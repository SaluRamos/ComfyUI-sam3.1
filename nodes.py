import logging
import os
from pathlib import Path
import tempfile
import threading
import urllib.request

import torch
import torch.nn.functional as F

import comfy.model_management as mm
import comfy.sd
import comfy.utils
from comfy.ops import cast_to_input
from comfy_execution.graph import ExecutionBlocker
import folder_paths

MODEL_FILENAME = "sam3.1_multiplex_fp16.safetensors"
MAX_SEGMENTS = 25
MODEL_URL = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/" + MODEL_FILENAME
_download_lock = threading.Lock()


def model_path():
    registered = folder_paths.get_full_path("checkpoints", MODEL_FILENAME)
    if registered:
        return registered
    root = Path(folder_paths.models_dir)
    with _download_lock:
        existing = next(root.rglob(MODEL_FILENAME), None)
        if existing is not None:
            return str(existing)
        target = root / "checkpoints" / MODEL_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        logging.info("SAM 3.1: downloading %s", MODEL_URL)
        temporary = None
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                progress = comfy.utils.ProgressBar(total or 1)
                downloaded = 0
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".part", delete=False) as output:
                    temporary = output.name
                    while chunk := response.read1(256 * 1024):
                        mm.throw_exception_if_processing_interrupted()
                        output.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            progress.update_absolute(downloaded, total)
                if downloaded == 0 or (total and downloaded != total):
                    raise RuntimeError("Download incompleto do SAM 3.1. Execute novamente para tentar baixar.")
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                os.unlink(temporary)
        return str(target)


def select_masks(candidates, nms_threshold, sort_by):
    # Suppress duplicates on the decoder grid before resizing to the original image.
    selected = []
    for logits, score, stability in sorted(candidates, key=lambda item: item[1] * item[2], reverse=True):
        mask = logits > 0
        duplicate = False
        for other, _, _ in selected:
            other_mask = other > 0
            union = (mask | other_mask).sum().item()
            if union and (mask & other_mask).sum().item() / union > nms_threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append((logits, score, stability))
    if sort_by == "area":
        selected.sort(key=lambda item: (item[0] > 0).sum().item(), reverse=True)
    return selected[:MAX_SEGMENTS]


def crop_segment(image, mask, padding, output_mode):
    ys, xs = torch.where(mask)
    if not len(xs):
        return None
    x0, x1 = xs.min().item(), xs.max().item() + 1
    y0, y1 = ys.min().item(), ys.max().item() + 1
    cropped_mask = mask[y0:y1, x0:x1, None].float()
    if output_mode == "mask":
        crop = cropped_mask.expand(-1, -1, 3)
    else:
        crop = image[y0:y1, x0:x1, :3] * cropped_mask
    # Pad the cropped canvas, including objects touching the source image edges.
    return F.pad(crop.permute(2, 0, 1), (padding,) * 4).permute(1, 2, 0).unsqueeze(0).contiguous()


class SAM31Automatic:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE", {"tooltip": "Imagem a segmentar. Os recortes preservam a resolução original. Se a entrada for um lote, image_index escolhe qual imagem será usada."}),
            "points_per_side": ("INT", {"default": 12, "min": 2, "max": 32, "tooltip": "Quantidade de pontos em cada lado da grade automática. 12 gera 144 pontos; 24 gera 576. Aumentar ajuda a encontrar regiões pequenas e detalhes, mas aumenta o tempo de processamento. Reduzir acelera, podendo deixar objetos sem detectar."}),
            "points_per_batch": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "Quantidade de pontos processados juntos. Aumentar pode acelerar quando há VRAM disponível, mas consome mais memória da GPU. Reduza para 1 se faltar memória. Não altera a densidade da grade nem os filtros de seleção."}),
            "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Confiança mínima de presença do objeto prevista pelo tracker. Aumentar rejeita detecções incertas e pode reduzir o número de segmentos. Diminuir aceita mais candidatos, incluindo possíveis falsos positivos. Não representa a precisão dos contornos."}),
            "stability_threshold": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Estabilidade mínima da máscara quando seu limiar varia. Aumentar exige contornos mais estáveis, mas pode eliminar segmentos úteis. Diminuir aceita máscaras mais sensíveis ao limiar e pode recuperar regiões difíceis. A variação usada no teste é definida por stability_offset."}),
            "stability_offset": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1, "tooltip": "Quanto variar o limiar dos logits para medir a estabilidade. Aumentar torna o teste mais exigente e pode rejeitar mais máscaras. Diminuir facilita a aprovação. Com 0, máscaras não vazias têm estabilidade 1, tornando esse filtro ineficaz."}),
            "nms_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Sobreposição máxima entre máscaras, medida por IoU, antes de eliminar uma duplicada. Diminuir remove mais regiões sobrepostas, podendo eliminar objetos distintos. Aumentar preserva mais sobreposições e pode deixar duplicatas. Com 1, nenhuma máscara é suprimida por esse filtro."}),
            "min_area_pixels": ("INT", {"default": 100, "min": 1, "max": 10000000, "tooltip": "Área mínima do segmento em pixels da imagem original. Aumentar elimina pequenos fragmentos e ruído, mas também objetos pequenos. Diminuir preserva detalhes menores. A margem adicionada por padding não conta nessa área."}),
            "max_area_fraction": ("FLOAT", {"default": 0.95, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Fração máxima da imagem que uma máscara pode ocupar. 0,95 rejeita regiões que cobrem mais de 95% da imagem. Diminuir ajuda a excluir grandes regiões de fundo, mas pode eliminar objetos grandes. 1 permite máscaras que ocupam a imagem inteira."}),
            "padding": ("INT", {"default": 20, "min": 0, "max": 512, "tooltip": "Margem preta adicionada em cada lado do recorte, em pixels. 20 acrescenta 40 pixels à largura e à altura do limite da máscara. Aumentar dá mais espaço ao redor do objeto; 0 recorta rente. Não altera a segmentação nem recupera conteúdo fora da imagem."}),
            "output_mode": (["cutout", "mask"], {"tooltip": "cutout: objeto recortado da imagem original sobre fundo preto. mask: máscara branca sobre fundo preto. Ambos retornam IMAGE com três canais, recortada e com a margem escolhida."}),
            "sort_by": (["area", "confidence"], {"tooltip": "Define a ordem das saídas após remover duplicadas. area prioriza os maiores segmentos; confidence prioriza confiança multiplicada pela estabilidade. Se houver mais de 25 segmentos, somente os 25 primeiros dessa ordem serão emitidos."}),
            "image_index": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Índice da imagem escolhida quando a entrada contém um lote. 0 usa a primeira, 1 a segunda, e assim por diante. Apenas essa imagem é segmentada. Um índice maior que o lote disponível gera erro."}),
        }}

    RETURN_TYPES = ("IMAGE",) * MAX_SEGMENTS
    RETURN_NAMES = tuple(f"segment_{i}" for i in range(1, MAX_SEGMENTS + 1))
    FUNCTION = "segment"
    CATEGORY = "image/segmentation/SAM 3.1"
    DESCRIPTION = "Automatic point-grid segmentation. Unused IMAGE outputs are None; connect IsNull to skip empty branches."

    def segment(self, image, points_per_side, points_per_batch, confidence_threshold,
                stability_threshold, stability_offset, nms_threshold, min_area_pixels,
                max_area_fraction, padding, output_mode, sort_by, image_index):
        masks = self.segment_masks(image, points_per_side, points_per_batch, confidence_threshold,
            stability_threshold, stability_offset, nms_threshold, min_area_pixels,
            max_area_fraction, sort_by, image_index)
        original = image[image_index, ..., :3].detach().to(device="cpu", dtype=torch.float32)
        outputs = [crop_segment(original, mask, padding, output_mode) for mask in masks]
        logging.info("SAM 3.1: %d segmentos emitidos (limite: %d).", len(outputs), MAX_SEGMENTS)
        return tuple(outputs + [None] * (MAX_SEGMENTS - len(outputs)))

    def segment_masks(self, image, points_per_side=12, points_per_batch=4, confidence_threshold=0.5,
                      stability_threshold=0.9, stability_offset=1.0, nms_threshold=0.7,
                      min_area_pixels=100, max_area_fraction=0.95, sort_by="area", image_index=0):
        """Full-frame CPU boolean masks for consumers that project masks onto geometry."""
        if image_index >= image.shape[0]:
            raise ValueError(f"image_index={image_index}; o lote possui {image.shape[0]} imagem(ns).")
        original = image[image_index, ..., :3].detach().to(device="cpu", dtype=torch.float32)
        height, width = original.shape[:2]
        patcher = comfy.sd.load_checkpoint_guess_config(
            model_path(), output_vae=False, output_clip=False,
            disable_dynamic=True,
        )[0]
        try:
            mm.load_model_gpu(patcher)
            device = patcher.load_device
            dtype = patcher.model.get_dtype_inference()
            sam = patcher.model.diffusion_model
            if not sam.detector.backbone["vision_backbone"].multiplex:
                raise RuntimeError("O checkpoint precisa ser SAM 3.1 multiplex.")
            frame = comfy.utils.common_upscale(original.unsqueeze(0).movedim(-1, 1), 1008, 1008, "bilinear", "disabled")
            frame = (frame.to(device=device, dtype=dtype) - 0.5) / 0.5
            backbone = sam.detector.backbone["vision_backbone"]
            _, _, features, _ = backbone(frame, tracker_mode="interactive", tracker_only=True)
            high_res = list(features[:-1])
            embedding = features[-1]
            batch, channels, fh, fw = embedding.shape
            flat = embedding.flatten(2).permute(0, 2, 1)
            flat = flat + cast_to_input(sam.tracker.interactivity_no_mem_embed, flat)
            embedding = flat.reshape(batch, fh, fw, channels).permute(0, 3, 1, 2)

            axis = (torch.arange(points_per_side, device=device, dtype=torch.float32) + 0.5) * (1008 / points_per_side)
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            points = torch.stack((xx.flatten(), yy.flatten()), dim=-1).to(dtype=dtype)
            candidates = []
            progress = comfy.utils.ProgressBar(len(points))
            for start in range(0, len(points), points_per_batch):
                mm.throw_exception_if_processing_interrupted()
                coords = points[start:start + points_per_batch, None, :]
                low_res, _, _, object_logits = sam.tracker._forward_sam_heads(
                    backbone_features=embedding,
                    point_inputs={"point_coords": coords, "point_labels": torch.ones(coords.shape[:2], device=device, dtype=torch.int32)},
                    high_res_features=high_res,
                    multimask_output=True,
                )
                scores = object_logits.float().sigmoid().flatten().cpu()
                for logits, score in zip(low_res[:, 0].float().cpu(), scores.tolist()):
                    if score < confidence_threshold:
                        continue
                    mask = logits > 0
                    fraction = mask.float().mean().item()
                    if fraction * height * width < min_area_pixels or fraction > max_area_fraction:
                        continue
                    union = (logits > -stability_offset).sum().item()
                    stability = (logits > stability_offset).sum().item() / max(union, 1)
                    if stability >= stability_threshold:
                        candidates.append((logits, score, stability))
                progress.update_absolute(min(start + points_per_batch, len(points)), len(points))

            masks = []
            for logits, _, _ in select_masks(candidates, nms_threshold, sort_by):
                mask = F.interpolate(logits[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0] > 0
                if mask.sum().item() < min_area_pixels:
                    continue
                masks.append(mask)
            return masks
        finally:
            mm.unload_model_and_clones(patcher)


class AnyType(str):
    def __ne__(self, other):
        return False


ANY = AnyType("*")


class IsNull:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"value": (ANY,)}}

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("value",)
    FUNCTION = "passthrough"
    CATEGORY = "utils/flow"
    DESCRIPTION = "Passes non-null input unchanged; silently blocks downstream execution for None."

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def passthrough(self, value):
        return (ExecutionBlocker(None) if value is None else value,)
