# ComfyUI SAM 3.1

Segmentação automática de imagens usando o SAM 3.1 nativo do ComfyUI, com 25 saídas individuais e um filtro IsNull.

## Instalação

Coloque esta pasta em `ComfyUI/custom_nodes` e reinicie o ComfyUI. Requer uma versão do ComfyUI com `comfy.supported_models.SAM31` e o tracker multiplex nativo. Não requer instalar o pacote externo `sam3` nem dependências adicionais.

O checkpoint é procurado nas pastas registradas de checkpoints e em `ComfyUI/models`. Quando ausente, a primeira execução baixa automaticamente para:

`ComfyUI/models/checkpoints/sam3.1_multiplex_fp16.safetensors`

[Download do modelo](https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors) · [Repositório e licença do modelo](https://huggingface.co/Comfy-Org/sam3.1)

O arquivo ocupa aproximadamente 1,75 GB. Downloads incompletos são removidos e podem ser repetidos. Imagens não são enviadas à internet.

## SAM 3.1 Automatic Segmentation

Conecte uma IMAGE. O node cria uma grade regular de pontos, segmenta cada ponto com o decoder interativo do SAM 3.1, filtra os resultados e remove máscaras duplicadas. O encoder da imagem é executado uma única vez. Não exige texto, caixas ou cliques manuais.

Cada saída `segment_1` até `segment_25` contém uma IMAGE `[1,H,W,3]` em resolução original, recortada pelos limites da máscara. `cutout` mostra o objeto da imagem original sobre fundo preto; `mask` mostra uma máscara branca sobre fundo preto. A margem padrão é de **20 pixels de cada lado**, inclusive quando o objeto toca a borda da imagem original. As saídas podem ter tamanhos diferentes.

Com seis segmentos, as seis primeiras saídas contêm imagens e as 19 últimas contêm `None`. Se houver mais de 25 segmentos, são emitidos os 25 primeiros após a ordenação escolhida. A detecção pode incluir partes de objetos e regiões sobrepostas; uma grade de pontos não garante um segmento por objeto semântico.

| Atributo | Função |
| --- | --- |
| points_per_side | Densidade da grade: 12 significa 144 pontos. Aumentar encontra detalhes menores, mas demora mais. |
| points_per_batch | Pontos processados juntos. Reduza para 1 se faltar VRAM. |
| confidence_threshold | Confiança de presença do objeto, calculada pelo tracker. |
| stability_threshold / stability_offset | Rejeitam máscaras que mudam muito quando o limiar dos logits varia. |
| nms_threshold | IoU entre máscaras para remoção de duplicadas; valores menores suprimem mais sobreposições. |
| min_area_pixels | Descarta regiões pequenas, em pixels da imagem original. |
| max_area_fraction | Descarta regiões que ocupam uma fração excessiva da imagem; 1 permite a imagem inteira. |
| padding | Margem em pixels por lado; padrão 20. |
| output_mode | Objeto recortado (`cutout`) ou máscara (`mask`). |
| sort_by | Área decrescente ou confiança combinada com estabilidade. |
| image_index | Índice da imagem usada quando IMAGE contém um lote; padrão 0. |

O carregamento e descarregamento usam o gerenciamento de memória do ComfyUI. Este node usa o ModelPatcher tradicional (`disable_dynamic=True`) para contornar quedas nativas no carregamento com DynamicVRAM no Windows. A inferência continua usando a GPU selecionada pelo ComfyUI. O modelo não é mantido em cache pelo custom node entre execuções.

### Integração com meshes

`SAM31Automatic.segment_masks(...)` fornece as mesmas detecções como uma lista de máscaras booleanas CPU `[H,W]`, sem crop ou padding, na resolução original, com limite de 25 segmentos. Essa API é usada pelo `SAM 3.1 Mesh Segmentation` de `ComfyUI-Semantic-Visibility-UV-Param` para projetar máscaras de várias vistas nas faces 3D.

## IsNull

Entrada e saída coringa. Valores não nulos são devolvidos sem alteração. Uma entrada `None` produz `ExecutionBlocker(None)`, que impede a execução dos consumidores dessa saída, sem cancelar as demais ramificações.

Fluxo recomendado: `SAM 3.1 segment_N → IsNull → Preview Image / próximo node`.

Conectar uma saída vazia diretamente a um consumidor de IMAGE pode causar erro. Coloque IsNull em cada ramificação que possa estar vazia. Se um node juntar entradas de várias ramificações, uma entrada bloqueada também bloqueará esse node.
