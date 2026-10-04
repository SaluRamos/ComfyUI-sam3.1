from .nodes import SAM31Automatic, IsNull

NODE_CLASS_MAPPINGS = {
    "SAM31Automatic": SAM31Automatic,
    "SAM31IsNull": IsNull,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SAM31Automatic": "SAM 3.1 Automatic Segmentation",
    "SAM31IsNull": "IsNull",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
