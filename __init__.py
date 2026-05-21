from .klein_tiled_upscaler import KleinTiledUpscalerNode

NODE_CLASS_MAPPINGS = {
    "KleinTiledUpscaler": KleinTiledUpscalerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KleinTiledUpscaler": "Klein Tiled Upscaler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]