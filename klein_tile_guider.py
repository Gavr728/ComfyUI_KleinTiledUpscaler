import math
import torch
import comfy.samplers


class KleinTileAwareCFGGuider(comfy.samplers.CFGGuider):
    """
    Tile-aware CFG Guider for Flux-Klein + USDU upscaling.

    Use with USDU 'Ignore Overlap' mode, padding=0, tile_overlap=0.
    Pass original (non-upscaled) ref latent to ReferenceLatent.

    Tile detection: step counter (most reliable).
    Ref crop: proportional mapping from upscaled canvas to original ref.
    """

    LAT_SCALE = 16

    def __init__(self, model_patcher):
        super().__init__(model_patcher)
        self._call_count = 0
        self._original_ref_clone = None
        self.grid_cols = 3
        self.grid_rows = 4
        self.tile_size_px = 1024
        self.steps_per_tile = 4
        self.target_width = 2784
        self.target_height = 3584
        self.debug = True

    def _get_ref_list_and_original(self):
        try:
            cv = self.conds["positive"][0]["model_conds"]["ref_latents"].cond
            if (isinstance(cv, list) and len(cv) > 0
                    and isinstance(cv[0], torch.Tensor)
                    and cv[0].dim() == 4):
                if self._original_ref_clone is None:
                    self._original_ref_clone = cv[0].clone()
                    if self.debug:
                        print(f"[KLEIN] ref shape: {self._original_ref_clone.shape}")
                return cv, self._original_ref_clone
        except (KeyError, IndexError, AttributeError):
            pass
        return None, None

    def predict_noise(self, x, timestep, model_options=None, seed=None):
        tile_idx = self._call_count // self.steps_per_tile
        tile_idx = tile_idx % (self.grid_cols * self.grid_rows)
        self._call_count += 1

        c = tile_idx % self.grid_cols
        r = tile_idx // self.grid_cols

        # Raw tile position matching USDU's calc_rectangle: xi * tile_width
        x_px = c * self.tile_size_px
        y_px = r * self.tile_size_px

        ref_list, original_ref = self._get_ref_list_and_original()
        backup = None

        if original_ref is not None:
            ref_h = original_ref.shape[2]
            ref_w = original_ref.shape[3]

            # Proportional mapping: upscaled canvas → original ref token space
            x_tok = round(x_px * ref_w / self.target_width)
            y_tok = round(y_px * ref_h / self.target_height)

            # Crop size proportional to tile's fraction of canvas
            tile_w_px = x.shape[3] * self.LAT_SCALE
            tile_h_px = x.shape[2] * self.LAT_SCALE
            w_tok = max(1, round(tile_w_px * ref_w / self.target_width))
            h_tok = max(1, round(tile_h_px * ref_h / self.target_height))

            xs = max(0, min(x_tok, ref_w - w_tok))
            ys = max(0, min(y_tok, ref_h - h_tok))

            crop = original_ref[:, :, ys:ys+h_tok, xs:xs+w_tok].clone()

            if self.debug:
                print(f"[KLEIN] call={self._call_count-1} tile={tile_idx} ({c},{r}) "
                      f"px=({x_px},{y_px}) tok=({xs},{ys}) crop={crop.shape}")

            try:
                cv = self.conds["positive"][0]["model_conds"]["ref_latents"].cond
                if isinstance(cv, list) and len(cv) > 0:
                    ref_list = cv
                    backup = ref_list[0].clone()
                    ref_list[0] = crop
            except (KeyError, IndexError, AttributeError):
                pass

        try:
            result = super().predict_noise(x, timestep, model_options, seed)
        finally:
            if ref_list is not None and backup is not None:
                ref_list[0] = backup

        return result


class KleinTileAwareCFGGuiderNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("MODEL",),
                "positive":      ("CONDITIONING",),
                "negative":      ("CONDITIONING",),
                "cfg":           ("FLOAT", {"default": 1.0,  "min": 0.0,   "max": 100.0, "step": 0.1}),
                "grid_cols":     ("INT",   {"default": 3,    "min": 1,     "max": 64,    "step": 1, "tooltip": "Columns from USDU console"}),
                "grid_rows":     ("INT",   {"default": 4,    "min": 1,     "max": 64,    "step": 1, "tooltip": "Rows from USDU console"}),
                "tile_size":     ("INT",   {"default": 1024, "min": 64,    "max": 8192,  "step": 8, "tooltip": "USDU tile size in pixels"}),
                "steps":         ("INT",   {"default": 4,    "min": 1,     "max": 100,   "step": 1, "tooltip": "Must match sampler steps"}),
                "target_width":  ("INT",   {"default": 2784, "min": 64,    "max": 16384, "step": 8, "tooltip": "Upscaled canvas width from USDU 'Canva size'"}),
                "target_height": ("INT",   {"default": 3584, "min": 64,    "max": 16384, "step": 8, "tooltip": "Upscaled canvas height from USDU 'Canva size'"}),
            }
        }

    RETURN_TYPES = ("GUIDER",)
    FUNCTION = "create_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def create_guider(self, model, positive, negative, cfg,
                      grid_cols, grid_rows, tile_size, steps,
                      target_width, target_height):
        guider = KleinTileAwareCFGGuider(model)
        guider.set_conds(positive=positive, negative=negative)
        guider.set_cfg(cfg)
        guider.grid_cols = grid_cols
        guider.grid_rows = grid_rows
        guider.tile_size_px = tile_size
        guider.steps_per_tile = steps
        guider.target_width = target_width
        guider.target_height = target_height
        return (guider,)


NODE_CLASS_MAPPINGS = {
    "KleinTileAwareCFGGuider": KleinTileAwareCFGGuiderNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KleinTileAwareCFGGuider": "Klein Tile-Aware CFG Guider",
}