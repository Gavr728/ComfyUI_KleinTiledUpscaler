"""
Klein Tiled Upscaler - Self-contained tiling node for Flux2.Klein
Features flawless Single-Pass Smoothstep Matrix Inpainting. Zero Stairs. Zero Sharp Edges.
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn.functional as F

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.conds
import comfy.model_management
from nodes import VAEEncode, VAEDecode, VAEDecodeTiled

# -----------------------------------------------------------------------------
# Helpers 
# -----------------------------------------------------------------------------

def slice_sigmas(sigmas, denoise):
    if denoise >= 1.0:
        return sigmas
    if denoise <= 0.0:
        return sigmas[-1:]
    steps = len(sigmas) - 1
    start_step = int(steps * (1.0 - denoise))
    return sigmas[start_step:]

def pil_to_tensor(image):
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def tensor_to_pil(tensor, index=0):
    arr = tensor[index].cpu().numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def get_crop_region(mask, pad=0):
    coords = mask.getbbox()
    if coords is None:
        return (0, 0, mask.width, mask.height)
    x1, y1, x2, y2 = coords
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, mask.width)
    y2 = min(y2 + pad, mask.height)
    if x2 < mask.width:
        x2 -= 1
    if y2 < mask.height:
        y2 -= 1
    return x1, y1, x2, y2

def expand_and_align_crop(region, width, height, target_w, target_h):
    x1, y1, x2, y2 = region
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    new_x1 = cx - (target_w // 2)
    new_y1 = cy - (target_h // 2)
    
    new_x1 = (new_x1 // 32) * 32
    new_y1 = (new_y1 // 32) * 32
    new_x2 = new_x1 + target_w
    new_y2 = new_y1 + target_h
    
    if new_x1 < 0:
        new_x1 = 0
        new_x2 = target_w
    if new_y1 < 0:
        new_y1 = 0
        new_y2 = target_h
    if new_x2 > width:
        new_x2 = width
        new_x1 = width - target_w
    if new_y2 > height:
        new_y2 = height
        new_y1 = height - target_h
        
    new_x1 = (new_x1 // 32) * 32
    new_y1 = (new_y1 // 32) * 32
    new_x2 = new_x1 + target_w
    new_y2 = new_y1 + target_h
    return (new_x1, new_y1, new_x2, new_y2), (target_w, target_h)

def color_match_tensor(target, source):
    # Strict Linear Color Match: scales mean and standard deviations uniformly
    t = target.movedim(-1, 1) 
    s = source.movedim(-1, 1).to(t.device) 
    
    t_mean = t.mean(dim=(2, 3), keepdim=True)
    t_std = t.std(dim=(2, 3), keepdim=True) + 1e-6
    s_mean = s.mean(dim=(2, 3), keepdim=True)
    s_std = s.std(dim=(2, 3), keepdim=True) + 1e-6
    
    matched = (t - t_mean) * (s_std / t_std) + s_mean
    matched = torch.clamp(matched, 0.0, 1.0)
    return matched.movedim(1, -1)

# -----------------------------------------------------------------------------
# Advanced Masking
# -----------------------------------------------------------------------------

def create_smooth_matrix_mask(canvas_w, canvas_h, core_x1, core_y1, core_x2, core_y2, blend):
    mask = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    x1 = max(0, core_x1 - blend)
    y1 = max(0, core_y1 - blend)
    x2 = min(canvas_w, core_x2 + blend)
    y2 = min(canvas_h, core_y2 + blend)
    if x1 >= x2 or y1 >= y2:
        return Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    
    def smooth_curve(length):
        t = np.linspace(0, 1, length, dtype=np.float32)
        return 0.5 - 0.5 * np.cos(np.pi * t)
        
    x_grad = np.ones(x2 - x1, dtype=np.float32)
    y_grad = np.ones(y2 - y1, dtype=np.float32)
    
    blend_left = core_x1 - x1
    blend_right = x2 - core_x2
    blend_top = core_y1 - y1
    blend_bottom = y2 - core_y2
    
    if blend_left > 0:
        x_grad[:blend_left] = smooth_curve(blend_left)
    if blend_right > 0:
        x_grad[-blend_right:] = smooth_curve(blend_right)[::-1]
    if blend_top > 0:
        y_grad[:blend_top] = smooth_curve(blend_top)
    if blend_bottom > 0:
        y_grad[-blend_bottom:] = smooth_curve(blend_bottom)[::-1]
        
    mask_2d = np.outer(y_grad, x_grad)
    mask[y1:y2, x1:x2] = mask_2d
    return Image.fromarray((mask * 255).astype(np.uint8), mode='L')

# -----------------------------------------------------------------------------
# Tile preparation
# -----------------------------------------------------------------------------

def prepare_tile(image, core_x1, core_y1, actual_tw, actual_th, padding,
                 canvas_w, canvas_h, full_tile_w=None, full_tile_h=None):
    mask = Image.new("L", (canvas_w, canvas_h), "black")
    draw = ImageDraw.Draw(mask)
    draw.rectangle((core_x1, core_y1, core_x1 + actual_tw, core_y1 + actual_th), fill="white")
    crop_region = get_crop_region(mask, padding)
    
    use_tw = full_tile_w if full_tile_w is not None else actual_tw
    use_th = full_tile_h if full_tile_h is not None else actual_th
    target_w = math.ceil((use_tw + padding * 2) / 32) * 32
    target_h = math.ceil((use_th + padding * 2) / 32) * 32
    
    crop_region, tile_size = expand_and_align_crop(crop_region, canvas_w, canvas_h, target_w, target_h)
    tile = image.crop(crop_region)
    original_size = tile.size
    
    if tile.size != tile_size:
        tile = tile.resize(tile_size, Image.Resampling.LANCZOS)
    return tile, crop_region, tile_size, original_size

def composite_tile(canvas, tile_pil, crop_region, original_size, mask):
    if tile_pil.size != original_size:
        tile_pil = tile_pil.resize(original_size, Image.Resampling.LANCZOS)
    
    tile_mask = mask.crop((crop_region[0], crop_region[1], crop_region[2], crop_region[3]))
    canvas.paste(tile_pil, crop_region[:2], tile_mask.convert('L'))
    return canvas

# -----------------------------------------------------------------------------
# Conditioning and sampling
# -----------------------------------------------------------------------------

def patch_conditioning(cond, ref_crop):
    new_cond = []
    for c in cond:
        if isinstance(c, (list, tuple)) and len(c) == 2 and isinstance(c[1], dict):
            new_c = list(c)
            new_c[1] = c[1].copy()
            if "model_conds" not in new_c[1]:
                new_c[1]["model_conds"] = {}
            else:
                new_c[1]["model_conds"] = new_c[1]["model_conds"].copy()
            new_c[1]["model_conds"]["ref_latents"] = comfy.conds.CONDList([ref_crop.clone()])
            new_c[1]["reference_latents"] = [ref_crop.clone()]
            new_cond.append(new_c)
        else:
            new_cond.append(c)
    return new_cond

def crop_latent_for_tile(full_latent, crop_region, canvas_w):
    divisor = 16 # Flux2/Klein VAE spatial compression token hardcode
    x1, y1, x2, y2 = crop_region
    
    rx1 = max(0, x1 // divisor)
    ry1 = max(0, y1 // divisor)
    rx2 = min(x2 // divisor, full_latent.shape[3])
    ry2 = min(y2 // divisor, full_latent.shape[2])
    rx2 = max(rx1 + 1, rx2)
    ry2 = max(ry1 + 1, ry2)
    
    raw = full_latent[:, :, ry1:ry2, rx1:rx2]
    enc_w = max(1, (x2 - x1) // divisor)
    enc_h = max(1, (y2 - y1) // divisor)
    
    if raw.shape[2] != enc_h or raw.shape[3] != enc_w:
        raw = F.interpolate(raw.float(), size=(enc_h, enc_w), mode='bilinear', align_corners=False).to(full_latent.dtype)
    return raw.clone()

def crop_noise_for_tile(global_noise, crop_region, canvas_w):
    divisor = 16
    x1, y1, x2, y2 = crop_region
    
    rx1 = x1 // divisor
    ry1 = y1 // divisor
    rx2 = x2 // divisor
    ry2 = y2 // divisor

    tile_noise = global_noise[:, :, ry1:ry2, rx1:rx2].clone()

    enc_w = max(1, (x2 - x1) // divisor)
    enc_h = max(1, (y2 - y1) // divisor)
    
    if tile_noise.shape[2] != enc_h or tile_noise.shape[3] != enc_w:
        tile_noise = F.interpolate(tile_noise.float(), size=(enc_h, enc_w), mode='bilinear', align_corners=False).to(global_noise.dtype)

    mean = tile_noise.mean()
    std = tile_noise.std()
    if std > 1e-6:
        tile_noise = (tile_noise - mean) / std

    return tile_noise

def patch_rope(diffusion_model, shift_x, shift_y):
    if shift_x == 0 and shift_y == 0:
        return None
    
    original_forward = diffusion_model.forward
    
    def patched_forward(*args, **kwargs):
        new_args = list(args)
        if len(new_args) > 1 and isinstance(new_args[1], torch.Tensor) and new_args[1].shape[-1] == 3:
            img_ids = new_args[1].clone()
            img_ids[..., 1] += shift_y
            img_ids[..., 2] += shift_x
            new_args[1] = img_ids
        elif "img_ids" in kwargs and kwargs["img_ids"] is not None:
            img_ids = kwargs["img_ids"].clone()
            img_ids[..., 1] += shift_y
            img_ids[..., 2] += shift_x
            kwargs["img_ids"] = img_ids
            
        return original_forward(*new_args, **kwargs)
        
    diffusion_model.forward = patched_forward
    return original_forward

def sample_tile(guider, positive, negative, sampler, sigmas, latent, seed, tile_noise,
               full_ref_tensor, crop_region, canvas_w):
    latent_image = latent["samples"]
    crop = crop_latent_for_tile(full_ref_tensor, crop_region, canvas_w)
    patched_pos = patch_conditioning(positive, crop)
    guider.set_conds(positive=patched_pos, negative=negative)
    
    noise_mask = latent.get("noise_mask", None)

    samples = guider.sample(
        tile_noise, latent_image, sampler, sigmas,
        denoise_mask=noise_mask, callback=None,
        disable_pbar=False, seed=seed
    )

    out = latent.copy()
    out["samples"] = samples
    return out

# -----------------------------------------------------------------------------
# Main node
# -----------------------------------------------------------------------------

class KleinTiledUpscalerNode:
    TILING_STRATEGIES = ["Chess", "Linear", "Reverse Chess", "Spiral", "Detail-First"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider":        ("GUIDER",),
                "positive":      ("CONDITIONING",),
                "negative":      ("CONDITIONING",),
                "sampler":       ("SAMPLER",),
                "sigmas":        ("SIGMAS",),
                "vae":           ("VAE",),
                "image":         ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "seed":          ("INT",    {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scale_factor":  ("FLOAT",  {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.25}),
                "tiling_strategy": (cls.TILING_STRATEGIES, {"default": "Detail-First"}),
                "tile_size_mode": (["Auto", "Manual"], {"default": "Auto"}),
                "tile_width":    ("INT",    {"default": 1024, "min": 512, "max": 4096, "step": 16}),
                "tile_height":   ("INT",    {"default": 1024, "min": 512, "max": 4096, "step": 16}),
                "padding":       ("INT",    {"default": 128, "min": 0, "max": 512, "step": 16}),
                "color_match":   ("BOOLEAN", {"default": True}),
                "mask_blur":     ("INT",    {"default": 32, "min": 0, "max": 64, "step": 1}),
                "adaptive_tiling": ("BOOLEAN", {"default": False}),
                "tiled_decode":  ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "sampling/custom_sampling"

    def upscale(self, guider, positive, negative, sampler, sigmas, vae, image, upscale_model,
                seed, scale_factor, tiling_strategy, tile_size_mode, tile_width, tile_height, padding,
                color_match, mask_blur, adaptive_tiling, tiled_decode):

        import comfy_extras.nodes_upscale_model as upscale_nodes
        
        vae_encoder = VAEEncode()
        vae_decoder = VAEDecode()
        vae_decoder_tiled = VAEDecodeTiled()
        upscaler_node = upscale_nodes.ImageUpscaleWithModel()
        
        batch_size = image.shape[0]
        final_batch_outputs = []

        # Save the original unpatched native KSampler execution method globally
        current_sample_method = comfy.samplers.KSampler.sample
        
        def native_comfy_sample(self, noise, positive, negative, cfg, device, sampler, sigmas, 
                                model_options={}, latent_image=None, denoise_mask=None, 
                                callback=None, disable_pbar=False, seed=None):
            return comfy.samplers.sample(
                self.model, noise, positive, negative, cfg, device, sampler, sigmas, 
                model_options, latent_image=latent_image, denoise_mask=denoise_mask, 
                callback=callback, disable_pbar=disable_pbar, seed=seed
            )

        dm_to_restore = None
        orig_forward = None

        try:
            # Bypass legacy Tiled Diffusion monkeypatch
            comfy.samplers.KSampler.sample = native_comfy_sample
            
            for b in range(batch_size):
                print(f"[KLEIN] Processing Batch Element {b+1}/{batch_size}")
                
                img_b = image[b:b+1] # Ensure dimensionality remains [1, H, W, C]
                upscaled_t = upscaler_node.upscale(upscale_model, img_b)[0]
                
                target_w = (round(img_b.shape[2] * scale_factor) // 32) * 32
                target_h = (round(img_b.shape[1] * scale_factor) // 32) * 32
                
                upscaled_t = upscaled_t.movedim(-1, 1)
                upscaled_t = F.interpolate(upscaled_t, size=(target_h, target_w), mode='bicubic', antialias=True)
                upscaled_t = torch.clamp(upscaled_t, 0.0, 1.0).movedim(1, -1)

                canvas_w, canvas_h = upscaled_t.shape[2], upscaled_t.shape[1]
                canvas_np = (upscaled_t[0].cpu().numpy() * 255).astype(np.uint8)
                canvas = Image.fromarray(canvas_np)

                print(f"[KLEIN] Canvas: {canvas_w}x{canvas_h}")

                (upscaled_latent_dict,) = vae_encoder.encode(vae, upscaled_t)
                raw_latent = upscaled_latent_dict["samples"]
                model = guider.model_patcher.model
                
                if hasattr(model, "process_latent_in"):
                    full_ref_tensor = model.process_latent_in(raw_latent)
                else:
                    full_ref_tensor = raw_latent

                global_noise = comfy.sample.prepare_noise(raw_latent, seed, None)

                if tile_size_mode == "Auto":
                    cols = max(1, round(canvas_w / 1024))
                    rows = max(1, round(canvas_h / 1024))
                    current_tile_width = max(128, (canvas_w // cols // 32) * 32)
                    current_tile_height = max(128, (canvas_h // rows // 32) * 32)
                else:
                    current_tile_width = (tile_width // 32) * 32
                    current_tile_height = (tile_height // 32) * 32

                rows = math.ceil(canvas_h / current_tile_height)
                cols = math.ceil(canvas_w / current_tile_width)
                tiles_order = []

                for yi in range(rows):
                    for xi in range(cols):
                        core_x1 = xi * current_tile_width
                        core_y1 = yi * current_tile_height
                        actual_tw = min(current_tile_width, canvas_w - core_x1)
                        actual_th = min(current_tile_height, canvas_h - core_y1)
                        if actual_tw > 0 and actual_th > 0:
                            tiles_order.append((xi, yi, core_x1, core_y1, actual_tw, actual_th))

                total = len(tiles_order)

                gray_lr = (img_b[..., 0] * 0.299 + img_b[..., 1] * 0.587 + img_b[..., 2] * 0.114).unsqueeze(1)
                blur_kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=img_b.device) / 9.0
                gray_lr = F.conv2d(gray_lr, blur_kernel, padding=1)
                lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=img_b.device).view(1, 1, 3, 3)
                lap_map_lr = F.conv2d(gray_lr, lap_kernel, padding=1).abs()

                tile_variances = {}
                for xi, yi, core_x1, core_y1, actual_tw, actual_th in tiles_order:
                    core_x2, core_y2 = core_x1 + actual_tw, core_y1 + actual_th
                    lr_x1, lr_y1 = int(core_x1 / scale_factor), int(core_y1 / scale_factor)
                    lr_x2, lr_y2 = int(core_x2 / scale_factor), int(core_y2 / scale_factor)
                    
                    lr_x2, lr_y2 = min(lr_x2, lap_map_lr.shape[3]), min(lr_y2, lap_map_lr.shape[2])
                    tile_lap = lap_map_lr[:, :, lr_y1:lr_y2, lr_x1:lr_x2]
                    tile_variances[(xi, yi)] = float(tile_lap.mean()) if tile_lap.numel() > 0 else 0.0

                var_values = sorted(list(tile_variances.values()))
                if var_values:
                    pct_idx = int(len(var_values) * 0.60)
                    pct_idx = min(len(var_values) - 1, max(0, pct_idx))
                    ref_var = max(0.005, var_values[pct_idx])
                else:
                    ref_var = 1.0

                if tiling_strategy == "Chess":
                    tiles_order = [t for t in tiles_order if (t[0]+t[1])%2 == 0] + [t for t in tiles_order if (t[0]+t[1])%2 == 1]
                elif tiling_strategy == "Reverse Chess":
                    tiles_order = [t for t in tiles_order if (t[0]+t[1])%2 == 1] + [t for t in tiles_order if (t[0]+t[1])%2 == 0]
                elif tiling_strategy == "Spiral":
                    cx, cy = (cols - 1) / 2.0, (rows - 1) / 2.0
                    tiles_order.sort(key=lambda t: (t[0]-cx)**2 + (t[1]-cy)**2)
                elif tiling_strategy == "Detail-First":
                    tiles_order.sort(key=lambda t: tile_variances[(t[0], t[1])], reverse=True)

                pbar = comfy.utils.ProgressBar(total)

                for step_i, (xi, yi, core_x1, core_y1, actual_tw, actual_th) in enumerate(tiles_order):
                    tile_pil, crop_region, tile_size, orig_size = prepare_tile(
                        canvas, core_x1, core_y1, actual_tw, actual_th, padding,
                        canvas_w, canvas_h, full_tile_w=current_tile_width, full_tile_h=current_tile_height)

                    core_x2 = core_x1 + actual_tw
                    core_y2 = core_y1 + actual_th

                    visual_blend_size = max(16, padding - 64) if padding >= 64 else (padding // 2)
                    visual_mask = create_smooth_matrix_mask(
                        canvas_w, canvas_h, core_x1, core_y1, core_x2, core_y2, visual_blend_size)

                    if mask_blur > 0:
                        visual_mask = visual_mask.filter(ImageFilter.GaussianBlur(mask_blur))

                    tile_t = pil_to_tensor(tile_pil)
                    (latent,) = vae_encoder.encode(vae, tile_t)
                    
                    latent_image = latent["samples"]
                    tile_h_lat, tile_w_lat = latent_image.shape[2], latent_image.shape[3]
                    
                    latent_expand_px = max(0, padding - 32)
                    latent_expand_size = (latent_expand_px // 32) * 32 if padding >= 32 else 0
                    
                    l_x1_px = max(0, core_x1 - latent_expand_size)
                    l_y1_px = max(0, core_y1 - latent_expand_size)
                    l_x2_px = min(canvas_w, core_x2 + latent_expand_size)
                    l_y2_px = min(canvas_h, core_y2 + latent_expand_size)
                    
                    l_x1_lat = max(0, (l_x1_px - crop_region[0]) // 16)
                    l_y1_lat = max(0, (l_y1_px - crop_region[1]) // 16)
                    l_x2_lat = min(tile_w_lat, (l_x2_px - crop_region[0]) // 16)
                    l_y2_lat = min(tile_h_lat, (l_y2_px - crop_region[1]) // 16)
                    
                    latent_mask = torch.zeros((1, tile_h_lat, tile_w_lat), dtype=torch.float32, device=latent_image.device)
                    latent_mask[0, l_y1_lat:l_y2_lat, l_x1_lat:l_x2_lat] = 1.0
                    latent["noise_mask"] = latent_mask

                    try:
                        dm = guider.model_patcher.model.diffusion_model
                    except AttributeError:
                        dm = None
                    
                    if dm is not None and hasattr(dm, "forward"):
                        dm_to_restore = dm
                        shift_x = crop_region[0] // 16
                        shift_y = crop_region[1] // 16
                        orig_forward = patch_rope(dm, shift_x, shift_y)

                    tile_noise = crop_noise_for_tile(global_noise, crop_region, canvas_w)

                    tile_sigmas = sigmas
                    if adaptive_tiling:
                        var_ratio = min(1.0, tile_variances[(xi, yi)] / ref_var)
                        denoise_mult = 0.35 + 0.65 * var_ratio
                        actual_total_steps = len(sigmas) - 1
                        steps_to_keep_trans = max(2, int(actual_total_steps * denoise_mult + 0.5))
                        steps_to_keep_trans = min(actual_total_steps, steps_to_keep_trans) 
                        tile_sigmas = sigmas[-(steps_to_keep_trans + 1):]

                    try:
                        sampled = sample_tile(
                            guider, positive, negative, sampler, tile_sigmas, latent, seed, tile_noise,
                            full_ref_tensor, crop_region, canvas_w)
                    finally:
                        if orig_forward is not None:
                            dm_to_restore.forward = orig_forward
                            dm_to_restore = None
                            orig_forward = None

                    if tiled_decode:
                        (decoded,) = vae_decoder_tiled.decode(vae, sampled, 512)
                    else:
                        (decoded,) = vae_decoder.decode(vae, sampled)

                    if color_match:
                        orig_tile_crop = upscaled_t[:, crop_region[1]:crop_region[3], crop_region[0]:crop_region[2], :]
                        if decoded.shape[1:3] != orig_tile_crop.shape[1:3]:
                            orig_tile_crop = orig_tile_crop.movedim(-1, 1)
                            orig_tile_crop = F.interpolate(orig_tile_crop, size=(decoded.shape[1], decoded.shape[2]), mode='bilinear', align_corners=False)
                            orig_tile_crop = orig_tile_crop.movedim(1, -1)
                            
                        matched = color_match_tensor(decoded, orig_tile_crop)
                        decoded = matched * 0.75 + decoded * 0.25

                    decoded_pil = tensor_to_pil(decoded)
                    canvas = composite_tile(canvas, decoded_pil, crop_region, orig_size, visual_mask)

                    pbar.update(1)
                    comfy.model_management.throw_exception_if_processing_interrupted()

                out_np = np.array(canvas).astype(np.float32) / 255.0
                out_t = torch.from_numpy(out_np).unsqueeze(0)

                if color_match:
                    out_t = color_match_tensor(out_t, upscaled_t)

                final_batch_outputs.append(out_t)
                
                # Manual memory cleanup post-batch iteration
                comfy.model_management.soft_empty_cache()

        finally:
            comfy.samplers.KSampler.sample = current_sample_method
            
            if dm_to_restore is not None and orig_forward is not None:
                try:
                    dm_to_restore.forward = orig_forward
                except Exception:
                    pass
            guider.set_conds(positive=positive, negative=negative)

        # Batch preservation guaranteed 
        final_out = torch.cat(final_batch_outputs, dim=0)
        return (final_out,)

# -----------------------------------------------------------------------------
# Entrypoint Registration
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "KleinTiledUpscaler": KleinTiledUpscalerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KleinTiledUpscaler": "Klein Tiled Upscaler",
}