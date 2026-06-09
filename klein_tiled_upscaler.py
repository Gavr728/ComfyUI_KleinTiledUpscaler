"""
Klein Tiled Upscaler - Self-contained tiling node for Flux2.Klein
Features Single-Pass Smoothstep Matrix Inpainting with advanced alignment.
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn.functional as F

import comfy.samplers
import comfy.utils
import comfy.model_management

# -----------------------------------------------------------------------------
# Helpers 
# -----------------------------------------------------------------------------

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
    return x1, y1, x2, y2

def expand_and_align_crop(region, width, height, target_w, target_h):
    # Ensure target sizes do not exceed canvas dimensions aligned to 32
    target_w = min(target_w, (width // 32) * 32)
    target_h = min(target_h, (height // 32) * 32)
    
    x1, y1, x2, y2 = region
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    
    # Calculate base unaligned crop positions
    new_x1 = cx - (target_w // 2)
    new_y1 = cy - (target_h // 2)
    
    # Align starting coordinates strictly to 32
    new_x1 = (new_x1 // 32) * 32
    new_y1 = (new_y1 // 32) * 32
    
    # Shift starting coordinates if out of bounds to maintain target size exactly (no resizing)
    if new_x1 < 0:
        new_x1 = 0
    elif new_x1 + target_w > width:
        new_x1 = (width - target_w) // 32 * 32
        
    if new_y1 < 0:
        new_y1 = 0
    elif new_y1 + target_h > height:
        new_y1 = (height - target_h) // 32 * 32
        
    new_x2 = new_x1 + target_w
    new_y2 = new_y1 + target_h
    
    return (new_x1, new_y1, new_x2, new_y2), (target_w, target_h)

def color_match_tensor(target, source):
    # Strict Linear Color Match with stabilized variance scaling
    t = target.movedim(-1, 1) 
    s = source.movedim(-1, 1).to(device=t.device, dtype=t.dtype) 
    
    t_mean = t.mean(dim=(2, 3), keepdim=True)
    t_std = t.std(dim=(2, 3), keepdim=True) + 1e-6
    s_mean = s.mean(dim=(2, 3), keepdim=True)
    s_std = s.std(dim=(2, 3), keepdim=True) + 1e-6
    
    # Clamp standard deviation ratios to prevent contrast mismatch seams on flat/low-texture tiles
    ratio = torch.clamp(s_std / t_std, 0.75, 1.33)
    matched = (t - t_mean) * ratio + s_mean
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
            new_c[1]["reference_latents"] = [ref_crop.clone()]
            new_c[1]["reference_latents_method"] = "index"
            new_cond.append(new_c)
        else:
            new_cond.append(c)
    return new_cond

def crop_latent_for_tile(full_latent, crop_region, canvas_w):
    x1, y1, x2, y2 = crop_region
    latent_h, latent_w = full_latent.shape[2], full_latent.shape[3]
    
    scale_x = canvas_w / latent_w
    scale_y = scale_x
    canvas_h = round(latent_h * scale_x)
    
    rx1 = int(round(x1 / canvas_w * latent_w))
    rx2 = int(round(x2 / canvas_w * latent_w))
    ry1 = int(round(y1 / canvas_h * latent_h))
    ry2 = int(round(y2 / canvas_h * latent_h))
    
    rx1 = max(0, rx1)
    ry1 = max(0, ry1)
    rx2 = min(rx2, latent_w)
    ry2 = min(ry2, latent_h)
    rx2 = max(rx1 + 1, rx2)
    ry2 = max(ry1 + 1, ry2)
    
    raw = full_latent[:, :, ry1:ry2, rx1:rx2]
    enc_w = max(1, round((x2 - x1) / scale_x))
    enc_h = max(1, round((y2 - y1) / scale_y))
    
    if raw.shape[2] != enc_h or raw.shape[3] != enc_w:
        raw = F.interpolate(raw.float(), size=(enc_h, enc_w), mode='bilinear', align_corners=False).to(full_latent.dtype)
    return raw.clone()

def make_tile_noise(latent_image, seed, tile_index):
    g = torch.Generator(device="cpu").manual_seed(seed + tile_index)
    return torch.randn(latent_image.shape, generator=g, dtype=torch.float32).to(latent_image.device)


def sample_tile(guider, positive, negative, sampler, sigmas, latent, seed, tile_noise,
               raw_latent, crop_region, canvas_w):
    latent_image = latent["samples"]
    crop = crop_latent_for_tile(raw_latent, crop_region, canvas_w)
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
                "seed":          ("INT",    {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "scale_factor":  ("FLOAT",  {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.25,"tooltip": "Upscale ratio applied before tiling refine."}),
                "tiling_strategy": (cls.TILING_STRATEGIES, {"default": "Detail-First", "tooltip": "Tile sequence order (Detail-First analyzes variance)."}),
                "tile_size_mode": (["Auto", "Manual"], {"default": "Auto", "tooltip": "Auto divides canvas evenly; Manual allows custom height/width below."}),
                "tile_width":    ("INT",    {"default": 1024, "min": 512, "max": 4096, "step": 16, "tooltip": "Width of individual tiles when Manual mode is selected."}),
                "tile_height":   ("INT",    {"default": 1024, "min": 512, "max": 4096, "step": 16, "tooltip": "Height of individual tiles when Manual mode is selected."}),
                "padding":       ("INT",    {"default": 128, "min": 0, "max": 512, "step": 16}),
                "color_match":   ("BOOLEAN", {"default": True, "tooltip": "Matches colors dynamically against original image to avoid color drift."}),
                "mask_blur":     ("INT",    {"default": 32, "min": 0, "max": 64, "step": 1, "tooltip": "Blur radius applied to blend masks to remove seams."}),
                "adaptive_tiling": ("BOOLEAN", {"default": False, "tooltip": "Dynamically scales down sampling steps on flatter tiles to optimize speed and reduce hallucinations"}),
                "core_anchor":   ("FLOAT",  {"default": 1.0, "min": 0.5, "max": 1.0, "step": 0.01, "tooltip": "Lower = keep more original structure in tile core. 1.0 = full regen. 0.85 is subtle, 1.0-0.95 is the sweet spot."}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("IMAGE", "LATENT")
    FUNCTION = "upscale"
    CATEGORY = "sampling/custom_sampling"

    def upscale(self, guider, positive, negative, sampler, sigmas, vae, image,
                seed, scale_factor, tiling_strategy, tile_size_mode, tile_width, tile_height, padding,
                color_match, mask_blur, adaptive_tiling, core_anchor, upscale_model=None):

        batch_size = image.shape[0]
        final_batch_outputs = []
        final_batch_latents = []

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

        try:
            # Bypass legacy Tiled Diffusion monkeypatch
            comfy.samplers.KSampler.sample = native_comfy_sample
            
            for b in range(batch_size):
                print(f"[KLEIN] Processing Batch Element {b+1}/{batch_size}")
                
                img_b = image[b:b+1] # Ensure dimensionality remains [1, H, W, C]
                
                if upscale_model is not None:
                    import comfy_extras.nodes_upscale_model as upscale_nodes
                    upscaler_node = upscale_nodes.ImageUpscaleWithModel()
                    upscaled_t = upscaler_node.upscale(upscale_model, img_b)[0]
                else:
                    upscaled_t = img_b.clone()
                
                target_w = (round(img_b.shape[2] * scale_factor) // 32) * 32
                target_h = (round(img_b.shape[1] * scale_factor) // 32) * 32
                
                upscaled_t = upscaled_t.movedim(-1, 1)
                
                orig_dtype = upscaled_t.dtype
                upscaled_t = F.interpolate(upscaled_t.float(), size=(target_h, target_w), mode='bicubic', antialias=True)
                upscaled_t = torch.clamp(upscaled_t, 0.0, 1.0).to(orig_dtype).movedim(1, -1)

                canvas_w, canvas_h = upscaled_t.shape[2], upscaled_t.shape[1]
                canvas_np = (upscaled_t[0].cpu().numpy() * 255).astype(np.uint8)
                canvas = Image.fromarray(canvas_np)

                print(f"[KLEIN] Canvas: {canvas_w}x{canvas_h}")

                # Core VAE API direct call for encoding
                raw_latent = vae.encode(upscaled_t[:, :, :, :3])
                latent_divisor = round(canvas_w / raw_latent.shape[3])
                print(f"[KLEIN] Latent Divisor: {latent_divisor}")

                # Initialize the full canvas latent representation
                latent_canvas = raw_latent.clone()

                if tile_size_mode == "Auto":
                    cols = max(1, round(canvas_w / 1024))
                    rows = max(1, round(canvas_h / 1024))
                    current_tile_width = max(256, round((canvas_w / cols) / 32) * 32)
                    current_tile_height = max(256, round((canvas_h / rows) / 32) * 32)
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
                        
                        if actual_tw < 256 and canvas_w >= 256:
                            shift = 256 - actual_tw
                            core_x1 = max(0, core_x1 - shift)
                            actual_tw = 256
                        if actual_th < 256 and canvas_h >= 256:
                            shift = 256 - actual_th
                            core_y1 = max(0, core_y1 - shift)
                            actual_th = 256
                            
                        if actual_tw > 0 and actual_th > 0:
                            tiles_order.append((xi, yi, core_x1, core_y1, actual_tw, actual_th))

                total = len(tiles_order)

                print(f"[KLEIN] Grid: {rows}x{cols} = {total} tiles (Auto Mode: {tile_size_mode}), strategy={tiling_strategy}")

                gray_lr = (img_b[..., 0] * 0.299 + img_b[..., 1] * 0.587 + img_b[..., 2] * 0.114).unsqueeze(1).contiguous()
                blur_kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=img_b.device) / 9.0
                blur_kernel = blur_kernel.to(device=gray_lr.device, dtype=gray_lr.dtype)
                gray_lr = F.conv2d(gray_lr, blur_kernel, padding=1)
                
                lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=img_b.device).view(1, 1, 3, 3)
                lap_kernel = lap_kernel.to(device=gray_lr.device, dtype=gray_lr.dtype)
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
                    print(f"[KLEIN] Tile {step_i+1}/{total} ({xi},{yi}) core=({core_x1},{core_y1}) size={actual_tw}x{actual_th}")

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
                    
                    # Direct VAE API call for tile encoding (RGB-cropped)
                    latent_samples = vae.encode(tile_t[:, :, :, :3])
                    latent = {"samples": latent_samples}
                    
                    tile_h_lat, tile_w_lat = latent_samples.shape[2], latent_samples.shape[3]
                    
                    latent_expand_px = max(0, padding - 32)
                    latent_expand_size = (latent_expand_px // 32) * 32 if padding >= 32 else 0
                    
                    l_x1_px = max(0, core_x1 - latent_expand_size)
                    l_y1_px = max(0, core_y1 - latent_expand_size)
                    l_x2_px = min(canvas_w, core_x2 + latent_expand_size)
                    l_y2_px = min(canvas_h, core_y2 + latent_expand_size)
                    
                    vae_divisor = max(1, round(tile_pil.width / tile_w_lat))
                    l_x1_lat = max(0, (l_x1_px - crop_region[0]) // vae_divisor)
                    l_y1_lat = max(0, (l_y1_px - crop_region[1]) // vae_divisor)
                    l_x2_lat = min(tile_w_lat, (l_x2_px - crop_region[0]) // vae_divisor)
                    l_y2_lat = min(tile_h_lat, (l_y2_px - crop_region[1]) // vae_divisor)
                    
                    # Enforce strict 4D format [batch, channel, height, width]
                    latent_mask = torch.zeros((1, 1, tile_h_lat, tile_w_lat), dtype=torch.float32, device=latent_samples.device)
                    latent_mask[0, 0, l_y1_lat:l_y2_lat, l_x1_lat:l_x2_lat] = core_anchor
                    latent["noise_mask"] = latent_mask

                    tile_noise = make_tile_noise(latent_samples, seed, step_i)

                    tile_sigmas = sigmas
                    if adaptive_tiling:
                        var_ratio = min(1.0, tile_variances[(xi, yi)] / ref_var)
                        denoise_mult = 0.35 + 0.65 * var_ratio
                        actual_total_steps = len(sigmas) - 1
                        steps_to_keep_trans = max(2, int(actual_total_steps * denoise_mult + 0.5))
                        steps_to_keep_trans = min(actual_total_steps, steps_to_keep_trans) 
                        tile_sigmas = sigmas[-(steps_to_keep_trans + 1):]
                        print(f"[KLEIN] Adaptive denoise factor: {denoise_mult:.2f} (running {steps_to_keep_trans}/{actual_total_steps} steps)")

                    sampled = sample_tile(
                        guider, positive, negative, sampler, tile_sigmas, latent, seed, tile_noise,
                        raw_latent, crop_region, canvas_w)

                    # Composite the sampled latent tile directly back into the full-canvas latent
                    lh, lw = raw_latent.shape[2], raw_latent.shape[3]
                    sx = canvas_w / lw
                    sy = canvas_h / lh
                    rx1 = int(round(crop_region[0] / sx))
                    ry1 = int(round(crop_region[1] / sy))
                    
                    s = sampled["samples"]
                    sh, sw = s.shape[2], s.shape[3]
                    
                    # Clip boundaries to prevent precision errors
                    if ry1 + sh > lh:
                        sh = lh - ry1
                    if rx1 + sw > lw:
                        sw = lw - rx1
                    
                    latent_canvas[:, :, ry1:ry1+sh, rx1:rx1+sw] = s[:, :, :sh, :sw]

                    # Decode the individual tile using regular vae.decode
                    decoded = vae.decode(sampled["samples"])

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
                final_batch_latents.append(latent_canvas)
                comfy.model_management.soft_empty_cache()

        finally:
            comfy.samplers.KSampler.sample = current_sample_method
            guider.set_conds(positive=positive, negative=negative)

        # Batch preservation guaranteed 
        final_out = torch.cat(final_batch_outputs, dim=0)
        final_latent = torch.cat(final_batch_latents, dim=0).detach().cpu()
        final_latent_dict = {"samples": final_latent}

        return (final_out, final_latent_dict)

# -----------------------------------------------------------------------------
# Entrypoint Registration
# -----------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "KleinTiledUpscaler": KleinTiledUpscalerNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "KleinTiledUpscaler": "Klein Tiled Upscaler",
}