"""
Klein Tiled Upscaler - Self-contained tiling node for Flux2.Klein
Features Single-Pass Smoothstep Matrix Inpainting with advanced alignment.

"""

import logging
import math
import numpy as np
import torch
import torch.nn.functional as F

import comfy.utils
import comfy.model_management

log = logging.getLogger("KleinTiledUpscaler")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def expand_and_align_crop(region, width, height, target_w, target_h):
    target_w = min(target_w, (width // 32) * 32)
    target_h = min(target_h, (height // 32) * 32)

    x1, y1, x2, y2 = region
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    new_x1 = cx - (target_w // 2)
    new_y1 = cy - (target_h // 2)

    new_x1 = (new_x1 // 32) * 32
    new_y1 = (new_y1 // 32) * 32

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
    t = target.movedim(-1, 1)
    s = source.movedim(-1, 1).to(device=t.device, dtype=t.dtype)

    t_mean = t.mean(dim=(2, 3), keepdim=True)
    t_std = t.std(dim=(2, 3), keepdim=True) + 1e-6
    s_mean = s.mean(dim=(2, 3), keepdim=True)
    s_std = s.std(dim=(2, 3), keepdim=True) + 1e-6

    ratio = torch.clamp(s_std / t_std, 0.75, 1.33)
    matched = (t - t_mean) * ratio + s_mean
    matched = torch.clamp(matched, 0.0, 1.0)
    return matched.movedim(1, -1)


def lanczos_resize(t_bhwc, width, height):
    s = t_bhwc.movedim(-1, 1)
    s = comfy.utils.common_upscale(s, width, height, "lanczos", "disabled")
    return s.movedim(1, -1)

# -----------------------------------------------------------------------------
# Separable blend profiles
# -----------------------------------------------------------------------------
# The smoothstep matrix mask is an outer product of two 1D profiles; a gaussian
# blur of an outer product equals the outer product of the blurred profiles.
# The same profiles, average-pooled by the VAE divisor, give the latent-space
# blend mask -- so pixel and latent compositing are geometrically identical.

def _smooth_curve(length):
    t = np.linspace(0, 1, length, dtype=np.float32)
    return 0.5 - 0.5 * np.cos(np.pi * t)


def _gaussian_blur_1d(arr, sigma):
    if sigma <= 0:
        return arr
    radius = max(1, int(sigma * 3.0 + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(arr, radius, mode='edge')
    out = np.convolve(padded, kernel, mode='same')
    return out[radius:-radius].astype(np.float32)


def _axis_profile(canvas_len, core_a, core_b, blend, mask_blur):
    profile = np.zeros(canvas_len, dtype=np.float32)
    a = max(0, core_a - blend)
    b = min(canvas_len, core_b + blend)
    if a >= b:
        return profile
    grad = np.ones(b - a, dtype=np.float32)
    blend_lo = core_a - a
    blend_hi = b - core_b
    if blend_lo > 0:
        grad[:blend_lo] = _smooth_curve(blend_lo)
    if blend_hi > 0:
        grad[-blend_hi:] = _smooth_curve(blend_hi)[::-1]
    profile[a:b] = grad
    return _gaussian_blur_1d(profile, mask_blur)


def _pool_profile(profile, divisor):
    n = (len(profile) // divisor) * divisor
    return profile[:n].reshape(-1, divisor).mean(axis=1)


def make_blend_masks(canvas_w, canvas_h, core_x1, core_y1, core_x2, core_y2,
                     blend, mask_blur, crop_region, vae_divisor):
    xp = _axis_profile(canvas_w, core_x1, core_x2, blend, mask_blur)
    yp = _axis_profile(canvas_h, core_y1, core_y2, blend, mask_blur)
    cx1, cy1, cx2, cy2 = crop_region

    pixel_mask = torch.from_numpy(np.outer(yp[cy1:cy2], xp[cx1:cx2])).float()
    pixel_mask = pixel_mask.unsqueeze(0).unsqueeze(-1)  # [1, h, w, 1] BHWC

    xl = _pool_profile(xp, vae_divisor)
    yl = _pool_profile(yp, vae_divisor)
    return pixel_mask, xl, yl

# -----------------------------------------------------------------------------
# Tile preparation
# -----------------------------------------------------------------------------

def prepare_tile(canvas_t, core_x1, core_y1, actual_tw, actual_th, padding,
                 canvas_w, canvas_h, full_tile_w=None, full_tile_h=None):
    x1 = max(core_x1 - padding, 0)
    y1 = max(core_y1 - padding, 0)
    x2 = min(core_x1 + actual_tw + padding, canvas_w)
    y2 = min(core_y1 + actual_th + padding, canvas_h)
    crop_region = (x1, y1, x2, y2)

    use_tw = full_tile_w if full_tile_w is not None else actual_tw
    use_th = full_tile_h if full_tile_h is not None else actual_th
    target_w = math.ceil((use_tw + padding * 2) / 32) * 32
    target_h = math.ceil((use_th + padding * 2) / 32) * 32

    crop_region, tile_size = expand_and_align_crop(crop_region, canvas_w, canvas_h, target_w, target_h)
    cx1, cy1, cx2, cy2 = crop_region

    tile = canvas_t[:, cy1:cy2, cx1:cx2, :]
    original_size = (cx2 - cx1, cy2 - cy1)  # (w, h)

    if (tile.shape[2], tile.shape[1]) != tile_size:
        tile = lanczos_resize(tile, tile_size[0], tile_size[1])

    return tile, crop_region, tile_size, original_size

# -----------------------------------------------------------------------------
# Conditioning and sampling
# -----------------------------------------------------------------------------

def patch_conditioning(cond, ref_crop):
    # REPLACE semantics: a stale full-image reference alongside the tile crop
    # causes per-tile color/tone drift.
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


def conditioning_has_reference(cond):
    for c in cond:
        if isinstance(c, (list, tuple)) and len(c) == 2 and isinstance(c[1], dict):
            if c[1].get("reference_latents"):
                return True
    return False


def set_guider_conds(guider, positive, negative):
    try:
        guider.set_conds(positive=positive, negative=negative)
    except TypeError:
        guider.set_conds(positive=positive)


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


def make_tile_noise(latent_image, seed, xi, yi, cols):
    # Fallback / consistent_noise=False path: grid-keyed, strategy-independent.
    idx = yi * cols + xi
    tile_seed = (seed * 0x9E3779B97F4A7C15 + idx * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    g = torch.Generator(device="cpu").manual_seed(tile_seed)
    return torch.randn(latent_image.shape, generator=g, dtype=torch.float32).to(latent_image.device)


def crop_noise_field(noise_field, crop_region, vae_divisor, lat_shape):
    """Crop the full-canvas noise field at this tile's latent coordinates.
    Overlapping tiles share identical noise in shared regions."""
    cx1, cy1 = crop_region[0], crop_region[1]
    lx1 = cx1 // vae_divisor
    ly1 = cy1 // vae_divisor
    h, w = lat_shape[2], lat_shape[3]
    if (ly1 + h > noise_field.shape[2]) or (lx1 + w > noise_field.shape[3]):
        return None  # caller falls back to per-tile noise
    return noise_field[:, :, ly1:ly1 + h, lx1:lx1 + w].clone()


def sample_tile(guider, positive, negative, sampler, sigmas, latent, seed, tile_noise,
                raw_latent, crop_region, canvas_w):
    latent_image = latent["samples"]
    crop = crop_latent_for_tile(raw_latent, crop_region, canvas_w)
    patched_pos = patch_conditioning(positive, crop)
    set_guider_conds(guider, patched_pos, negative)

    noise_mask = latent.get("noise_mask", None)

    samples = guider.sample(
        tile_noise, latent_image, sampler, sigmas,
        denoise_mask=noise_mask, callback=None,
        disable_pbar=False, seed=seed
    )
    return samples

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
                "scale_factor":  ("FLOAT",  {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.25, "tooltip": "Upscale ratio applied before tiling refine."}),
                "tiling_strategy": (cls.TILING_STRATEGIES, {"default": "Detail-First", "tooltip": "Tile sequence order (Detail-First analyzes variance)."}),
                "tile_size_mode": (["Auto", "Manual"], {"default": "Auto", "tooltip": "Auto divides canvas evenly; Manual allows custom height/width below."}),
                "tile_width":    ("INT",    {"default": 1024, "min": 512, "max": 4096, "step": 32, "tooltip": "Width of individual tiles when Manual mode is selected."}),
                "tile_height":   ("INT",    {"default": 1024, "min": 512, "max": 4096, "step": 32, "tooltip": "Height of individual tiles when Manual mode is selected."}),
                "padding":       ("INT",    {"default": 128, "min": 0, "max": 512, "step": 16}),
                "color_match":   ("BOOLEAN", {"default": True, "tooltip": "Matches colors dynamically against original image to avoid color drift."}),
                "mask_blur":     ("INT",    {"default": 32, "min": 0, "max": 64, "step": 1, "tooltip": "Blur radius applied to blend masks to remove seams."}),
                "adaptive_tiling": ("BOOLEAN", {"default": False, "tooltip": "Dynamically scales down sampling steps on flatter tiles to optimize speed and reduce hallucinations"}),
                "skip_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Only works with Detail-First. Skip tiles whose detail variance is below this fraction of the reference variance (keeps bicubic content). 0.0 = never skip."}),
                "core_anchor":   ("FLOAT",  {"default": 1.0, "min": 0.5, "max": 1.0, "step": 0.01, "tooltip": "Lower = keep more original structure in tile core. 1.0 = full regen. 0.85 is subtle, 1.0-0.95 is the sweet spot."}),
                "consistent_noise": ("BOOLEAN", {"default": True, "tooltip": "Sample all tiles from one shared full-canvas noise field. Overlapping regions get identical noise. Off = independent per-tile noise."}),
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
                color_match, mask_blur, adaptive_tiling, core_anchor,
                consistent_noise=True, skip_threshold=0.0, upscale_model=None):

        if upscale_model is not None:
            import comfy_extras.nodes_upscale_model as upscale_nodes

        if conditioning_has_reference(positive):
            log.warning("[KLEIN] Incoming positive conditioning already carries reference_latents; "
                        "they will be replaced per-tile with local crops (required for tiling).")

        batch_size = image.shape[0]
        final_batch_outputs = []
        final_batch_latents = []

        try:
            for b in range(batch_size):
                log.info(f"[KLEIN] Processing Batch Element {b+1}/{batch_size}")

                img_b = image[b:b+1]

                if upscale_model is not None:
                    upscaler_node = upscale_nodes.ImageUpscaleWithModel()
                    upscaled_t = upscaler_node.upscale(upscale_model, img_b)[0]
                else:
                    upscaled_t = img_b.clone()

                target_w = (round(img_b.shape[2] * scale_factor) // 32) * 32
                target_h = (round(img_b.shape[1] * scale_factor) // 32) * 32

                upscaled_t = upscaled_t.movedim(-1, 1)
                upscaled_t = F.interpolate(upscaled_t.float(), size=(target_h, target_w), mode='bicubic', antialias=True)
                upscaled_t = torch.clamp(upscaled_t, 0.0, 1.0).movedim(1, -1)

                canvas_w, canvas_h = upscaled_t.shape[2], upscaled_t.shape[1]
                canvas_t = upscaled_t.detach().to("cpu", dtype=torch.float32).clone()

                log.info(f"[KLEIN] Canvas: {canvas_w}x{canvas_h}")

                raw_latent = vae.encode(upscaled_t[:, :, :, :3])
                latent_divisor = max(1, round(canvas_w / raw_latent.shape[3]))
                log.info(f"[KLEIN] Latent Divisor: {latent_divisor}")

                # LATENT output canvas: starts as bicubic encode, refined tiles
                # get blended in at latent resolution (no extra encodes).
                latent_canvas = raw_latent.clone()

                # Spatially consistent noise field: one noise tensor for the
                # whole canvas; tiles crop their window from it, so overlapping
                # regions share identical noise (anti-ghosting).
                noise_field = None
                if consistent_noise:
                    g = torch.Generator(device="cpu").manual_seed(seed)
                    noise_field = torch.randn(
                        (raw_latent.shape[0], raw_latent.shape[1], raw_latent.shape[2], raw_latent.shape[3]),
                        generator=g, dtype=torch.float32)

                # --- Tile layout: identical to original implementation ---
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
                log.info(f"[KLEIN] Grid: {rows}x{cols} = {total} tiles ({tile_size_mode} mode), strategy={tiling_strategy}")

                # --- Variance analysis ---
                gray_lr = (img_b[..., 0] * 0.299 + img_b[..., 1] * 0.587 + img_b[..., 2] * 0.114).unsqueeze(1).contiguous()
                blur_kernel = torch.ones((1, 1, 3, 3), dtype=gray_lr.dtype, device=gray_lr.device) / 9.0
                gray_lr = F.conv2d(gray_lr, blur_kernel, padding=1)

                lap_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=gray_lr.dtype, device=gray_lr.device).view(1, 1, 3, 3)
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
                    tiles_order = [t for t in tiles_order if (t[0]+t[1]) % 2 == 0] + [t for t in tiles_order if (t[0]+t[1]) % 2 == 1]
                elif tiling_strategy == "Reverse Chess":
                    tiles_order = [t for t in tiles_order if (t[0]+t[1]) % 2 == 1] + [t for t in tiles_order if (t[0]+t[1]) % 2 == 0]
                elif tiling_strategy == "Spiral":
                    ccx, ccy = (cols - 1) / 2.0, (rows - 1) / 2.0
                    tiles_order.sort(key=lambda t: (t[0]-ccx)**2 + (t[1]-ccy)**2)
                elif tiling_strategy == "Detail-First":
                    tiles_order.sort(key=lambda t: tile_variances[(t[0], t[1])], reverse=True)

                pbar = comfy.utils.ProgressBar(total)

                for step_i, (xi, yi, core_x1, core_y1, actual_tw, actual_th) in enumerate(tiles_order):
                    # Optional flat-tile skip (inert at 0.0). Canvas + latent
                    # canvas keep their consistent bicubic content there.
                    var_ratio = min(1.0, tile_variances[(xi, yi)] / ref_var)

                    if skip_threshold > 0.0 and var_ratio < skip_threshold:
                        log.info(f"[KLEIN] Tile {step_i+1}/{total} ({xi},{yi}) low detail "
                                 f"({var_ratio:.2f} < {skip_threshold:.2f}) -> not refined; "
                                 f"upscaled content kept (already used as reference/padding by detail tiles)")
                        pbar.update(1)
                        if tiling_strategy == "Detail-First":
                            remaining = total - (step_i + 1)
                            if remaining > 0:
                                log.info(f"[KLEIN] Detail-First cutoff: {remaining} remaining low-detail tile(s) "
                                         f"left as upscale (ground-truth anchor)")
                                pbar.update(remaining)
                            break
                        continue

                    log.info(f"[KLEIN] Tile {step_i+1}/{total} ({xi},{yi}) core=({core_x1},{core_y1}) size={actual_tw}x{actual_th}")

                    tile_t, crop_region, tile_size, orig_size = prepare_tile(
                        canvas_t, core_x1, core_y1, actual_tw, actual_th, padding,
                        canvas_w, canvas_h, full_tile_w=current_tile_width, full_tile_h=current_tile_height)

                    core_x2 = core_x1 + actual_tw
                    core_y2 = core_y1 + actual_th

                    visual_blend_size = max(16, padding - 64) if padding >= 64 else (padding // 2)
                    pixel_mask, x_lat_profile, y_lat_profile = make_blend_masks(
                        canvas_w, canvas_h, core_x1, core_y1, core_x2, core_y2,
                        visual_blend_size, mask_blur, crop_region, latent_divisor)

                    latent_samples = vae.encode(tile_t[:, :, :, :3])
                    latent = {"samples": latent_samples}

                    tile_h_lat, tile_w_lat = latent_samples.shape[2], latent_samples.shape[3]

                    latent_expand_px = max(0, padding - 32)
                    latent_expand_size = (latent_expand_px // 32) * 32 if padding >= 32 else 0

                    l_x1_px = max(0, core_x1 - latent_expand_size)
                    l_y1_px = max(0, core_y1 - latent_expand_size)
                    l_x2_px = min(canvas_w, core_x2 + latent_expand_size)
                    l_y2_px = min(canvas_h, core_y2 + latent_expand_size)

                    vae_divisor = max(1, round(tile_t.shape[2] / tile_w_lat))
                    l_x1_lat = max(0, (l_x1_px - crop_region[0]) // vae_divisor)
                    l_y1_lat = max(0, (l_y1_px - crop_region[1]) // vae_divisor)
                    l_x2_lat = min(tile_w_lat, (l_x2_px - crop_region[0]) // vae_divisor)
                    l_y2_lat = min(tile_h_lat, (l_y2_px - crop_region[1]) // vae_divisor)

                    latent_mask = torch.zeros((1, 1, tile_h_lat, tile_w_lat), dtype=torch.float32, device=latent_samples.device)
                    latent_mask[0, 0, l_y1_lat:l_y2_lat, l_x1_lat:l_x2_lat] = core_anchor
                    latent["noise_mask"] = latent_mask

                    tile_noise = None
                    if noise_field is not None:
                        tile_noise = crop_noise_field(noise_field, crop_region, latent_divisor, latent_samples.shape)
                        if tile_noise is not None:
                            tile_noise = tile_noise.to(latent_samples.device)
                    if tile_noise is None:
                        tile_noise = make_tile_noise(latent_samples, seed, xi, yi, cols)

                    tile_sigmas = sigmas
                    if adaptive_tiling:
                        # Ramp floor = skip line, so the flattest *kept* tiles get
                        # minimum steps and detail rises smoothly to full at ref_var.
                        # No sharpness cliff between skipped and barely-refined tiles.
                        floor = skip_threshold if skip_threshold > 0.0 else 0.0
                        ramp = (var_ratio - floor) / max(1e-6, 1.0 - floor)
                        ramp = max(0.0, min(1.0, ramp))
                        denoise_mult = 0.35 + 0.65 * ramp
                        actual_total_steps = len(sigmas) - 1
                        steps_to_keep = max(2, int(actual_total_steps * denoise_mult + 0.5))
                        steps_to_keep = min(actual_total_steps, steps_to_keep)
                        tile_sigmas = sigmas[-(steps_to_keep + 1):]
                        log.info(f"[KLEIN] Adaptive denoise factor: {denoise_mult:.2f} "
                                 f"({steps_to_keep}/{actual_total_steps} steps, ratio={var_ratio:.2f})")

                    samples = sample_tile(
                        guider, positive, negative, sampler, tile_sigmas, latent, seed, tile_noise,
                        raw_latent, crop_region, canvas_w)

                    # --- Latent compositing: same smoothstep geometry, latent res ---
                    cx1, cy1, cx2, cy2 = crop_region
                    lx1, ly1 = cx1 // latent_divisor, cy1 // latent_divisor
                    s = samples.to(device=latent_canvas.device, dtype=latent_canvas.dtype)
                    sh = min(s.shape[2], latent_canvas.shape[2] - ly1)
                    sw = min(s.shape[3], latent_canvas.shape[3] - lx1)
                    if sh > 0 and sw > 0:
                        lat_mask = torch.from_numpy(
                            np.outer(y_lat_profile[ly1:ly1 + sh], x_lat_profile[lx1:lx1 + sw])
                        ).to(device=latent_canvas.device, dtype=latent_canvas.dtype).unsqueeze(0).unsqueeze(0)
                        region_l = latent_canvas[:, :, ly1:ly1 + sh, lx1:lx1 + sw]
                        latent_canvas[:, :, ly1:ly1 + sh, lx1:lx1 + sw] = torch.lerp(region_l, s[:, :, :sh, :sw], lat_mask)

                    # --- Pixel compositing ---
                    decoded = vae.decode(samples)

                    if color_match:
                        orig_tile_crop = upscaled_t[:, cy1:cy2, cx1:cx2, :]
                        if decoded.shape[1:3] != orig_tile_crop.shape[1:3]:
                            orig_tile_crop = orig_tile_crop.movedim(-1, 1)
                            orig_tile_crop = F.interpolate(orig_tile_crop, size=(decoded.shape[1], decoded.shape[2]), mode='bilinear', align_corners=False)
                            orig_tile_crop = orig_tile_crop.movedim(1, -1)

                        matched = color_match_tensor(decoded, orig_tile_crop)
                        decoded = matched * 0.75 + decoded * 0.25

                    decoded = decoded.detach().to("cpu", dtype=torch.float32).clamp(0.0, 1.0)

                    if (decoded.shape[2], decoded.shape[1]) != orig_size:
                        decoded = lanczos_resize(decoded, orig_size[0], orig_size[1]).clamp(0.0, 1.0)

                    region = canvas_t[:, cy1:cy2, cx1:cx2, :]
                    canvas_t[:, cy1:cy2, cx1:cx2, :] = torch.lerp(region, decoded, pixel_mask)

                    pbar.update(1)
                    comfy.model_management.throw_exception_if_processing_interrupted()

                out_t = canvas_t

                if color_match:
                    out_t = color_match_tensor(out_t, upscaled_t)
                out_t = out_t.clamp(0.0, 1.0)

                final_batch_outputs.append(out_t)
                final_batch_latents.append(latent_canvas.detach().cpu())
                comfy.model_management.soft_empty_cache()

        finally:
            try:
                set_guider_conds(guider, positive, negative)
            except Exception:
                pass

        final_out = torch.cat(final_batch_outputs, dim=0)
        final_latent_dict = {"samples": torch.cat(final_batch_latents, dim=0)}

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