# Klein Tiled Upscaler for ComfyUI

A highly optimized, self-contained, **inpainting-based** tiling upscale node specifically engineered for **creative upscaling** in **Flux2.Klein** specifically.


> **Vibecoded Disclaimer:** This custom node is 100% vibecoded. The author does not know how to code. This entire repository was built by prompting LLMs. If something breaks, please copy the `klein_tiled_upscaler.py` file along with your console logs, feed them into an LLM (like Claude or GPT), and ask it to help you fix it. But it would be helpful if you share your findings. Also perhaps you know how to code and found an obvious way to improve this node that I have no idea about, you can share it so I can later go ask Gemini to implement it. 

---

## Why An Inpainting-Based Tile Upscaler?

Unlike purely mathematical patch-mixers, this node runs a sequential, single-pass **inpainting pipeline** on a tiled grid. It is designed specifically for **creative upscaling**—where you don't just want to sharpen existing pixels, but rather want the AI to imagine rich, coherent new high-frequency details (organic textures, surface imperfections, realistic grain) within each tile. It also vram friednly. 

---

## Key Features

### 1. Symmetrical Canvas Partitioning (`Auto` Mode)
Traditional tilers often leave an odd-sized, smaller partial tile at the right and bottom edges of the image, causing poor boundary integration. 
* **Auto Slicing:** Dynamically divides your canvas into perfectly identical, equal-sized tiles near your target size. 
* For example, a $3584 \times 2784$ canvas is partitioned into exactly four equal $896$-pixel columns and three equal $928$-pixel rows. Symmetrical tile dimensions result in completely uniform structural generations.

### 2. Built-In Reference Latents
Symmetrical reference latent extraction, model patching, and structural guidance are handled natively inside the node's execution pipeline.

### 3. Ground-Truth Laplacian Detail Analysis
Instead of basic pixel variance (which is easily tricked by smooth sky gradients), this node runs a GPU-accelerated $3\times3$ Laplacian edge convolution over the grayscale representation of your **original low-resolution input image**:
* **No Bicubic Noise Inflation:** Bypasses the high-frequency ringing and overshoots created by bicubic upscaling.
* **3x3 Pre-Convolution Blur:** Smooths out microscopic sensor noise and JPEG compression artifacts before analysis. 
* **The Result:** The node correctly differentiates between flat sky/walls (which drop smoothly to 2 steps) and detailed elements (which run at 4 steps), preventing flat areas from generating unwanted detailed "hallucinations".

### 4. Symmetrical Crop Sizes
Even when edge tiles are positioned near boundaries, pass-through parameters force every single crop passed to the VAE Encoder to remain **100% identical in size**. This completely eliminates shape-mismatch artifacts during latent processing.

### 5. VRAM-Friendly Architecture
Because this node processes the canvas sequentially tile-by-tile instead of merging massive global attention maps or holding multiple high-resolution noise layers in memory simultaneously, its VRAM footprint remains low. This allows you to generate massive, high-fidelity upscales even on budget GPUs.

### 6. LoRa's support
All loras for Flux2.Klein should work as expected. Including loras for upscaling, consistency, style.

---

## Limitations & Testing Configuration

* **Prompt Sensitivity:** Highly descriptive or structurally-mismatching prompts can cause stylistic tile drift. Keep prompts focused on the overall material and details of the scene. A basic prompt like `"upscale this image"` works great, whereas a complex prompt describing specific objects in one corner can cause those objects to hallucinate in other tiles.
* **Development Disclaimer:** For development and debugging reasons, the vast majority of tests and calibrations were made with a basic, fast configuration: **4 steps, Euler sampler, CFG 1.0 (Guidance 1.0), 1024 tile size, and 2x upscale**. 
* **Beyond Defaults:** If you go outside these values (e.g. running 20+ steps, higher CFG/guidance models, or extreme 8x upscales), you may encounter unexpected rendering behaviors, contrast shifts, or alignment quirks that I have not accounted for.

---

## Comparison With Other Methods

* **Standard Hi-Res Fix (Latent Upscale):** Upscales the entire latent space at once. While visually coherent, it can causes Out-of-Memory (OOM) crashes on high target resolutions.
* **Ultimate SD Upscale:** Runs sequential pixel-space blending. It frequently struggles with tile boundaries, grid seams, and completely lacks advanced RoPE alignment essential for Flux2.Klein.
* **SDXL Tile ControlNet Upscalers:** Relies on a ControlNet Tile model to guide boundaries. ControlNet Tile models do not exist natively or performantly for Flux2.Klein.
* **SeedVR2 Upscaler:** While SeedVR2 produces incredible blur removal, it is exceptionally computationally heavy, slow to run, and highly VRAM-intensive. Klein Tiled Upscaler runs exceptionally fast with a fraction of the VRAM usage. But if you want you can use it with this node, just use 1x upscale and connect the image that was upscaled with SeedVR.

---

## Installation

Navigate to your ComfyUI `custom_nodes/` directory and clone this repository:

```bash
cd custom_nodes
git clone https://github.com/Gavr728/ComfyUI_KleinTiledUpscaler.git
```

Restart ComfyUI, and the node will be available in the ComfyUI right-click search menu as **Klein Tiled Upscaler**.

---

## Parameter Settings Guide

* **`tile_size_mode` (Auto / Manual):** 
  * `Auto` (recommended): Automatically calculates equal, perfectly symmetrical tile boundaries.
  * `Manual`: Allows you to set custom `tile_width` and `tile_height` values.
* **`tiling_strategy` (Detail-First / Spiral / Chess / Linear):**
  * `Detail-First`: Analyzes the scene and processes the highly textured tiles first. Flat zones (skies, walls) are processed last, allowing them to anchor cleanly to the finalized, sharp boundaries of the foreground details.
* **`color_match` (True / False):**
  * Performs linear histogram matching of each tile against the original upscaled canvas. This should eliminate visible blocky lighting variations, contrast drifts, or color blocks across seams.
* **`adaptive_tiling` (True / False):**
  * Dynamically reduces denoiser steps in low-detail zones (skies/walls) to save render time. Flat skies/walls scale down to 50% steps (2 steps), while detailed zones keep 100% steps (4 steps).
* **`tiled_decode` (True / False):**
  * Decodes the latent canvas in tiles. Turn this on if you are upscaling to extremely high resolutions (8k+) to prevent GPU VRAM OOM crashes.Might increase color difference between tiles.

---

## Troubleshooting & Visible Seams

If you see visible grid lines, tile boxes, or color transitions in your output, go through this checklist:
1. **Enable Color Match:** Ensure `color_match` is set to `True`. This locks the luminance and contrast of the tiles to the original reference.
2. **Increase Mask Blur:** Increase `mask_blur` to `48` or `64`. This softens the physical crossfade mask boundaries.
3. **Use Detail-First Strategy:** Set your `tiling_strategy` to `Detail-First`. This forces the generator to build sharp foreground structures first, establishing anchor points for skies and walls to blend into later.
4. **Turn Off/On Adaptive Tiling:** In my test it helps with eliminating some visible difference between tiles. But the sudden shift in steps (e.g. 4 steps vs 2 steps) can occasionally cause minor contrast transitions on difficult images. Turning it `False` forces all tiles to run at uniform step counts, guaranteeing perfect rendering consistency.Also this mode can introduce visible noise that wasn't denoized by the model.
5. **Consitency loras:** You can use loras for consistency, they will work as expected. But will limit the upscaler strength. Also some upscaler/fix details loras for Flux2.Klein have some consistency capabilities built in, so you can try them. If you do you should probably increase the steps, at 4 steps the effect was very minor. 

**Known bugs**. Sometimes, with a specifically 3x upscale factor, the model begins to heavily hallucinate and lose any context of reference latent. I could not find the exact reason why. Some images work fine, some don't at all. One thing I discovered is that it is happening with the q8 model but doesn't with the INT8 model.

---

## License

This software is distributed under a custom **Non-Commercial License**. Free use is granted for personal, non-commercial, research, and hobbyist projects. Any commercial deployment, hosting on commercial generation platforms, or use for paid contract work requires written permission from the copyright holder. See `LICENSE.txt` for details.