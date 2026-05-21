# Klein Tiled Upscaler for ComfyUI

A highly optimized, self-contained, context-aware tiling upscale node specifically engineered for high-fidelity generation in **Flux.1 (including Flux-Klein)** models. 

This node resolves the classic visual issues associated with tiled upscaling (such as blocky edge color shifts, halo artifacts, and detailed tiles mismatched against smooth backgrounds) without introducing PyTorch Out-of-Memory (OOM) errors.

---

## Key Features

### 1. Symmetrical Canvas Partitioning (`Auto` Mode)
Traditional tilers often leave an odd-sized, smaller partial tile at the right and bottom edges of the image, causing poor boundary integration. 
* **Auto Slicing:** Dynamically divides your canvas into perfectly identical, equal-sized tiles near your target size [2.1.2]. 
* For example, a $3584 \times 2784$ canvas is partitioned into exactly four equal $896$-pixel columns and three equal $928$-pixel rows [2.1.2]. Symmetrical tile dimensions result in completely uniform structural generations.

### 2. Ground-Truth Laplacian Detail Analysis
Instead of basic pixel variance (which is easily tricked by smooth sky gradients), this node runs a GPU-accelerated $3\times3$ Laplacian edge convolution over the grayscale representation of your **original low-resolution input image** [2]:
* **No Bicubic Noise Inflation:** Bypasses the high-frequency ringing and overshoots created by bicubic upscaling [2].
* **3x3 Pre-Convolution Blur:** Smooths out microscopic sensor noise and JPEG compression artifacts before analysis [2]. 
* **The Result:** The node correctly differentiates between flat sky/walls (which drop smoothly to 2 steps) and detailed elements (which run at 4 steps), preventing flat areas from generating unwanted detailed "hallucinations" [1.1].

### 3. Symmetrical Crop Sizes
Even when edge tiles are positioned near boundaries, pass-through parameters force every single crop passed to the VAE Encoder to remain **100% identical in size** [1.1]. This completely eliminates shape-mismatch artifacts during latent processing.

### 4. Dynamic RoPE Alignment
Automatically shifts Flux's Rotary Position Embedding (RoPE) coordinates dynamically on the model's forward pass to match each tile's exact coordinate location [1.1]. This ensures that structural elements seamlessly align across tile boundaries [1.1].

---

## Installation

Navigate to your ComfyUI `custom_nodes/` directory and clone this repository:

```bash
cd custom_nodes
git clone https://github.com/YOUR_GITHUB_USERNAME/ComfyUI_KleinTileGuider.git
```

Restart ComfyUI, and the node will be available under `sampling/custom_sampling`.

---

## Parameter Settings Guide

* **`tile_size_mode` (Auto / Manual):** 
  * `Auto` (highly recommended): Automatically calculates equal, perfectly symmetrical tile boundaries [2.1.2].
  * `Manual`: Allows you to set custom `tile_width` and `tile_height` values.
* **`tiling_strategy` (Detail-First / Spiral / Chess / Linear):**
  * `Detail-First` (highly recommended for Flux): Analyzes the scene and processes the highly textured tiles first [1.1.2]. Flat zones (skies, background walls) are processed last, allowing them to anchor cleanly to the finalized, sharp boundaries of the foreground details [1.1.2].
* **`color_match` (True / False):**
  * Performs linear histogram matching of each tile against the original upscaled canvas [3]. This completely eliminates visible blocky lighting variations, contrast drifts, or color blocks across seams [3].
* **`adaptive_tiling` (True / False):**
  * Dynamically reduces denoiser steps in low-detail zones (skies/walls) to save render time [1.1]. Flat skies/walls scale down to 50% steps (2 steps), while detailed zones keep 100% steps (4 steps) [1.1].
* **`tiled_decode` (True / False):**
  * Decodes the latent canvas in tiles. Turn this on if you are upscaling to extremely high resolutions (8k+) to prevent GPU VRAM OOM crashes [1.1].

---

## License

This software is distributed under a custom **Non-Commercial License**. Free use is granted for personal, non-commercial, research, and hobbyist projects. Any commercial deployment, hosting on commercial generation platforms, or use for paid contract work requires written permission from the copyright holder. See `LICENSE.txt` for details.