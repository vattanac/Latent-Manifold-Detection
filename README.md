# AI-Generated Image Detection using Latent Manifolds

A complete, interactive technical guide explaining how latent autoencoders — the silent workhorses behind Stable Diffusion — can be repurposed as **training-free detectors** that separate real photographs from AI-generated images.

Built as a single self-contained HTML file with a modern 2025-style UI: dark glassmorphism theme, sticky left-side navigation, scroll-spy highlighting, gradient accents, and responsive layout.

---

## Quick Start

Open the guide directly in any modern browser:

```bash
# Option 1: Double-click the file
index.html

# Option 2: Open from terminal
open index.html      # macOS
xdg-open index.html  # Linux
start index.html     # Windows
```

From the landing page, click the **"View Visualizations"** button in the hero section
to jump into the interactive diagrams (`visualization.html`).

No server, no build step, no dependencies. Everything is inlined.

---

## What's Inside

The guide is organized into **19 sections across 6 categories**:

### Foundations
1. **Overview** — The core insight in one paragraph
2. **Autoencoders** — How encoders and decoders compress/reconstruct images
3. **Latent Diffusion** — Why Stable Diffusion works in compressed space
4. **The Manifold** — The curved "world of images" the decoder can reach

### Core Theory
5. **The Asymmetry** — Why real images sit off the manifold but generated ones sit on it
6. **Mathematical Formulation** — The scoring function S(x) = d(x, D(E(x)))
7. **Detection Pipeline** — 8-step workflow from image to verdict
8. **Distance Metrics** — L2, L1, LPIPS, SSIM, DreamSim, DISTS compared

### Signal Analysis
9. **Complex Regions** — Where reconstruction error concentrates (hair, foliage, text, skin)
10. **Frequency Domain** — DCT/FFT analysis of residuals

### Practice
11. **Code Example** — Full PyTorch implementation using Stable Diffusion's VAE and LPIPS
12. **Calibration** — Threshold selection, ROC curves, asymmetric error costs
13. **AEROBLADE Paper** — Historical context (Ricker et al., CVPR 2024)

### Limitations
14. **Failure Cases** — Cross-model generalization, compression, inpainting, post-editing
15. **Adversarial Attacks** — Gradient-based attacks and defenses
16. **Alternative Methods** — DIRE, CNNDetect, CLIP-based, watermarking, C2PA

### Beyond
17. **Applications** — Journalism, moderation, forensics, KYC, dataset hygiene
18. **Future Directions** — Universal autoencoders, patch-level scoring, video extension
19. **Glossary** — Key terms defined

---

## The Core Idea in 30 Seconds

Stable Diffusion generates images inside a compressed **latent space**, then decodes them through a pretrained **autoencoder**. Because every generated image is literally the output of this decoder, it sits *on* the autoencoder's **manifold** — the surface of images the decoder can produce.

Real photographs do *not* sit on that manifold. When you encode a real photo and decode it back, the autoencoder has to "round it off" to the nearest manifold point, losing detail in the process.

```
Real image   →  encode → decode  →  reconstruction error is LARGE
Fake image   →  encode → decode  →  reconstruction error is TINY
```

Measure that error, compare to a threshold, and you have a detector.

---

## Formula

```
S(x) = d(x, D(E(x)))

classify(x) = GENERATED  if S(x) < τ
              REAL       if S(x) ≥ τ
```

Where `E` is the encoder, `D` is the decoder, `d` is a distance metric (LPIPS works best), and `τ` is an empirically calibrated threshold.

---

## Minimal Code Example

```python
# pip install diffusers torch torchvision lpips pillow
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image
import lpips

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to("cuda").eval()
lpips_fn = lpips.LPIPS(net="vgg").to("cuda")

preprocess = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def reconstruction_score(image_path):
    img = Image.open(image_path).convert("RGB")
    x = preprocess(img).unsqueeze(0).to("cuda")
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mean
        x_hat = vae.decode(z).sample
        score = lpips_fn(x, x_hat).item()
    return score

THRESHOLD = 0.18
score = reconstruction_score("suspect.jpg")
label = "GENERATED" if score < THRESHOLD else "REAL"
print(f"Score={score:.4f}  →  {label}")
```

Expected score ranges on typical data:
- Generated images: **0.05 – 0.12**
- Real photographs: **0.20 – 0.40**

---

## Primary Source

**AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error**
Ricker, J., Lukovnikov, D., & Fischer, A.
*CVPR 2024.*

Related work: DIRE (Wang et al., ICCV 2023), CNNDetect (Wang et al., CVPR 2020), SeDID.

---

## Visualizations Page

A companion file, `visualization.html`, complements the main guide with five
standalone SVG diagrams that illustrate the core ideas visually:

1. **The Manifold Concept** — real vs. generated points around a curved manifold, with dashed arrows showing reconstruction distance.
2. **Encode → Decode Pipeline** — the full `S(x) = d(x, D(E(x)))` flow as labeled boxes.
3. **Score Distributions** — overlapping density curves for real and generated images with the decision threshold τ.
4. **Reconstruction Error Maps** — side-by-side residual heatmaps highlighting where the signal concentrates on real vs. generated images.
5. **Score Examples** — worked examples showing how two sample scores fall on either side of the threshold.

You can open `visualization.html` directly, or click the **"View Visualizations"**
button at the top of `index.html`.

---

## Features of the Guide

- **Modern UI** — Dark glassmorphism theme with gradient accents (purple → cyan → pink)
- **Sticky sidebar** — Always-visible navigation with active-section highlighting
- **Scroll spy** — Current section updates automatically as you scroll
- **Responsive** — Works on desktop, tablet, and mobile
- **Self-contained** — Single HTML file, no external assets, works offline
- **Rich visuals** — Pipeline diagrams, formula blocks, comparison tables, card layouts
- **Syntax-highlighted code** — PyTorch example with proper coloring
- **Reference tables** — Distance metrics, failure cases, applications

---

## When to Use This Method

Latent-manifold detection is a good choice when:

- You need a **training-free** solution
- You're detecting images from a **known generator family** (Stable Diffusion, SDXL, FLUX, etc.)
- You have access to the generator's autoencoder (or a closely related VAE)
- You can tolerate some cross-model weakness in exchange for simplicity

It is **not** the right choice when:

- You need to detect GAN output, autoregressive image models, or pixel-space diffusion
- Images have been heavily compressed, resized, or post-edited
- Adversaries may have white-box access to the detector
- You need to localize the AI-generated region within a hybrid image (use patch-level methods instead)

For production deployments, **combine this detector with a CLIP-based classifier and a frequency-domain check** — no single method dominates across all conditions.

---

## File Structure

```
Latent-Manifold-Detection/
├── index.html           # The interactive guide (open this)
├── visualization.html   # Five SVG visualizations of the core ideas
└── README.md            # This file
```

---

## License & Credits

Educational material. Built as a comprehensive reference for understanding the latent-manifold approach to AI image detection. All technical claims are traceable to AEROBLADE (CVPR 2024) and related published research.