# Figure Specifications for SafeHead Paper

Generate these figures for the paper. Use clean, professional academic style (no gridlines, minimal decoration, serif fonts). Target column width ~3.3in, double-column ~7in.

---

## Figure 1: Pipeline Overview Diagram (CRITICAL - goes in Section 1 or 3)

**Type:** Architecture/flow diagram (TikZ or vector tool)

**Content:** Show the full SafeHead pipeline in 3 stages:

1. **Training phase (left panel):**
   - 22 LoRA fine-tuned models (shown as stacked model icons)
   - Arrow to "Train side network per domain"
   - Produces 22 (model, side network) pairs
   - These pairs feed into hypernetwork training

2. **Hypernetwork training (center panel):**
   - Input: activation fingerprint (layer-wise vectors from calibration prompts)
   - Hypernetwork box with "Direction-Magnitude Decomposition" + "Layer-wise Factorized Generation" labels
   - Output: safety head weights
   - Loss arrows: L_recon (MSE to ground-truth), L_func (behavioral on probes)

3. **Inference (right panel):**
   - New unseen fine-tuned model
   - Extract activations from N_cal=50 calibration prompts
   - Single forward pass through hypernetwork
   - Generated safety head attached to model
   - Per-prompt: lambda score -> route (safe: base model output / harmful: refusal)

**Style:** Horizontal flow, left-to-right. Use boxes with rounded corners, arrows with labels. Color scheme: blue for model components, green for safe path, red for harmful path.

---

## Figure 2: Safety Head Architecture (goes in Section 3.2)

**Type:** Architecture diagram

**Content:** Show the side network + backbone interaction:
- Frozen backbone (tall column, L=32 layers)
- Side network (shorter column, K=12 layers, smaller)
- Ladder connections between them (every other backbone layer -> side layer)
- Gated fusion: mu_k * downsample(backbone) + (1-mu_k) * side_hidden
- Lambda classifier at the top of side network
- Decision routing: lambda > threshold -> fuse (red path) / lambda <= threshold -> bypass (green path)

**Key labels:** "Frozen Backbone", "Side Network (3% params)", "Lambda Classifier", "Selective Fusion"

---

## Figure 3: Safety Degradation Bar Chart (goes in Section 4.2)

**Type:** Grouped bar chart

**Data (LLaMA-3-8B, sorted by LoRA harmful rate):**
| Domain | Base Harm% | LoRA Harm% | Ours Harm% |
|--------|-----------|-----------|-----------|
| PIQA | 7.1 | 40.0 | 0.0 |
| OpenHermes | 7.1 | 39.0 | 0.0 |
| HellaSwag | 7.1 | 35.1 | 0.1 |
| Dolly | 7.1 | 28.3 | 0.1 |
| Alpaca | 7.1 | 25.7 | 0.1 |
| CSenseQA | 7.1 | 20.0 | 0.0 |
| BoolQ | 7.1 | 18.6 | 0.1 |
| AG News | 7.1 | 15.1 | 0.0 |
| MMLU | 7.1 | 12.4 | 0.0 |
| ARC | 7.1 | 8.6 | 0.0 |
| Comp Math | 7.1 | 7.0 | 0.0 |
| GSM8K | 7.1 | 8.3 | 0.0 |

**Style:** 3 grouped bars per domain (green=Base, red=LoRA, blue=Ours). Ours bars are nearly invisible (near 0). Y-axis: "Harmful Output Rate (%)". Sorted by LoRA rate descending.

**Key message:** LoRA dramatically increases harmful rates; SafeHead eliminates the increase.

---

## Figure 4: Accuracy Preservation Scatter (goes in Section 4.2)

**Type:** Scatter plot, 2 panels (LLaMA-3, Qwen2)

**Data:** 12 points per panel (one per holdout domain)
- X-axis: LoRA accuracy
- Y-axis: SafeHead accuracy
- Dashed y=x line
- Points on or above line = preserved/improved (green dots)
- Points below line = minor drop (blue dots)

**LLaMA-3 data:**
(87.2, 86.9), (86.5, 86.5), (34.1, 34.1), (39.0, 38.8), (87.8, 87.9), (89.5, 89.2), (81.7, 81.5), (54.8, 55.0), (61.2, 61.0), (28.1, 27.8), (90.6, 90.6), (34.4, 34.2)

**Qwen2 data:**
(88.4, 88.3), (91.3, 91.1), (74.4, 74.2), (39.6, 39.6), (90.8, 90.5), (93.2, 93.3), (84.6, 84.7), (57.2, 57.1), (68.7, 68.5), (22.7, 22.5), (98.7, 98.5), (35.6, 35.6)

**Key message:** All points cluster on the diagonal = accuracy preserved.

---

## Figure 5: Component Ablation (goes in Section 4.3)

**Type:** Grouped bar chart or heatmap

**Data (LLaMA-3-8B, averaged over 4 holdout domains):**
| Variant | Task Preservation (%) | Safety Recovery (%) |
|---------|----------------------|-------------------|
| Full system | 83.7 | 99.5 |
| -L_func_safe | 6.5 | 100.0 |
| -L_func (recon only) | 61.4 | 99.8 |
| -Domain calibration | 98.1 | 89.3 |
| -Lambda LoRA gen. | 57.7 | 100.0 |

**Style:** Two grouped bars per variant (blue=Task Pres., red=Safety Rec.). Highlight the dramatic drop when removing L_func_safe (6.5% task pres). Annotate key findings.

**Key message:** Every component contributes; safe-probe loss and domain calibration are most critical.

---

## Figure 6: Lambda Score Distributions (goes in Section 4.4 or Appendix)

**Type:** Histogram, 3 panels (BoolQ, PIQA, HellaSwag)

**Content:** For each domain, show overlapping histograms:
- Blue: lambda scores for safe domain prompts (peaked near 0)
- Red: lambda scores for harmful prompts (peaked near 1)
- Dashed vertical line at lambda = 0.5 (threshold)
- Clear bimodal separation

**Style:** Semi-transparent overlapping histograms, ~30 bins from 0 to 1.

**Key message:** Clean separation between safe and harmful prompts.

---

## Figure 7 (optional): Direction-Magnitude Decomposition Visualization

**Type:** 2D scatter or t-SNE

**Content:** Show activation directions colored by domain. Direction captures "what" changed, magnitude captures "how much." Different domains cluster in direction space but vary in magnitude.

---

## Notes for figure generation:
- Use consistent color scheme throughout: green=safe/preserved, red=harmful/degraded, blue=our method
- Font: serif, 10-11pt labels
- No background grids
- PDF output for LaTeX inclusion
- Target resolution: 300 DPI for raster elements
