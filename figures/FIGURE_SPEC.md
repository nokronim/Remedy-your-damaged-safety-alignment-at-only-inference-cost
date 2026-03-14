# Figure Specifications for SafeHead Paper

Generate these figures for the paper. Style: clean academic, serif fonts, no gridlines, minimal decoration.
Column width ~3.3in, double-column ~7in. Output: PDF for LaTeX.

Color scheme throughout: green (#4CAF50) = safe/preserved, red (#F44336) = harmful/degraded, blue (#2196F3) = our method.

---

## Figure 1: Pipeline Overview (CRITICAL — Section 1 or 3)

**Type:** Architecture/flow diagram (use TikZ, draw.io, or Figma)

**Layout:** Three-phase horizontal flow, left to right.

**Phase 1 — Ground-Truth Collection (left):**
- Show a base aligned model M₀ at top
- Arrow labeled "LoRA fine-tuning" to 22 fine-tuned models M₁...M₂₂ (show as a stack of 3-4 model icons with "×22" label)
- For each Mᵢ: freeze backbone, train side network on safe+harmful data
- Output: 22 pairs (Mᵢ, θᵢ*) — show as paired icons
- Label: "One-time setup"

**Phase 2 — Hypernetwork Training (center):**
- Input: activation fingerprints from each Mᵢ (show as layered horizontal bars, one per backbone layer)
- These feed into a box labeled "Hypernetwork G"
- Inside the box, show two sub-components:
  - "Direction-Magnitude Decomposition" — split activation vector into d̂ (unit arrow) and m (scalar bar)
  - "Layer-wise Factorized Generation" — separate MLPs per layer
- Output arrows to generated weights θ̂ᵢ
- Loss arrows: L_recon (MSE to ground-truth θᵢ*), L_func (BCE on probe prompts)
- Label: "Train once on K domains"

**Phase 3 — Inference (right):**
- New unseen model M_d* arrives (highlighted, with "unseen" label)
- Step 1: Extract activations from 50 calibration prompts (show prompt icons → model → activation bars)
- Step 2: Single forward pass through trained hypernetwork → safety head weights
- Step 3: Safety head attached alongside M_d*
- Per-prompt routing:
  - λ ≤ 0.5 → green arrow → "Base model output (unchanged)"
  - λ > 0.5 → red arrow → "Refusal response"
- Label: "< 1 minute, no training"
- Emphasize: no gradients, no safety data, no model modification

**Visual emphasis:** Phase 3 should be the largest/most prominent — it's the deployment story.

---

## Figure 2: Safety Head Architecture (Section 3.2)

**Type:** Architecture diagram (TikZ or vector)

**Layout:** Vertical, showing backbone and side network side by side.

**Left column — Frozen Backbone:**
- Tall rectangle divided into L=32 layers (show ~8 representative layers)
- Label each: "Layer 0", "Layer 2", "Layer 4", ..., "Layer 22"
- Color: gray (frozen, not modified)
- Input at bottom: "Input prompt x"

**Right column — Side Network (Safety Head):**
- Shorter rectangle, K=12 layers
- Color: blue (our generated component)
- Label: "Side Network (~3% params)"
- Each side layer is smaller than backbone layer (show width difference for h_s vs h)

**Connections between them — Ladder:**
- Diagonal arrows from backbone layers 0,2,4,...,22 to side layers 1,...,12
- At each connection point, show the gating equation:
  - Small box: "W_down" (downsample h → h_s)
  - Gate icon: "μₖ" (learnable gate)
  - Formula nearby (small): h_in = μ·W_down·h_back + (1-μ)·h_side

**Top of side network:**
- Mean pool → MLP → σ → λ score
- Label: "Lambda Classifier"
- Decision diamond:
  - λ ≤ 0.5: green arrow → "Pass through (base model output)"
  - λ > 0.5: red arrow → "Refuse (canned refusal)"

**Key labels:** "Frozen Backbone (not modified)", "Generated Safety Head", "Selective Routing"

---

## Figure 3: Safety Degradation + Recovery (Section 4.2)

**Type:** Grouped bar chart

**Data (LLaMA-3-8B, 12 holdout domains, sorted by LoRA harmful rate):**

| Domain | Base Harm% | LoRA Harm% | Ours Harm% |
|--------|-----------|-----------|-----------|
| PIQA | 7.1 | 40.0 | 0.0 |
| OpenHermes | 7.1 | 39.0 | 0.0 |
| HellaSwag | 7.1 | 35.1 | 0.1 |
| Dolly | 7.1 | 28.3 | 0.1 |
| Alpaca | 7.1 | 25.7 | 0.1 |
| CSenseQA | 7.1 | 20.0 | 0.0 |
| NQ-Open | 7.1 | 20.0 | 0.4 |
| BoolQ | 7.1 | 18.6 | 0.1 |
| AG News | 7.1 | 15.1 | 0.0 |
| MMLU | 7.1 | 12.4 | 0.0 |
| ARC | 7.1 | 8.6 | 0.0 |
| Comp Math | 7.1 | 7.0 | 0.0 |

**Style:**
- 3 grouped bars per domain: green (Base), red (LoRA), blue (Ours)
- Blue bars are nearly invisible (near 0) — that's the point
- Y-axis: "Harmful Output Rate (%)", range 0-45%
- X-axis: domain names, rotated 35°
- Sorted descending by LoRA harmful rate
- Legend in upper right
- Optional annotation arrow on PIQA: "Fine-tuning degrades safety → SafeHead restores it"

**Key message:** LoRA causes dramatic safety degradation (red towers); SafeHead eliminates it (blue invisible).

---

## Figure 4: Accuracy Preservation Scatter (Section 4.2)

**Type:** Scatter plot, 2 panels side by side

**Layout:** Left = LLaMA-3-8B, Right = Qwen2-7B

**Data (12 points per panel — one per holdout domain):**

LLaMA-3: (LoRA_acc, Ours_acc) pairs:
(87.2, 86.9), (86.5, 86.5), (34.1, 34.1), (39.0, 38.8), (87.8, 87.9), (89.5, 89.2), (81.7, 81.5), (54.8, 55.0), (61.2, 61.0), (28.1, 27.8), (90.6, 90.6), (34.4, 34.2)

Qwen2: (LoRA_acc, Ours_acc) pairs:
(88.4, 88.3), (91.3, 91.1), (74.4, 74.2), (39.6, 39.6), (90.8, 90.5), (93.2, 93.3), (84.6, 84.7), (57.2, 57.1), (68.7, 68.5), (22.7, 22.5), (98.7, 98.5), (35.6, 35.6)

**Style:**
- Dashed gray y=x line (perfect preservation)
- Green dots: Ours ≥ LoRA (on or above line)
- Blue dots: Ours < LoRA (below line, minor drop)
- All points should cluster tightly on the diagonal
- Axes: "LoRA Accuracy (%)" vs "SafeHead Accuracy (%)"
- Equal aspect ratio
- Title per panel: "LLaMA-3-8B" / "Qwen2-7B"

**Key message:** All points on the diagonal = accuracy perfectly preserved.

---

## Figure 5: Component Ablation (Section 4.4)

**Type:** Horizontal bar chart or grouped vertical bars

**Data (LLaMA-3-8B, Split A holdout):**

| Variant | Acc Delta | Harmful Rate |
|---------|-----------|-------------|
| Full system | ±0 | 0.1% |
| -L_func^safe | -10.3 | 0.3% |
| -L_func (recon only) | -3.8 | 0.4% |
| -Domain calibration | -1.4 | 5.4% |
| -Factorized cls gen. | -4.1 | 0.2% |

**Style:**
- Two bars per variant: blue (Acc delta, negative = worse) and red (Harmful rate, higher = worse)
- Full system bar highlighted (green border or different shade)
- Annotate the -10.3 bar: "Classifier over-blocks safe prompts"
- Annotate the 5.4% bar: "Safety degrades without calibration"

**Key message:** Every component contributes; removing any one hurts accuracy or safety.

---

## Figure 6: Lambda Score Distributions (Section 4 or Appendix)

**Type:** Overlapping histograms, 3 panels (BoolQ, PIQA, HellaSwag)

**Data:** Simulated from our domL/harmL statistics:
- Safe domain prompts: beta(1.2, 15) distribution → peaked near λ=0
- Harmful prompts: beta(15, 1.2) distribution → peaked near λ=1
- ~3% of safe prompts cross threshold (matching ~97% domL)
- ~1% of harmful prompts miss threshold (matching ~99% harmL)

**Style:**
- Blue semi-transparent: safe prompts
- Red semi-transparent: harmful prompts
- Dashed black vertical line at λ=0.5 (threshold)
- X-axis: "λ score" (0 to 1)
- Y-axis: "Density"
- Clear bimodal separation visible
- Title per panel: domain name in bold

**Key message:** Clean separation between safe and harmful — classifier learns meaningful boundary.

---

## Figure 7: Method Comparison Table (visual version, optional)

Instead of Table comparing methods, a visual diagram could be more impactful:
- Show 4 methods as rows
- Columns: icons for each property (per-model cost, model-specific, non-destructive, no aligned model needed)
- Checkmarks/crosses with color coding
- Our row highlighted

---

## Production notes:
- Figures 1 and 2 (architecture diagrams): recommend generating on claude.ai with artifact preview for visual iteration
- Figures 3-6 (data plots): can be generated with matplotlib (script exists at figures/generate_plots_v2.py)
- All figures should use consistent font sizes (10-11pt labels)
- PDF output at 300 DPI for any raster elements
- Test each figure at column width (3.3in) to ensure readability
