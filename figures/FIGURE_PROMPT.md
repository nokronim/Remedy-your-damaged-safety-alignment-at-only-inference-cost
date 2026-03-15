# Prompts for Claude Web — Generate SVG Figures

Copy each prompt below into Claude web. Ask for an SVG artifact. Iterate until satisfied, then save the SVG.

---

## Figure 1: Pipeline Overview

```
Create a clean, professional SVG diagram for an academic ML paper (700×500 viewBox). Use serif font (Times New Roman). No emojis. Muted, professional color palette.

The diagram shows a 2-row layout:

TOP ROW — "Training (one-time)" inside a light gray rounded container:

Left section "Collect Pairs":
- A single box "Aligned LLM" at top
- Arrow down labeled "LoRA"
- A stack of 3 slightly offset rounded boxes labeled "Fine-tuned Models" (showing there are many)
- Arrow down labeled "train"
- A stack of 3 slightly offset rounded boxes in a different color labeled "Safety Heads"
- These two stacks are the training pairs

Middle section "Train Hypernetwork":
- Box "Activation Fingerprint" at top
- Splits into two smaller boxes "Direction" and "Magnitude" via two arrows
- Both feed into a prominent larger box "Hypernetwork G" (this is the key component, make it stand out with a slightly darker/bolder style)
- Below it: "Generated Weights" box

Right section "Losses":
- Small panel showing three lines of text:
  "L_recon — weight matching"
  "L_cls — classifier accuracy"
  "L_func — behavioral probes"
- Connected to the hypernetwork with a thin dashed line

Dashed arrows connect left section to middle section (pairs feed hypernetwork training).

BOTTOM ROW — "Inference (per model)" inside a light green/teal container:
- Linear left-to-right flow with arrows between each box:
  "New Model (unseen)" → "Extract Activations" → "Hypernetwork G (frozen)" [dashed border] → "Safety Head" → "Route"
- "Route" splits into two paths:
  - Down-left to green pill "Safe output"
  - Down-right to red/coral pill "Refuse"

Between the two rows: a banner/pill shape saying "Deploy in seconds — no training needed"

Style: Use soft, muted fills (not saturated). Thin borders. Rounded corners (8px). Small, clean arrows. Professional academic look, similar to NeurIPS/ICML papers. White background.
```

---

## Figure 2: Architecture

```
Create a clean, professional SVG diagram for an academic ML paper (680×420 viewBox). Use serif font (Times New Roman). Muted professional colors.

The diagram shows the safety head architecture with a HORIZONTAL layout:

TOP BAND — "Frozen Backbone":
- A wide horizontal rounded rectangle containing 14 small squares arranged in a row (representing transformer layers)
- Label above: "Frozen backbone (L layers)"
- Arrow entering from the left labeled "Input x"
- Arrow exiting to the right (to LM output)
- Use a muted purple/lavender color scheme for the backbone

DIAGONAL DASHED ARROWS:
- 6 diagonal dashed arrows going from selected backbone layer squares DOWN to the side network squares below
- These represent "ladder connections" with gated fusion
- Use a teal/green color for these arrows
- Small label near one arrow: "Gated fusion μ_k"

BOTTOM BAND — "Side Network":
- A shorter horizontal rounded rectangle (about 60% width of backbone) containing 11 smaller squares
- Label to the right: "Side network — K layers, ~3% params"
- Use a teal/green color scheme (lighter than the arrows)
- Positioned below and slightly left-aligned with the backbone

BELOW SIDE NETWORK:
- Arrow going down from the center of the side network, labeled "Pool + MLP"
- Arrow leads to a prominent rounded box "Safety Score λ" (use warm amber/gold color to make it the focal point)

ROUTING (below safety score):
- Two curved arrows diverging from the safety score box:
  - Left curve (green) → green rounded pill "Pass through" with subtitle "Base model output unchanged"
  - Right curve (red/coral) → coral rounded pill "Refuse" with subtitle "Safety head intervenes"
- Labels on the curves: "safe" (green) and "harmful" (coral)

Style: Muted fills, thin borders, rounded corners. The safety score box should be the visual focal point (slightly warmer/bolder color). Everything else should be subtle. White background. No grid lines.
```

---

## Figure 3: Bar Chart (Safety Degradation)

```
Create a clean SVG grouped bar chart for an academic paper (680×320 viewBox). Serif font.

Data (12 domains sorted by LoRA harmful rate, descending):
PIQA: Base=7.1, LoRA=40.0, Ours=0.0
OpenHermes: 7.1, 39.0, 0.0
HellaSwag: 7.1, 35.1, 0.1
Dolly: 7.1, 28.3, 0.1
Alpaca: 7.1, 25.7, 0.1
CSenseQA: 7.1, 20.0, 0.0
NQ-Open: 7.1, 20.0, 0.4
BoolQ: 7.1, 18.6, 0.1
AG News: 7.1, 15.1, 0.0
MMLU: 7.1, 12.4, 0.0
ARC: 7.1, 8.6, 0.0
Math: 7.1, 7.0, 0.0

Three grouped bars per domain:
- Green (#7CB97A) = Base (aligned model)
- Red/coral (#E07A6A) = LoRA (fine-tuned, safety degraded)
- Blue (#5B9BD5) = Ours (SafeHead, safety restored)

Y-axis: "Harmful Output Rate (%)", range 0 to 45
X-axis: domain names, rotated 35-40 degrees
Legend: horizontal, top-right corner inside the plot

The "Ours" bars should be nearly invisible (they're all 0-0.4%). That contrast is the visual point — tall red bars vs invisible blue bars.

No grid lines. Clean axis lines only. White background. Thin bars with small gaps.
```

---

## Figure 4: Scatter Plot (Accuracy Preservation)

```
Create a clean SVG scatter plot for an academic paper, two panels side by side (680×280 viewBox). Serif font.

Left panel: "LLaMA-3-8B"
Right panel: "Qwen2-7B"

Each panel has:
- X-axis: "LoRA Accuracy (%)"
- Y-axis: "SafeHead Accuracy (%)"
- A dashed gray diagonal line (y=x, representing perfect preservation)
- 12 blue filled circles (radius 4px, with thin white border)

LLaMA-3 data points (x, y):
(87.2, 86.9), (86.5, 86.5), (34.1, 34.1), (39.0, 38.9),
(87.8, 87.7), (89.5, 89.1), (81.7, 81.5), (54.8, 54.6),
(61.2, 61.0), (28.1, 27.7), (90.6, 90.6), (34.4, 34.4)

Qwen2 data points (x, y):
(88.4, 88.2), (91.3, 91.3), (74.4, 74.1), (39.6, 39.5),
(90.8, 90.6), (93.2, 92.8), (84.6, 84.5), (57.2, 57.0),
(68.7, 68.4), (22.7, 22.2), (97.6, 97.6), (35.6, 35.4)

Key visual: all points should cluster tightly on or just below the diagonal line, showing accuracy is preserved.

Equal aspect ratio for both panels. No grid, just axis lines. Panel titles in bold. Muted blue for dots (#5B9BD5). White background.
```

---

## Tips for iterating on Claude web:
- Start with "Create an SVG artifact" to get the preview
- Say "make the colors more muted" if too saturated
- Say "increase spacing between elements" if crowded
- Say "make the hypernetwork box more prominent" for emphasis
- Say "use thinner lines and borders" for a cleaner look
- Say "match the style of NeurIPS papers" for academic look
- When done, copy the SVG code and save as .svg file
