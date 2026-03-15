# Prompts for Claude Web — Generate SVG Figures

Copy each prompt into Claude web and ask for an SVG artifact. Iterate until it looks good.

---

## Figure 1: Pipeline Overview

```
Design a visually appealing SVG figure for an academic ML paper showing the SafeHead pipeline. This should look like a figure from a top NeurIPS/ICML paper — clean, modern, professional.

The system has two phases:

TRAINING PHASE (one-time, offline):
1. We start with an aligned base LLM and create many fine-tuned variants via LoRA
2. For each fine-tuned model, we train a small "safety head" — a classifier that detects harmful prompts
3. This gives us pairs of (fine-tuned model, safety head)
4. We extract "activation fingerprints" from each model — these are layer-wise activation vectors that capture how fine-tuning changed the model
5. Each fingerprint is decomposed into Direction (what changed) and Magnitude (how much)
6. A hypernetwork G learns to predict safety head weights from these fingerprints
7. Three training losses guide this: L_recon (weight matching), L_cls (classifier accuracy), L_func (behavioral probes)

INFERENCE PHASE (per new model, takes seconds):
1. Given any new fine-tuned model (never seen before)
2. Extract its activation fingerprint from a few calibration prompts
3. One forward pass through the frozen hypernetwork produces a safety head
4. The safety head classifies each prompt: safe prompts pass through unchanged, harmful prompts get refused
5. No training, no gradients, no safety data needed

Make this visually clear and elegant. Use muted, professional colors. Serif font. The training phase and inference phase should be visually distinct. The hypernetwork should be the visual focal point. Show the flow clearly with arrows. 700x500 viewBox.
```

---

## Figure 2: Safety Head Architecture

```
Design a visually appealing SVG figure for an academic paper showing how the safety head works alongside a frozen LLM backbone.

Key components to show:

- A frozen transformer backbone with L layers (show as a horizontal band of small blocks). This is the original fine-tuned model — it's not modified at all.

- A smaller "side network" with K layers (fewer and smaller blocks than the backbone). This is the safety head — only ~3% of the backbone's parameters.

- "Ladder connections" — diagonal dashed arrows from selected backbone layers down to corresponding side network layers. These carry downsampled hidden states through learnable gates (gated fusion).

- At the end of the side network: a Pool + MLP step that produces a single "Safety Score" (this should be visually prominent — it's the key output)

- The safety score routes the input:
  - If safe → "Pass through" (base model output unchanged)
  - If harmful → "Refuse" (safety head intervenes)

The backbone should look large and solid (but frozen/gray). The side network should look lightweight and active. The safety score should be the visual focal point of the diagram. The routing at the bottom should clearly show the two paths.

Use a horizontal layout. Muted professional colors. Serif font. 680x420 viewBox. Style it like a figure from a top ML conference paper.
```

---

## Figure 3: Safety Degradation Bar Chart

```
Create a clean, publication-quality SVG bar chart showing how LoRA fine-tuning degrades LLM safety and how our method (SafeHead) restores it.

Data for LLaMA-3-8B across 12 domains (sorted by LoRA harmful rate):

Domain          | Base | LoRA | SafeHead
PIQA            | 7.1  | 40.0 | 0.0
OpenHermes      | 7.1  | 39.0 | 0.0
HellaSwag       | 7.1  | 35.1 | 0.1
Dolly           | 7.1  | 28.3 | 0.1
Alpaca          | 7.1  | 25.7 | 0.1
CommonsenseQA   | 7.1  | 20.0 | 0.0
NQ-Open         | 7.1  | 20.0 | 0.4
BoolQ           | 7.1  | 18.6 | 0.1
AG News         | 7.1  | 15.1 | 0.0
MMLU            | 7.1  | 12.4 | 0.0
ARC             | 7.1  | 8.6  | 0.0
Comp. Math      | 7.1  | 7.0  | 0.0

Y-axis: "Harmful Output Rate (%)"
Three grouped bars per domain: green (Base), red/coral (LoRA), blue (SafeHead)

The key visual insight: the LoRA bars (red) tower up to 40%, while the SafeHead bars (blue) are nearly invisible at 0%. This dramatic contrast IS the point of the figure.

Make it clean, professional, academic. Muted colors. Serif font. 680x320 viewBox.
```

---

## Figure 4: Accuracy Preservation Scatter

```
Create a clean SVG scatter plot for an academic paper with two side-by-side panels.

Left panel: "LLaMA-3-8B"
Right panel: "Qwen2-7B"

Each panel plots SafeHead accuracy (y-axis) vs LoRA accuracy (x-axis) for 12 held-out domains. A dashed diagonal line shows y=x (perfect accuracy preservation).

LLaMA-3 points (LoRA_acc, SafeHead_acc):
(87.2, 86.9), (86.5, 86.5), (34.1, 34.1), (39.0, 38.9),
(87.8, 87.7), (89.5, 89.1), (81.7, 81.5), (54.8, 54.6),
(61.2, 61.0), (28.1, 27.7), (90.6, 90.6), (34.4, 34.4)

Qwen2 points (LoRA_acc, SafeHead_acc):
(88.4, 88.2), (91.3, 91.3), (74.4, 74.1), (39.6, 39.5),
(90.8, 90.6), (93.2, 92.8), (84.6, 84.5), (57.2, 57.0),
(68.7, 68.4), (22.7, 22.2), (97.6, 97.6), (35.6, 35.4)

The key visual: every point clusters tightly on the diagonal — SafeHead preserves LoRA's task accuracy almost perfectly.

Professional academic style. Muted blue dots. Equal aspect ratio. 680x280 viewBox.
```

---

## Iteration tips:
- "make it more visually appealing"
- "the colors are too saturated, make them more muted"
- "add more whitespace between elements"
- "make the [X] more prominent"
- "style it like a NeurIPS paper figure"
- "the arrows are too thick"
- "can you try a different layout?"
