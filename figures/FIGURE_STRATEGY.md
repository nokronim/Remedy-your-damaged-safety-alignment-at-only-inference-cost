# Figure Strategy — 3 Figures, No Overlap

Each figure has ONE job. No repeated elements across figures.

---

## Figure 1: Pipeline ("What do I need to do?")
**Placement:** Section 1 (Introduction) or Section 3 (Method overview)
**Size:** double-column (figure*)

**Shows:** The end-to-end system at a high level.
**Does NOT show:** Internal architecture of the safety head, routing logic, or how the side network works.

**Content:**
- Training phase: Aligned LLM → LoRA → fine-tuned models → train safety heads → (model, safety head) pairs
- Hypernetwork training: activation fingerprints → direction/magnitude decomposition → Hypernetwork G → generated weights
- Inference: new unseen model → extract activations → frozen Hypernetwork G → safety head (done)
- Ends at "Safety Head deployed" — no safe/refuse routing arrows

**Key message:** One-time training amortizes safety across all future models. Deployment takes seconds.

**Claude web prompt:**
```
Design a clean SVG figure for an academic ML paper showing the SafeHead pipeline. Professional, muted colors, serif font. 700×420 viewBox.

Two-row layout:

TOP ROW — "Training (one-time)" in a light container:
- Left: Aligned LLM → LoRA → stack of fine-tuned models → train → stack of safety heads. These are training pairs.
- Middle: Activation fingerprints split into Direction and Magnitude → feed into a prominent "Hypernetwork G" box → outputs generated weights. Show three losses (L_recon, L_cls, L_func) as a small side panel.
- Connect left to middle with dashed arrows.

BOTTOM ROW — "Inference (per model)" in a different colored container:
- Linear flow: "New Model (unseen)" → "Extract Activations" → "Hypernetwork G (frozen)" with dashed border → "Safety Head"
- End here. Do NOT add routing arrows or safe/refuse outputs.
- Small label: "Deploy in seconds — no training needed"

The hypernetwork should be the visual focal point in both rows.
```

---

## Figure 2: Architecture ("How does it work technically?")
**Placement:** Section 3.2 (Safety Head Architecture)
**Size:** double-column (figure*)

**Shows:** The deployed safety head inside the model. How it processes prompts and routes them.
**Does NOT show:** How the hypernetwork works, training phase, activation extraction, or direction-magnitude decomposition.

**Content:**
- Frozen backbone as a horizontal band of layer blocks
- Side network as a shorter band below, connected via diagonal dashed ladder arrows
- Gated fusion label on the connections
- Side network → Pool + MLP → Safety Score (focal point)
- Safety score splits into: safe → "Pass through (base model unchanged)" and harmful → "Refuse (safety head intervenes)"

**Key message:** Lightweight classifier (~3% params) that routes prompts without modifying the backbone.

**Claude web prompt:**
```
Design a clean SVG figure for an academic paper showing how a safety head works alongside a frozen LLM backbone. Professional, muted colors, serif font. 680×400 viewBox.

Horizontal layout:
- Top: frozen backbone as a wide band of small layer blocks (muted purple/lavender). Label: "Frozen backbone (L layers, not modified)". Input arrow from left.
- Diagonal dashed arrows (teal/green) going from selected backbone layers down to a shorter band of side network blocks below. Label one arrow: "Gated fusion".
- Side network band (teal/green, visually smaller than backbone). Label: "Side network — K layers, ~3% params"
- Arrow down from side network center: "Pool + MLP"
- Prominent "Safety Score" box (warm amber — visual focal point)
- Two curved paths from safety score:
  - Left (green): "Pass through — base model output unchanged"
  - Right (coral): "Refuse — safety head intervenes"

The safety score box should be the most prominent element. Everything else supports it.
```

---

## Figure 3: Results ("Does it work?")
**Placement:** Section 4.2 (Main Results)
**Size:** single-column (figure) or double-column if needed

**Shows:** The key experimental result — LoRA degrades safety, SafeHead restores it.
**Does NOT show:** Architecture, pipeline, or method details.

**Content:** Grouped bar chart.
- 12 domains on x-axis (sorted by LoRA harmful rate)
- 3 bars per domain: Base (green), LoRA (red/coral), SafeHead (blue)
- Y-axis: "Harmful Output Rate (%)"
- Visual punchline: tall red bars (up to 40%) vs invisible blue bars (0%)

**Key message:** Fine-tuning breaks safety dramatically. SafeHead fixes it completely.

**Claude web prompt:**
```
Design a clean SVG grouped bar chart for an academic paper. Professional style, serif font. 680×300 viewBox.

12 domains sorted by LoRA harmful rate (descending):
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
- Muted green = Base (aligned model)
- Coral/red = LoRA (safety degraded)
- Blue = SafeHead (safety restored)

The blue bars should be nearly invisible — that dramatic contrast is the whole point. Tall red vs flat blue.

Y-axis: "Harmful Output Rate (%)", 0 to 45.
X-axis: domain names, rotated ~35°.
Legend: top-right, horizontal.
No grid lines. White background.
```

---

## Summary

| Figure | Section | Job | Shows | Doesn't show |
|--------|---------|-----|-------|-------------|
| 1 | Intro/Method | What to do | Pipeline end-to-end | Routing, internal architecture |
| 2 | Method 3.2 | How it works | Side network + routing | Hypernetwork, training |
| 3 | Results 4.2 | Does it work | Safety degradation bars | Any method details |
