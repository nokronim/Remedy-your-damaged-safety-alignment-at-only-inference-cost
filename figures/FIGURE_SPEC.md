# Figure Specifications for SafeHead Paper

## Design Language (apply to ALL figures)

**Color Palette:**
- Primary blue: `#4A90D9` (model components)
- Primary red/coral: `#E8726A` (harmful / safety head)
- Primary green: `#5CB85C` (safe / preserved)
- Accent purple: `#8E7CC3` (hypernetwork)
- Accent orange: `#F5A623` (activations)
- Background fills: use 10% opacity versions of the above
- Text: `#333333` (dark gray, never pure black)
- Subtle borders: `#CCCCCC`

**Typography:**
- Headings in boxes: 11pt, semibold, dark gray
- Labels on arrows: 9pt, regular, 60% gray
- Phase titles: 14pt, bold, colored to match phase
- Math: use italic where needed but keep it minimal

**Visual Style:**
- Rounded rectangles with 8px corner radius
- Soft drop shadows (2px offset, 10% opacity, 4px blur)
- Arrows: 2px stroke, rounded caps, slight curve (not straight lines)
- Use icons where possible (model = stacked layers icon, data = document icon)
- White space between elements — don't crowd
- Subtle gradient fills on key boxes (top-light to bottom-darker, same hue)

---

## Figure 1: Pipeline Overview (double-column width, ~7in × 2.5in)

**Layout:** Three panels left-to-right, connected by thick dashed arrows. Each panel has a colored background rectangle (very light fill, rounded corners 12px).

### Panel 1 — "Collect Training Pairs" (blue theme)
**Background:** `#4A90D9` at 5% opacity

**Elements top to bottom:**
1. **Title:** "Phase 1: Collect Training Pairs" in blue, 14pt bold
2. **Base model box:** Rounded rect, `#E8E8E8` fill, white border
   - Text: "Aligned LLM" with a small shield icon ️🛡
   - Size: 140×40px
3. **Downward arrow** labeled "LoRA fine-tuning" (gray, 9pt)
4. **Stack of 3 overlapping boxes** (offset by 4px each), blue gradient fill
   - Front box text: "Fine-tuned Models"
   - Show 3 stacked to imply many
   - Blue fill from `#D6E4F0` to `#B8D0E8`
5. **Downward arrow** labeled "train side network per domain"
6. **Stack of 3 overlapping boxes**, coral/red gradient fill
   - Front box text: "Safety Heads (ground truth)"
   - Red fill from `#F5D5D0` to `#E8B8B0`
7. **Small label below:** "K domain pairs" in 8pt gray italic

### Panel 2 — "Train Hypernetwork" (purple theme)
**Background:** `#8E7CC3` at 5% opacity

**Elements top to bottom:**
1. **Title:** "Phase 2: Train Hypernetwork" in purple, 14pt bold
2. **Activation box:** Rounded rect, orange fill `#FEF0D5`
   - Text: "Activation Fingerprint"
   - Small bar-chart icon to the left of text (representing layer activations)
3. **Split into two arrows** going down-left and down-right:
   - Left box: "Direction" (orange fill, italic "what changed")
   - Right box: "Magnitude" (orange fill, italic "how much")
4. **Both arrows merge into:**
5. **Large central box** — the hypernetwork, purple gradient fill
   - Text: "Hypernetwork G" in white, bold
   - Size: 180×60px, prominent
   - Subtle purple gradient from `#A594D6` to `#7B68AE`
6. **Downward arrow**
7. **Output box:** "Generated Safety Head" in coral fill
8. **Side annotation** (small, to the right): three loss labels stacked:
   - "L_recon" "L_cls" "L_func"
   - Connected to output box with a thin dashed line

### Panel 3 — "Deploy" (green theme)
**Background:** `#5CB85C` at 5% opacity

**Elements top to bottom:**
1. **Title:** "Phase 3: Deploy" in green, 14pt bold
2. **New model box:** Green border (2px), white fill, green glow/shadow
   - Text: "New Fine-tuned Model"
   - Small badge above: "unseen" in green italic 9pt
3. **Downward arrow** labeled "calibration prompts"
4. **Hypernetwork box** (same purple as Panel 2 but lighter, implying "frozen")
   - Text: "Hypernetwork G" with a ❄ snowflake or lock icon
   - Dashed border to show it's frozen
5. **Downward arrow** labeled "single forward pass" with a ⚡ bolt icon
6. **Safety head box:** Coral fill, text "Safety Head"
7. **Fork into two paths:**
   - **Left path (green arrow, curved):** → green box "Safe output ✓"
     - Label on arrow: "safe" in green
   - **Right path (red arrow, curved):** → red box "Refused ✗"
     - Label on arrow: "harmful" in red
8. **Small label centered below:** "< 1 minute · no training · no modification"

### Panel connectors:
- Between Panel 1→2: thick dashed arrow (4px, gray), labeled "training data"
- Between Panel 2→3: thick dashed arrow, labeled "trained model"
- Arrows should be slightly curved, not straight

---

## Figure 2: Safety Head Architecture (single-column, ~3.3in × 4in)

**Layout:** Vertical, two columns side by side.

### Left column — Backbone
- **Title above:** "Frozen Backbone" in gray
- **Tall rounded rectangle** divided into 8 horizontal bands (layers)
  - Fill: light gray `#F0F0F0`
  - Each band has very subtle horizontal line separators
  - Label bottom band: "Layer 0" (9pt gray)
  - Label top band: "Layer L"
  - Ice crystal / lock icon in top-right corner (frozen indicator)
- **Input arrow** from below: "Input prompt x"

### Right column — Safety Head
- **Title above:** "Safety Head" in blue
- **Shorter rounded rectangle** (visually ~60% height of backbone), 8 bands
  - Fill: light blue gradient
  - Visually narrower than backbone (showing smaller hidden dim)
- **Brace on right side** with label "~3% params"

### Connections (ladder)
- **Diagonal dashed arrows** from each backbone layer to corresponding side layer
  - Orange colored, 1.5px, dashed
  - Small diamond/dot at connection point
  - One arrow labeled "gate μ_k" (just one, to avoid clutter)

### Top section — Classifier + Routing
- Above the side network: **"Safety Score λ" box** (coral fill, rounded)
  - Connected from side network top via arrow labeled "pool → MLP"
- Above λ box: **fork into two curved paths**
  - Left curve → green rounded box with checkmark: "Pass through"
  - Right curve → red rounded box with X: "Refuse"
  - Labels on curves: "safe" (green) and "harmful" (red)
  - No numbers, no thresholds

---

## Figure 3: Safety Degradation Bar Chart (single-column, ~3.3in × 2.5in)

**Style:** Clean grouped bars, no grid, white background.

**Data (12 domains, sorted by LoRA harmful rate):**
```
PIQA       | Base 7.1 | LoRA 40.0 | Ours 0.0
OpenHermes | Base 7.1 | LoRA 39.0 | Ours 0.0
HellaSwag  | Base 7.1 | LoRA 35.1 | Ours 0.1
Dolly      | Base 7.1 | LoRA 28.3 | Ours 0.1
Alpaca     | Base 7.1 | LoRA 25.7 | Ours 0.1
CSenseQA   | Base 7.1 | LoRA 20.0 | Ours 0.0
NQ-Open    | Base 7.1 | LoRA 20.0 | Ours 0.4
BoolQ      | Base 7.1 | LoRA 18.6 | Ours 0.1
AG News    | Base 7.1 | LoRA 15.1 | Ours 0.0
MMLU       | Base 7.1 | LoRA 12.4 | Ours 0.0
ARC        | Base 7.1 | LoRA  8.6 | Ours 0.0
Math       | Base 7.1 | LoRA  7.0 | Ours 0.0
```

**Design:**
- 3 bars per domain: green (Base), red (LoRA), blue (Ours)
- Bar width: ~8px with 2px gap between groups
- Y-axis: "Harmful Output Rate (%)", range 0–45, ticks at 0,10,20,30,40
- X-axis: domain names rotated 40°
- Legend: top-right, horizontal, inside plot
- Blue "Ours" bars are nearly invisible — that IS the visual point
- Optional: thin horizontal dashed line at y=7.1 (base rate) for reference

---

## Figure 4: Accuracy Preservation Scatter (single-column, ~3.3in × 1.8in)

**Layout:** Two panels side by side (LLaMA-3, Qwen2)

**Data points (LoRA_acc, Ours_acc):**

LLaMA-3: (87.2,86.9) (86.5,86.5) (34.1,34.1) (39.0,38.9) (87.8,87.7) (89.5,89.1) (81.7,81.5) (54.8,54.6) (61.2,61.0) (28.1,27.7) (90.6,90.6) (34.4,34.4)

Qwen2: (88.4,88.2) (91.3,91.3) (74.4,74.1) (39.6,39.5) (90.8,90.6) (93.2,92.8) (84.6,84.5) (57.2,57.0) (68.7,68.4) (22.7,22.2) (97.6,97.6) (35.6,35.4)

**Design:**
- Dashed gray diagonal line (y=x, "perfect preservation")
- Blue filled circles (6px radius) with thin white border
- All points cluster on/near the diagonal
- Axes: "LoRA Accuracy (%)" vs "SafeHead Accuracy (%)"
- Equal aspect ratio
- Panel titles: model names in bold
- No grid, just axis lines

---

## Production Workflow

1. **Option A (recommended): Canva**
   - Go to canva.com → Custom size → set to 7×2.5in (Fig 1) or 3.3×4in (Fig 2)
   - Use "Elements" search for icons (shield, lightning bolt, lock, checkmark)
   - Copy the color codes from above
   - Export as PDF

2. **Option B: Google Slides**
   - One slide per figure, set custom slide size
   - Use shapes, connectors, text boxes
   - Download as PDF

3. **Option C: Figma** (best quality but steeper learning curve)
   - Use auto-layout for alignment
   - Export as PDF/SVG

4. **Option D: TikZ** (already in repo as fig_pipeline.tex, fig_architecture.tex)
   - Compile with paper to see result
   - Iterate by editing coordinates
   - Text is native LaTeX (best for math symbols)

For Figures 3-4 (data plots), use matplotlib with the exact data above, or plot in Google Sheets and style manually.
