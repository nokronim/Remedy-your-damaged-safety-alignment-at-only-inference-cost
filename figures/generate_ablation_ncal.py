"""Compact one-panel plot of calibration-set-size sensitivity.

Designed to sit beside tab:ablation_ncal in a side-by-side layout (each
occupying ~half the column width). Single panel with two y-axes:

  left  = average task accuracy across BoolQ, ARC, Math, Alpaca (blue)
  right = harmful rate (red)

Both lines share the log-scale N_cal x-axis. A dashed "default" line at
N_cal=50 marks the value used elsewhere in the paper.

Outputs:
  figures/ablation_ncal.pdf
  figures/ablation_ncal.png
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ncal = [10, 25, 50, 100, 200]
accs_per_domain = [
    [82.3, 84.1, 32.8, 35.6],
    [84.7, 85.4, 33.5, 37.2],
    [86.9, 86.5, 34.1, 38.9],
    [87.1, 86.6, 34.2, 39.1],
    [87.2, 86.5, 34.1, 39.0],
]
avg_accs = [sum(a) / len(a) for a in accs_per_domain]   # 58.7, 60.2, 61.6, 61.75, 61.7
harms = [0.0, 0.0, 0.1, 0.4, 0.3]

ACC_COLOR = "#4F6E94"   # matches ablation_summary.pdf
HARM_COLOR = "#B94A48"
GUIDE_COLOR = "#666666"
DEFAULT_N = 50

fig, ax_acc = plt.subplots(figsize=(2.8, 2.2))
ax_harm = ax_acc.twinx()

# "default" reference line at N_cal=50.
ax_acc.axvline(
    DEFAULT_N, color=GUIDE_COLOR, linewidth=0.7,
    linestyle=(0, (4, 2.5)), zorder=1,
)
# A short label for the dashed line, placed just above the plot area.
ax_acc.text(
    DEFAULT_N, 1.02, "default", transform=ax_acc.get_xaxis_transform(),
    ha="center", va="bottom", fontsize=7, color=GUIDE_COLOR, style="italic",
)

# Accuracy curve (left y-axis).
ax_acc.plot(
    ncal, avg_accs, marker="o", markersize=4.5,
    color=ACC_COLOR, linewidth=1.5, zorder=3,
)
ax_acc.set_xlabel(r"$N_{\mathrm{cal}}$ (number of calibration prompts)")
ax_acc.set_ylabel("Avg. task accuracy (%)", color=ACC_COLOR)
ax_acc.tick_params(axis="y", colors=ACC_COLOR, length=2.5)
ax_acc.spines["left"].set_color(ACC_COLOR)
ax_acc.spines["right"].set_visible(False)

# Harmful rate curve (right y-axis).
ax_harm.plot(
    ncal, harms, marker="s", markersize=4,
    color=HARM_COLOR, linewidth=1.5, zorder=3,
)
ax_harm.set_ylabel("Harmful rate (%)", color=HARM_COLOR)
ax_harm.tick_params(axis="y", colors=HARM_COLOR, length=2.5)
ax_harm.spines["right"].set_color(HARM_COLOR)
ax_harm.spines["top"].set_visible(False)
ax_harm.spines["left"].set_visible(False)

# Log x-axis with explicit ticks at the swept values.
ax_acc.set_xscale("log")
ax_acc.set_xticks(ncal)
ax_acc.set_xticklabels(ncal)
ax_acc.minorticks_off()
ax_acc.tick_params(axis="x", length=2.5, color="#999999")

# Tight, data-aware y-limits.
ax_acc.set_ylim(57.5, 63.0)
ax_harm.set_ylim(-0.05, 0.65)

ax_acc.set_axisbelow(True)
ax_acc.grid(True, which="major", linestyle=":", linewidth=0.4, color="#CCCCCC")

fig.tight_layout()
fig.savefig(ROOT / "ablation_ncal.pdf", bbox_inches="tight")
fig.savefig(ROOT / "ablation_ncal.png", bbox_inches="tight", dpi=300)
print("Wrote ablation_ncal.pdf and ablation_ncal.png")
