"""Ablation summary chart: delta from Full system, per metric.

Each ablation row shows two paired bars:
  left  = Delta task accuracy (negative = accuracy worsens)
  right = Delta harmful rate  (positive = safety worsens)

Both share a 0 line (= Full system performance), so a row reads directly as
"removing X costs Y accuracy points and adds Z harm points." Bigger bar =
component matters more.

Single-row layout: panels (a) and (b) sit side by side.

Outputs:
  figures/ablation_summary.pdf
  figures/ablation_summary.png
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Source rows: (name, [per-domain accuracies], harm rate -- scalar or list).
loss_rows = [
    ("Full system",                                          [86.9, 86.5, 34.1, 38.9], 0.1),
    (r"w/o $\mathcal{L}_{\mathrm{func}}$",   [68.3, 71.2, 28.6, 27.5], 0.0),
    (r"w/o $\mathcal{L}_{\mathrm{cls}}$",                  [84.2, 85.8, 33.6, 36.1], 0.6),
    (r"w/o domain calib.",                                   [84.4, 85.7, 33.5, 36.8], 0.3),
]
data_rows = [
    ("Full (18)",                  [86.9, 86.5, 34.1, 38.9], [0.1, 0.0, 0.0, 0.1]),
    (r"w/o math/code (16)",        [86.8, 86.5, 33.4, 38.6], [0.1, 0.0, 0.1, 0.1]),
    (r"w/o QA/read (15)",          [86.1, 85.7, 34.0, 38.2], [0.3, 0.1, 0.0, 0.2]),
    (r"w/o instruction (12)",      [86.8, 86.4, 33.8, 38.7], [6.3, 3.1, 1.7, 8.4]),
]


def avg(x):
    return sum(x) / len(x) if isinstance(x, list) else x


def to_deltas(rows):
    """Skip the first row (full system); return (name, dAcc, dHarm) tuples."""
    full_acc = avg(rows[0][1])
    full_harm = avg(rows[0][2])
    return [(name, avg(accs) - full_acc, avg(harms) - full_harm)
            for name, accs, harms in rows[1:]]


loss_deltas = to_deltas(loss_rows)
data_deltas = to_deltas(data_rows)

ACC_COLOR = "#4F6E94"
HARM_COLOR = "#B94A48"
GOOD_COLOR = "#4C8C6A"


def fmt_delta(v):
    if abs(v) < 0.005:
        return r"$\pm$0"
    return f"{v:+.1f}" if abs(v) >= 1 else f"{v:+.2f}"


VALUE_PAD_PT = 10  # gap (in points) between bar end and numeric label
NOTE_PAD_PT = 6    # gap (in points) between zero line and any italic caveat


def plot_metric(ax, ys, values, title, xlabel, *,
                harm=False, xlim=None, value_notes=None):
    if harm:
        colors = [HARM_COLOR if v >= 0 else GOOD_COLOR for v in values]
    else:
        colors = [ACC_COLOR] * len(values)
    ax.barh(ys, values, height=0.55, color=colors,
            edgecolor="white", linewidth=0.5, zorder=3)
    ax.axvline(0, color="#444444", linewidth=0.7,
               linestyle=(0, (4, 2.5)), zorder=2)

    if xlim is not None:
        ax.set_xlim(*xlim)

    xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
    value_notes = value_notes or {}
    for i, (y, v) in enumerate(zip(ys, values)):
        label = fmt_delta(v)
        # Numeric label: anchored at the bar end, offset in display points so
        # the visual gap is independent of the data-axis scale.
        if abs(v) < xspan * 0.0025:
            ax.annotate(label, xy=(0, y),
                        xytext=(VALUE_PAD_PT, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=7, color="#888888")
        elif v >= 0:
            ax.annotate(label, xy=(v, y),
                        xytext=(VALUE_PAD_PT, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=7, color="#222222")
        else:
            ax.annotate(label, xy=(v, y),
                        xytext=(-VALUE_PAD_PT, 0), textcoords="offset points",
                        ha="right", va="center", fontsize=7, color="#222222")
        # Optional caveat: anchor at zero, push into the empty side of the row.
        note = value_notes.get(i, "")
        if note:
            if v <= 0:
                ax.annotate(note, xy=(0, y),
                            xytext=(NOTE_PAD_PT, 0), textcoords="offset points",
                            ha="left", va="center",
                            fontsize=6.5, color="#888888", style="italic")
            else:
                ax.annotate(note, xy=(0, y),
                            xytext=(-NOTE_PAD_PT, 0), textcoords="offset points",
                            ha="right", va="center",
                            fontsize=6.5, color="#888888", style="italic")

    ax.set_yticks([])
    ax.set_ylim(-0.6, len(ys) - 0.4)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle=":", linewidth=0.4, color="#CCCCCC")
    ax.tick_params(axis="x", length=2.5, color="#999999")
    ax.set_title(title, loc="left", fontsize=8.5, pad=2, color="#444444")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.5, color="#666666", labelpad=2)


def plot_panel(ax_label, ax_acc, ax_harm, deltas, *,
               acc_xlim, harm_xlim, harm_notes=None):
    n = len(deltas)
    ys = np.arange(n)[::-1]
    names = [d[0] for d in deltas]
    d_accs = [d[1] for d in deltas]
    d_harms = [d[2] for d in deltas]

    ax_label.set_xlim(0, 1)
    ax_label.set_ylim(-0.6, n - 0.4)
    ax_label.axis("off")
    for y, name in zip(ys, names):
        ax_label.text(1.0, y, name, va="center", ha="right",
                      fontsize=8.5, color="#222222")

    plot_metric(ax_acc, ys, d_accs,
                r"$\Delta$ task accuracy (%)",
                r"$\leftarrow$ Accuracy decreases",
                harm=False, xlim=acc_xlim)
    plot_metric(ax_harm, ys, d_harms,
                r"$\Delta$ harmful rate (%)",
                r"Safety decreases $\rightarrow$",
                harm=True, xlim=harm_xlim, value_notes=harm_notes)


fig = plt.figure(figsize=(7.0, 2.5))
# Width-ratios redistributed: variant-label gutters shrunk to just-enough
# (panel A: fits "w/o Domain calib."; panel B: fits "w/o Instruction (12)"),
# the reclaimed width is given back to the four metric subplots so the value
# labels have more breathing room.
gs = fig.add_gridspec(
    1, 6,
    width_ratios=[0.80, 0.95, 0.95, 0.95, 0.95, 0.95],
    wspace=0.06,
    left=0.005, right=0.995, top=0.78, bottom=0.20,
)

# Panel (a): loss components.
ax_la = fig.add_subplot(gs[0, 0])
ax_aa = fig.add_subplot(gs[0, 1])
ax_ah = fig.add_subplot(gs[0, 2])
plot_panel(
    ax_la, ax_aa, ax_ah, loss_deltas,
    acc_xlim=(-19.0, 1.0),
    harm_xlim=(-0.50, 0.95),
)

# Panel (b): training data composition.
ax_lb = fig.add_subplot(gs[0, 3])
ax_ab = fig.add_subplot(gs[0, 4])
ax_bh = fig.add_subplot(gs[0, 5])
plot_panel(
    ax_lb, ax_ab, ax_bh, data_deltas,
    acc_xlim=(-1.05, 0.3),
    harm_xlim=(-0.8, 6.2),
)

# Panel titles, positioned by panel midpoint over (label + acc + harm).
def panel_center(left_ax, right_ax):
    return (left_ax.get_position().x0 + right_ax.get_position().x1) / 2

fig.text(panel_center(ax_la, ax_ah), 0.91, "(a) Loss components",
         ha="center", fontweight="bold", fontsize=10)
fig.text(panel_center(ax_lb, ax_bh), 0.91, "(b) Training data composition",
         ha="center", fontweight="bold", fontsize=10)

fig.savefig(ROOT / "ablation_summary.pdf", bbox_inches="tight")
fig.savefig(ROOT / "ablation_summary.png", bbox_inches="tight", dpi=300)
print("Wrote ablation_summary.pdf and ablation_summary.png")
