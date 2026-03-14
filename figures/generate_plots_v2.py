"""Generate additional paper figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})

# ============================================================
# Figure: Safety degradation under fine-tuning (motivation)
# Shows Base -> LoRA harmful rates across domains
# ============================================================
fig, ax = plt.subplots(figsize=(8, 3.5))

domains = ['PIQA', 'OpenH.', 'HellaS.', 'Dolly', 'Alpaca', 'CSenseQA',
           'BoolQ', 'AG News', 'MMLU', 'ARC', 'Math', 'GSM8K']
base_harm = [7.1]*12  # LLaMA3 base
lora_harm = [40.0, 39.0, 35.1, 28.3, 25.7, 20.0, 18.6, 15.1, 12.4, 8.6, 7.0, 8.3]
ours_harm = [0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]

x = np.arange(len(domains))
w = 0.25

bars1 = ax.bar(x - w, base_harm, w, label='Base (aligned)', color='#4CAF50', alpha=0.8)
bars2 = ax.bar(x, lora_harm, w, label='LoRA (fine-tuned)', color='#F44336', alpha=0.8)
bars3 = ax.bar(x + w, ours_harm, w, label='+ SafeHead (ours)', color='#2196F3', alpha=0.8)

ax.set_ylabel('Harmful Output Rate (%)')
ax.set_xticks(x)
ax.set_xticklabels(domains, rotation=35, ha='right', fontsize=9)
ax.set_ylim(0, 48)
ax.legend(loc='upper right', framealpha=0.95)
ax.set_title('Safety Degradation under LoRA Fine-Tuning (LLaMA-3-8B)', fontweight='bold')

# Add annotation
ax.annotate('Fine-tuning\ndegrades safety', xy=(0, 40), xytext=(3, 44),
            fontsize=9, color='#F44336', fontweight='bold', ha='center',
            arrowprops=dict(arrowstyle='->', color='#F44336', lw=1.5))

plt.tight_layout()
plt.savefig('safety_degradation.pdf', bbox_inches='tight')
plt.savefig('safety_degradation.png', bbox_inches='tight')
print("Saved safety_degradation.pdf/png")

# ============================================================
# Figure: Combined ablation (loss + data) - single clean figure
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: Loss component ablation
ax = axes[0]
variants = ['Full', '$-\\mathcal{L}_{func}^{safe}$', '$-\\mathcal{L}_{func}$', '$-$Domain\ncalib.', '$-$Lambda\nLoRA']
tp = [83.7, 6.5, 61.4, 98.1, 57.7]
sr = [99.5, 100.0, 99.8, 89.3, 100.0]

x = np.arange(len(variants))
w = 0.35
b1 = ax.bar(x - w/2, tp, w, label='Task Pres.', color='#2196F3', edgecolor='white', linewidth=0.5)
b2 = ax.bar(x + w/2, sr, w, label='Safety Rec.', color='#F44336', edgecolor='white', linewidth=0.5)

ax.set_ylabel('Score (%)')
ax.set_xticks(x)
ax.set_xticklabels(variants, fontsize=8)
ax.set_ylim(0, 110)
ax.legend(loc='upper center', fontsize=9, framealpha=0.9)
ax.set_title('(a) Component Ablation', fontweight='bold')

for b, vals in [(b1, tp), (b2, sr)]:
    for bar, v in zip(b, vals):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., y + 1.5, f'{v:.0f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

# Right: Data composition
ax = axes[1]
configs = ['Full\n(18)', '$-$QA\n(15)', '$-$Math\n(15)', '$-$Instr.\n(12)', 'Excl. 1\n(17)', 'Excl. 2\n(16)']
tp2 = [83.7, 80.3, 83.1, 99.6, 83.4, 82.9]
sr2 = [99.5, 100.0, 99.7, 91.2, 99.4, 99.3]

x = np.arange(len(configs))
b1 = ax.bar(x - w/2, tp2, w, label='Task Pres.', color='#2196F3', edgecolor='white', linewidth=0.5)
b2 = ax.bar(x + w/2, sr2, w, label='Safety Rec.', color='#F44336', edgecolor='white', linewidth=0.5)

ax.set_ylabel('Score (%)')
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylim(75, 105)
ax.legend(loc='lower left', fontsize=9, framealpha=0.9)
ax.set_title('(b) Training Data Composition', fontweight='bold')

# Highlight the instruction removal
ax.annotate('Safety drops\nwithout instruction data', xy=(3 + w/2, sr2[3]), xytext=(2.0, 83),
            fontsize=8, color='red', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

for b, vals in [(b1, tp2), (b2, sr2)]:
    for bar, v in zip(b, vals):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., y + 0.5, f'{v:.0f}',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig('ablation_combined.pdf', bbox_inches='tight')
plt.savefig('ablation_combined.png', bbox_inches='tight')
print("Saved ablation_combined.pdf/png")

# ============================================================
# Figure: Accuracy preservation scatter (updated with green deltas)
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

# LLaMA-3 (from updated table)
l3_lora = [87.2, 86.5, 34.1, 39.0, 87.8, 89.5, 81.7, 54.8, 61.2, 28.1, 90.6, 34.4]
l3_ours = [86.9, 86.5, 34.1, 38.8, 87.9, 89.2, 81.5, 55.0, 61.0, 27.8, 90.6, 34.2]

# Qwen2 (from updated table)
q2_lora = [88.4, 91.3, 74.4, 39.6, 90.8, 93.2, 84.6, 57.2, 68.7, 22.7, 98.7, 35.6]
q2_ours = [88.3, 91.1, 74.2, 39.6, 90.5, 93.3, 84.7, 57.1, 68.5, 22.5, 98.5, 35.6]

for ax, lora, ours, title in [(ax1, l3_lora, l3_ours, 'LLaMA-3-8B'), (ax2, q2_lora, q2_ours, 'Qwen2-7B')]:
    # Color: green if ours >= lora, blue otherwise
    colors = ['#4CAF50' if o >= l else '#2196F3' for o, l in zip(ours, lora)]
    ax.scatter(lora, ours, c=colors, s=55, zorder=3, edgecolors='white', linewidth=0.5)
    lims = [min(min(lora), min(ours)) - 3, max(max(lora), max(ours)) + 3]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='$y = x$')
    ax.set_xlabel('LoRA Accuracy (%)')
    ax.set_ylabel('SafeHead Accuracy (%)')
    ax.set_title(title, fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    ax.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('accuracy_preservation.pdf', bbox_inches='tight')
plt.savefig('accuracy_preservation.png', bbox_inches='tight')
print("Saved accuracy_preservation.pdf/png")

print("\nAll v2 figures generated!")
