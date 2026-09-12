"""
make_figure.py - Generates Figure 1 for the RetrievalBench paper.

Left panel  : grouped bar chart  - Recall@1, MRR, nDCG@5 by configuration
Right panel : accuracy vs latency scatter on log-x scale

Run:
    py make_figure.py
Output:
    plots/figure1.png
"""

import os
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────

configs      = ["Dense-only", "Hybrid", "Hybrid +\nReranked"]
colors       = ["#4C72B0", "#DD8452", "#55A868"]
mean_latency = [62.7, 13.8, 701.8]

data = {
    "Recall@1": [0.8929, 0.9286, 0.9643],
    "MRR":      [0.9464, 0.9643, 0.9821],
    "nDCG@5":   [0.9605, 0.9736, 0.9868],
}
accuracy_metric = "MRR"

# ── Figure ────────────────────────────────────────────────────────────────────

fig, (ax_bar, ax_sc) = plt.subplots(
    1, 2, figsize=(13, 5),
    gridspec_kw={"width_ratios": [1.6, 1]},
)
fig.suptitle(
    "Figure 1. RetrievalBench: Retrieval Quality and Latency Tradeoff",
    fontsize=12, fontweight="bold", y=1.01,
)

# ── (a) Grouped bar chart ─────────────────────────────────────────────────────

metrics   = list(data.keys())
n_metrics = len(metrics)
bar_w     = 0.22
x         = np.arange(n_metrics)
offsets   = np.array([-1, 0, 1]) * bar_w

for i, (cfg, color, offset) in enumerate(zip(configs, colors, offsets)):
    vals = [data[m][i] for m in metrics]
    bars = ax_bar.bar(
        x + offset, vals,
        width=bar_w * 0.92, color=color, label=cfg,
        zorder=3, edgecolor="white", linewidth=0.5,
    )
    for bar, val in zip(bars, vals):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=7.5, color="#222222",
        )

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(metrics, fontsize=11)
ax_bar.set_ylabel("Score", fontsize=11)
ax_bar.set_ylim(0.84, 1.025)
ax_bar.set_title("(a) Retrieval Quality", fontsize=11, fontweight="bold")
ax_bar.legend(fontsize=9, loc="lower right", framealpha=0.9)
ax_bar.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax_bar.set_axisbelow(True)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

# ── (b) Accuracy vs latency scatter ──────────────────────────────────────────

acc_vals = data[accuracy_metric]

# Dashed connector in latency order
order = sorted(range(len(configs)), key=lambda i: mean_latency[i])
ax_sc.plot(
    [mean_latency[i] for i in order],
    [acc_vals[i]     for i in order],
    linestyle="--", color="#bbbbbb", linewidth=1.2, zorder=1,
)

# Points + labels
label_offsets = {
    "Dense-only":       (8, -16),
    "Hybrid + Reranked": (8,   6),
    "Hybrid":           (8,   6),
}
for cfg, color, lat, acc in zip(configs, colors, mean_latency, acc_vals):
    ax_sc.scatter(lat, acc, s=160, color=color, zorder=3,
                  edgecolors="white", linewidths=1.5)
    clean = cfg.replace("\n", " ")
    xoff, yoff = label_offsets.get(clean, (8, 6))
    ax_sc.annotate(
        clean, (lat, acc),
        textcoords="offset points", xytext=(xoff, yoff),
        fontsize=8.5, color=color, fontweight="bold",
    )

# Pareto annotation - text bottom-left, arrow curves up to Hybrid point
ax_sc.annotate(
    "Hybrid dominates Dense-only\n(higher accuracy, lower latency)",
    xy=(13.8, 0.9643),
    xytext=(9, 0.938),
    fontsize=7.2, color="#555555",
    arrowprops=dict(
        arrowstyle="->", color="#999999", lw=1.1,
        connectionstyle="arc3,rad=0.35",
    ),
)

ax_sc.set_xscale("log")
ax_sc.set_xlim(6, 1800)
ax_sc.set_ylim(0.933, 0.995)
ax_sc.set_xlabel("Mean Latency per Query (ms)  [log scale]", fontsize=11)
ax_sc.set_ylabel(accuracy_metric, fontsize=11)
ax_sc.set_title("(b) Accuracy vs. Latency Tradeoff", fontsize=11, fontweight="bold")
ax_sc.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda v, _: f"{int(v)}" if v >= 1 else f"{v:.1f}"
))
ax_sc.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax_sc.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
ax_sc.set_axisbelow(True)
ax_sc.spines["top"].set_visible(False)
ax_sc.spines["right"].set_visible(False)

# ── Save ──────────────────────────────────────────────────────────────────────

os.makedirs("plots", exist_ok=True)
out = os.path.join("plots", "figure1.png")
fig.tight_layout()
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved -> {out}")
