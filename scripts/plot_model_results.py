import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
figure_dir = f"{output_root}/figures"
family = os.environ.get("SPF_FAMILY", "PF00018")
metric = os.environ.get("SPF_METRIC", "test_spearman_per_protein")
papers = ['lehner', 'rocklin']
targets = ['raw', 'zscore']
models = ['ridge', 'rf', 'xgb']
encodings = ['onehot', 'ESM2-150M']

os.makedirs(figure_dir, exist_ok = True)

slide = os.environ.get("SPF_SLIDE", "") == "1"
scale = 1.4 if slide else 1.0
# okabe-ito pair, checked for colourblind separation against a light surface
blue, vermillion = "#0072B2", "#D55E00"
ink, muted = "#1a1a1a", "#6b6b6b"
colour = {'onehot': blue, 'ESM2-150M': vermillion}
paper_label = {'lehner': 'Lehner 2025  (human)', 'rocklin': 'Rocklin 2023  (across nature)'}
target_label = {'raw': 'raw DMS score', 'zscore': 'z-scored within dataset'}
metric_label = {'test_spearman_per_protein': "Spearman $\\rho$, averaged within protein",
                'test_spearman': "Spearman $\\rho$, pooled across proteins",
                'test_r2': "$R^2$ on held-out domains"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11 * scale,
    "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": muted, "ytick.color": muted,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "#e4e4e2", "grid.linewidth": 0.8,
})

results = pd.read_csv(f"{results_dir}/model_results_{family}.csv")
print(f"{len(results)} cells, {results['replicate'].nunique()} replicates")

rng = np.random.default_rng(67)
fig, axes = plt.subplots(len(targets), len(papers), figsize = (12 * scale, 8 * scale), sharey = True)

for row, target in enumerate(targets):
    for col, paper in enumerate(papers):
        ax = axes[row, col]
        ax.grid(True, axis = 'y', zorder = 0)
        ax.set_axisbelow(True)
        ax.axhline(0, color = muted, linewidth = 1, zorder = 1)

        for offset, encoding in zip([-0.16, 0.16], encodings):
            for position, model in enumerate(models):
                cell = results[(results['paper'] == paper) & (results['target'] == target) &
                               (results['encoding'] == encoding) & (results['model'] == model)]
                if cell.empty:
                    continue
                values = cell[metric].values
                # every replicate as a point, so the spread is the message rather
                # than a single number that depends on which domains landed in test
                ax.scatter(np.full(len(values), position + offset) + rng.uniform(-0.05, 0.05, len(values)),
                           values, s = 30, color = colour[encoding], alpha = 0.6,
                           edgecolor = 'none', zorder = 3,
                           label = encoding if (row == 0 and col == 0 and position == 0) else None)
                ax.hlines(np.median(values), position + offset - 0.12, position + offset + 0.12,
                          color = colour[encoding], linewidth = 2.5, zorder = 4)

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([m.upper() if m != 'ridge' else 'Ridge' for m in models])
        ax.set_xlim(-0.5, len(models) - 0.5)
        if row == 0:
            ax.set_title(paper_label[paper], loc = 'left', fontsize = 11.5 * scale)
        if col == 0:
            ax.set_ylabel(f"{target_label[target]}\n{metric_label[metric]}", fontsize = 10 * scale)

axes[0, 0].legend(loc = 'upper left', fontsize = 10 * scale)
headline = ("averaged within each held-out protein"
            if metric == 'test_spearman_per_protein' else "pooled across held-out proteins")
fig.suptitle(f"{family}: held-out domain performance, {headline}\n"
             f"one point per split replicate, clusters at 60% identity",
             fontsize = 13 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.93])
suffix = "_per_protein" if metric == "test_spearman_per_protein" else ""
path = f"{figure_dir}/model_results_{family}{suffix}.png"
plt.savefig(path)
plt.close()
print(f"saved {path}")

summary = (results.groupby(['paper', 'encoding', 'model', 'target'])[
               [c for c in ['test_spearman_per_protein', 'test_spearman', 'test_r2'] if c in results]]
           .agg(['median', 'std', 'count']).round(3))
summary.to_csv(f"{results_dir}/model_summary_{family}.csv")
print()
print(summary.to_string())
