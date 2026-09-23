import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
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
baseline = "ESM2-150M"
challenger = "onehot"
papers = ['lehner', 'rocklin']
targets = ['raw', 'zscore']
models = ['ridge', 'rf', 'xgb']

os.makedirs(figure_dir, exist_ok = True)

slide = os.environ.get("SPF_SLIDE", "") == "1"
scale = 1.4 if slide else 1.0
blue, vermillion = "#0072B2", "#D55E00"
ink, muted = "#1a1a1a", "#6b6b6b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11 * scale,
    "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": muted, "ytick.color": muted,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "#e4e4e2", "grid.linewidth": 0.8,
})

results = pd.read_csv(f"{results_dir}/model_results_{family}.csv")

# replicates are paired: every cell saw the same split assignments, so the
# difference can be taken replicate by replicate rather than comparing two
# independent spreads
rows = []
for paper in papers:
    for target in targets:
        for model in models:
            cell = results[(results['paper'] == paper) & (results['target'] == target) &
                           (results['model'] == model)]
            a = cell[cell['encoding'] == challenger].set_index('replicate')[metric]
            b = cell[cell['encoding'] == baseline].set_index('replicate')[metric]
            shared = a.index.intersection(b.index)
            if len(shared) < 3:
                continue
            difference = (a.loc[shared] - b.loc[shared]).values
            statistic, p_value = wilcoxon(difference)
            rows.append({
                'paper': paper, 'target': target, 'model': model,
                'n_replicates': len(shared),
                f'{challenger}_median': round(float(a.loc[shared].median()), 4),
                f'{baseline}_median': round(float(b.loc[shared].median()), 4),
                'median_difference': round(float(np.median(difference)), 4),
                'wins': int((difference > 0).sum()),
                'p_value': round(float(p_value), 5),
            })

report = pd.DataFrame(rows)
print("=" * 100)
print(f"PAIRED COMPARISON — {challenger} minus {baseline}, Wilcoxon signed rank over shared replicates")
print("=" * 100)
print(report.to_string(index = False))

fig, axes = plt.subplots(1, len(targets), figsize = (12 * scale, 5 * scale), sharey = True)
rng = np.random.default_rng(67)
for ax, target in zip(axes, targets):
    ax.grid(True, axis = 'y', zorder = 0)
    ax.set_axisbelow(True)
    ax.axhline(0, color = ink, linewidth = 1.2, zorder = 2)
    positions, labels = [], []
    for index, paper in enumerate(papers):
        for offset, model in zip([-0.22, 0, 0.22], models):
            cell = results[(results['paper'] == paper) & (results['target'] == target) &
                           (results['model'] == model)]
            a = cell[cell['encoding'] == challenger].set_index('replicate')[metric]
            b = cell[cell['encoding'] == baseline].set_index('replicate')[metric]
            shared = a.index.intersection(b.index)
            if len(shared) < 3:
                continue
            difference = (a.loc[shared] - b.loc[shared]).values
            x = index + offset
            colour = blue if np.median(difference) > 0 else vermillion
            ax.scatter(np.full(len(difference), x) + rng.uniform(-0.05, 0.05, len(difference)),
                       difference, s = 34, color = colour, alpha = 0.6, edgecolor = 'none', zorder = 3)
            ax.hlines(np.median(difference), x - 0.09, x + 0.09, color = colour, linewidth = 2.6, zorder = 4)
            positions.append(x)
            labels.append(model.upper() if model != 'ridge' else 'Ridge')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize = 9.5 * scale)
    # the study name sits below the model ticks, not on top of the middle one
    for index, paper in enumerate(papers):
        ax.annotate(paper.capitalize(), xy = (index, -0.115), xycoords = ('data', 'axes fraction'),
                    ha = 'center', va = 'top', fontsize = 11.5 * scale)
    ax.set_xlim(-0.45, len(papers) - 1 + 0.45)
    ax.set_title('raw DMS score' if target == 'raw' else 'z-scored within dataset', loc = 'left')
axes[0].set_ylabel("Spearman $\\rho$ difference per replicate")
fig.suptitle(f"{family}: one-hot minus ESM2-150M, paired on the same split\n"
             f"above the line means one-hot won that replicate",
             fontsize = 12.5 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0.05, 1, 0.91])
suffix = "_per_protein" if metric == "test_spearman_per_protein" else ""
path = f"{figure_dir}/paired_test_{family}{suffix}.png"
plt.savefig(path)
plt.close()
print()
print(f"saved {path}")
report.to_csv(f"{results_dir}/paired_test_{family}{suffix}.csv", index = False)
