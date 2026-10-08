import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PF00018 gives one-hot ridge 0.595 on held-out domains. On its own that number could be
# the family rather than the method, so this reads every family that was deep enough to
# split and shows the spread. The question is how much of the result travels.

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
figure_dir = f"{output_root}/figures"
paper = os.environ.get("SPF_PAPER", "lehner")
target = os.environ.get("SPF_TARGET", "zscore")
encoding = os.environ.get("SPF_ENCODING", "onehot")
metric = os.environ.get("SPF_METRIC", "test_spearman_per_protein")
# a family scored on one or two held-out domains is a number, not an average
min_folds = int(os.environ.get("SPF_MIN_FOLDS", "3"))

os.makedirs(figure_dir, exist_ok = True)

paths = sorted(glob.glob(f"{results_dir}/loco_results_PF*.csv"))
if not paths:
    raise SystemExit(f"no loco_results_PF*.csv under {results_dir}")

rows = []
for path in paths:
    match = re.search(r"loco_results_(PF\d+)\.csv$", os.path.basename(path))
    if not match:
        continue
    frame = pd.read_csv(path)
    frame = frame[(frame['paper'] == paper) & (frame['encoding'] == encoding) &
                  (frame['target'] == target)]
    if frame.empty:
        continue
    frame = frame.assign(family = match.group(1))
    rows.append(frame[['family', 'model', 'fold', metric, 'n_test_domains']])
if not rows:
    raise SystemExit(f"no rows for paper={paper} encoding={encoding} target={target}")
cells = pd.concat(rows, ignore_index = True)
cells = cells[cells[metric].notna()]

per_family = (cells.groupby(['family', 'model'])[metric]
              .agg(folds = 'count', mean = 'mean', median = 'median', sd = 'std')
              .reset_index())
thin = sorted(per_family.loc[per_family['folds'] < min_folds, 'family'].unique())
if thin:
    print(f"families with fewer than {min_folds} folds, left out: {', '.join(thin)}")
per_family = per_family[per_family['folds'] >= min_folds]
if per_family.empty:
    raise SystemExit(f"no family has {min_folds} folds or more")

models = sorted(per_family['model'].unique())
print(f"\n{per_family['family'].nunique()} families, {paper}, {encoding}, {target}")
across = (per_family.groupby('model')['mean']
          .agg(families = 'count', mean = 'mean', sd = 'std', min = 'min', max = 'max'))
print(across.to_string(float_format = lambda v: f"{v:+.3f}"))

wide = per_family.pivot_table(index = 'family', columns = 'model', values = 'mean')
wide['folds'] = per_family.groupby('family')['folds'].max()
print()
print(wide.sort_values(models[0], ascending = False)
      .to_string(float_format = lambda v: f"{v:+.3f}"))

slide = os.environ.get("SPF_SLIDE", "") == "1"
scale = 1.4 if slide else 1.0
palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
colour = dict(zip(models, palette))
ink, muted = "#1a1a1a", "#6b6b6b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11 * scale,
    "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": muted, "ytick.color": muted,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "#e4e4e2", "grid.linewidth": 0.8,
})

order = wide.sort_values(models[0]).index.tolist()
height = max(4.5, 0.34 * len(order)) * scale
fig, ax = plt.subplots(figsize = (9 * scale, height))
ax.grid(True, axis = 'x', zorder = 0)
ax.set_axisbelow(True)
ax.axvline(0, color = muted, linewidth = 1, zorder = 1)

positions = np.arange(len(order))
for model in models:
    values = [wide.loc[family, model] for family in order]
    ax.scatter(values, positions, s = 48, color = colour[model], alpha = 0.85,
               edgecolor = 'none', zorder = 3, label = model.upper() if model != 'ridge'
               else 'Ridge')
pilot = 'PF00018'
if pilot in order:
    ax.axhline(order.index(pilot), color = "#e4e4e2", linewidth = 14, zorder = 0)

ax.set_yticks(positions)
ax.set_yticklabels([f"{family}  ({int(wide.loc[family, 'folds'])})" for family in order],
                   fontsize = 9.5 * scale)
ax.set_ylim(-0.8, len(order) - 0.2)
ax.set_xlabel("Spearman $\\rho$ within the held-out protein, averaged over folds")
ax.legend(loc = 'lower right', fontsize = 9.5 * scale)

headline = across.loc[models[0]] if models[0] in across.index else None
subtitle = (f"{encoding}, leave-one-cluster-out at 60% identity, "
            f"held-out domains per family in brackets")
fig.suptitle(f"{paper}: how much of the PF00018 result is the method and how much is "
             f"the family\n{subtitle}", fontsize = 12.5 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.93])
out = f"{figure_dir}/family_variance_{paper}_{encoding}.png"
plt.savefig(out)
plt.close()

table = f"{results_dir}/family_variance_{paper}_{encoding}.csv"
per_family.to_csv(table, index = False)
print(f"\nsaved {out}")
print(f"saved {table}")
