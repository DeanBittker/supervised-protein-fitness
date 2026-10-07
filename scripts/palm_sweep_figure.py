import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ridge beats PALM on every held-out domain, and no configuration tried changes that.
# A table of twenty-two rows does not carry it; one domain per line, ridge against the
# whole span PALM covers across configurations, shows both at once: the gap, and that
# the configurations sit on top of each other inside it.

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
figure_dir = f"{output_root}/figures"
family = os.environ.get("SPF_FAMILY", "PF00018")
paper = os.environ.get("SPF_PAPER", "lehner")

os.makedirs(figure_dir, exist_ok = True)

path = f"{results_dir}/palm_sweep_{family}_{paper}.csv"
if not os.path.exists(path):
    raise SystemExit(f"{path} not found\n  run scripts/palm_sweep_summary.py first")

frame = pd.read_csv(path)
if 'dataset' not in frame or 'ridge' not in frame:
    raise SystemExit(f"{path} needs a dataset and a ridge column, has {list(frame.columns)}")
configs = [c for c in frame.columns if c.startswith('dropout')]
# PALM's own defaults are batch 5000 with SGD, which is about six optimiser steps an
# epoch here and was much worse. Every run in this sweep uses batch 256 with Adam, so
# this series is the sweep's baseline rather than PALM as distributed, and the figure
# says so instead of letting "default" stand for both.
baseline = os.environ.get("SPF_BASELINE", "dropout 0.25, kernel 5")
if not configs:
    raise SystemExit(f"no configuration columns in {path}: {list(frame.columns)}")

frame = frame[frame['ridge'].notna()].copy()
# the gene symbol is what anyone reads; the rest of the label is the same on every row
frame['label'] = frame['dataset'].str.split('_').str[0]
frame['palm_low'] = frame[configs].min(axis = 1)
frame['palm_high'] = frame[configs].max(axis = 1)
frame = frame.sort_values('ridge').reset_index(drop = True)

slide = os.environ.get("SPF_SLIDE", "") == "1"
scale = 1.4 if slide else 1.0
blue, vermillion, green, purple = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
ink, muted = "#1a1a1a", "#6b6b6b"
colour = dict(zip(configs, [vermillion, green, purple]))
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11 * scale,
    "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": muted, "ytick.color": muted,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "#e4e4e2", "grid.linewidth": 0.8,
})

height = max(5.0, 0.32 * len(frame)) * scale
fig, ax = plt.subplots(figsize = (9 * scale, height))
ax.grid(True, axis = 'x', zorder = 0)
ax.set_axisbelow(True)
ax.axvline(0, color = muted, linewidth = 1, zorder = 1)

rows = np.arange(len(frame))
# the span PALM covers across configurations, so the reader can see it never reaches ridge
ax.hlines(rows, frame['palm_low'], frame['palm_high'], color = "#c9c9c4",
          linewidth = 5, zorder = 2)
for config in configs:
    ax.scatter(frame[config], rows, s = 34, color = colour[config], alpha = 0.9,
               edgecolor = 'none', zorder = 3,
               label = (f"PALM {'baseline, ' if config == baseline else ''}{config}"
                        f"  ({frame[config].mean():+.3f})"))
ax.scatter(frame['ridge'], rows, s = 72, color = blue, marker = 'D',
           edgecolor = 'none', zorder = 4,
           label = f"one-hot ridge  ({frame['ridge'].mean():+.3f})")

ax.set_yticks(rows)
ax.set_yticklabels(frame['label'], fontsize = 9.5 * scale)
ax.set_ylim(-0.8, len(frame) - 0.2)
ax.set_xlabel("Spearman $\\rho$ within the held-out protein")
# the empty corner is where the low-scoring domains are not, at the top left
ax.legend(loc = 'upper left', fontsize = 9.5 * scale)

# the headline has to be the count, not "every domain": PALM does win a couple
best = frame[configs].max(axis = 1)
beaten = int((frame['ridge'] > best).sum())
fig.suptitle(f"{family} {paper}: ridge beats PALM on {beaten} of {len(frame)} "
             f"held-out domains\nleave-one-cluster-out, every PALM configuration at "
             f"batch 256 with Adam",
             fontsize = 12.5 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.94])
out = f"{figure_dir}/palm_sweep_{family}_{paper}.png"
plt.savefig(out)
plt.close()

wins = int((frame[configs].max(axis = 1) > frame['ridge']).sum())
print(f"{len(frame)} domains, ridge {frame['ridge'].mean():+.3f}")
for config in configs:
    print(f"  {config:28s} {frame[config].mean():+.3f}")
print(f"PALM beats ridge on {wins} of {len(frame)} domains under its best configuration")
print(f"saved {out}")
