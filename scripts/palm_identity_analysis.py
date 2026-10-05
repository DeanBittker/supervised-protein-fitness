import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Transfer to a held-out domain is not all or nothing: inside PF00018 one domain scored
# 0.53 and another 0.21, and PF00046 as a family did better than PF00018 did. The
# obvious candidate is how close the held-out domain sits to the nearest domain the
# model trained on. This puts each domain's score against that distance.

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
figure_dir = f"{output_root}/figures"
family = os.environ.get("SPF_FAMILY", "PF00018")
paper = os.environ.get("SPF_PAPER", "lehner")
threshold = float(os.environ.get("SPF_THRESHOLD", "0.6"))
encoding = os.environ.get("SPF_ENCODING", "onehot")
model = os.environ.get("SPF_MODEL", "ridge")
target = os.environ.get("SPF_TARGET", "zscore")

os.makedirs(figure_dir, exist_ok = True)

identity_path = f"{results_dir}/identity/{paper}_{family}.csv"
folds_path = f"{results_dir}/loco_{family}.csv"
for path in (identity_path, folds_path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found\n"
                         f"  the identity matrices are written on the machine that has "
                         f"biotite, so rsync results/identity across if this is O2")

identity = pd.read_csv(identity_path, index_col = 0)
folds = pd.read_csv(folds_path)
folds = folds[(folds['scope'] == paper) & np.isclose(folds['threshold'], threshold)]
if folds.empty:
    raise SystemExit(f"no folds for {paper} at {threshold} in {folds_path}")

paths = sorted(glob.glob(f"{results_dir}/palm_test_spearman_loco*.csv"))
if not paths:
    raise SystemExit(f"no palm_test_spearman_loco*.csv under {results_dir}")
palm = pd.concat([pd.read_csv(path) for path in paths], ignore_index = True)
palm = palm[['dataset', 'n', 'spearman']].drop_duplicates(subset = ['dataset'])

# the nearest training domain is what a model could have learned the held-out one from.
# The mean is reported too, since a family that is uniformly similar is not the same as
# one with a single close relative
rows = []
for fold, group in folds.groupby('fold'):
    test = group.loc[group['split'] == 'test', 'label'].tolist()
    train = group.loc[group['split'] == 'train', 'label'].tolist()
    known = [name for name in train if name in identity.columns]
    for domain in test:
        if domain not in identity.index or not known:
            continue
        distances = identity.loc[domain, known].astype(float)
        rows.append({'fold': fold, 'dataset': domain,
                     'max_identity': float(distances.max()),
                     'mean_identity': float(distances.mean()),
                     'n_train_domains': len(known)})
closeness = pd.DataFrame(rows)
if closeness.empty:
    raise SystemExit("no held-out domain matched the identity matrix's labels")

frame = closeness.merge(palm, on = 'dataset', how = 'inner')
frame = frame.rename(columns = {'spearman': 'palm'})

ridge_path = f"{results_dir}/loco_results_{family}.csv"
if os.path.exists(ridge_path):
    ridge = pd.read_csv(ridge_path)
    ridge = ridge[(ridge['paper'] == paper) & (ridge['encoding'] == encoding) &
                  (ridge['model'] == model) & (ridge['target'] == target)]
    frame = frame.merge(ridge[['fold', 'test_spearman_per_protein']].rename(
        columns = {'test_spearman_per_protein': 'ridge'}), on = 'fold', how = 'left')

if frame.empty:
    raise SystemExit("no domain had both a PALM score and an identity")

print(f"{len(frame)} held-out domains with a score and a nearest training neighbour")
for column in ('palm', 'ridge'):
    if column not in frame or frame[column].isna().all():
        continue
    usable = frame[frame[column].notna()]
    if len(usable) < 3:
        print(f"  {column}: {len(usable)} domains, too few to correlate")
        continue
    against_max = spearmanr(usable['max_identity'], usable[column])
    against_mean = spearmanr(usable['mean_identity'], usable[column])
    print(f"  {column}: mean {usable[column].mean():+.3f}, "
          f"rho against nearest neighbour {against_max.statistic:+.3f} "
          f"(p {against_max.pvalue:.3f}), against family mean "
          f"{against_mean.statistic:+.3f} (p {against_mean.pvalue:.3f})")

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

fig, ax = plt.subplots(figsize = (8 * scale, 5.2 * scale))
ax.grid(True, zorder = 0)
ax.set_axisbelow(True)
ax.axhline(0, color = muted, linewidth = 1, zorder = 1)
for column, colour, label in (('ridge', blue, 'one-hot ridge'), ('palm', vermillion, 'PALM')):
    if column not in frame or frame[column].isna().all():
        continue
    usable = frame[frame[column].notna()]
    ax.scatter(usable['max_identity'] * 100, usable[column], s = 55, color = colour,
               alpha = 0.75, edgecolor = 'none', zorder = 3, label = label)
    if len(usable) >= 3:
        fit = np.polyfit(usable['max_identity'] * 100, usable[column], 1)
        span = np.linspace(usable['max_identity'].min() * 100,
                           usable['max_identity'].max() * 100, 20)
        ax.plot(span, np.polyval(fit, span), color = colour, linewidth = 1.6,
                linestyle = '--', zorder = 2)
ax.set_xlabel("identity to the nearest training domain (percent)")
ax.set_ylabel("Spearman $\\rho$ within the held-out protein")
ax.legend(loc = 'best', fontsize = 10 * scale)
fig.suptitle(f"{family} {paper}: does a closer relative in training mean better transfer?\n"
             f"one point per held-out domain, leave-one-cluster-out at "
             f"{threshold:.0%} identity",
             fontsize = 12.5 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.88])
path = f"{figure_dir}/palm_identity_{family}_{paper}.png"
plt.savefig(path)
plt.close()

table = f"{results_dir}/palm_identity_{family}_{paper}.csv"
frame.to_csv(table, index = False)
print()
print(frame.sort_values('max_identity', ascending = False).to_string(index = False))
print(f"\nsaved {path}")
print(f"saved {table}")
