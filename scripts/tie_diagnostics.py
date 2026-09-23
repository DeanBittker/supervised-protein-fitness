import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
figure_dir = f"{output_root}/figures"
variants_path = f"{results_dir}/variants.parquet"
family = os.environ.get("SPF_FAMILY", "PF00018")
score_column = "DMS_score"
papers = ['lehner', 'rocklin']
min_variants = 10
rng_seed = 67

os.makedirs(figure_dir, exist_ok = True)
rng = np.random.default_rng(rng_seed)

# a rank correlation is only as informative as the ordering it is given. if an assay
# reports the same number for a large share of its variants, no model can recover an
# order that the measurement never resolved. this characterises that limit for each
# dataset, before any model is fitted, so the ceiling is a property of the data.


def tie_ceiling(values, draws = 5):
    # the highest spearman a tie-free predictor could reach against this target:
    # correlate the midranks spearman actually uses against a perfect ordering that
    # breaks each tied block arbitrarily. averaged over draws because the breaking
    # is arbitrary by construction
    values = np.asarray(values, dtype = float)
    n = values.size
    tied = rankdata(values)
    reached = []
    for _ in range(draws):
        order = np.lexsort((rng.random(n), values))
        broken = np.empty(n)
        broken[order] = np.arange(n)
        reached.append(spearmanr(tied, broken)[0])
    return float(np.mean(reached))


def describe(values):
    values = np.asarray(values, dtype = float)
    values = values[np.isfinite(values)]
    n = values.size
    if n < min_variants:
        return None
    counts = pd.Series(values).value_counts()
    # share of variant pairs the assay left unordered
    tied_pairs = float((counts * (counts - 1) / 2).sum())
    total_pairs = n * (n - 1) / 2
    modal_value = counts.index[0]
    low, high = values.min(), values.max()
    if modal_value == low:
        position = 'floor'
    elif modal_value == high:
        position = 'ceiling'
    else:
        position = 'interior'
    return {
        'n_variants': n,
        'n_distinct': int(counts.size),
        'frac_tied_pairs': tied_pairs / total_pairs,
        'modal_share': float(counts.iloc[0]) / n,
        'modal_position': position,
        'frac_at_extreme': float(((values == low) | (values == high)).sum()) / n,
        'rho_ceiling': tie_ceiling(values),
        # spread a spearman would show on this many variants if the model were useless,
        # which is the other half of why a per-protein number can swing
        'null_band': 1.96 / np.sqrt(n - 1),
    }


variants = pd.read_parquet(variants_path)
print(f"{len(variants)} variants, {variants['dataset'].nunique()} datasets, "
      f"{variants['pfam_acc'].nunique()} families")

rows = []
for (paper, dataset), group in variants.groupby(['paper', 'dataset']):
    summary = describe(group[score_column])
    if summary is None:
        continue
    families = group['pfam_acc'].dropna().unique()
    rows.append({'paper': paper, 'dataset': dataset,
                 'pfam_acc': families[0] if len(families) else None, **summary})

report = pd.DataFrame(rows)
report.to_csv(f"{results_dir}/tie_diagnostics.csv", index = False)

print("\n" + "=" * 95)
print("HOW MUCH ORDERING DOES EACH ASSAY ACTUALLY RESOLVE?")
print("=" * 95)
for scope, frame in [("all families", report), (family, report[report['pfam_acc'] == family])]:
    if frame.empty:
        continue
    print(f"\n{scope}  ({len(frame)} datasets)")
    print(frame.groupby('paper')[['n_variants', 'frac_tied_pairs', 'modal_share',
                                  'frac_at_extreme', 'rho_ceiling', 'null_band']]
          .median().round(3).to_string())
    print(frame.groupby(['paper', 'modal_position']).size().to_string())

focus = report[report['pfam_acc'] == family]
if focus.empty:
    print(f"\nno {family} datasets to plot")
    raise SystemExit

slide = os.environ.get("SPF_SLIDE", "") == "1"
scale = 1.4 if slide else 1.0
blue, vermillion = "#0072B2", "#D55E00"
ink, muted = "#1a1a1a", "#6b6b6b"
colour = {'lehner': blue, 'rocklin': vermillion}
paper_label = {'lehner': 'Lehner 2025  (human)', 'rocklin': 'Rocklin 2023  (across nature)'}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11 * scale,
    "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": muted, "ytick.color": muted,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "#e4e4e2", "grid.linewidth": 0.8,
})

measures = [('frac_tied_pairs', 'share of variant pairs the assay\nleaves tied', (0, 1)),
            ('rho_ceiling', 'best Spearman $\\rho$ a tie-free\nmodel could reach', (0, 1.02)),
            ('null_band', 'Spearman $\\rho$ spread expected\nfrom a useless model', None)]

fig, axes = plt.subplots(len(papers), len(measures),
                         figsize = (13 * scale, 6.4 * scale), sharex = 'col')
for row, paper in enumerate(papers):
    frame = focus[focus['paper'] == paper]
    for col, (measure, label, limits) in enumerate(measures):
        ax = axes[row, col]
        ax.grid(True, axis = 'y', zorder = 0)
        ax.set_axisbelow(True)
        if frame.empty:
            ax.set_axis_off()
            continue
        ax.hist(frame[measure], bins = 14, color = colour[paper], alpha = 0.75,
                edgecolor = 'white', linewidth = 0.8, zorder = 3)
        ax.axvline(frame[measure].median(), color = ink, linewidth = 1.6,
                   linestyle = '--', zorder = 4)
        if limits:
            ax.set_xlim(*limits)
        if row == len(papers) - 1:
            ax.set_xlabel(label, fontsize = 10 * scale)
        if col == 0:
            ax.set_ylabel(f"{paper_label[paper]}\ndatasets", fontsize = 10 * scale)

fig.suptitle(f"{family}: how much of the ranking the measurement resolves\n"
             f"one dataset per count, dashed line is the median",
             fontsize = 13 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.93])
path = f"{figure_dir}/tie_diagnostics_{family}.png"
plt.savefig(path)
plt.close()
print(f"\nsaved {path}")
