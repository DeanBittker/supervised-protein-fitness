import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal

# Across 22 Lehner families the zinc fingers came out on top and the small helical
# interaction modules at the bottom. Before that is a statement about biology it has to
# survive three duller explanations: a family scored on five folds has a noisy mean, a
# family whose domains resemble each other is easier to transfer between, and a family
# with more variants per domain has more to learn from. This tests the trend against all
# three at once, and reports the counterexamples rather than averaging over them: LIM
# binds two zincs and scores low, the EF-hand pair binds calcium and scores lowest.

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
figure_dir = f"{output_root}/figures"
paper = os.environ.get("SPF_PAPER", "lehner")
encoding = os.environ.get("SPF_ENCODING", "onehot")
model = os.environ.get("SPF_MODEL", "ridge")
tag = os.environ.get("SPF_TAG", "").strip()
suffix = f"_{tag}" if tag else ""
annotation_path = os.environ.get("SPF_ANNOTATION", f"{project_dir}/reference/pfam_annotation.csv")

os.makedirs(figure_dir, exist_ok = True)

variance_path = f"{results_dir}/family_variance_{paper}_{encoding}{suffix}.csv"
if not os.path.exists(variance_path):
    raise SystemExit(f"{variance_path} not found\n  run scripts/family_variance.py first")
if not os.path.exists(annotation_path):
    raise SystemExit(f"{annotation_path} not found")

scores = pd.read_csv(variance_path)
scores = scores[scores['model'] == model][['family', 'folds', 'mean', 'sd']]
scores = scores.rename(columns = {'mean': 'score'})
annotation = pd.read_csv(annotation_path)
if 'source' in annotation and (annotation['source'] == 'unverified').any():
    print("WARNING: some annotations are marked unverified. They were written from "
          "memory and need checking against Pfam or InterPro before publication.\n")

frame = scores.merge(annotation, left_on = 'family', right_on = 'accession', how = 'left')
missing = sorted(frame.loc[frame['functional_class'].isna(), 'family'])
if missing:
    print(f"no annotation for {', '.join(missing)}, left out of the class comparison")

# how alike a family's own domains are. A family of near-duplicates is easier to
# transfer between for reasons that have nothing to do with what the domain does
identities = {}
for family in scores['family']:
    path = f"{results_dir}/identity/{paper}_{family}.csv"
    if not os.path.exists(path):
        continue
    matrix = pd.read_csv(path, index_col = 0).astype(float).values
    off_diagonal = matrix[~np.eye(len(matrix), dtype = bool)]
    if off_diagonal.size:
        identities[family] = float(off_diagonal.mean())
frame['within_family_identity'] = frame['family'].map(identities)

variants_path = f"{results_dir}/variants.parquet"
if os.path.exists(variants_path):
    variants = pd.read_parquet(variants_path)
    variants = variants[variants['paper'] == paper]
    counts = (variants.groupby('pfam_acc')
              .agg(domains = ('dataset', 'nunique'), variants = ('dataset', 'size')))
    counts['variants_per_domain'] = counts['variants'] / counts['domains']
    frame = frame.merge(counts[['domains', 'variants_per_domain']],
                        left_on = 'family', right_index = True, how = 'left')
else:
    print(f"{variants_path} not found, skipping the variants-per-domain confound")

print(f"{len(frame)} families, {paper}, {encoding}, {model}\n")

# --- the confounds, each against the score on its own -------------------------------
print("does anything duller explain the ranking?")
for column, label in (('folds', 'folds scored'),
                      ('within_family_identity', 'within-family identity'),
                      ('domains', 'domains in family'),
                      ('variants_per_domain', 'variants per domain')):
    if column not in frame or frame[column].isna().all():
        continue
    usable = frame[frame[column].notna()]
    if len(usable) < 5:
        continue
    result = spearmanr(usable[column], usable['score'])
    flag = "  <-- confounded" if result.pvalue < 0.05 else ""
    print(f"  {label:26s} rho {result.statistic:+.3f}  p {result.pvalue:.3f}  "
          f"n {len(usable)}{flag}")

# --- the functional grouping --------------------------------------------------------
grouped = frame[frame['functional_class'].notna()]
print(f"\nby functional class")
summary = (grouped.groupby('functional_class')['score']
           .agg(['count', 'mean', 'median', 'min', 'max'])
           .sort_values('mean', ascending = False))
print(summary.to_string(float_format = lambda v: f"{v:+.3f}"))

groups = [g['score'].values for _, g in grouped.groupby('functional_class') if len(g) >= 3]
if len(groups) >= 2:
    test = kruskal(*groups)
    print(f"\n  Kruskal-Wallis across {len(groups)} classes of 3 or more: "
          f"H {test.statistic:.2f}, p {test.pvalue:.3f}")

# A class difference that disappears once a confound is removed was that confound. The
# residual is taken on ranks, because with twenty-two families a rank fit is all the
# data supports and the scores are read as a rank correlation anyway.
for column, label in (('folds', 'fold count'),
                      ('within_family_identity', 'within-family identity')):
    if column not in grouped or grouped[column].isna().any():
        continue
    usable = grouped[grouped[column].notna()].copy()
    if len(usable) < 8:
        continue
    x = usable[column].rank().values
    y = usable['score'].rank().values
    slope, intercept = np.polyfit(x, y, 1)
    usable['residual'] = y - (slope * x + intercept)
    parts = [g['residual'].values for _, g in usable.groupby('functional_class')
             if len(g) >= 3]
    if len(parts) < 2:
        continue
    after = kruskal(*parts)
    verdict = ("class still separates them" if after.pvalue < 0.05
               else f"nothing left once {label} is removed")
    print(f"  after removing {label:24s} H {after.statistic:.2f}, "
          f"p {after.pvalue:.3f}   {verdict}")
    print("    " + "  ".join(
        f"{name[:12]} {part.mean():+.1f}"
        for name, part in usable.groupby('functional_class')['residual']
        if len(part) >= 3))

# --- metal binding, where the counterexamples live ----------------------------------
if 'binds_metal' in grouped:
    print("\nby metal bound")
    print(grouped.groupby('binds_metal')['score']
          .agg(['count', 'mean', 'min', 'max'])
          .to_string(float_format = lambda v: f"{v:+.3f}"))
    odd = grouped[(grouped['binds_metal'] != 'none') &
                  (grouped['functional_class'] != 'zinc_finger')]
    if not odd.empty:
        print("\n  metal-binding families that are not zinc fingers, which is where a "
              "\"metal coordination is easier\" story breaks:")
        for _, row in odd.sort_values('score').iterrows():
            print(f"    {row['family']} {row['name']:24s} {row['binds_metal']:8s} "
                  f"{row['score']:+.3f}")

# --- figure -------------------------------------------------------------------------
slide = os.environ.get("SPF_SLIDE", "") == "1"
scale = 1.4 if slide else 1.0
palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6b6b6b"]
ink, muted = "#1a1a1a", "#6b6b6b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11 * scale,
    "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": muted, "ytick.color": muted,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "#e4e4e2", "grid.linewidth": 0.8,
})

classes = summary.index.tolist()
colour = dict(zip(classes, palette))
fig, axes = plt.subplots(1, 2, figsize = (13 * scale, 5.4 * scale))

ax = axes[0]
ax.grid(True, axis = 'y', zorder = 0)
ax.set_axisbelow(True)
rng = np.random.default_rng(67)
for position, name in enumerate(classes):
    values = grouped.loc[grouped['functional_class'] == name, 'score'].values
    ax.scatter(np.full(len(values), position) + rng.uniform(-0.12, 0.12, len(values)),
               values, s = 60, color = colour[name], alpha = 0.85, edgecolor = 'none',
               zorder = 3)
    ax.hlines(np.mean(values), position - 0.25, position + 0.25, color = colour[name],
              linewidth = 2.5, zorder = 4)
ax.set_xticks(range(len(classes)))
ax.set_xticklabels([name.replace('_', '\n') for name in classes], fontsize = 10 * scale)
ax.set_ylabel(f"Spearman $\\rho$, {model}, averaged over folds")
ax.set_title("by what the domain does", loc = 'left', fontsize = 11.5 * scale)

ax = axes[1]
ax.grid(True, zorder = 0)
ax.set_axisbelow(True)
usable = grouped[grouped['within_family_identity'].notna()]
if not usable.empty:
    for name in classes:
        part = usable[usable['functional_class'] == name]
        ax.scatter(part['within_family_identity'] * 100, part['score'], s = 60,
                   color = colour[name], alpha = 0.85, edgecolor = 'none', zorder = 3,
                   label = name.replace('_', ' '))
    if len(usable) >= 3:
        fit = np.polyfit(usable['within_family_identity'] * 100, usable['score'], 1)
        span = np.linspace(usable['within_family_identity'].min() * 100,
                           usable['within_family_identity'].max() * 100, 20)
        ax.plot(span, np.polyval(fit, span), color = muted, linewidth = 1.4,
                linestyle = '--', zorder = 2)
    ax.legend(loc = 'lower right', fontsize = 9 * scale)
ax.set_xlabel("mean identity between the family's own domains (percent)")
ax.set_title("against the duller explanation", loc = 'left', fontsize = 11.5 * scale)

fig.suptitle(f"{paper}: is the family ranking about what the domain does?\n"
             f"{encoding}, {model}, leave-one-cluster-out, one point per family",
             fontsize = 12.5 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.9])
out = f"{figure_dir}/family_function_{paper}_{model}{suffix}.png"
plt.savefig(out)
plt.close()

table = f"{results_dir}/family_function_{paper}_{model}{suffix}.csv"
frame.sort_values('score', ascending = False).to_csv(table, index = False)
print(f"\nsaved {out}")
print(f"saved {table}")
