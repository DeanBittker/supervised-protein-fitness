import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import biotite.sequence as seq
import biotite.sequence.align as align
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
# outputs go to the repo by default; on O2 set SPF_OUTPUT to the shared project
# folder so results are computed once and pulled down rather than recomputed
output_root = os.environ.get("SPF_OUTPUT", project_dir)
data_root = os.environ.get("SPF_DATA", f"{project_dir}/data")
constructs_path = f"{output_root}/results/constructs.csv"
results_dir = f"{output_root}/results"
figure_dir = f"{output_root}/figures"
family = os.environ.get("SPF_FAMILY", "PF00018")
thresholds = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40]
scopes = ['lehner', 'rocklin', 'pooled']
ratios = {'train': 0.80, 'validate': 0.10, 'test': 0.10}
n_replicates = 20
identity_mode = "shortest"
linkage_method = "average"
rng_seed = 67

for d in [results_dir, figure_dir]:
    os.makedirs(d, exist_ok = True)


def identity_matrix(sequences):
    proteins = [seq.ProteinSequence(s) for s in sequences]
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignment = align.align_multiple(proteins, matrix, gap_penalty = (-10, -1))[0]
    identity = align.get_pairwise_sequence_identity(alignment, mode = identity_mode)
    np.fill_diagonal(identity, 1.0)
    return identity


def cluster_at(identity, threshold):
    distance = 1 - identity
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2
    tree = linkage(squareform(distance, checks = False), method = linkage_method)
    return fcluster(tree, t = 1 - threshold, criterion = 'distance')


def assign_once(sizes, rng):
    # whole clusters only. shuffle first so ties break differently per replicate,
    # then place the largest remaining cluster into whichever split is furthest
    # below its target share
    total = sum(sizes.values())
    targets = {k: v * total for k, v in ratios.items()}
    running = {k: 0 for k in ratios}
    assignment = {}
    order = list(sizes.items())
    rng.shuffle(order)
    for cluster_id, n in sorted(order, key = lambda x: -x[1]):
        pick = max(running, key = lambda k: targets[k] - running[k])
        assignment[cluster_id] = pick
        running[pick] += n
    return assignment


constructs = pd.read_csv(constructs_path)
annotated = constructs.dropna(subset = ['pfam_acc'])
domains = (annotated[annotated['pfam_acc'] == family]
           .groupby('wt_sequence')
           .agg(n_variants = ('n_variants', 'sum'),
                papers = ('paper', lambda s: '+'.join(sorted(set(s)))),
                label = ('construct', 'first'))
           .reset_index())
print(f"family {family}: {len(domains)} distinct wild-type domains")

split_rows = []
loco_rows = []
summary_rows = []

for scope in scopes:
    pool = domains if scope == 'pooled' else domains[domains['papers'].str.contains(scope)]
    pool = pool.reset_index(drop = True)
    if len(pool) < 5:
        print(f"  {scope}: only {len(pool)} domains, skipped")
        continue
    identity = identity_matrix(pool['wt_sequence'].tolist())

    for threshold in thresholds:
        labels = cluster_at(identity, threshold)
        pool = pool.assign(cluster = labels)
        sizes = pool.groupby('cluster').size().to_dict()
        variants = pool.groupby('cluster')['n_variants'].sum().to_dict()

        # leave one cluster out: every cluster takes a turn as the test set
        for cluster_id in sorted(sizes):
            for _, row in pool.iterrows():
                loco_rows.append({
                    'scope': scope, 'threshold': threshold, 'fold': int(cluster_id),
                    'label': row['label'], 'cluster': int(row['cluster']),
                    'split': 'test' if row['cluster'] == cluster_id else 'train',
                    'n_variants': int(row['n_variants']),
                })

        # a spread of valid 80/10/10 assignments, so the split itself has a variance
        rng = np.random.default_rng(rng_seed)
        achieved = []
        seen = set()
        for replicate in range(n_replicates):
            assignment = assign_once(dict(sizes), rng)
            pool_split = pool['cluster'].map(assignment)
            counts = pool_split.value_counts()
            achieved.append({k: 100 * counts.get(k, 0) / len(pool) for k in ratios})
            seen.add(tuple(pool_split))
            for (_, row), split in zip(pool.iterrows(), pool_split):
                split_rows.append({
                    'scope': scope, 'threshold': threshold, 'replicate': replicate,
                    'label': row['label'], 'cluster': int(row['cluster']),
                    'split': split, 'n_variants': int(row['n_variants']),
                    'papers': row['papers'],
                })

        achieved = pd.DataFrame(achieved)
        test_domains = [int(round(a * len(pool) / 100)) for a in achieved['test']]
        summary_rows.append({
            'scope': scope, 'threshold': threshold, 'n_domains': len(pool),
            'n_clusters': len(sizes),
            'n_singletons': sum(1 for n in sizes.values() if n == 1),
            'largest_cluster': max(sizes.values()),
            'pct_in_largest': round(100 * max(sizes.values()) / len(pool), 1),
            'train_pct_mean': round(achieved['train'].mean(), 1),
            'test_pct_mean': round(achieved['test'].mean(), 1),
            'test_pct_min': round(achieved['test'].min(), 1),
            'test_pct_max': round(achieved['test'].max(), 1),
            'test_domains_min': min(test_domains),
            'test_domains_max': max(test_domains),
            'n_distinct_assignments': len(seen),
        })

# record every cluster's size so the size distribution can be plotted per threshold
size_rows = []
for scope in scopes:
    pool = domains if scope == 'pooled' else domains[domains['papers'].str.contains(scope)]
    pool = pool.reset_index(drop = True)
    if len(pool) < 5:
        continue
    identity = identity_matrix(pool['wt_sequence'].tolist())
    for threshold in thresholds:
        for size in pd.Series(cluster_at(identity, threshold)).value_counts():
            size_rows.append({'scope': scope, 'threshold': threshold, 'size': int(size)})
sizes_frame = pd.DataFrame(size_rows)

summary = pd.DataFrame(summary_rows)
splits = pd.DataFrame(split_rows)
loco = pd.DataFrame(loco_rows)

print()
print("=" * 110)
print(f"SPLIT QUALITY ACROSS THRESHOLDS — {family}")
print("=" * 110)
print(summary.to_string(index = False))

print()
print("leave-one-cluster-out folds available:")
print(loco.groupby(['scope', 'threshold'])['fold'].nunique().to_string())

splits.to_csv(f"{results_dir}/splits_{family}.csv", index = False)
loco.to_csv(f"{results_dir}/loco_{family}.csv", index = False)
summary.to_csv(f"{results_dir}/split_summary_{family}.csv", index = False)
blue, vermillion, green = "#0072B2", "#D55E00", "#009E73"
ink, muted = "#1a1a1a", "#6b6b6b"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": muted, "ytick.color": muted,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "#e4e4e2", "grid.linewidth": 0.8,
})

labels = {'lehner': 'Lehner 2025  (human)', 'rocklin': 'Rocklin 2023  (across nature)', 'pooled': 'Both studies pooled'}
fig, axes = plt.subplots(1, 3, figsize = (14, 4.6), sharey = True)
rng = np.random.default_rng(rng_seed)
for ax, scope in zip(axes, scopes):
    ax.grid(True, axis = 'y', zorder = 0)
    ax.set_axisbelow(True)
    scoped = sizes_frame[sizes_frame['scope'] == scope]
    if scoped.empty:
        continue
    n_domains = int(summary[summary['scope'] == scope]['n_domains'].iloc[0])
    # one dot per cluster, jittered, so the whole size distribution is visible
    ax.scatter(scoped['threshold'] * 100 + rng.uniform(-1.6, 1.6, len(scoped)), scoped['size'],
               s = 26, color = blue, alpha = 0.55, zorder = 3, edgecolor = 'none')
    medians = scoped.groupby('threshold')['size'].median()
    ax.plot(medians.index * 100, medians.values, marker = 'o', markersize = 7,
            linewidth = 2, color = vermillion, zorder = 4, label = 'median cluster size')
    ideal = n_domains * ratios['test']
    ax.axhline(ideal, color = green, linestyle = '--', linewidth = 1.6, zorder = 2,
               label = '10% test share')
    ax.annotate(f'{ideal:.1f} domains', xy = (0.985, ideal), xycoords = ('axes fraction', 'data'),
                ha = 'right', va = 'bottom', fontsize = 9, color = green)
    ax.set_xlabel('sequence identity threshold (percent)')
    ax.set_title(f"{labels[scope]} — {n_domains} domains", loc = 'left', fontsize = 11)
axes[0].set_ylabel('domains in cluster')
axes[0].legend(loc = 'upper left', fontsize = 9)
fig.suptitle(f"{family}: cluster size distribution by threshold. Clusters above the dashed line cannot fit in a 10% test set",
             fontsize = 12.5, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.92])
figure_path = f"{figure_dir}/cluster_sizes_{family}.png"
plt.savefig(figure_path)
plt.close()
print()
print(f"Cluster size figure saved to {figure_path}")

sizes_frame.to_csv(f"{results_dir}/cluster_sizes_{family}.csv", index = False)
print(f"Split replicates saved to {results_dir}/splits_{family}.csv")
print(f"Leave-one-cluster-out folds saved to {results_dir}/loco_{family}.csv")
print(f"Summary saved to {results_dir}/split_summary_{family}.csv")
