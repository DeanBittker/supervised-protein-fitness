import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance, rankdata
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
# outputs go to the repo by default; on O2 set SPF_OUTPUT to the shared project
# folder so results are computed once and pulled down rather than recomputed
output_root = os.environ.get("SPF_OUTPUT", project_dir)
data_root = os.environ.get("SPF_DATA", f"{project_dir}/data")
results_dir = f"{output_root}/results"
figure_dir = f"{output_root}/figures"
variants_path = f"{results_dir}/variants.parquet"
clusters_path = f"{results_dir}/domain_clusters.csv"
score_column = "DMS_score"
scaling = "zscore"          # zscore or rank
cluster_column = "cluster_60"
focus_family = "PF00018"
min_datasets = 5
rng_seed = 67

os.makedirs(figure_dir, exist_ok = True)
rng = np.random.default_rng(rng_seed)


def scaled(values):
    # the two studies report on different scales, so every distribution is put on a
    # common footing before any distance is taken
    values = np.asarray(values, dtype = float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    if scaling == "rank":
        return rankdata(values) / len(values)
    spread = values.std()
    return (values - values.mean()) / spread if spread > 0 else values - values.mean()


variants = pd.read_parquet(variants_path)
clusters = pd.read_csv(clusters_path)
pooled_clusters = clusters[clusters['scope'] == 'pooled'] if 'scope' in clusters.columns else clusters

print(f"variants loaded: {len(variants):,}")
print(f"scaling: {scaling}, cluster column: {cluster_column}")

rows = []
for (paper, family), group in variants.dropna(subset = ['pfam_acc']).groupby(['paper', 'pfam_acc']):
    datasets = sorted(group['dataset'].unique())
    if len(datasets) < min_datasets:
        continue
    distributions = {d: scaled(group.loc[group['dataset'] == d, score_column]) for d in datasets}
    datasets = [d for d in datasets if distributions[d].size > 0]
    if len(datasets) < min_datasets:
        continue

    n = len(datasets)
    distance = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            w = wasserstein_distance(distributions[datasets[i]], distributions[datasets[j]])
            distance[i, j] = distance[j, i] = w

    # does assay-distribution similarity track sequence clustering? if it does, the
    # heatmap shows blocks along the diagonal and a cluster split would confound
    # "new domain" with "new assay distribution"
    members = pooled_clusters[pooled_clusters['pfam_acc'] == family]
    label_of = dict(zip(members['label'], members[cluster_column])) if cluster_column in members.columns else {}
    assigned = np.array([label_of.get(d.replace('_substitutions', '').replace('_indels', ''), -1) for d in datasets])

    within, between = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if assigned[i] < 0 or assigned[j] < 0:
                continue
            (within if assigned[i] == assigned[j] else between).append(distance[i, j])

    record = {'paper': paper, 'pfam_acc': family, 'n_datasets': n,
              'median_distance': round(float(np.median(distance[np.triu_indices(n, 1)])), 4),
              'n_within_pairs': len(within), 'n_between_pairs': len(between)}
    if len(within) >= 3 and len(between) >= 3:
        observed = np.mean(between) - np.mean(within)
        combined = np.array(within + between)
        n_within = len(within)
        null = []
        for _ in range(2000):
            shuffled = rng.permutation(combined)
            null.append(shuffled[n_within:].mean() - shuffled[:n_within].mean())
        record['mean_within'] = round(float(np.mean(within)), 4)
        record['mean_between'] = round(float(np.mean(between)), 4)
        record['gap'] = round(float(observed), 4)
        record['p_value'] = round(float((np.sum(np.array(null) >= observed) + 1) / (len(null) + 1)), 4)
    rows.append(record)

    if family == focus_family:
        order = leaves_list(linkage(squareform(distance, checks = False), method = 'average'))
        fig, ax = plt.subplots(figsize = (7.5, 6.2))
        sns.heatmap(distance[np.ix_(order, order)], cmap = "mako_r", square = True,
                    xticklabels = False, yticklabels = False,
                    cbar_kws = {"label": f"Wasserstein distance ({scaling}-scaled)"}, ax = ax)
        ax.set_title(f"Assay distribution distances: {family}, {paper}\n"
                     f"{n} datasets, ordered by distance clustering", fontsize = 11)
        plt.tight_layout()
        path = f"{figure_dir}/assay_distance_{family}_{paper}.png"
        plt.savefig(path, dpi = 150)
        plt.close()
        print(f"saved {path}")

report = pd.DataFrame(rows)
print()
print("=" * 95)
print("ASSAY DISTRIBUTION DISTANCES, WITHIN EACH STUDY")
print("=" * 95)
print(report.to_string(index = False))
print()
print("gap = mean between-cluster distance minus mean within-cluster distance.")
print("a positive gap with a small p-value means assay distributions track sequence")
print("clusters, which is the diagonal-block pattern we do not want.")

report.to_csv(f"{results_dir}/assay_distance_report.csv", index = False)
print()
print(f"Report saved to {results_dir}/assay_distance_report.csv")
