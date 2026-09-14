import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import biotite.sequence as seq
import biotite.sequence.align as align
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
constructs_path = f"{project_dir}/results/constructs.csv"
output_dir = f"{project_dir}/results"
min_wt_seqs = 5
focus_family = "PF00018"
identity_mode = "shortest"
linkage_method = "average"
thresholds = [0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30]

os.makedirs(output_dir, exist_ok = True)


def identity_matrix(sequences):
    # all-by-all msa first, then identity over the shorter sequence of each pair,
    # which is the same convention used in the PHYLO notebook
    proteins = [seq.ProteinSequence(s) for s in sequences]
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignment = align.align_multiple(proteins, matrix, gap_penalty = (-10, -1))[0]
    identity = align.get_pairwise_sequence_identity(alignment, mode = identity_mode)
    np.fill_diagonal(identity, 1.0)
    return identity, len(alignment)


def cluster_at(identity, threshold):
    distance = 1 - identity
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2
    tree = linkage(squareform(distance, checks = False), method = linkage_method)
    return fcluster(tree, t = 1 - threshold, criterion = 'distance')


constructs = pd.read_csv(constructs_path)
annotated = constructs.dropna(subset = ['pfam_acc'])

# one row per distinct wild-type domain, with its variants summed across constructs
domains = (annotated.groupby(['pfam_acc', 'wt_sequence'])
           .agg(n_variants = ('n_variants', 'sum'),
                papers = ('paper', lambda s: '+'.join(sorted(set(s)))),
                label = ('construct', 'first'))
           .reset_index())
domains['wt_length'] = domains['wt_sequence'].str.len()

family_sizes = domains.groupby('pfam_acc').size()
families = sorted(family_sizes[family_sizes >= min_wt_seqs].index)
print(f"clustering {len(families)} families with at least {min_wt_seqs} distinct wild-type domains")
print(f"identity mode = {identity_mode}, linkage = {linkage_method}")

rows = []
assignments = []
for family in families:
    members = domains[domains['pfam_acc'] == family].reset_index(drop = True)
    identity, n_columns = identity_matrix(members['wt_sequence'].tolist())
    off_diagonal = identity[np.triu_indices(len(members), 1)]

    member_clusters = members[['pfam_acc', 'wt_sequence', 'label', 'papers',
                               'n_variants', 'wt_length']].copy()
    for threshold in thresholds:
        labels = cluster_at(identity, threshold)
        sizes = pd.Series(labels).value_counts()
        variants = members.groupby(labels)['n_variants'].sum()
        member_clusters[f"cluster_{int(threshold * 100)}"] = labels
        rows.append({
            'pfam_acc': family,
            'n_domains': len(members),
            'msa_columns': n_columns,
            'median_identity': round(float(np.median(off_diagonal)), 3),
            'max_identity': round(float(off_diagonal.max()), 3),
            'threshold': threshold,
            'n_clusters': int(sizes.size),
            'n_singletons': int((sizes == 1).sum()),
            'largest_cluster': int(sizes.max()),
            'pct_in_largest': round(100 * sizes.max() / len(members), 1),
            'pct_variants_in_largest': round(100 * variants.max() / members['n_variants'].sum(), 1),
        })
    assignments.append(member_clusters)

    if family == focus_family:
        order = leaves_list(linkage(squareform(1 - identity, checks = False), method = linkage_method))
        fig, ax = plt.subplots(figsize = (7, 6))
        sns.heatmap(identity[np.ix_(order, order)], cmap = "viridis", vmin = 0, vmax = 1, square = True,
                    xticklabels = False, yticklabels = False,
                    cbar_kws = {"label": "sequence identity"}, ax = ax)
        ax.set_title(f"Pairwise identity: {family}\n"
                     f"{len(members)} distinct wild-type domains, identity mode = {identity_mode}",
                     fontsize = 10)
        plt.tight_layout()
        heatmap_path = f"{output_dir}/identity_{family}.png"
        plt.savefig(heatmap_path, dpi = 150)
        print(f"Identity heatmap saved to {heatmap_path}")

clusters = pd.concat(assignments, ignore_index = True)
report = pd.DataFrame(rows)

print()
print("=" * 90)
print(f"CLUSTERING OF {focus_family}")
print("=" * 90)
print(report[report['pfam_acc'] == focus_family].to_string(index = False))

print()
print("=" * 90)
print("ACROSS ALL FAMILIES: HOW BALANCED ARE THE CLUSTERS AT EACH THRESHOLD")
print("=" * 90)
balance = report.groupby('threshold').agg(
    families = ('pfam_acc', 'nunique'),
    median_clusters = ('n_clusters', 'median'),
    median_singleton_frac = ('n_singletons', 'median'),
    median_pct_in_largest = ('pct_in_largest', 'median'),
    families_all_singletons = ('pct_in_largest', lambda s: int((s < 15).sum())),
    families_one_big_cluster = ('pct_in_largest', lambda s: int((s > 50).sum())),
)
print(balance.to_string())

# a threshold is usable for splitting only if a family has enough clusters to hold
# out whole clusters without one of them dominating the family
usable = report[(report['n_clusters'] >= 5) & (report['pct_in_largest'] <= 50)]
print()
print("families with >=5 clusters and no cluster holding >50% of domains:")
print(usable.groupby('threshold').size().to_string())

report_path = f"{output_dir}/cluster_report.csv"
report.to_csv(report_path, index = False)
print()
print(f"Cluster report saved to {report_path}")

clusters_path = f"{output_dir}/domain_clusters.csv"
clusters.to_csv(clusters_path, index = False)
print(f"Domain cluster assignments saved to {clusters_path}")
