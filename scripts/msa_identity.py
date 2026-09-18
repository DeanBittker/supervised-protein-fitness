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
# outputs go to the repo by default; on O2 set SPF_OUTPUT to the shared project
# folder so results are computed once and pulled down rather than recomputed
output_root = os.environ.get("SPF_OUTPUT", project_dir)
data_root = os.environ.get("SPF_DATA", f"{project_dir}/data")
constructs_path = f"{output_root}/results/constructs.csv"
results_dir = f"{output_root}/results"
msa_dir = f"{results_dir}/msa"
identity_dir = f"{results_dir}/identity"
figure_dir = f"{output_root}/figures/identity"
min_wt_seqs = 5
identity_mode = "shortest"
linkage_method = "average"
thresholds = [0.90, 0.60, 0.30]
split_threshold = 0.60
split_ratios = {'train': 0.80, 'validate': 0.10, 'test': 0.10}
heatmap_min_domains = 10
scopes = ['pooled', 'lehner', 'rocklin']
rng_seed = 67

for d in [msa_dir, identity_dir, figure_dir]:
    os.makedirs(d, exist_ok = True)
rng = np.random.default_rng(rng_seed)


def align_family(sequences):
    proteins = [seq.ProteinSequence(s) for s in sequences]
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    alignment = align.align_multiple(proteins, matrix, gap_penalty = (-10, -1))[0]
    identity = align.get_pairwise_sequence_identity(alignment, mode = identity_mode)
    np.fill_diagonal(identity, 1.0)
    return alignment, identity


def cluster_at(identity, threshold):
    distance = 1 - identity
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2
    tree = linkage(squareform(distance, checks = False), method = linkage_method)
    return fcluster(tree, t = 1 - threshold, criterion = 'distance'), tree


def assign_splits(sizes):
    # whole clusters go to one split. largest first, each cluster to whichever split
    # is furthest below its target share, so clusters are never broken across splits
    total = sum(n for _, n in sizes)
    targets = {k: v * total for k, v in split_ratios.items()}
    running = {k: 0 for k in split_ratios}
    assignment = {}
    for cluster_id, n in sorted(sizes, key = lambda x: -x[1]):
        deficit = {k: targets[k] - running[k] for k in running}
        pick = max(deficit, key = deficit.get)
        assignment[cluster_id] = pick
        running[pick] += n
    return assignment, running


constructs = pd.read_csv(constructs_path)
annotated = constructs.dropna(subset = ['pfam_acc'])
domains = (annotated.groupby(['pfam_acc', 'wt_sequence'])
           .agg(n_variants = ('n_variants', 'sum'),
                papers = ('paper', lambda s: '+'.join(sorted(set(s)))),
                label = ('construct', 'first'))
           .reset_index())

member_rows = []
split_rows = []
summary_rows = []

for scope in scopes:
    pool = domains if scope == 'pooled' else domains[domains['papers'].str.contains(scope)]
    sizes = pool.groupby('pfam_acc').size()
    families = sorted(sizes[sizes >= min_wt_seqs].index)
    print(f"{scope}: {len(families)} families with at least {min_wt_seqs} domains")

    for family in families:
        members = pool[pool['pfam_acc'] == family].reset_index(drop = True)
        alignment, identity = align_family(members['wt_sequence'].tolist())

        # save the alignment itself, gaps included, so it can be inspected later
        gapped = alignment.get_gapped_sequences()
        with open(f"{msa_dir}/{scope}_{family}.fasta", 'w') as handle:
            for label, row in zip(members['label'], gapped):
                handle.write(f">{label}\n{row}\n")

        frame = pd.DataFrame(identity, index = members['label'], columns = members['label'])
        frame.round(4).to_csv(f"{identity_dir}/{scope}_{family}.csv")

        off_diagonal = identity[np.triu_indices(len(members), 1)]
        summary_rows.append({
            'scope': scope, 'pfam_acc': family, 'n_domains': len(members),
            'msa_columns': len(alignment),
            'min_identity': round(float(off_diagonal.min()), 3),
            'median_identity': round(float(np.median(off_diagonal)), 3),
            'max_identity': round(float(off_diagonal.max()), 3),
            'n_variants': int(members['n_variants'].sum()),
        })

        for threshold in thresholds:
            labels, tree = cluster_at(identity, threshold)
            members['cluster'] = labels
            for cluster_id, group in members.groupby('cluster'):
                member_rows.append({
                    'scope': scope, 'pfam_acc': family, 'threshold': threshold,
                    'cluster': int(cluster_id),
                    'n_members': len(group),
                    'n_variants': int(group['n_variants'].sum()),
                    'members': ';'.join(sorted(group['label'])),
                })

        # the split protocol runs at one chosen threshold only
        labels, tree = cluster_at(identity, split_threshold)
        members['cluster'] = labels
        cluster_sizes = [(c, len(g)) for c, g in members.groupby('cluster')]
        assignment, running = assign_splits(cluster_sizes)
        members['split'] = members['cluster'].map(assignment)
        for _, row in members.iterrows():
            split_rows.append({
                'scope': scope, 'pfam_acc': family, 'label': row['label'],
                'cluster': int(row['cluster']), 'split': row['split'],
                'n_variants': int(row['n_variants']), 'papers': row['papers'],
            })

        if len(members) >= heatmap_min_domains:
            order = leaves_list(linkage(squareform(1 - identity, checks = False), method = linkage_method))
            fig, ax = plt.subplots(figsize = (7, 6))
            sns.heatmap(identity[np.ix_(order, order)], cmap = "viridis", vmin = 0, vmax = 1,
                        square = True, xticklabels = False, yticklabels = False,
                        cbar_kws = {"label": "sequence identity"}, ax = ax)
            ax.set_title(f"{family} — {scope}\n{len(members)} wild-type domains, identity over shorter sequence",
                         fontsize = 11)
            plt.tight_layout()
            plt.savefig(f"{figure_dir}/{scope}_{family}.png", dpi = 150)
            plt.close()

summary = pd.DataFrame(summary_rows)
member_table = pd.DataFrame(member_rows)
splits = pd.DataFrame(split_rows)

print()
print("=" * 95)
print("ALIGNMENT SUMMARY (top 12 families by size, pooled)")
print("=" * 95)
print(summary[summary['scope'] == 'pooled'].sort_values('n_domains', ascending = False).head(12).to_string(index = False))

print()
print("=" * 95)
print("MEMBERS AND VARIANTS PER CLUSTER — PF00018, pooled")
print("=" * 95)
focus = member_table[(member_table['pfam_acc'] == 'PF00018') & (member_table['scope'] == 'pooled')]
for threshold in thresholds:
    at = focus[focus['threshold'] == threshold].sort_values('n_members', ascending = False)
    print(f"\nidentity >= {threshold:.0%}: {len(at)} clusters")
    print(f"  members per cluster:  {list(at['n_members'])}")
    print(f"  variants per cluster: {list(at['n_variants'])}")

print()
print("=" * 95)
print(f"PROPOSED 80/10/10 SPLIT AT {split_threshold:.0%} IDENTITY, WHOLE CLUSTERS HELD OUT")
print("=" * 95)
for scope in scopes:
    scoped = splits[splits['scope'] == scope]
    if scoped.empty:
        continue
    by_split = scoped.groupby('split').agg(domains = ('label', 'size'), variants = ('n_variants', 'sum'))
    by_split['pct_domains'] = (100 * by_split['domains'] / by_split['domains'].sum()).round(1)
    by_split['pct_variants'] = (100 * by_split['variants'] / by_split['variants'].sum()).round(1)
    print(f"\n{scope} ({scoped['pfam_acc'].nunique()} families, {len(scoped)} domains):")
    print(by_split.to_string())

summary.to_csv(f"{results_dir}/msa_summary.csv", index = False)
member_table.to_csv(f"{results_dir}/cluster_members.csv", index = False)
splits.to_csv(f"{results_dir}/splits.csv", index = False)

print()
print(f"Alignments written to {msa_dir}/ ({len(os.listdir(msa_dir))} files)")
print(f"Identity matrices written to {identity_dir}/ ({len(os.listdir(identity_dir))} files)")
print(f"Heatmaps written to {figure_dir}/ ({len(os.listdir(figure_dir))} files)")
print(f"Tables: msa_summary.csv, cluster_members.csv, splits.csv")
