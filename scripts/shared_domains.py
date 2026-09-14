import os
import numpy as np
import pandas as pd
import biotite.sequence as seq
import biotite.sequence.align as align

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
constructs_path = f"{project_dir}/results/constructs.csv"
output_dir = f"{project_dir}/results"
min_wt_seqs = 5
identity_mode = "shortest"
near_identical = 0.95

os.makedirs(output_dir, exist_ok = True)

constructs = pd.read_csv(constructs_path)

# exact duplicates first: the same wild-type sequence assayed more than once
exact = (constructs.groupby('wt_sequence')
         .agg(n_constructs = ('construct', 'nunique'),
              papers = ('paper', lambda s: '+'.join(sorted(set(s)))),
              constructs_list = ('construct', lambda s: ';'.join(sorted(s))),
              pfam_acc = ('pfam_acc', 'first'),
              n_variants = ('n_variants', 'sum'))
         .reset_index())
repeated = exact[exact['n_constructs'] > 1]
cross_study = repeated[repeated['papers'] == 'lehner+rocklin']

print("=" * 70)
print("EXACT WILD-TYPE SEQUENCE MATCHES")
print("=" * 70)
print(f"distinct wild-type sequences        {len(exact):>6,}")
print(f"  assayed by more than one construct {len(repeated):>5,}")
print(f"  of those, measured by both papers  {len(cross_study):>5,}")
if len(repeated):
    print()
    print(repeated[['wt_sequence', 'pfam_acc', 'papers', 'constructs_list']].to_string(index = False, max_colwidth = 45))

# near-identical pairs across studies, found within each family's alignment
annotated = constructs.dropna(subset = ['pfam_acc'])
domains = (annotated.groupby(['pfam_acc', 'wt_sequence'])
           .agg(paper = ('paper', lambda s: '+'.join(sorted(set(s)))),
                label = ('construct', 'first'),
                n_variants = ('n_variants', 'sum'))
           .reset_index())

family_sizes = domains.groupby('pfam_acc').size()
families = sorted(family_sizes[family_sizes >= min_wt_seqs].index)
matrix = align.SubstitutionMatrix.std_protein_matrix()

pairs = []
for family in families:
    members = domains[domains['pfam_acc'] == family].reset_index(drop = True)
    proteins = [seq.ProteinSequence(s) for s in members['wt_sequence']]
    alignment = align.align_multiple(proteins, matrix, gap_penalty = (-10, -1))[0]
    identity = align.get_pairwise_sequence_identity(alignment, mode = identity_mode)
    for i, j in zip(*np.triu_indices(len(members), 1)):
        if identity[i, j] < near_identical:
            continue
        if members.loc[i, 'paper'] == members.loc[j, 'paper']:
            continue
        pairs.append({
            'pfam_acc': family,
            'identity': round(float(identity[i, j]), 3),
            'label_a': members.loc[i, 'label'],
            'paper_a': members.loc[i, 'paper'],
            'n_variants_a': members.loc[i, 'n_variants'],
            'label_b': members.loc[j, 'label'],
            'paper_b': members.loc[j, 'paper'],
            'n_variants_b': members.loc[j, 'n_variants'],
        })

shared = pd.DataFrame(pairs)
print()
print("=" * 70)
print(f"CROSS-STUDY PAIRS AT >= {near_identical:.0%} IDENTITY")
print("=" * 70)
print(f"pairs found across {len(families)} families: {len(shared)}")
if len(shared):
    print()
    print(shared.sort_values('identity', ascending = False).to_string(index = False))
    print()
    print("families contributing such pairs:")
    print(shared.groupby('pfam_acc').size().sort_values(ascending = False).to_string())

shared_path = f"{output_dir}/shared_domains.csv"
shared.to_csv(shared_path, index = False)
print()
print(f"Cross-study pairs saved to {shared_path}")
