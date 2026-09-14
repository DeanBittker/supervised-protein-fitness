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
              pfam_acc = ('pfam_acc', 'first'))
         .reset_index())
repeated = exact[exact['n_constructs'] > 1]

print("=" * 70)
print("EXACT WILD-TYPE SEQUENCE MATCHES")
print("=" * 70)
print(f"distinct wild-type sequences         {len(exact):>6,}")
print(f"  assayed by more than one construct {len(repeated):>6,}")
print(f"  of those, measured by both papers  {(repeated['papers'] == 'lehner+rocklin').sum():>6,}")

# near-identical pairs across studies. a domain is keyed on its wild-type sequence,
# so one lehner domain matching four rocklin constructs of the same protein is one
# domain pair, not four. only the best rocklin match per lehner domain is kept.
annotated = constructs.dropna(subset = ['pfam_acc'])
annotated = annotated.assign(
    lehner_name = np.where(annotated['paper'] == 'lehner', annotated['construct'], None),
    rocklin_name = np.where(annotated['paper'] == 'rocklin', annotated['construct'], None))


def first_name(values):
    names = sorted(v for v in values if isinstance(v, str))
    return names[0] if names else None


domains = (annotated.groupby(['pfam_acc', 'wt_sequence'])
           .agg(papers = ('paper', lambda s: set(s)),
                lehner_label = ('lehner_name', first_name),
                rocklin_label = ('rocklin_name', first_name),
                lehner_variants = ('n_variants', lambda s: s[annotated.loc[s.index, 'paper'] == 'lehner'].sum()),
                rocklin_variants = ('n_variants', lambda s: s[annotated.loc[s.index, 'paper'] == 'rocklin'].sum()))
           .reset_index())

family_sizes = domains.groupby('pfam_acc').size()
families = sorted(family_sizes[family_sizes >= min_wt_seqs].index)
matrix = align.SubstitutionMatrix.std_protein_matrix()

best = []
for family in families:
    members = domains[domains['pfam_acc'] == family].reset_index(drop = True)
    proteins = [seq.ProteinSequence(s) for s in members['wt_sequence']]
    alignment = align.align_multiple(proteins, matrix, gap_penalty = (-10, -1))[0]
    identity = align.get_pairwise_sequence_identity(alignment, mode = identity_mode)
    np.fill_diagonal(identity, 0.0)

    in_lehner = members['papers'].apply(lambda p: 'lehner' in p).values
    in_rocklin = members['papers'].apply(lambda p: 'rocklin' in p).values

    for i in np.where(in_lehner)[0]:
        # the partner must carry rocklin data and must not be this same sequence
        candidates = np.where(in_rocklin)[0]
        candidates = candidates[candidates != i]
        if len(candidates) == 0:
            continue
        j = candidates[np.argmax(identity[i, candidates])]
        if identity[i, j] < near_identical:
            continue
        best.append({
            'pfam_acc': family,
            'identity': round(float(identity[i, j]), 3),
            'lehner_domain': members.loc[i, 'lehner_label'],
            'lehner_variants': members.loc[i, 'lehner_variants'],
            'rocklin_domain': members.loc[j, 'rocklin_label'],
            'rocklin_variants': members.loc[j, 'rocklin_variants'],
            'same_sequence': bool(identity[i, j] >= 1.0),
        })

shared = pd.DataFrame(best).sort_values('identity', ascending = False)
print()
print("=" * 70)
print(f"CROSS-STUDY DOMAIN PAIRS AT >= {near_identical:.0%} IDENTITY")
print("=" * 70)
print(f"distinct lehner domains with a rocklin match: {len(shared)}")
print(f"  of those, sequence-identical:               {int(shared['same_sequence'].sum()) if len(shared) else 0}")
if len(shared):
    print()
    print(shared.to_string(index = False))
    print()
    print("families contributing:")
    print(shared.groupby('pfam_acc').size().sort_values(ascending = False).to_string())
    print()
    print(f"total variants available on the lehner side:  {shared['lehner_variants'].sum():>8,.0f}")
    print(f"total variants available on the rocklin side: {shared['rocklin_variants'].sum():>8,.0f}")

shared_path = f"{output_dir}/shared_domains.csv"
shared.to_csv(shared_path, index = False)
print()
print(f"Cross-study pairs saved to {shared_path}")
