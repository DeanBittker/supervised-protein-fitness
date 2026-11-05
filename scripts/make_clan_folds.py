import os
import numpy as np
import pandas as pd

# Turns the extracted domain instances into a leave-one-clan-out fold assignment, and
# writes down what each fold actually holds. The fold sizes matter as much as the
# assignment: the clans are wildly uneven, so a flat mean over folds would weight a
# 95-instance clan the same as a 19-instance one, and fold difficulty is confounded
# with how many papers and how long a domain the clan carries.
#
# A clan is only worth holding out if what is left to test on is a real test set, so
# clans under the protein threshold stay in training for every fold rather than
# becoming one-protein folds that only add noise.
#
#   SPF_OUTPUT=<shared folder> python scripts/make_clan_folds.py

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
extract_dir = f"{output_root}/results/extraction"
instances_path = f"{extract_dir}/instances.csv"

min_proteins = int(os.environ.get("SPF_MIN_FOLD_PROTEINS", 5))

if not os.path.exists(instances_path):
    raise SystemExit(f"{instances_path} not found, run scripts/pfam_extract.py first")

instances = pd.read_csv(instances_path)
usable = instances[(instances['is_primary_condition'] == True) &
                   (instances['score_distribution_status'] == 'ok')].copy()
print(f"{len(usable):,} usable instances in {usable['group'].nunique():,} groups")

per_group = usable.groupby('group').agg(
    instances = ('filename', 'size'), datasets = ('filename', 'nunique'),
    proteins = ('protein', 'nunique'), papers = ('paper', 'nunique'),
    families = ('pfam', 'nunique'), median_residues = ('covered', 'median'))

held_out = per_group[per_group['proteins'] >= min_proteins].sort_values(
    'proteins', ascending = False)
print(f"{len(held_out)} groups carry {min_proteins}+ proteins and become folds; "
      f"the other {len(per_group) - len(held_out)} stay in training throughout "
      f"({int(per_group.loc[~per_group.index.isin(held_out.index), 'instances'].sum()):,} "
      f"instances)")

# one row per fold per instance would be enormous, so the assignment is stored as the
# held-out group per fold and read back by joining on 'group'
folds = held_out.reset_index()
folds.insert(0, 'fold', range(len(folds)))
folds['train_instances'] = len(usable) - folds['instances']
folds['test_share'] = (folds['instances'] / len(usable)).round(4)

print("\nfolds, deepest first. test_share is how much of the data one fold removes, "
      "and papers is how many different labs' assays sit inside it.")
print(folds[['fold', 'group', 'instances', 'proteins', 'papers', 'families',
             'median_residues', 'test_share']].to_string(index = False))

spread = folds['instances']
print(f"\nimbalance: largest fold {int(spread.max())} instances, smallest "
      f"{int(spread.min())}, ratio {spread.max() / spread.min():.1f}x")
print(f"  median residues across folds runs {int(folds['median_residues'].min())} to "
      f"{int(folds['median_residues'].max())}, a "
      f"{folds['median_residues'].max() / folds['median_residues'].min():.1f}x spread, "
      f"so a per-fold score is not comparable across folds without saying so")
multi = folds[folds['papers'] >= 10]
if not multi.empty:
    print(f"  {len(multi)} folds draw on ten or more papers "
          f"({', '.join(multi['group'])}), so within those folds assay heterogeneity "
          f"is mixed into whatever the held-out score shows")

# the instance-level view, so a training script never has to redo this grouping
assignment = usable[['filename', 'protein', 'paper', 'pfam', 'clan', 'group', 'repeat',
                     'covered_start', 'covered_end', 'covered']].copy()
assignment['is_fold'] = assignment['group'].isin(set(folds['group']))
assignment = assignment.merge(folds[['fold', 'group']], on = 'group', how = 'left')

for frame, name, index in [(folds, 'folds', False), (assignment, 'fold_assignment', False),
                           (per_group, 'group_summary', True)]:
    path = f"{extract_dir}/{name}.csv"
    frame.to_csv(path, index = index)
    print(f"\nwrote {path}")

# every dataset these folds touch, training groups included, as a plain list for
# embed_esm.py. A clan spans families, so the embedding step cannot be selected by
# family and needs to be told exactly which datasets to cover.
needed = sorted(assignment['filename'].unique())
list_path = f"{extract_dir}/fold_datasets.txt"
with open(list_path, "w") as handle:
    handle.write("\n".join(needed) + "\n")
print(f"\nwrote {list_path} ({len(needed):,} datasets)")
