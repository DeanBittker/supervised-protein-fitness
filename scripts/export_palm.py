import os
import numpy as np
import pandas as pd

from encoding import filter_rows

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
export_dir = f"{output_root}/palm"
variants_path = f"{results_dir}/variants.parquet"
family = os.environ.get("SPF_FAMILY", "PF00018")
threshold = float(os.environ.get("SPF_THRESHOLD", "0.6"))
split_mode = os.environ.get("SPF_SPLIT_MODE", "replicate")
fold = int(os.environ.get("SPF_FOLD", "0"))
target = os.environ.get("SPF_TARGET", "zscore")
# PALM pads short sequences with zeros, so an insertion is fine, but a deletion is a
# gap character its alphabet has no token for
row_filter = os.environ.get("SPF_ROWS", "drop_deletions")
papers = ['lehner', 'rocklin']

os.makedirs(export_dir, exist_ok = True)

# PALM reads one csv per dataset holding a sequence column, the target under the name
# its task expects, and a data_split column of train/val/test. Our splits assign whole
# domains rather than rows, so the split a variant gets is the split of its wild type.

variants = pd.read_parquet(variants_path)
variants = variants[variants['pfam_acc'] == family].copy()
variants = filter_rows(variants, row_filter)

grouped = variants.groupby('dataset')['DMS_score']
variants['value_raw'] = variants['DMS_score']
# z-scoring within a dataset first stops one assay's range dominating the min-max
# scaling PALM applies afterwards
variants['value_zscore'] = (variants['DMS_score'] - grouped.transform('mean')) / \
                           grouped.transform('std').replace(0, np.nan)

constructs = pd.read_csv(f"{results_dir}/constructs.csv")
wt_of = dict(zip(constructs['construct'], constructs['wt_sequence']))

# "within" is the control for the other two: the same variants and the same model, but
# split at random inside each protein instead of holding whole proteins out. A model
# that cannot learn this one is not telling us anything about transfer across domains,
# it is telling us the wiring is wrong
if split_mode == 'within':
    splits = None
else:
    if split_mode == 'loco':
        splits = pd.read_csv(f"{results_dir}/loco_{family}.csv")
        fold_column = 'fold'
    else:
        splits = pd.read_csv(f"{results_dir}/splits_{family}.csv")
        fold_column = 'replicate'
    splits = splits[np.isclose(splits['threshold'], threshold)]
    splits = splits[splits[fold_column] == fold]


def split_within(rows, seed):
    """Assign each variant of a protein to train, validate or test at eight to one to one.

    Drawn inside each dataset so every protein appears in all three, which is the whole
    point of the control: nothing is held out that the model has not seen a relative of.
    """
    generator = np.random.default_rng(seed)
    membership = pd.Series(index = rows.index, dtype = object)
    for _, index in rows.groupby('dataset').groups.items():
        order = generator.permutation(len(index))
        cut_train = int(0.8 * len(index))
        cut_val = int(0.9 * len(index))
        labels = np.empty(len(index), dtype = object)
        labels[order[:cut_train]] = 'train'
        labels[order[cut_train:cut_val]] = 'val'
        labels[order[cut_val:]] = 'test'
        membership.loc[index] = labels
    return membership

print(f"family {family}, {split_mode} {fold}, target {target}, rows {row_filter}")
for paper in papers:
    rows = variants[variants['paper'] == paper]
    if rows.empty:
        print(f"  {paper}: nothing to export")
        continue
    labels = None if splits is None else splits[splits['scope'] == paper]
    if labels is not None and labels.empty:
        print(f"  {paper}: nothing to export")
        continue

    if labels is None:
        membership = split_within(rows, seed = 67 + fold)
    else:
        # several constructs can share a wild-type sequence, so membership is routed
        # through the sequence rather than the label, or those variants go unassigned
        split_of_wt = {wt_of[row['label']]: row['split']
                       for _, row in labels.iterrows() if row['label'] in wt_of}
        membership = rows['construct'].map(wt_of).map(split_of_wt)
    # PALM expects val where our splits say validate, and leave-one-cluster-out has no
    # validation fold at all, so part of train stands in for it
    membership = membership.replace({'validate': 'val'})

    frame = pd.DataFrame({
        'sequence': rows['mutated_sequence'].astype(str),
        'value_real': rows[f"value_{target}"],
        'data_split': membership,
        'dataset': rows['dataset'],
    })
    unassigned = int(frame['data_split'].isna().sum())
    frame = frame[frame['data_split'].notna() & frame['value_real'].notna()]
    # Rows arrive grouped by domain, so a loader that takes the file in order gets a
    # batch holding one protein and a gradient that lurches between domains instead of
    # averaging over them. Whether PALM shuffles is PALM's business; shuffling here
    # means the question never arises, and the control is exported the same way.
    frame = frame.sample(frac = 1, random_state = 67 + fold).reset_index(drop = True)

    path = f"{export_dir}/{family}_{paper}_{split_mode}{fold}.csv"
    frame.to_csv(path)
    counts = frame['data_split'].value_counts().to_dict()
    domains = frame.groupby('data_split')['dataset'].nunique().to_dict()
    print(f"  {paper}: {len(frame):,} rows, {unassigned:,} unassigned and dropped")
    print(f"    rows per split    {counts}")
    print(f"    domains per split {domains}")
    if 'val' not in counts:
        print("    WARNING: no validation split, PALM needs one unless the config "
              "gives val a fraction of its own")
    print(f"    wrote {path}")
