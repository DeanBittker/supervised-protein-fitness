import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
# outputs go to the repo by default; on O2 set SPF_OUTPUT to the shared project
# folder so results are computed once and pulled down rather than recomputed
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
embedding_dir = f"{output_root}/embeddings"
variants_path = f"{results_dir}/variants.parquet"
family = os.environ.get("SPF_FAMILY", "PF00018")
threshold = float(os.environ.get("SPF_THRESHOLD", "0.6"))
n_replicates = int(os.environ.get("SPF_REPLICATES", "10"))
n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
cache_tag = "esm150M"
papers = ['lehner', 'rocklin']
encodings = ['onehot', 'ESM2-150M']
models = ['ridge', 'rf', 'xgb']
targets = ['raw', 'zscore']
drop_insertions = True
dry_run = os.environ.get("SPF_DRYRUN", "") == "1"

os.makedirs(results_dir, exist_ok = True)
print(f"family {family}, threshold {threshold:.0%}, {n_replicates} replicates, n_jobs {n_jobs}")


def r2(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.size == 0:
        return np.nan
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def rho(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.size == 0:
        return np.nan
    return spearmanr(y, p)[0]


def read_alignment(path):
    """Return {ungapped sequence: aligned sequence} from a gapped fasta."""
    aligned = {}
    label, chunks = None, []
    for line in open(path):
        line = line.strip()
        if line.startswith('>'):
            if label is not None:
                row = ''.join(chunks)
                aligned[row.replace('-', '')] = row
            label, chunks = line[1:], []
        elif line:
            chunks.append(line)
    if label is not None:
        row = ''.join(chunks)
        aligned[row.replace('-', '')] = row
    return aligned


def to_alignment_columns(sequences, wt_sequence, aligned_wt):
    """Scatter each variant into its wild type's alignment columns.

    Domains in a family differ in length, so raw position i means a different
    thing in each one and a fixed-length encoding cannot be shared across them.
    Projecting onto the alignment gives every domain the same column space.
    A variant has the same length as its own wild type, because deletions are
    gap characters in place and insertions have been dropped.
    """
    columns = [i for i, c in enumerate(aligned_wt) if c != '-']
    width = len(aligned_wt)
    block = np.full((len(sequences), width), '-', dtype='<U1')
    placed = np.zeros(len(sequences), dtype=bool)
    for row, sequence in enumerate(sequences):
        if len(sequence) != len(wt_sequence):
            continue
        for position, column in enumerate(columns):
            block[row, column] = sequence[position]
        placed[row] = True
    return block, placed


def build_model(name):
    if name == 'ridge':
        return Ridge(alpha = 1.0)
    if name == 'rf':
        return RandomForestRegressor(n_estimators = 100, random_state = 67, n_jobs = n_jobs)
    return xgb.XGBRegressor(n_estimators = 100, max_depth = 6, learning_rate = 0.1,
                            random_state = 67, n_jobs = n_jobs)


variants = pd.read_parquet(variants_path)
variants = variants[variants['pfam_acc'] == family].copy()
if drop_insertions:
    variants = variants[~variants['has_insertion']]
variants = variants.sort_values(['dataset']).reset_index(drop = True)

# the z-scored target is standardised within each dataset, so a model is never
# rewarded for learning which domain a variant came from
grouped = variants.groupby('dataset')['DMS_score']
variants['score_raw'] = variants['DMS_score']
variants['score_zscore'] = (variants['DMS_score'] - grouped.transform('mean')) / grouped.transform('std')

constructs = pd.read_csv(f"{results_dir}/constructs.csv")
splits = pd.read_csv(f"{results_dir}/splits_{family}.csv")
splits = splits[np.isclose(splits['threshold'], threshold)]

# embeddings are cached one file per dataset, in the same row order as the
# variant table sorted by dataset
# an embedding file must have exactly one row per surviving variant of its
# dataset, in the same order. a mismatch means the cache was built against a
# different filter, so it is rejected rather than silently misaligned.
embeddings = {}
stale = []
counts = variants.groupby('dataset').size()
for dataset in variants['dataset'].unique():
    path = f"{embedding_dir}/{dataset}_{cache_tag}.npy"
    if not os.path.exists(path):
        continue
    array = np.load(path)
    if array.shape[0] != counts[dataset]:
        stale.append((dataset, array.shape[0], int(counts[dataset])))
        continue
    embeddings[dataset] = array
missing = [d for d in variants['dataset'].unique() if d not in embeddings]
if stale:
    print(f"WARNING: {len(stale)} embedding files have the wrong row count and were rejected")
    for dataset, got, want in stale[:3]:
        print(f"  {dataset}: {got} rows cached, {want} expected")
    print("  rebuild them with embed_esm.py, which applies the same insertion filter")
if missing:
    print(f"WARNING: {len(missing)} datasets have no usable embedding, ESM2-150M cells will be skipped")

cache_path = f"{results_dir}/model_results_{family}.csv"
done = set()
if os.path.exists(cache_path):
    previous = pd.read_csv(cache_path)
    done = set(map(tuple, previous[['paper', 'encoding', 'model', 'target', 'replicate']].values))
    print(f"resuming: {len(done)} cells already done")

rows = []
for paper in papers:
    paper_rows = variants[variants['paper'] == paper]
    if paper_rows.empty:
        continue
    labels = splits[splits['scope'] == paper]
    if labels.empty:
        print(f"no split for {paper} at this threshold, skipped")
        continue

    # a split assigns whole domains, so map each domain label to its variants
    # the embedding matrix is built in this same dataset order, before any row filter
    datasets_present = sorted(paper_rows['dataset'].unique())
    # project every variant onto its family's alignment columns once per paper
    alignment = read_alignment(f"{results_dir}/msa/{paper}_{family}.fasta")
    wt_of = dict(zip(constructs['construct'], constructs['wt_sequence']))
    # a split names one representative construct per domain, but several constructs
    # can share a wild-type sequence, so route through the sequence rather than the
    # label or those variants are silently left out of every partition
    split_of_wt = {wt_of[row['label']]: row['split']
                   for _, row in labels.iterrows() if row['label'] in wt_of}
    width = len(next(iter(alignment.values())))
    blocks, placed_parts, gaps = [], [], []
    for dataset in sorted(paper_rows['dataset'].unique()):
        rows_here = paper_rows[paper_rows['dataset'] == dataset]
        wt = wt_of.get(rows_here['construct'].iloc[0])
        if wt is None or wt not in alignment:
            print(f"  no alignment row for {dataset}, its variants cannot be encoded")
            blocks.append(np.full((len(rows_here), width), '-', dtype='<U1'))
            placed_parts.append(np.zeros(len(rows_here), dtype=bool))
            continue
        block, placed = to_alignment_columns(
            rows_here['mutated_sequence'].astype(str).tolist(), wt, alignment[wt])
        blocks.append(block)
        placed_parts.append(placed)
        lengths = rows_here['mutated_sequence'].astype(str).str.len().values
        gaps.extend((lengths[~placed] - len(wt)).tolist())
    chars = np.concatenate(blocks)
    placed = np.concatenate(placed_parts)

    # a variant that does not match its wild type's length cannot be projected onto
    # the alignment. those rows are dropped from every encoding, not just one-hot,
    # so the encodings are always compared on an identical row set.
    if (~placed).any():
        offsets = pd.Series(gaps).value_counts().head(5)
        print(f"{paper}: {int((~placed).sum()):,} variants do not match their wild-type length and are dropped")
        print(f"  length difference from wild type: {dict(offsets)}")
    paper_rows = paper_rows[placed].copy()
    chars = chars[placed]
    print(f"{paper}: {chars.shape[1]} alignment columns, {len(paper_rows):,} variants encoded")

    if dry_run:
        usable = [d for d in paper_rows['dataset'].unique() if d in embeddings]
        print(f"  embeddings usable for {len(usable)} of {paper_rows['dataset'].nunique()} datasets")
        print(f"  variants placed into alignment columns: {len(chars):,}")
        first = labels[labels['replicate'] == 0]
        first_of_wt = {wt_of[row['label']]: row['split']
                       for _, row in first.iterrows() if row['label'] in wt_of}
        membership = paper_rows['construct'].map(wt_of).map(first_of_wt)
        print(f"  replicate 0 rows -> train {int((membership == 'train').sum()):,}  "
              f"validate {int((membership == 'validate').sum()):,}  "
              f"test {int((membership == 'test').sum()):,}  "
              f"unmapped {int(membership.isna().sum()):,}")
        continue

    for encoding in encodings:
        if encoding == 'ESM2-150M' and any(d not in embeddings for d in paper_rows['dataset'].unique()):
            print(f"skipping {paper} / {encoding}: embeddings incomplete")
            continue

        for replicate in range(n_replicates):
            assignment = labels[labels['replicate'] == replicate]
            split_of = {wt_of[row['label']]: row['split']
                        for _, row in assignment.iterrows() if row['label'] in wt_of}
            membership = paper_rows['construct'].map(wt_of).map(split_of)
            train_mask = (membership == 'train').values
            test_mask = (membership == 'test').values
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                continue

            if encoding == 'onehot':
                encoder = onehot(sparse_output = False, drop = "first", handle_unknown = "ignore")
                X_train = encoder.fit_transform(chars[train_mask])
                X_test = encoder.transform(chars[test_mask])
            else:
                matrix = np.concatenate([embeddings[d] for d in sorted(datasets_present)])[placed]
                X_train = matrix[train_mask]
                X_test = matrix[test_mask]

            for target in targets:
                column = f"score_{target}"
                for model_name in models:
                    key = (paper, encoding, model_name, target, replicate)
                    if key in done:
                        continue
                    Y_train = paper_rows.loc[train_mask, column].values
                    Y_test = paper_rows.loc[test_mask, column].values
                    keep_train = np.isfinite(Y_train)
                    keep_test = np.isfinite(Y_test)

                    model = build_model(model_name)
                    model.fit(X_train[keep_train], Y_train[keep_train])
                    train_pred = model.predict(X_train[keep_train])
                    test_pred = model.predict(X_test[keep_test])

                    # ranking variants from independent experiments against each
                    # other is not meaningful, so the per-protein average is the
                    # primary figure and the pooled one is kept for comparison
                    scored = paper_rows[test_mask][keep_test].assign(
                        truth = Y_test[keep_test], prediction = test_pred)
                    per_protein = [rho(group['truth'], group['prediction'])
                                   for _, group in scored.groupby('dataset')
                                   if len(group) > 10 and group['truth'].std() > 0]

                    rows.append({
                        'paper': paper, 'encoding': encoding, 'model': model_name,
                        'target': target, 'replicate': replicate,
                        'n_train': int(keep_train.sum()), 'n_test': int(keep_test.sum()),
                        'n_train_domains': int(assignment[assignment['split'] == 'train'].shape[0]),
                        'n_test_domains': int(assignment[assignment['split'] == 'test'].shape[0]),
                        'train_r2': round(float(r2(Y_train[keep_train], train_pred)), 4),
                        'test_r2': round(float(r2(Y_test[keep_test], test_pred)), 4),
                        'train_spearman': round(float(rho(Y_train[keep_train], train_pred)), 4),
                        'test_spearman': round(float(rho(Y_test[keep_test], test_pred)), 4),
                        'test_spearman_per_protein': round(float(np.mean(per_protein)), 4)
                            if per_protein else np.nan,
                        'test_spearman_per_protein_sd': round(float(np.std(per_protein)), 4)
                            if len(per_protein) > 1 else np.nan,
                        'n_test_proteins_scored': len(per_protein),
                    })
                    print(f"  {paper:8s} {encoding:10s} {model_name:5s} {target:6s} rep {replicate:2d}  "
                          f"pooled rho {rows[-1]['test_spearman']:7.4f}  "
                          f"per-protein rho {rows[-1]['test_spearman_per_protein']:7.4f} "
                          f"(n={rows[-1]['n_test_proteins_scored']})", flush=True)

                    # written every cell so a preempted job resumes rather than restarts
                    frame = pd.DataFrame(rows)
                    if os.path.exists(cache_path):
                        frame = pd.concat([pd.read_csv(cache_path), frame], ignore_index = True)
                        frame = frame.drop_duplicates(
                            subset = ['paper', 'encoding', 'model', 'target', 'replicate'], keep = 'last')
                    frame.to_csv(cache_path, index = False)
                    rows = []

if dry_run:
    print()
    print("dry run: nothing was fitted. unset SPF_DRYRUN to run the grid.")
    raise SystemExit(0)

print()
print(f"Results saved to {cache_path}")
final = pd.read_csv(cache_path)
print(f"{len(final)} cells complete")
print()
print(final.groupby(['paper', 'encoding', 'model', 'target'])[
    ['test_spearman_per_protein', 'test_spearman']].agg(['median', 'std']).round(3).to_string())
