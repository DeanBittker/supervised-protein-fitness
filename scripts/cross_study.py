import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from encoding import to_alignment_columns, esm_models, filter_rows

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
figure_dir = f"{output_root}/figures"
embedding_dir = f"{output_root}/embeddings"
variants_path = f"{results_dir}/variants.parquet"
family = os.environ.get("SPF_FAMILY", "PF00018")
# 70% is where both directions stay viable: below it one large cluster absorbs most
# rocklin domains together with lehner ones and there is almost nothing left to test on
threshold = int(float(os.environ.get("SPF_THRESHOLD", "0.7")) * 100)
esm_name = os.environ.get("SPF_ESM", "ESM2-150M")
cache_tag = esm_models[esm_name][2]
row_filter = os.environ.get("SPF_ROWS", "drop_insertions")
n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
# ridge alone answers the question in minutes; random forest over ESM embeddings is
# tens of thousands of variants against hundreds of dimensions and dominates the runtime
# while losing to ridge in every cell so far. SPF_MODELS=ridge to get the answer first
models = os.environ.get("SPF_MODELS", "ridge,rf,xgb").split(',')
targets = ['raw', 'zscore']
encodings = ['onehot', esm_name]
directions = [('rocklin', 'lehner'), ('lehner', 'rocklin')]
min_test_variants = 10
# twenty amino acids and the gap character; insertions are filtered out upstream
residue_alphabet = 'ACDEFGHIKLMNPQRSTVWY-'

os.makedirs(results_dir, exist_ok = True)

# Can a model trained on one study's domains predict the other's? Within a study the
# assay and the protocol are fixed, so a within-study number isolates generalisation
# across domains. Across studies it also has to survive a change of measurement:
# rocklin reports stability and lehner abundance. The gap between the two is therefore
# assay transfer on top of domain distance, and only means something next to the
# within-study leave-one-cluster-out numbers computed on the same target.


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


def build_model(name):
    if name == 'ridge':
        return Ridge(alpha = 1.0)
    if name == 'rf':
        return RandomForestRegressor(n_estimators = 100, random_state = 67, n_jobs = n_jobs)
    return xgb.XGBRegressor(n_estimators = 100, max_depth = 6, learning_rate = 0.1,
                            random_state = 67, n_jobs = n_jobs)


variants = pd.read_parquet(variants_path)
variants = variants[variants['pfam_acc'] == family].copy()
variants = filter_rows(variants, row_filter)
variants = variants.sort_values(['dataset']).reset_index(drop = True)
grouped = variants.groupby('dataset')['DMS_score']
variants['score_raw'] = variants['DMS_score']
variants['score_zscore'] = (variants['DMS_score'] - grouped.transform('mean')) / \
                           grouped.transform('std').replace(0, np.nan)

constructs = pd.read_csv(f"{results_dir}/constructs.csv")
wt_of = dict(zip(constructs['construct'], constructs['wt_sequence']))

# clustering and the alignments are built locally, because biotite will not install
# against the python on O2. they are small, so they are copied up rather than rebuilt,
# and saying so here saves working out what a missing file means
for needed in [f"{results_dir}/domain_clusters.csv", f"{results_dir}/msa/pooled_{family}.fasta"]:
    if not os.path.exists(needed):
        raise SystemExit(
            f"missing {needed}\n"
            f"  it is built locally, not on the cluster, and results/ is not in git.\n"
            f"  copy it up from the repo on your laptop:\n"
            f"    rsync -avh results/domain_clusters.csv results/msa/ "
            f"<user>@transfer.rc.hms.harvard.edu:{results_dir}/")

clusters = pd.read_csv(f"{results_dir}/domain_clusters.csv")
clusters = clusters[(clusters['pfam_acc'] == family) & (clusters['scope'] == 'pooled')]
cluster_column = f"cluster_{threshold}"
if cluster_column not in clusters:
    raise SystemExit(f"{cluster_column} is not in domain_clusters.csv")

# a cluster holding domains from both studies cannot be used to test transfer: the
# model would have seen a close relative of the held-out domain during training
papers_in = clusters.groupby(cluster_column)['papers'].apply(
    lambda s: {p for entry in s for p in str(entry).split(',')})
shared = {c for c, p in papers_in.items() if len(p) > 1}
cluster_of_wt = dict(zip(clusters['wt_sequence'], clusters[cluster_column]))

alignment = read_alignment(f"{results_dir}/msa/pooled_{family}.fasta")
width = len(next(iter(alignment.values())))
print(f"models: {', '.join(models)}")
print(f"{family}: {len(clusters)} domains, {clusters[cluster_column].nunique()} clusters "
      f"at {threshold}%, {len(shared)} shared across studies")

# project every variant onto the pooled alignment once, so both studies share columns
blocks, placed_parts = [], []
for dataset in sorted(variants['dataset'].unique()):
    rows_here = variants[variants['dataset'] == dataset]
    wt = wt_of.get(rows_here['construct'].iloc[0])
    if wt is None or wt not in alignment:
        blocks.append(np.full((len(rows_here), width), '-', dtype='<U1'))
        placed_parts.append(np.zeros(len(rows_here), dtype=bool))
        continue
    block, placed_here = to_alignment_columns(
        rows_here['mutated_sequence'].astype(str).tolist(), wt, alignment[wt])
    blocks.append(block)
    placed_parts.append(placed_here)
chars = np.concatenate(blocks)
placed = np.concatenate(placed_parts)

datasets_present = sorted(variants['dataset'].unique())
embeddings, stale = {}, 0
counts = variants.groupby('dataset').size()
for dataset in datasets_present:
    path = f"{embedding_dir}/{dataset}_{cache_tag}.npy"
    if not os.path.exists(path):
        continue
    array = np.load(path)
    if array.shape[0] != counts[dataset]:
        stale += 1
        continue
    embeddings[dataset] = array
have_all = all(d in embeddings for d in datasets_present)
if not have_all:
    print(f"{esm_name} cells skipped: {len(datasets_present) - len(embeddings)} datasets "
          f"without a usable embedding ({stale} rejected on row count)")

variants = variants[placed].copy()
chars = chars[placed]
matrix = (np.concatenate([embeddings[d] for d in datasets_present])[placed]
          if have_all else None)
print(f"{len(variants):,} variants encoded over {chars.shape[1]} alignment columns")

variants['cluster'] = variants['construct'].map(wt_of).map(cluster_of_wt)

rows = []
for train_paper, test_paper in directions:
    train_mask = (variants['paper'] == train_paper).values
    # drop from test rather than from train: a held-out domain with a relative in the
    # other study is unusable, but every training domain is still worth keeping
    test_mask = ((variants['paper'] == test_paper) &
                 (~variants['cluster'].isin(shared))).values
    n_test_datasets = variants[test_mask]['dataset'].nunique()
    # rocklin splits a construct into _substitutions and _indels, so the unit here is
    # the dataset rather than the domain and the two counts differ
    print(f"\n{train_paper} -> {test_paper}: train {int(train_mask.sum()):,} variants "
          f"over {variants[train_mask]['dataset'].nunique()} datasets, "
          f"test {int(test_mask.sum()):,} over {n_test_datasets} datasets")
    if test_mask.sum() == 0 or train_mask.sum() == 0:
        print("  nothing to do")
        continue

    for encoding in encodings:
        if encoding != 'onehot' and matrix is None:
            continue
        if encoding == 'onehot':
            # the alphabet is fixed rather than learned from the training study. fitted
            # on train alone, a residue that only appears in the test study is unknown,
            # and with drop="first" an unknown row is all zeros, which is exactly how
            # the dropped reference residue is encoded. the two would be indistinguishable
            encoder = onehot(sparse_output = False, drop = "first",
                             categories = [list(residue_alphabet)] * chars.shape[1])
            X_train = encoder.fit_transform(chars[train_mask])
            X_test = encoder.transform(chars[test_mask])
        else:
            X_train, X_test = matrix[train_mask], matrix[test_mask]

        for target in targets:
            column = f"score_{target}"
            Y_train = variants.loc[train_mask, column].values
            Y_test = variants.loc[test_mask, column].values
            keep_train, keep_test = np.isfinite(Y_train), np.isfinite(Y_test)
            for model_name in models:
                model = build_model(model_name)
                model.fit(X_train[keep_train], Y_train[keep_train])
                prediction = model.predict(X_test[keep_test])

                scored = variants[test_mask][keep_test].assign(
                    truth = Y_test[keep_test], prediction = prediction)
                per_domain = []
                for dataset, group in scored.groupby('dataset'):
                    if len(group) > min_test_variants and group['truth'].std() > 0:
                        per_domain.append({
                            'train_paper': train_paper, 'test_paper': test_paper,
                            'encoding': encoding, 'model': model_name, 'target': target,
                            'dataset': dataset, 'n_variants': len(group),
                            'spearman': spearmanr(group['truth'], group['prediction'])[0],
                        })
                rows.extend(per_domain)
                values = [r['spearman'] for r in per_domain]
                print(f"  {encoding:10s} {model_name:5s} {target:6s} "
                      f"median rho {np.median(values):6.3f} over {len(values)} datasets"
                      if values else
                      f"  {encoding:10s} {model_name:5s} {target:6s} no scorable dataset")

report = pd.DataFrame(rows)
if report.empty:
    raise SystemExit("nothing scored")
path = f"{results_dir}/cross_study_{family}.csv"
report.to_csv(path, index = False)

print("\n" + "=" * 90)
print("CAN ONE STUDY'S DOMAINS PREDICT THE OTHER'S?")
print("=" * 90)
summary = (report.groupby(['train_paper', 'test_paper', 'encoding', 'model', 'target'])
           ['spearman'].agg(['median', 'std', 'count']).round(3))
print(summary.to_string())
print(f"\nsaved {path}")
