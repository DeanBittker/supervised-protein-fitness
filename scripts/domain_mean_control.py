import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import OneHotEncoder as onehot
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
figure_dir = f"{output_root}/figures"
embedding_dir = f"{output_root}/embeddings"
variants_path = f"{results_dir}/variants.parquet"
family = os.environ.get("SPF_FAMILY", "PF00018")
threshold = float(os.environ.get("SPF_THRESHOLD", "0.6"))
n_replicates = int(os.environ.get("SPF_REPLICATES", "10"))
cache_tag = "esm150M"
papers = ['lehner', 'rocklin']
encodings = ['onehot', 'ESM2-150M']
target_column = 'score_raw'

os.makedirs(figure_dir, exist_ok = True)

# this control asks a single question: is a model predicting which domain a variant
# came from, or the effect of the variant itself? ridge alone, because the question
# is about the features rather than the learner, and it keeps the run to minutes.

slide = os.environ.get("SPF_SLIDE", "") == "1"
scale = 1.4 if slide else 1.0
blue, vermillion = "#0072B2", "#D55E00"
ink, muted = "#1a1a1a", "#6b6b6b"
colour = {'onehot': blue, 'ESM2-150M': vermillion}
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11 * scale,
    "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": muted, "ytick.color": muted,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "#e4e4e2", "grid.linewidth": 0.8,
})


def read_alignment(path):
    aligned, label, chunks = {}, None, []
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
    columns = [i for i, c in enumerate(aligned_wt) if c != '-']
    block = np.full((len(sequences), len(aligned_wt)), '-', dtype='<U1')
    placed = np.zeros(len(sequences), dtype=bool)
    for row, sequence in enumerate(sequences):
        if len(sequence) != len(wt_sequence):
            continue
        for position, column in enumerate(columns):
            block[row, column] = sequence[position]
        placed[row] = True
    return block, placed


variants = pd.read_parquet(variants_path)
variants = variants[variants['pfam_acc'] == family]
variants = variants[~variants['has_insertion']].sort_values(['dataset']).reset_index(drop = True)
grouped = variants.groupby('dataset')['DMS_score']
variants['score_raw'] = variants['DMS_score']
variants['domain_mean'] = grouped.transform('mean')

constructs = pd.read_csv(f"{results_dir}/constructs.csv")
splits = pd.read_csv(f"{results_dir}/splits_{family}.csv")
splits = splits[np.isclose(splits['threshold'], threshold)]

rows = []
for paper in papers:
    paper_rows = variants[variants['paper'] == paper]
    labels = splits[splits['scope'] == paper]
    if paper_rows.empty or labels.empty:
        continue
    datasets_present = sorted(paper_rows['dataset'].unique())
    alignment = read_alignment(f"{results_dir}/msa/{paper}_{family}.fasta")
    wt_of = dict(zip(constructs['construct'], constructs['wt_sequence']))

    blocks, placed_parts = [], []
    for dataset in datasets_present:
        here = paper_rows[paper_rows['dataset'] == dataset]
        wt = wt_of.get(here['construct'].iloc[0])
        if wt is None or wt not in alignment:
            blocks.append(np.full((len(here), len(next(iter(alignment.values())))), '-', dtype='<U1'))
            placed_parts.append(np.zeros(len(here), dtype=bool))
            continue
        block, placed_here = to_alignment_columns(
            here['mutated_sequence'].astype(str).tolist(), wt, alignment[wt])
        blocks.append(block)
        placed_parts.append(placed_here)
    chars = np.concatenate(blocks)
    placed = np.concatenate(placed_parts)
    matrix = np.concatenate([np.load(f"{embedding_dir}/{d}_{cache_tag}.npy") for d in datasets_present])

    paper_rows = paper_rows[placed].copy()
    chars, matrix = chars[placed], matrix[placed]
    print(f"{paper}: {len(paper_rows):,} variants across {paper_rows['dataset'].nunique()} datasets")

    for replicate in range(n_replicates):
        assignment = labels[labels['replicate'] == replicate]
        split_of = {wt_of[row['label']]: row['split']
                    for _, row in assignment.iterrows() if row['label'] in wt_of}
        membership = paper_rows['construct'].map(wt_of).map(split_of)
        train_mask = (membership == 'train').values
        test_mask = (membership == 'test').values
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        for encoding in encodings:
            if encoding == 'onehot':
                encoder = onehot(sparse_output = False, drop = "first", handle_unknown = "ignore")
                X_train = encoder.fit_transform(chars[train_mask])
                X_test = encoder.transform(chars[test_mask])
            else:
                X_train, X_test = matrix[train_mask], matrix[test_mask]

            Y_train = paper_rows.loc[train_mask, target_column].values
            test_frame = paper_rows[test_mask]
            model = Ridge(alpha = 1.0)
            model.fit(X_train, Y_train)
            prediction = model.predict(X_test)

            # three readings of the same prediction:
            #   overall   - correlation over every held-out variant, what we reported before
            #   within    - correlation inside each held-out domain, averaged; the part that
            #               is genuinely about the variant
            #   to_mean   - correlation with the domain's mean score, the part that is
            #               only about which domain the variant belongs to
            overall = spearmanr(test_frame[target_column].values, prediction)[0]
            within = []
            for _, group in test_frame.assign(prediction = prediction).groupby('dataset'):
                if len(group) > 10 and group[target_column].std() > 0:
                    within.append(spearmanr(group[target_column], group['prediction'])[0])
            to_mean = spearmanr(test_frame['domain_mean'].values, prediction)[0]
            spread = np.var(pd.Series(prediction).groupby(test_frame['dataset'].values).transform('mean'))
            rows.append({
                'paper': paper, 'encoding': encoding, 'replicate': replicate,
                'overall_rho': round(float(overall), 4),
                'within_domain_rho': round(float(np.mean(within)), 4) if within else np.nan,
                'rho_with_domain_mean': round(float(to_mean), 4),
                'between_domain_variance_fraction': round(float(spread / np.var(prediction)), 4)
                    if np.var(prediction) > 0 else np.nan,
                'n_test_domains': int(test_frame['dataset'].nunique()),
            })
            print(f"  rep {replicate:2d} {encoding:10s} overall {overall:6.3f}  "
                  f"within {np.mean(within) if within else float('nan'):6.3f}  "
                  f"vs domain mean {to_mean:6.3f}", flush = True)

report = pd.DataFrame(rows)
print()
print("=" * 95)
print("IS THE MODEL PREDICTING THE DOMAIN OR THE VARIANT?")
print("=" * 95)
summary = report.groupby(['paper', 'encoding'])[
    ['overall_rho', 'within_domain_rho', 'rho_with_domain_mean',
     'between_domain_variance_fraction']].median().round(3)
print(summary.to_string())

fig, axes = plt.subplots(1, 2, figsize = (12 * scale, 5 * scale))
measures = ['within_domain_rho', 'between_domain_variance_fraction']
titles = ['Correlation within each held-out protein\n(genuine variant-level signal)',
          'Fraction of prediction variance that is\nbetween proteins rather than within']
rng = np.random.default_rng(67)
for ax, measure, title in zip(axes, measures, titles):
    ax.grid(True, axis = 'y', zorder = 0)
    ax.set_axisbelow(True)
    ax.axhline(0, color = ink, linewidth = 1.2, zorder = 2)
    for index, paper in enumerate(papers):
        for offset, encoding in zip([-0.18, 0.18], encodings):
            values = report[(report['paper'] == paper) &
                            (report['encoding'] == encoding)][measure].dropna().values
            if not len(values):
                continue
            x = index + offset
            ax.scatter(np.full(len(values), x) + rng.uniform(-0.04, 0.04, len(values)),
                       values, s = 34, color = colour[encoding], alpha = 0.6,
                       edgecolor = 'none', zorder = 3,
                       label = encoding if (index == 0 and measure == measures[0]) else None)
            ax.hlines(np.median(values), x - 0.1, x + 0.1,
                      color = colour[encoding], linewidth = 2.6, zorder = 4)
    ax.set_xticks(range(len(papers)))
    ax.set_xticklabels(['Lehner 2025', 'Rocklin 2023'])
    ax.set_title(title, loc = 'left', fontsize = 10.5 * scale)
    if measure == 'between_domain_variance_fraction':
        ax.set_ylim(0, 1)
        ax.set_ylabel('fraction of prediction variance')
axes[0].set_ylabel("Spearman $\\rho$")
axes[0].legend(loc = 'upper left', fontsize = 10 * scale)
fig.suptitle(f"{family}: what the prediction is actually tracking (Ridge, raw target)",
             fontsize = 12.5 * scale, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.93])
path = f"{figure_dir}/domain_mean_control_{family}.png"
plt.savefig(path)
plt.close()
print()
print(f"saved {path}")
report.to_csv(f"{results_dir}/domain_mean_control_{family}.csv", index = False)
