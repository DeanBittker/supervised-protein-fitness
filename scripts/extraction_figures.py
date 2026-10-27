import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Three figures for the Pfam extraction, each answering one question a reader will
# ask about it: was the coverage cutoff a real choice, are the folds comparable to
# each other, and what does clan grouping cost the encoder.
#
#   SPF_OUTPUT=<shared folder> python scripts/extraction_figures.py

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
extract_dir = f"{output_root}/results/extraction"
figure_dir = f"{output_root}/figures"
clans_path = os.environ.get("SPF_CLANS", f"{output_root}/reference/pfam_clans.tsv")
# the same floor pfam_extract.py applies, so the curve below lands on the same
# numbers the extraction reports rather than a coverage-only count that disagrees
min_residues = int(os.environ.get("SPF_MIN_RESIDUES", 30))

blue = "#0072B2"
vermillion = "#D55E00"
ink = "#1a1a1a"
muted = "#6b6b6b"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.edgecolor": muted, "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": muted, "ytick.color": muted,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "grid.color": "#e4e4e2", "grid.linewidth": 0.8,
})

os.makedirs(figure_dir, exist_ok = True)
for name in ('coverage', 'folds'):
    if not os.path.exists(f"{extract_dir}/{name}.csv"):
        raise SystemExit(f"{extract_dir}/{name}.csv not found, run pfam_extract.py "
                         f"and make_clan_folds.py first")
coverage = pd.read_csv(f"{extract_dir}/coverage.csv")
folds = pd.read_csv(f"{extract_dir}/folds.csv")

# clan names make the fold labels readable; without the table the accessions stand
names = {}
if os.path.exists(clans_path):
    table = pd.read_csv(clans_path, sep = '\t', header = None, usecols = [1, 2],
                        names = ['clan', 'name'], dtype = str).dropna()
    names = dict(zip(table['clan'], table['name']))
folds['label'] = [f"{g}  {names.get(g, '')}".strip() for g in folds['group']]
written = []


def save(fig, name):
    path = f"{figure_dir}/{name}.png"
    fig.savefig(path, bbox_inches = 'tight')
    plt.close(fig)
    written.append(path)


# --- 1. was the cutoff a real choice -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize = (11.5, 4.3))

ax = axes[0]
ax.hist(coverage['coverage'].clip(0, 1), bins = 40, color = blue, edgecolor = 'white',
        linewidth = 0.5)
ax.set_yscale('log')
ax.set_xlabel('fraction of the Pfam region the construct covers')
ax.set_ylabel('domain instances (log)')
ax.set_title('Coverage is bimodal, not graded', loc = 'left')
ax.axvline(0.5, color = vermillion, linewidth = 2, zorder = 3)
ax.annotate('50% cutoff', xy = (0.5, ax.get_ylim()[1] * 0.4), xytext = (8, 0),
            textcoords = 'offset points', color = vermillion, fontsize = 10)
ax.grid(axis = 'y', alpha = 0.7)
ax.set_axisbelow(True)

ax = axes[1]
cuts = np.arange(0.05, 1.001, 0.05)
long_enough = coverage['covered'] >= min_residues
kept = [int(((coverage['coverage'] >= c) & long_enough).sum()) for c in cuts]
ax.plot(cuts * 100, kept, color = blue, linewidth = 2, marker = 'o', markersize = 5,
        clip_on = False)
ax.axvline(50, color = vermillion, linewidth = 2, zorder = 3)
at_fifty = int(((coverage['coverage'] >= 0.5) & long_enough).sum())
at_full = int(((coverage['coverage'] >= 1.0) & long_enough).sum())
ax.annotate(f"{at_fifty:,} kept", xy = (50, at_fifty), xytext = (8, 12),
            textcoords = 'offset points', color = vermillion, fontsize = 10)
ax.annotate(f"{at_full:,}", xy = (100, at_full), xytext = (-4, 12),
            textcoords = 'offset points', color = ink, fontsize = 10, ha = 'right')
ax.set_xlabel('coverage cutoff (%)')
ax.set_ylabel(f'instances surviving (and {min_residues}+ residues)')
ax.set_title('The cutoff is nearly free until it reaches 100%', loc = 'left')
ax.grid(axis = 'y', alpha = 0.7)
ax.set_axisbelow(True)
save(fig, 'extraction_cutoff')

# --- 2. fold sizes, and where one-hot can still run ----------------------------------
# one-hot projects variants onto one family's alignment columns, so a fold spanning
# several families has no shared column space and only embeddings can run on it
folds = folds.sort_values('instances')
single = folds['families'] == 1
fig, ax = plt.subplots(figsize = (8.6, 9))
positions = np.arange(len(folds))
ax.barh(positions, folds['instances'],
        color = np.where(single, blue, vermillion), height = 0.72)
for y, (count, families, is_single) in enumerate(
        zip(folds['instances'], folds['families'], single)):
    note = '1 family' if is_single else f"{families} families"
    ax.annotate(f"{count}  ·  {note}", xy = (count, y), xytext = (5, 0),
                textcoords = 'offset points', va = 'center', fontsize = 9,
                color = blue if is_single else vermillion)
ax.set_yticks(positions)
ax.set_yticklabels(folds['label'], fontsize = 9)
ax.set_xlabel('domain instances held out by the fold')
ax.set_xlim(0, folds['instances'].max() * 1.32)
ax.set_title(f"Leave-one-clan-out: {len(folds)} folds, "
             f"{int(folds['instances'].max() / folds['instances'].min())}x "
             f"size spread\n"
             f"one-hot over alignment columns can only run on the "
             f"{int(single.sum())} single-family folds", loc = 'left')
handles = [plt.Rectangle((0, 0), 1, 1, color = blue),
           plt.Rectangle((0, 0), 1, 1, color = vermillion)]
ax.legend(handles, ['one family: one-hot and embeddings',
                    'several families: embeddings only'],
          loc = 'lower right', fontsize = 10)
ax.grid(axis = 'x', alpha = 0.7)
ax.set_axisbelow(True)
save(fig, 'extraction_folds')

# --- 3. the folds are not comparable to one another ----------------------------------
fig, ax = plt.subplots(figsize = (8.2, 5.6))
sizes = 18 + 7 * folds['instances']
ax.scatter(folds['median_residues'], folds['papers'], s = sizes, color = blue,
           alpha = 0.55, edgecolor = 'white', linewidth = 1.2, zorder = 3)
ax.set_xscale('log')
# plain residue counts: nobody reads a domain length as 3 x 10^1
ticks = [30, 50, 75, 100, 150, 250, 400, 600]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# headroom so the topmost labels clear the title
ax.set_ylim(0, folds['papers'].max() * 1.18)
ax.set_xlabel('median residues covered per instance (log scale)')
ax.set_ylabel('papers contributing to the fold')
ax.set_title('Fold difficulty is confounded with domain length and assay mix\n'
             'marker area is how much data the fold holds out', loc = 'left')
worst = pd.concat([folds.nlargest(3, 'papers'), folds.nlargest(2, 'median_residues'),
                   folds.nlargest(1, 'instances')]).drop_duplicates('group')
for _, row in worst.iterrows():
    ax.annotate(row['label'], xy = (row['median_residues'], row['papers']),
                xytext = (0, 16), textcoords = 'offset points', ha = 'center',
                fontsize = 9, color = ink)
ax.grid(alpha = 0.7)
ax.set_axisbelow(True)
save(fig, 'extraction_confounds')

print(f"{len(coverage):,} instances named, {int(coverage['kept'].sum()):,} kept")
print(f"{len(folds)} folds, {int((folds['families'] == 1).sum())} of them single-family "
      f"holding {int(folds.loc[folds['families'] == 1, 'instances'].sum())} instances "
      f"of {int(folds['instances'].sum())} in folds")
for path in written:
    print(f"wrote {path}")
