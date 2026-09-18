import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
# outputs go to the repo by default; on O2 set SPF_OUTPUT to the shared project
# folder so results are computed once and pulled down rather than recomputed
output_root = os.environ.get("SPF_OUTPUT", project_dir)
data_root = os.environ.get("SPF_DATA", f"{project_dir}/data")
results_dir = f"{output_root}/results"
output_dir = f"{output_root}/figures"
min_wt_seqs = 5
identity_low = 0.60
identity_high = 0.70
studies = [('lehner', 'Lehner 2025  (human)'), ('rocklin', 'Rocklin 2023  (across nature)')]
thresholds = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 26, 30]

os.makedirs(output_dir, exist_ok = True)

# okabe-ito subset, validated colourblind-safe against a light surface
blue = "#0072B2"
vermillion = "#D55E00"
green = "#009E73"
ink = "#1a1a1a"
muted = "#6b6b6b"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": muted,
    "axes.labelcolor": ink,
    "text.color": ink,
    "xtick.color": muted,
    "ytick.color": muted,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "grid.color": "#e4e4e2",
    "grid.linewidth": 0.8,
})


def recede(ax):
    ax.grid(True, axis = 'y', zorder = 0)
    ax.set_axisbelow(True)


constructs = pd.read_csv(f"{output_root}/results/constructs.csv")
annotated = constructs.dropna(subset = ['pfam_acc'])

# one row per distinct wild-type domain per study, so the two studies are never pooled
domains = (annotated.groupby(['pfam_acc', 'wt_sequence'])['paper']
           .apply(lambda s: set(s)).reset_index(name = 'papers'))

# ---------------------------------------------------------------- figure 1
# family depth within each study: how many families survive each cutoff
fig, axes = plt.subplots(1, 2, figsize = (11.5, 4.6), sharey = True)

for ax, (study, label) in zip(axes, studies):
    recede(ax)
    present = domains[domains['papers'].apply(lambda p: study in p)]
    depth = present.groupby('pfam_acc').size()
    kept = [int((depth >= t).sum()) for t in thresholds]
    variants = []
    for t in thresholds:
        families = depth[depth >= t].index
        subset = annotated[(annotated['paper'] == study) & (annotated['pfam_acc'].isin(families))]
        total = annotated[annotated['paper'] == study]['n_variants'].sum()
        variants.append(100 * subset['n_variants'].sum() / total)

    ax.axvline(min_wt_seqs, color = muted, linestyle = ':', linewidth = 1.2, zorder = 1)
    ax.plot(thresholds, kept, marker = 'o', markersize = 6, linewidth = 2,
            color = blue, label = 'families retained', zorder = 3)
    at_cutoff = kept[thresholds.index(min_wt_seqs)]
    pct_at_cutoff = variants[thresholds.index(min_wt_seqs)]
    ax.annotate(f"{at_cutoff} families\n{pct_at_cutoff:.0f}% of this study's variants",
                xy = (min_wt_seqs, at_cutoff), xytext = (14, 14),
                textcoords = 'offset points', color = blue, fontsize = 10)
    ax.set_xlim(0, 31)
    ax.set_xlabel('minimum distinct wild-type domains per family')
    ax.set_title(label, loc = 'left')

axes[0].set_ylabel('number of Pfam families')
axes[0].legend(loc = 'upper right')
fig.suptitle("Family depth within each study: Lehner carries 25 families at a cutoff of 5, Rocklin only 9",
             fontsize = 13, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.93])
plt.savefig(f"{output_dir}/family_size_threshold.png")
plt.close()
print("saved family_size_threshold.png")

# ---------------------------------------------------------------- figure 2
# clustering behaviour within each study
report = pd.read_csv(f"{results_dir}/cluster_report.csv")
report['usable'] = (report['n_clusters'] >= 5) & (report['pct_in_largest'] <= 50)
report['fragmented'] = report['pct_in_largest'] < 15
report['dominated'] = report['pct_in_largest'] > 50

fig, axes = plt.subplots(1, 2, figsize = (11.5, 5), sharey = False)

for ax, (study, label) in zip(axes, studies):
    recede(ax)
    scoped = report[report['scope'] == study]
    n_families = scoped['pfam_acc'].nunique()
    by_threshold = scoped.groupby('threshold')[['usable', 'fragmented', 'dominated']].sum().reset_index()

    ax.axvspan(identity_low * 100, identity_high * 100, color = "#eef2f6", zorder = 0)
    ax.plot(by_threshold['threshold'] * 100, by_threshold['usable'], marker = 'o', markersize = 7,
            linewidth = 2.2, color = blue, label = 'usable for splitting', zorder = 3)
    ax.plot(by_threshold['threshold'] * 100, by_threshold['dominated'], marker = 's', markersize = 7,
            linewidth = 2, color = vermillion, label = 'one dominant cluster', zorder = 3)
    ax.plot(by_threshold['threshold'] * 100, by_threshold['fragmented'], marker = '^', markersize = 7,
            linewidth = 2, color = green, label = 'shattered into singletons', zorder = 3)
    ax.set_xlim(27, 94)
    ax.set_ylim(0, n_families)
    ax.set_xlabel('sequence identity threshold (percent)')
    ax.set_ylabel(f'number of families (of {n_families})')
    ax.set_title(f"{label} — {n_families} families", loc = 'left')

axes[0].text((identity_low + identity_high) / 2 * 100, 18, 'operating\nrange',
             ha = 'center', va = 'center', fontsize = 9, color = muted)
axes[0].legend(loc = 'center left', fontsize = 10)
fig.suptitle("Clustering behaviour by study: the 60-70% range holds in Lehner; Rocklin is too thin to discriminate",
             fontsize = 13, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.93])
plt.savefig(f"{output_dir}/identity_threshold.png")
plt.close()
print("saved identity_threshold.png")

# ---------------------------------------------------------------- figure 3
# domain lengths, one panel per study
fig, axes = plt.subplots(1, 2, figsize = (11.5, 4.6), sharex = True)
bins = np.arange(15, 105, 5)

for ax, (study, label) in zip(axes, studies):
    recede(ax)
    lengths = constructs[constructs['paper'] == study]['wt_length']
    ax.hist(lengths, bins = bins, color = blue, zorder = 3)
    if study == 'rocklin':
        ax.axvline(72, color = vermillion, linestyle = '--', linewidth = 1.6, zorder = 4)
        ax.annotate('72 aa assay cap', xy = (72, ax.get_ylim()[1] * 0.9), xytext = (6, 0),
                    textcoords = 'offset points', fontsize = 10, color = vermillion)
    ax.set_xlabel('wild-type domain length (amino acids)')
    ax.set_title(f"{label} — n={len(lengths)}, range {int(lengths.min())}-{int(lengths.max())} aa",
                 loc = 'left', fontsize = 11)

axes[0].set_ylabel('number of constructs')
fig.suptitle("Domain lengths by study: Rocklin is capped near 72 aa by assay design, Lehner is not",
             fontsize = 13, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.93])
plt.savefig(f"{output_dir}/domain_lengths.png")
plt.close()
print("saved domain_lengths.png")

# ---------------------------------------------------------------- figure 4
# the sh3 pilot family, again one panel per study
fig, axes = plt.subplots(1, 2, figsize = (11.5, 4.3), sharex = True, sharey = True)
sh3_bins = np.arange(54.5, 68.5, 1)

for ax, (study, label) in zip(axes, studies):
    recede(ax)
    lengths = constructs[(constructs['pfam_acc'] == 'PF00018') & (constructs['paper'] == study)]['wt_length']
    ax.hist(lengths, bins = sh3_bins, color = blue, zorder = 3)
    ax.set_xticks(np.arange(55, 68, 2))
    ax.set_xlabel('wild-type domain length (amino acids)')
    ax.set_title(f"{label} — n={len(lengths)}, median {lengths.median():.0f} aa", loc = 'left', fontsize = 11)

axes[0].set_ylabel('number of constructs')
fig.suptitle("SH3 domains (PF00018), the pilot family: comparable length distributions in both studies",
             fontsize = 13, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.92])
plt.savefig(f"{output_dir}/sh3_lengths.png")
plt.close()
print("saved sh3_lengths.png")

print()
print(f"figures written to {output_dir}")
