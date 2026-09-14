import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
results_dir = f"{project_dir}/results"
output_dir = f"{project_dir}/figures"
min_wt_seqs = 5
identity_low = 0.60
identity_high = 0.70

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


# ---------------------------------------------------------------- figure 1
# why at least five domains per family: what survives each candidate cutoff
survival = pd.read_csv(f"{results_dir}/pfam_survival.csv")

fig, axes = plt.subplots(1, 2, figsize = (11, 4.4))

ax = axes[0]
recede(ax)
ax.axvline(min_wt_seqs, color = muted, linestyle = ':', linewidth = 1.2, zorder = 1)
ax.plot(survival['min_wt_seqs'], survival['n_families'], marker = 'o', markersize = 6,
        linewidth = 2, color = blue, label = 'families retained', zorder = 3)
ax.plot(survival['min_wt_seqs'], survival['families_in_both'], marker = 's', markersize = 6,
        linewidth = 2, color = vermillion, label = 'retained and present in both studies', zorder = 3)
chosen = survival[survival['min_wt_seqs'] == min_wt_seqs].iloc[0]
ax.annotate(f"{int(chosen['n_families'])} families", xy = (min_wt_seqs, chosen['n_families']),
            xytext = (8, 10), textcoords = 'offset points', color = blue, fontsize = 10)
ax.text(0.04, 0.05, f"{int(chosen['families_in_both'])} of them present in both studies",
        transform = ax.transAxes, color = vermillion, fontsize = 10)
ax.set_xlim(0, 31)
ax.set_xlabel('minimum distinct wild-type domains per family')
ax.set_ylabel('number of Pfam families')
ax.set_title('Families retained at each cutoff', loc = 'left')
ax.legend(loc = 'upper right')

ax = axes[1]
recede(ax)
ax.axvline(min_wt_seqs, color = muted, linestyle = ':', linewidth = 1.2, zorder = 1)
ax.plot(survival['min_wt_seqs'], survival['pct_variants'], marker = 'o', markersize = 6,
        linewidth = 2, color = blue, zorder = 3)
ax.annotate(f"{chosen['pct_variants']:.0f}% of variants kept\nat a cutoff of {min_wt_seqs}",
            xy = (min_wt_seqs, chosen['pct_variants']), xytext = (14, 6),
            textcoords = 'offset points', fontsize = 10, color = ink)
ax.set_ylim(0, 100)
ax.set_xlim(0, 31)
ax.set_xlabel('minimum distinct wild-type domains per family')
ax.set_ylabel('percent of all variants retained')
ax.set_title('Data retained at each cutoff', loc = 'left')

fig.suptitle("Choosing a family-size threshold: 5 domains keeps 32 families and 65% of the data",
             fontsize = 13, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.94])
path = f"{output_dir}/family_size_threshold.png"
plt.savefig(path)
plt.close()
print(f"saved {path}")

# ---------------------------------------------------------------- figure 2
# why 60-70 percent identity: usable families against the two failure modes
report = pd.read_csv(f"{results_dir}/cluster_report.csv")
report['usable'] = (report['n_clusters'] >= 5) & (report['pct_in_largest'] <= 50)
report['fragmented'] = report['pct_in_largest'] < 15
report['dominated'] = report['pct_in_largest'] > 50
by_threshold = report.groupby('threshold')[['usable', 'fragmented', 'dominated']].sum().reset_index()
n_families = report['pfam_acc'].nunique()

fig, ax = plt.subplots(figsize = (8, 5))
recede(ax)
ax.axvspan(identity_low * 100, identity_high * 100, color = "#eef2f6", zorder = 0)
ax.plot(by_threshold['threshold'] * 100, by_threshold['usable'], marker = 'o', markersize = 7,
        linewidth = 2.2, color = blue, label = 'usable for splitting', zorder = 3)
ax.plot(by_threshold['threshold'] * 100, by_threshold['dominated'], marker = 's', markersize = 7,
        linewidth = 2, color = vermillion, label = 'collapsed into one dominant cluster', zorder = 3)
ax.plot(by_threshold['threshold'] * 100, by_threshold['fragmented'], marker = '^', markersize = 7,
        linewidth = 2, color = green, label = 'shattered into singletons', zorder = 3)
ax.text((identity_low + identity_high) / 2 * 100, n_families * 0.99, 'operating range',
        ha = 'center', va = 'top', fontsize = 10, color = muted)
end = by_threshold.iloc[-1]
for value, colour in [(end['usable'], blue), (end['dominated'], vermillion), (end['fragmented'], green)]:
    ax.annotate(f"{int(value)}", xy = (end['threshold'] * 100, value), xytext = (10, -4),
                textcoords = 'offset points', color = colour, fontsize = 11, fontweight = 'bold')
ax.set_xlim(27, 96)
ax.set_xlabel('sequence identity threshold (percent)')
ax.set_ylabel(f'number of families (of {n_families})')
ax.set_ylim(0, n_families)
ax.set_title("Choosing a clustering threshold: 60-70% identity keeps most families splittable\n"
             "usable = at least 5 clusters with no cluster holding over half the domains",
             loc = 'left', fontsize = 12)
ax.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.14), ncol = 3, fontsize = 10)
plt.tight_layout()
path = f"{output_dir}/identity_threshold.png"
plt.savefig(path)
plt.close()
print(f"saved {path}")

# ---------------------------------------------------------------- figure 3
# domain length distributions, overall and for the sh3 pilot family
constructs = pd.read_csv(f"{results_dir}/constructs.csv")
lehner = constructs[constructs['paper'] == 'lehner']['wt_length']
rocklin = constructs[constructs['paper'] == 'rocklin']['wt_length']
sh3 = constructs[constructs['pfam_acc'] == 'PF00018']['wt_length']

fig, axes = plt.subplots(1, 2, figsize = (11, 4.4))

ax = axes[0]
recede(ax)
bins = np.arange(15, 105, 5)
ax.hist(lehner, bins = bins, histtype = 'step', linewidth = 2.2, color = blue,
        label = f'Lehner 2025  (n={len(lehner)})', zorder = 3)
ax.hist(rocklin, bins = bins, histtype = 'step', linewidth = 2.2, linestyle = '--', color = vermillion,
        label = f'Rocklin 2023  (n={len(rocklin)})', zorder = 3)
ax.axvline(72, color = muted, linestyle = ':', linewidth = 1.2, zorder = 1)
ax.annotate('72 aa assay cap', xy = (72, 70), xytext = (4, 0), textcoords = 'offset points',
            fontsize = 9, color = muted)
ax.set_xlabel('wild-type domain length (amino acids)')
ax.set_ylabel('number of constructs')
ax.set_title('All domains, by study', loc = 'left')
ax.legend(loc = 'upper left', fontsize = 10)

ax = axes[1]
recede(ax)
ax.hist(sh3, bins = np.arange(54.5, 68.5, 1), color = blue, zorder = 3)
ax.set_xticks(np.arange(55, 68, 2))
ax.set_xlabel('wild-type domain length (amino acids)')
ax.set_ylabel('number of constructs')
ax.set_title(f'SH3 domains (PF00018), n={len(sh3)} constructs', loc = 'left')
ax.annotate(f"range {int(sh3.min())}-{int(sh3.max())} aa, median {sh3.median():.0f}",
            xy = (0.97, 0.93), xycoords = 'axes fraction', ha = 'right', fontsize = 10, color = muted)

fig.suptitle("Domain length distributions: Rocklin is capped near 72 aa by assay design",
             fontsize = 13, x = 0.02, ha = 'left')
plt.tight_layout(rect = [0, 0, 1, 0.94])
path = f"{output_dir}/domain_lengths.png"
plt.savefig(path)
plt.close()
print(f"saved {path}")

print()
print(f"figures written to {output_dir}")
