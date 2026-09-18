import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
# outputs go to the repo by default; on O2 set SPF_OUTPUT to the shared project
# folder so results are computed once and pulled down rather than recomputed
output_root = os.environ.get("SPF_OUTPUT", project_dir)
data_root = os.environ.get("SPF_DATA", f"{project_dir}/data")
constructs_path = f"{output_root}/results/constructs.csv"
output_dir = f"{output_root}/results"
thresholds = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 26, 30]

os.makedirs(output_dir, exist_ok = True)

constructs = pd.read_csv(constructs_path)
annotated = constructs.dropna(subset = ['pfam_acc']).copy()

print("=" * 70)
print("PFAM COVERAGE")
print("=" * 70)
print(f"constructs total            {len(constructs):>6,}")
print(f"constructs with a pfam acc  {len(annotated):>6,}")
print(f"constructs without one      {len(constructs) - len(annotated):>6,}")
print(f"distinct pfam families      {annotated['pfam_acc'].nunique():>6,}")
print()
print("unannotated constructs by paper:")
print(constructs[constructs['pfam_acc'].isna()].groupby('paper').size().to_string())

summary = annotated.groupby('pfam_acc').agg(
    n_constructs = ('construct', 'nunique'),
    n_wt_seq = ('wt_sequence', 'nunique'),
    n_accession = ('accession', 'nunique'),
    n_variants = ('n_variants', 'sum'),
    wt_length_min = ('wt_length', 'min'),
    wt_length_max = ('wt_length', 'max'),
)

# a family is only usable for cross-study work if both papers contribute domains to it
per_paper = annotated.groupby(['pfam_acc', 'paper'])['wt_sequence'].nunique().unstack(fill_value = 0)
for paper in ['lehner', 'rocklin']:
    summary[f"wt_{paper}"] = per_paper[paper] if paper in per_paper.columns else 0
summary['in_both'] = (summary['wt_lehner'] > 0) & (summary['wt_rocklin'] > 0)
summary = summary.sort_values('n_wt_seq', ascending = False)

print()
print("=" * 70)
print("TOP 25 FAMILIES BY DISTINCT WILD-TYPE DOMAIN SEQUENCES")
print("=" * 70)
print(summary.head(25).to_string())

print()
print("families present in both papers:", int(summary['in_both'].sum()))
print("families only in lehner:        ", int(((summary['wt_lehner'] > 0) & (summary['wt_rocklin'] == 0)).sum()))
print("families only in rocklin:       ", int(((summary['wt_rocklin'] > 0) & (summary['wt_lehner'] == 0)).sum()))

print()
print("=" * 70)
print("SURVIVAL AT CANDIDATE FAMILY-SIZE THRESHOLDS")
print("=" * 70)
rows = []
total_variants = summary['n_variants'].sum()
for t in thresholds:
    kept = summary[summary['n_wt_seq'] >= t]
    rows.append({
        'min_wt_seqs': t,
        'n_families': len(kept),
        'n_wt_seqs': int(kept['n_wt_seq'].sum()),
        'n_variants': int(kept['n_variants'].sum()),
        'pct_variants': round(100 * kept['n_variants'].sum() / total_variants, 1),
        'families_in_both': int(kept['in_both'].sum()),
    })
survival = pd.DataFrame(rows)
print(survival.to_string(index = False))

fig, axes = plt.subplots(1, 2, figsize = (12, 4.5))
sns.histplot(summary['n_wt_seq'], bins = 40, color = 'royalblue', edgecolor = 'black', ax = axes[0])
axes[0].set_yscale('log')
axes[0].set_xlabel('distinct wild-type domains in family')
axes[0].set_ylabel('number of families (log)')
axes[0].set_title(f"Family depth\n{len(summary)} pfam families, {len(annotated)} annotated constructs", fontsize = 10)

axes[1].plot(survival['min_wt_seqs'], survival['n_families'], marker = 'o', label = 'families kept')
axes[1].plot(survival['min_wt_seqs'], survival['families_in_both'], marker = 's', label = 'families in both papers')
axes[1].set_xlabel('minimum distinct wild-type domains per family')
axes[1].set_ylabel('number of families')
axes[1].set_title("Families surviving a size threshold", fontsize = 10)
axes[1].legend()

plt.tight_layout()
figure_path = f"{output_dir}/pfam_family_depth.png"
plt.savefig(figure_path, dpi = 150)
print()
print(f"Figure saved to {figure_path}")

summary_path = f"{output_dir}/pfam_summary.csv"
summary.to_csv(summary_path)
print(f"Family summary saved to {summary_path}")

survival_path = f"{output_dir}/pfam_survival.csv"
survival.to_csv(survival_path, index = False)
print(f"Survival table saved to {survival_path}")
