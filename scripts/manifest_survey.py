import os
import re
import numpy as np
import pandas as pd

# A read-only pass over the ProteinGym2 manifest, answering how much of it can carry a
# held-out-domain question before any modelling is committed to. Nothing here writes
# outside its own survey folder and nothing reads the per-dataset files, so it can be
# run against the shared manifest without touching it.
#
#   SPF_MANIFEST=<the annotated manifest csv> python scripts/manifest_survey.py

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
# its own folder, so a survey can never land on top of a results file
survey_dir = f"{output_root}/results/survey"
manifest_path = os.environ.get("SPF_MANIFEST", "")
if not manifest_path:
    raise SystemExit("set SPF_MANIFEST to the annotated manifest csv")
if not os.path.exists(manifest_path):
    raise SystemExit(f"{manifest_path} not found")

os.makedirs(survey_dir, exist_ok = True)

PFAM = 'pfam accession'
INDICES = 'pfam indices'
PAPER = 'last author last name'
PROTEIN = 'accession'

manifest = pd.read_csv(manifest_path, low_memory = False)
print(f"{len(manifest):,} rows, {manifest.shape[1]} columns\n")

required = [PFAM, INDICES, PAPER, PROTEIN, 'filename', 'is_primary_condition',
            'score_distribution_status', 'directionality', 'phenotype']
absent = [c for c in required if c not in manifest.columns]
if absent:
    raise SystemExit(f"manifest is missing {absent}")

written = []


def save(frame, name, index = True):
    path = f"{survey_dir}/{name}.csv"
    frame.to_csv(path, index = index)
    written.append(path)


# --- 1. what survives each filter ---------------------------------------------------
def pfam_set(series):
    """Every accession named anywhere in the column. Several Pfams in one row are
    separated by a colon, and the ranges within one Pfam by a comma."""
    return (series.dropna().astype(str).str.split(':').explode().str.strip())


primary = manifest['is_primary_condition'] == True
has_pfam = manifest[PFAM].notna()
scored = manifest['score_distribution_status'] == 'ok'
directed = manifest['directionality'].notna()

stages = [
    ('all rows', pd.Series(True, index = manifest.index)),
    ('primary condition only', primary),
    ('and annotated with a pfam', primary & has_pfam),
    ('and score distribution ok', primary & has_pfam & scored),
    ('and directionality known', primary & has_pfam & scored & directed),
]
funnel = []
for name, mask in stages:
    part = manifest[mask]
    funnel.append({'stage': name, 'datasets': len(part),
                   'unique_pfams': pfam_set(part[PFAM]).nunique(),
                   'proteins': part[PROTEIN].nunique(),
                   'papers': part[PAPER].nunique()})
funnel = pd.DataFrame(funnel).set_index('stage')
print("what survives each filter")
print(funnel.to_string())
save(funnel, 'funnel')
print("\n  a dataset is dropped by directionality only because nobody recorded the sign, "
      "so that stage is a curation gap rather than a property of the data")

# --- 2. the pfam column's grammar, and whether the filename agrees -------------------
annotated = manifest[has_pfam]
per_row = annotated[PFAM].astype(str).str.count(':') + 1
print(f"\npfams named per annotated row: "
      f"{ {int(k): int(v) for k, v in per_row.value_counts().sort_index().items()} }")
print(f"  rows naming more than one: {int((per_row > 1).sum()):,} of {len(annotated):,}")

from_filename = manifest['filename'].astype(str).str.extract(r'(PF\d{5})', expand = False)
comparable = manifest[from_filename.notna() & has_pfam].copy()
comparable['filename_pfam'] = from_filename[comparable.index]
comparable['column_pfams'] = (comparable[PFAM].astype(str).str.split(':')
                              .apply(lambda names: {n.strip() for n in names}))
disagree = comparable[[name not in names for name, names
                       in zip(comparable['filename_pfam'], comparable['column_pfams'])]]
print(f"\nfilenames carrying a pfam: {int(from_filename.notna().sum()):,}; "
      f"of those comparable, {len(disagree)} disagree with the column")
if not disagree.empty:
    print("  by paper: " + ", ".join(f"{k} {v}" for k, v in
                                     disagree[PAPER].value_counts().items()))
    save(disagree[['filename', PFAM, 'filename_pfam', PAPER]],
         'pfam_filename_disagreements', index = False)
    print("  the column is the later annotation where these differ, so it is the one to "
          "key on; the list is saved")

# --- 3. per family ------------------------------------------------------------------
usable = manifest[primary & has_pfam & scored].copy()
exploded = usable.assign(pfam = usable[PFAM].astype(str).str.split(':')).explode('pfam')
exploded['pfam'] = exploded['pfam'].str.strip()

families = exploded.groupby('pfam').agg(
    datasets = ('filename', 'nunique'), proteins = (PROTEIN, 'nunique'),
    papers = (PAPER, 'nunique'), phenotypes = ('phenotype', 'nunique'))
phenotype_counts = (exploded.pivot_table(index = 'pfam', columns = 'phenotype',
                                         values = 'filename', aggfunc = 'nunique',
                                         fill_value = 0))
families = families.join(phenotype_counts).sort_values('proteins', ascending = False)
save(families, 'families')

print("\nfamilies deep enough to hold a domain out "
      "(primary conditions, annotated, score distribution ok)")
for cut in (2, 3, 5, 10):
    print(f"  {int((families['proteins'] >= cut).sum()):4d} families with {cut:2d}+ "
          f"distinct proteins   "
          f"({int((families['datasets'] >= cut).sum()):4d} with {cut}+ datasets)")
print("\n  deepest ten:")
print(families.head(10).to_string())

# --- 4. per protein -----------------------------------------------------------------
proteins = manifest[primary].groupby(PROTEIN).agg(
    datasets = ('filename', 'nunique'), papers = (PAPER, 'nunique'),
    phenotypes = ('phenotype', 'nunique')).sort_values('datasets', ascending = False)
save(proteins, 'proteins')
print(f"\nat full protein level: {len(proteins):,} proteins, "
      f"{int((proteins['datasets'] >= 2).sum())} with 2+ datasets, "
      f"{int((proteins['datasets'] >= 3).sum())} with 3+, "
      f"{int((proteins['papers'] >= 2).sum())} measured by more than one paper")

# --- 5. substitution order ----------------------------------------------------------
order_columns = ['num_single_mutants', 'num_double_mutants', 'num_triple_mutants',
                 'num_higher_order', 'num_insertion_mutants', 'num_deletion_mutants']
present = [c for c in order_columns if c in manifest.columns]
if present:
    counts = manifest.loc[primary, present].apply(pd.to_numeric, errors = 'coerce')
    order = pd.DataFrame({'datasets_with_any': (counts > 0).sum(),
                          'total_variants': counts.fillna(0).sum().astype('int64')})
    save(order, 'substitution_order')
    print("\nsubstitution order, primary conditions")
    print(order.to_string())
    beyond = counts[['num_double_mutants', 'num_triple_mutants', 'num_higher_order']]
    multi = beyond.fillna(0).sum(axis = 1) > 0
    by_paper = manifest.loc[primary][multi.values][PAPER].value_counts()
    print(f"\n  {int(multi.sum())} primary datasets carry multi-substitution variants, "
          f"led by " + ", ".join(f"{k} {v}" for k, v in by_paper.head(5).items()))
    save(by_paper.rename('datasets_with_multi').to_frame(), 'multi_substitution_by_paper')

# --- 6. the unsupervised baselines already scored -----------------------------------
baseline_columns = [c for c in manifest.columns
                    if c.endswith('_spearman') and not c.endswith('_pval')]
if baseline_columns:
    per_dataset = manifest[baseline_columns].notna().sum(axis = 1)
    coverage = (manifest.assign(baselines = per_dataset).groupby(PAPER)
                .agg(datasets = ('filename', 'size'), mean_baselines = ('baselines', 'mean'))
                .sort_values('datasets', ascending = False))
    save(coverage, 'baseline_coverage')
    print(f"\n{len(baseline_columns)} zero-shot baselines scored in the manifest; "
          f"{int((per_dataset > 0).sum()):,} datasets carry at least one and "
          f"{int((per_dataset >= 20).sum()):,} carry twenty or more")
    print("  so the supervised numbers can be set against zero-shot on the same assay "
          "without running anything")

# --- 7. where directionality is missing ---------------------------------------------
missing = (manifest.assign(unknown = manifest['directionality'].isna()).groupby(PAPER)
           .agg(datasets = ('filename', 'size'), unknown = ('unknown', 'sum')))
missing['percent_unknown'] = (100 * missing['unknown'] / missing['datasets']).round(0)
missing = missing.sort_values('datasets', ascending = False)
save(missing, 'directionality_by_paper')
print("\ndirectionality unknown, by paper, for the ten largest")
print(missing.head(10).to_string())
print("  it clusters by paper rather than by dataset, so a paper's convention can be "
      "settled once instead of a dataset at a time")

print("\nwrote:")
for path in written:
    print(f"  {path}")
