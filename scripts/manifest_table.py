import os
import re
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
manifest_path = f"{project_dir}/data/260822_manifests_R1_plus_R2.csv"
output_dir = f"{project_dir}/results"
dois = ['10.1038/s41586-023-06328-6', '10.1038/s41586-024-08370-4']
paper_names = {'10.1038/s41586-023-06328-6': 'rocklin', '10.1038/s41586-024-08370-4': 'lehner'}

os.makedirs(output_dir, exist_ok = True)

months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
          'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}


def repair_range(value):
    # excel silently converted "1-76" into the date "Jan-76" (and "1-22" into "22-Jan"),
    # so the month abbreviation is really the start of the range and the number is the end
    if not isinstance(value, str):
        return (np.nan, np.nan)
    parts = value.strip().split('-')
    if len(parts) != 2:
        return (np.nan, np.nan)
    numbers = []
    for p in parts:
        key = p.strip().lower()[:3]
        if key in months:
            numbers.append(months[key])
        elif p.strip().isdigit():
            numbers.append(int(p.strip()))
        else:
            return (np.nan, np.nan)
    return (min(numbers), max(numbers))


def construct_id(filename):
    # rocklin splits one assayed construct across a _substitutions row and an _indels row
    return re.sub(r'_(substitutions|indels)$', '', str(filename))


manifest = pd.read_csv(manifest_path, low_memory = False)
data = manifest[manifest['DOI'].isin(dois)].copy()
data['paper'] = data['DOI'].map(paper_names)
data['construct'] = data['filename'].apply(construct_id)
data['n_variants'] = pd.to_numeric(data['number of variants'], errors = 'coerce')
data['wt_length'] = data['sequence'].astype(str).str.len()
data[['range_start', 'range_end']] = pd.DataFrame(
    data['range mutated'].apply(repair_range).tolist(), index = data.index)

print("=" * 70)
print("UNIT COUNTS")
print("=" * 70)
for paper, group in data.groupby('paper'):
    print(f"{paper}:")
    print(f"  manifest rows      {len(group):>8,}")
    print(f"  constructs         {group['construct'].nunique():>8,}")
    print(f"  distinct wt seqs   {group['sequence'].nunique():>8,}")
    print(f"  uniprot accessions {group['accession'].nunique():>8,}")
    print(f"  variants           {group['n_variants'].sum():>8,.0f}")
print("total:")
print(f"  manifest rows      {len(data):>8,}")
print(f"  constructs         {data['construct'].nunique():>8,}")
print(f"  distinct wt seqs   {data['sequence'].nunique():>8,}")
print(f"  variants           {data['n_variants'].sum():>8,.0f}")

# collapse to one row per construct, which is the dataset unit used downstream
order_cols = ['num_single_mutants', 'num_double_mutants', 'num_triple_mutants',
              'num_higher_order', 'num_insertion_mutants', 'num_deletion_mutants']
for col in order_cols:
    data[col] = pd.to_numeric(data[col], errors = 'coerce').fillna(0)

constructs = data.groupby('construct').agg(
    paper = ('paper', 'first'),
    uniprot_id = ('uniprot ID (preferably GENE_ORGANISM)', 'first'),
    accession = ('accession', 'first'),
    pfam_acc = ('pfam accession', 'first'),
    organism = ('source organism', 'first'),
    phenotype = ('phenotype', 'first'),
    transformation = ('transformation', 'first'),
    wt_sequence = ('sequence', 'first'),
    wt_length = ('wt_length', 'first'),
    range_start = ('range_start', 'min'),
    range_end = ('range_end', 'max'),
    n_variants = ('n_variants', 'sum'),
    n_single = ('num_single_mutants', 'sum'),
    n_double = ('num_double_mutants', 'sum'),
    n_triple = ('num_triple_mutants', 'sum'),
    n_higher = ('num_higher_order', 'sum'),
    n_insertion = ('num_insertion_mutants', 'sum'),
    n_deletion = ('num_deletion_mutants', 'sum'),
    n_rows = ('filename', 'size'),
    csv_paths = ('csv_path', lambda s: ';'.join(sorted(set(s.dropna().astype(str))))),
).reset_index()

print()
print("=" * 70)
print("QC")
print("=" * 70)
print(f"constructs                         {len(constructs):>8,}")
print(f"  missing pfam accession           {constructs['pfam_acc'].isna().sum():>8,}")
print(f"  missing wt sequence              {constructs['wt_sequence'].isna().sum():>8,}")
print(f"  missing csv path                 {(constructs['csv_paths'] == '').sum():>8,}")
print(f"  duplicated wt sequence           {constructs['wt_sequence'].duplicated().sum():>8,}")
print(f"  range mutated unparsed           {constructs['range_end'].isna().sum():>8,}")

# the repaired range should end at the wild-type length if the whole construct was scanned
scanned = constructs.dropna(subset = ['range_end'])
mismatch = scanned[scanned['range_end'] != scanned['wt_length']]
print(f"  range_end != wt_length           {len(mismatch):>8,}")

# the per-order counts should add up to the reported variant total
counted = constructs[['n_single', 'n_double', 'n_triple', 'n_higher',
                      'n_insertion', 'n_deletion']].sum(axis = 1)
off_by = (counted - constructs['n_variants']).abs()
print(f"  order counts != n_variants       {(off_by > 1).sum():>8,}")
print(f"  (median absolute discrepancy)    {off_by.median():>8,.0f}")

print()
print("mutational order by paper:")
print(constructs.groupby('paper')[['n_single', 'n_double', 'n_triple', 'n_higher',
                                   'n_insertion', 'n_deletion']].sum().to_string())
print()
print("wt length by paper:")
print(constructs.groupby('paper')['wt_length'].describe()[['count', 'min', '50%', 'max']].to_string())

constructs_path = f"{output_dir}/constructs.csv"
constructs.to_csv(constructs_path, index = False)
print()
print(f"Construct table saved to {constructs_path}")

mismatch_path = f"{output_dir}/constructs_qc_mismatches.csv"
mismatch.to_csv(mismatch_path, index = False)
print(f"QC mismatches saved to {mismatch_path}")
