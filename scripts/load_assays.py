import os
import glob
import zipfile
import tarfile
import io
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
archive_dir = "/n/groups/marks/projects/ProteinGym2/supervised/260914_domainome_megascale"
constructs_path = f"{project_dir}/results/constructs.csv"
output_dir = f"{project_dir}/results"
score_column = "DMS_score"
sequence_column = "mutated_sequence"
inspect_only = False

os.makedirs(output_dir, exist_ok = True)


def inner_csvs(path):
    # a .pgarchive is a zipped folder, but fall back to tar in case some are packed differently
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith('.csv')]
            return names, {n: archive.read(n) for n in names}
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            names = [n for n in archive.getnames() if n.lower().endswith('.csv')]
            return names, {n: archive.extractfile(n).read() for n in names}
    return [], {}


archives = sorted(glob.glob(f"{archive_dir}/*.pgarchive"))
if not archives:
    archives = sorted(glob.glob(f"{archive_dir}/*"))
print(f"archive directory: {archive_dir}")
print(f"files found: {len(archives)}")
if not archives:
    raise SystemExit("nothing found - check the path")

print()
print("first five entries:")
for path in archives[:5]:
    print(f"  {os.path.basename(path)}")

constructs = pd.read_csv(constructs_path)
known = set(constructs['construct'])

print()
print("=" * 70)
print("NAME MAPPING")
print("=" * 70)
stems = {os.path.splitext(os.path.basename(p))[0]: p for p in archives}
matched = {s: p for s, p in stems.items() if s in known}
unmatched = sorted(s for s in stems if s not in known)
print(f"archives                        {len(stems):>6,}")
print(f"  stem matches a construct      {len(matched):>6,}")
print(f"  stem does not match           {len(unmatched):>6,}")
print(f"constructs with no archive      {len(known - set(stems)):>6,}")
if unmatched:
    print()
    print("unmatched archive stems (first 10):")
    for s in unmatched[:10]:
        print(f"  {s}")
missing = sorted(known - set(stems))
if missing:
    print()
    print("constructs with no matching archive (first 10):")
    for s in missing[:10]:
        print(f"  {s}")

print()
print("inner contents of the first archive:")
names, blobs = inner_csvs(archives[0])
print(f"  csv entries: {names}")
if names:
    peek = pd.read_csv(io.BytesIO(blobs[names[0]]))
    print(f"  shape: {peek.shape}")
    print(f"  columns: {list(peek.columns)}")
    print(peek.head(3).to_string())

if inspect_only:
    raise SystemExit("inspect_only is set, stopping before the full load")

frames = []
problems = []
for stem, path in sorted(stems.items()):
    names, blobs = inner_csvs(path)
    if len(names) != 1:
        problems.append({'construct': stem, 'issue': f"{len(names)} csv entries"})
        if not names:
            continue
    assay = pd.read_csv(io.BytesIO(blobs[names[0]]))
    if score_column not in assay.columns or sequence_column not in assay.columns:
        problems.append({'construct': stem, 'issue': f"columns {list(assay.columns)[:6]}"})
        continue
    keep = assay[[c for c in [sequence_column, score_column, 'mutant'] if c in assay.columns]].copy()
    keep['construct'] = stem
    frames.append(keep)

variants = pd.concat(frames, ignore_index = True)
variants['mut_length'] = variants[sequence_column].astype(str).str.len()

print()
print("=" * 70)
print("LOADED ASSAY DATA")
print("=" * 70)
print(f"archives read        {len(frames):>8,}")
print(f"problem archives     {len(problems):>8,}")
print(f"variant rows         {len(variants):>8,}")
print(f"null scores          {variants[score_column].isna().sum():>8,}")

annotated = variants.merge(constructs[['construct', 'paper', 'pfam_acc', 'wt_sequence', 'n_variants']],
                           on = 'construct', how = 'left')
unmatched_rows = annotated['paper'].isna().sum()
print()
print("rows by paper:")
print(annotated.groupby('paper').size().to_string())
print(f"rows from archives with no manifest match: {unmatched_rows:,}")

counts = annotated.groupby('construct').size().rename('loaded')
check = constructs.set_index('construct')[['n_variants']].join(counts, how = 'inner')
check['difference'] = check['loaded'] - check['n_variants']
print()
print(f"constructs where loaded rows differ from the manifest count: {(check['difference'].abs() > 1).sum()}")
print(f"  median absolute difference: {check['difference'].abs().median():.0f}")

print()
print("score distribution by paper:")
print(annotated.groupby('paper')[score_column].describe()[['count', 'mean', 'std', 'min', 'max']].to_string())

variants_path = f"{output_dir}/variants.parquet"
annotated.to_parquet(variants_path, index = False)
print()
print(f"Variant table saved to {variants_path}")

if problems:
    problems_path = f"{output_dir}/load_problems.csv"
    pd.DataFrame(problems).to_csv(problems_path, index = False)
    print(f"Problem archives saved to {problems_path}")
