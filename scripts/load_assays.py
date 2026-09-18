import os
import re
import glob
import zipfile
import tarfile
import io
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
# outputs go to the repo by default; on O2 set SPF_OUTPUT to the shared project
# folder so results are computed once and pulled down rather than recomputed
output_root = os.environ.get("SPF_OUTPUT", project_dir)
data_root = os.environ.get("SPF_DATA", f"{project_dir}/data")
archive_dir = os.environ.get("SPF_ARCHIVES",
    "/n/groups/marks/projects/ProteinGym2/supervised/260914_domainome_megascale")
manifest_path = f"{data_root}/260822_manifests_R1_plus_R2.csv"
output_dir = f"{output_root}/results"
score_column = "DMS_score"
sequence_column = "mutated_sequence"
dois = ['10.1038/s41586-023-06328-6', '10.1038/s41586-024-08370-4']
paper_names = {'10.1038/s41586-023-06328-6': 'rocklin', '10.1038/s41586-024-08370-4': 'lehner'}
inspect_only = os.environ.get("SPF_INSPECT_ONLY", "") == "1"
force_reload = os.environ.get("SPF_FORCE_RELOAD", "") == "1"

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


# archive stems match the manifest filename column, suffixes included, so the join is
# at the manifest-row level. the construct is that name with the rocklin
# _substitutions / _indels suffix removed.
manifest = pd.read_csv(manifest_path, low_memory = False)
rows = manifest[manifest['DOI'].isin(dois)].copy()
rows['paper'] = rows['DOI'].map(paper_names)
rows['construct'] = rows['filename'].astype(str).str.replace(r'_(substitutions|indels)$', '', regex = True)
rows['n_variants'] = pd.to_numeric(rows['number of variants'], errors = 'coerce')
lookup = rows.set_index('filename')

# reading a thousand archives is the slow step, so it is cached: if the parquet is
# already there this is a no-op unless SPF_FORCE_RELOAD=1
cached = f"{output_dir}/variants.parquet"
if os.path.exists(cached) and not force_reload:
    existing = pd.read_parquet(cached)
    print(f"cached variant table already present: {cached}")
    print(f"  {len(existing):,} rows over {existing['dataset'].nunique():,} datasets")
    print("  set SPF_FORCE_RELOAD=1 to rebuild it")
    raise SystemExit(0)

archives = sorted(glob.glob(f"{archive_dir}/*.pgarchive"))
print(f"archive directory: {archive_dir}")
print(f"pgarchive files found: {len(archives)}")
if not archives:
    raise SystemExit("no .pgarchive files found - check the path")

stems = {os.path.basename(p)[:-len('.pgarchive')]: p for p in archives}
known = set(lookup.index)

print()
print("=" * 70)
print("NAME MAPPING (archive stem against manifest filename)")
print("=" * 70)
matched = sorted(set(stems) & known)
extra = sorted(set(stems) - known)
missing = sorted(known - set(stems))
print(f"archives                              {len(stems):>6,}")
print(f"  matching a manifest row for our two studies  {len(matched):>6,}")
print(f"  not in our two studies (other papers)        {len(extra):>6,}")
print(f"manifest rows for our studies with no archive  {len(missing):>6,}")
for label, names in [("unmatched archive stems", extra), ("manifest rows with no archive", missing)]:
    if names:
        print(f"\n{label} (first 8):")
        for n in names[:8]:
            print(f"  {n}")

print()
print("inner contents of the first matching archive:")
probe = stems[matched[0]] if matched else archives[0]
names, blobs = inner_csvs(probe)
print(f"  {os.path.basename(probe)}")
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
for i, stem in enumerate(matched):
    if i % 100 == 0:
        print(f"  reading {i}/{len(matched)}", flush = True)
    names, blobs = inner_csvs(stems[stem])
    if len(names) != 1:
        problems.append({'dataset': stem, 'issue': f"{len(names)} csv entries"})
        if not names:
            continue
    assay = pd.read_csv(io.BytesIO(blobs[names[0]]))
    if score_column not in assay.columns or sequence_column not in assay.columns:
        problems.append({'dataset': stem, 'issue': f"columns {list(assay.columns)[:6]}"})
        continue
    keep = assay[[c for c in [sequence_column, score_column, 'mutant'] if c in assay.columns]].copy()
    keep['dataset'] = stem
    frames.append(keep)

variants = pd.concat(frames, ignore_index = True)
variants['construct'] = lookup.loc[variants['dataset'], 'construct'].values
variants['paper'] = lookup.loc[variants['dataset'], 'paper'].values
variants['pfam_acc'] = lookup.loc[variants['dataset'], 'pfam accession'].values
variants['mut_length'] = variants[sequence_column].astype(str).str.len()

print()
print("=" * 70)
print("LOADED ASSAY DATA")
print("=" * 70)
print(f"datasets read        {len(frames):>8,}")
print(f"problem datasets     {len(problems):>8,}")
print(f"variant rows         {len(variants):>8,}")
print(f"null scores          {variants[score_column].isna().sum():>8,}")
print()
print("rows by paper:")
print(variants.groupby('paper').size().to_string())

loaded = variants.groupby('dataset').size().rename('loaded')
check = lookup.loc[matched, ['n_variants']].join(loaded)
check['difference'] = check['loaded'] - check['n_variants']
print()
print(f"datasets where loaded rows differ from the manifest count: {(check['difference'].abs() > 1).sum()}")
print(f"  median absolute difference: {check['difference'].abs().median():.0f}")

print()
print("score distribution by paper:")
print(variants.groupby('paper')[score_column].describe()[['count', 'mean', 'std', 'min', 'max']].to_string())

variants_path = f"{output_dir}/variants.parquet"
variants.to_parquet(variants_path, index = False)
print()
print(f"Variant table saved to {variants_path}")

check.to_csv(f"{output_dir}/load_rowcount_check.csv")
if problems:
    pd.DataFrame(problems).to_csv(f"{output_dir}/load_problems.csv", index = False)
    print(f"Problem datasets saved to {output_dir}/load_problems.csv")
