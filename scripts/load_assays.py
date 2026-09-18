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
# the archives do not use one spelling: seen so far are "DMS Score" and "DMS_score",
# "sequence" and "mutated_sequence". match on a normalised key instead of exact text.
score_aliases = ["dmsscore", "score", "dms"]
sequence_aliases = ["mutatedsequence", "sequence", "seq"]
dois = ['10.1038/s41586-023-06328-6', '10.1038/s41586-024-08370-4']
paper_names = {'10.1038/s41586-023-06328-6': 'rocklin', '10.1038/s41586-024-08370-4': 'lehner'}
inspect_only = os.environ.get("SPF_INSPECT_ONLY", "") == "1"
force_reload = os.environ.get("SPF_FORCE_RELOAD", "") == "1"

os.makedirs(output_dir, exist_ok = True)


def normalise(name):
    return ''.join(c for c in str(name).lower() if c.isalnum())


def resolve(columns, aliases):
    lookup = {normalise(c): c for c in columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def read_archive(path):
    """A .pgarchive is a zip holding dataset.pgdata (itself a zip) and slices.json.

    The nested archive lays its contents out as assays/<name>.csv alongside
    sequences/, structures/ and msas/ directories, so the assay table is two zip
    layers down rather than at the top level.
    """
    with zipfile.ZipFile(path) as outer:
        names = outer.namelist()
        if 'dataset.pgdata' not in names:
            return None, names, None
        payload = outer.read('dataset.pgdata')
        slices = outer.read('slices.json') if 'slices.json' in names else None

    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        entries = inner.namelist()
        assays = [n for n in entries if n.startswith('assays/') and n.lower().endswith('.csv')]
        if not assays:
            return None, entries, slices
        table = pd.read_csv(io.BytesIO(inner.read(assays[0])))
    return table, entries, slices


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
peek, entries, _ = read_archive(probe)
print(f"  {os.path.basename(probe)}")
for entry in entries:
    print(f"    {entry}")
if peek is not None:
    print(f"  assay shape: {peek.shape}")
    print(f"  assay columns: {list(peek.columns)}")
    print(peek.head(3).to_string())
else:
    raise SystemExit("could not find an assays/*.csv inside the archive - format changed?")

if inspect_only:
    raise SystemExit("inspect_only is set, stopping before the full load")

frames = []
problems = []
inventory = []
for i, stem in enumerate(matched):
    if i % 100 == 0:
        print(f"  reading {i}/{len(matched)}", flush = True)
    try:
        assay, entries, slices = read_archive(stems[stem])
    except Exception as error:
        problems.append({'dataset': stem, 'issue': f"unreadable: {error}"})
        continue
    if assay is None:
        problems.append({'dataset': stem, 'issue': "no assays/*.csv inside"})
        continue
    found_score = resolve(assay.columns, score_aliases)
    found_sequence = resolve(assay.columns, sequence_aliases)
    if found_score is None or found_sequence is None:
        problems.append({'dataset': stem, 'issue': f"columns {list(assay.columns)[:6]}"})
        continue
    assay = assay.rename(columns = {found_score: score_column, found_sequence: sequence_column})

    # each archive also carries a fasta, a structure and two msas; record what is
    # present so the inventory is known without reopening a thousand zips later
    inventory.append({
        'dataset': stem,
        'has_fasta': any(e.startswith('sequences/') for e in entries),
        'has_structure': any(e.startswith('structures/') for e in entries),
        'n_msas': sum(1 for e in entries if e.startswith('msas/')),
        'has_slices': slices is not None,
    })

    keep = assay[[c for c in [sequence_column, score_column, 'mutant'] if c in assay.columns]].copy()
    keep['dataset'] = stem
    frames.append(keep)

if not frames:
    # nothing was readable: dump why, so the failure is diagnosable from the log
    print()
    print("=" * 70)
    print("NO DATA LOADED - every archive was rejected")
    print("=" * 70)
    issues = pd.DataFrame(problems)
    if len(issues):
        print(issues['issue'].value_counts().head(10).to_string())
        issues.to_csv(f"{output_dir}/load_problems.csv", index = False)
        print(f"\nfull list: {output_dir}/load_problems.csv")
    raise SystemExit(1)

variants = pd.concat(frames, ignore_index = True)
variants['construct'] = lookup.loc[variants['dataset'], 'construct'].values
variants['paper'] = lookup.loc[variants['dataset'], 'paper'].values
variants['pfam_acc'] = lookup.loc[variants['dataset'], 'pfam accession'].values
variants['mut_length'] = variants[sequence_column].astype(str).str.len()
# insertions are written as lowercase residues and deletions as gap characters,
# following the a3m convention, so both are flaggable straight off the sequence
variants['has_insertion'] = variants[sequence_column].astype(str).str.contains('[a-z]', regex = True)
variants['has_deletion'] = variants[sequence_column].astype(str).str.contains('-', regex = False)

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
print()
print(f"rows with an insertion (lowercase residue): {variants['has_insertion'].sum():,}")
print(f"rows with a deletion (gap character):       {variants['has_deletion'].sum():,}")

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
pd.DataFrame(inventory).to_csv(f"{output_dir}/archive_inventory.csv", index = False)
print(f"Archive inventory saved to {output_dir}/archive_inventory.csv")
if problems:
    pd.DataFrame(problems).to_csv(f"{output_dir}/load_problems.csv", index = False)
    print(f"Problem datasets saved to {output_dir}/load_problems.csv")
