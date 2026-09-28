import os
import io
import glob
import zipfile
import numpy as np
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
archive_dir = os.environ.get(
    "SPF_ARCHIVES",
    "/n/groups/marks/projects/ProteinGym2/supervised/260914_domainome_megascale")
n_archives = int(os.environ.get("SPF_N_ARCHIVES", "20"))

# kermut and proteinNPT both want structure alongside sequence, and both assume one
# structure for the assay rather than one per variant. this reports which of those two
# the archives actually hold, because the second would rule those models out entirely.


def archive_entries(path):
    with zipfile.ZipFile(path) as outer:
        if 'dataset.pgdata' not in outer.namelist():
            return None, None
        payload = outer.read('dataset.pgdata')
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        entries = inner.namelist()
        assays = [n for n in entries if n.startswith('assays/') and n.lower().endswith('.csv')]
        table = pd.read_csv(io.BytesIO(inner.read(assays[0]))) if assays else None
    return entries, table


archives = sorted(glob.glob(f"{archive_dir}/*.pgarchive"))
print(f"{len(archives)} archives in {archive_dir}")
if not archives:
    raise SystemExit("no .pgarchive files found - check SPF_ARCHIVES")

step = max(1, len(archives) // n_archives)
rows = []
for path in archives[::step][:n_archives]:
    stem = os.path.basename(path)[:-len('.pgarchive')]
    entries, table = archive_entries(path)
    if entries is None:
        rows.append({'dataset': stem, 'note': 'no dataset.pgdata'})
        continue
    structures = [e for e in entries if e.startswith('structures/') and not e.endswith('/')]
    extensions = sorted({os.path.splitext(e)[1].lower() for e in structures})
    rows.append({
        'dataset': stem,
        'n_variants': len(table) if table is not None else None,
        'n_structures': len(structures),
        'extensions': ','.join(extensions) if extensions else '',
        # one structure for the whole assay is what kermut expects; anything close to
        # one per variant means the archive is carrying per-variant models instead
        'per_variant': (len(structures) > 1 and table is not None
                        and len(structures) >= 0.5 * len(table)),
        'first': structures[0] if structures else '',
        'note': '',
    })

report = pd.DataFrame(rows)
os.makedirs(results_dir, exist_ok = True)
report.to_csv(f"{results_dir}/structure_inventory.csv", index = False)

print("\n" + "=" * 100)
print("ONE STRUCTURE PER ASSAY, OR ONE PER VARIANT?")
print("=" * 100)
print(report.to_string(index = False))
if 'n_structures' in report:
    clean = report[report['n_structures'].notna()]
    print(f"\nstructures per archive: {clean['n_structures'].value_counts().to_dict()}")
    print(f"file types: {clean['extensions'].value_counts().to_dict()}")
    print(f"archives carrying one structure per variant: {int(clean['per_variant'].sum())}")
print(f"\nsaved {results_dir}/structure_inventory.csv")
