import os
import re
import glob
import json
import io
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
n_archives = int(os.environ.get("SPF_N_ARCHIVES", "12"))

# slices.json carries a kfold list of boolean masks over an assay's rows, but nothing
# that names the scheme those masks came from. the scheme matters: a split by position
# modulo k, or by contiguous position blocks, holds out a region of the protein, while
# a random split does not. this works out which one it is from the masks themselves.


def read_archive(path):
    # the same two-layer unpack load_assays.py does, repeated here because that file is
    # a pipeline script that would run its whole body on import
    with zipfile.ZipFile(path) as outer:
        names = outer.namelist()
        if 'dataset.pgdata' not in names:
            return None, None
        payload = outer.read('dataset.pgdata')
        slices = outer.read('slices.json') if 'slices.json' in names else None
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        assays = [n for n in inner.namelist()
                  if n.startswith('assays/') and n.lower().endswith('.csv')]
        if not assays:
            return None, slices
        table = pd.read_csv(io.BytesIO(inner.read(assays[0])))
    return table, slices


def positions(mutant):
    # a mutant reads like A12G, or A12G:K30R for a double. the scheme tests only make
    # sense on singles, so anything with more than one substitution is set aside
    found = re.findall(r'[A-Za-z](\d+)[A-Za-z]', str(mutant))
    return [int(p) for p in found]


def classify(fold_positions, n_folds):
    # contiguous: the fold's positions fill an unbroken run
    # modulo: every position in the fold has the same remainder against the fold count
    spans, remainders = [], []
    for held in fold_positions:
        if len(held) < 2:
            continue
        spans.append(len(held) == (max(held) - min(held) + 1))
        remainders.append(len({p % n_folds for p in held}) == 1)
    if not spans:
        return 'too few positions'
    if all(remainders):
        return 'modulo'
    if all(spans):
        return 'contiguous'
    return 'random'


archives = sorted(glob.glob(f"{archive_dir}/*.pgarchive"))
print(f"{len(archives)} archives in {archive_dir}")
if not archives:
    raise SystemExit("no .pgarchive files found - check SPF_ARCHIVES")

step = max(1, len(archives) // n_archives)
sample = archives[::step][:n_archives]

rows = []
for path in sample:
    stem = os.path.basename(path)[:-len('.pgarchive')]
    table, raw = read_archive(path)
    if raw is None or table is None:
        rows.append({'dataset': stem, 'note': 'no slices' if raw is None else 'no assay'})
        continue
    payload = json.loads(raw)
    folds = payload.get('kfold', [])
    masks = []
    for fold in folds:
        assays = fold.get('assays') or []
        if assays and assays[0].get('records') is not None:
            masks.append(np.asarray(assays[0]['records'], dtype = bool))
    if not masks or len({m.size for m in masks}) != 1:
        rows.append({'dataset': stem, 'note': f'{len(masks)} unusable masks'})
        continue

    masks = np.vstack(masks)
    per_row = masks.sum(axis = 0)
    mutant_column = next((c for c in table.columns if c.strip().lower() == 'mutant'), None)

    scheme = 'no mutant column'
    if mutant_column is not None and masks.shape[1] == len(table):
        singles = table[mutant_column].map(lambda m: positions(m))
        held = []
        for mask in masks:
            found = {p[0] for p, keep in zip(singles, mask) if keep and len(p) == 1}
            held.append(sorted(found))
        scheme = classify(held, masks.shape[0])

    rows.append({
        'dataset': stem,
        'n_folds': masks.shape[0],
        'n_records': masks.shape[1],
        'n_assay_rows': len(table),
        'lengths_match': masks.shape[1] == len(table),
        'partition': bool((per_row == 1).all()),
        'rows_in_no_fold': int((per_row == 0).sum()),
        'rows_in_many_folds': int((per_row > 1).sum()),
        'smallest_fold': int(masks.sum(axis = 1).min()),
        'largest_fold': int(masks.sum(axis = 1).max()),
        'scheme': scheme,
        'note': '',
    })

report = pd.DataFrame(rows)
os.makedirs(results_dir, exist_ok = True)
report.to_csv(f"{results_dir}/slices_inspection.csv", index = False)

print("\n" + "=" * 100)
print("WHAT THE PREDEFINED KFOLD SLICES ARE")
print("=" * 100)
print(report.to_string(index = False))
if 'n_folds' in report:
    clean = report[report['n_folds'].notna()]
    print(f"\nfold counts: {clean['n_folds'].value_counts().to_dict()}")
    print(f"masks partition the assay rows: {clean['partition'].value_counts().to_dict()}")
    print(f"schemes: {clean['scheme'].value_counts().to_dict()}")
print(f"\nsaved {results_dir}/slices_inspection.csv")
