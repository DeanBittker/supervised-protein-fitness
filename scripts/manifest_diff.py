import os
import re
import pandas as pd

# Compares two manifest vintages on the Pfam annotation, because "some of them were
# redone" decides whether results keyed to the older one still stand. A dataset whose
# family assignment moved between manifests was trained and scored under a family it
# is no longer in, which is a different claim rather than a smaller one.
#
# Date damage is repaired on both sides before comparing, so a range a spreadsheet
# mangled in one file and not the other does not read as a re-annotation.
#
#   SPF_MANIFEST_OLD=... SPF_MANIFEST_NEW=... python scripts/manifest_diff.py

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
data_root = os.environ.get("SPF_DATA", f"{output_root}/data")
old_path = os.environ.get("SPF_MANIFEST_OLD",
                          f"{data_root}/260822_manifests_R1_plus_R2.csv")
new_path = os.environ.get("SPF_MANIFEST_NEW",
                          f"{data_root}/260930_manifest_annotated.csv")
out_dir = f"{output_root}/results/survey"

for path in (old_path, new_path):
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found")
os.makedirs(out_dir, exist_ok = True)

PFAM = 'pfam accession'
INDICES = 'pfam indices'
months = {name: number for number, name in enumerate(
    ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
     'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], start = 1)}


def repair(text):
    """Undo a spreadsheet reading a range as a date, on both sides, so corruption in
    one vintage is not mistaken for a changed annotation."""
    def fix(match):
        name = match.group(1).lower()
        return f"{months[name]}-{match.group(2)}" if name in months else match.group(0)
    return re.sub(r'([A-Za-z]{3})-(\d+)', fix, str(text))


def normalise(frame, column):
    """Order within a row should not count as a difference: a row naming two Pfams in
    the other order is the same annotation."""
    def key(text):
        if pd.isna(text):
            return ''
        return ':'.join(sorted(part.strip() for part in repair(text).split(':')))
    return frame[column].map(key)


old = pd.read_csv(old_path, low_memory = False)
new = pd.read_csv(new_path, low_memory = False)
print(f"old {os.path.basename(old_path)}: {len(old):,} rows, {old.shape[1]} columns")
print(f"new {os.path.basename(new_path)}: {len(new):,} rows, {new.shape[1]} columns")

gained = [c for c in new.columns if c not in set(old.columns)]
lost = [c for c in old.columns if c not in set(new.columns)]
print(f"\ncolumns added: {len(gained)}" + (f"  {gained}" if gained else ""))
print(f"columns dropped: {len(lost)}" + (f"  {lost}" if lost else ""))

for frame, label in ((old, 'old'), (new, 'new')):
    if 'filename' not in frame.columns:
        raise SystemExit(f"the {label} manifest has no filename column to key on")
    if frame['filename'].duplicated().any():
        print(f"  note: {label} has {int(frame['filename'].duplicated().sum())} "
              f"duplicate filenames, keeping the first of each")

old = old.drop_duplicates('filename').set_index('filename')
new = new.drop_duplicates('filename').set_index('filename')
only_old = old.index.difference(new.index)
only_new = new.index.difference(old.index)
shared = old.index.intersection(new.index)
print(f"\ndatasets: {len(shared):,} in both, {len(only_new):,} only in new, "
      f"{len(only_old):,} only in old")

both = pd.DataFrame({
    'old_pfam': normalise(old.loc[shared], PFAM),
    'new_pfam': normalise(new.loc[shared], PFAM),
    'old_indices': normalise(old.loc[shared], INDICES),
    'new_indices': normalise(new.loc[shared], INDICES),
})
both['paper'] = new.loc[shared, 'last author last name'].values

family_moved = both['old_pfam'] != both['new_pfam']
indices_moved = (~family_moved) & (both['old_indices'] != both['new_indices'])
gained_pfam = family_moved & (both['old_pfam'] == '') & (both['new_pfam'] != '')
lost_pfam = family_moved & (both['old_pfam'] != '') & (both['new_pfam'] == '')
reassigned = family_moved & ~gained_pfam & ~lost_pfam

print(f"\nof the {len(shared):,} shared datasets")
print(f"  {int((~family_moved & ~indices_moved).sum()):6,}  unchanged")
print(f"  {int(gained_pfam.sum()):6,}  newly annotated (had no Pfam before)")
print(f"  {int(lost_pfam.sum()):6,}  annotation removed")
print(f"  {int(reassigned.sum()):6,}  REASSIGNED to a different family set")
print(f"  {int(indices_moved.sum()):6,}  same families, coordinates moved")

if int(reassigned.sum()):
    print("\n  reassignments by paper: " + ", ".join(
        f"{k} {v}" for k, v in both.loc[reassigned, 'paper'].value_counts().head(8).items()))
    print("\n  first ten:")
    print(both.loc[reassigned, ['paper', 'old_pfam', 'new_pfam']].head(10).to_string())
    print("\n  any result keyed to the old manifest for these datasets was scored "
          "under a family they are no longer in")

changed = both[family_moved | indices_moved]
path = f"{out_dir}/manifest_diff.csv"
changed.to_csv(path)
print(f"\nwrote {path} ({len(changed):,} changed datasets)")
