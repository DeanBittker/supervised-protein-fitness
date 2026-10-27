import os
import re
import numpy as np
import pandas as pd

# Turns the ProteinGym2 manifest into one row per Pfam domain instance, which is the
# unit a held-out-domain split needs. Four calls had to be made to get there, and each
# is a knob rather than a buried constant so the cost of the call can be measured:
#
#   partial coverage  a region the assay construct only partly covers is kept if the
#                     construct holds at least SPF_MIN_COVERAGE of it
#   repeats           a Pfam named with several ranges is several instances, not one,
#                     so each repeat is judged and held out on its own
#   overlaps          where two families claim the same residues, one wins them whole
#                     and the other is trimmed to its longest surviving run
#   clans             the held-out unit is the clan where one is known, so related
#                     families cannot sit on both sides of a split
#
#   SPF_MANIFEST=<annotated manifest csv> python scripts/pfam_extract.py

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
extract_dir = f"{output_root}/results/extraction"

manifest_path = os.environ.get("SPF_MANIFEST", "")
if not manifest_path:
    raise SystemExit("set SPF_MANIFEST to the annotated manifest csv")
if not os.path.exists(manifest_path):
    raise SystemExit(f"{manifest_path} not found")

min_coverage = float(os.environ.get("SPF_MIN_COVERAGE", 0.50))
min_residues = int(os.environ.get("SPF_MIN_RESIDUES", 30))
# which family wins contested residues. 'fraction' gives them to the family they are a
# larger part of, so a small domain nested in a sprawling one keeps itself whole;
# 'length' gives them to the longer annotation instead.
overlap_rule = os.environ.get("SPF_OVERLAP_RULE", "fraction")
if overlap_rule not in ('fraction', 'length'):
    raise SystemExit("SPF_OVERLAP_RULE must be 'fraction' or 'length'")
clans_path = os.environ.get("SPF_CLANS", f"{project_dir}/reference/pfam_clans.tsv")

os.makedirs(extract_dir, exist_ok = True)

PFAM = 'pfam accession'
INDICES = 'pfam indices'
START = 'start index'
SEQUENCE = 'sequence'
PAPER = 'last author last name'
PROTEIN = 'accession'

manifest = pd.read_csv(manifest_path, low_memory = False)
print(f"{len(manifest):,} manifest rows")

# --- the index column's grammar -----------------------------------------------------
# Several Pfams in one row are separated by a colon and line up positionally with the
# colon-separated groups in the index column; the ranges within one group, separated by
# commas, are the repeats of that one family.
months = {name: number for number, name in enumerate(
    ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
     'jul', 'aug', 'sep', 'oct', 'nov', 'dec'], start = 1)}


def repair(piece):
    """Undo a spreadsheet's reading of a range as a date. A start residue of 1 to 12
    looks like a month, so '8-43' comes back as 'Aug-43'; the month name is the start
    and the trailing number the end. Anything else is returned untouched."""
    match = re.fullmatch(r'([A-Za-z]{3})-(\d+)', piece)
    if match and match.group(1).lower() in months:
        return f"{months[match.group(1).lower()]}-{match.group(2)}"
    return piece


def parse_group(text):
    """One family's index group as a list of (start, end), 1-based inclusive in the
    reference protein's numbering. Returns None if any piece cannot be read."""
    segments = []
    for piece in str(text).split(','):
        piece = piece.strip()
        if not piece:
            continue
        match = re.fullmatch(r'(\d+)-(\d+)', repair(piece))
        if not match:
            return None
        start, end = int(match.group(1)), int(match.group(2))
        if end < start:
            return None
        segments.append((start, end))
    return segments or None


# --- one row per repeat -------------------------------------------------------------
instances, dropped = [], []


def drop(row, accession, reason, **extra):
    dropped.append({'filename': row['filename'], 'pfam': accession,
                    'paper': row[PAPER], 'reason': reason, **extra})


annotated = manifest[manifest[PFAM].notna()]
for _, row in annotated.iterrows():
    accessions = [a.strip() for a in str(row[PFAM]).split(':')]
    groups = str(row[INDICES]).split(':') if pd.notna(row[INDICES]) else []
    if len(groups) != len(accessions):
        drop(row, ':'.join(accessions), 'accession and index groups do not line up')
        continue
    sequence = row[SEQUENCE]
    start_index = pd.to_numeric(row[START], errors = 'coerce')
    if not isinstance(sequence, str) or not sequence or pd.isna(start_index):
        drop(row, ':'.join(accessions), 'no construct sequence or start index')
        continue
    # the window of the reference protein the construct actually holds
    construct_start = int(start_index)
    construct_end = construct_start + len(sequence) - 1

    for accession, group in zip(accessions, groups):
        segments = parse_group(group)
        if segments is None:
            drop(row, accession, 'index group could not be read', indices = group)
            continue
        for repeat, (start, end) in enumerate(segments, start = 1):
            instances.append({
                'filename': row['filename'], 'protein': row[PROTEIN],
                'paper': row[PAPER], 'pfam': accession,
                'repeat': repeat, 'repeats_in_row': len(segments),
                'pfam_start': start, 'pfam_end': end, 'pfam_length': end - start + 1,
                'construct_start': construct_start, 'construct_end': construct_end,
                'construct_length': len(sequence),
                'is_primary_condition': row['is_primary_condition'],
                'score_distribution_status': row['score_distribution_status'],
            })

instances = pd.DataFrame(instances)
print(f"{len(instances):,} domain instances named, "
      f"{int((instances['repeats_in_row'] > 1).sum()):,} of them repeats of a family "
      f"appearing more than once in its row")

# --- coverage by the construct ------------------------------------------------------
# A region is only usable to the extent the assay construct contains it; the overlap of
# the two windows is what there are scores for.
overlap_start = instances[['pfam_start', 'construct_start']].max(axis = 1)
overlap_end = instances[['pfam_end', 'construct_end']].min(axis = 1)
instances['covered_start'] = overlap_start
instances['covered_end'] = overlap_end
instances['covered'] = (overlap_end - overlap_start + 1).clip(lower = 0)
instances['coverage'] = instances['covered'] / instances['pfam_length']

print(f"\ncoverage of the named region by the construct")
for low, high in [(1.0, 1.01), (0.9, 1.0), (0.5, 0.9), (0.0, 0.5)]:
    part = instances[(instances['coverage'] >= low) & (instances['coverage'] < high)]
    label = 'complete' if low == 1.0 else f"{low:.0%} to {high:.0%}"
    print(f"  {len(part):6,}  {label}")

# --- overlaps between families on one construct -------------------------------------
# Two families can claim the same residues, and a residue that belongs to both would
# put one domain on each side of a held-out split. One family wins the contested run
# whole; the loser keeps its longest surviving run.
instances = instances.reset_index(drop = True)
instances['trimmed'] = False
contested = 0

for filename, block in instances.groupby('filename', sort = False):
    live = block[block['covered'] > 0]
    if len(live) < 2:
        continue
    # residues each instance currently holds, as a set per index
    held = {index: set(range(int(r['covered_start']), int(r['covered_end']) + 1))
            for index, r in live.iterrows()}
    order = list(held)
    for a_pos, a in enumerate(order):
        for b in order[a_pos + 1:]:
            if instances.at[a, 'pfam'] == instances.at[b, 'pfam']:
                continue  # repeats of one family are its own business
            shared = held[a] & held[b]
            if not shared:
                continue
            contested += 1
            if overlap_rule == 'fraction':
                # the family the contested run is a larger part of has the stronger claim
                claim_a = len(shared) / instances.at[a, 'pfam_length']
                claim_b = len(shared) / instances.at[b, 'pfam_length']
            else:
                claim_a = instances.at[a, 'pfam_length']
                claim_b = instances.at[b, 'pfam_length']
            loser = b if claim_a > claim_b else a
            held[loser] -= shared
            instances.at[loser, 'trimmed'] = True
    for index in order:
        residues = sorted(held[index])
        if not residues:
            instances.at[index, 'covered'] = 0
            continue
        # longest run left, since a domain split in two is not one domain
        best_start = best_end = run_start = residues[0]
        for previous, current in zip(residues, residues[1:]):
            if current != previous + 1:
                if previous - run_start > best_end - best_start:
                    best_start, best_end = run_start, previous
                run_start = current
        if residues[-1] - run_start > best_end - best_start:
            best_start, best_end = run_start, residues[-1]
        instances.at[index, 'covered_start'] = best_start
        instances.at[index, 'covered_end'] = best_end
        instances.at[index, 'covered'] = best_end - best_start + 1

instances['coverage'] = instances['covered'] / instances['pfam_length']
print(f"\n{contested:,} overlapping pairs of different families on one construct, "
      f"{int(instances['trimmed'].sum()):,} instances trimmed by the '{overlap_rule}' rule")

# --- the cutoff ---------------------------------------------------------------------
keep = (instances['coverage'] >= min_coverage) & (instances['covered'] >= min_residues)
for _, r in instances[~keep].iterrows():
    dropped.append({'filename': r['filename'], 'pfam': r['pfam'], 'paper': r['paper'],
                    'reason': f"under the cutoff ({r['coverage']:.2f} of "
                              f"{r['pfam_length']} residues)"})
print(f"\ncutoff at {min_coverage:.0%} of the region and {min_residues} residues: "
      f"{int(keep.sum()):,} instances kept, {int((~keep).sum()):,} dropped")
print("  what the cutoff costs, as it moves:")
for cut in (0.3, 0.5, 0.7, 0.9, 1.0):
    at = ((instances['coverage'] >= cut) & (instances['covered'] >= min_residues))
    print(f"    {cut:.0%}  {int(at.sum()):6,} instances  "
          f"{instances.loc[at, 'pfam'].nunique():4,} families  "
          f"{instances.loc[at, 'protein'].nunique():4,} proteins")

kept = instances[keep].copy()

# --- clans --------------------------------------------------------------------------
# Pfam splits some genuinely single folds across several families, so two families in
# one clan on opposite sides of a split would leak. The clan is the held-out unit where
# one is known and the family stands in for itself where none is.
if os.path.exists(clans_path):
    clans = pd.read_csv(clans_path, sep = '\t', header = None, usecols = [0, 1],
                        names = ['pfam', 'clan'], dtype = str)
    clans['clan'] = clans['clan'].replace('\\N', np.nan)
    mapping = clans.dropna(subset = ['clan']).set_index('pfam')['clan']
    kept['clan'] = kept['pfam'].map(mapping)
    kept['group'] = kept['clan'].fillna(kept['pfam'])
    print(f"\nclans read from {clans_path}: "
          f"{int(kept['clan'].notna().sum()):,} of {len(kept):,} instances sit in one, "
          f"{kept['group'].nunique():,} held-out groups against "
          f"{kept['pfam'].nunique():,} families")
else:
    kept['clan'] = np.nan
    kept['group'] = kept['pfam']
    print(f"\nno clan table at {clans_path}, so every family stands alone and related "
          f"families can still straddle a split. Pfam-A.clans.tsv fixes this.")

# --- what is left to model ----------------------------------------------------------
usable = kept[(kept['is_primary_condition'] == True) &
              (kept['score_distribution_status'] == 'ok')]
groups = (usable.groupby('group').agg(
    instances = ('filename', 'size'), datasets = ('filename', 'nunique'),
    proteins = ('protein', 'nunique'), papers = ('paper', 'nunique'),
    families = ('pfam', 'nunique'), median_covered = ('covered', 'median'))
    .sort_values('proteins', ascending = False))

print(f"\nprimary conditions with a usable score distribution: {len(usable):,} instances "
      f"in {len(groups):,} groups")
for cut in (2, 3, 5, 10):
    print(f"  {int((groups['proteins'] >= cut).sum()):4d} groups with {cut:2d}+ distinct "
          f"proteins   ({int((groups['datasets'] >= cut).sum()):4d} with {cut}+ datasets)")
print("\n  deepest ten:")
print(groups.head(10).to_string())

dropped = pd.DataFrame(dropped)
for frame, name in [(kept, 'instances'), (dropped, 'dropped'), (groups, 'groups')]:
    path = f"{extract_dir}/{name}.csv"
    frame.to_csv(path, index = name == 'groups')
    print(f"\nwrote {path}")
