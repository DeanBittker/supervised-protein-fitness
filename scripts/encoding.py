import numpy as np

# a few manifest rows record a wild type longer than the region actually mutated, by one
# or two residues. anything beyond this is a genuine mismatch, not an overhang
max_wt_overhang = 4


def to_alignment_columns(sequences, wt_sequence, aligned_wt):
    """Scatter each variant into its wild type's alignment columns.

    Domains in a family differ in length, so raw position i means a different
    thing in each one and a fixed-length encoding cannot be shared across them.
    Projecting onto the alignment gives every domain the same column space.
    A variant usually has the same length as its own wild type, because deletions
    are gap characters in place and insertions have been dropped. Some manifest
    rows record a wild type one or two residues longer than the region that was
    actually mutated, though, so the variant is a window of it rather than the
    same length. Such a variant still differs from its wild type at only one or
    two positions, so the window is placed where it leaves the fewest mismatches.
    """
    columns = [i for i, c in enumerate(aligned_wt) if c != '-']
    width = len(aligned_wt)
    block = np.full((len(sequences), width), '-', dtype='<U1')
    placed = np.zeros(len(sequences), dtype=bool)
    for row, sequence in enumerate(sequences):
        gap = len(wt_sequence) - len(sequence)
        if gap < 0 or gap > max_wt_overhang:
            continue
        shift = 0 if gap == 0 else min(
            range(gap + 1),
            key = lambda start: sum(a != b for a, b in zip(sequence, wt_sequence[start:])))
        for position, column in enumerate(columns[shift:shift + len(sequence)]):
            block[row, column] = sequence[position]
        placed[row] = True
    return block, placed
