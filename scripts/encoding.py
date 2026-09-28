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


# the esm2 family, by the fair-esm loader name and the layer to read. the layer is
# always the last one, so an embedding is the final representation rather than an
# intermediate. kermut's sequence kernel is mean-pooled 650M, which is why that one
# matters beyond curiosity
esm_models = {
    'ESM2-8M':   ('esm2_t6_8M_UR50D',     6, 'esm8M',   16),
    'ESM2-35M':  ('esm2_t12_35M_UR50D',  12, 'esm35M',  16),
    'ESM2-150M': ('esm2_t30_150M_UR50D', 30, 'esm150M', 16),
    'ESM2-650M': ('esm2_t33_650M_UR50D', 33, 'esm650M',  8),
}


def filter_rows(variants, mode):
    """Drop the rows a given encoding cannot represent, and say so.

    Three experiments want three different row sets and the difference is easy to
    lose track of: a fixed-length encoding cannot take an insertion, a site-indexed
    model such as Kermut cannot take either kind of indel, and a convolutional model
    over padded sequences handles insertions but not gaps. Naming the mode rather
    than hardcoding one keeps a result honest about which rows produced it.
    """
    before = len(variants)
    if mode == 'drop_insertions':
        variants = variants[~variants['has_insertion']]
    elif mode == 'drop_indels':
        variants = variants[~variants['has_insertion'] & ~variants['has_deletion']]
    elif mode == 'drop_deletions':
        variants = variants[~variants['has_deletion']]
    elif mode != 'keep_all':
        raise SystemExit(f"unknown row filter {mode!r}, expected one of "
                         f"drop_insertions, drop_indels, drop_deletions, keep_all")
    print(f"row filter {mode}: dropped {before - len(variants):,}, "
          f"{len(variants):,} remain")
    return variants


def result_stem(family, split_mode, row_filter, esm_name):
    """Name a results file after the experiment that produced it.

    A run over a different row set, or a different language model, is a different
    experiment and must not be merged into the default file. The defaults carry no
    suffix so existing results keep their names.
    """
    stem = f"loco_results_{family}" if split_mode == 'loco' else f"model_results_{family}"
    if row_filter != 'drop_insertions':
        stem += f"_{row_filter}"
    if esm_name != 'ESM2-150M':
        stem += f"_{esm_models[esm_name][2]}"
    return stem
