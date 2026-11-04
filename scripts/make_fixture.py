import os
import sys
import numpy as np
import pandas as pd

# A small synthetic family that run_models.py will accept, so a change to an encoder
# or a model can be run end to end on a laptop before it reaches the cluster.
#
# This exists because it was skipped once. A gaussian process was added, tested only
# as a class against synthetic arrays of the wrong shape, and submitted: the kernel
# collapsed on real data, a second array was needed after the fix, and two further
# jobs resumed stale cells and exited having computed nothing. Every one of those
# failures reproduces here in under a minute.
#
#   python scripts/make_fixture.py /tmp/fixture
#   SPF_OUTPUT=/tmp/fixture SPF_FAMILY=PFTEST SPF_SPLIT_MODE=loco \
#     SPF_ENCODINGS=onehot SPF_MODELS=gp,ridge SPF_MAX_TRAIN=500 \
#     python scripts/run_models.py
#
# It deliberately matches the real geometry rather than being convenient: one
# category per alignment column, variants differing from their wild type at a single
# position, and whole domains held out per fold. Uniform random features do not
# reproduce the kernel failure, which is how it was missed.

root = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SPF_FIXTURE", "")
if not root:
    raise SystemExit("usage: python scripts/make_fixture.py <output directory>")

family = os.environ.get("SPF_FIXTURE_FAMILY", "PFTEST")
n_domains = int(os.environ.get("SPF_FIXTURE_DOMAINS", "8"))
per_domain = int(os.environ.get("SPF_FIXTURE_VARIANTS", "320"))
length = int(os.environ.get("SPF_FIXTURE_LENGTH", "45"))
n_folds = int(os.environ.get("SPF_FIXTURE_FOLDS", "4"))
if n_domains < 2 * n_folds:
    raise SystemExit(f"{n_domains} domains cannot fill {n_folds} folds of two")

results_dir = f"{root}/results"
os.makedirs(f"{results_dir}/msa", exist_ok = True)

alphabet = list("ACDEFGHIKLMNPQRSTVWY")
index_of = {letter: i for i, letter in enumerate(alphabet)}
generator = np.random.default_rng(11)
# a position-by-residue effect table, so the score is additive over the sequence and
# a linear model should do well: the point is to exercise the pipeline, not to be hard
effect = generator.normal(size = (length, len(alphabet)))

wild_types, variants, constructs = [], [], []
for domain in range(n_domains):
    wild_type = "".join(generator.choice(alphabet, length))
    wild_types.append(wild_type)
    constructs.append({'construct': f"C{domain}", 'wt_sequence': wild_type})
    for _ in range(per_domain):
        sequence = list(wild_type)
        sequence[int(generator.integers(0, length))] = \
            alphabet[int(generator.integers(0, len(alphabet)))]
        sequence = "".join(sequence)
        score = sum(effect[position, index_of[letter]]
                    for position, letter in enumerate(sequence))
        variants.append({
            'pfam_acc': family, 'dataset': f"D{domain}", 'paper': 'lehner',
            'construct': f"C{domain}", 'mutated_sequence': sequence,
            'DMS_score': score + generator.normal(scale = 0.4),
            'has_insertion': False, 'has_deletion': False})

pd.DataFrame(variants).to_parquet(f"{results_dir}/variants.parquet")
pd.DataFrame(constructs).to_csv(f"{results_dir}/constructs.csv", index = False)

# leave-one-cluster-out over whole domains, two held out per fold
splits = []
for fold in range(n_folds):
    held = {f"C{2 * fold}", f"C{2 * fold + 1}"}
    for domain in range(n_domains):
        splits.append({'scope': 'lehner', 'threshold': 0.6, 'fold': fold,
                       'label': f"C{domain}",
                       'split': 'test' if f"C{domain}" in held else 'train'})
pd.DataFrame(splits).to_csv(f"{results_dir}/loco_{family}.csv", index = False)

# every wild type is the same length here, so the alignment is the sequences as they
# are; to_alignment_columns still exercises its projection against it
with open(f"{results_dir}/msa/lehner_{family}.fasta", "w") as handle:
    for domain, wild_type in enumerate(wild_types):
        handle.write(f">C{domain}\n{wild_type}\n")

print(f"wrote {results_dir}")
print(f"  {len(variants):,} variants over {n_domains} domains, {n_folds} folds, "
      f"{length} alignment columns")
print(f"\nSPF_OUTPUT={root} SPF_FAMILY={family} SPF_SPLIT_MODE=loco "
      f"SPF_ENCODINGS=onehot SPF_MODELS=ridge,gp python scripts/run_models.py")
