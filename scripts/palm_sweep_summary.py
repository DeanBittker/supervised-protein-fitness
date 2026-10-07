import os
import glob
import numpy as np
import pandas as pd

# The default PALM configuration is badly under-regularised for this task: on the three
# folds tried by hand, raising dropout or widening the kernel rescued the domains it was
# failing on (ITK 0.046 to 0.474, RASA1 -0.146 to 0.310) and cost a little on the one it
# already handled. This reads the full sweep so one configuration can be chosen across
# every fold, rather than the best one per fold, which would be fitting on the test set.
#
# A fold holding exactly one domain has pooled and per-protein Spearman equal, so PALM's
# own test.spearman_r is usable directly. A fold holding several does not, and is left
# out rather than quietly mixed in.

mlruns = os.environ.get("PALM_MLRUNS", os.path.expanduser("~/PALM/mlruns"))
script_dir = os.path.dirname(os.path.abspath(__file__))
output_root = os.environ.get("SPF_OUTPUT", os.path.dirname(script_dir))
results_dir = f"{output_root}/results"
family = os.environ.get("SPF_FAMILY", "PF00018")
paper = os.environ.get("SPF_PAPER", "lehner")
threshold = float(os.environ.get("SPF_THRESHOLD", "0.6"))
encoding = os.environ.get("SPF_ENCODING", "onehot")
model = os.environ.get("SPF_MODEL", "ridge")
target = os.environ.get("SPF_TARGET", "zscore")

prefix = f"{family}_{paper}_loco"


def read_param(run_dir, name, default = None):
    path = f"{run_dir}/params/{name}"
    return open(path).read().strip() if os.path.exists(path) else default


def final_metric(run_dir, name):
    path = f"{run_dir}/metrics/{name}"
    if not os.path.exists(path):
        return None
    last = None
    for line in open(path):
        parts = line.split()
        if len(parts) >= 2:
            last = float(parts[1])
    return last


rows = []
for run_dir in sorted(glob.glob(f"{mlruns}/*/*")):
    data_name = read_param(run_dir, 'dataset.data_name')
    if not data_name or not data_name.startswith(prefix):
        continue
    suffix = data_name[len(prefix):]
    if not suffix.isdigit():
        continue
    score = final_metric(run_dir, 'test.spearman_r')
    if score is None:
        continue
    dropout = read_param(run_dir, 'predictor.hparams.dropout', '?')
    kernel = read_param(run_dir, 'predictor.hparams.kernel_size', '?')
    config = f"dropout {dropout}, kernel {kernel}"
    rows.append({'fold': int(suffix), 'config': config, 'palm': score,
                 'run': os.path.basename(run_dir)[:8],
                 'written': os.path.getmtime(run_dir)})

if not rows:
    raise SystemExit(f"no {prefix}* runs with test.spearman_r under {mlruns}")
sweep = pd.DataFrame(rows)
# the folds tried by hand before the arrays went out were run a second time by them, so
# a fold and configuration can have more than one run. Keep the newest of each rather
# than averaging, which would hide that the pair disagreed.
sweep = sweep.sort_values('written')
repeats = int(sweep.duplicated(subset = ['fold', 'config']).sum())
if repeats:
    spread = (sweep.groupby(['fold', 'config'])['palm']
              .agg(lambda values: values.max() - values.min()))
    print(f"{repeats} fold-configuration pairs ran more than once, keeping the newest; "
          f"widest disagreement between repeats {spread.max():.3f}")
sweep = sweep.drop_duplicates(subset = ['fold', 'config'], keep = 'last')

# a fold is comparable only when it held exactly one domain out
folds_path = f"{results_dir}/loco_{family}.csv"
if not os.path.exists(folds_path):
    raise SystemExit(f"{folds_path} not found")
folds = pd.read_csv(folds_path)
folds = folds[(folds['scope'] == paper) & np.isclose(folds['threshold'], threshold) &
              (folds['split'] == 'test')]
held = folds.groupby('fold')['label'].agg(['count', 'first'])
single = held[held['count'] == 1]
multi = sorted(held[held['count'] > 1].index)

usable = sweep[sweep['fold'].isin(single.index)].copy()
usable['dataset'] = usable['fold'].map(single['first'])
if multi:
    print(f"folds holding more than one domain, left out: {multi}")
    print("  pooled and per-protein Spearman differ there, so they need the inference "
          "pass rather than PALM's own metric")

ridge_path = f"{results_dir}/loco_results_{family}.csv"
ridge = None
if os.path.exists(ridge_path):
    frame = pd.read_csv(ridge_path)
    frame = frame[(frame['paper'] == paper) & (frame['encoding'] == encoding) &
                  (frame['model'] == model) & (frame['target'] == target)]
    ridge = frame.set_index('fold')['test_spearman_per_protein']

print(f"\n{len(usable)} runs over {usable['fold'].nunique()} single-domain folds")
summary = (usable.groupby('config')['palm']
           .agg(['count', 'mean', 'median', 'std', 'min', 'max'])
           .sort_values('mean', ascending = False))
print(summary.to_string(float_format = lambda v: f"{v:+.3f}"))

if ridge is not None:
    shared = sorted(set(usable['fold']) & set(ridge.index))
    complete = [c for c, g in usable.groupby('config')
                if set(shared).issubset(set(g['fold']))]
    if complete:
        print(f"\non the {len(shared)} folds ridge also covers:")
        print(f"  ridge{'':24s} {ridge.loc[shared].mean():+.3f}")
        for config in complete:
            got = usable[(usable['config'] == config) & (usable['fold'].isin(shared))]
            got = got.set_index('fold')['palm'].loc[shared]
            print(f"  {config:28s} {got.mean():+.3f}   "
                  f"PALM wins {int((got.values > ridge.loc[shared].values).sum())} "
                  f"of {len(shared)}")

wide = usable.pivot_table(index = ['fold', 'dataset'], columns = 'config', values = 'palm')
if ridge is not None:
    wide['ridge'] = [ridge.get(fold, np.nan) for fold, _ in wide.index]
print()
print(wide.sort_index().to_string(float_format = lambda v: f"{v:+.3f}"))

path = f"{results_dir}/palm_sweep_{family}_{paper}.csv"
wide.to_csv(path)
print(f"\nsaved {path}")
