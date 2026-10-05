import os
import glob
import numpy as np
import pandas as pd

# Collect the per-fold scores the leave-one-cluster-out inference array wrote and put
# them beside ridge's, domain by domain. Both held the same domain out of the same
# family, so this is the comparison the replicate split could only make on two domains,
# where they disagreed badly: 0.53 against 0.21.

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
output_root = os.environ.get("SPF_OUTPUT", project_dir)
results_dir = f"{output_root}/results"
family = os.environ.get("SPF_FAMILY", "PF00018")
paper = os.environ.get("SPF_PAPER", "lehner")
threshold = float(os.environ.get("SPF_THRESHOLD", "0.6"))
# the ridge cells to read, matching how PALM was trained
encoding = os.environ.get("SPF_ENCODING", "onehot")
model = os.environ.get("SPF_MODEL", "ridge")
target = os.environ.get("SPF_TARGET", "zscore")

paths = sorted(glob.glob(f"{results_dir}/palm_test_spearman_loco*.csv"))
if not paths:
    raise SystemExit(f"no palm_test_spearman_loco*.csv under {results_dir}\n"
                     f"  run jobs/run_palm_loco_inference.sbatch first")

palm = pd.concat([pd.read_csv(path) for path in paths], ignore_index = True)
palm = palm.rename(columns = {'spearman': 'palm', 'pooled': 'palm_pooled'})
# a fold holding a cluster of several domains scores each of them, so keep them all
palm = palm[['run', 'dataset', 'n', 'palm']].drop_duplicates(subset = ['dataset'])

print(f"{len(paths)} folds scored, {len(palm)} domains")
print(f"  PALM, averaged within protein   {palm['palm'].mean():+.3f} "
      f"(median {palm['palm'].median():+.3f}, sd {palm['palm'].std():.3f})")

ridge_path = f"{results_dir}/loco_results_{family}.csv"
folds_path = f"{results_dir}/loco_{family}.csv"
if not (os.path.exists(ridge_path) and os.path.exists(folds_path)):
    print(f"\nno ridge comparison: {ridge_path} or {folds_path} missing")
    raise SystemExit(0)

folds = pd.read_csv(folds_path)
folds = folds[(folds['scope'] == paper) & np.isclose(folds['threshold'], threshold) &
              (folds['split'] == 'test')][['fold', 'label']]
ridge = pd.read_csv(ridge_path)
ridge = ridge[(ridge['paper'] == paper) & (ridge['encoding'] == encoding) &
              (ridge['model'] == model) & (ridge['target'] == target)]
ridge = folds.merge(ridge[['fold', 'test_spearman_per_protein']], on = 'fold', how = 'left')
ridge = ridge.rename(columns = {'label': 'dataset', 'test_spearman_per_protein': 'ridge'})

# a fold of several domains gets one ridge number for the fold, so the join repeats it
both = palm.merge(ridge[['dataset', 'ridge']], on = 'dataset', how = 'inner')
if both.empty:
    print("\nno domain matched between PALM's scores and ridge's folds")
    raise SystemExit(0)

both['palm_minus_ridge'] = both['palm'] - both['ridge']
print(f"  ridge, same domains             {both['ridge'].mean():+.3f}")
print(f"  PALM on those domains           {both['palm'].mean():+.3f}")
print(f"  PALM wins on {int((both['palm_minus_ridge'] > 0).sum())} of {len(both)} domains")
print()
print(both.sort_values('ridge', ascending = False)
      [['dataset', 'n', 'ridge', 'palm', 'palm_minus_ridge']].to_string(index = False))

path = f"{results_dir}/palm_vs_ridge_loco_{family}_{paper}.csv"
both.to_csv(path, index = False)
print(f"\nsaved {path}")
