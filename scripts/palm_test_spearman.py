import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# PALM reports one Spearman over every test row at once. That number is the one the
# rest of this project argues against: pooled across proteins it rewards a model for
# placing whole domains correctly, which a model that cannot rank variants inside a
# protein can still do. This reads PALM's own predictions back and reports the metric
# the other models are judged by, so the two sit on the same axis.

predictions_path = os.environ.get("PALM_PREDICTIONS", "")
test_path = os.environ.get("PALM_TEST", "")
label = os.environ.get("PALM_LABEL", "")
# the column holding the prediction, when more than one numeric column is a candidate
pred_column = os.environ.get("PALM_PRED_COLUMN", "")

if not predictions_path or not test_path:
    raise SystemExit('set PALM_PREDICTIONS=<the _sequences.csv inference wrote> '
                     'and PALM_TEST=<the rows it was run on>')

predictions = pd.read_csv(predictions_path)
truth = pd.read_csv(test_path)

if not pred_column:
    reserved = {'value_real', 'value_bool', 'len', 'index'}
    numeric = [c for c in predictions.columns
               if c not in reserved and pd.api.types.is_numeric_dtype(predictions[c])]
    if not numeric:
        raise SystemExit(f"no numeric column in {predictions_path}: "
                         f"{list(predictions.columns)}\n  set PALM_PRED_COLUMN")
    pred_column = numeric[0]
    if len(numeric) > 1:
        print(f"several numeric columns {numeric}, using {pred_column!r}; "
              f"set PALM_PRED_COLUMN to choose another")

# joining on the sequence is safer than trusting two files to stay in step, but a
# sequence can repeat across domains, so fall back to position when it is not unique
if 'sequence' in predictions.columns and truth['sequence'].is_unique:
    frame = truth.merge(predictions[['sequence', pred_column]], on = 'sequence', how = 'left')
    missing = int(frame[pred_column].isna().sum())
    if missing:
        raise SystemExit(f"{missing} test rows got no prediction from the join on sequence")
else:
    if len(predictions) != len(truth):
        raise SystemExit(f"cannot join: {len(predictions)} predictions against "
                         f"{len(truth)} test rows, and sequences are not unique")
    print("sequences are not unique, joining by row order instead")
    frame = truth.copy()
    frame[pred_column] = predictions[pred_column].values

frame = frame[frame['value_real'].notna()]
pooled = spearmanr(frame[pred_column], frame['value_real']).statistic

rows = []
for dataset, group in frame.groupby('dataset'):
    # a protein whose variants all share one score has no ranking to get right
    if len(group) < 10 or group['value_real'].nunique() < 2:
        continue
    rows.append({'dataset': dataset, 'n': len(group),
                 'spearman': spearmanr(group[pred_column], group['value_real']).statistic})
per_protein = pd.DataFrame(rows)
if per_protein.empty:
    raise SystemExit("no protein had enough test rows to rank")

print(f"{len(frame):,} test rows over {frame['dataset'].nunique()} proteins")
print(f"  pooled across proteins   {pooled:+.3f}")
print(f"  averaged within protein  {per_protein['spearman'].mean():+.3f} "
      f"(median {per_protein['spearman'].median():+.3f}, "
      f"{len(per_protein)} proteins scored)")
print()
print(per_protein.sort_values('spearman', ascending = False).to_string(index = False))

if label:
    output_root = os.environ.get("SPF_OUTPUT", os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    path = f"{output_root}/results/palm_test_spearman_{label}.csv"
    per_protein.assign(pooled = pooled, run = label).to_csv(path, index = False)
    print(f"\nsaved {path}")
