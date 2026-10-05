#!/bin/bash
#SBATCH --job-name=spf_palm_loco_infer
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_palm_loco_infer_%A_%a.out
#SBATCH --error=logs/spf_palm_loco_infer_%A_%a.err

# Score every leave-one-cluster-out fold on its held-out domain. The array index is the
# fold, and the mlflow run is found by the data_name it trained on rather than by an id
# copied across by hand, because matching 24 ids to 24 folds is where mistakes live.
#
#   sbatch --array=1-24 jobs/run_palm_loco_inference.sbatch
#
# Then collect the folds:
#   python scripts/palm_loco_summary.py

set -o pipefail
mkdir -p logs

PALM_DIR=${PALM_DIR:-$HOME/PALM}
PALM_ENV=${PALM_ENV:-$PALM_DIR/.conda_palm}
PROJECT_DIR=${PROJECT_DIR:-$HOME/supervised-protein-fitness}
export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
SPF_FAMILY=${SPF_FAMILY:-PF00018}
SPF_PAPER=${SPF_PAPER:-lehner}
FOLD=${SLURM_ARRAY_TASK_ID:-${SPF_FOLD:-1}}

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PALM_ENV"
export PYTHONPATH=$PALM_DIR:$PYTHONPATH
export MLFLOW_ALLOW_FILE_STORE=true

data_name="${SPF_FAMILY}_${SPF_PAPER}_loco${FOLD}"
name="loco${FOLD}"

# -x so loco1 does not match loco15. A fold run twice leaves two runs, and one of them
# may be an interrupted attempt with no saved model, so take the newest run that
# actually finished rather than the newest run.
run_dir=""
for param in $(ls -t $(grep -lx "$data_name" "$PALM_DIR"/mlruns/*/*/params/dataset.data_name 2>/dev/null) 2>/dev/null); do
  candidate=$(dirname "$(dirname "$param")")
  if [ -f "$candidate/artifacts/model_state_dict.pt" ]; then
    run_dir="$candidate"
    break
  fi
  echo "skipping $(basename "$candidate"): no model_state_dict.pt, it did not finish"
done
if [ -z "$run_dir" ]; then
  echo "no finished mlflow run trained on $data_name" >&2
  exit 0
fi
artifacts="$run_dir/artifacts"
echo "fold $FOLD -> run $(basename "$run_dir")"

fold_dir="$PALM_DIR/PALM_models/$name/fold0"
mkdir -p "$fold_dir"
cp "$artifacts"/model_state_dict.pt "$fold_dir"/
cp "$artifacts"/*LightAttention*.yaml "$fold_dir"/model.yaml
cp "$artifacts"/*targetscaler.skops "$fold_dir"/

rows="$PALM_DIR/${name}_test_rows.csv"
python - "$SPF_OUTPUT/palm/${data_name}.csv" "$rows" <<'PY'
import sys
import pandas as pd
frame = pd.read_csv(sys.argv[1])
test = frame[frame['data_split'] == 'test'][['sequence', 'value_real', 'dataset']]
test.to_csv(sys.argv[2], index = False)
print(f"{len(test):,} test rows over {test['dataset'].nunique()} domains: "
      f"{', '.join(sorted(test['dataset'].unique()))}")
PY

cd "$PALM_DIR"
python src/cli/inference.py \
  --csv "$rows" \
  --model_name "$name" \
  --no_ensemble \
  --models_dir ./PALM_models \
  --data_root . \
  --output_prefix "predictions_${name}_test" || exit 1

cd "$PROJECT_DIR"
PALM_PREDICTIONS="$PALM_DIR/predictions_${name}_test__sequences.csv" \
PALM_TEST="$rows" \
PALM_LABEL="$name" \
python scripts/palm_test_spearman.py

echo "finished: $(date)"
