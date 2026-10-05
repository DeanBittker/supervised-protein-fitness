#!/bin/bash
#SBATCH --job-name=spf_palm_infer
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=0-02:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_palm_infer_%j.out
#SBATCH --error=logs/spf_palm_infer_%j.err

# Score a trained PALM run on held-out rows, so its predictions can be read per
# protein rather than pooled. PALM's own test.spearman_r is pooled across every test
# row at once, which is the metric the rest of this project argues against.
#
# The inference CLI wants a fold directory holding model.yaml, model_state_dict.pt and
# the target scaler, so this assembles one from the mlflow run's artifacts.
#
#   PALM_RUN=<mlflow run id> PALM_NAME=esm_bce_held sbatch jobs/run_palm_inference.sbatch
#   PALM_RUN=... PALM_NAME=... SPF_SPLIT=val sbatch jobs/run_palm_inference.sbatch

set -o pipefail
mkdir -p logs

PALM_DIR=${PALM_DIR:-$HOME/PALM}
PALM_ENV=${PALM_ENV:-$PALM_DIR/.conda_palm}
export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
SPF_FAMILY=${SPF_FAMILY:-PF00018}
SPF_PAPER=${SPF_PAPER:-lehner}
SPF_FOLD=${SPF_FOLD:-replicate0}
# test is the honest split; val has more proteins but is the one the epoch was chosen on
SPF_SPLIT=${SPF_SPLIT:-test}

if [ -z "$PALM_RUN" ]; then
  echo "set PALM_RUN to an mlflow run id" >&2
  exit 1
fi
PALM_NAME=${PALM_NAME:-${PALM_RUN:0:8}}

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PALM_ENV"
export PYTHONPATH=$PALM_DIR:$PYTHONPATH
export MLFLOW_ALLOW_FILE_STORE=true

artifacts=$(ls -d "$PALM_DIR"/mlruns/*/"$PALM_RUN"*/artifacts 2>/dev/null | head -1)
if [ -z "$artifacts" ]; then
  echo "no mlflow run matching $PALM_RUN under $PALM_DIR/mlruns" >&2
  exit 1
fi

fold="$PALM_DIR/PALM_models/$PALM_NAME/fold0"
mkdir -p "$fold"
cp "$artifacts"/model_state_dict.pt "$fold"/
cp "$artifacts"/*LightAttention*.yaml "$fold"/model.yaml
# loaded at init even in inference mode, so it has to be here even though a monotone
# rescaling cannot change a rank correlation
cp "$artifacts"/*targetscaler.skops "$fold"/

data="$SPF_OUTPUT/palm/${SPF_FAMILY}_${SPF_PAPER}_${SPF_FOLD}.csv"
rows="$PALM_DIR/${PALM_NAME}_${SPF_SPLIT}_rows.csv"
python - "$data" "$rows" "$SPF_SPLIT" <<'PY'
import sys
import pandas as pd
frame = pd.read_csv(sys.argv[1])
rows = frame[frame['data_split'] == sys.argv[3]][['sequence', 'value_real', 'dataset']]
rows.to_csv(sys.argv[2], index = False)
print(f"{len(rows):,} {sys.argv[3]} rows over {rows['dataset'].nunique()} proteins")
PY

echo "host: $(hostname)   run: $PALM_RUN   name: $PALM_NAME   split: $SPF_SPLIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no gpu visible"
echo "started: $(date)"

cd "$PALM_DIR"
python src/cli/inference.py \
  --csv "$rows" \
  --model_name "$PALM_NAME" \
  --no_ensemble \
  --models_dir ./PALM_models \
  --data_root . \
  --output_prefix "predictions_${PALM_NAME}_${SPF_SPLIT}"

echo "finished: $(date)"
echo
echo "score it with:"
echo "  PALM_PREDICTIONS=$PALM_DIR/predictions_${PALM_NAME}_${SPF_SPLIT}__sequences.csv \\"
echo "  PALM_TEST=$rows PALM_LABEL=${PALM_NAME}_${SPF_SPLIT} \\"
echo "  python scripts/palm_test_spearman.py"
