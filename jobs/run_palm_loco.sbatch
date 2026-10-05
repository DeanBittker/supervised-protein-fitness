#!/bin/bash
#SBATCH --job-name=spf_palm_loco
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --time=0-04:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_palm_loco_%A_%a.out
#SBATCH --error=logs/spf_palm_loco_%A_%a.err

# One PALM run per held-out cluster, so its per-protein Spearman rests on every domain
# rather than the two that happened to land in a replicate split's test share. Those two
# disagreed badly, 0.53 against 0.21, which is why a single replicate cannot settle it.
#
# The array index is the fold. Count the folds first and size the array to match:
#   python -c "import pandas as pd, numpy as np, os; f = pd.read_csv(os.environ['SPF_OUTPUT'] \
#     + '/results/loco_PF00018.csv'); f = f[(f.scope=='lehner') & np.isclose(f.threshold,0.6)]; \
#     print(sorted(f.fold.unique()))"
#   sbatch --array=1-25 jobs/run_palm_loco.sbatch
#
# Each task exports its own fold before training, so nothing has to be prepared by hand.

set -o pipefail
mkdir -p logs

PALM_DIR=${PALM_DIR:-$HOME/PALM}
PALM_ENV=${PALM_ENV:-$PALM_DIR/.conda_palm}
PROJECT_DIR=${PROJECT_DIR:-$HOME/supervised-protein-fitness}
export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
export SPF_FAMILY=${SPF_FAMILY:-PF00018}
export SPF_PAPER=${SPF_PAPER:-lehner}
export SPF_THRESHOLD=${SPF_THRESHOLD:-0.6}
export SPF_SPLIT_MODE=loco
export SPF_FOLD=${SLURM_ARRAY_TASK_ID:-${SPF_FOLD:-1}}
SPF_EMBEDDER=${SPF_EMBEDDER:-esm2_t6_8M}
SPF_KERNEL=${SPF_KERNEL:-5}
SPF_EXTRA=${SPF_EXTRA:-"predictor.hparams.batch_size=256 predictor.hparams.optimizer_type=adam"}
export PALM_LOSS=${PALM_LOSS:-bce}

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PALM_ENV"
export PYTHONPATH=$PALM_DIR:$PYTHONPATH
export MLFLOW_ALLOW_FILE_STORE=true

echo "host: $(hostname)   family: $SPF_FAMILY   paper: $SPF_PAPER   fold: $SPF_FOLD"
echo "started: $(date)"

cd "$PROJECT_DIR"
python scripts/export_palm.py || exit 1

data_name="${SPF_FAMILY}_${SPF_PAPER}_loco${SPF_FOLD}"
if [ ! -f "$SPF_OUTPUT/palm/${data_name}.csv" ]; then
  echo "no export for fold $SPF_FOLD, nothing to train" >&2
  exit 0
fi

cd "$PALM_DIR"
python src/cli/training.py \
  dataset=regression \
  +dataset.use_sample_weights=false \
  dataset.data_name="$data_name" \
  dataset.use_predefined_split=true \
  dataset.target_scaling=MinMaxScaler \
  persistence.data_root_folder="$SPF_OUTPUT" \
  persistence.training_data=palm \
  predictor=lightattention \
  predictor.model_type=regression \
  predictor.hparams.kernel_size="$SPF_KERNEL" \
  embedder="$SPF_EMBEDDER" \
  embedder.mean_pool=false \
  $SPF_EXTRA

echo "finished: $(date)"
