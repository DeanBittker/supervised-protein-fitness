#!/bin/bash
#SBATCH --job-name=spf_palm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_palm_%j.out
#SBATCH --error=logs/spf_palm_%j.err

# PALM's LightAttention over per-residue ESM2 embeddings, on our whole-domain splits.
# The point of this model is that it keeps the embedding per residue and convolves
# along it, so a filter spanning a mutated site can see the change. Mean pooling would
# throw that away, which is why mean_pool is false here as it is in every published
# PALM model, even though the embedder configs default it to true.
#
# Needs: export_palm.py already run, patch_palm.py already applied to the checkout.
#   sbatch jobs/run_palm.sbatch
#   SPF_PAPER=rocklin SPF_EMBEDDER=esm2_t33_650M sbatch jobs/run_palm.sbatch

set -o pipefail
mkdir -p logs

PALM_DIR=${PALM_DIR:-$HOME/PALM}
PALM_ENV=${PALM_ENV:-$PALM_DIR/.conda_palm}
export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
SPF_FAMILY=${SPF_FAMILY:-PF00018}
SPF_PAPER=${SPF_PAPER:-lehner}
SPF_FOLD=${SPF_FOLD:-replicate0}
SPF_EMBEDDER=${SPF_EMBEDDER:-esm2_t6_8M}
SPF_KERNEL=${SPF_KERNEL:-5}

module load conda/miniforge3/24.11.3-0 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PALM_ENV"
export PYTHONPATH=$PALM_DIR:$PYTHONPATH
# mlflow refuses a filesystem backend without this, and a database backend buys us
# nothing for a single run
export MLFLOW_ALLOW_FILE_STORE=true

echo "host: $(hostname)"
echo "data: ${SPF_FAMILY}_${SPF_PAPER}_${SPF_FOLD}   embedder: $SPF_EMBEDDER   kernel: $SPF_KERNEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no gpu visible"
echo "started: $(date)"

cd "$PALM_DIR"
python src/cli/training.py \
  dataset=regression \
  +dataset.use_sample_weights=false \
  dataset.data_name="${SPF_FAMILY}_${SPF_PAPER}_${SPF_FOLD}" \
  dataset.use_predefined_split=true \
  dataset.target_scaling=MinMaxScaler \
  persistence.data_root_folder="$SPF_OUTPUT" \
  persistence.training_data=palm \
  predictor=lightattention \
  predictor.model_type=regression \
  predictor.hparams.kernel_size="$SPF_KERNEL" \
  embedder="$SPF_EMBEDDER" \
  embedder.mean_pool=false

echo "finished: $(date)"
