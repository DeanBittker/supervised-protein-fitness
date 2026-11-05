#!/bin/bash
#SBATCH --job-name=spf_clan
#SBATCH --partition=short
#SBATCH --time=0-08:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_clan_%A_%a.out
#SBATCH --error=logs/spf_clan_%A_%a.err

# Models over the leave-one-clan-out folds, one array task per held-out clan. Each
# writes clan_results_<group>.csv, so tasks never contend for a file.
#
# Embeddings only: one-hot projects onto a single family's alignment columns and a
# clan spans families that share none, so run_models.py refuses it in this mode.
#
#   tail -n +2 $SPF_OUTPUT/results/extraction/folds.csv | wc -l
#   sbatch --array=1-25 jobs/run_models_clan.sbatch
#
# To match training size across models, as the family-level comparison does:
#   SPF_MAX_TRAIN=1000 sbatch --array=1-25 jobs/run_models_clan.sbatch

set -o pipefail
mkdir -p logs

PROJECT_DIR=${PROJECT_DIR:-$HOME/supervised-protein-fitness}
export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
export SPF_SPLIT_MODE=clan
export SPF_ENCODINGS=${SPF_ENCODINGS:-ESM2-150M}
export SPF_MODELS=${SPF_MODELS:-ridge,gp}
export SPF_THRESHOLD=${SPF_THRESHOLD:-0.6}

folds="$SPF_OUTPUT/results/extraction/folds.csv"
if [ ! -f "$folds" ]; then
  echo "no folds at $folds, run make_clan_folds.py first" >&2
  exit 1
fi
index=${SLURM_ARRAY_TASK_ID:-1}
# skip the header, take the group column of the index-th fold
export SPF_GROUP=$(tail -n +2 "$folds" | sed -n "${index}p" | cut -d, -f2)
if [ -z "$SPF_GROUP" ]; then
  echo "no fold on line $index of $folds" >&2
  exit 0
fi

module load gcc/9.2.0 2>/dev/null || true
cd "$PROJECT_DIR"
source .venv/bin/activate

echo "host: $(hostname)   group: $SPF_GROUP   encodings: $SPF_ENCODINGS   models: $SPF_MODELS"
echo "started: $(date)"
python scripts/run_models.py
echo "finished: $(date)"
