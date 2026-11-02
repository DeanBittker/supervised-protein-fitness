#!/bin/bash
#SBATCH --job-name=spf_gp
#SBATCH --partition=short
#SBATCH --time=0-08:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_gp_%A_%a.out
#SBATCH --error=logs/spf_gp_%A_%a.err

# The gaussian process baselines the scoping document asks for, which ridge against
# XGBoost does not answer. GP only: the tree and ridge cells already exist in the same
# results files and are keyed by model, so this extends them rather than redoing them.
#
#   wc -l < results/families.txt
#   sbatch --array=1-22 jobs/run_models_gp.sbatch
#
# To check the cap is not driving the answer, rerun one family deeper:
#   SPF_GP_MAX_TRAIN=3000 sbatch --array=1 --time=1-00:00 jobs/run_models_gp.sbatch

set -o pipefail
mkdir -p logs

PROJECT_DIR=${PROJECT_DIR:-$HOME/supervised-protein-fitness}
export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
families=${SPF_FAMILIES:-$SPF_OUTPUT/results/families.txt}
export SPF_SPLIT_MODE=${SPF_SPLIT_MODE:-loco}
export SPF_MODELS=${SPF_MODELS:-gp}
export SPF_THRESHOLD=${SPF_THRESHOLD:-0.6}
# left at the measured default; raise it only for a deliberate deeper check
export SPF_GP_MAX_TRAIN=${SPF_GP_MAX_TRAIN:-1000}

if [ ! -f "$families" ]; then
  echo "no family list at $families" >&2
  exit 1
fi
index=${SLURM_ARRAY_TASK_ID:-1}
export SPF_FAMILY=$(sed -n "${index}p" "$families" | tr -d '[:space:]')
if [ -z "$SPF_FAMILY" ]; then
  echo "no family on line $index of $families" >&2
  exit 0
fi

module load gcc/9.2.0 2>/dev/null || true
cd "$PROJECT_DIR"
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo "no .venv in $PROJECT_DIR" >&2
  exit 1
fi
# only sklearn is needed for gp, but run_models imports xgboost at module level
python -c "import sklearn, xgboost" || {
  echo "sklearn or xgboost missing from this environment" >&2
  exit 1
}

echo "host: $(hostname)   family: $SPF_FAMILY   models: $SPF_MODELS   cap: $SPF_GP_MAX_TRAIN"
echo "started: $(date)"

for needed in "$SPF_OUTPUT/results/splits_${SPF_FAMILY}.csv" \
              "$SPF_OUTPUT/results/loco_${SPF_FAMILY}.csv"; do
  if [ ! -f "$needed" ]; then
    echo "missing $needed, run make_splits.py for this family first" >&2
    exit 0
  fi
done

python scripts/run_models.py
echo "finished: $(date)"
