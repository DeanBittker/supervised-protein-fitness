#!/bin/bash
#SBATCH --job-name=spf_families
#SBATCH --partition=short
#SBATCH --time=0-06:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_families_%A_%a.out
#SBATCH --error=logs/spf_families_%A_%a.err

# The classical models on every family deep enough to split, so 0.595 on PF00018 can be
# read against the spread rather than on its own. One-hot only: it needs the alignment
# and nothing else, where the language model would mean embedding tens of thousands of
# sequences per family first.
#
# The array index is a line in the family list:
#   wc -l < results/families.txt
#   sbatch --array=1-25 jobs/run_models_families.sbatch

set -o pipefail
mkdir -p logs

PROJECT_DIR=${PROJECT_DIR:-$HOME/supervised-protein-fitness}
export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
families=${SPF_FAMILIES:-$SPF_OUTPUT/results/families.txt}
export SPF_SPLIT_MODE=${SPF_SPLIT_MODE:-loco}
export SPF_ENCODINGS=${SPF_ENCODINGS:-onehot}
export SPF_THRESHOLD=${SPF_THRESHOLD:-0.6}

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

# the same environment run_models.sbatch uses, so these numbers sit beside the
# PF00018 run rather than beside whatever a different environment resolves to. The
# PALM environment has no xgboost, which is what killed the first attempt.
module load gcc/9.2.0 2>/dev/null || true
cd "$PROJECT_DIR"
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
else
  echo "no .venv in $PROJECT_DIR, falling back to conda" >&2
  module load conda/miniforge3/24.11.3-0 2>/dev/null || true
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${SPF_ENV:-$HOME/PALM/.conda_palm}"
fi
python -c "import xgboost" 2>/dev/null || {
  echo "xgboost missing from this environment, ridge and rf would run but the" >&2
  echo "comparison would be missing a model, so stopping instead" >&2
  exit 1
}

echo "host: $(hostname)   family: $SPF_FAMILY   mode: $SPF_SPLIT_MODE   encodings: $SPF_ENCODINGS"
echo "started: $(date)"

for needed in "$SPF_OUTPUT/results/splits_${SPF_FAMILY}.csv" \
              "$SPF_OUTPUT/results/loco_${SPF_FAMILY}.csv"; do
  if [ ! -f "$needed" ]; then
    echo "missing $needed, run make_splits.py for this family and rsync it across" >&2
    exit 0
  fi
done

python scripts/run_models.py
echo "finished: $(date)"
