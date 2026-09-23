#!/bin/bash
#SBATCH --job-name=spf_models
#SBATCH --partition=short
#SBATCH --time=0-11:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_models_%j.out
#SBATCH --error=logs/spf_models_%j.err

# The model grid: 2 encodings x 3 models x 2 papers x 2 targets, across split
# replicates. Every finished cell is appended to model_results_<family>.csv and
# skipped on a re-run, so this resumes after a timeout or a requeue.
#
# Needs embeddings already cached by run_embed_gpu.sbatch for the ESM cells;
# the one-hot cells run regardless.

set -o pipefail
mkdir -p logs

module load gcc/9.2.0 2>/dev/null || true
source .venv/bin/activate

export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
export SPF_FAMILY=${SPF_FAMILY:-PF00018}
export SPF_THRESHOLD=${SPF_THRESHOLD:-0.6}
export SPF_REPLICATES=${SPF_REPLICATES:-10}
# set SPF_SPLIT_MODE=loco to hold out one cluster at a time instead of a random tenth,
# which is the only way every domain reaches a test set. writes loco_results_<family>.csv
export SPF_SPLIT_MODE=${SPF_SPLIT_MODE:-replicate}

echo "host: $(hostname)   cores: $SLURM_CPUS_PER_TASK"
echo "family: $SPF_FAMILY   threshold: $SPF_THRESHOLD   replicates: $SPF_REPLICATES   split mode: $SPF_SPLIT_MODE"
echo "started: $(date)"

python3 scripts/run_models.py
python3 scripts/plot_model_results.py

echo "finished: $(date)"
