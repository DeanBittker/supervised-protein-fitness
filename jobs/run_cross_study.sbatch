#!/bin/bash
#SBATCH --job-name=spf_cross
#SBATCH --partition=short
#SBATCH --time=0-08:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_cross_%j.out
#SBATCH --error=logs/spf_cross_%j.err

# Train on one study's domains, test on the other's. Random forest over ESM embeddings
# is the slow cell here, tens of thousands of variants against hundreds of dimensions,
# so this wants the cores a login or interactive node will not give it.
#
#   sbatch jobs/run_cross_study.sbatch
#   SPF_THRESHOLD=0.6 sbatch jobs/run_cross_study.sbatch

set -o pipefail
mkdir -p logs

module load gcc/9.2.0 2>/dev/null || true
source .venv/bin/activate

export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
export SPF_FAMILY=${SPF_FAMILY:-PF00018}
export SPF_THRESHOLD=${SPF_THRESHOLD:-0.7}
export SPF_ESM=${SPF_ESM:-ESM2-150M}

echo "host: $(hostname)   cores: $SLURM_CPUS_PER_TASK"
echo "family: $SPF_FAMILY   identity: $SPF_THRESHOLD   model: $SPF_ESM"
echo "started: $(date)"

python3 -u scripts/cross_study.py

echo "finished: $(date)"
