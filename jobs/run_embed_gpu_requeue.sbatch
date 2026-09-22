#!/bin/bash
#SBATCH --job-name=spf_embed_rq
#SBATCH --partition=gpu_requeue
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=0-06:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_embed_%j.out
#SBATCH --error=logs/spf_embed_%j.err

# ESM-2 150M mean-pooled embeddings, one .npy per dataset. Safe to re-run: a
# dataset whose cache already has the right row count is skipped, so a job that
# runs out of time simply continues where it stopped.
#
# fair-esm downloads weights on first use, and compute nodes may have no network.
# Warm the cache once on a login node before submitting:
#   source .venv/bin/activate
#   python3 -c "import esm; esm.pretrained.esm2_t30_150M_UR50D()"

set -o pipefail
mkdir -p logs

module load gcc/9.2.0 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
source .venv/bin/activate

export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
export SPF_FAMILY=${SPF_FAMILY:-PF00018}
export TORCH_HOME=${TORCH_HOME:-$SPF_OUTPUT/torch_cache}

echo "host: $(hostname)  gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "family: $SPF_FAMILY   output: $SPF_OUTPUT"
echo "started: $(date)"
python3 scripts/embed_esm.py
echo "finished: $(date)"
