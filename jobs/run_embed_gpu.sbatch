#!/bin/bash
#SBATCH --job-name=spf_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-08:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_embed_%j.out
#SBATCH --error=logs/spf_embed_%j.err

# Dedicated gpu partition: the job runs to completion without preemption. Use this
# when you want a clean uninterrupted run; use run_embed_gpu_requeue.sbatch to soak
# up idle capacity instead, at the cost of possible preemption.
# Either way embed_mean_pooled.py writes one .npy per dataset and skips what is
# already on disk, so re-running is always safe.
#
# The model must be cached BEFORE submitting, because compute nodes may have no
# network. On a login node, once:
#   source .venv/bin/activate
#   python3 -c "from transformers import AutoModel, AutoTokenizer; \
#     m='facebook/esm2_t30_150M_UR50D'; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"

set -e
mkdir -p logs
module load gcc/9.2.0 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
module load python/3.10.11 2>/dev/null || module load python/3.9.14 2>/dev/null || true
source .venv/bin/activate

export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
export HF_HOME=${HF_HOME:-$SPF_OUTPUT/hf_cache}
export SPF_BATCH=${SPF_BATCH:-32}

echo "host: $(hostname)   gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
echo "started: $(date)"
python3 scripts/embed_mean_pooled.py
echo "finished: $(date)"
