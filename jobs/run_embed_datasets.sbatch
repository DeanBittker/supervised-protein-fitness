#!/bin/bash
#SBATCH --job-name=spf_embed_ds
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_embed_ds_%A_%a.out
#SBATCH --error=logs/spf_embed_ds_%A_%a.err

# Embeddings for a list of datasets rather than one family, which is what a
# clan-level run needs: a clan spans many families, so SPF_FAMILY cannot select it.
# make_clan_folds.py writes the list.
#
# Each array task takes every nth dataset, largest first, so the big ones are spread
# across tasks rather than landing in one. Safe to re-run: a dataset whose cache has
# the right row count is skipped, so a task that runs out of time continues where it
# stopped.
#
#   wc -l < $SPF_OUTPUT/results/extraction/fold_datasets.txt
#   sbatch --array=1-10 jobs/run_embed_datasets.sbatch
#
# Check the plan without a GPU first:
#   SPF_DRYRUN=1 SPF_SHARDS=10 SPF_SHARD_INDEX=1 python scripts/embed_esm.py
#
# fair-esm downloads weights on first use and compute nodes may have no network, so
# warm the cache once on a login node:
#   source .venv/bin/activate
#   python3 -c "import esm; esm.pretrained.esm2_t30_150M_UR50D()"

set -o pipefail
mkdir -p logs

PROJECT_DIR=${PROJECT_DIR:-$HOME/supervised-protein-fitness}
export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
export SPF_DATASETS=${SPF_DATASETS:-$SPF_OUTPUT/results/extraction/fold_datasets.txt}
export SPF_ESM=${SPF_ESM:-ESM2-150M}
export SPF_ROWS=${SPF_ROWS:-drop_insertions}
export TORCH_HOME=${TORCH_HOME:-$SPF_OUTPUT/torch_cache}

module load gcc/9.2.0 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
cd "$PROJECT_DIR"
source .venv/bin/activate

if [ ! -f "$SPF_DATASETS" ]; then
  echo "no dataset list at $SPF_DATASETS, run make_clan_folds.py first" >&2
  exit 1
fi

echo "host: $(hostname)   shard ${SLURM_ARRAY_TASK_ID:-1} of ${SLURM_ARRAY_TASK_COUNT:-1}"
echo "list: $SPF_DATASETS ($(wc -l < "$SPF_DATASETS") datasets)   model: $SPF_ESM"
echo "started: $(date)"

python3 scripts/embed_esm.py
echo "finished: $(date)"
