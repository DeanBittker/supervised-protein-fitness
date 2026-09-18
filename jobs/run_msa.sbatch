#!/bin/bash
#SBATCH --job-name=spf_msa
#SBATCH --partition=short
#SBATCH --time=0-00:30
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/spf_msa_%j.out
#SBATCH --error=logs/spf_msa_%j.err

# Sequence-level work only: needs the manifest, not the assay archives, so it can
# run before or independently of the data load.
#   sbatch jobs/run_msa.sbatch

set -e
mkdir -p logs results figures

module load gcc/9.2.0 2>/dev/null || true
module load python/3.10.11 2>/dev/null || module load python/3.9.14 2>/dev/null || true
source .venv/bin/activate

export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
export SPF_DATA=${SPF_DATA:-$SPF_OUTPUT/data}
mkdir -p "$SPF_OUTPUT"/{results,figures,data}
echo "output: $SPF_OUTPUT"

echo "host: $(hostname)"
echo "started: $(date)"

echo; echo "=== 1/4 manifest table ==="
python3 scripts/manifest_table.py

echo; echo "=== 2/4 pfam summary ==="
python3 scripts/pfam_summary.py

echo; echo "=== 3/4 msa, identity matrices, clusters, 80/10/10 split ==="
python3 scripts/msa_identity.py

echo; echo "=== 4/4 clustering sweep across thresholds ==="
python3 scripts/cluster_family.py

echo; echo "=== cross-study overlap ==="
python3 scripts/shared_domains.py

echo; echo "=== figures ==="
python3 scripts/make_figures.py

echo; echo "finished: $(date)"
