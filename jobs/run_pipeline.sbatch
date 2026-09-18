#!/bin/bash
#SBATCH --job-name=spf_full
#SBATCH --partition=short
#SBATCH --time=0-02:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=logs/spf_%j.out
#SBATCH --error=logs/spf_%j.err

# Submit from the repo root, after jobs/setup_o2.sh has been run once:
#   sbatch jobs/run_pipeline.sbatch

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
echo

echo "=== 1/5 manifest table ==="
python3 scripts/manifest_table.py

echo
echo "=== 2/5 pfam summary ==="
python3 scripts/pfam_summary.py

echo
echo "=== 3/7 msa, identity matrices, clusters, 80/10/10 split ==="
python3 scripts/msa_identity.py

echo
echo "=== 4/7 clustering sweep across thresholds ==="
python3 scripts/cluster_family.py

echo
echo "=== 5/7 cross-study overlap ==="
python3 scripts/shared_domains.py

echo
echo "=== 6/7 loading assay archives ==="
python3 scripts/load_assays.py

echo
echo "=== 7/7 assay distribution distances ==="
python3 scripts/assay_distances.py

echo
echo "=== figures ==="
python3 scripts/make_figures.py

echo
echo "finished: $(date)"
ls -la results figures
