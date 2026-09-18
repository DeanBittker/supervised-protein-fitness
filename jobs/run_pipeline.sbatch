#!/bin/bash
#SBATCH --job-name=spf_exploratory
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

echo "host: $(hostname)"
echo "started: $(date)"
echo

echo "=== 1/5 manifest table ==="
python3 scripts/manifest_table.py

echo
echo "=== 2/5 pfam summary ==="
python3 scripts/pfam_summary.py

echo
echo "=== 3/5 family clustering ==="
python3 scripts/cluster_family.py

echo
echo "=== 4/5 loading assay archives ==="
python3 scripts/load_assays.py

echo
echo "=== 5/5 assay distribution distances ==="
python3 scripts/assay_distances.py

echo
echo "=== figures ==="
python3 scripts/make_figures.py

echo
echo "finished: $(date)"
ls -la results figures
