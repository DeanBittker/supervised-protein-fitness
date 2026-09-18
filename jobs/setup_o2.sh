#!/bin/bash
# Run this ONCE on an O2 login node, from the repo root. Login nodes have network
# access for pip; compute nodes may not, so the environment is built here.
set -e

module load gcc/9.2.0 2>/dev/null || true
module load python/3.10.11 2>/dev/null || module load python/3.9.14 2>/dev/null || true
echo "python: $(which python3)  $(python3 --version)"

python3 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy pandas scipy matplotlib seaborn biotite pyarrow
echo "environment ready"

mkdir -p data results figures
if [ ! -f data/260822_manifests_R1_plus_R2.csv ]; then
  echo
  echo "MISSING: data/260822_manifests_R1_plus_R2.csv"
  echo "searching the project directory for a copy..."
  found=$(find /n/groups/marks/projects/ProteinGym2 -maxdepth 3 -name '*260822_manifests*' 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    cp "$found" data/260822_manifests_R1_plus_R2.csv
    echo "copied from $found"
  else
    echo "not found on O2 - copy it from your laptop with:"
    echo "  scp ~/Desktop/Debbie_Marks_Lab/code/Supervised_ML/260822_manifests_R1_plus_R2.csv \\"
    echo "      <user>@o2.hms.harvard.edu:~/supervised-protein-fitness/data/"
    exit 1
  fi
fi
echo "manifest present. next: sbatch jobs/run_pipeline.sbatch"
