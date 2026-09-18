#!/bin/bash
# Run ONCE on an O2 login node from the repo root. Login nodes have network access
# for pip and for caching the model; compute nodes may not.
set -e

SHARED=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}

module load gcc/9.2.0 2>/dev/null || true
module load python/3.10.11 2>/dev/null || module load python/3.9.14 2>/dev/null || true
echo "python: $(which python3)  $(python3 --version)"

python3 -m venv .venv
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet numpy pandas scipy matplotlib seaborn biotite pyarrow
echo "environment ready"

echo
echo "shared output folder: $SHARED"
mkdir -p "$SHARED"/{data,results,figures,embeddings}
mkdir -p logs
echo "created data/ results/ figures/ embeddings/"

if [ ! -f "$SHARED/data/260822_manifests_R1_plus_R2.csv" ]; then
  found=$(find /n/groups/marks/projects -maxdepth 4 -name '*260822_manifests*' 2>/dev/null | head -1)
  if [ -n "$found" ]; then
    cp "$found" "$SHARED/data/260822_manifests_R1_plus_R2.csv"
    echo "manifest copied from $found"
  else
    echo
    echo "MISSING: $SHARED/data/260822_manifests_R1_plus_R2.csv"
    echo "from a LOCAL terminal run:"
    echo "  scp ~/Desktop/Debbie_Marks_Lab/code/Supervised_ML/260822_manifests_R1_plus_R2.csv \\"
    echo "      deb278@o2.hms.harvard.edu:$SHARED/data/"
    exit 1
  fi
fi

echo
echo "add this to ~/.bashrc so every session points at the shared folder:"
echo "  export SPF_OUTPUT=$SHARED"
echo "  export SPF_DATA=\$SPF_OUTPUT/data"
echo
echo "next: sbatch jobs/run_msa.sbatch"
