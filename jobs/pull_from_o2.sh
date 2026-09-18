#!/bin/bash
# Run LOCALLY, from the repo root, to pull O2-computed results down instead of
# recomputing them. Everything except the archive read can then run on your laptop.
set -e
USER_ID=${SPF_O2_USER:-deb278}
SHARED=${SPF_REMOTE:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}

mkdir -p results figures
echo "pulling results from O2..."
rsync -avh --progress \
  "$USER_ID@o2.hms.harvard.edu:$SHARED/results/" results/
echo
echo "pulling figures..."
rsync -avh --progress \
  "$USER_ID@o2.hms.harvard.edu:$SHARED/figures/" figures/
echo
echo "done. variants.parquet is the expensive one - with it local, these all run here:"
echo "  python3 scripts/assay_distances.py"
echo "  python3 scripts/make_figures.py"
