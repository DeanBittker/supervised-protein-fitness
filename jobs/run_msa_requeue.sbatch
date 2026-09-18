#!/bin/bash
#SBATCH --job-name=spf_msa
#SBATCH --partition=short
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --time=0-02:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --output=logs/spf_msa_%j.out
#SBATCH --error=logs/spf_msa_%j.err

# Requeue-safe CPU chain. Every step is idempotent and the expensive archive read is
# cached, so a preempted job that restarts picks up where it left off rather than
# redoing work. Check available partitions with: sinfo -s
# If your cluster exposes a dedicated requeue partition, change --partition above.

set -e
mkdir -p logs
module load gcc/9.2.0 2>/dev/null || true
module load python/3.10.11 2>/dev/null || module load python/3.9.14 2>/dev/null || true
source .venv/bin/activate

export SPF_OUTPUT=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
export SPF_DATA=${SPF_DATA:-$SPF_OUTPUT/data}
mkdir -p "$SPF_OUTPUT"/{results,figures,data}

echo "host: $(hostname)   output: $SPF_OUTPUT"
echo "started: $(date)"

for step in manifest_table pfam_summary msa_identity cluster_family shared_domains; do
  echo; echo "=== $step ==="
  python3 scripts/$step.py
done

echo; echo "=== load_assays (cached; SPF_FORCE_RELOAD=1 to rebuild) ==="
python3 scripts/load_assays.py

echo; echo "=== assay_distances ==="
python3 scripts/assay_distances.py || echo "skipped: needs variants.parquet"

echo; echo "=== figures ==="
python3 scripts/make_figures.py

echo; echo "finished: $(date)"
