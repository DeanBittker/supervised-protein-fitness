#!/bin/bash
# Fetch the Pfam family-to-clan table, which is what lets a held-out split hold out a
# clan rather than a family and so stop two names for one fold sitting on both sides.
#
# Run on an O2 login node: compute nodes have no network, and the session Claude runs
# in cannot reach the Pfam mirrors at all.
#
#   bash jobs/fetch_pfam_clans.sh
set -u

# It lands beside the other shared inputs rather than in the repo's reference folder,
# which is tracked: a 20k-line third-party table does not belong in the history, and
# putting it here means every job on every node reads the same copy.
SHARED=${SPF_OUTPUT:-/n/groups/marks/projects/ProteinGym_supervised/260918_domain_clustering}
target="$SHARED/reference/pfam_clans.tsv"

if [ -s "$target" ]; then
  echo "already have $target ($(wc -l < "$target") lines)"
  exit 0
fi
mkdir -p "$SHARED/reference" || { echo "cannot write to $SHARED/reference" >&2; exit 1; }

mirrors=(
  "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz"
  "https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam37.0/Pfam-A.clans.tsv.gz"
)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT

for url in "${mirrors[@]}"; do
  echo "trying $url"
  # --max-time as well as --connect-timeout: a proxy that accepts the connection and
  # then stalls would otherwise hang here with no output and nothing to interrupt
  if command -v curl > /dev/null; then
    curl -fsSL --connect-timeout 20 --max-time 120 -o "$scratch/clans.gz" "$url" && break
  else
    wget -q -T 20 -O "$scratch/clans.gz" "$url" && break
  fi
  echo "  failed"
done

if [ ! -s "$scratch/clans.gz" ]; then
  echo "could not reach any mirror. If the login node is also blocked, download" >&2
  echo "Pfam-A.clans.tsv.gz on your laptop and copy it across with:" >&2
  echo "  scp Pfam-A.clans.tsv deb278@o2.hms.harvard.edu:$target" >&2
  exit 1
fi

gunzip -c "$scratch/clans.gz" > "$scratch/clans.tsv" || {
  echo "the download is not gzip - a proxy probably returned an error page" >&2; exit 1; }

# the table is five tab-separated columns with no header, the second being the clan or
# empty. Check that before anything downstream trusts it.
read -r lines fields <<< "$(awk -F'\t' 'NR==1{f=NF} END{print NR, f}' "$scratch/clans.tsv")"
first=$(head -1 "$scratch/clans.tsv" | cut -f1)
if [ "$fields" != "5" ] || [[ ! "$first" =~ ^PF[0-9]{5}$ ]]; then
  echo "unexpected shape: $lines lines, $fields fields, first id '$first'" >&2
  exit 1
fi
mv "$scratch/clans.tsv" "$target"
clanned=$(awk -F'\t' '$2 != "" && $2 != "\\N"' "$target" | wc -l)
echo "wrote $target: $lines families, $clanned of them in a clan"
