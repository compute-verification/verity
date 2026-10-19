#!/usr/bin/env bash
# Copy the files a sha256 list names (paths relative to the sweep, as match_gpu.sh's match.sha256 and build_record.sh's
# rowdir.sha256 write them) from one node to another: STREAMS tar streams through this VM (`research pods ssh`), then every file
# checked against the list on the destination.  /workspace/fs is not shared between vy-glm53-eu and the CPU nodes.
#   bash integrations/vllm/manifests/glm_v2_tree/jobs/copy_rowdir.sh SRC_NODE DST_NODE LIST.sha256 [SW]
# env: [STREAMS=8], [RESEARCH="uv run --frozen --no-sync research"].
set -euo pipefail
SRC=${1:?SRC_NODE} DST=${2:?DST_NODE} LIST=${3:?LIST.sha256} SW=${4:-/workspace/cp/circuits/glm-v2/rows}
read -r -a R <<<"${RESEARCH:-uv run --frozen --no-sync research}"
N=${STREAMS:-8}
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
cut -c67- "$LIST" >"$T/names"
split -n "r/$N" "$T/names" "$T/part."
"${R[@]}" pods ssh "$DST" -- "mkdir -p '$SW'"
pids=()
for p in "$T"/part.*; do
  ( "${R[@]}" pods ssh "$SRC" -- "cd '$SW' && tar -cf - -T -" <"$p" | "${R[@]}" pods ssh "$DST" -- "cd '$SW' && tar -xf -" ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" = 0 ] || { echo "a tar stream failed"; exit 1; }
"${R[@]}" pods ssh "$DST" -- "cd '$SW' && sha256sum --quiet -c -" <"$LIST" && echo "$(wc -l <"$LIST") files on $DST equal $SRC's"
