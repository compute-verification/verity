#!/usr/bin/env bash
# Copy the files a sha256 list names (paths relative to the sweep, as match_gpu.sh's match.sha256 and build_record.sh's
# rowdir.sha256 write them) from one node to another: STREAMS tar streams through this VM (`research pods ssh`), then every file
# checked against the list on the destination.  /workspace/fs is not shared between vy-glm53-eu and the CPU nodes.
# A row's `<row>.tree` is not copied as a file: the Build of record and the Match GPU legs are each their node's first job, so the
# two nodes' trees may differ.  The destination ends with the older of the two, which must be one tree or the other's ancestor
# changing only pins, this runbook and its named Commit-only fixes (tree_guard.sh's PINS_ONLY and COMMIT_FIXES); otherwise nothing
# is copied (exit 2).
#   bash integrations/vllm/manifests/glm_v2_tree/jobs/copy_rowdir.sh SRC_NODE DST_NODE LIST.sha256 [SW]
# env: [STREAMS=8], [RESEARCH="uv run --frozen --no-sync research"].
set -euo pipefail
SRC=${1:?SRC_NODE} DST=${2:?DST_NODE} LIST=${3:?LIST.sha256} SW=${4:-/workspace/cp/circuits/glm-v2/rows}
read -r -a R <<<"${RESEARCH:-uv run --frozen --no-sync research}"
N=${STREAMS:-8} TREE=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
source "$TREE/integrations/vllm/manifests/glm_v2_tree/jobs/tree_guard.sh"
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
pins_only() {  # pins_only OLD NEW: NEW descends from OLD and changes only pins, this runbook and COMMIT_FIXES since
  git -C "$TREE" merge-base --is-ancestor "$1" "$2" 2>/dev/null &&
    [ -z "$(git -C "$TREE" diff --name-only "$1" "$2" | { grep -vE "$PINS_ONLY" || true; } |
      while read -r p; do commit_fix "$1" "$2" "$p" || echo "$p"; done)" ]
}
awk 'substr($0, 67) !~ /\.tree$/' "$LIST" >"$T/list"
: >"$T/trees"
for t in $(cut -c67- "$LIST" | grep -E '\.tree$' || true); do
  s=$("${R[@]}" pods ssh "$SRC" -- "cat '$SW/$t'") || { echo "no $SW/$t on $SRC"; exit 2; }
  d=$("${R[@]}" pods ssh "$DST" -- "cat '$SW/$t' 2>/dev/null || true")
  [ "$d" != "$s" ] || continue
  if [ -n "$d" ]; then
    for x in "${s##*@}" "${d##*@}"; do
      git -C "$TREE" cat-file -e "$x^{commit}" 2>/dev/null || git -C "$TREE" fetch -q origin "${s%@*}" || true
    done
    if [ "${d%@*}" = "${s%@*}" ] && pins_only "${d##*@}" "${s##*@}"; then
      echo "$t: $DST keeps $d ($SRC's $s changes only pins, this runbook and COMMIT_FIXES since)"; continue
    fi
    [ "${d%@*}" = "${s%@*}" ] && pins_only "${s##*@}" "${d##*@}" ||
      { echo "$t: $DST is at $d and $SRC at $s, neither the other's pins-only descendant: a row's jobs run at one tree"; exit 2; }
    echo "$t: $SRC's $s replaces $DST's $d (which changes only pins, this runbook and COMMIT_FIXES since)"
  fi
  echo "$t $s" >>"$T/trees"
done
cut -c67- "$T/list" >"$T/names"
split -n "r/$N" "$T/names" "$T/part."
"${R[@]}" pods ssh "$DST" -- "mkdir -p '$SW'"
pids=()
for p in "$T"/part.*; do
  ( "${R[@]}" pods ssh "$SRC" -- "cd '$SW' && tar -cf - -T -" <"$p" | "${R[@]}" pods ssh "$DST" -- "cd '$SW' && tar -xf -" ) &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
[ "$rc" = 0 ] || { echo "a tar stream failed"; exit 1; }
"${R[@]}" pods ssh "$DST" -- "cd '$SW' && sha256sum --quiet -c -" <"$T/list" && echo "$(wc -l <"$T/list") files on $DST equal $SRC's"
while read -r t s; do
  printf '%s\n' "$s" | "${R[@]}" pods ssh "$DST" -- "cat >'$SW/$t'" && echo "$t on $DST: $s"
done <"$T/trees"
