#!/usr/bin/env bash
# fetch-records.sh PARTS: for each line "LO HI ART" of PARTS (ART a run record whose out/part holds segments [LO, HI) as
# part-out.sh writes them), fetch the segments this machine's mem69-v2 store lacks, K processes an art, each its own store
# root; then check them and the store's own against their digests and move them in (join.py). Scratch outside the run dir.
#   bash benchmarks/pous/p2_v1/parts/fetch-records.sh PARTS      # on the card's node, with read access to the evidence store
set -euo pipefail
PARTS=$1 K=${K:-2}
ST=${ST:-/workspace/research/pous-store/mem69-v2}
T=$(mktemp -d /workspace/research/tmp-mem69-records.XXXX)
trap 'rm -rf "$T"' EXIT
fetch() {
  PYTHONPATH=$PWD/tools/research/src python3 -m research data fetch "$1" --store "$T/store-$2" --to "$3" "${@:4}" >/dev/null
}
pids=() dirs=()
while read -r LO HI ART; do
  [ -n "${ART:-}" ] || continue
  d="$T/$(echo "$ART" | cut -c5-16)"
  dirs+=("$d/out/part")
  fetch "$ART" "$LO-meta" "$d" --path out/part/setup.json
  for k in $(seq 0 $((K - 1))); do
    args=()
    for s in $(seq "$LO" $((HI - 1))); do
      n=$(printf %05d "$s")
      [ $((s % K)) -eq "$k" ] && [ ! -f "$ST/seg/$n.json" ] || continue
      args+=(--path "out/part/seg/$n.c" --path "out/part/seg/$n.json" --path "out/part/rows/$n.rows")
    done
    [ ${#args[@]} -eq 0 ] && continue
    fetch "$ART" "$LO-$k" "$d" "${args[@]}" &
    pids+=($!)
  done
done < "$PARTS"
for p in "${pids[@]}"; do wait "$p"; done
for d in "${dirs[@]}"; do mkdir -p "$d/seg" "$d/rows"; done
python3 "$(dirname "$0")/join.py" "$ST" "${dirs[@]}" | tee "$OUT/join.json"
ls "$ST/seg" | grep -c '\.json$' > "$OUT/segments-on-card.txt"
