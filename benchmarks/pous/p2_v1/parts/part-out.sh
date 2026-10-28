#!/usr/bin/env bash
# part-out.sh LO HI: hard-link segments [LO, HI) of this machine's mem69-v2 store (seg/, rows/) and its setup.json into
# $OUT/part, so the run's custody publishes them in its run record (a node run holds no key to put them in the store itself).
# Keep a part under 4 GiB (26 segments), since custody uploads don't resume; the record's art is in the run's `.custody`.
#   OUT=DIR bash benchmarks/pous/p2_v1/parts/part-out.sh LO HI
set -euo pipefail
LO=$1 HI=$2
ST=${ST:-/workspace/research/pous-store/mem69-v2}
mkdir -p "$OUT/part/seg" "$OUT/part/rows"
for s in $(seq -f %05g "$LO" $((HI - 1))); do
  for f in "seg/$s.c" "rows/$s.rows" "seg/$s.json"; do
    [ -f "$ST/$f" ] || { echo "missing $f" >&2; exit 1; }
    ln "$ST/$f" "$OUT/part/$f"
  done
done
cp "$ST/setup.json" "$OUT/part/setup.json"
(cd "$OUT/part" && sha512sum seg/*.c rows/*.rows > SHA512SUMS)
printf '{"lo": %d, "hi": %d, "host": "%s", "store": "%s"}\n' "$LO" "$HI" "$(hostname)" "$ST" > "$OUT/part/part.json"
echo "part [$LO, $HI): $(ls "$OUT/part/seg" | grep -c '\.json$') segments" >&2
