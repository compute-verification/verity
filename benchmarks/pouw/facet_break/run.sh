#!/usr/bin/env bash
# One FACET-BREAK batch: generate the seeded grid, build and gate it (the gate's refusals are dropped by name and the rest
# rebuilt in chunks of 64, so one bad sink loses only its chunk), gate again and time on one leased GPU, then load every
# program on every facet. From a pushed commit:
#
#   research run --on vy-nebius-2 --project verity --campaign pouw --source <tree> --cwd source --custody-ttl 12h \
#     --env GPU_LEASE_WHO=facet-break -- bash benchmarks/pouw/facet_break/run.sh [generate.py flags: --seed S --n N ...]
#
# $OUT gets fb_*.cu (the sources timed), kept.txt, gate0.txt (the first gate, refusals included), sass/, chunks/*/ (each
# chunk's gate.json, times.jsonl, rows.jsonl), rows.jsonl (every chunk's), loads.jsonl and summary.txt.
set -uo pipefail
OUT=${OUT:?run under research run}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROBE="$HERE/../opcode_probe/probe.py"
export PYTHONDONTWRITEBYTECODE=1
NVCC=${NVCC:-/usr/local/cuda-13.0/bin/nvcc}
[ -x "$NVCC" ] || NVCC=$(command -v nvcc)
export NVCC PATH="$(dirname "$NVCC"):$PATH"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$OUT/sass" "$OUT/chunks"

python3 "$HERE/generate.py" "$@" --emit "$WORK/all.cu" || exit 1
python3 "$PROBE" build --out "$WORK/b0" "$WORK/all.cu" >/dev/null 2>"$OUT/build0.log" || { cat "$OUT/build0.log" >&2; exit 1; }
python3 "$PROBE" gate "$WORK/b0/all" --json "$WORK/gate0.json" >"$OUT/gate0.txt" 2>&1
python3 -c "import json,sys; print('\n'.join(v['probe'] for v in json.load(open(sys.argv[1]))['verdicts'] if v['passed']))" \
  "$WORK/gate0.json" >"$OUT/kept.txt"
echo "kept $(wc -l <"$OUT/kept.txt") of $(grep -c '"probe"' "$WORK/gate0.json") programs"
split -l 64 -d -a 2 "$OUT/kept.txt" "$WORK/chunk_"
bins=()
for c in "$WORK"/chunk_*; do
  i=${c##*_}
  python3 "$HERE/generate.py" "$@" --only "$c" --emit "$OUT/fb_$i.cu" || exit 1
  python3 "$PROBE" build --out "$WORK/b" "$OUT/fb_$i.cu" >/dev/null 2>>"$OUT/build.log" || { cat "$OUT/build.log" >&2; exit 1; }
  cuobjdump -sass "$WORK/b/fb_$i" >"$OUT/sass/fb_$i.sass" 2>&1
  bins+=("$WORK/b/fb_$i")
done
python3 "$PROBE" gate "${bins[@]}" --json "$OUT/gate-build.json" >"$OUT/gate-build.txt" 2>&1 \
  || { cat "$OUT/gate-build.txt" >&2; exit 1; }

cat >"$WORK/time.sh" <<EOF
rc=0
for b in ${bins[*]}; do
  python3 "$PROBE" run "\$b" --out "$OUT/chunks/\$(basename "\$b")" >"$OUT/chunks/\$(basename "\$b").log" 2>&1 || rc=1
done
exit \$rc
EOF
# a chunk times in about a minute: the lease is 2 minutes a chunk and 5 more, so a big batch isn't cut off mid-chunk
gpu-lease 1 --wait --max-min $(( ${#bins[@]} * 2 + 5 )) -- bash "$WORK/time.sh"
rc=$?
cat "$OUT"/chunks/*/rows.jsonl >"$OUT/rows.jsonl" 2>/dev/null
python3 "$HERE/loads.py" table "$OUT/rows.jsonl" >"$OUT/loads.jsonl"
python3 "$HERE/loads.py" summary "$OUT/rows.jsonl" | tee "$OUT/summary.txt"
exit $rc
