#!/usr/bin/env bash
# Pearl-C's served decode overhead over stock FP8, without hashing and with it, across a list of points (noise_sweep.py's
# summary), on an RTX PRO 6000 node: untimed preemptible screens (tier 2a), not panel rows.  From a run tree's root:
#   research run --on MACHINE --project verity --source <tree> --cwd source --declared-output 'out/*' \
#     --env GPU_LEASE_WHO=LANE --env UNTIL=<ISO time> --send points.txt -- \
#     bash benchmarks/pouw/pearl_c_vllm/noise_sweep.sh "$RESEARCH_RUN_DIR"/inputs/points.txt
# A points file has one point a line, its name then served_profile.sh's variables (BATCH, PROMPT, SCHEME, GRAPHS, ...):
#   eager-b32-p64 BATCH=32 PROMPT=64 SCHEME=pearl-c-sm120-v1-h3
# GPU 1's ship is built from this tree (CPU) once per kernel directory's git tree, kept in SHIPS under that tree's hash (the
# ship's sources are that directory; served_profile.sh checks its manifest and run.py), and reused.  Then each point's REPLICATES runs (each its own beacon, so its own seeds)
# go PAR at a time, each in its own one-GPU preemptible lease of at most LEASE_MIN minutes that ends by UNTIL (its --max-min,
# and a timeout inside it), served_profile.sh with NSYS=0 writing out/points/<point>.r<k>/; a failed run (a preemption) is
# tried once more as .t2.  A point with NSYS=1 (and NSYS_ARGS=--cuda-graph-trace=node for graphs) is profiled, and its
# served-profile.json is analyzed after the lease.  Each finished run is a line of out/progress.jsonl; out/sweep.{json,md} summarize what finished.
# SETUP=1 runs setup.sh first (CPU only).
set -uo pipefail
D=${RESEARCH_RUN_DIR:?run this under research run}; OUT=$D/out; mkdir -p "$OUT/points"
POINTS=${1:?a points file}; UNTIL=${UNTIL:?an absolute UTC end, e.g. 2026-10-06T13:58:00Z}
PAR=${PAR:-5} REPLICATES=${REPLICATES:-2} LEASE_MIN=${LEASE_MIN:-20} MIN_MIN=${MIN_MIN:-8}
export UNTIL OUT LEASE_MIN MIN_MIN
export WORK=${WORK:-/workspace/pouw/mvp-e2e}
S=$(mktemp -d /workspace/pouw/tmp-noise-sweep.XXXXXX); trap 'rm -rf "$S"' EXIT
HEAD=$(git rev-parse HEAD); echo "tree $HEAD" | tee "$OUT/tree.txt"; cp "$POINTS" "$OUT/points.txt"
stamp() { echo "noise-sweep: $* at $(date -u +%H:%M:%SZ)"; }

if [ "${SETUP:-0}" = 1 ]; then
  stamp "setup.sh"
  bash benchmarks/pouw/pearl_c_vllm/setup.sh > "$OUT/setup.log" 2>&1 || { stamp "SETUP_FAIL"; tail -20 "$OUT/setup.log"; exit 10; }
  tail -2 "$OUT/setup.log"
fi
KTREE=$(git rev-parse HEAD:benchmarks/pouw/pearl_c_sm120); SHIPS=${SHIPS:-/workspace/pouw/noise-sweep/ships}
export SHIP_TAR=$SHIPS/$KTREE.tar
if [ -f "$SHIP_TAR" ]; then
  stamp "ship of kernel tree $KTREE from $SHIP_TAR"
else
  stamp "ship build"
  CUDA=/usr/local/cuda-13.0 bash benchmarks/pouw/pearl_c_sm120/build.sh "$S/ship-${HEAD:0:8}" > "$OUT/build.log" 2>&1 \
    || { stamp "BUILD_FAIL"; tail -20 "$OUT/build.log"; exit 11; }
  mkdir -p "$SHIPS" && tar -cf "$S/ship.tar" -C "$S" "ship-${HEAD:0:8}" && mv "$S/ship.tar" "$SHIP_TAR.$$" && mv "$SHIP_TAR.$$" "$SHIP_TAR"
fi
tar -xOf "$SHIP_TAR" --wildcards '*/source.txt' > "$OUT/source.txt"; tar -xOf "$SHIP_TAR" --wildcards '*/sass.json' > "$OUT/sass.json"
echo "$KTREE $(sha256sum "$SHIP_TAR" | cut -c1-64)" > "$OUT/ship.txt"

one() {  # one run of one point: <dir name> <VAR=value>...
  local name=$1; shift
  local dir=$OUT/points/$name left max rc
  mkdir -p "$dir"
  left=$(( ($(date -u -d "$UNTIL" +%s) - $(date -u +%s)) / 60 ))
  if [ "$left" -lt "$MIN_MIN" ]; then echo "{\"point\": \"$name\", \"rc\": null, \"skipped\": \"$left min to $UNTIL\"}" >> "$OUT/progress.jsonl"; return 0; fi
  max=$(( left < LEASE_MIN ? left : LEASE_MIN ))
  env NSYS=0 "$@" OUT="$dir" gpu-lease 1 --wait --preemptible --max-min "$max" -- bash -c '
    left=$(( $(date -u -d "$UNTIL" +%s) - $(date -u +%s) )); [ "$left" -gt 240 ] || exit 75
    exec timeout --kill-after=30 "$left" bash benchmarks/pouw/pearl_c_vllm/served_profile.sh gpu' > "$dir/log" 2>&1
  rc=$?
  if [ -f "$dir/served.nsys-rep" ]; then   # a point with NSYS=1: its analysis after the lease, the report kept only if that fails
    nsys export --type sqlite --force-overwrite true -o "$dir/served.sqlite" "$dir/served.nsys-rep" >> "$dir/log" 2>&1 \
      && "$WORK/venv312/bin/python" benchmarks/pouw/pearl_c_vllm/served_profile.py --analyze "$dir/served.sqlite" \
           --out "$dir/served-profile.json" >> "$dir/log" 2>&1 \
      && rm -f "$dir/served.nsys-rep"
    rm -f "$dir/served.sqlite"
  fi
  echo "{\"point\": \"$name\", \"rc\": $rc, \"ended\": \"$(date -u +%FT%TZ)\", \"e2e\": $( [ -f "$dir/e2e.json" ] && echo true || echo false)}" >> "$OUT/progress.jsonl"
  return "$rc"
}
export -f one

stamp "points: $(grep -cv '^\s*\(#\|$\)' "$POINTS") x $REPLICATES replicates, $PAR at a time, leases to $UNTIL"
for k in $(seq 1 "$REPLICATES"); do
  grep -v '^\s*\(#\|$\)' "$POINTS" | while read -r name vars; do echo "$name.r$k $vars"; done
done > "$S/jobs.txt"
xargs -P "$PAR" -L 1 bash -c 'one "$0" "$@" || one "$0.t2" "$@" || true' < "$S/jobs.txt"

stamp "summary"
"$WORK/venv312/bin/python" benchmarks/pouw/pearl_c_vllm/noise_sweep.py "$OUT/points" --json "$OUT/sweep.json" --md "$OUT/sweep.md" --result
n=$(find "$OUT/points" -name e2e.json | wc -l); stamp "$n run(s) with an e2e.json"
[ "$n" -gt 0 ]
