#!/usr/bin/env bash
# The covert sender's rate through the online prover's declared sync (covert.py) on the injected clock: active_replay.py
# over every configuration of every trace, per sync policy, in parallel, then covert.py pools each policy's summaries.
# From the shipped tree's root, with the traces' tarball and the calibration sent as inputs:
#   research run --on MACHINE --project verity --source <tree> --cwd source --campaign CAMPAIGN \
#     --tool benchmarks.network_traces.tool:COVERT_SYNC --declared-output 'out/*' \
#     --send traces.tar.gz --send calibration.json [--env JOBS=32] -- \
#     bash benchmarks/network_traces/covert.sh last cover-3-2 envelope
# A fixed policy (`envelope`, `model`) is the control: the same sender steers nothing.
set -euo pipefail
OUT=${OUT:-${RESEARCH_RUN_DIR:?run this under research run}/out}
IN=${IN:-$RESEARCH_RUN_DIR/inputs}
mkdir -p "$OUT"
VENV=${VENV:-/workspace/jobs/venv312}
export PYTHONPATH=$PWD${PYTHONPATH:+:$PYTHONPATH}
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
tar xzf "$IN/traces.tar.gz" -C "$d"
cp "$IN/calibration.json" "$OUT/calibration.json"
one() {  # policy trace
  local n
  n=$(basename "$2" .json)
  mkdir -p "$OUT/$1/$n"
  "$VENV/bin/python" benchmarks/network_traces/active_replay.py --clock virtual --syncs "$1" \
    --strategies "$([ "$1" = envelope ] || [ "$1" = model ] && echo honest,covert || echo honest,covert,bits)" \
    --seed "${SEED:-0}" --windows 100000 --config-windows 1:100000 --calibration "$IN/calibration.json" \
    --out "$d/$1-$n" "$2" > "$OUT/$1/$n/run.log" 2>&1 || echo "exit $?" >> "$OUT/$1/$n/run.log"
  cp "$d/$1-$n"/{summary.json,verdicts.jsonl,selection.json} "$OUT/$1/$n/" 2>/dev/null || true
  rm -rf "${d:?}/$1-$n"
}
export -f one
export IN OUT VENV d
for policy in "$@"; do
  for t in "$d"/traces/r*-????.json; do echo "$policy $t"; done
done | xargs -P "${JOBS:-16}" -L 1 bash -c 'one "$0" "$1"'
for policy in "$@"; do
  "$VENV/bin/python" benchmarks/network_traces/covert.py --calibration "$IN/calibration.json" \
    --out "$OUT/pooled-$policy.json" "$OUT/$policy"/*/summary.json | tee "$OUT/pooled-$policy.txt"
done
