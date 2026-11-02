#!/usr/bin/env bash
# candidate.sh CLASS START [TRAIN_ARGS...]: one candidate on one leased GPU (fixed). islands/CLASS/train.py runs inside the
# harness (`python -m te_core.run --script`), which counts its MACs, times its steps, keeps the held-out sets unreadable from
# it and stops it at 1.2x GPU_SECONDS_CAP; then evaluate.py checks the witnesses against the harness's record and scores the
# submission. $OUT gets run/ (the harness's run.json and guard.log), witness.json, timing.json, the submission's meta.json and
# steps.jsonl, E.bin when it is at most 1 MB (else its sha256), the candidate's train.py and its logs. The submission (with
# the model) lives in a scratch directory removed at exit.
set -u
here=$(cd "$(dirname "$0")" && pwd)
P=${CW_PYTHON:-/workspace/jobs/venv312/bin/python}
R=${CW_ROOT:-/workspace/research/scratch/covert-witness}
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-training:covert-witness}
# the seconds cap, the process's hard limit (1.2x the cap, plus loading, the harness's evaluations and the family checks) and
# the GPU lease's, in minutes
read -r cap kill lease <<< "$("$P" -c "import sys; sys.path.insert(0, '$here'); import prepare; c = prepare.GPU_SECONDS_CAP
print(int(c), int(1.2 * c + 900), int((1.2 * c + 900) / 60 + 10))")"

if [ "${1:-}" = --leased ]; then
  shift
  klass=$1 start=$2 S=$3
  shift 3
  fraction=0.98 prev=
  for x in "$@"; do
    case "$x" in --budget-fraction=*) fraction=${x#*=} ;; esac
    [ "$prev" = --budget-fraction ] && fraction=$x
    prev=$x
  done
  meta="{\"class\": \"$klass\", \"start\": \"$start\", \"launcher\": \"covert_witness/candidate.sh\"}"
  t0=$(date +%s.%N)
  (cd "$here" && timeout --signal=TERM --kill-after=30 "$kill" "$P" -m te_core.run --script "islands/$klass/train.py" \
    --out "$OUT/run" --families "$S/families" --budget-fraction "$fraction" \
    --budget-minutes "$("$P" -c "print($cap / 60)")" --meta "$meta" \
    -- --start "$start" --out "$S/sub" "$@") > "$OUT/train.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  echo "{\"process_seconds\": $("$P" -c "print(round($t1 - $t0, 3))"), \"rc\": $rc, \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null | head -1)\"}" > "$S/timing.json"
  cp "$S/timing.json" "$OUT/timing.json"
  [ -f "$S/sub/meta.json" ] || { echo "candidate: train.py wrote no submission (rc $rc)"; tail -3 "$OUT/train.log"; exit 1; }
  [ -f "$OUT/run/run.json" ] || { echo "candidate: the harness wrote no run.json (rc $rc)"; tail -3 "$OUT/train.log"; exit 1; }
  "$P" "$here/evaluate.py" --submission "$S/sub" --class "$klass" --timing "$S/timing.json" --run "$OUT/run/run.json" \
    --out "$OUT/witness.json" > "$OUT/evaluate.log" 2>&1
  erc=$?
  tail -1 "$OUT/evaluate.log"
  exit $erc
fi

klass=$1 start=$2
shift 2
[ -f "$here/islands/$klass/train.py" ] || { echo "candidate: no island $klass"; exit 2; }
mkdir -p "$R/candidates" "$OUT"
S=$(mktemp -d -p "$R/candidates" "$klass.XXXXXX")
trap 'rm -rf "$S"' EXIT
cp "$here/islands/$klass/train.py" "$OUT/train.py"
echo "{\"class\": \"$klass\", \"start\": \"$start\", \"args\": \"$*\", \"commit\": \"${RESEARCH_SOURCE_SHA:-}\"}" > "$OUT/candidate.json"
gpu-lease 1 --wait --max-min "$lease" -- bash "$0" --leased "$klass" "$start" "$S" "$@"
rc=$?
for f in meta.json steps.jsonl; do [ -f "$S/sub/$f" ] && cp "$S/sub/$f" "$OUT/"; done
if [ -f "$S/sub/E.bin" ]; then
  if [ "$(stat -c %s "$S/sub/E.bin")" -le 1000000 ]; then cp "$S/sub/E.bin" "$OUT/E.bin"; fi
  sha256sum "$S/sub/E.bin" | cut -d' ' -f1 > "$OUT/E.sha256"
fi
exit $rc
