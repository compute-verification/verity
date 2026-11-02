#!/usr/bin/env bash
# The scorer's final-split scoring, on one leased GPU, run in an entry's own commit (research run --source --cwd source),
# this file sent with --send:
#
#   final.sh --starts S...              each start's own bpb on the search and the final set -> $OUT/start.<S>.<split>.json
#   final.sh CLASS START [TRAIN_ARGS]   one rerun of an entry: islands/CLASS/train.py under guard.py, timed and capped as
#                                       candidate.sh does, then evaluate.py on the search set and on the final set
#                                       (witness.search.json, witness.final.json, each with its curve) and the start on both
#
# $OUT also gets timing.json, the submission's meta.json and steps.jsonl, and the logs. The submission (with the model) lives
# in a scratch directory removed at exit. The fixed side (guard.py, evaluate.py, prepare.py) is the commit's own.
set -u
cw=$PWD/benchmarks/covert_witness
P=${CW_PYTHON:-/workspace/jobs/venv312/bin/python}
R=${CW_ROOT:-/workspace/research/scratch/covert-witness}
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-training:scorer-final}
[ -f "$cw/evaluate.py" ] || { echo "final: no $cw/evaluate.py (run with --cwd source)"; exit 2; }

starts() {
  for s in "$@"; do
    for split in search final; do
      "$P" "$cw/evaluate.py" --start "$s" --split "$split" --out "$OUT/start.$s.$split.json" >> "$OUT/evaluate.start.log" 2>&1 \
        || { echo "final: start $s on $split failed"; return 1; }
    done
  done
}

if [ "${1:-}" = --leased-starts ]; then
  shift
  starts "$@"
  exit $?
fi

if [ "${1:-}" = --leased ]; then
  shift
  klass=$1 start=$2 S=$3
  shift 3
  cap=$("$P" -c "import sys; sys.path.insert(0, '$cw'); import prepare; print(int(prepare.GPU_SECONDS_CAP))")
  t0=$(date +%s.%N)
  CW_GUARD_LOG="$OUT/guard.log" timeout --signal=TERM --kill-after=30 $((cap + 60)) \
    "$P" "$cw/guard.py" "$cw/islands/$klass/train.py" --start "$start" --out "$S/sub" "$@" > "$OUT/train.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  echo "{\"gpu_seconds\": $("$P" -c "print(round($t1 - $t0, 3))"), \"rc\": $rc, \"gpu\": \"$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null)\"}" > "$S/timing.json"
  cp "$S/timing.json" "$OUT/timing.json"
  [ -f "$S/sub/meta.json" ] || { echo "final: train.py wrote no submission (rc $rc)"; exit 1; }
  erc=0
  for split in search final; do
    "$P" "$cw/evaluate.py" --submission "$S/sub" --class "$klass" --timing "$S/timing.json" --split "$split" \
      --out "$OUT/witness.$split.json" > "$OUT/evaluate.$split.log" 2>&1 || erc=$?
    tail -1 "$OUT/evaluate.$split.log"
  done
  starts "$start" || erc=1
  exit $erc
fi

mkdir -p "$R/candidates" "$OUT"
if [ "${1:-}" = --starts ]; then
  shift
  gpu-lease 1 --wait --max-min 30 -- bash "$0" --leased-starts "$@"
  exit $?
fi
klass=$1 start=$2
shift 2
[ -f "$cw/islands/$klass/train.py" ] || { echo "final: no island $klass"; exit 2; }
S=$(mktemp -d -p "$R/candidates" "final-$klass.XXXXXX")
trap 'rm -rf "$S"' EXIT
cp "$cw/islands/$klass/train.py" "$OUT/train.py"
echo "{\"class\": \"$klass\", \"start\": \"$start\", \"args\": \"$*\", \"commit\": \"${RESEARCH_SOURCE_SHA:-}\", \"split\": \"final\"}" > "$OUT/candidate.json"
gpu-lease 1 --wait --max-min 30 -- bash "$0" --leased "$klass" "$start" "$S" "$@"
rc=$?
for f in meta.json steps.jsonl; do [ -f "$S/sub/$f" ] && cp "$S/sub/$f" "$OUT/"; done
[ -f "$S/sub/E.bin" ] && sha256sum "$S/sub/E.bin" | cut -d' ' -f1 > "$OUT/E.sha256"
exit $rc
