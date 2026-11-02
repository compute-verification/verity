#!/usr/bin/env bash
# The scorer's final-split scoring, on one leased GPU, run in an entry's own commit (research run --source --cwd source),
# this file sent with --send:
#
#   final.sh --starts S...       each start's own bpb on the search and the final set -> $OUT/start.<S>.<split>.json
#   final.sh [--seed N] [--direction D] [--mechanism M] [--reference] CLASS START [TRAIN_ARGS]
#                                one rerun of an entry, as its commit's candidate.sh runs it, then the start on both sets.
#                                A covert_witness v2.1 commit (candidate.sh runs te_core.run): islands/CLASS/train.py inside
#                                the harness at tier 2 and seed N, which scores every checkpoint on the search and the final
#                                set (run/run.json); evaluate.py --run on both sets (witness.search.json, witness.final.json);
#                                the run ingested into a ledger of its own ($OUT/ledger, agent scorer-final, not published)
#                                as direction D, mechanism M. An older commit: train.py under guard.py, then evaluate.py on
#                                both sets.
#
# $OUT also gets timing.json, the submission's meta.json and steps.jsonl, and the logs. The submission (with the model) lives
# in a scratch directory removed at exit. The fixed side (te_core, evaluate.py, prepare.py) is the commit's own.
set -u
cw=$PWD/benchmarks/covert_witness
P=${CW_PYTHON:-/workspace/jobs/venv312/bin/python}
R=${CW_ROOT:-/workspace/research/scratch/covert-witness}
export GPU_LEASE_WHO=${GPU_LEASE_WHO:-training:scorer-final}
[ -f "$cw/evaluate.py" ] || { echo "final: no $cw/evaluate.py (run with --cwd source)"; exit 2; }
harness=0
grep -q "te_core.run" "$cw/candidate.sh" && harness=1

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
  seed=$1 direction=$2 mechanism=$3 reference=$4 klass=$5 start=$6 S=$7
  shift 7
  if [ "$harness" = 1 ]; then
    read -r cap kill <<< "$("$P" -c "import sys; sys.path.insert(0, '$cw'); import prepare; c = prepare.GPU_SECONDS_CAP
print(int(c), int(1.2 * c + 900))")"
    fraction=0.98 prev=
    for x in "$@"; do
      case "$x" in --budget-fraction=*) fraction=${x#*=} ;; esac
      [ "$prev" = --budget-fraction ] && fraction=$x
      prev=$x
    done
    meta="{\"class\": \"$klass\", \"start\": \"$start\", \"launcher\": \"covert_witness/scorer/final.sh\"}"
    t0=$(date +%s.%N)
    (cd "$cw" && timeout --signal=TERM --kill-after=30 "$kill" "$P" -m te_core.run --script "islands/$klass/train.py" \
      --out "$OUT/run" --families "$S/families" --budget-fraction "$fraction" --tier 2 --seed "$seed" \
      --budget-minutes "$("$P" -c "print($cap / 60)")" --meta "$meta" \
      -- --start "$start" --out "$S/sub" "$@") > "$OUT/train.log" 2>&1
    rc=$?
  else
    cap=$("$P" -c "import sys; sys.path.insert(0, '$cw'); import prepare; print(int(prepare.GPU_SECONDS_CAP))")
    t0=$(date +%s.%N)
    CW_GUARD_LOG="$OUT/guard.log" timeout --signal=TERM --kill-after=30 $((cap + 60)) \
      "$P" "$cw/guard.py" "$cw/islands/$klass/train.py" --start "$start" --out "$S/sub" "$@" > "$OUT/train.log" 2>&1
    rc=$?
  fi
  t1=$(date +%s.%N)
  secs=$("$P" -c "print(round($t1 - $t0, 3))")
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null | head -1)
  echo "{\"gpu_seconds\": $secs, \"process_seconds\": $secs, \"rc\": $rc, \"gpu\": \"$gpu\"}" > "$S/timing.json"
  cp "$S/timing.json" "$OUT/timing.json"
  [ -f "$S/sub/meta.json" ] || { echo "final: train.py wrote no submission (rc $rc)"; tail -3 "$OUT/train.log"; exit 1; }
  run=()
  if [ "$harness" = 1 ]; then
    [ -f "$OUT/run/run.json" ] || { echo "final: the harness wrote no run.json (rc $rc)"; tail -3 "$OUT/train.log"; exit 1; }
    run=(--run "$OUT/run/run.json")
  fi
  erc=0
  for split in search final; do
    "$P" "$cw/evaluate.py" --submission "$S/sub" --class "$klass" --timing "$S/timing.json" "${run[@]}" --split "$split" \
      --out "$OUT/witness.$split.json" > "$OUT/evaluate.$split.log" 2>&1 || erc=$?
    tail -1 "$OUT/evaluate.$split.log"
  done
  if [ "$harness" = 1 ]; then
    ref=()
    [ "$reference" = 1 ] && ref=(--reference)
    (cd "$cw" && "$P" -m te_core.ledger --root "$OUT/ledger" --agent scorer-final ingest "$OUT/run" \
      --direction "$direction" --mechanism "$mechanism" --run-id "${RESEARCH_RUN_ID:-}" "${ref[@]}") > "$OUT/ingest.log" 2>&1 \
      || { echo "final: ledger ingest failed"; tail -3 "$OUT/ingest.log"; erc=1; }
  fi
  starts "$start" || erc=1
  exit $erc
fi

mkdir -p "$R/candidates" "$OUT"
if [ "${1:-}" = --starts ]; then
  shift
  gpu-lease 1 --wait --max-min 30 -- bash "$0" --leased-starts "$@"
  exit $?
fi
seed=0 direction=scorer-final mechanism=rerun reference=0
while [ $# -gt 0 ]; do
  case "$1" in
    --seed) seed=$2; shift 2 ;;
    --direction) direction=$2; shift 2 ;;
    --mechanism) mechanism=$2; shift 2 ;;
    --reference) reference=1; shift ;;
    *) break ;;
  esac
done
klass=$1 start=$2
shift 2
[ -f "$cw/islands/$klass/train.py" ] || { echo "final: no island $klass"; exit 2; }
S=$(mktemp -d -p "$R/candidates" "final-$klass.XXXXXX")
trap 'rm -rf "$S"' EXIT
cp "$cw/islands/$klass/train.py" "$OUT/train.py"
echo "{\"class\": \"$klass\", \"start\": \"$start\", \"args\": \"$*\", \"commit\": \"${RESEARCH_SOURCE_SHA:-}\", \"split\": \"final\", \"seed\": $seed, \"harness\": $harness}" > "$OUT/candidate.json"
lease=$("$P" -c "import sys; sys.path.insert(0, '$cw'); import prepare; print(int((1.2 * prepare.GPU_SECONDS_CAP + 900) / 60 + 25))")
gpu-lease 1 --wait --max-min "$lease" -- bash "$0" --leased "$seed" "$direction" "$mechanism" "$reference" "$klass" "$start" "$S" "$@"
rc=$?
for f in meta.json steps.jsonl; do [ -f "$S/sub/$f" ] && cp "$S/sub/$f" "$OUT/"; done
if [ -f "$S/sub/E.bin" ]; then
  if [ "$(stat -c %s "$S/sub/E.bin")" -le 1000000 ]; then cp "$S/sub/E.bin" "$OUT/E.bin"; fi
  sha256sum "$S/sub/E.bin" | cut -d' ' -f1 > "$OUT/E.sha256"
fi
exit $rc
