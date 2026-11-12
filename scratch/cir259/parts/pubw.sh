#!/usr/bin/env bash
# stage.py's --public-weights arms by the bound default and by --unbound, and origin/main's stage.py (inputs/stage_main.py) with
# no flag: the unbound circuits must be main's, and the bound ones stage. From the source tree's root, after `uv sync`.
set -uo pipefail
SRC=$(pwd); OUT=${RESEARCH_RUN_DIR:?}/out; IN=$RESEARCH_RUN_DIR/inputs; mkdir -p "$OUT"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
export PYTHONPATH=$SRC:$SRC/verity/protocols/flock/prover/python:$SRC/adapters/vllm:$SRC/tools/verity/circuit_check/src
PYM=$SRC/.venv/bin/python
UNIT=${PW_UNIT:-fp8-coord:K=256}
echo "source ${RESEARCH_SOURCE_SHA:-?}; $UNIT N=2" | tee "$OUT/job.log"
for arm in public packed; do
  flags=--public-weights; [ $arm = packed ] && flags="--public-weights --packed"
  for way in B U M; do
    d=$W/$arm/$way; mkdir -p "$d" "$OUT/$arm/$way"
    case $way in
      B) "$PYM" benchmarks/served_units/stage.py "$UNIT" 2 "$d" $flags ;;
      U) "$PYM" benchmarks/served_units/stage.py "$UNIT" 2 "$d" $flags --unbound ;;
      M) "$PYM" "$IN/stage_main.py" "$UNIT" 2 "$d" $flags ;;
    esac > "$OUT/$arm/$way/stage.log" 2>&1
    rc=$?
    (cd "$d" && sha512sum circuit.txt inst.bin pub.bin partition.json program.json 2>/dev/null) > "$OUT/$arm/$way/files.sha512"
    cp "$d/stage.json" "$d/pin.txt" "$OUT/$arm/$way/" 2>/dev/null
    echo "$arm/$way: rc $rc; pin $(cut -c1-16 "$d/pin.txt" 2>/dev/null); files $(cut -c1-16,129- "$OUT/$arm/$way/files.sha512" | tr '\n' ' '); $(tail -1 "$OUT/$arm/$way/stage.log" | cut -c1-300)" | tee -a "$OUT/job.log"
  done
  u=$(sort "$OUT/$arm/U/files.sha512"); m=$(sort "$OUT/$arm/M/files.sha512")
  echo "PUBW $arm: unbound = main's: $([ -n "$u" ] && [ "$u" = "$m" ] && echo yes || echo NO); bound staged with partition and program: $(grep -c 'partition.json\|program.json' "$OUT/$arm/B/files.sha512")/2" | tee -a "$OUT/job.log"
done
