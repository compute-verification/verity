#!/usr/bin/env bash
# The recursion bench over an inner statement rec_inner.py stages rather than 85's build (85's STAGE=0): N seeded draws of
# SUBCIRCUIT (`rec_inner.py stage --subcircuit`), or with WINDOW every unit of a served window's member MEMBER
# (`rec_inner.py stage-window`: prove_path.py commit's window dir, the prover's files included). Both write
# $FLOCK_WORK/data/$TAG/stage (plain leaves, the recursive inner) and stage-zk (hm96, today's --zk on the same units) with
# the partition, program and full draw beside them (own-partition), then run job.sh on them with STAGE=0 and N the
# statement's instances. Every other KEY=VALUE goes on to job.sh. Run from the verity tree, as job.sh is:
#   research run --on vy-nebius-cpu-2 --project verity --cwd source --mem-gb 128 --timeout 600m -- \
#     taskset -c 0-31 bash benchmarks/recursion_e2e/stage.sh SUBCIRCUIT=silu-mul-v2/i8192/bf16 N=3 GPU=0 \
#     FLOCK_WORK=/workspace/cache/rec-bench STEPS=build,inner,icheck,vstage,vcap,oprove,oreplay,lean,negative
# env: SUBCIRCUIT (a census-style id)  N=3 (seeded draws; WINDOW: the member's own n)  SEED (rec_inner.py's)
#      WINDOW=DIR MEMBER=0  FLOCK_WORK=/workspace/jobs/rec-bench  TAG=rb-<run id>
#      SERVE_RUN=DIR SERVE_WINDOW=0 (with WINDOW: the window's run root must be the one DIR/windows.jsonl published for it)
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
: "${SUBCIRCUIT:?SUBCIRCUIT=<census-style subcircuit id>}"
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:-/tmp/rec-stage-$$}/out}; mkdir -p "$O"
export FLOCK_WORK=${FLOCK_WORK:-/workspace/jobs/rec-bench} TAG=${TAG:-rb-${RESEARCH_RUN_ID:-$$}}
DD=$FLOCK_WORK/data/$TAG W=$FLOCK_WORK/flock-circuit
mkdir -p "$DD" "$W" || exit 1
[ -w "$HOME" ] || { export HOME=$FLOCK_WORK/home; mkdir -p "$HOME"; }
# 85's own Python (its STEP=build makes the same venv, and keeps it)
UV=$(command -v uv || echo $HOME/.local/bin/uv)
[ -x $UV ] || { curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; UV=$HOME/.local/bin/uv; }
[ -x $W/py/bin/python ] || { $UV venv -q --python 3.12 $W/py && $UV pip install -q --python $W/py/bin/python numpy blake3; } || exit 1
PY=$W/py/bin/python
export PYTHONPATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
S=$DD/inputs
rm -rf "$DD/stage" "$DD/stage-zk" "$S"
if [ -n "${WINDOW:-}" ]; then
  if [ -n "${SERVE_RUN:-}" ]; then   # the window's rows the served run's: its run root the one windows.jsonl published
    $PY -c 'import json, sys
w = json.load(open(sys.argv[1] + "/window.json"))
pub = [x for x in map(json.loads, open(sys.argv[2] + "/windows.jsonl")) if int(x["window"]) == int(sys.argv[3])]
ok = len(pub) == 1 and pub[0]["run_root"] == w["run"]
print(json.dumps({"served_root_held": ok, "window_run_root": w["run"], "published": [x["run_root"] for x in pub],
                  "registration_digest": w["registration_digest"]}))
sys.exit(0 if ok else 1)' "$WINDOW" "$SERVE_RUN" "${SERVE_WINDOW:-0}" | tee "$O/served-root.json" || exit 1
  fi
  /usr/bin/time -v -o "$O/time-stage.txt" $PY verity/ml/flock/pod/rec_inner.py stage-window --window "$WINDOW" \
    --member "${MEMBER:-0}" --subcircuit "$SUBCIRCUIT" --sets "$S/sets" --out "$DD/stage" --zk-out "$DD/stage-zk" \
    > "$O/stage.json" 2> "$O/stage.err" || { tail -20 "$O/stage.err"; exit 1; }
else
  N=${N:-3}
  for d in stage stage-zk; do
    /usr/bin/time -v -o "$O/time-$d.txt" $PY verity/ml/flock/pod/rec_inner.py stage --n "$N" --subcircuit "$SUBCIRCUIT" \
      ${SEED:+--seed $SEED} --sets "$S/sets" --out "$DD/$d" $([ $d = stage ] && echo --plain-leaves) \
      > "$O/$d.json" 2> "$O/$d.err" || { tail -20 "$O/$d.err"; exit 1; }
  done
fi
touch "$DD/stage/own-partition"
N=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["n"])' "$O/stage.json") || exit 1
cat "$O/stage.json"
# job.sh removes data/$TAG at its end unless KEEP_DATA=1
exec bash benchmarks/recursion_e2e/job.sh "$@" STAGE=0 N=$N TAG=$TAG FLOCK_WORK=$FLOCK_WORK
