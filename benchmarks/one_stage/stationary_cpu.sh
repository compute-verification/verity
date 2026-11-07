#!/usr/bin/env bash
# Weights-stationary proof blocks against the per-expert cut's, proved on the node's CPUs: each ST_UNITS type of
# benchmarks/served_units/stage.py (st-w13:M=8, pe-w13, read, ...) staged at each of ST_COUNTS blocks a statement on one built
# circuit (stage.py's N1,N2), each statement proved by `flock-circuit prove --zk` (zk_cost.py prove --cpu, arm nofw), one warm
# session then ST_RUNS timed, every block drawn. Two counts a type give its seconds a block (the marginal) and a statement's
# fixed seconds; a type the prover refuses (past Flock's range) is logged as refused and the rest go on.
# Run from the verity tree in a proofs slot:
#   bash benchmarks/one_stage/stationary_cpu.sh ST_UNITS="pe-w13 st-w13:M=8" ST_COUNTS=2,8
# env: ST_UNITS (stage.py's type specs)  ST_COUNTS=2,8  ST_RUNS=2
# Kept in $OUT: per type and count its stage.json and the sessions' prove.out, prove.err and serve.log, and the records;
# never a circuit, a prover's file or a proof.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH PYTHONDONTWRITEBYTECODE=1
{ hostname; lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; free -g; git rev-parse HEAD; } > "$O/host.txt" 2>&1
log "host: $(nproc) CPUs, $(free -g | awk '/Mem:/{print $2}') GB"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
PY=$REPO/.venv/bin/python
mkdir -p "$O/m0-build"
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$W MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
  || { tail -40 "$O/m0-build/build.log"; exit 6; }
BIN=$W/flock-gpu-link/flock/target/release/flock-circuit
sha256sum "$BIN" | tee "$O/binaries.sha256" | tee -a "$O/job.log"
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/numerical/python
export RAYON_NUM_THREADS=$(nproc) RUST_MIN_STACK=$((64 << 20)) FC_PHASE_TIMES=1 FLOCK_PHASE_TIMES=1
Z=$W/zk; mkdir -p "$Z/records"; echo '[]' > "$Z/plan.json"
COUNTS=${ST_COUNTS:-2,8}; rc=0
keep() {
  (cd "$Z" && find . -type f -size -64M ! -name circuit.txt ! -name '*.bin' ! -path '*/sessions/*' -print0 \
    | xargs -0 -r -I{} cp --parents {} "$O/")
}
for u in ${ST_UNITS:?}; do
  t=$(echo "$u" | tr ':,=' '---' | tr -d '-' ); sd=$Z/$t; mkdir -p "$sd"
  log "$u: stage $COUNTS"
  for c in ${COUNTS//,/ }; do mkdir -p "$sd/c$c/stage"; done
  "$PY" benchmarks/served_units/stage.py "$u" "$COUNTS" "$sd/c{n}/stage" > "$sd/stage.log" 2>&1
  src=$?
  if [ $src != 0 ]; then log "$u: stage refused or failed (rc $src): $(grep -vE '^\s' "$sd/stage.log" | tail -c 600)"; keep; continue; fi
  pts=$("$PY" - "$Z" "$u" "$t" "$sd" "$COUNTS" 2>> "$O/job.log" <<'EOF'
import json, sys
from pathlib import Path
z, u, t, sd, counts = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), sys.argv[5]
plan, pts = json.loads((z / "plan.json").read_text()), []
for c in (int(x) for x in counts.split(",")):
    d = sd / f"c{c}"
    st = json.loads((d / "stage" / "stage.json").read_text())
    (d / f"draw-k{c}.json").write_text(json.dumps({"law": "subset", "population": c, "k": c, "units": list(range(c))}))
    plan.append({"unit": u, "tag": t, "c": c, "k": c, "arms": ["nofw"], "dir": str(d), "stage_dir": str(d / "stage"),
                 "pin": (d / "stage" / "pin.txt").read_text().strip(), "inst": str(d / "stage" / "inst.bin"),
                 "pub": str(d / "stage" / "pub.bin"), "partition": None})
    pts.append(str(len(plan) - 1))
    print({k: st.get(k) for k in ("n", "k_log", "ands", "unit_nnz", "macs", "maxrss_gb")}, st["seconds"], file=sys.stderr)
(z / "plan.json").write_text(json.dumps(plan, indent=1))
print(",".join(pts))
EOF
)
  [ -n "$pts" ] || { log "$u: no points"; rc=1; continue; }
  log "$u: staged; prove points $pts"
  for p in ${pts//,/ }; do
    "$PY" benchmarks/one_stage/zk_cost.py prove --cpu --bin "$BIN" --arms nofw --cold 0 --warm 1 --runs "${ST_RUNS:-2}" \
      --points "$p" --out "$Z" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST | tee -a "$O/job.log"
    [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  done
  keep
done
log "stationary: done, rc $rc"
exit "$rc"
