#!/usr/bin/env bash
# Where a served proof block's CPU proving time goes: each SPLIT_UNITS type of benchmarks/served_units/stage.py, staged with
# SPLIT_N instances, proved by `flock-circuit prove --zk` on the node's CPUs (zk_cost.py prove --cpu, arm nofw: the host's
# witness, no send check), one warm session then SPLIT_RUNS timed, with the prover's phase timers on: the witness's laps by
# circuit (FC_PHASE_TIMES: CONVERGE, WITNESSCPU, the block's LAYOUT) and the CPU prover's phases (FLOCK_PHASE_TIMES:
# CPUPHASE, 60-circuit.sh's flock-ptime patch). prove_split.py reads them back into hashing and arithmetic.
# Run from the verity tree in a proofs slot:
#   bash benchmarks/one_stage/prove_split_cpu.sh SPLIT_UNITS="gq-k1536 fi-linear-k2048" SPLIT_N=16
# env: SPLIT_UNITS (stage.py's type names)  SPLIT_N=16  SPLIT_RUNS=2  SPLIT_ARMS=nofw (zk_cost.py's --arms; plain: no --zk)
# Kept in $OUT: per type its stage.json and the sessions' prove.out, prove.err and serve.log, the records, and split.json;
# never a circuit, a prover's file or a proof.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$PATH PYTHONDONTWRITEBYTECODE=1
{ hostname; lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; free -g; cat /sys/fs/cgroup/memory.max; git rev-parse HEAD; } > "$O/host.txt" 2>&1
log "host: $(nproc) CPUs, $(free -g | awk '/Mem:/{print $2}') GB, cgroup max $(cat /sys/fs/cgroup/memory.max 2>/dev/null)"
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
N=${SPLIT_N:-16}; rc=0
keep() {
  (cd "$Z" && find . -type f -size -64M ! -name circuit.txt ! -name '*.bin' ! -path '*/sessions/*' -print0 \
    | xargs -0 -r -I{} cp --parents {} "$O/")
}
for u in ${SPLIT_UNITS:?}; do
  sd=$Z/$u-c$N; mkdir -p "$sd"
  log "$u: stage $N"
  "$PY" benchmarks/served_units/stage.py "$u" "$N" "$sd/stage" > "$sd/stage.log" 2>&1 \
    || { log "$u: stage failed: $(tail -c 400 "$sd/stage.log")"; rc=1; keep; continue; }
  i=$("$PY" - "$Z" "$u" "$sd" "${SPLIT_ARMS:-nofw}" <<'EOF'
import json, sys
from pathlib import Path
z, u, sd, arms = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4].split(",")
st = json.loads((sd / "stage" / "stage.json").read_text())
c = st["n"]
(sd / f"draw-k{c}.json").write_text(json.dumps({"law": "subset", "population": c, "k": c, "units": list(range(c))}))
part = sd / "stage" / "partition.json"
plan = json.loads((z / "plan.json").read_text())
plan.append({"unit": u, "tag": u, "c": c, "k": c, "arms": arms, "dir": str(sd), "stage_dir": str(sd / "stage"),
             "pin": (sd / "stage" / "pin.txt").read_text().strip(), "inst": str(sd / "stage" / "inst.bin"),
             "pub": str(sd / "stage" / "pub.bin"), "partition": str(part) if part.exists() else None})
(z / "plan.json").write_text(json.dumps(plan, indent=1))
print(len(plan) - 1)
EOF
)
  log "$u: staged ($("$PY" -c 'import json, sys; s = json.load(open(sys.argv[1])); print({k: s.get(k) for k in ("n", "k_log", "vus_per_block", "ands", "maxrss_gb")}, s["seconds"]["total"], "s")' "$sd/stage/stage.json")); prove"
  "$PY" benchmarks/one_stage/zk_cost.py prove --cpu --bin "$BIN" --arms "${SPLIT_ARMS:-nofw}" --cold 0 --warm 1 --runs "${SPLIT_RUNS:-2}" \
    --points "$i" --out "$Z" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST | tee -a "$O/job.log"
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  keep
  "$PY" benchmarks/one_stage/prove_split.py "$O" > "$O/split.md" 2>> "$O/job.log" || true
done
log "prove-split: done, rc $rc"
exit "$rc"
