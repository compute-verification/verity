#!/usr/bin/env bash
# C-Flock's zero-knowledge session cost on a CPU node (zk_cost.py --cpu): zk_cost.sh's sweep of W8A8 coordinates in statements
# of c units, each session proving k sampled by Lean, by `flock-circuit prove --zk` on the node's CPUs with the ZK server's send
# check, then Lean `verify --zk` on every session; what one drawn statement costs a CPU node, for sizing a block's sample.
# The prover is built as prove_path/run.sh builds it (60-circuit.sh GPU=0), Lean's flock-verify and flock-firewall in the tree.
# Run from the verity tree in a proofs slot (one point at a time, so a session's peak is its own):
#   research run --on vy-nebius-cpu-3 --pool proofs --project verity --source . --cwd clone \
#     -- bash benchmarks/one_stage/zk_cost_cpu.sh ZK_KS=4096,16384 ZK_SIZES=1,8 ZK_DRAWS=1,c
# env: ZK_KS=4096,16384  ZK_SIZES=1,8  ZK_DRAWS=1,c  ZK_ARMS=fw  ZK_COLD=1  ZK_RUNS=0 (no warm process)
# Kept in $OUT: the records (stage, prove and verify .jsonl; each prove record's peak_rss_kb is the session's summed peak), the
# fit and the logs; never a circuit, a prover's file or a proof.
for kv in "$@"; do case $kv in *=*) export "$kv" ;; esac; done
set -uo pipefail
REPO=$(pwd); O=${OUT:-${RESEARCH_RUN_DIR:?}/out}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
log() { echo "[$(date -u +%H:%M:%SZ)] $*" | tee -a "$O/job.log"; }
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH PYTHONDONTWRITEBYTECODE=1
{ lscpu | grep -E 'Model name|^CPU\(s\)'; nproc; free -g; git rev-parse HEAD; } > "$O/host.txt" 2>&1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync -q --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
PY=$REPO/.venv/bin/python

log "zk-cost-cpu: source $(git rev-parse HEAD), builds in $W"
(cd verity/core/service/flock && lake build flock-verify flock-firewall) > "$O/build-lean.log" 2>&1 &
LP=$!
mkdir -p "$O/m0-build"
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$W MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
  || { tail -40 "$O/m0-build/build.log"; exit 6; }
BIN=$W/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 "$O/build-lean.log"; exit 7; }
LEAN=$REPO/verity/core/service/flock/.lake/build/bin/flock-verify
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
sha256sum "$BIN" "$LEAN" "$FC_FIREWALL_LEAN" | tee "$O/binaries.sha256" | tee -a "$O/job.log"
export PYTHONPATH=$REPO:$REPO/verity/ml/flock/python:$REPO/integrations/vllm:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/numerical/python
export RAYON_NUM_THREADS=$(nproc)
# flock-circuit's rayon pool keeps Rust's default 2 MiB thread stack, which a --zk prover at m=27 can overflow on a many-core
# node (prove_path/run.sh)
export RUST_MIN_STACK=$((64 << 20))
Z=$W/zk
ZK() { "$PY" benchmarks/one_stage/zk_cost.py "$@" --out "$Z"; }

log "stage: K ${ZK_KS:-4096,16384}, sizes ${ZK_SIZES:-1,8}, draws ${ZK_DRAWS:-1,c}; $RAYON_NUM_THREADS threads"
ZK stage --lean "$LEAN" --ks "${ZK_KS:-4096,16384}" --sizes "${ZK_SIZES:-1,8}" --draws "${ZK_DRAWS:-1,c}" 2>&1 \
  | tee -a "$O/zk.log" | grep ZKCOST
rc=${PIPESTATUS[0]}
if [ "$rc" = 0 ]; then
  log "prove: arms ${ZK_ARMS:-fw}, ${ZK_COLD:-1} cold, ${ZK_RUNS:-0} warm sessions"
  ZK prove --cpu --bin "$BIN" --arms "${ZK_ARMS:-fw}" --cold "${ZK_COLD:-1}" --warm 0 --runs "${ZK_RUNS:-0}" 2>&1 \
    | tee -a "$O/zk.log" | grep ZKCOST
  rc=${PIPESTATUS[0]}
  log "prove: exit $rc; verify"
  ZK verify --lean "$LEAN" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
  [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  ZK fit 2>&1 | tee -a "$O/zk.log"
fi
(cd "$Z" && find . -type f -size -8M ! -name circuit.txt ! -name 'inst-*' ! -name 'pub-*' ! -path '*/archive/*' \
  \( ! -path '*/sessions/*/*' -o -name session.json \) -print0 | xargs -0 -r -I{} cp --parents {} "$O/")
log "zk-cost-cpu: done, rc $rc"
exit "$rc"
