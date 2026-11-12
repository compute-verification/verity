#!/usr/bin/env bash
# Flock's zero-knowledge session cost on a CPU node (zk_cost.py --cpu): zk_cost.sh's sweep of W8A8 coordinates in statements
# of c units, each session proving k sampled by Lean, by `flock-circuit prove --zk` on the node's CPUs with the ZK server's send
# check, then Lean `verify --zk` on every session; what one sampled statement costs a CPU node, for sizing a block's sample.
# The prover is built as prove_path/run.sh builds it (60-circuit.sh GPU=0), the Lean verifier and ZK server binaries in the tree,
# and flock-read-tmpl, the template a table read (stage.py's `table-read`) is staged on; with ZK_UNITS, flock-tmpl-unit in a
# copy of the package (hidden_zk/job.sh's build_tmpl_unit), which places a templated unit's rows for its prover.
# Run from the verity tree in a proofs slot (one point at a time, so a session's peak is its own):
#   bash benchmarks/one_stage/zk_cost_cpu.sh ZK_KS=4096,16384 ZK_SIZES=1,8 ZK_DRAWS=1,c
#   (a job run from a clone of the tree)
# env: ZK_KS=4096,16384  ZK_SIZES=1,8  ZK_DRAWS=1,c  ZK_ARMS=fw  ZK_COLD=1  ZK_WARM=0  ZK_RUNS=0 (no warm process)
#      ZK_UNITS="gq:K=384" (served or census types instead of A6's coordinates, zk_cost.py's --units; overhead --card none
#      instead of fit: each session's parts fitted over its sizes)
#      ZK_PASSES=1 (the plan this many times, each type's sizes in ABBA order: ZK_PASSES=2 proves c=2, 64, 64, 2)
#      ZK_VERIFY=0 (no Lean verify: the sessions' prices alone, each accepted by its ZK server)
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
TU=$W/tmpl-unit
{ (cd verity/protocols/flock/verifier && lake build flock-verify flock-firewall flock-read-tmpl) && if [ -n "${ZK_UNITS:-}" ]; then
    mkdir -p "$TU" && (cd verity/protocols/flock/verifier && tar --exclude=.lake -cf - .) | tar -xf - -C "$TU" \
      && cp verity/protocols/flock/prover/tests/lean/TmplUnit.lean "$TU/" \
      && printf '\n[[lean_exe]]\nname = "flock-tmpl-unit"\nroot = "TmplUnit"\n' >> "$TU/lakefile.toml" \
      && (cd "$TU" && lake build flock-tmpl-unit); fi; } > "$O/build-lean.log" 2>&1 &
LP=$!
mkdir -p "$O/m0-build"
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$W MODE=build GPU=0 bash verity/protocols/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
  || { tail -40 "$O/m0-build/build.log"; exit 6; }
BIN=$W/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 "$O/build-lean.log"; exit 7; }
LEAN=$REPO/verity/protocols/flock/verifier/.lake/build/bin/flock-verify
export FC_FIREWALL_LEAN=${LEAN%/*}/flock-firewall
sha256sum "$BIN" "$LEAN" "$FC_FIREWALL_LEAN" | tee "$O/binaries.sha256" | tee -a "$O/job.log"
export PYTHONPATH=$REPO:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm:$REPO/tools/verity/circuit_check/src:$REPO/benchmarks/numerical/python
export RAYON_NUM_THREADS=$(nproc)
# flock-circuit's rayon pool keeps Rust's default 2 MiB thread stack, which a --zk prover at m=27 can overflow on a many-core
# node (prove_path/run.sh)
export RUST_MIN_STACK=$((64 << 20))
Z=$W/zk
ZK() { "$PY" benchmarks/one_stage/zk_cost.py "$@" --out "$Z"; }

if [ -n "${ZK_UNITS:-}" ]; then
  log "stage: units $ZK_UNITS, sizes ${ZK_SIZES:-1,8}, samples ${ZK_DRAWS:-1,c}; $RAYON_NUM_THREADS threads"
  ZK stage --units "$ZK_UNITS" --sizes "${ZK_SIZES:-1,8}" --draws "${ZK_DRAWS:-1,c}" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
else
  log "stage: K ${ZK_KS:-4096,16384}, sizes ${ZK_SIZES:-1,8}, samples ${ZK_DRAWS:-1,c}; $RAYON_NUM_THREADS threads"
  ZK stage --lean "$LEAN" --ks "${ZK_KS:-4096,16384}" --sizes "${ZK_SIZES:-1,8}" --draws "${ZK_DRAWS:-1,c}" 2>&1 \
    | tee -a "$O/zk.log" | grep ZKCOST
fi
rc=${PIPESTATUS[0]}
if [ "$rc" = 0 ]; then
  log "prove: arms ${ZK_ARMS:-fw}, ${ZK_COLD:-1} cold, ${ZK_WARM:-0}+${ZK_RUNS:-0} warm sessions, ${ZK_PASSES:-1} passes"
  ZK prove --cpu --bin "$BIN" --arms "${ZK_ARMS:-fw}" --cold "${ZK_COLD:-1}" --warm "${ZK_WARM:-0}" --runs "${ZK_RUNS:-0}" \
    --passes "${ZK_PASSES:-1}" --tmpl-unit-bin "$TU/.lake/build/bin/flock-tmpl-unit" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
  rc=${PIPESTATUS[0]}
  if [ "${ZK_VERIFY:-1}" = 1 ]; then
    log "prove: exit $rc; verify"
    ZK verify --lean "$LEAN" 2>&1 | tee -a "$O/zk.log" | grep ZKCOST
    [ "${PIPESTATUS[0]}" = 0 ] || rc=1
  fi
  if [ -n "${ZK_UNITS:-}" ]; then ZK overhead --card none 2>&1 | tee -a "$O/zk.log"
  else ZK fit 2>&1 | tee -a "$O/zk.log"; fi
fi
(cd "$Z" && find . -type f -size -8M ! -name circuit.txt ! -name 'inst*.bin' ! -name 'pub*.bin' ! -path '*/archive/*' \
  \( ! -path '*/sessions/*/*' -o -name session.json \) -print0 | xargs -0 -r -I{} cp --parents {} "$O/")
log "zk-cost-cpu: done, rc $rc"
exit "$rc"
