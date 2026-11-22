# The two-stage scripts' builds, sourced from the tree's root (two_stage_small.sh, two_stage_served.sh): uv and the
# workspace, flock-circuit, verity-verify-proof and SPOT's Lean verifier built from this tree, and main's first-parent
# line for a local challenger session's pin. Reads O (the records), W (scratch), FW (flock-circuit's build, default W)
# and GPU (default 0); sets BIN, LEAN and M0PATH, and writes machine.txt, the builds' logs, binaries.sha256 and main.txt
# under O.
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD; } | tee -a "$O/machine.txt"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }
{ (cd verity/protocols/proof/flock/verifier && lake build verity-verify-proof flock-zk-server) \
    && (cd verity/protocols/proof/spot/verifier/lean && lake build spot-verifier); } > "$O/build-lean.log" 2>&1 &
LP=$!
mkdir -p "$O/m0-build"
FW=${FW:-$W}
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$FW MODE=build GPU=${GPU:-0} bash verity/protocols/proof/flock/prover/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
  || { tail -40 "$O/m0-build/build.log"; exit 6; }
BIN=$FW/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 "$O/build-lean.log"; exit 7; }
LEAN=$REPO/verity/protocols/proof/flock/verifier/.lake/build/bin/verity-verify-proof
sha256sum "$BIN" "$LEAN" | tee "$O/binaries.sha256"
M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/proof/flock/prover/python:$REPO/adapters/vllm:$REPO/adapters/torch
export RAYON_NUM_THREADS=$(nproc)
# a local challenger session's pin is held to main's first-parent line: a run's clone has the shipped tree as its origin,
# which holds main as its own main or origin/main
UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
{ git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main; } > "$O/fetch-main.log" 2>&1 \
  && git rev-list --first-parent refs/remotes/origin/main > "$O/main.txt" || { tail -5 "$O/fetch-main.log"; exit 4; }
