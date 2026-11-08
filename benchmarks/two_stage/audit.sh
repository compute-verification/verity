#!/usr/bin/env bash
# The two-stage toy's audit on a CPU node (audit.py): flock-circuit, flock-verify and the sampled-proofs verifier (the
# stream's draws) built from this tree, then the honest run and the --altered run, each one's records under $OUT/<case>
# (never a circuit, a prover file or a private body).
#   research run --on vy-nebius-cpu-11 --pool research --as proofs --project verity --source . --cwd clone -- \
#     bash benchmarks/two_stage/audit.sh
#   TWO_FLOCK_WORK=DIR keeps flock-circuit's build there across runs (default: scratch); TWO_CASES="honest" runs those only;
#   TWO_ARGS="--n 8 --q 4" passes audit.py more
set -euo pipefail
REPO=$(git rev-parse --show-toplevel); cd "$REPO"
O=${OUT:?research run sets OUT}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD HEAD:verity/ml/flock; } | tee "$O/machine.txt"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }

{ (cd verity/core/service/flock && lake build flock-verify flock-firewall) \
    && (cd verity/core/service/sampled_proofs && lake build sampled-proofs-verifier); } > "$O/build-lean.log" 2>&1 &
LP=$!
mkdir -p "$O/m0-build"
FW=${TWO_FLOCK_WORK:-$W}
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/ml/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
  || { tail -40 "$O/m0-build/build.log"; exit 6; }
BIN=$FW/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 "$O/build-lean.log"; exit 7; }
LEAN=$REPO/verity/core/service/flock/.lake/build/bin/flock-verify
sha256sum "$BIN" "$LEAN" | tee "$O/binaries.sha256"

M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/ml/flock/python:$REPO/integrations/vllm
export RAYON_NUM_THREADS=$(nproc)
rc=0
for c in ${TWO_CASES:-honest altered}; do
  flags=(); [ "$c" = altered ] && flags=(--altered)
  # shellcheck disable=SC2086
  PYTHONPATH=$M0PATH uv run --no-sync python benchmarks/two_stage/audit.py --m0-pythonpath "$M0PATH" --bin "$BIN" --lean "$LEAN" \
    --out "$W/$c" "${flags[@]}" ${TWO_ARGS:-} > "$O/$c.log" 2>&1 || rc=1
  grep '^TWO ' "$O/$c.log" || tail -20 "$O/$c.log"
  mkdir -p "$O/$c"
  (cd "$W/$c" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -name '*-private.bin' \
     ! -name index.json ! -name index-salt.bin ! -path './*/archive/*' -print0 | xargs -0 -r -I{} cp --parents {} "$O/$c/")
done
exit $rc
