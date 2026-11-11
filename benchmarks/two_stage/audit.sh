#!/usr/bin/env bash
# The two-stage toy's audit on a CPU node (audit.py): flock-circuit, flock-verify and SPOT's Lean verifier (the
# stream's draws) built from this tree, then the honest run and the --altered run, each one's records under $OUT/<case>
# (never a circuit, a prover file, a private body, or a challenger's key or secret: its *-work/).
#   research run --on vy-nebius-cpu-11 --pool research --as proofs --project verity --source . --cwd clone -- \
#     bash benchmarks/two_stage/audit.sh
#   TWO_FLOCK_WORK=DIR keeps flock-circuit's build there across runs (default: scratch); TWO_CASES="honest" runs those only;
#   TWO_ARGS="--n 8 --q 4" passes audit.py more; FLOCK_CHALLENGER=1 gives each session a challenger of its own, audited
#   at the sessions' end (in Lean too, which builds SPOT's difftest)
set -euo pipefail
REPO=$(git rev-parse --show-toplevel); cd "$REPO"
O=${OUT:?research run sets OUT}; mkdir -p "$O"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
export PATH=$HOME/.local/bin:$HOME/.cargo/bin:$HOME/.elan/bin:$PATH
{ nproc; free -g | head -2; git rev-parse HEAD HEAD:verity/protocols/flock/prover HEAD:verity/protocols/flock/pod HEAD:verity/protocols/flock/agreement HEAD:verity/protocols/flock/tool.py; } | tee "$O/machine.txt"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
command -v lake >/dev/null || curl -sSfL https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null
uv sync --all-packages --extra torch-cpu > "$O/sync.log" 2>&1 || { tail -20 "$O/sync.log"; exit 5; }

# under FLOCK_CHALLENGER=1 each session's challenger is audited in Lean too (the difftest's challenger pieces)
SPD=; [ "${FLOCK_CHALLENGER:-0}" = 1 ] && SPD=sampled-proofs-difftest
{ (cd verity/protocols/flock/verifier && lake build flock-verify flock-firewall) \
    && (cd verity/protocols/spot/verifier/lean && lake build sampled-proofs-verifier $SPD); } > "$O/build-lean.log" 2>&1 &
LP=$!
mkdir -p "$O/m0-build"
FW=${TWO_FLOCK_WORK:-$W}
RESEARCH_RUN_DIR=$O/m0-build FLOCK_WORK=$FW MODE=build GPU=0 bash verity/protocols/flock/pod/60-circuit.sh > "$O/m0-build/build.log" 2>&1 \
  || { tail -40 "$O/m0-build/build.log"; exit 6; }
BIN=$FW/flock-gpu-link/flock/target/release/flock-circuit
wait $LP || { tail -40 "$O/build-lean.log"; exit 7; }
LEAN=$REPO/verity/protocols/flock/verifier/.lake/build/bin/flock-verify
sha256sum "$BIN" "$LEAN" | tee "$O/binaries.sha256"

M0PATH=$REPO:$REPO/benchmarks/numerical/python:$REPO/verity/protocols/flock/prover/python:$REPO/adapters/vllm
export RAYON_NUM_THREADS=$(nproc)
# the challenger session's pin is held to main's first-parent line: a run's clone has the shipped tree as its origin, which
# holds main as its own main or origin/main
UNSHALLOW=(); [ "$(git rev-parse --is-shallow-repository)" = true ] && UNSHALLOW=(--unshallow)
{ git fetch -q "${UNSHALLOW[@]}" origin +refs/heads/main:refs/remotes/origin/main \
    || git fetch -q "${UNSHALLOW[@]}" origin +refs/remotes/origin/main:refs/remotes/origin/main; } > "$O/fetch-main.log" 2>&1 \
  && git rev-list --first-parent refs/remotes/origin/main > "$O/main.txt" || { tail -5 "$O/fetch-main.log"; exit 4; }
rc=0
for c in ${TWO_CASES:-honest altered}; do
  flags=(); [ "$c" = altered ] && flags=(--altered)
  # shellcheck disable=SC2086
  PYTHONPATH=$M0PATH uv run --no-sync python benchmarks/two_stage/audit.py --m0-pythonpath "$M0PATH" --bin "$BIN" --lean "$LEAN" \
    --main "$O/main.txt" --out "$W/$c" "${flags[@]}" ${TWO_ARGS:-} > "$O/$c.log" 2>&1 || rc=1
  grep '^TWO ' "$O/$c.log" || tail -20 "$O/$c.log"
  mkdir -p "$O/$c"
  (cd "$W/$c" && find . -type f ! -name circuit.txt ! -name 'inst-*' ! -name 'm*-inst-*' ! -name '*-private.bin' \
     ! -name index.json ! -name index-salt.bin ! -path './*/archive/*' ! -path '*-work/*' -print0 \
     | xargs -0 -r -I{} cp --parents {} "$O/$c/")
done
exit $rc
