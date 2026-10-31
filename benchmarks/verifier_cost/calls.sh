#!/usr/bin/env bash
# One verifier call's cost on a large program, step by step: calls.py builds population.py's population of N units with
# every composite padded by K XORs (K = 290 at the default hierarchy is a descriptor of about 300 MB over about 8,100
# Definitions) and a valid v1 registration of it, and times the Python reference, the program's encoding, the request
# line and `verifier.check` through `sampled-proofs-verifier`, C calls in one process; calls-time (calls/Main.lean,
# built in a scratch Lake package against the checkout's verifier) times each step the Lean takes on that request line.
# Run from a clone (the build writes the verifier's .lake):
#   research run --as proofs --project verity --on vy-nebius-cpu-2 --source . --pool proofs --cwd clone --mem-gb 90 -- \
#     bash benchmarks/verifier_cost/calls.sh 6963296 290 3
# $OUT gets calls.jsonl (one row per Python step) and lean.jsonl (one row per Lean step).
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -ge 2 ] || { echo "usage: calls.sh N K [C]" >&2; exit 2; }
L=$PWD/verity/core/service/sampled_proofs
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
mkdir -p "$S/time"
cp benchmarks/verifier_cost/calls/Main.lean "$L/lean-toolchain" "$S/time/"
cat > "$S/time/lakefile.toml" <<EOF
name = "calls_time"
defaultTargets = ["calls-time"]

[[require]]
name = "core_sampled_proofs"
path = "$L"

[[lean_exe]]
name = "calls-time"
root = "Main"
EOF
(cd "$S/time" && lake build calls-time) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
uv run python benchmarks/verifier_cost/calls.py "$S/in" "$1" --pad "$2" --calls "${3:-2}" | tee "$O/calls.jsonl"
"$S/time/.lake/build/bin/calls-time" "$S/in/registration.line" | grep '^PHASE ' | sed 's/^PHASE //' | tee "$O/lean.jsonl"
