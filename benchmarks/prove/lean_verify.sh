#!/usr/bin/env bash
# The Lean verifier timed on one kept proof, as a research run: builds the verifier from the run's clone (the executable
# its lakefile roots at FlockVerify, by the name the lakefile gives it), fetches the proof's art (a tree: one session
# directory and the statement it was proved on) to scratch, and runs `verity bench prove lean-verify` on it, whose
# record ($OUT/lean-verify.json) `verity bench prove` reads once the run is labelled
# `run_of_record=prove-WORKLOAD-lean-verify`. The flags after ART are lean-verify's (--repeat, --word, --k-log, ...).
# The job exits nonzero when the verifier refuses the proof, and runs it no further. Run from a clone (the build writes
# the verifier's .lake), on a CPU node with the memory the statement's circuit file needs:
#   research run --project verity --on NODE --mem-gb GB --timeout SECONDS --source . --cwd clone -- \
#     bash benchmarks/prove/lean_verify.sh ART [--repeat N] [--word TYPE] [--definition ID] [--k-log K] [--blocks N]
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
art=${1:?usage: lean_verify.sh ART [lean-verify flags]}
shift
L=$PWD/verity/protocols/flock/verifier
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
exe=$(uv run python -c 'import sys, tomllib
[name] = [e["name"] for e in tomllib.load(open(sys.argv[1], "rb"))["lean_exe"] if e.get("root") == "FlockVerify"]
print(name)' "$L/lakefile.toml")
(cd "$L" && lake build "$exe") > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
research data fetch "$art" --to "$S/proof" > "$O/fetch.log" 2>&1 || { tail -20 "$O/fetch.log" >&2; exit 1; }
(cd "$S/proof" && find . -type f -printf '%s %p\n' | sort -k2) > "$O/proof-files.txt"
uv run verity bench prove lean-verify --lean "$L/.lake/build/bin/$exe" --proof "$S/proof" --out "$O" "$@"
