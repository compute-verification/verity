#!/usr/bin/env bash
# The Lean verifier timed on one kept proof, as a research run: fetches the proof's art (kind `flock-proof/v1`, a tree:
# one session directory and what of its statement the art holds) to scratch, and beside it each file the statement keeps
# in another art (`--with ART PATH`: PATH of ART, as a circuit too large for the proof's art is kept, or a program or
# partition file made after it), builds the verifier from the run's clone (the executable its lakefile roots at
# FlockVerify, by the name the lakefile gives it), and runs `verity bench prove lean-verify` on the tree, whose record
# ($OUT/lean-verify.json) `verity bench prove` reads once the run is labelled `run_of_record=prove-WORKLOAD-lean-verify`.
# The flags after the arts are lean-verify's (--repeat, --word, --k-log, ...). The job exits nonzero when the verifier
# refuses the proof, and runs it no further; a partition file that isn't the one the circuit binds ends it before the
# verifier starts (the program is held to nothing the circuit states: lean_verify.py). Run from a clone (the build
# writes the verifier's .lake) with the store's read-only key (--store-read: the job fetches the arts), on a CPU node
# with the memory the statement's circuit file needs; scratch (mktemp's, so $TMPDIR's) holds the circuit twice, the
# fetched file and the verifier's archive of it:
#   research run --project verity --on NODE --mem-gb GB --timeout SECONDS --source . --cwd clone --store-read -- \
#     bash benchmarks/prove/lean_verify.sh ART [--with ART PATH]... [--repeat N] [--word TYPE] [--k-log K] [--blocks N]
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
art=${1:?usage: lean_verify.sh ART [--with ART PATH]... [lean-verify flags]}
shift
L=$PWD/verity/protocols/flock/verifier
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
fetch() { uv run research data fetch "$@" >> "$O/fetch.log" 2>&1 || { tail -20 "$O/fetch.log" >&2; exit 1; }; }
fetch "$art" --to "$S/tree/proof"
n=0
while [ "${1:-}" = --with ]; do
  [ $# -ge 3 ] || { echo "lean_verify.sh: --with takes an art and a path in it: --with ART PATH" >&2; exit 2; }
  n=$((n + 1))
  fetch "$2" --path "$3" --to "$S/tree/with-$n"
  shift 3
done
(cd "$S/tree" && find . -type f -printf '%s %p\n' | sort -k2) > "$O/proof-files.txt"
exe=$(uv run python -c 'import sys, tomllib
[name] = [e["name"] for e in tomllib.load(open(sys.argv[1], "rb"))["lean_exe"] if e.get("root") == "FlockVerify"]
print(name)' "$L/lakefile.toml")
(cd "$L" && lake build "$exe") > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
uv run verity bench prove lean-verify --lean "$L/.lake/build/bin/$exe" --proof "$S/tree" --out "$O" "$@"
