#!/usr/bin/env bash
# The Lean verifier's SHA-512 rate and a table root's cost at a pinned capacity of 2^17 rows: table_root/Main.lean, built
# in a scratch Lake package against the checkout's verifier (main's cached Flock build, `setup.sh`, and the spot package's
# `Spot.HistoryTable`), times the bare hash, the frame-v3 node, leaf and tree leaf, a 2^17-leaf root, and whole table
# roots (`Spot.HistoryTable.rootAt`) at the lengths given. Run from a clone:
#   research run --as proofs --project verity --on vy-nebius-cpu-1 --source . --pool proofs --cwd clone -- \
#     bash benchmarks/verifier_cost/table_root/table_root.sh [--tables T] [--lengths 4096,40960,131072]
# $OUT gets table_root.jsonl (one row per phase: calls, ns a call, calls a second, ms, peak RSS) and build.log.
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
F=$PWD/verity/protocols/flock/verifier
P=$PWD/verity/protocols/spot/verifier/lean
S=$(mktemp -d)
trap 'rm -rf "$S"' EXIT
bash tools/verity/lean/setup.sh "$F" > "$O/setup.log" 2>&1 || { tail -40 "$O/setup.log" >&2; exit 1; }
export PATH="$HOME/.elan/bin:$PATH"
mkdir -p "$S/time"
cp benchmarks/verifier_cost/table_root/Main.lean "$F/lean-toolchain" "$S/time/"
cat > "$S/time/lakefile.toml" <<EOF
name = "table_root_time"
defaultTargets = ["table-root-time"]

[[require]]
name = "core_spot"
path = "$P"

[[lean_exe]]
name = "table-root-time"
root = "Main"
EOF
(cd "$S/time" && lake build table-root-time) > "$O/build.log" 2>&1 || { tail -60 "$O/build.log" >&2; exit 1; }
lscpu | grep -E '^(Model name|CPU\(s\)|CPU max MHz)' > "$O/cpu.txt" || true
"$S/time/.lake/build/bin/table-root-time" "$@" | grep '^PHASE ' | sed 's/^PHASE //' | tee "$O/table_root.jsonl"
