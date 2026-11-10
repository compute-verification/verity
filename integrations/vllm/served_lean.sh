#!/usr/bin/env bash
# Scratch, never for landing: the small served phase's circuit through the record form at this checkout, the reference
# first (served_rule.py, which also writes the one difftest case) and then the compiled Lean on that case.
#   bash integrations/vllm/served_lean.sh        (from the repository's root, in a clone)
set -euo pipefail
root=$(pwd)
pkg=verity/core/protocols/network/nci/lean
scratch=$(mktemp -d)
uv sync -q --all-packages --extra torch-cpu
(cd integrations/vllm && uv run python served_rule.py --case "$scratch")
ls -la "$scratch"
bash tools/verity/lean/setup.sh "$pkg" > "$scratch/setup.log" 2>&1 || { tail -40 "$scratch/setup.log"; exit 1; }
export PATH="$HOME/.elan/bin:$PATH"
cd "$root/$pkg"
lake build nci-record-difftest > "$scratch/build.log" 2>&1 || { tail -60 "$scratch/build.log"; exit 1; }
tail -3 "$scratch/build.log"
echo "the compiled Lean form on the served case:"
time .lake/build/bin/nci-record-difftest "$scratch/cases.jsonl" "$scratch/python.jsonl" 1
echo "what the reference said of it (python.jsonl):"
cat "$scratch/python.jsonl"
