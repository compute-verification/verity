#!/usr/bin/env bash
set -euo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
export TMPDIR="$d"
python3 tools/verity/lean/lean_changed.py --records Security/Proofs --out "$RESEARCH_RUN_DIR/lean-audit/proofs"
