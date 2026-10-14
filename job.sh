#!/usr/bin/env bash
set -euo pipefail
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
export TMPDIR="$d"
python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security"
