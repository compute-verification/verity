#!/usr/bin/env bash
set -euo pipefail
python3 tools/verity/lean/check.py --build --update Security Security/Proofs --out "$RESEARCH_RUN_DIR/lean-audit"
