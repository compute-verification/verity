#!/usr/bin/env bash
# The one-stage verifier's checks on a5's path for a sample on the verifier's own coins, against the registration's ports
# (stream.py: a toy window of two templates with N ports, in a5's order, as a5 asks them, `a5_memo.memo`, and with
# NO_MEMO=1 each asked every time). Run from a checkout; the first check builds sampled-proofs-verifier with lake:
#   research run --as proofs --project verity --on vy-nebius-cpu-11 --source . --cwd source -- \
#     bash benchmarks/verifier_cost/stream.sh 10000 20000 40000
# $OUT gets stream.jsonl: one row per N (accepted, every negative refused, and per check and per Lean piece its calls and
# seconds).
set -euo pipefail
O=${OUT:?run under research run, which sets OUT}
[ $# -gt 0 ] || { echo "usage: stream.sh N..." >&2; exit 2; }
command -v lake > /dev/null || export PATH="$HOME/.elan/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu
uv run python benchmarks/verifier_cost/stream.py ${NO_MEMO:+--no-memo} "$@" | tee -a "$O/stream.jsonl"
