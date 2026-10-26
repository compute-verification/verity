#!/usr/bin/env bash
# The frontier census: atoms.py counts every atom frontier.py composes (property/v1 on vLLM's registry Definitions; CPU only,
# no torch), holds them to the int8 census's measured Llama-3.1-8B, and frontier.py gives the table. Keeps atoms.json,
# atoms.log, frontier.json and frontier.md in $OUT.
#
#   research run --queue --on vy-nebius-1 --cpus 4 --mem-gb 16 --max-min 120 --source . --project verity \
#     --cwd clone/integrations/vllm -- bash ../../benchmarks/stacked_share/frontier_job.sh
set -euo pipefail
export PY=/workspace/jobs/venv312/bin/python3 PYTHONPATH=.:../..:../../tools/research/src
HERE=../../benchmarks/stacked_share
/usr/bin/time -v "$PY" "$HERE/atoms.py" --out "$OUT/atoms.json" 2> >(tee "$OUT/atoms.log" >&2)
"$PY" "$HERE/frontier.py" --atoms "$OUT/atoms.json" --out "$OUT/frontier.json"
