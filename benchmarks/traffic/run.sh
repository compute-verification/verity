#!/usr/bin/env bash
# The traffic job: fetch and tokenize every source into $OUT/units, then the tables into $OUT. Run from the checkout:
#   research run --on vy-nebius-cpu-5 --pool research --as training --project verity --source . --cwd source \
#     --timeout 90m -- bash benchmarks/traffic/run.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:?set by research run: the run dir out/}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
export TRAFFIC_CACHE="$scratch/cache" HF_HOME="$scratch/hf" UV_CACHE_DIR="${UV_CACHE_DIR:-$scratch/uv}"
mkdir -p "$TRAFFIC_CACHE"
py=(uv run --no-project --python 3.12 --with numpy --with tokenizers --with huggingface_hub --with pyarrow python)
nproc; free -g | head -2
"${py[@]}" "$here/sessions.py" "$OUT" --frac "${FRAC:-0.05}" ${ONLY:+--only "$ONLY"}
"${py[@]}" "$here/measure.py" "$OUT/units" --out "$OUT"
