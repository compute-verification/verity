#!/usr/bin/env bash
# One update-price run: the fine-tune (train.py), then the pricing (price.py). Outputs go to $OUT; downloads and
# checkpoints to a scratch directory removed at exit. From the repository root, as a research run:
#   research run --on auto --cpus 12 --mem-gb 32 --max-min 300 --cwd source -- bash benchmarks/update_price/run.sh
# Arguments go to train.py (a trial: --steps 3 --save 1,3 --state 1 --replay 1).
set -euo pipefail
: "${OUT:?run it under research run, which sets OUT}"
S=$(mktemp -d "${TMPDIR:-/tmp}/update-price.XXXXXX")
trap 'rm -rf "$S"' EXIT
df -h "$S" | tail -1
uv sync -q --all-packages --extra torch-cpu
W=(uv run --no-sync --with transformers==4.57.1 --with pyarrow --with zstandard)
"${W[@]}" python benchmarks/update_price/train.py --ckpt "$S/ckpt" --out "$OUT" --cache "$S/cache" --threads "$(nproc)" "$@"
du -sh "$S/ckpt"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  "${W[@]}" python benchmarks/update_price/price.py --ckpt "$S/ckpt" --train "$OUT/train.json" --out "$OUT" \
  --workers "${PRICE_WORKERS:-5}" --sel-workers "${PRICE_SEL_WORKERS:-2}"
