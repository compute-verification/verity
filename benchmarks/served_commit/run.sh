#!/usr/bin/env bash
# `verity bench served_commit [ARGS]`: served vLLM against served vLLM with every value committed (the spec's Commit arm,
# prof_arm.py's rows-sha512 arm in the served specs),
# interleaved, through `verity-bench vllm` on one GPU, from the source tree: the spec SPEC (default
# qwen3-8b-bf16-h200-served) with its arms, sweep and reps unless ARGS name others (`--sweep quick --reps 1`). The
# result, its table and the run's log go to $OUT (a research run's, else ./out).
#
#   research run --on vy-glm53-eu2 --project verity --source . --cwd source --mem-gb 400 \
#     --env CUDA_VISIBLE_DEVICES=7 --env VERITY_BENCH_VENV=/workspace/cp/venv312 \
#     --env NATIVE_COLLECT_BUILD=/workspace/cp/nc_build -- bash benchmarks/served_commit/run.sh
set -euo pipefail
. benchmarks/harness/env.sh
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
exec "$VENV/bin/python" -m verity_bench vllm --preset "${SPEC:-qwen3-8b-bf16-h200-served}" --out "$OUT" "$@"
