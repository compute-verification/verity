#!/usr/bin/env bash
# `verity bench served_commit [ARGS]`: served vLLM against served vLLM with every value committed (the spec's Commit arm,
# prof_arm.py's rows-sha512 arm in the served specs),
# interleaved, through `verity-bench vllm` on one GPU, from the source tree: the spec SPEC (one of the served arm's,
# benchmarks/specs/*-rtx-served-arm*.toml) with its arms, sweep and reps unless ARGS name others (`--sweep quick --reps 1`).
# The result, its table and the run's log go to $OUT (else $RESEARCH_RUN_DIR/out, else ./out). served_arm.sh runs it, and
# its header has the served arm's command.
set -euo pipefail
. benchmarks/harness/env.sh
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
exec "$VENV/bin/python" -m verity_bench vllm --preset "${SPEC:?SPEC names a served spec, e.g. qwen3-8b-bf16-rtx-served-arm}" --out "$OUT" "$@"
