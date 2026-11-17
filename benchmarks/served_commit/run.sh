#!/usr/bin/env bash
# `verity bench served_commit [ARGS]`: served vLLM against served vLLM with every value committed (the spec's Commit arm,
# prof_arm.py's rows-sha512 arm in the served specs),
# interleaved, through `verity-bench vllm` on one GPU, from the source tree: the spec SPEC (one of the served arm's,
# benchmarks/harness/specs/*-served-arm*.toml) with its arms, sweep and reps unless ARGS name others (`--sweep quick --reps 1`).
# The result, its table and the run's log go to $OUT (else $RESEARCH_RUN_DIR/out, else ./out). served_arm.sh runs it, and
# its header has the served arm's command. record.py then writes the Commit arm's ratios over stock (metrics.toml), and its percent
# over stock's time per output token at batch 32 (benchmarks/harness/metrics.toml).
# `verity bench served_commit --hash-rate`: rows_floor.sh, the committing kernels' hashing rate on one GPU.
set -euo pipefail
[ "${1:-}" != --hash-rate ] || { shift; exec bash benchmarks/served_commit/rows_floor.sh "$@"; }
. benchmarks/harness/env.sh
OUT=${OUT:-${RESEARCH_RUN_DIR:-.}/out}
rc=0
"$VENV/bin/python" -m verity_bench vllm --preset "${SPEC:?SPEC names a served spec, e.g. qwen3-8b-bf16-rtx-served-arm}" --out "$OUT" "$@" || rc=$?
[ ! -f "$OUT/result.json" ] || "$VENV/bin/python" benchmarks/served_commit/record.py "$OUT/result.json"
exit $rc
