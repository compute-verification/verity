#!/usr/bin/env bash
# Scratch: one variant (wired | unwired | field-out | positions) of the wired two-template window through the served audit.
set -euo pipefail
variant="${1:?variant}"
if [ "$variant" != wired ]; then git apply "scratch-green/$variant.patch"; fi
git diff --stat
uv sync -q --all-packages --extra torch-cpu
(cd verity/protocols/proofs/flock/verifier && lake build verity-verify-proof flock-zk-server 2>&1 | tail -5)
GREEN_VARIANT="$variant" uv run python -m pytest tests/benchmarks/one_stage/test_green_wired_run.py -s -p no:cacheprovider -rs 2>&1 \
  | tee "${OUT:-/tmp}/pytest-$variant.log"
