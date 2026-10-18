#!/usr/bin/env bash
# The loop's job.sh at the commit under audit, run under the H200's Build of record whatever that commit's job.sh knows of pins:
# prof.sh (identical from 599be4b to c17e521b0) reads PIN_BUILD itself, so iteration 0's job.sh runs pinned too.
# Sent with the build as inputs/qwen3-8b-build.json; cwd is the audited commit (--cwd source).
set -uo pipefail
export PIN_BUILD=${RESEARCH_RUN_DIR:?}/inputs/qwen3-8b-build.json
[ -f "$PIN_BUILD" ] || { echo "no $PIN_BUILD sent"; exit 9; }
git rev-parse HEAD 2>/dev/null || true
exec bash benchmarks/hash/loop/job.sh
