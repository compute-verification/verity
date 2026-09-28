#!/usr/bin/env bash
# rhoD_atomMixP_flat's domain conditions (hfloor, hlive, hN at ρ_max) on every credited row of one retained pass, CPU only,
# read-only on the pass (forming_domain.py):
#   bash benchmarks/pouw/pearl_c_vllm/forming_domain.sh <pass run id> <label> [J]
set -euo pipefail
WORK=${WORK:-/workspace/pouw/mvp-e2e}; PY=$WORK/venv312/bin/python
PASS=$WORK/passes/${1:?the pass run id}; LABEL=${2:?a label}; JOBS=${3:-12}; OUT=${RESEARCH_RUN_DIR:-.}/out
export PYTHONPATH=$PWD:$PWD/integrations/vllm:$PWD/benchmarks/pouw
"$PY" -u benchmarks/pouw/pearl_c_vllm/forming_domain.py "$PASS" --jobs "$JOBS" --out "$OUT/$LABEL-$1.json"
