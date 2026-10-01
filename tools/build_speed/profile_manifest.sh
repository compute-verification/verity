#!/usr/bin/env bash
# py-spy over one B=1 manifest build (`manifest build`, as the Build's required-manifest step runs it) of a kept request Program dir
# (PROGRAM: its instances.json.gz, result.json, artifact.json, descriptor.json.gz), reading the unit rules from RULES; collapsed stacks,
# the manifest log and its digest to $RESEARCH_RUN_DIR/out/. Measurement only.
set -euo pipefail
ROW=$1 PROGRAM=$2 RULES=$3
SRC=$(pwd)
RID=${RESEARCH_RUN_ID:?}
SCR=/workspace/research/build-speed/runs/$RID
mkdir -p "$SCR/tmp" "$RESEARCH_RUN_DIR/out"
trap 'rm -rf "$SCR"' EXIT
export TMPDIR=$SCR/tmp PY=/workspace/jobs/venv312/bin/python OMP_NUM_THREADS=16 BUILD_JOBS=auto VERITY_UNIT_RULE_CACHE=$RULES
cd "$SRC/integrations/vllm"
export PYTHONPATH=.:$SRC/packages/verity/src:$SRC/tools/research/src
"$PY" -m compileall -q -j 16 "$SRC/packages/verity/src" verity_vllm "$SRC/tools/research/src" > /dev/null 2>&1 || true
t0=$(date +%s.%N)
"/workspace/research/build-speed/spyenv/bin/py-spy" record --rate 20 --subprocesses --format raw --nonblocking -o "$RESEARCH_RUN_DIR/out/manifest.collapsed" -- \
  "$PY" -m verity_vllm.pipeline.cli manifest build --program "$PROGRAM" --workload "workloads/$ROW.json" --norm-scales --router-softmax \
  --vocab-range --guarded-max --out "$SCR/manifest.json" > "$RESEARCH_RUN_DIR/out/manifest.log" 2>&1
t1=$(date +%s.%N)
"$PY" -c "import json,sys; m=json.load(open(sys.argv[1])); print(json.dumps({'wall': round(float(sys.argv[3]) - float(sys.argv[2]), 1), 'manifest_digest': m.get('manifest_digest'), 'program_digest': m.get('program_digest')}))" \
  "$SCR/manifest.json" "$t0" "$t1" | tee "$RESEARCH_RUN_DIR/out/manifest_times.json"
