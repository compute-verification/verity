#!/usr/bin/env bash
# One config-run Build of ROW on vy-nebius-1 as the dispatcher's `config-run` build task runs it (infra/nebius sky/jobs/config-run.yaml:
# `row stage build ROW ROLE REPO REV --config-run 1 --replay-k 460`, same interpreter, env and caches), with the row dir in scratch outside
# the run dir. Measurement only (cursor/build-speed-measure-8c79), never merged.
#   BENCH_RULES=seed|cold|shared   the unit-rule cache: a private copy of the seed snapshot (default), empty, or the host's shared one
#   BUILD_JOBS                     passed through (default auto)
set -euo pipefail
ROW=$1 ROLE=$2 REPO=$3 REV=$4
shift 4
SRC=$(pwd)
RID=${RESEARCH_RUN_ID:?}
BASE=/workspace/research/build-speed
SCR=$BASE/runs/$RID
mkdir -p "$SCR/sweep" "$SCR/tmp" "$RESEARCH_RUN_DIR/out"
trap 'rm -rf "$SCR"' EXIT
export TMPDIR=$SCR/tmp SWEEP_DIR=$SCR/sweep
export PY=/workspace/jobs/venv312/bin/python PY312=/workspace/jobs/venv312/bin/python
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 CUDA_HOME=/workspace/jobs/cuda-12.9 TORCH_EXTENSIONS_DIR=/workspace/jobs/torch-extensions
export OMP_NUM_THREADS=16 BUILD_JOBS=${BUILD_JOBS:-auto} BUILD_RAM_BUDGET_GB=${BUILD_RAM_BUDGET_GB:-160}
case "${BENCH_RULES:-seed}" in
  seed) cp -a "$BASE/unit-rules-seed" "$SCR/unit-rules"; export VERITY_UNIT_RULE_CACHE=$SCR/unit-rules ;;
  cold) mkdir -p "$SCR/unit-rules"; export VERITY_UNIT_RULE_CACHE=$SCR/unit-rules ;;
  shared) export VERITY_UNIT_RULE_CACHE=/workspace/jobs/cache/unit-rules ;;
esac
export PATH=/workspace/jobs/venv312/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=$SRC/integrations/vllm:$SRC/tools/research/src CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/workspace/jobs/cuda-driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
"$PY" -m compileall -q -j 16 packages/verity/src integrations/vllm/verity_vllm tools/research/src protocols > /dev/null 2>&1 || true
{
  echo "cpus_allowed=$(python3 -c 'import os; print(len(os.sched_getaffinity(0)))') nproc=$(nproc) cgroup=$(cat /proc/self/cgroup)"
  echo "BUILD_JOBS=$BUILD_JOBS BENCH_RULES=${BENCH_RULES:-seed} rules_before=$(find "$VERITY_UNIT_RULE_CACHE" -type f | wc -l)"
} | tee "$RESEARCH_RUN_DIR/out/bench.txt"
t0=$(date +%s)
rc=0
"$PY" -m verity_vllm.pipeline.cli row stage build "$ROW" "$ROLE" "$REPO" "$REV" --config-run 1 --replay-k 460 "$@" || rc=$?
t1=$(date +%s)
D=$SWEEP_DIR/$ROW
cg=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)
{
  echo "rc=$rc wall_s=$((t1 - t0)) memory_peak=$(cat "$cg/memory.peak" 2> /dev/null || echo na)"
  echo "rules_after=$(find "$VERITY_UNIT_RULE_CACHE" -type f | wc -l)"
} | tee -a "$RESEARCH_RUN_DIR/out/bench.txt"
cp "$D"/stages.txt "$D"/row.log "$D"/timeline.jsonl "$D"/build_summary.json "$D"/strict_word.json "$RESEARCH_RUN_DIR/out/" 2> /dev/null || true
cp "$D"/*.log "$RESEARCH_RUN_DIR/out/" 2> /dev/null || true
cp "$D"/build_workload/workload_program.json "$RESEARCH_RUN_DIR/out/" 2> /dev/null || true
"$PY" - "$D" > "$RESEARCH_RUN_DIR/out/digests.json" << 'EOF' || true
import glob, hashlib, json, os, sys
d = sys.argv[1]
out = {}
for f in sorted(glob.glob(f"{d}/build_*/*")) + [f"{d}/manifest.json", f"{d}/strict_word.json"]:
    if os.path.isfile(f) and not f.endswith((".log", ".rc")) and "derive-report" not in f:
        out[os.path.relpath(f, d)] = hashlib.sha256(open(f, "rb").read()).hexdigest()
m = json.load(open(f"{d}/manifest.json")) if os.path.isfile(f"{d}/manifest.json") else {}
out["_manifest_digest"] = m.get("manifest_digest")
out["_program_digest"] = m.get("program_digest")
json.dump(out, sys.stdout, indent=1, sort_keys=True)
EOF
exit $rc
