#!/usr/bin/env bash
# One request derive of ROW's workload (LP, T), as the Build's derive runs it (bench_build.sh's environment): its phase times, the
# Program digest, and the sha256 of instances.json.gz's text and of descriptor.json.gz's, with each output's write time read from the
# files' mtimes, to $RESEARCH_RUN_DIR/out/derive_once.json. Measurement only.
set -euo pipefail
ROW=$1 LP=$2 T=$3 MML=$4
SRC=$(pwd)
RID=${RESEARCH_RUN_ID:?}
SCR=/workspace/research/build-speed/runs/$RID
mkdir -p "$SCR/tmp" "$RESEARCH_RUN_DIR/out"
trap 'rm -rf "$SCR"' EXIT
export TMPDIR=$SCR/tmp
export PY=/workspace/jobs/venv312/bin/python
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 CUDA_HOME=/workspace/jobs/cuda-12.9 TORCH_EXTENSIONS_DIR=/workspace/jobs/torch-extensions
export OMP_NUM_THREADS=16 VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false
export PATH=/workspace/jobs/venv312/bin:$CUDA_HOME/bin:$PATH
cd "$SRC/integrations/vllm"
export PYTHONPATH=.:$SRC/packages/verity/src:$SRC/tools/research/src CUDA_VISIBLE_DEVICES=
export LD_LIBRARY_PATH=/workspace/jobs/cuda-driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
"$PY" -m compileall -q -j 16 "$SRC/packages/verity/src" verity_vllm "$SRC/tools/research/src" > /dev/null 2>&1 || true
TGT=$("$PY" -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['target']))" "workloads/$ROW.json")
t0=$(date +%s.%N)
"$PY" -m verity_vllm.pipeline.cli build --model unsloth/Llama-3.2-1B --revision 9535bd9b1d1dea6acafbdc4813b728796aeb28da --wrapper request \
  --LP "$LP" --T "$T" --max-model-len "$MML" --workload "workloads/$ROW.json" --out "$SCR/derive" --target "$TGT" > "$RESEARCH_RUN_DIR/out/derive.log" 2>&1
t1=$(date +%s.%N)
"$PY" - "$SCR/derive" "$t0" "$t1" << 'EOF' | tee "$RESEARCH_RUN_DIR/out/derive_once.json"
import gzip, hashlib, json, os, sys
d, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
r = json.load(open(f"{d}/result.json"))


def text_sha(p):
    h = hashlib.sha256()
    with gzip.open(p, "rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


m = {f: os.path.getmtime(f"{d}/{f}") for f in ("descriptor.json.gz", "derive-report.json", "instances.json.gz", "artifact.json")}
print(json.dumps({"wall": round(t1 - t0, 1), "export": r["export"]["secs"], "derive": r["derive_secs"], "total": r.get("wall_secs_total"),
                  "program_digest": r["program"]["digest"], "correspondence_digest": r["program"].get("correspondence_digest"),
                  "descriptor_sha256": r["program"].get("descriptor_sha256"), "instances_text_sha256": text_sha(f"{d}/instances.json.gz"),
                  "instances_gz_bytes": os.path.getsize(f"{d}/instances.json.gz"),
                  "instances_write_s": round(m["instances.json.gz"] - m["derive-report.json"], 1),
                  "descriptor_to_report_s": round(m["derive-report.json"] - m["descriptor.json.gz"], 1)}, indent=1))
EOF
