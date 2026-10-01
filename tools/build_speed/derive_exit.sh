#!/usr/bin/env bash
# One request derive of ROW's shape (LP, T) under several exit paths at once, in parallel (same host load): how long the process
# takes between `main` returning and the parent seeing it exit, and the Program digest it derived. Measurement only
# (cursor/build-speed-measure-8c79), never merged.
#   derive_exit.sh ROW LP T MML VARIANT...     VARIANT = exit (sys.exit) | freeze (gc.freeze(), then sys.exit) | osexit (flush, os._exit)
set -euo pipefail
ROW=$1 LP=$2 T=$3 MML=$4
shift 4
SRC=$(pwd)
RID=${RESEARCH_RUN_ID:?}
SCR=/workspace/research/build-speed/runs/$RID
mkdir -p "$SCR" "$RESEARCH_RUN_DIR/out"
trap 'rm -rf "$SCR"' EXIT
export TMPDIR=$SCR HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 CUDA_HOME=/workspace/jobs/cuda-12.9 TORCH_EXTENSIONS_DIR=/workspace/jobs/torch-extensions
export OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES= VLLM_BATCH_INVARIANT=1 TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=/workspace/jobs/cuda-driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/workspace/jobs/venv312/bin/python
export PYTHONPATH=.:$SRC/packages/verity/src:$SRC/tools/research/src
cd "$SRC/integrations/vllm"
"$PY" -m compileall -q -j 16 "$SRC/packages/verity/src" verity_vllm "$SRC/tools/research/src" > /dev/null 2>&1 || true
TGT=$("$PY" -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['target']))" "workloads/$ROW.json")
i=0
pids=()
for V in "$@"; do
  i=$((i + 1))
  (
    t0=$(date +%s.%N)
    rc=0
    EXIT_VARIANT=$V $PY -c "
import gc, os, sys, time
from verity_vllm.pipeline import cli
sys.argv = ['cli', *sys.argv[1:]]
rc = cli.main()
print(f'[exit-ab] returned {time.time():.3f}', file=sys.stderr, flush=True)
v = os.environ['EXIT_VARIANT']
if v == 'freeze':
    gc.freeze()
if v == 'osexit':
    sys.stdout.flush(); sys.stderr.flush(); os._exit(rc)
sys.exit(rc)" \
      build --model unsloth/Llama-3.2-1B --revision 9535bd9b1d1dea6acafbdc4813b728796aeb28da --wrapper request --LP "$LP" --T "$T" \
      --max-model-len "$MML" --workload "workloads/$ROW.json" --out "$SCR/d$i" --target "$TGT" > "$RESEARCH_RUN_DIR/out/derive_$i.log" 2>&1 || rc=$?
    t1=$(date +%s.%N)
    $PY - "$SCR/d$i" "$V" "$rc" "$t0" "$t1" "$RESEARCH_RUN_DIR/out/derive_$i.log" > "$RESEARCH_RUN_DIR/out/derive_$i.json" << 'EOF'
import json, os, re, sys
d, v, rc, t0, t1, log = sys.argv[1:]
r = json.load(open(f"{d}/result.json")) if rc == "0" else {}
last = max(os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d))
m = re.search(r"\[exit-ab\] returned ([0-9.]+)", open(log).read())
ret = float(m.group(1)) if m else None
print(json.dumps({"variant": v, "rc": int(rc), "wall_s": round(float(t1) - float(t0), 1), "derive": r.get("derive_secs"),
                  "last_write_to_return_s": round(ret - last, 1) if ret else None, "return_to_exit_s": round(float(t1) - ret, 1) if ret else None,
                  "digest": (r.get("program") or {}).get("digest"), "correspondence_digest": (r.get("program") or {}).get("correspondence_digest")}))
EOF
    rm -rf "$SCR/d$i"
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || true; done
cat "$RESEARCH_RUN_DIR"/out/derive_*.json
