#!/bin/bash
# Lane SD: correspondence of every recorded speculative decode step (both sides) against the typed programs.
# usage: sd_corr_all.sh <commit store> <out dir> [parallel=12] [sides="target draft"]
# CPU only (venv312: verity_vllm_numerics exact models); cos_sin tables are the CUDA-built ones.
set -u
COMMIT=$1; OUT=$2; PAR=${3:-12}; SIDES=${4:-"target draft"}
cd /workspace/veritor-sd/vllm-poc
mkdir -p "$OUT"
PY312=${PY312:-/workspace/venv312/bin/python}
TGT=/workspace/hf/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323/model.safetensors
DRF=/workspace/hf/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987/model.safetensors
CS_T=/workspace/ta1-store/B1/ta1eager/corr/cos_sin_cache_cuda.npy
CS_TC=/workspace/ta1-store/B1/ta1eager/corr/cos_sin_cache_cpu.npy
CS_D=/workspace/results/sd/corr/cos_sin_draft_cuda.npy
CS_DC=/workspace/results/sd/corr/cos_sin_draft_cpu.npy
# every (request, step) with a speculative block (k > 0)
$PY312 - "$COMMIT" > "$OUT/steps.txt" <<'EOF'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
for req in sorted(p.name for p in (root / "ru").iterdir() if p.is_dir()):
    for d in sorted(p for p in (root / "ru" / req).iterdir() if p.is_dir()):
        sp = json.loads((d / "index.json").read_text()).get("speculative") or {}
        if sp.get("k"):
            print(req, int(d.name))
EOF
echo "[$(date -u +%H:%M:%SZ)] $(wc -l < "$OUT/steps.txt") speculative steps; sides: $SIDES; parallel $PAR" | tee -a "$OUT/all.log"
run_one() {
  req=$1; step=$2; side=$3
  tag="$req-s$step-$side"
  [ -s "$OUT/$tag.json" ] && { echo "skip $tag"; return; }
  if [ "$side" = target ]; then ck=$TGT; cs="--cos-sin-npy $CS_T --cos-sin-cpu-npy $CS_TC"; else ck=$DRF; cs="--cos-sin-npy $CS_D --cos-sin-cpu-npy $CS_DC"; fi
  PYTHONPATH=.:../src VERITY_ARCH=sm_80 $PY312 -m verity_vllm_adapter.sd_corr --commit "$COMMIT" --request "$req" --step "$step" --side "$side" \
      --checkpoint "$ck" $cs --out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  echo "[$(date -u +%H:%M:%SZ)] $tag rc=$? $(tail -1 "$OUT/$tag.log" | cut -c1-160)" >> "$OUT/all.log"
}
export -f run_one; export COMMIT OUT PY312 TGT DRF CS_T CS_TC CS_D CS_DC
for side in $SIDES; do awk -v s="$side" '{print $1, $2, s}' "$OUT/steps.txt"; done | xargs -P "$PAR" -L 1 bash -c 'run_one $0 $1 $2'
echo "[$(date -u +%H:%M:%SZ)] ALL_DONE" | tee -a "$OUT/all.log"
