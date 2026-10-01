#!/usr/bin/env bash
# Research question: what does crediting R1-H's residual-basis atom slices once across layers (the residual-state rule against
# a faulty unit's carried residual, the red team's 4:31 AM PDT C5) cost honest Llama-3.1-8B rows, per layer and class, on
# natural text and the adversarial token workloads, at layers 4 apart and at adjacent layers?
#
# Node 2, CPU only: r1_xlayer.py on every capture present (natural; repeat, phrase, lowent and long from r1_adv_job.sh;
# layers 0, 4, ..., 28), then on a 512-token natural capture of every layer (pearlc_capture.py --every 1, made once into
# $CAPTURES/natural-every1-512, outside the run dir; ADJ=0 skips it).
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/hf}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
MODEL=unsloth/Meta-Llama-3.1-8B-Instruct
REV=a2856192dd7c25b842431f39c179a6c2c2f627d1
SNAP="$HF_HOME/hub/models--unsloth--Meta-Llama-3.1-8B-Instruct/snapshots/$REV"
CAPTURES="${CAPTURES:-/workspace/pouw/pouw-design/captures}"
OUT="$RESEARCH_RUN_DIR/out"
WORKERS="${WORKERS:-8}"
SCRATCH="$(mktemp -d)"
trap 'rm -rf "$SCRATCH"' EXIT
CAP=(uv run --no-project --with torch --with transformers --with safetensors --with numpy --with huggingface_hub --with datasets)
RUN=(env PYTHONPATH=packages/verity/src:protocols/pouw uv run --no-project --with numpy python benchmarks/pouw/pearl_c_sm120/r1_xlayer.py
     --weights "$SNAP" --workers "$WORKERS" --d "${D:-6}" --scratch "$SCRATCH")
mkdir -p "$OUT"
rc=0
for w in natural ${WORKLOADS:-repeat phrase lowent long}; do
  d="$CAPTURES/$w"; first=0
  [[ "$w" == natural ]] && d=/workspace/pouw/fp8-security/llama8b-w0
  [[ "$w" == long ]] && first=6144
  [[ -f "$d/manifest.json" ]] || { echo "$w: no capture" >&2; continue; }
  "${RUN[@]}" --acts "$d" --first-row "$first" --out "$OUT/$w.json" > "$OUT/$w.log" 2>&1 || { rc=1; tail -5 "$OUT/$w.log" >&2; }
done
if [[ "${ADJ:-1}" == 1 ]]; then
  d="$CAPTURES/natural-every1-512"
  if [[ ! -f "$d/manifest.json" ]]; then
    "${CAP[@]}" python benchmarks/pouw/pearlc_capture.py --model "$MODEL" --revision "$REV" --tokens 512 --every 1 \
      --threads "$WORKERS" --budget-min 25 --out "$d" > "$OUT/capture-every1.log" 2>&1 || { rc=1; tail -5 "$OUT/capture-every1.log" >&2; }
  fi
  if [[ -f "$d/manifest.json" ]]; then
    "${RUN[@]}" --acts "$d" --layers "$(seq -s, 0 31)" --out "$OUT/natural-every1.json" > "$OUT/natural-every1.log" 2>&1 \
      || { rc=1; tail -5 "$OUT/natural-every1.log" >&2; }
  fi
fi
exit "$rc"
