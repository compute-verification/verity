#!/usr/bin/env bash
# The CPU gate on a pod, over every staged row dir: required-value manifest (the row's Build dir x workload, as row_pod builds it) ->
# runtime tree (the model on the META device, as Build instantiates it) -> `verity_vllm.acquire.gate` (the row's acquisition plan).
#
#   ROWS=/workspace/rows OUT=$RESEARCH_RUN_DIR/acquire_gate SCRATCH=/workspace/acquire_gate bash verity_vllm/ops/pod_gate.sh [row-id ...]
#
# Small documents (acquisition_plan.json, runtime_tree.json, logs) go to $OUT/<row>/ (fetched with the
# run); the manifest (hundreds of MB at B=64) stays in $SCRATCH/<row>/manifest.json.  Exit 0 iff every row's gate exits 0.
set -uo pipefail
cd "$(dirname "$0")/../.."                      # integrations/vllm
PY=${PY:-/workspace/venv312/bin/python}
ROWS=${ROWS:-/workspace/rows}
OUT=${OUT:-${RESEARCH_RUN_DIR:-out}/acquire_gate}
SCRATCH=${SCRATCH:-/workspace/acquire_gate}
# (the core `verity` package must be importable: the manifest build reads the registry)
export PYTHONPATH=${PYTHONPATH:-.:$PWD/../../packages/verity/src} HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p "$OUT" "$SCRATCH"
RC=0
if [ $# -gt 0 ]; then ROWLIST=("$@"); else ROWLIST=(); for d in "$ROWS"/*/; do ROWLIST+=("$(basename "$d")"); done; fi
for ROW in "${ROWLIST[@]}"; do
  RD=$ROWS/$ROW; WL=workloads/$ROW.json
  [ -d "$RD" ] || { echo "[pod_gate] $ROW: no row dir $RD"; RC=1; continue; }
  [ -f "$WL" ] || { echo "[pod_gate] $ROW: no workload $WL"; RC=1; continue; }
  O=$OUT/$ROW; S=$SCRATCH/$ROW; mkdir -p "$O" "$S"
  read -r B MODEL REV MML <<< "$($PY - "$WL" <<'EOF'
import json, sys
w = json.load(open(sys.argv[1])); s = w["sweep"]
mml = (w.get("engine_args_required") or {}).get("max_model_len") or (int(s["max_input"]) + int(s["max_output"]))
print(s["batch"], w["model"], w["tokenizer_revision"], mml)
EOF
)"
  echo "[pod_gate] $ROW: B=$B model=$MODEL@${REV:0:10} max_model_len=$MML"
  # 1. the required-value manifest of record: the row's own when staged, else built exactly as row_pod.sh does (build-global at B>1, build at B=1)
  MAN=$S/manifest.json
  if [ -f "$RD/manifest.json" ]; then
    cp "$RD/manifest.json" "$MAN"; echo "[pod_gate] $ROW: manifest = the staged row's manifest.json"
  elif [ ! -f "$MAN" ]; then
    T0=$(date +%s)
    if [ "$B" -gt 1 ]; then
      WD=$($PY -c "import json,sys;d=json.load(open(sys.argv[1]));print(d.get('workload_digest') or d.get('program_digest') or d.get('digest') or '')" "$RD/build_workload/workload_program.json" 2>/dev/null || true)
      $PY -m verity_vllm.pipeline.cli build-global --fixture "$WL" --programs-root "$RD" ${WD:+--workload-digest "$WD"} --out "$MAN" > "$O/manifest.log" 2>&1 || { echo "[pod_gate] $ROW: manifest build-global FAILED (see manifest.log)"; tail -n 5 "$O/manifest.log"; RC=1; continue; }
    else
      $PY -m verity_vllm.pipeline.cli build --program "$RD/build_request" --workload "$WL" --out "$MAN" > "$O/manifest.log" 2>&1 || { echo "[pod_gate] $ROW: manifest build FAILED (see manifest.log)"; tail -n 5 "$O/manifest.log"; RC=1; continue; }
    fi
    echo "[pod_gate] $ROW: manifest built in $(( $(date +%s) - T0 )) s: $(tail -n 1 "$O/manifest.log" | cut -c1-200)"
  fi
  ls -l "$MAN" | awk '{print "[pod_gate] manifest bytes " $5}'
  $PY - "$MAN" "$O/manifest_header.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
hdr = {k: v for k, v in m.items() if k not in ("identities",)}
hdr["identities_n"] = len(m.get("identities") or [])
json.dump(hdr, open(sys.argv[2], "w"), indent=1, default=str)
print("[pod_gate] manifest:", m.get("manifest_digest", "")[:16], "identities", hdr["identities_n"], "families", (m.get("populations") or {}).get("by_family"))
EOF
  # 2. the runtime tree: the model instantiated on the meta device as Build does
  T0=$(date +%s)
  # (HF_HUB_OFFLINE=0: a row whose checkpoint is not pre-warmed on this pod needs its config.json fetched -- shapes only, no weights)
  if HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} $PY -m verity_vllm.correspondence.runtime_tree --model "$MODEL" --revision "$REV" --max-model-len "$MML" --out "$O/runtime_tree.json" > "$O/runtime_tree.log" 2>&1; then
    echo "[pod_gate] $ROW: runtime tree (meta) in $(( $(date +%s) - T0 )) s: $(tail -n 1 "$O/runtime_tree.log")"
  else
    echo "[pod_gate] $ROW: runtime tree (meta) FAILED (see runtime_tree.log)"; tail -n 8 "$O/runtime_tree.log"; RC=1; continue
  fi
  # 3. the gate (--programs-root: the owning modules come off the row's request Programs' runtime-correspondence/v1 records when the Build
  #    carries them, else off the manifest's v1 annotations -- the source is in acquisition_plan.json `correspondence`;
  #    AGAINST=<dir of acquisition_plan.json per row>: compare this plan Value by Value with that run's -> plan_compare.json)
  T0=$(date +%s)
  AG=(); [ -n "${AGAINST:-}" ] && [ -f "$AGAINST/$ROW/acquisition_plan.json" ] && AG=(--against "$AGAINST/$ROW/acquisition_plan.json")
  if $PY -m verity_vllm.acquire.gate --manifest "$MAN" --runtime "$O/runtime_tree.json" --out "$O" --row-id "$ROW" --programs-root "$RD" ${AG[@]+"${AG[@]}"} > "$O/gate.log" 2>&1; then
    echo "[pod_gate] $ROW: GATE PASS in $(( $(date +%s) - T0 )) s"
  else
    echo "[pod_gate] $ROW: GATE rc=$? in $(( $(date +%s) - T0 )) s"; RC=1
  fi
  cat "$O/gate.log"
done
echo "[pod_gate] done rc=$RC -> $OUT"
exit $RC
