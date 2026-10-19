#!/usr/bin/env bash
# Experiments A and B1 of "Settling F3: routed-mixture experiments" on one node: every model in manifest.json scores
# $WINDOWS, one model per GPU lane, then analyze writes g, its intervals, D at g and the verdict to $OUT. A finished
# score file is never redone, so a rerun after a failure scores only what is missing. Not launched: it needs a budget
# line (see README.md).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
: "${OUT:?OUT is where scores, routing files and the analysis go (research run sets it)}"
: "${WINDOWS:?WINDOWS is the evaluation windows file, from probe.py windows}"
: "${CALIBRATION:?CALIBRATION is the codebook calibration windows file, disjoint from WINDOWS}"
: "${HF_HOME:?HF_HOME must be a model cache outside the run dir with about 1 TB free}"
MANIFEST=${MANIFEST:-$HERE/manifest.json}
GPUS=${GPUS:-8}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
read -r -a PY <<< "${PY:-uv run --no-project --python 3.12 --with torch==2.14.1 --with transformers==5.20.1 --with accelerate==1.15.0 python}"

mkdir -p "$OUT/scores" "$OUT/routing" "$OUT/logs"
rm -f "$OUT/failed.txt"
ROUTING=$OUT/routing
"${PY[@]}" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)" \
  | tee "$OUT/env.txt"
{ git -C "$HERE" rev-parse HEAD 2>/dev/null || true; nvidia-smi -L 2>/dev/null || true; } >> "$OUT/env.txt"
cp "$MANIFEST" "$OUT/manifest.json"

# name repo revision role confine, one line per model of the given roles
jobs() {
  "${PY[@]}" - "$MANIFEST" "$@" <<'EOF'
import json, sys
models = json.load(open(sys.argv[1]))["models"]
for name, m in models.items():
    if m.get("role", "score") in sys.argv[2:]:
        print(name, m["repo"], m.get("revision", "main"), m.get("role", "score"), m.get("confine", "none"))
EOF
}

score() {  # name repo revision role confine
  local name=$1 repo=$2 rev=$3 role=$4 confine=$5
  local args=(--windows "$WINDOWS")
  case $role in
    calibration) args=(--windows "$CALIBRATION" --routing-out "$ROUTING/calibration.pt" --profiles) ;;
    record) args+=(--routing-out "$ROUTING/windows.pt" --codebook "$ROUTING/codebook.pt") ;;
    confine) args+=(--confine "$confine" --routing "$ROUTING/windows.pt" --codebook "$ROUTING/codebook.pt") ;;
  esac
  "${PY[@]}" "$HERE/probe.py" score --model "$repo" --revision "$rev" --out "$OUT/scores/$name.jsonl" \
    --dtype "$DTYPE" --device "$DEVICE" "${args[@]}"
}

phase() {  # roles...: their models, split round-robin over $GPUS lanes
  local list
  list=$(mktemp)
  jobs "$@" > "$list"
  for ((lane = 0; lane < GPUS; lane++)); do
    (
      awk -v lane=$lane -v n="$GPUS" 'NR % n == lane' "$list" | while read -r name repo rev role confine; do
        [ -s "$OUT/scores/$name.jsonl" ] && continue
        echo "lane $lane: $name ($repo@$rev, $confine)"
        CUDA_VISIBLE_DEVICES=$lane score "$name" "$repo" "$rev" "$role" "$confine" < /dev/null \
          > "$OUT/logs/$name.log" 2>&1 || echo "$name" >> "$OUT/failed.txt"
      done
    ) &
  done
  wait
  rm -f "$list"
  if [ -s "$OUT/failed.txt" ]; then
    echo "failed (logs in $OUT/logs):" >&2
    cat "$OUT/failed.txt" >&2
    exit 1
  fi
}

phase score calibration
[ -s "$ROUTING/codebook.pt" ] || "${PY[@]}" "$HERE/probe.py" codebook --routing "$ROUTING/calibration.pt" \
  --out "$ROUTING/codebook.pt" --paths 16
phase record
phase confine
"${PY[@]}" "$HERE/probe.py" analyze "$MANIFEST" --base "$OUT" --out "$OUT/analysis.json" | tee "$OUT/verdict.txt"
