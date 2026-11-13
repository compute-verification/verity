set -uo pipefail
# CIR-274 (REQBLOCK): collect43.py's TP window with the ranks compiled in parallel (collect_par.py), for each layer count in LAYERS_LIST.
uv sync -q --all-packages --extra torch-cpu || { echo "REQBLOCK uv sync failed"; exit 2; }
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/source_sha.txt" 2>/dev/null
echo "REQBLOCK collect_par tree $(git rev-parse --short=12 HEAD 2>/dev/null || echo '?') from $(date -u +%H:%M:%SZ)"
cd adapters/vllm
for L in ${LAYERS_LIST:-4}; do
  ../../.venv/bin/python ../../scratch/cir274/collect_par.py "$OUT" "$L" "${LP:-43}" 2>&1 | tee -a "$OUT/collect.log" \
    | grep -E '^REQBLOCK' | cut -c1-3200
done
echo "REQBLOCK collect_par done at $(date -u +%H:%M:%SZ)"
