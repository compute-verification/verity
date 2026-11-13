set -uo pipefail
# CIR-271 (REQBLOCK): class G's TP window of collective sites at (LP, 2 outputs) for each layer count in LAYERS_LIST.
uv sync -q --all-packages --extra torch-cpu || { echo "REQBLOCK uv sync failed"; exit 2; }
mkdir -p "$OUT"
echo "REQBLOCK collect tree $(git rev-parse --short=12 HEAD 2>/dev/null || echo '?') from $(date -u +%H:%M:%SZ)"
cd adapters/vllm
for L in ${LAYERS_LIST:-4}; do
  ../../.venv/bin/python ../../scratch/cir271/collect43.py "$OUT" "$L" "${LP:-43}" 2>&1 | tee -a "$OUT/collect.log" \
    | grep -E '^REQBLOCK' | cut -c1-3200
done
echo "REQBLOCK collect done at $(date -u +%H:%M:%SZ)"
