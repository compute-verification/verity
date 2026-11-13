set -uo pipefail
# CIR-271 (REQBLOCK): node request Programs at a ladder of shapes, at LAYERS (default 4), to find the smallest class keeping
# every Definition. SHAPES is "LP:T ...".
uv sync -q --all-packages --extra torch-cpu || { echo "REQBLOCK uv sync failed"; exit 2; }
mkdir -p "$OUT"
echo "REQBLOCK smallest tree $(git rev-parse --short=12 HEAD 2>/dev/null || echo '?') from $(date -u +%H:%M:%SZ)"
for L in ${LAYERS_LIST:-4}; do
  [ "$L" = all ] && L=""
  .venv/bin/python scratch/cir271/smallest.py "$OUT/l${L:-all}" "$L" "${SHAPES:-1:1 2:1 8:1 16:1 32:1 64:1 128:1 256:1 256:31}" 2>&1 \
    | tee -a "$OUT/smallest.log" | grep -E '^REQBLOCK' | cut -c1-1600
done
echo "REQBLOCK smallest done at $(date -u +%H:%M:%SZ)"
