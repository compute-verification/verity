set -uo pipefail
# CIR-271 part 1 (REQBLOCK): the node request Program at (256, 31), eager then compiled, each in its own process.
# EXECUTIONS (default "eager compiled"), LAYERS (default: the whole model) and OUTPUTS (default 32) narrow it.
uv sync -q --all-packages --extra torch-cpu || { echo "REQBLOCK uv sync failed"; exit 2; }
mkdir -p "$OUT"
echo "REQBLOCK tree $(git rev-parse --short=12 HEAD 2>/dev/null || echo '?') on $(hostname) from $(date -u +%H:%M:%SZ) layers=${LAYERS:-all}"
rc=0
for ex in ${EXECUTIONS:-eager compiled}; do
  .venv/bin/python scratch/cir271/node1.py "$OUT" "$ex" "${LAYERS:-}" 2>&1 | tee "$OUT/$ex-o${OUTPUTS:-32}.log" | grep -E '^REQBLOCK|Error|error' | cut -c1-2000
  r=${PIPESTATUS[0]}; [ "$r" = 0 ] || rc=$r
  echo "REQBLOCK $ex rc=$r at $(date -u +%H:%M:%SZ)"
done
exit $rc
