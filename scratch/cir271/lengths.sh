set -uo pipefail
# CIR-271 part 3 (REQBLOCK): window_circuit's class Program under class G, and the padded form against the (256, 31) prefill.
uv sync -q --all-packages --extra torch-cpu || { echo "REQBLOCK uv sync failed"; exit 2; }
mkdir -p "$OUT"
echo "REQBLOCK lengths tree $(git rev-parse --short=12 HEAD 2>/dev/null || echo '?') on $(hostname) from $(date -u +%H:%M:%SZ) layers=${LAYERS:-all}"
.venv/bin/python scratch/cir271/lengths.py "$OUT" "${LAYERS:-}" 2>&1 | tee "$OUT/lengths.log" | grep -E '^REQBLOCK' | cut -c1-2600
r=${PIPESTATUS[0]}
echo "REQBLOCK lengths rc=$r at $(date -u +%H:%M:%SZ)"
exit $r
