set -uo pipefail
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu || exit 3
git log -1 --format='tree at %H' 2>/dev/null || true
ls -la /etc/vy/runs-via-slot 2>&1 | cut -c1-120
cd tools/research && uv run -q pytest tests/test_auto.py -q -p no:cacheprovider 2>&1 | tee "$OUT/test-auto.txt" | tail -120
echo "PYTEST exit ${PIPESTATUS[0]}"
exit 0
