set -uo pipefail
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu || exit 3
git fetch -q origin main 2>/dev/null || true
uv run tools/verity/check/suites.py --changed origin/main --quick 2>&1 | tee "$OUT/suites-quick.txt" | tail -90
rc=${PIPESTATUS[0]}
echo "SUITES exit $rc"
exit $rc
