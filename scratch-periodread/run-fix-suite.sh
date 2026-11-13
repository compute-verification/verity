set -uo pipefail
export PATH="$HOME/.elan/bin:$HOME/.local/bin:$PATH"
uv sync -q --all-packages --extra torch-cpu || exit 3
git log -1 --format='tree at %H' 2>/dev/null || true
uv run tools/verity/check/suites.py research --quick --fresh 2>&1 | tee "$OUT/suite-research.txt" | tail -60
echo "SUITES exit ${PIPESTATUS[0]}"
exit 0
