set -u
echo "== source $(git rev-parse HEAD)"
uv sync -q --all-packages --extra torch-cpu
uv run tools/verity/check/suites.py --changed c68135b49d63b1dc38ce60d8a3037405c9996066 --quick --fresh --json "$OUT/suites.json" --logs "$OUT/logs" 2>&1 | rg -v "^\s+(delete|warnings\.warn)" | tail -40
echo "suites rc=${PIPESTATUS[0]}"
