# The window Program this tree declares for LLAMA32_1B_INT8_K2048 at T 512 under remote_run.sh's protocols config, at
# S 1, 4 and 16 (served_verdicts.py program: no checkpoint, no GPU), to hold against the gate's model registry.
set -uo pipefail
uv sync -q --all-packages --extra torch-cpu || exit 1
T=$(mktemp -d)
echo "TREE $(git rev-parse HEAD 2>/dev/null || echo "${RESEARCH_TREE_SHA:-unknown}")"
for s in 1 4 16; do
  uv run python benchmarks/one_stage/served_verdicts.py program "$T/p$s" --rows 512 --steps "$s" \
    --protocols-config '{"pouw": {"ks": [2048]}}' 2>&1 | tail -1
  python3 -c 'import json, sys; print("DIGEST S", sys.argv[2], json.load(open(sys.argv[1]))["program_digest"])' \
    "$T/p$s/window-program.meta.json" 2>/dev/null || ls "$T/p$s"
done
rm -rf "$T"
