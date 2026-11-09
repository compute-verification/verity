# The window Program this tree declares for LLAMA32_1B_INT8_K2048 at T 512 under remote_run.sh's protocols config, at
# S 1, 4 and 16 (served_verdicts.py program: no checkpoint, no GPU), to hold against the gate's model registry.
set -uo pipefail
uv sync -q --all-packages --extra torch-cpu || exit 1
T=$(mktemp -d)
echo "TREE $(git rev-parse HEAD 2>/dev/null || echo "${RESEARCH_TREE_SHA:-unknown}")"
for s in 1 4 16; do
  uv run python benchmarks/one_stage/served_verdicts.py program "$T/p$s" --rows 512 --steps "$s" \
    --protocols-config '{"pouw": {"ks": [2048]}}' 2>&1 | tail -1
  d=$(grep -o -E '"program_digest": *"[0-9a-f]{128}"' "$T/p$s/window-program.meta.json" | head -1 | grep -o -E '[0-9a-f]{128}')
  if [ -n "$d" ]; then echo "DIGEST S $s $d"; else echo "DIGEST S $s unread:"; ls -la "$T/p$s"; fi
done
rm -rf "$T"
