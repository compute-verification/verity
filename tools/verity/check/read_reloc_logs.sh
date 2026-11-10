#!/bin/bash
# reads only: the logs and scripts an earlier relocation of this node's check cache left beside it, for the rates it measured
d=/workspace/research/cache/verity-check-root
for f in "$d"/reloc-*.sh "$d"/relocate-*.log "$d"/reloc-*.log; do
  [ -f "$f" ] || continue
  echo "=== $(basename "$f") $(stat -c '%s bytes, %y' "$f" | cut -c1-40) lines $(wc -l < "$f")"
  head -n 45 "$f" | cut -c1-220
  echo "--- tail"
  tail -n 12 "$f" | cut -c1-220
done
