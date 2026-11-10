#!/bin/bash
# reads only: who owns each check cache directory under /workspace/research/cache, when its top-level entries last changed, and
# which users' ~/.cache/verity-check point where
for d in /workspace/research/cache/verity-check*; do
  stat -c "%n owner=%U mtime=%y" "$d" "$d"/* 2>/dev/null | cut -c1-150
done
for h in /root /home/*; do
  printf "%s/.cache/verity-check: " "$h"
  if [ -L "$h/.cache/verity-check" ]; then readlink "$h/.cache/verity-check"; elif [ -d "$h/.cache/verity-check" ]; then echo "a directory"; else echo "absent or unreadable"; fi
done
id -un
# every link up to three levels under the research user's check cache, and what it points at
find /home/research/.cache/verity-check/ -maxdepth 3 -type l -printf 'link %P -> %l\n' 2>/dev/null | sort | head -200
# the root disk's largest users beside the check cache, one level under the research user's home and its ~/.cache
du -x -B1G --max-depth=1 /home/research 2>/dev/null | sort -rn | head -8
du -x -B1G --max-depth=1 /home/research/.cache 2>/dev/null | sort -rn | head -8
