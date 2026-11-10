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
