#!/usr/bin/env bash
# watch.sh RUN...: fetch each (running or not) and print its memory line
cd /agent/repos/verity || exit 1
for r in "$@"; do
  timeout 300 research fetch "$r" --all >/dev/null 2>&1
  python3 /tmp/cir269/peak.py "$r"
done
