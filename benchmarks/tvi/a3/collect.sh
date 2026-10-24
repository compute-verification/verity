#!/usr/bin/env bash
# Fetch A3's runs and aggregate every result they copied into $OUT/results. Usage: collect.sh DEST RUN...
set -euo pipefail
dest=$1; shift
mkdir -p "$dest/results"
for r in "$@"; do
  timeout 120 research fetch "$r" --on vy-nebius-4 --all >/dev/null 2>&1 || true
  cp -n ~/.research/runs/"$r"/out/results/*.json "$dest/results/" 2>/dev/null || true
  [ -f ~/.research/runs/"$r"/out/base_pass64.json ] && cp ~/.research/runs/"$r"/out/base_pass64.json "$dest/"
done
py=${PY:-/tmp/tokvenv/bin/python}
"$py" "$(dirname "$0")/aggregate.py" --results "$dest/results" --base "$dest/base_pass64.json" --out "$dest" --node vy-nebius-4
