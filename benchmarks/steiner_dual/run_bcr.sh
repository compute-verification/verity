#!/usr/bin/env bash
# CMP-8 worker A1b: exact UCR and BCR values for <2,2,2> over F2 (bcr.py). Scratch venv in a mktemp dir.
# Run from the clone's root: bash benchmarks/steiner_dual/run_bcr.sh   (writes nothing outside the scratch dir and $OUT)
set -u
d=$(mktemp -d)
trap 'rm -rf "$d"' EXIT
PY=python3
if ! python3 -c "import numpy, scipy" 2>/dev/null; then
  if command -v uv >/dev/null; then
    UV_CACHE_DIR="$d/uvcache" uv venv -q "$d/venv" --python 3.12 \
      && UV_CACHE_DIR="$d/uvcache" uv pip install -q --python "$d/venv/bin/python" numpy scipy \
      && PY="$d/venv/bin/python"
  else
    python3 -m pip install -q --target "$d/lib" numpy scipy && export PYTHONPATH="$d/lib"
  fi
fi
"$PY" -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
cd "$(dirname "$0")"
for a in "2 1 2" "1 2 2" "2 2 2 ucr" "2 2 2"; do
  /usr/bin/time -v "$PY" bcr.py $a 2>&1 | grep -v -E "^\s+(Average|Socket|Signals|Swaps|File system|Page size|Exit status|Minor|Major|Voluntary|Involuntary|Percent|System time|User time)"
done
