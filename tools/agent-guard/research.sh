#!/bin/sh
# /usr/local/bin/research on an agent VM (install.sh): runs origin/main's tools/research/src/research/at_main.py, which
# runs this checkout's research or, when the checkout's is only behind, origin/main's. The checkout is the one the
# command runs in, else /workspace. Without origin/main's at_main.py it runs the checkout's research.
ws=$(git rev-parse --show-toplevel 2>/dev/null)
[ -n "$ws" ] && [ -f "$ws/tools/research/src/research/cli.py" ] || ws=/workspace
py=$ws/.venv/bin/python
[ -x "$py" ] || py=python3
if code=$(git -C "$ws" show origin/main:tools/research/src/research/at_main.py 2>/dev/null) && [ -n "$code" ]; then
  RESEARCH_CHECKOUT=$ws exec "$py" -c "$code" "$@"
fi
PYTHONPATH=$ws/tools/research/src${PYTHONPATH:+:$PYTHONPATH} RESEARCH_AT_CHECKOUT=1 exec "$py" -m research "$@"
