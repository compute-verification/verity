#!/usr/bin/env bash
# Option (b) of card 82c1e288 (`pearlc_rowcap_window.py`): honest acceptance under each widened row cap at k = 256, Gaussian
# bands and every TP8 slice in SLICES of GLM-5.3's down projections (SNAP), a `research run --cwd source` job writing $OUT.
# Extra arguments go to the script (e.g. `--gaussian-runs 1 --bands 1 --slices 1` for a trial).
set -uo pipefail
SNAP=${SNAP:-/workspace/hf/hub/models--zai-org--GLM-5.3/snapshots/aca966e4e02791568aa6a4ced368624b3d897f42}
uv sync -q --all-packages --extra torch-cpu || exit 2
uv run --no-sync python benchmarks/pouw/pearlc_rowcap_window.py --out "$OUT" --snapshot "$SNAP" --jobs "${JOBS:-48}" "$@"
