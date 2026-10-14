#!/usr/bin/env bash
set -uo pipefail

export PATH="$HOME/.elan/bin:$PATH"
rc=0

python3 tools/verity/lean/lean_changed.py --records Security --update --out "$OUT/security" \
  > "$OUT/security.log" 2>&1 || rc=1
uv run --locked --extra torch-cpu pytest -q verity/ml/flock/tests/test_rec_vstage.py -k shared_rows \
  > "$OUT/shared-rows-pytest.log" 2>&1 || rc=1

exit "$rc"
