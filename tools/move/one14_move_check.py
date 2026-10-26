"""`move_check` for ONE-14, the protocols' circuit builders' move to verity.ml.protocols: BASE..HEAD with `--map` extras
from one14_map.toml.

`pouw/circuit/` moves whole but for `live_reads`, which stays in core, so no rename pairs the directory, and a reference
to it (a suite's inputs, PROTOCOL.md) follows the circuits as the map's `[split]` says. Each `[paths]` entry is an extra.

    python3 tools/move/one14_move_check.py BASE HEAD [--json]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from defs_move_check import extras

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    check = HERE.parent / "verity" / "check" / "move_check.py"
    os.execv(sys.executable, [sys.executable, str(check), *sys.argv[1:],
                              *(f"--map={o}={n}" for o, n in extras(HERE / "one14_map.toml").items())])
