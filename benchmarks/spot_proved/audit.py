"""a11.py's audit (``--load-values``) with the registration window naming the replay's interior commitment.

    python audit.py <a11.py's arguments, with --load-values DIR and --kinds DIR/kinds.json>

DIR is stage.py's output: the sampled proof blocks of one replayed step, opened from the capture, checked against the
interior root, and held to the captured outputs. ``DIR/window.json`` names the replay block and its roots.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "one_stage"))

import a11  # noqa: E402


def main(argv: list[str]) -> int:
    d = Path(argv[argv.index("--load-values") + 1])
    w = json.loads((d / "window.json").read_text())
    a11.WINDOW_OF = w["of"]
    a11.LINK = w["link"]
    a11.ROW = w["row"]
    sys.argv = ["a11.py", *argv]
    return a11.main()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
