"""Run a candidate's train.py with the held-out sets unreadable from its process (fixed).

    python guard.py ISLAND/train.py ARGS...

An audit hook, which Python code cannot remove, refuses every open, listing or stat under `prepare.HELD_OUT` and logs each
subprocess the candidate starts to `$CW_GUARD_LOG`. A subprocess runs without the hook; program.md forbids reading the held-out
sets from one, and a candidate that does is a loophole to report.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare  # noqa: E402

FORBIDDEN = str(prepare.HELD_OUT.resolve())
LOG = os.environ.get("CW_GUARD_LOG")
PATH_EVENTS = {"open", "os.listdir", "os.scandir", "os.stat", "glob.glob", "os.chdir", "os.symlink", "os.link"}


def _hook(event: str, args: tuple) -> None:
    if event in PATH_EVENTS and args:
        p = args[0]
        if isinstance(p, (str, bytes, os.PathLike)):
            try:
                s = os.path.realpath(os.fsdecode(p))
            except (TypeError, ValueError):
                return
            if s == FORBIDDEN or s.startswith(FORBIDDEN + os.sep):
                raise PermissionError(f"guard: {s} is held out; only evaluate.py reads it")
    elif event in ("subprocess.Popen", "os.system", "os.exec", "os.posix_spawn") and LOG:
        with open(LOG, "a") as f:
            f.write(f"{event} {args[:2]!r}\n")


def main() -> int:
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    sys.addaudithook(_hook)
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
