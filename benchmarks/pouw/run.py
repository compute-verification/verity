"""The PoUW benchmarks' front door: ``benchmarks/pouw/run.py NAME [ARGS]`` runs ``benchmarks/pouw/NAME/``'s runner with
ARGS, which is what ``verity bench pouw NAME [ARGS]`` does. A directory's runner is its ``run.sh``, else its ``run.py``.
Given no NAME, or one that has no runner, it names the directories that have one.

The runner takes this process's place, so its exit status, its output and a signal sent to the job are the runner's
own. A runner that needs a GPU or a whole node is the command of a ``research run``, as its header says. Standard
library only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
#: a runner's file name with the program that starts it; a directory's runner is the first of these it holds
RUNNERS = (("run.sh", "bash"), ("run.py", sys.executable))


def runners(here: Path = HERE) -> dict[str, list[str]]:
    """Each directory under ``here`` that has a runner, by name, with the command that starts it."""
    found = {}
    for d in sorted(p for p in here.iterdir() if p.is_dir()):
        for file, program in RUNNERS:
            if (d / file).is_file():
                found[d.name] = [program, str(d / file)]
                break
    return found


def main(argv: list[str]) -> None:
    known = runners()
    name = argv[0].replace("-", "_") if argv else ""
    if name not in known:
        raise SystemExit("benchmarks/pouw/run.py NAME [ARGS] (a - in NAME is a _): NAME is one of " + ", ".join(known))
    command = [*known[name], *argv[1:]]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main(sys.argv[1:])
