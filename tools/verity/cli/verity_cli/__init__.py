"""`verity`: the repository's one command, dispatching to the tools beside it.

    verity check [ARGS]             the check a merge requires (check/check.py)
    verity check tests [ARGS]       the test suites whose inputs changed (check/suites.py)
    verity check lean [ARGS]        the Lean check (lean/audit.py)
    verity check circuits [ARGS]    the property checks every circuit passes (circuit-check)
    verity check move BASE HEAD     certify a pure move (check/move_check.py)
    verity bench NAME [ARGS]        benchmarks/NAME's runner (run.py or run.sh)
    verity check --remote POD       `check` on POD, recorded as an Attempt (check.py --record --on POD)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
ROOT = HERE.parents[1]
CHECK_PARTS = {"tests": HERE / "check" / "suites.py", "lean": HERE / "lean" / "audit.py",
               "move": HERE / "check" / "move_check.py"}


def _run(argv: list[str]) -> int:
    return subprocess.run(argv, cwd=ROOT).returncode


def _check(args: list[str]) -> int:
    if args and args[0] == "circuits":
        return _run([sys.executable, "-c", "import sys; from circuit_check.cli import main; sys.exit(main())", *args[1:]])
    if args and args[0] in CHECK_PARTS:
        return _run([sys.executable, str(CHECK_PARTS[args[0]]), *args[1:]])
    if "--remote" in args:
        i = args.index("--remote")
        if i + 1 >= len(args):
            raise SystemExit("verity check --remote POD: name the pod")
        args = [*args[:i], "--record", "--on", args[i + 1], *args[i + 2:]]
    return _run([sys.executable, str(HERE / "check" / "check.py"), *args])


def _bench(args: list[str]) -> int:
    names = sorted(p.name for p in (ROOT / "benchmarks").iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
    if not args or args[0] not in names:
        raise SystemExit("verity bench NAME [ARGS]: NAME is one of " + ", ".join(names))
    d = ROOT / "benchmarks" / args[0]
    for runner, argv in (("run.py", [sys.executable]), ("run.sh", ["bash"])):
        if (d / runner).exists():
            return _run([*argv, str(d / runner), *args[1:]])
    raise SystemExit(f"benchmarks/{args[0]} has no run.py or run.sh: run its tool through `research run`")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "check":
        return _check(args[1:])
    if args[0] == "bench":
        return _bench(args[1:])
    raise SystemExit(f"verity: no command {args[0]!r}\\n{__doc__}")


if __name__ == "__main__":
    sys.exit(main())
