"""`verity`: the repository's one command, dispatching to the tools beside it.

    verity check [ARGS]             the whole check, run on `main` after each landing (check/check.py)
    verity check PART... [ARGS]     only those parts of it, tests, circuits or lean (their jobs in check/ci.toml)
    verity check --remote [ARGS]    the check recorded as an Attempt, on a free pod unless --on POD names one
    verity check move BASE HEAD     certify a pure move (check/move_check.py)
    verity bench NAME [ARGS]        benchmarks/NAME's runner (run.py or run.sh)
    verity run CONFIG [--party ROLE ...] [--machine NAME] [--periods N] [--replay DIR] [--state DIR]
                                    a deployment configuration's parties on this machine, and its verdict (run.py)
    verity config new|check|hash    write, check or hash a deployment configuration (run.py)
    verity example NAME [ARGS]      examples/NAME: `verity run` its config.toml, else its run.py or run.sh
    verity lean try FILE [--on NODE] [--pool P] [--as LANE]
                                    elaborate one Lean file on a node's warm slot tree (lean/lean_changed.py --try) and
                                    print Lean's output: the edit loop (default vy-nebius-cpu-1, pool research)
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
ROOT = HERE.parents[1]
#: where `verity lean try` runs when --on names no node, and that node's pool for it
TRY_NODE, TRY_POOL = "vy-nebius-cpu-1", "research"
#: `research run --on`'s line naming the run it started (or refused)
LAUNCHED = re.compile(r"^research: run (\S+) (?:launched|refused) on ", re.M)


def _run(argv: list[str]) -> int:
    return subprocess.run(argv, cwd=ROOT).returncode


def _check(args: list[str]) -> int:
    if args and args[0] == "move":
        return _run([sys.executable, str(HERE / "check" / "move_check.py"), *args[1:]])
    return _run([sys.executable, str(HERE / "check" / "check.py"), *args])


def _runner(top: str, command: str, args: list[str]) -> int:
    """Run ``top/NAME``'s run.py or run.sh with the rest of ``args``."""
    here = ROOT / top
    names = sorted(p.name for p in here.iterdir() if p.is_dir() and not p.name.startswith((".", "_"))) if here.is_dir() else []
    if args:
        args = [args[0].replace("-", "_"), *args[1:]]
    if not args or args[0] not in names:
        raise SystemExit(f"verity {command} NAME [ARGS] (a - in NAME is a _): NAME is one of " + (", ".join(names) or "none"))
    d = here / args[0]
    for runner, argv in (("run.py", [sys.executable]), ("run.sh", ["bash"])):
        if (d / runner).exists():
            return _run([*argv, str(d / runner), *args[1:]])
    raise SystemExit(f"{top}/{args[0]} has no run.py or run.sh: run its tool through `research run`")


def _git(*argv: str, env: dict | None = None, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *argv], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


def _snapshot(tmp: Path) -> Path:
    """A checkout `research run --source` ships as is: this one when it is clean, else a detached worktree under ``tmp`` of a
    commit of its working tree (untracked files that aren't ignored included), made without touching its branch or index."""
    if not _git("status", "--porcelain"):
        return ROOT
    index = tmp / "index"
    index.write_bytes((ROOT / _git("rev-parse", "--git-path", "index")).read_bytes())
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    _git("add", "-A", env=env)
    commit = _git("commit-tree", _git("write-tree", env=env), "-p", "HEAD", "-m", "verity lean try: the working tree")
    _git("worktree", "add", "-q", "--detach", str(tmp / "tree"), commit)
    return tmp / "tree"


def _lean(args: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verity lean try", description="Elaborate one Lean file on a node's warm Lean slot tree "
                                 "(tools/verity/lean/lean_changed.py --try through `research run`) and print Lean's output.")
    ap.add_argument("file", metavar="FILE", help="the .lean file, from the repository's root")
    ap.add_argument("--on", default=TRY_NODE, metavar="NODE", help=f"the node (default {TRY_NODE})")
    ap.add_argument("--pool", help=f"its `research run` pool (default {TRY_POOL} on {TRY_NODE}, else the node's default)")
    ap.add_argument("--as", dest="lane", metavar="LANE", help="the lane the pool's slot names (default $RESEARCH_LANE)")
    if not args or args[0] != "try":
        raise SystemExit("verity lean try FILE [--on NODE] [--pool P] [--as LANE]")
    a = ap.parse_args(args[1:])
    file = a.file if (ROOT / a.file).is_file() else os.path.relpath(Path(a.file).resolve(), ROOT)
    pool = a.pool or (TRY_POOL if a.on == TRY_NODE else None)
    research = [sys.executable, "-m", "research"]
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="verity-lean-try-") as tmp:
        source = _snapshot(Path(tmp))
        try:
            p = subprocess.Popen([*research, "run", "--on", a.on, *(["--pool", pool] if pool else []),
                                  *(["--as", a.lane] if a.lane else []), "--project", "verity",
                                  "--source", str(source), "--cwd", "clone", "--wait", "--wait-every", "5", "--",
                                  "python3", "tools/verity/lean/lean_changed.py", "--try", file],
                                 cwd=ROOT, stderr=subprocess.PIPE, text=True)
            said = []
            for line in p.stderr:
                sys.stderr.write(line)
                said.append(line)
            rc = p.wait()
        finally:
            if source != ROOT:
                subprocess.run(["git", "worktree", "remove", "--force", str(source)], cwd=ROOT, capture_output=True)
    if not (m := LAUNCHED.search("".join(said))):
        print(f"verity lean try: `research run` exited {rc} without starting a run", file=sys.stderr)
        return rc or 1
    run_id = m.group(1)
    subprocess.run([*research, "fetch", run_id, "--on", a.on, "--all"], cwd=ROOT, stdout=subprocess.DEVNULL)
    d = Path(os.environ.get("RESEARCH_RUNS", Path.home() / ".research" / "runs")).expanduser() / run_id
    for name, to in (("stdout.log", sys.stdout), ("stderr.log", sys.stderr)):
        if (d / name).is_file():
            to.write((d / name).read_text(errors="replace"))
    print(f"verity lean try: run {run_id} on {a.on}: exit {rc} in {time.monotonic() - t0:.0f} s wall", flush=True)
    return rc


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if args[0] == "check":
        return _check(args[1:])
    if args[0] == "bench":
        return _runner("benchmarks", "bench", args[1:])
    if args[0] in ("run", "config"):
        from . import run

        return (run.run_main if args[0] == "run" else run.config_main)(args[1:])
    if args[0] == "example":
        if args[1:] and (ROOT / "examples" / args[1] / "config.toml").is_file():
            from . import run

            return run.run_main([str(ROOT / "examples" / args[1] / "config.toml"), *args[2:]])
        return _runner("examples", "example", args[1:])
    if args[0] == "lean":
        return _lean(args[1:])
    raise SystemExit(f"verity: no command {args[0]!r}\\n{__doc__}")


if __name__ == "__main__":
    sys.exit(main())
