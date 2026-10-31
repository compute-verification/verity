"""`verity`: the repository's one command, dispatching to the tools beside it.

    verity check [ARGS]             the whole check, run on `main` after each landing (check/check.py)
    verity check PART... [ARGS]     only those parts of it, tests, circuits or lean (their jobs in check/ci.toml)
    verity check --record [ARGS]    the check recorded as an Attempt (check.py --record; a lane's own: --on M --pool NAME)
    verity check move BASE HEAD     certify a pure move (check/move_check.py)
    verity bench NAME [ARGS]        benchmarks/NAME's runner (run.py or run.sh)
    verity run CONFIG [--party ROLE ...] [--on MACHINE] [--periods N] [--state DIR]
                                    a deployment configuration's parties on this machine, and its verdict (run.py)
    verity config new|check|hash    write, check or hash a deployment configuration (run.py)
    verity example NAME [ARGS]      examples/NAME's command (`verity run` its config.toml, else its run.py or run.sh): its
                                    workload on its placement
    verity example --status         every example's protocol, workload, run and proof, and each one's next missing evidence
    verity lean try FILE            elaborate one Lean file, the edit loop: `lake env lean FILE` in the Lake package that
                                    holds it, exiting with lean's status (on a project machine with --on NODE [--pool P])
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
ROOT = HERE.parents[1]
LAKEFILES = ("lakefile.lean", "lakefile.toml")
USAGE = "verity lean try FILE [--on NODE [--pool P] [--as LANE]]"
#: the node whose pool `verity lean try --on` takes when --pool names none, and that pool
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


def _tree(file: str) -> Path:
    """The checkout `verity lean try` ships: the one ``file`` lies in, else the current directory's, else the CLI's own
    (``ROOT``).  ``ROOT`` doesn't come first: a lane in a `git worktree` runs the main checkout's venv, and shipping that tree
    would build its version of the file, or miss the file."""
    p = Path(file)
    where = p.resolve().parent if p.is_file() else Path.cwd()
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=where, capture_output=True, text=True)
    if top.returncode:
        return ROOT
    tree = Path(top.stdout.strip())
    if not (tree / "tools" / "verity" / "lean" / "lean_changed.py").is_file():
        raise SystemExit(f"verity lean try: {tree} is not a Verity checkout (no tools/verity/lean/lean_changed.py)")
    return tree


def _snapshot(tree: Path, tmp: Path) -> Path:
    """A checkout `research run --source` ships as is: ``tree`` when it is clean, else a detached worktree under ``tmp`` of a
    commit of its working tree (untracked files that aren't ignored included), made without touching its branch or index."""
    if not _git("status", "--porcelain", cwd=tree):
        return tree
    index = tmp / "index"
    index.write_bytes((tree / _git("rev-parse", "--git-path", "index", cwd=tree)).read_bytes())
    env = {**os.environ, "GIT_INDEX_FILE": str(index)}
    _git("add", "-A", env=env, cwd=tree)
    commit = _git("commit-tree", _git("write-tree", env=env, cwd=tree), "-p", "HEAD", "-m", "verity lean try: the working tree",
                  cwd=tree)
    _git("worktree", "add", "-q", "--detach", str(tmp / "tree"), commit, cwd=tree)
    return tmp / "tree"


def _lean_here(file: str) -> int:
    """`lake env lean FILE` in the directory of the nearest lakefile above FILE, against that package's built imports."""
    path = (Path(file) if Path(file).is_file() else _tree(file) / file).resolve()
    pkg = next((d for d in path.parents if any((d / name).is_file() for name in LAKEFILES)), None)
    if not path.is_file() or path.suffix != ".lean" or pkg is None:
        raise SystemExit(f"verity lean try: {file} is not a .lean file in a Lake package (a directory with a "
                         f"{' or '.join(LAKEFILES)})")
    where = pkg.relative_to(ROOT) if pkg.is_relative_to(ROOT) else pkg
    lake = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    if not Path(lake).is_file():
        raise SystemExit(f"verity lean try: no `lake` here: `bash tools/verity/lean/setup.sh {where}` installs elan, the "
                         "toolchain and the package's dependencies, then `lake build` there builds its imports")
    rel = os.path.relpath(path, pkg)
    print(f"verity lean try: lake env lean {rel} (in {where})", flush=True)
    return subprocess.run([lake, "env", "lean", rel], cwd=pkg).returncode


def _lean(args: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verity lean try", description="Elaborate one Lean file: `lake env lean FILE` in the Lake "
                                 "package that holds it (the nearest lakefile above FILE), against that package's built "
                                 "imports (`lake build` there first), exiting with lean's status.")
    ap.add_argument("file", metavar="FILE", help="the .lean file: a path from here, or from the root of "
                    "the checkout you're in, which is the tree it elaborates")
    ap.add_argument("--on", metavar="NODE", help="elaborate it on a project machine instead")
    ap.add_argument("--pool", help="with --on: the machine's pool for it")
    ap.add_argument("--as", dest="lane", metavar="LANE", help="with --on: the lane the pool's slot names (default $RESEARCH_LANE)")
    if not args or args[0] != "try":
        raise SystemExit(USAGE)
    a = ap.parse_args(args[1:])
    if a.on is None:
        if a.pool or a.lane:
            ap.error("--pool and --as go with --on")
        return _lean_here(a.file)
    tree = _tree(a.file)
    file = a.file if not Path(a.file).is_file() else os.path.relpath(Path(a.file).resolve(), tree)
    if not (tree / file).is_file():
        raise SystemExit(f"verity lean try: no {file} in {tree}, the checkout it ships")
    pool = a.pool or (TRY_POOL if a.on == TRY_NODE else None)
    research = [sys.executable, "-m", "research"]
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="verity-lean-try-") as tmp:
        source = _snapshot(tree, Path(tmp))
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
            if source != tree:
                subprocess.run(["git", "worktree", "remove", "--force", str(source)], cwd=tree, capture_output=True)
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
    if args[0] == "example" and args[1:] == ["--status"]:
        from verity_cli.examples import status
        return status(ROOT)
    if args[0] == "example":
        if args[1:] and (config := ROOT / "examples" / args[1].replace("-", "_") / "config.toml").is_file():
            from . import run

            return run.run_main([str(config), *args[2:]])
        return _runner("examples", "example", args[1:])
    if args[0] == "lean":
        return _lean(args[1:])
    raise SystemExit(f"verity: no command {args[0]!r}\n{__doc__}")


if __name__ == "__main__":
    sys.exit(main())
