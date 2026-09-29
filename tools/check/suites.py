"""`suites`: every test suite in the repository, each run in its own process from its own directory, with passes cached.

    uv run tools/check/suites.py                     every suite whose inputs changed since it last passed here
    uv run tools/check/suites.py verity flock        only these suites (a package name, a directory, or a unique part of one)
    uv run tools/check/suites.py --quick             without the tests marked `slow`
    uv run tools/check/suites.py --fresh             run even where a pass is cached
    uv run tools/check/suites.py --list              each suite, its inputs, and whether a pass is cached
    uv run tools/check/suites.py flock -- -k tanh    arguments after `--` go to pytest

A suite is a directory whose `pyproject.toml` configures pytest (`[tool.pytest.ini_options]`). Its inputs are its own directory,
the directories of the workspace packages it depends on (its `[project]` dependencies, transitively), the paths it lists under
`[tool.verity.tests] inputs`, and the environment: the installed distributions, Python, the platform, whether `lake`, `cargo`
and `nvidia-smi` are on the PATH, and the `VERITY_*` and `FLOCK_*` variables tests switch on. The cache key hashes all of them,
read from git's view of the working tree, so uncommitted edits count; a tree without .git (what `research run --source` ships)
is keyed from the sha256 of every file in it. A pass is kept under its key in `~/.cache/verity/tests`
(`VERITY_TEST_CACHE` overrides it) and reused while the key is unchanged.

Every run loads `guard/verity_suite_guard.py`, which fails the suite if its tests open a repository file outside its inputs,
so a cached pass never hides a change to something the tests read. Programs a test starts as a subprocess are not traced:
list what they read under `inputs`.

`--jobs` is how many test processes run at once. A suite that declares `[tool.verity.tests] parallel = true` spreads its tests
over all of them with pytest-xdist, whose workers load the guard too. Parallel suites run first, one at a time, then the
others side by side, longest first. Without pytest-xdist installed, a parallel suite runs in one process.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GUARD = HERE / "guard"
CACHE = Path(os.environ.get("VERITY_TEST_CACHE") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "verity" / "tests")
#: Files every key includes besides the workspace's metadata (every pyproject.toml and uv.lock): the runner itself.
RUNNER = ("tools/check/suites.py", "tools/check/guard/verity_suite_guard.py")
XDIST = importlib.util.find_spec("xdist") is not None
TOOLS = ("lake", "cargo", "nvidia-smi")
ENV_PREFIXES = ("VERITY_", "FLOCK_")
QUICK = ["-m", "not slow"]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(req: str) -> str:
    return _norm(re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req).group(1))


def cpus() -> int:
    """The CPUs this process may use: its affinity (a container's cpuset) and its cgroup's quota. A pod's `os.cpu_count()` is the
    host's (384 on the check pods, which have 8)."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 2
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(quota) // int(period)))
    except (OSError, ValueError):
        pass
    return n


def _git(*args: str, env: dict | None = None, stdin: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, env=env, input=stdin, capture_output=True, text=True, check=True).stdout


#: What a machine adds to a shipped tree (`research run --source` ships a `git archive`: no .git) besides the commit's files.
_NOT_SOURCE = {".git", ".venv", "__pycache__", ".lake", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}


def _in_git() -> bool:
    p = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True)
    return p.returncode == 0 and Path(p.stdout.strip()).resolve() == ROOT


#: Whether this tree is a git checkout; a shipped tree is keyed from its files instead.
GIT = _in_git()


def _files(path: str) -> list[str]:
    """The source files under `path` (relative to the root) in a tree without git."""
    base = ROOT / path
    if base.is_file():
        return [path]
    out = []
    for d, dirs, files in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x not in _NOT_SOURCE)
        out += [os.path.relpath(os.path.join(d, f), ROOT) for f in sorted(files)]
    return [f for f in out if f != "READY.json"]


def digests() -> dict[str, str]:
    """Every source file's sha256, in a tree without git. Files .gitignore covers count too: eight tracked fixtures match it."""
    return {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in _files(".")}


def changed(before: dict[str, str]) -> list[str]:
    """In a tree without git, what a run changed since `before` (its `digests()`): files modified or removed, and new files the
    tree's .gitignore files do not cover, as git's view of a checkout would show them."""
    after = digests()
    new = sorted(set(after) - set(before))
    ignored: set[str] = set()
    if new:
        with tempfile.TemporaryDirectory() as tmp:        # a repository with nothing tracked answers from the ignore rules alone
            subprocess.run(["git", "init", "-q", tmp], capture_output=True, check=True)
            p = subprocess.run(["git", f"--git-dir={tmp}/.git", f"--work-tree={ROOT}", "check-ignore", "--stdin"], cwd=ROOT,
                               input="\n".join(new) + "\n", capture_output=True, text=True)
            ignored = set(p.stdout.splitlines())
    return sorted({f for f in before if after.get(f) != before[f]} | (set(new) - ignored))


@dataclass(frozen=True)
class Suite:
    name: str
    path: str                    # relative to the repository root; "." for the repository's own suite
    requires: tuple[str, ...]    # names of the distributions it depends on
    inputs: tuple[str, ...]      # paths it declares under [tool.verity.tests] inputs
    parallel: bool = False       # [tool.verity.tests] parallel: its tests may run in several processes at once


def _pyproject(path: str) -> dict:
    return tomllib.loads((ROOT / path / "pyproject.toml").read_text())


def _requires(project: dict, data: dict) -> tuple[str, ...]:
    reqs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        reqs += extra
    for group in data.get("dependency-groups", {}).values():
        reqs += [r for r in group if isinstance(r, str)]
    return tuple(sorted({_requirement_name(r) for r in reqs}))


def members() -> dict[str, str]:
    """The workspace packages: {distribution name: directory}."""
    out = {}
    for path in _pyproject(".").get("tool", {}).get("uv", {}).get("workspace", {}).get("members", []):
        out[_norm(_pyproject(path)["project"]["name"])] = path
    return out


def metadata() -> list[str]:
    """The workspace's metadata, an input of every suite: every pyproject.toml, and the lockfile."""
    if GIT:
        listed = _git("ls-files", "--cached", "--others", "--exclude-standard", "--", "pyproject.toml", "*/pyproject.toml").split()
    else:
        listed = [f for f in _files(".") if os.path.basename(f) == "pyproject.toml"]
    return sorted(set(listed)) + ["uv.lock"]


def discover() -> list[Suite]:
    suites = []
    for f in metadata()[:-1]:
        path = str(Path(f).parent)
        data = _pyproject(path)
        if "pytest" not in data.get("tool", {}):
            continue
        project = data.get("project", {})
        name = "repository" if path == "." else _norm(project["name"]) if project else path
        tests = data["tool"].get("verity", {}).get("tests", {})
        suites.append(Suite(name, path, _requires(project, data), tuple(tests.get("inputs", [])), tests.get("parallel", False) is True))
    return suites


def closure(suite: Suite, pkgs: dict[str, str]) -> list[str]:
    """Directories of the workspace packages `suite` depends on, directly or through another."""
    seen: dict[str, str] = {}
    todo = [r for r in suite.requires if r in pkgs]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen[name] = pkgs[name]
        data = _pyproject(pkgs[name])
        todo += [r for r in _requires(data["project"], data) if r in pkgs]
    return sorted(p for p in seen.values() if p != suite.path)


def worktree() -> str | None:
    """The tree git would commit now: tracked files as they are on disk, plus untracked files that aren't ignored. None
    without git."""
    if not GIT:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "index"
        real = ROOT / _git("rev-parse", "--git-path", "index").strip()
        if real.exists():
            shutil.copyfile(real, index)
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        _git("add", "-A", env=env)
        return _git("write-tree", env=env).strip()


def object_ids(tree: str | None, paths: list[str], files: dict[str, str] | None = None) -> dict[str, str]:
    """Each path's git object id in `tree`, or without git a sha256 over the digests (`files`) of the files under it."""
    if tree is None:
        out = {}
        for p in paths:
            under = sorted(f for f in files if p == "." or f == p or f.startswith(p + "/"))
            out[p] = hashlib.sha256("".join(f"{f}\0{files[f]}\n" for f in under).encode()).hexdigest() if under else "absent"
        return out
    specs = [tree if p == "." else f"{tree}:{p}" for p in paths]
    out = _git("cat-file", "--batch-check=%(objectname)", stdin="\n".join(specs) + "\n").splitlines()
    return {p: ("absent" if line.endswith(" missing") else line) for p, line in zip(paths, out)}


def environment() -> dict:
    dists = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
            "distributions": hashlib.sha256("\n".join(dists).encode()).hexdigest(),
            "tools": {t: shutil.which(t) is not None for t in TOOLS},
            "variables": {k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES) and k != "VERITY_TEST_CACHE"}}


def inputs_of(suite: Suite, pkgs: dict[str, str]) -> list[str]:
    """Its directory, its dependencies' directories, and its declared inputs; a declared input that is a workspace package
    brings that package's dependencies too."""
    paths = {suite.path, *closure(suite, pkgs), *suite.inputs}
    by_path = {p: n for n, p in pkgs.items()}
    for p in suite.inputs:
        if p in by_path:
            paths.update(closure(Suite(by_path[p], p, (by_path[p],), ()), pkgs))
    return sorted(paths)


def key(suite: Suite, ids: dict[str, str], env: dict, args: list[str]) -> str:
    blob = json.dumps({"suite": suite.path, "inputs": ids, "environment": env, "pytest": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def cached(k: str) -> dict | None:
    f = CACHE / f"{k}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _durations() -> dict[str, float]:
    f = CACHE / "durations.json"
    return json.loads(f.read_text()) if f.exists() else {}


@dataclass
class Result:
    suite: Suite
    status: str                  # passed, cached, failed
    seconds: float = 0.0
    counts: dict | None = None
    outside: list | None = None
    unmarked_slow: list | None = None
    log: Path | None = None
    since: dict | None = None    # for a cached pass, the run it came from
    key: str = ""
    workers: int = 1
    unguarded: list | None = None    # xdist workers that sent the guard no record
    reported: bool = True            # whether the guard wrote its report


#: pytest with the guard loaded by name (`-p`), which xdist repeats in each worker, from `guard/`: appended to sys.path, and
#: holding that one module only, so it shadows nothing a suite imports.
_PYTEST = "import sys, pytest; sys.path.append(sys.argv.pop(1)); sys.exit(pytest.main(sys.argv[1:]))"
#: xdist's scheduling: an idle worker takes queued tests from a busy one, which evens out suites whose tests differ in length.
DIST = ["--dist", "worksteal"]


def run(suite: Suite, allowed: list[str], args: list[str], logs: Path, workers: int = 1) -> Result:
    slug = suite.path.replace("/", "_") if suite.path != "." else "repository"
    log, report = logs / f"{slug}.log", logs / f"{slug}.json"
    report.unlink(missing_ok=True)
    env = dict(os.environ, VERITY_SUITE_ROOT=str(ROOT), VERITY_SUITE_ALLOW=json.dumps(allowed), VERITY_SUITE_REPORT=str(report))
    spread = ["-n", str(workers), *DIST] if workers > 1 else []
    start = time.monotonic()
    with log.open("w") as out:
        rc = subprocess.run([sys.executable, "-c", _PYTEST, str(GUARD), "-p", "verity_suite_guard", "-p", "no:cacheprovider",
                             *spread, *args], cwd=ROOT / suite.path, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
    seconds = time.monotonic() - start
    rep = json.loads(report.read_text()) if report.exists() else None
    passed = rc == 0 and rep is not None and not rep["outside"] and not rep["unguarded"]
    rep = rep or {}
    return Result(suite, "passed" if passed else "failed", seconds, rep.get("counts"), rep.get("outside"),
                  rep.get("unmarked_slow"), log, workers=workers, unguarded=rep.get("unguarded"), reported=bool(rep))


class Slots:
    """`--jobs` test processes, handed out in the order the suites were planned: a suite waits for its turn, then for as many
    free slots as it runs processes."""

    def __init__(self, n: int):
        self.free, self.turn, self.cond = n, 0, threading.Condition()

    @contextmanager
    def take(self, turn: int, n: int):
        with self.cond:
            self.cond.wait_for(lambda: self.turn == turn and self.free >= n)
            self.free -= n
            self.turn += 1
            self.cond.notify_all()
        try:
            yield
        finally:
            with self.cond:
                self.free += n
                self.cond.notify_all()


def select(suites: list[Suite], names: list[str]) -> list[Suite]:
    if not names:
        return suites
    chosen = []
    for n in names:
        exact = [s for s in suites if n in (s.name, s.path, s.path.rstrip("/"))]
        hits = exact or [s for s in suites if n in s.name or n in s.path]
        if len(hits) != 1:
            sys.exit(f"suites: {n!r} matches {len(hits)} suites: {', '.join(s.name for s in hits) or 'none'} (see --list)")
        chosen += hits
    return list(dict.fromkeys(chosen))


def _count(r: Result) -> str:
    c = r.counts or {}
    return ", ".join(f"{c[k]} {k}" for k in ("passed", "failed", "error", "skipped") if c.get(k)) or "no tests"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(prog="suites", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="*", help="suites to run (default: all)")
    ap.add_argument("--quick", action="store_true", help="deselect the tests marked slow")
    ap.add_argument("--fresh", action="store_true", help="run even where a pass is cached")
    ap.add_argument("--list", action="store_true", help="list the suites, their inputs, and whether a pass is cached")
    ap.add_argument("--skip", action="append", default=[], metavar="SUITE", help="leave this suite out (repeatable)")
    ap.add_argument("--jobs", "-j", type=int, default=cpus(),
                    help="test processes at once, a parallel suite's xdist workers included (default: the CPUs this process may use)")
    ap.add_argument("--logs", type=Path, help="write each suite's log and report here (default: under the cache)")
    ap.add_argument("--json", type=Path, help="write the outcome of every selected suite here")
    ap.add_argument("--commit", help="the commit these files are, in a tree without git (check passes RESEARCH_SOURCE_SHA)")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)

    pkgs, suites = members(), discover()
    skipped = select(suites, a.skip) if a.skip else []
    chosen = [s for s in select(suites, a.suites) if s not in skipped]
    args = (QUICK if a.quick else []) + extra
    tree, env = worktree(), environment()
    files = None if GIT else digests()
    plan = []
    meta = metadata()
    for s in chosen:
        paths = inputs_of(s, pkgs)
        ids = object_ids(tree, sorted(set(paths) | set(meta) | set(RUNNER)), files)
        keys = [key(s, ids, env, args)] + ([key(s, ids, env, extra)] if a.quick else [])   # a full pass also covers --quick
        plan.append((s, paths, keys, next((c for c in map(cached, keys) if c), None)))

    if a.list:
        for s, paths, keys, hit in plan:
            state = f"cached from {hit['commit'][:10]} {hit['when']}" if hit else "not cached"
            print(f"{s.name:28} {s.path:28} {state}\n    inputs: {', '.join(paths)}")
        return 0

    logs = a.logs or CACHE / "logs" / f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    logs.mkdir(parents=True, exist_ok=True)
    head = _git("rev-parse", "HEAD").strip() if GIT else a.commit or "unknown"
    dirty = bool(_git("status", "--porcelain").strip()) if GIT else False
    results: list[Result] = [Result(s, "cached", since=hit, key=hit["key"]) for s, _, _, hit in plan if hit and not a.fresh]
    todo = [(s, paths, keys) for s, paths, keys, hit in plan if a.fresh or not hit]
    known = _durations()
    todo.sort(key=lambda t: (not t[0].parallel, -known.get(t[0].path, float("inf"))))
    for r in results:
        print(f"cached  {r.suite.name:28} passed at {r.since['commit'][:10]} {r.since['when']}")
    for s in skipped:
        print(f"skipped {s.name:28} (--skip)")
    jobs = max(1, a.jobs)
    spread = jobs if XDIST else 1
    if spread == 1 and jobs > 1 and any(s.parallel for s, _, _ in todo):
        print("pytest-xdist is not installed: each parallel suite runs in one process")
    slots = Slots(jobs)

    def start(turn: int, s: Suite, paths: list[str]) -> Result:
        n = spread if s.parallel else 1
        with slots.take(turn, n):
            return run(s, paths + meta + list(RUNNER), args, logs, n)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(todo))) as pool:
        futures = {pool.submit(start, i, s, paths): (s, keys[0]) for i, (s, paths, keys) in enumerate(todo)}
        for fut in concurrent.futures.as_completed(futures):
            s, k = futures[fut]
            r = fut.result()
            r.key = k
            results.append(r)
            known[s.path] = r.seconds
            print(f"{r.status:7} {s.name:28} {_count(r)} in {r.seconds:.0f} s" + (f" on {r.workers} workers" if r.workers > 1 else ""))
            if r.status == "passed":
                CACHE.mkdir(parents=True, exist_ok=True)
                (CACHE / f"{k}.json").write_text(json.dumps({"suite": s.path, "key": k, "commit": head, "dirty": dirty,
                                                              "when": time.strftime("%Y-%m-%d %H:%M"), "counts": r.counts,
                                                              "seconds": round(r.seconds, 1)}, indent=1) + "\n")
            elif r.outside:
                print(f"        its tests read {len(r.outside)} files outside its inputs, for example {', '.join(r.outside[:3])}:"
                      f" depend on the package, or list the path under [tool.verity.tests] inputs in {s.path}/pyproject.toml")
            elif r.unguarded:
                print(f"        xdist workers {', '.join(r.unguarded)} sent the guard no record of what they read\n        log: {r.log}")
            else:
                if not r.reported:
                    print("        the guard wrote no report, so nothing vouches for what the tests read")
                print("        " + "\n        ".join(r.log.read_text().splitlines()[-25:]) + f"\n        log: {r.log}")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "durations.json").write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
    if not todo:
        written = []
    elif GIT:
        written = _git("diff-tree", "-r", "--name-only", tree, worktree()).split()
    else:
        written = changed(files)
    if written:
        print(f"\nthe tests changed {len(written)} files in the repository (tests write under tmp_path):\n  " + "\n  ".join(written[:20]))

    slow = [(r.suite.name, t, s) for r in results for t, s in (r.unmarked_slow or [])]
    if slow:
        print(f"\n{len(slow)} tests took over 5 s without the `slow` mark, so --quick still runs them:")
        for name, test, secs in sorted(slow, key=lambda x: -x[2])[:10]:
            print(f"  {secs:6.1f} s  {name}: {test}")
    failed = [r for r in results if r.status == "failed"]
    ran = sum(r.seconds for r in results)
    print(f"\n{len(results)} suites: {len(results) - len(failed)} passed ({sum(r.status == 'cached' for r in results)} cached),"
          f" {len(failed)} failed; {ran / 60:.1f} min of test time; logs in {logs}")
    if a.json:
        a.json.write_text(json.dumps([{"suite": r.suite.path, "name": r.suite.name, "status": r.status, "key": r.key,
                                       "seconds": round(r.seconds, 1), "workers": r.workers, "counts": r.counts,
                                       "outside": r.outside, "unguarded": r.unguarded, "since": r.since} for r in results]
                                     + [{"suite": s.path, "name": s.name, "status": "skipped"} for s in skipped], indent=1) + "\n")
    return 1 if failed or written else 0


if __name__ == "__main__":
    sys.exit(main())
