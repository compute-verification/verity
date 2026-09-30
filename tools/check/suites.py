"""`suites`: every test suite in the repository, each run in its own process from its own directory, with passes cached.

    uv run tools/check/suites.py                     every suite whose inputs changed since it last passed here
    uv run tools/check/suites.py verity flock        only these suites (a package name, a directory, or a unique part of one)
    uv run tools/check/suites.py --quick             without the tests marked `slow`
    uv run tools/check/suites.py --fresh             run even where a pass is cached
    uv run tools/check/suites.py --list              each suite, its inputs, and whether a pass is cached
    uv run tools/check/suites.py flock -- -k tanh    arguments after `--` go to pytest

A suite is a directory whose `pyproject.toml` configures pytest (`[tool.pytest.ini_options]`). Its inputs are its own directory,
the directories of the workspace packages it depends on (its `[project]` dependencies, transitively), the paths it lists under
`[tool.verity.tests] inputs`, the build outputs it lists under `artifacts` (each file's digest: git ignores them), what it
resolves to in `uv.lock` (the lock's entry for each distribution it depends on, transitively, with the workspace's dependency
groups, which bring pytest) and the installed version of each, and the environment: Python, the platform, whether `lake`,
`cargo` and `nvidia-smi` are on the PATH, and the `VERITY_*` and `FLOCK_*` variables tests switch on. The cache key hashes all
of them, read from git's view of the working tree, so uncommitted edits count; a tree without .git (what `research run
--source` ships) is keyed from the sha256 of every file in it. So a change to the lockfile or to another package's
`pyproject.toml` re-runs only the suites whose resolved dependencies it changed. A pass is kept under its key in
`~/.cache/verity/tests` (`VERITY_TEST_CACHE` overrides it) and reused while the key is unchanged.

Fixtures too large for git are entries of `fixtures/artifacts.json` whose paths git does not track. Before the suites run, the
entries under their inputs are fetched (`research data fetch-fixtures`: every byte checked against its id), and each suite's key
adds the ids of those under its inputs, since git's tree ids leave ignored files out. A suite whose fixtures cannot be fetched
fails without running.

Every run loads `guard/verity_suite_guard.py`, which fails the suite if its tests open a repository file outside its inputs,
so a cached pass never hides a change to something the tests read. It also records what the key leaves out but the tests
used: the workspace metadata files they opened, the installed distributions they imported outside their resolved
dependencies, and the imports that found nothing. A pass is reused only while each of those is as it was. Programs a test
starts as a subprocess are not traced: list what they read under `inputs`.

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
#: Files every key includes: the runner itself.
RUNNER = ("tools/check/suites.py", "tools/check/guard/verity_suite_guard.py")
XDIST = importlib.util.find_spec("xdist") is not None
TOOLS = ("lake", "cargo", "nvidia-smi")
ENV_PREFIXES = ("VERITY_", "FLOCK_")
QUICK = ["-m", "not slow"]
REGISTRY = "fixtures/artifacts.json"


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
    artifacts: tuple[str, ...] = ()   # [tool.verity.tests] artifacts: build outputs its tests run or skip on


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
    """The workspace's metadata: every pyproject.toml, and the lockfile. A suite's key holds what it resolves to (`resolved`);
    the tests may read the files themselves, which the guard records."""
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
        suites.append(Suite(name, path, _requires(project, data), tuple(tests.get("inputs", [])), tests.get("parallel", False) is True,
                            tuple(tests.get("artifacts", []))))
    return suites


def artifact_ids(suite: Suite) -> dict[str, str]:
    """Each declared build output as the key sees it: a file's sha256, "directory", or "absent" (git ignores build outputs, so the
    tree's object ids never see them)."""
    out = {}
    for a in suite.artifacts:
        p = ROOT / a
        if p.is_file():
            with open(p, "rb") as f:
                out[a] = hashlib.file_digest(f, "sha256").hexdigest()
        else:
            out[a] = "directory" if p.is_dir() else "absent"
    return out


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


def locked() -> dict[str, dict]:
    """{distribution: sha256 of its entries in uv.lock, and the names they depend on (dependencies, extras, groups)}."""
    lock = ROOT / "uv.lock"
    by: dict[str, list] = {}
    for p in tomllib.loads(lock.read_text()).get("package", []) if lock.exists() else []:
        by.setdefault(_norm(p["name"]), []).append(p)
    out = {}
    for n, entries in by.items():
        deps = [d for p in entries for d in p.get("dependencies", [])]
        deps += [d for p in entries for g in ("optional-dependencies", "dev-dependencies") for ds in p.get(g, {}).values() for d in ds]
        out[n] = {"entry": hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
                  "requires": sorted({_norm(d["name"]) for d in deps})}
    return out


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _state(name: str, lock: dict[str, dict]) -> list:
    """[its lock entry's sha256, its installed version]."""
    return [lock[name]["entry"] if name in lock else None, _version(name)]


def resolved(names: list[str], lock: dict[str, dict]) -> dict[str, list]:
    """`_state` of ``names`` and of everything they depend on in the lock."""
    out: dict[str, list] = {}
    todo = [_norm(n) for n in names]
    while todo:
        n = todo.pop()
        if n in out or n not in lock:
            continue
        out[n] = _state(n, lock)
        todo += lock[n]["requires"]
    return out


def roots(suite: Suite, pkgs: dict[str, str]) -> list[str]:
    """What a suite resolves from: its own distribution, the workspace packages it lists as inputs, and the workspace's
    dependency groups, which every suite runs with (pytest)."""
    root = _pyproject(".")
    own = root.get("project", {}).get("name") if suite.path == "." else _pyproject(suite.path).get("project", {}).get("name")
    by_path = {p: n for n, p in pkgs.items()}
    groups = [_requirement_name(r) for g in root.get("dependency-groups", {}).values() for r in g if isinstance(r, str)]
    return sorted({*([own] if own else []), *(by_path[p] for p in suite.inputs if p in by_path), *groups})


def environment() -> dict:
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
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


def _under(path: str, prefixes: list[str]) -> bool:
    return any(p == "." or path == p or path.startswith(p.rstrip("/") + "/") for p in prefixes)


def fetched_fixtures(tree: str | None, paths: set[str]) -> dict[str, str]:
    """{repo path: art id} of the registry's entries under `paths` that git's view of the tree does not hold (in a tree without
    git, every entry under them: the fetch leaves one already exact alone)."""
    p = ROOT / REGISTRY
    if not p.is_file():
        return {}
    reg = {v: k for k, v in json.loads(p.read_text()).items() if k.startswith("art:") and isinstance(v, str) and _under(v, sorted(paths))}
    if tree is None or not reg:
        return reg
    held = object_ids(tree, sorted(reg))
    return {v: k for v, k in reg.items() if held[v] == "absent"}


def fetch_fixtures(fixtures: dict[str, str], logs: Path) -> set[str]:
    """Fetch the entries (`research data fetch-fixtures`); returns the paths that are still missing or different afterwards."""
    if not fixtures:
        return set()
    log = logs / "fetch-fixtures.log"
    p = subprocess.run([sys.executable, "-m", "research", "data", "fetch-fixtures", "--repo", str(ROOT), "--json", *sorted(fixtures)],
                       cwd=ROOT, capture_output=True, text=True)
    log.write_text(p.stdout + p.stderr)
    try:
        rep = json.loads(p.stdout)
    except ValueError:
        print(f"fetch-fixtures failed (rc {p.returncode}); log: {log}")
        return set(fixtures)
    bad = {e["path"] for e in rep["entries"] if e["status"] in ("missing", "stale")}
    blobs = rep["fetch"]["blobs"]
    print(f"fixtures: {len(fixtures)} registry entries checked, {blobs['fetched']} blobs ({blobs['bytes'] / 1e6:.1f} MB) downloaded"
          + (f", {len(bad)} missing (log: {log})" if bad else ""))
    return bad


def key(suite: Suite, ids: dict[str, str], env: dict, args: list[str], deps: dict[str, list] | None = None) -> str:
    blob = json.dumps({"suite": suite.path, "inputs": ids, "artifacts": artifact_ids(suite), "resolved": deps or {},
                       "environment": env, "pytest": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def cached(k: str) -> dict | None:
    f = CACHE / f"{k}.json"
    return json.loads(f.read_text()) if f.exists() else None


def holds(hit: dict, tree: str | None, files: dict[str, str] | None, lock: dict[str, dict], provided: set[str]) -> bool:
    """Whether what a pass used beyond its key is as it was: the metadata files its tests read, the distributions they
    imported outside their resolved dependencies, and the imports no installed distribution provided (``provided``: the
    top-level names installed distributions provide now)."""
    watched = hit.get("watched") or {}
    return ((not watched or object_ids(tree, sorted(watched), files) == watched)
            and all(_state(n, lock) == v for n, v in (hit.get("imported") or {}).items())
            and not provided.intersection(hit.get("missing") or []))


def used(r: "Result", paths: list[str], deps: dict[str, list], tree: str | None, files: dict[str, str] | None,
         lock: dict[str, dict]) -> dict:
    """What a pass keeps beside its key, for `holds`."""
    watched = sorted(f for f in r.watched or [] if not _under(f, paths))
    return {"watched": object_ids(tree, watched, files) if watched else {},
            "imported": {n: _state(n, lock) for n in sorted({_norm(d) for d in r.distributions or []} - set(deps))},
            "missing": r.missing or []}


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
    watched: list | None = None      # workspace metadata files the tests read
    distributions: list | None = None    # installed distributions the tests imported
    missing: list | None = None      # top-level imports that found nothing


#: pytest with the guard loaded by name (`-p`), which xdist repeats in each worker, from `guard/`: appended to sys.path, and
#: holding that one module only, so it shadows nothing a suite imports.
_PYTEST = "import sys, pytest; sys.path.append(sys.argv.pop(1)); sys.exit(pytest.main(sys.argv[1:]))"
#: xdist's scheduling: an idle worker takes queued tests from a busy one, which evens out suites whose tests differ in length.
DIST = ["--dist", "worksteal"]


def run(suite: Suite, allowed: list[str], args: list[str], logs: Path, workers: int = 1, watch: list[str] = ()) -> Result:
    slug = suite.path.replace("/", "_") if suite.path != "." else "repository"
    log, report = logs / f"{slug}.log", logs / f"{slug}.json"
    report.unlink(missing_ok=True)
    env = dict(os.environ, VERITY_SUITE_ROOT=str(ROOT), VERITY_SUITE_ALLOW=json.dumps(allowed), VERITY_SUITE_REPORT=str(report),
               VERITY_SUITE_WATCH=json.dumps(list(watch)))
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
                  rep.get("unmarked_slow"), log, workers=workers, unguarded=rep.get("unguarded"), reported=bool(rep),
                  watched=rep.get("watched"), distributions=rep.get("distributions"), missing=rep.get("missing"))


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
    inputs = {s: inputs_of(s, pkgs) for s in chosen}
    fixtures = fetched_fixtures(tree, {p for ps in inputs.values() for p in ps})
    logs = a.logs or CACHE / "logs" / f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    logs.mkdir(parents=True, exist_ok=True)
    # the repository's own suite keys on the whole tree but reads tracked files only, so it alone never makes a fetch
    needed = {p: art for p, art in fixtures.items() if any(_under(p, [x for x in ps if x != "."]) for ps in inputs.values())}
    unfetched = set() if a.list else fetch_fixtures(needed, logs)
    files = None if GIT else digests()
    plan = []
    meta, lock, provided = metadata(), locked(), set(importlib.metadata.packages_distributions())
    for s in chosen:
        paths = inputs[s]
        ids = object_ids(tree, sorted(set(paths) | set(RUNNER)), files)
        ids.update({f"fixture:{p}": art for p, art in fixtures.items() if _under(p, paths)})
        deps = resolved(roots(s, pkgs), lock)
        keys = [key(s, ids, env, args, deps)] + ([key(s, ids, env, extra, deps)] if a.quick else [])   # a full pass also covers --quick
        hit = next((c for c in map(cached, keys) if c and holds(c, tree, files, lock, provided)), None)
        plan.append((s, paths, keys, hit, deps))

    if a.list:
        for s, paths, keys, hit, _deps in plan:
            state = f"cached from {hit['commit'][:10]} {hit['when']}" if hit else "not cached"
            print(f"{s.name:28} {s.path:28} {state}\n    inputs: {', '.join(paths)}")
        return 0

    head = _git("rev-parse", "HEAD").strip() if GIT else a.commit or "unknown"
    dirty = bool(_git("status", "--porcelain").strip()) if GIT else False
    results: list[Result] = [Result(s, "cached", since=hit, key=hit["key"]) for s, _, _, hit, _ in plan if hit and not a.fresh]
    blocked = [s for s, paths, _, hit, _ in plan if (a.fresh or not hit) and any(_under(p, [x for x in paths if x != "."]) for p in unfetched)]
    for s in blocked:
        print(f"failed  {s.name:28} its fixtures could not be fetched: "
              f"{', '.join(sorted(p for p in unfetched if _under(p, inputs[s])))[:300]} (log: {logs / 'fetch-fixtures.log'})")
        results.append(Result(s, "failed", log=logs / "fetch-fixtures.log"))
    todo = [(s, paths, keys, deps) for s, paths, keys, hit, deps in plan if (a.fresh or not hit) and s not in blocked]
    known = _durations()
    todo.sort(key=lambda t: (not t[0].parallel, -known.get(t[0].path, float("inf"))))
    for r in results:
        print(f"cached  {r.suite.name:28} passed at {r.since['commit'][:10]} {r.since['when']}")
    for s in skipped:
        print(f"skipped {s.name:28} (--skip)")
    jobs = max(1, a.jobs)
    spread = jobs if XDIST else 1
    if spread == 1 and jobs > 1 and any(t[0].parallel for t in todo):
        print("pytest-xdist is not installed: each parallel suite runs in one process")
    slots = Slots(jobs)

    def start(turn: int, s: Suite, paths: list[str]) -> Result:
        n = spread if s.parallel else 1
        with slots.take(turn, n):
            return run(s, paths + meta + list(RUNNER), args, logs, n, [m for m in meta if not _under(m, paths)])

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(todo))) as pool:
        futures = {pool.submit(start, i, s, paths): (s, keys[0], paths, deps) for i, (s, paths, keys, deps) in enumerate(todo)}
        for fut in concurrent.futures.as_completed(futures):
            s, k, paths, deps = futures[fut]
            r = fut.result()
            r.key = k
            results.append(r)
            known[s.path] = r.seconds
            print(f"{r.status:7} {s.name:28} {_count(r)} in {r.seconds:.0f} s" + (f" on {r.workers} workers" if r.workers > 1 else ""))
            if r.status == "passed":
                CACHE.mkdir(parents=True, exist_ok=True)
                (CACHE / f"{k}.json").write_text(json.dumps({"suite": s.path, "key": k, "commit": head, "dirty": dirty,
                                                              "when": time.strftime("%Y-%m-%d %H:%M"), "counts": r.counts,
                                                              "seconds": round(r.seconds, 1), **used(r, paths, deps, tree, files, lock)},
                                                             indent=1) + "\n")
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
