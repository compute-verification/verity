"""`suites`: every test suite in the repository, each run in its own process from its own directory, with passes cached.

    uv run tools/verity/check/suites.py                     every suite whose inputs changed since it last passed here
    uv run tools/verity/check/suites.py verity flock        only these suites (a package name, a directory, or a unique part of one)
    uv run tools/verity/check/suites.py --quick             without the tests marked `slow`
    uv run tools/verity/check/suites.py --changed REV       only the suites the change from REV touches (`touched`)
    uv run tools/verity/check/suites.py --fresh             run even where a pass is cached
    uv run tools/verity/check/suites.py --list              each suite, its inputs, and whether a pass is cached
    uv run tools/verity/check/suites.py flock -- -k tanh    arguments after `--` go to pytest

A suite is a directory whose `pyproject.toml` configures pytest (`[tool.pytest.ini_options]`). Its inputs are its own directory,
the directories of the workspace packages it depends on (its dependencies, extras and dependency groups, and theirs
transitively without their dependency groups, which are their own tests' tooling; a package that lists `exports` under
`[tool.verity.tests]` is those paths of it, what its dependents read, in place of its directory), the paths it lists under
`[tool.verity.tests] inputs` (with `inputs_only = true`, those alone in place of its dependencies' directories, for a suite whose
tests read a little of a large dependency), the build outputs it lists under `artifacts` (each file's digest: git ignores them), what it
resolves to in `uv.lock` (the lock's entry for each distribution it depends on, transitively, with the workspace's dependency
groups, which bring pytest) and the installed version of each, and the environment: Python, the platform, the base image (the
os-release `ID` and `VERSION_ID` and the C library: `base_image`), whether `lake`, `cargo` and `nvidia-smi` are on the PATH,
and the `VERITY_*` and `FLOCK_*` variables tests switch on, and the code that runs the suites (`RUNNER`: this runner and its guard,
not the rest of check, so an edit to `check.py` reruns its own suite alone). The cache key hashes all
of them, read from git's view of the working tree, so uncommitted edits count. The tree must be a git checkout: `research run
--source` ships one, so a machine keys a commit as a laptop does. So a change to the lockfile or to another package's
`pyproject.toml` re-runs only the suites whose resolved dependencies it changed. A pass is kept under its key in
`~/.cache/verity/tests` (`VERITY_TEST_CACHE` overrides it) and reused while the key is unchanged, on this machine or, through a
verdict pack (`verdicts.py`; a recorded check gathers one from the newest recorded checks), on any other. A pass names the
recorded run that earned it (`--run`, which check passes), and a reuse prints that run.

The key is content, not location. Whatever is at or under the suite's own directory is keyed by its path from there (`_rel`,
written `<suite>/...`): the directory's git tree id (which hashes each file's path inside it and its blob, and not the
directory's own name), the declared inputs, build outputs, fixtures and runner files under it, its test modules, and in
`uv.lock` the workspace paths that name it (its `source`, a `requires-dist` on it). An input outside it (a dependency's
directory, a declared path) is keyed by its path from the root, as before. A suite is the same suite after a move by its name:
its distribution's name (`[project] name`, in the pyproject the tree id hashes), `repository` for the root's, and its path for a
suite without one, which is a new suite where it lands. The key also holds the suite's depth (`depth`): its tests find the
root by counting parents of `__file__`, so a suite moved to another depth would read elsewhere. So a suite renamed beside itself,
or moved to the same depth, with every file as it was and every input outside it where it was, hits its pass and its tests'
passes; any changed byte under it, or an outside input moved or changed, misses. What this can't see is a test that names its
own suite's directory from the root (`ROOT / "packages/verity/tests/x"`): such a test fails after the move, which its kept pass
hides until the suite next runs (`check.py --no-cache`, the nightly `audit-main` run). Locate a suite's own files from
`__file__`.

Fixtures too large for git are entries of `fixtures/artifacts.json` whose paths git does not track. Before the suites run, the
entries under their inputs are fetched (`research data fetch-fixtures`: every byte checked against its id), and each suite's key
adds the ids of those under its inputs, since git's tree ids leave ignored files out. A suite whose fixtures cannot be fetched
fails without running.

Every run loads `guard/verity_suite_guard.py`, which fails the suite if its tests open a repository file outside its inputs,
so a cached pass never hides a change to something the tests read; the workspace metadata, and what a store fetch reads
(`STORE_READS`), every suite may read. Markdown is prose: a suite's directories key without their `*.md`, and its guard fails a
test that opens one, unless the suite lists a `.md` under `inputs` (`markdown`), when its directories key whole. It also records what the key leaves out but the tests
used: the workspace metadata files they opened, the installed distributions they imported outside their resolved
dependencies, and the imports that found nothing. A pass is reused only while each of those is as it was. A Python a test
starts, or a Python started by a program it starts, traces its own opens while it keeps the suite's PYTHONPATH and environment
(`guard/child/`): one that opens a file outside the inputs fails the suite as the test would, and so does one whose tracer
doesn't start; the failure names the test and the file. Nothing sees into other programs (bash, git, `uv`, a Python run with
`-I`, `-E` or `-S`, or on a PYTHONPATH without the tracer): what they read is unchecked, and checking it would mean keying
their suite on the whole tree. List what they read under `inputs`, or give the test a stand-in for it. The one exception is the
suite's own build outputs (`artifacts`, a Lean executable reads only what its arguments name): such a program's start is taken
as reading the repository paths it is given, a file with the files beside it and a directory whole, checked against the inputs
like any read, and `lake build` of them in their package as reading nothing the key doesn't hold (`VERITY_SUITE_PROGRAMS`).

Per-test verdicts. In a suite that declares `parallel = true`, whose tests run in any order and in any process, each test that
passed is also kept on its own, even when its suite failed: under its module's key (`module_key`), which is the suite's with the
suite's test modules that the module's tests can't reach left out (`Modules.reach`: what the module and the conftests mention by
import or by name in a string, and so on; every test module whose import acts on the process; none left out when a module
imports by a computed name), and with the other test modules those tests opened recorded beside it (the guard). A run whose
suite key missed deselects every test whose module's key and recorded reads are as they were, and counts it as reused, so a fix
to one test module re-runs that module and what can reach it, not the suite.

Traced. Where the guard traced the run (every file the suite's processes opened or loaded and every directory they listed, its
children's included, `child/verity_suite_reads.py`), a module's pass is kept under a narrower key instead (`traced_key`): the suite's
file inputs, fixtures, runner and configuration whole, its test modules as above, and in place of the directories it depends on the
run's read set (`TRACED`), each file's object id (a file git doesn't hold by its sha256) and each listed directory's names. Such a pass
is reused while every read in its set finds what it found, so a change to a package file no test read re-runs nothing. A module
whose tests started a program no tracer sees into (anything but a Python that loads the tracer or a build output it declares) is kept on the suite's inputs whole,
and so is every module when that happened outside one. A pass of a whole suite traced so (every test ran, none started such a
program) is also kept as the suite's (`PerTest.keep_whole`), under `traced_key` with every test module in it, and a run whose
suite key missed reuses it while its read set holds, without starting pytest: the one traced reuse a suite that doesn't declare
`parallel` gets, since it answers for every test in the order they ran. What tracing can't see: a test that only stats a path (`exists`) and reads
outside the repository; `check.py --no-cache`, the nightly `audit-main` run, reports a kept pass a fresh failure contradicts.

`--per-test-min-s` (or `$VERITY_TEST_CACHE_MIN_S`)
keeps only passes that took at least that long; `--no-per-test` turns it off; `--fresh` reuses nothing, and names what failed
that a cached pass (the suite's or a test's) answers for: the nightly cold check's signal that a key misses an input
(`check.py --no-cache`, the nightly `audit-main` job). A stop mid-run (SIGINT) still keeps the passes of runs that finished.

Each suite's pytest runs in a process group of its own, and whatever is left in the group when pytest ends (a multiprocessing
forkserver and its workers, after a kill) is killed with it. A suite whose pytest dies by a signal is reported with it, its peak
RSS and, when the kernel's OOM count rose meanwhile, as an OOM kill, with the machine's memory and the way to a check slot.

Every suite, parallel or not, records each test's outcome under its module's key (a suite that isn't parallel keys every module on
all its files, and reuses nothing): a test that both passed and failed under one key is `nondeterministic`, since its inputs were
identical, and is never reused. Each pass keeps its key's components beside it (`components`: every input's object id, the build
outputs, the resolved dependencies, the environment, the pytest arguments), and `--explain SUITE` diffs a miss against the nearest
pass kept here, naming a changed directory's files from git's trees, and says per test module what it reaches that moved.

`--jobs` is how many test processes run at once, by default the CPUs this process may use but at most one per
`TEST_PROCESS_GB` of available memory. A suite that declares `[tool.verity.tests] parallel = true` spreads its tests
over all of them with pytest-xdist, whose workers load the guard too; each worker's share starts with one of the tests whose
kept passes took longest (`longest`), so the run doesn't wait on a long test that started late. The parallel suites run one at a
time, beside the others, which run side by side on as few of the processes as let them finish about when the longest of them
does (`serial_slots`, from each suite's last duration); each lane goes longest first. Without pytest-xdist installed, a parallel
suite runs in one process.

`--timeouts` (`check` passes it) bounds each suite's run by its `[tool.verity.tests] timeout_s`, else `SUITE_TIMEOUT_S`: a suite
still running then has its process group killed and fails as timed out (`timed_out` in --json), so a hung test costs the check
an hour, not its slot until the run's own timeout.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
GUARD = HERE / "guard"
#: on the tests' PYTHONPATH: its `sitecustomize` makes every Python a test starts trace what it reads
CHILD = GUARD / "child"
CACHE = Path(os.environ.get("VERITY_TEST_CACHE") or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "verity" / "tests")
#: Files every key includes: the code that decides a suite's verdict, the runner and its guard and tracer (Daniel, 9 Oct). Not
#: check.py: what it hands the runner that can change a verdict is keyed as such (the pytest arguments, `environment()`'s
#: variables, the tools on the PATH), and its own suite (verity-check) keys it as its own file.
RUNNER = ("tools/verity/check/suites.py", "tools/verity/check/guard/verity_suite_guard.py",
          "tools/verity/check/guard/child/sitecustomize.py", "tools/verity/check/guard/child/verity_suite_reads.py")
XDIST = importlib.util.find_spec("xdist") is not None
TOOLS = ("lake", "cargo", "nvidia-smi")
ENV_PREFIXES = ("VERITY_", "FLOCK_")
#: variables of those prefixes that say where results are kept or which are kept, not what a test computes
UNKEYED = frozenset({"VERITY_TEST_CACHE", "VERITY_TEST_CACHE_MIN_S"})
#: and the runner's own, which it sets for the tests (`VERITY_SUITE_ROOT`, `..._MODULE`, `..._READS`, ...)
UNKEYED_PREFIX = "VERITY_SUITE_"
#: what a machine's runner gives a run's workload and check's steps never get (check.py's `ENV_PASS`): the run's `RESEARCH_*` and
#: the machine's PYTHONPYCACHEPREFIX. Tests that start runs of their own would inherit them, so a suite run inside a `research
#: run` would reach another verdict than inside check (test_remote_cwd, r20261008-010104-2b69)
RUNNER_PREFIX, RUNNER_ENV = "RESEARCH_", ("PYTHONPYCACHEPREFIX",)
#: the suites whose tests run `lake` or what `lake build` builds in the Lean verifier: check runs them after its `lean-build`
#: step (its `lean-suites`), and on its own this runs that build before them, so they run or skip alike in both
AFTER_LEAN = ("verity-flock", "verity-lean-audit", "verity-difftest", "pous-check", "core-identity", "verity-core-composition",
              "verity-core-evaluate", "verity-core-firewall",
              "verity/core/service/vstar", "verity-core-sampled-proofs",
              "verity/core/protocols/network", "verity-pouw-window-difftest",
              "verity-pouw-check-difftest", "memory-challenger-lean", "benchmarks/pous",
              "verity/core/protocols/sampled_proofs", "verity/core/protocols/certifier", "verity-pouw-benchmarks")
#: the toolchain `lean-toolchain` pins, through elan when `lake` is missing (as `ci-pod.sh` does)
ELAN = ('command -v lake >/dev/null || command -v "$HOME/.elan/bin/lake" >/dev/null || { curl -sSf '
        'https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y --default-toolchain none >/dev/null; }; '
        'export PATH="$HOME/.elan/bin:$PATH"; ')
QUICK = ["-m", "not slow"]
REGISTRY = "fixtures/artifacts.json"
#: what a store fetch from inside the tree reads of it, which every suite may read: the registry, for repo replicas (indexing the
#: remote's manifest on an empty local store), and the bucket pin, when the remote is the environment's (`R2_*`). Neither changes
#: what a fetch returns, which is checked against its id whichever replica or remote serves it.
STORE_READS = (REGISTRY, "tools/research/store.pod.toml")


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


#: what one test process may hold at its peak, its children included: `test_flock_rows.py` on the rmsnorm-triton 4096 layout peaks at
#: 8.8 GB (the lowering in the test's Python and `flock-rows` at once), and two at once on a 15 GB machine were OOM-killed
TEST_PROCESS_GB = 9


def default_jobs(meminfo: Path = Path("/proc/meminfo")) -> int:
    """The default `--jobs`: the CPUs this process may use (`cpus`), or fewer where the memory available holds fewer test
    processes at `TEST_PROCESS_GB` each."""
    n = cpus()
    try:
        kb = next(int(ln.split()[1]) for ln in meminfo.read_text().splitlines() if ln.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError, IndexError):
        return n
    return min(n, max(1, kb // (TEST_PROCESS_GB << 20)))


def _git(*args: str, env: dict | None = None, stdin: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, env=env, input=stdin, capture_output=True, text=True, errors="surrogateescape",
                          check=True).stdout


def in_git() -> bool:
    """Whether the root is a git checkout's top level: the one tree `suites` keys."""
    p = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True)
    return p.returncode == 0 and Path(p.stdout.strip()).resolve() == ROOT


@dataclass(frozen=True)
class Suite:
    name: str
    path: str                    # relative to the repository root; "." for the repository's own suite
    requires: tuple[str, ...]    # names of the distributions it depends on
    inputs: tuple[str, ...]      # paths it declares under [tool.verity.tests] inputs
    parallel: bool = False       # [tool.verity.tests] parallel: its tests may run in several processes at once
    artifacts: tuple[str, ...] = ()   # [tool.verity.tests] artifacts: build outputs its tests run or skip on
    timeout_s: float = 0.0       # [tool.verity.tests] timeout_s: the bound `--timeouts` puts on its run (0: SUITE_TIMEOUT_S)
    inputs_only: bool = False    # [tool.verity.tests] inputs_only: its dependencies' directories are left out (`inputs_of`)


def _pyproject(path: str) -> dict:
    return tomllib.loads((ROOT / path / "pyproject.toml").read_text())


def _requires(project: dict, data: dict, groups: bool = True) -> tuple[str, ...]:
    """The distributions a member names: its dependencies and extras, and with ``groups`` its dependency groups (its own tests'
    tooling, which code that imports it never reads)."""
    reqs = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        reqs += extra
    for group in data.get("dependency-groups", {}).values() if groups else ():
        reqs += [r for r in group if isinstance(r, str)]
    return tuple(sorted({_requirement_name(r) for r in reqs}))


def members() -> dict[str, str]:
    """The workspace packages: {distribution name: directory}, the root's own distribution under "." when it builds one (uv makes
    it a member)."""
    root = _pyproject(".")
    out = {_norm(root["project"]["name"]): "."} if _declared(".", root) is not None else {}
    for path in root.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", []):
        out[_norm(_pyproject(path)["project"]["name"])] = path
    return out


def shipped(path: str) -> list[str]:
    """The paths a workspace package's dependents key on: those it lists under `[tool.verity.tests] exports`, else its directory,
    or for the root, which holds the whole tree, the packages its wheel ships."""
    data = _pyproject(path)
    if (exports := data.get("tool", {}).get("verity", {}).get("tests", {}).get("exports")) is not None:
        return list(exports)
    if path != ".":
        return [path]
    return list(data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"])


def tests_pythonpath() -> str:
    """The tests' PYTHONPATH: the tracer's directory (`CHILD`), then the caller's entries inside this tree. An entry naming another
    checkout, such as an `export PYTHONPATH=...` left in a shell, would run this tree's tests against that checkout's code."""
    return os.pathsep.join([str(CHILD), *(p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                                          if p and Path(p).resolve().is_relative_to(ROOT))])


#: prints where each top-level module named on its command line would be imported from, without importing any of them
_ORIGINS = ("import importlib.util as u, json, sys; "
            "print(json.dumps({n: getattr(u.find_spec(n), 'origin', None) for n in sys.argv[1:]}))")


def _declared(path: str, data: dict) -> list[str] | None:
    """The package directories (relative to the member) that the member's build configuration ships, or None when it is a
    virtual member, which installs nothing. Hatch's `packages`, uv_build's module, setuptools' packages, else hatch's default
    (`src/<name>` or `<name>`)."""
    tool, project = data.get("tool", {}), data.get("project", {})
    if tool.get("uv", {}).get("package") is False or not project or "build-system" not in data:
        return None
    if (wheel := tool.get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})).get("packages"):
        return list(wheel["packages"])
    module = _norm(project["name"]).replace("-", "_")
    if (ub := tool.get("uv", {}).get("build-backend")) is not None:
        names = ub.get("module-name", module)
        return [f"{ub.get('module-root', 'src')}/{n.replace('.', '/')}" for n in ([names] if isinstance(names, str) else names)]
    if (st := tool.get("setuptools")) is not None:
        listed, where = st.get("packages"), ["."]
        if isinstance(listed, list):
            base = st.get("package-dir", {}).get("", ".")
            return [f"{base}/{p}" for p in listed if "." not in p]
        if isinstance(listed, dict):
            where = listed.get("find", {}).get("where", ["."])
        return [str(p.parent.relative_to(ROOT / path)) for w in where for p in sorted((ROOT / path / w).glob("*/__init__.py"))]
    return [p for p in (f"src/{module}", module) if (ROOT / path / p / "__init__.py").is_file()][:1]


def importable(pkgs: dict[str, str]) -> dict[str, str]:
    """The members' importable top-level packages, as their build configurations declare them: {import name: member directory}.
    A member that builds but whose packages can't be found maps "" to its directory."""
    out = {}
    for d in pkgs.values():
        found = _declared(d, _pyproject(d))
        if found is None:
            continue
        names = {Path(p).name.removesuffix(".py") for p in found if not Path(p).is_absolute()}
        for n in names or {""}:
            out.setdefault(n, d)
    return dict(sorted(out.items()))


def foreign(pkgs: dict[str, str], env: dict[str, str], python: str = sys.executable) -> str | None:
    """The members' packages that `python` under `env` would import from outside this tree, or can't import at all, each with
    where from, or None. A virtualenv installed from another checkout runs this tree's tests against that checkout's code; one
    that lacks a member's package can't say which code it would run."""
    names = importable(pkgs)
    away = {f"{d}'s package": "nowhere (its build configuration declares none)" for n, d in names.items() if not n}
    if names := sorted(n for n in names if n):
        r = subprocess.run([python, "-P", "-c", _ORIGINS, *names], env=env, capture_output=True, text=True)
        if r.returncode:
            return f"{python} can't say where it imports them from: {r.stderr.strip()[-300:]}"
        for n, o in json.loads(r.stdout.splitlines()[-1]).items():
            if not o:
                away[n] = "nowhere (not importable)"
            elif not Path(o).resolve().is_relative_to(ROOT):
                away[n] = o
    return ", ".join(f"{n} from {o}" for n, o in sorted(away.items())) or None


def metadata() -> list[str]:
    """The workspace's metadata: every pyproject.toml, and the lockfile. A suite's key holds what it resolves to (`resolved`);
    the tests may read the files themselves, which the guard records."""
    listed = [p for p in _git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "pyproject.toml",
                              "*/pyproject.toml").split("\0") if p]
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
                            tuple(tests.get("artifacts", [])), float(tests.get("timeout_s", 0)), tests.get("inputs_only", False) is True))
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
    """Directories of the workspace packages `suite` depends on, directly (its dependencies, extras and dependency groups) or
    through another's dependencies and extras: a dependency's groups are its own tests' tooling, not what code importing it reads."""
    seen: dict[str, str] = {}
    todo = [r for r in suite.requires if r in pkgs]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen[name] = pkgs[name]
        data = _pyproject(pkgs[name])
        todo += [r for r in _requires(data["project"], data, groups=False) if r in pkgs]
    return sorted(s for p in seen.values() if p != suite.path for s in shipped(p))


def worktree() -> str:
    """The tree git would commit now: tracked files as they are on disk, plus untracked files that aren't ignored."""
    with tempfile.TemporaryDirectory() as tmp:
        index = Path(tmp) / "index"
        real = ROOT / _git("rev-parse", "--git-path", "index").strip()
        if real.exists():
            shutil.copy2(real, index)          # its mtime too: git re-reads an entry as recent as the index (racy git)
        env = dict(os.environ, GIT_INDEX_FILE=str(index))
        _git("add", "-A", env=env)
        return _git("write-tree", env=env).strip()


def object_ids(tree: str, paths: list[str]) -> dict[str, str]:
    """Each path's git object id in `tree`."""
    specs = [tree if p == "." else f"{tree}:{p}" for p in paths]
    out = _git("cat-file", "--batch-check=%(objectname)", stdin="\n".join(specs) + "\n").splitlines()
    return {p: ("absent" if line.endswith(" missing") else line) for p, line in zip(paths, out)}


def markdown(suite: Suite) -> bool:
    """Whether the suite's tests read Markdown, which it says by listing a `.md` file under `inputs`: its directories key whole.
    Any other suite's directories key without their `*.md` (`input_ids`), and its guard fails a test that opens one, so an edit
    to prose alone re-runs only the suites that read it."""
    return any(p.endswith(".md") for p in suite.inputs)


_PROSE: dict[tuple[str, str], str] = {}
#: each digest `input_ids` wrote -> the git tree it read it from, which `components` keeps beside the key for `--explain`
_TREES: dict[str, str] = {}


def input_ids(tree: str, paths: list[str], prose: bool = True) -> dict[str, str]:
    """`object_ids`, with each directory, when not ``prose``, as a digest of its files but its `*.md` (each by its path from the
    directory, as git's tree id hashes them)."""
    ids = object_ids(tree, paths)
    for p in paths if not prose else ():
        if ids[p] != "absent" and (ROOT / p).is_dir():
            if (tree, p) not in _PROSE:
                cut = "" if p == "." else p + "/"
                _PROSE[tree, p] = "nomd:" + _digest(sorted((f.removeprefix(cut), i) for f, i in own_ids(tree, p).items()
                                                           if not f.endswith(".md")))
            _TREES[_PROSE[tree, p]] = ids[p]
            ids[p] = _PROSE[tree, p]
    return ids


def locked(text: str | None = None) -> dict[str, dict]:
    """{distribution: its entries in uv.lock, and the names they depend on (dependencies, extras, groups)}; of ``text``, another
    revision's lock, when given."""
    lock = ROOT / "uv.lock"
    if text is None:
        text = lock.read_text() if lock.exists() else ""
    by: dict[str, list] = {}
    for p in tomllib.loads(text).get("package", []):
        by.setdefault(_norm(p["name"]), []).append(p)
    out = {}
    for n, entries in by.items():
        deps = [d for p in entries for d in p.get("dependencies", [])]
        deps += [d for p in entries for g in ("optional-dependencies", "dev-dependencies") for ds in p.get(g, {}).values() for d in ds]
        out[n] = {"entries": entries, "requires": sorted({_norm(d["name"]) for d in deps})}
    return out


#: Where uv.lock writes a workspace path: a source (`{ editable = "verity" }`) and a requirement's (`requires-dist`).
_LOCAL = frozenset({"editable", "virtual", "directory", "path"})
#: How a key writes the suite's own directory, so that the same files in another directory key alike (`_rel`).
SELF = "<suite>"


def _relocated(doc, own: str):
    """``doc`` (lock entries) with each workspace path at or under ``own`` written from it (`SELF`); every other path as it is."""
    if isinstance(doc, list):
        return [_relocated(x, own) for x in doc]
    if isinstance(doc, dict):
        return {k: (_rel(own, v) if k in _LOCAL and isinstance(v, str) else _relocated(v, own)) for k, v in doc.items()}
    return doc


def _rel(own: str, path: str) -> str:
    """``path`` (repository-relative) as a key writes it for the suite at ``own``: from `SELF` when it is at or under ``own``, else
    as it is; the repository's own suite (".") keys every path as it is."""
    if own == "." or not (path == own or path.startswith(own + "/")):
        return path
    return SELF + path[len(own):]


def _abs(own: str, path: str) -> str:
    """The inverse of `_rel`."""
    return own + path[len(SELF):] if own != "." and (path == SELF or path.startswith(SELF + "/")) else path


def _rel_keys(own: str, d: dict) -> dict:
    """``d`` with its keys (paths, or `fixture:` and a path) written by `_rel`."""
    return {(f"fixture:{_rel(own, k.removeprefix('fixture:'))}" if k.startswith("fixture:") else _rel(own, k)): v for k, v in d.items()}


def depth(suite: "Suite") -> int:
    """How many directories the suite is below the root: its tests find the root by counting parents (`Path(__file__).parents[4]`),
    so a suite moved to another depth reads elsewhere, and its key moves with it."""
    return 0 if suite.path == "." else suite.path.count("/") + 1


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _state(name: str, lock: dict[str, dict], own: str = ".") -> list:
    """[the sha256 of its lock entries, with the workspace paths at or under ``own`` written from it (`_relocated`), its installed
    version]."""
    return [_digest(_relocated(lock[name]["entries"], own)) if name in lock else None, _version(name)]


def resolved(names: list[str], lock: dict[str, dict], own: str = ".") -> dict[str, list]:
    """`_state` of ``names`` and of everything they depend on in the lock, for the suite at ``own``."""
    out: dict[str, list] = {}
    todo = [_norm(n) for n in names]
    while todo:
        n = todo.pop()
        if n in out or n not in lock:
            continue
        out[n] = _state(n, lock, own)
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


def flock_lean() -> Path:
    """The Lean verifier of record, the package check's `lean-build` step builds (check.py's `FLOCK_LEAN`)."""
    return ROOT / "verity" / "core" / "service" / "flock"


def lean_on_path() -> None:
    """Put elan's `lake` on this process's PATH, installing elan when there is none, as check's Lean steps have it; the suites'
    keys then see `lake` as they do inside check."""
    p = subprocess.run(["bash", "-c", ELAN + "command -v lake"], capture_output=True, text=True)
    lake = p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else ""
    if lake and shutil.which("lake") is None:
        os.environ["PATH"] = os.pathsep.join([str(Path(lake).parent), os.environ.get("PATH", "")])


def lean_build(log: Path) -> str | None:
    """check's `lean-build` step, `lake build` in the verifier: None when it built (or the tree has no verifier), else why not."""
    lean = flock_lean()
    if not (lean / "lakefile.toml").exists():
        return None
    print(f"suites: `lake build` in {lean.relative_to(ROOT)} (check's lean-build step) before {', '.join(AFTER_LEAN)}")
    with log.open("w") as out:
        rc = subprocess.run(["bash", "-c", ELAN + "lake build"], cwd=lean, stdout=out, stderr=subprocess.STDOUT).returncode
    return None if rc == 0 else f"`lake build` in {lean.relative_to(ROOT)} exited {rc}"


def base_image() -> dict:
    """The machine's operating system as the key sees it: its os-release `ID` and `VERSION_ID`, and its C library."""
    try:
        release = platform.freedesktop_os_release()
    except OSError:
        release = {}
    return {"os": f"{release.get('ID', '?')} {release.get('VERSION_ID', '?')}", "libc": " ".join(platform.libc_ver())}


def environment() -> dict:
    return {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(), "image": base_image(),
            "tools": {t: shutil.which(t) is not None for t in TOOLS},
            "variables": {k: v for k, v in sorted(os.environ.items())
                          if k.startswith(ENV_PREFIXES) and k not in UNKEYED and not k.startswith(UNKEYED_PREFIX)}}


def inputs_of(suite: Suite, pkgs: dict[str, str]) -> list[str]:
    """Its directory, its dependencies' directories, and its declared inputs. A declared workspace package is its directory
    alone: the guard allows what this returns, so a test that reads that package's own dependencies fails until they are
    declared too. A suite that declares `inputs_only = true` leaves its dependencies' directories out: its `inputs` name what
    its tests read of them, which the guard holds it to."""
    return sorted({suite.path, *([] if suite.inputs_only else closure(suite, pkgs)), *suite.inputs})


def _under(path: str, prefixes: list[str]) -> bool:
    return any(p == "." or path == p or path.startswith(p.rstrip("/") + "/") for p in prefixes)


def touched(suites: list[Suite], pkgs: dict[str, str], rev: str, tree: str, lock: dict[str, dict]) -> list[Suite]:
    """The suites whose key the change from ``rev`` to ``tree`` moves by path: a changed file under their inputs (the repository's
    own suite keys on all of them) or in the runner, or a changed resolution in `uv.lock`. What their tests read beside the key
    (`used`) is left out, so a suite this skips can still miss its cached pass."""
    files = [f for f in _git("diff", "--no-renames", "--name-only", "-z", rev, tree).split("\0") if f]   # a rename names its source too
    if any(f in RUNNER for f in files):
        return list(suites)
    before = lock
    if "uv.lock" in files:
        r = subprocess.run(["git", "show", f"{rev}:uv.lock"], cwd=ROOT, capture_output=True, text=True)
        before = locked(r.stdout if r.returncode == 0 else "")
    return [s for s in suites if any(_under(f, inputs_of(s, pkgs)) for f in files if markdown(s) or not f.endswith(".md"))
            or resolved(roots(s, pkgs), before, s.path) != resolved(roots(s, pkgs), lock, s.path)]


def fetched_fixtures(tree: str, paths: set[str]) -> dict[str, str]:
    """{repo path: art id} of the registry's entries under `paths` that git's view of the tree does not hold."""
    p = ROOT / REGISTRY
    if not p.is_file():
        return {}
    reg = {v: k for k, v in json.loads(p.read_text()).items() if k.startswith("art:") and isinstance(v, str) and _under(v, sorted(paths))}
    if not reg:
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


def _digest(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def components(suite: Suite, ids: dict[str, str], env: dict, args: list[str], deps: dict[str, list] | None = None) -> dict:
    """What a suite's key hashes: its name and depth, each input's object id (under its own directory by the path from there:
    `_rel`), the build outputs, the resolved dependencies, the environment and the pytest arguments. A pass keeps them beside it,
    so `--explain` can say what a miss changed; beside them, unhashed, the git tree of each directory keyed without its Markdown
    (`trees`), which lists the files a change moved."""
    inputs = _rel_keys(suite.path, ids)
    out = {"suite": suite.name, "depth": depth(suite), "inputs": inputs,
           "artifacts": _rel_keys(suite.path, artifact_ids(suite)), "resolved": deps or {}, "environment": env, "pytest": args}
    if trees := {q: _TREES[i] for q, i in inputs.items() if i in _TREES}:
        out["trees"] = trees
    return out


def key(suite: Suite, ids: dict[str, str], env: dict, args: list[str], deps: dict[str, list] | None = None) -> str:
    c = components(suite, ids, env, args, deps)
    c.pop("trees", None)
    return _digest(c)


def cached(k: str) -> dict | None:
    f = CACHE / f"{k}.json"
    return json.loads(f.read_text()) if f.exists() else None


def holds(hit: dict, tree: str, lock: dict[str, dict], provided: set[str]) -> bool:
    """Whether what a pass used beyond its key is as it was: the metadata files its tests read, the distributions they
    imported outside their resolved dependencies, and the imports no installed distribution provided (``provided``: the
    top-level names installed distributions provide now)."""
    watched = hit.get("watched") or {}
    return ((not watched or object_ids(tree, sorted(watched)) == watched)
            and all(_state(n, lock) == v for n, v in (hit.get("imported") or {}).items())
            and not provided.intersection(hit.get("missing") or []))


def used(r: "Result", paths: list[str], deps: dict[str, list], tree: str, lock: dict[str, dict]) -> dict:
    """What a pass keeps beside its key, for `holds`."""
    watched = sorted(f for f in r.watched or [] if not _under(f, paths))
    return {"watched": object_ids(tree, watched) if watched else {},
            "imported": {n: _state(n, lock) for n in sorted({_norm(d) for d in r.distributions or []} - set(deps))},
            "missing": r.missing or []}


# ---- per-test verdicts ---------------------------------------------------------------------------------------------------------------

#: Under the cache: one entry per test module and key, `per-test/<suite>/<module>--<key>.json`, packed like every verdict.
PER_TEST = "per-test"
#: A passing test whose call took less than this many seconds is not kept (default 0: every pass).
MIN_S_ENV = "VERITY_TEST_CACHE_MIN_S"
#: The newest entries kept per test module: an older one only answers for an older tree.
KEEP_PER_MODULE = 3
#: Under a suite's per-test directory: the read sets traced entries were kept on (`Traced`), each `<digest>.json`.
TRACED = "traced"
#: Under a suite's per-test directory, the stem of its whole-suite traced passes (`PerTest.keep_whole`): no test module's path
#: (each ends in `.py`).
WHOLE = "(suite)"
#: Calls that import a module by a name computed at run time, or run code from a path: `Modules.reach` can't bound what they load.
_COMPUTED = frozenset({"import_module", "__import__", "spec_from_file_location", "run_path", "run_module", "exec", "execfile",
                       "reload", "importorskip"})
#: Directories whose `test_*.py` files are data (a test may hand one to a subprocess): never left out of a key.
_DATA_DIRS = frozenset({"fixtures", "data", "expected", "testdata"})
#: Top-level statements that only define.
_DEFINES = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Pass, ast.Global, ast.Nonlocal)


@dataclass(frozen=True)
class Scan:
    names: frozenset          # the names its imports mention (each dotted part, each name imported), and the `test_...` words
                              # in its strings (a path, a node id, a plugin)
    package: bool             # it imports by a name computed from its own package's (`__name__`, `__package__`)
    computed: bool            # it imports by any other computed name, or a string of it names test modules by a pattern
    effects: bool             # importing it acts on the process: a top-level call, or an assignment to an attribute or item


#: A test module's name in a string; a bare `test_` is the start of a computed one (`f"test_{x}.py"`, `"test_*.py"`).
_TEST_WORD = re.compile(r"\btest_\w*")


def _own_name(e) -> bool:
    return isinstance(e, ast.Name) and e.id in ("__name__", "__package__")


def _effects(body: list) -> bool:
    for s in body:
        if isinstance(s, _DEFINES) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)):
            continue
        if isinstance(s, (ast.If, ast.Try, ast.With, ast.For, ast.While)):
            inner = [*s.body, *getattr(s, "orelse", []), *getattr(s, "finalbody", []), *(x for h in getattr(s, "handlers", []) for x in h.body)]
            if _effects(inner):
                return True
            continue
        if isinstance(s, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = s.targets if isinstance(s, ast.Assign) else [s.target]
            if all(isinstance(e, ast.Name) for t in targets for e in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t])):
                continue
        return True
    return False


def scan(text: str) -> Scan | None:
    """What a module can load by name, and whether importing it acts on the process; None when it doesn't parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return None
    names: set[str] = set()
    package = computed = False
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names.update(p for a in n.names for p in a.name.split("."))
        elif isinstance(n, ast.ImportFrom):
            names.update([*(n.module or "").split("."), *(a.name for a in n.names)])
        elif isinstance(n, ast.Constant) and isinstance(n.value, str):
            words = set(_TEST_WORD.findall(n.value))
            computed = computed or "test_" in words
            names.update(words)
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            if name in _COMPUTED and not (n.args and isinstance(n.args[0], ast.Constant)):
                a = n.args[0] if n.args else None
                own = ((isinstance(a, ast.JoinedStr) and a.values and isinstance(a.values[0], ast.FormattedValue)
                        and _own_name(a.values[0].value)) or (isinstance(a, ast.BinOp) and _own_name(a.left))
                       or any(_own_name(x) for x in n.args[1:]) or any(_own_name(k.value) for k in n.keywords))
                package, computed = package or own, computed or not own
    return Scan(frozenset(names - {""}), package, computed, _effects(tree.body))


class Modules:
    """The Python files under a suite's test paths (`testpaths`), scanned once, and what each of its test modules can reach."""

    def __init__(self, suite: Suite, texts: dict[str, str]):
        cfg = _pyproject(suite.path).get("tool", {}).get("pytest", {}).get("ini_options", {})
        base = "" if suite.path == "." else suite.path + "/"
        self.testpaths = [base + t.strip("/") if t not in (".", "") else suite.path for t in cfg.get("testpaths", ["."])]
        self.texts = {p: t for p, t in texts.items() if _under(p, self.testpaths)}
        self.scans = {p: scan(t) for p, t in self.texts.items()}
        by_name: dict[str, list[str]] = {}
        for p in self.texts:
            parts = p.split("/")
            by_name.setdefault(parts[-2] if parts[-1] == "__init__.py" and len(parts) > 1 else parts[-1][:-3], []).append(p)
        self.mentions = {p: [f for n in (s.names if s else ()) for f in by_name.get(n, ())] for p, s in self.scans.items()}
        self.conftests = sorted(p for p in self.texts if p.rsplit("/", 1)[-1] == "conftest.py")
        self.effects = {p for p, s in self.scans.items() if self.narrowable(p) and (s is None or s.effects)}

    def narrowable(self, path: str) -> bool:
        """A test module a key may leave out: a `test_*.py` file under the test paths, outside a data directory."""
        parts = path.split("/")
        return (parts[-1].startswith("test_") and parts[-1].endswith(".py") and not _DATA_DIRS.intersection(parts[:-1])
                and _under(path, self.testpaths))

    def modules(self) -> list[str]:
        return sorted(p for p in self.texts if self.narrowable(p))

    def reach(self, module: str) -> set[str] | None:
        """The test modules a test of ``module`` can depend on: what its file and the suite's conftests mention by name (an import;
        for a test module also its name in a string: a path, a node id, a plugin), and so on through what those mention; and every
        test module whose import acts on the process (under xdist each worker imports them all). None when one of them imports by a
        computed name: then none is left out. One that imports by a name computed from its own package's reaches that package."""
        seen, todo = set(), [module, *self.conftests]
        while todo:
            p = todo.pop()
            if p in seen or p not in self.scans:
                continue
            seen.add(p)
            s = self.scans[p]
            if s is None or s.computed:
                return None
            if s.package:
                todo += [q for q in self.texts if q.rsplit("/", 1)[0] == p.rsplit("/", 1)[0]]
            todo += self.mentions[p]
        return {p for p in seen if self.narrowable(p)} | self.effects


def own_ids(tree: str, path: str) -> dict[str, str]:
    """{file: git object id} for every file under ``path``."""
    out = {}
    for line in _git("ls-tree", "-r", "-z", tree, *(["--", path] if path != "." else [])).split("\0"):
        if "\t" in line:
            meta, f = line.split("\t", 1)
            out[f] = meta.split()[2]
    return out


def module_key(suite: Suite, module: str, keep: set[str] | None, mods: Modules, ids: dict[str, str], own: dict[str, str], env: dict,
               args: list[str], deps: dict[str, list] | None) -> str:
    """A test module's key: its suite's (`key`), with the suite's directory taken file by file and the test modules outside
    ``keep`` (`Modules.reach`) left out; with ``keep`` None, none is."""
    inputs = {p: i for p, i in ids.items() if p != suite.path}
    inputs.update({f: i for f, i in own.items() if keep is None or f in keep or not mods.narrowable(f)})
    blob = json.dumps({"suite": suite.name, "depth": depth(suite), "module": _rel(suite.path, module),
                       "inputs": _rel_keys(suite.path, inputs), "artifacts": _rel_keys(suite.path, artifact_ids(suite)),
                       "resolved": deps or {}, "environment": env, "pytest": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def traced_key(suite: Suite, module: str, keep: set[str] | None, mods: Modules, ids: dict[str, str], own: dict[str, str], env: dict,
               args: list[str], deps: dict[str, list] | None) -> str:
    """A test module's key when its run traced what its tests read (`PerTest`, "Traced"): its suite's file inputs, fixtures and
    runner whole, its suite's configuration, and the suite's test modules ``keep`` names (all with ``keep`` None); not the
    directories the suite depends on, whose files a traced entry's read set holds one by one."""
    config = "pyproject.toml" if suite.path == "." else f"{suite.path}/pyproject.toml"
    inputs = {p: i for p, i in ids.items() if p != suite.path and not (ROOT / p).is_dir()}
    inputs.update({f: i for f, i in own.items() if f == config or (mods.narrowable(f) and (keep is None or f in keep))})
    blob = json.dumps({"schema": "traced/v2", "suite": suite.name, "depth": depth(suite), "module": _rel(suite.path, module),
                       "inputs": _rel_keys(suite.path, inputs), "artifacts": _rel_keys(suite.path, artifact_ids(suite)),
                       "resolved": deps or {}, "environment": env, "pytest": args}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def per_test_dir(suite: Suite) -> Path:
    """Where a suite's per-test verdicts are kept: under its name, which a move leaves as it is."""
    return CACHE / PER_TEST / suite.name.replace("/", "_")


class PerTest:
    """One suite's per-test verdicts for one run: which passes it may reuse (`reusable`), and keeping the passes it earns (`keep`)."""

    def __init__(self, suite: Suite, ids: dict[str, str], env: dict, arg_sets: list[list[str]], deps: dict[str, list] | None,
                 tree: str):
        self.suite, self.tree = suite, tree
        self.dir = per_test_dir(suite)
        self.own = {f: i for f, i in own_ids(tree, suite.path).items() if markdown(suite) or not f.endswith(".md")}
        self.narrow = suite.parallel             # a suite that doesn't declare order independence keys each module on everything
        texts = {}
        for f in self.own if self.narrow else ():
            if f.endswith(".py"):
                try:
                    texts[f] = (ROOT / f).read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    texts[f] = ""          # scans as nothing reachable, and is kept in every key when it is a test module
        self.mods = Modules(suite, texts)
        if self.narrow:
            self.keeps = {m: self.mods.reach(m) for m in self.mods.modules()}
        else:
            self.keeps = {m: None for m in sorted(f for f in self.own if f.endswith(".py") and self.mods.narrowable(f))}
        self.keys = {m: [module_key(suite, m, keep, self.mods, ids, self.own, env, a, deps) for a in arg_sets] for m, keep in self.keeps.items()}
        self.tkeys = {m: [traced_key(suite, m, keep, self.mods, ids, self.own, env, a, deps) for a in arg_sets]
                      for m, keep in self.keeps.items()} if self.narrow else {}
        self.wkeys = [traced_key(suite, suite.path, None, self.mods, ids, self.own, env, a, deps) for a in arg_sets]
        self.fixtures = {p.removeprefix("fixture:"): i for p, i in ids.items() if p.startswith("fixture:")}
        self._index: tuple[dict[str, str], dict[str, str]] | None = None
        self._held: dict[str, bool] = {}
        self.components = {"suite_tree": _TREES.get(ids.get(suite.path), ids.get(suite.path)), "depth": depth(suite),
                           "outside": _rel_keys(suite.path, {p: i for p, i in ids.items() if p != suite.path}),
                           "artifacts": _rel_keys(suite.path, artifact_ids(suite)), "environment": _digest(env),
                           "resolved": _digest(deps or {}), "pytest": arg_sets[0]}
        self.nondeterministic: list[str] = []

    def _rel(self, path: str) -> str:
        return _rel(self.suite.path, path)

    def _abs(self, path: str) -> str:
        return _abs(self.suite.path, path)

    def _stem(self, module: str) -> str:
        """An entry's file name before its key: the module's path from the suite's directory."""
        return self._rel(module).removeprefix(SELF + "/").replace("/", "~")

    def _keep(self, keep: set[str] | None) -> list[str] | None:
        return sorted(map(self._rel, keep)) if keep is not None else None

    def _entry(self, module: str, k: str) -> Path:
        return self.dir / f"{self._stem(module)}--{k}.json"

    def _holds(self, e: dict, lock: dict[str, dict], provided: set[str]) -> bool:
        return (all(self.own.get(self._abs(f), "absent") == i for f, i in (e.get("reads") or {}).items())
                and holds(e, self.tree, lock, provided))

    def _tree(self) -> tuple[dict[str, str], dict[str, str]]:
        """({file: git object id}, {directory: a digest of its entries' names}) for the whole tree, once."""
        if self._index is None:
            files = own_ids(self.tree, ".")
            names: dict[str, set[str]] = {}
            for f in files:
                parts = f.split("/")
                for i in range(len(parts)):
                    names.setdefault("/".join(parts[:i]) or ".", set()).add(parts[i])
            self._index = files, {d: _digest(sorted(n)) for d, n in names.items()}
        return self._index

    def _now(self, item: str) -> str:
        """What a traced read (`F<tab>file`, `L<tab>directory`) finds now: a file's git object id, a fixture's id, or the sha256 of
        a file git doesn't hold (a build output); a directory's entry names as git holds them, or as the disk does for one git
        doesn't hold; `absent`. A read under the suite's directory is kept by its path from there (`_rel`)."""
        kind, path = item.split("\t", 1)
        path = self._abs(path)
        files, dirs = self._tree()
        if kind == "L":
            if path in dirs:
                return dirs[path]
            return _digest(sorted(os.listdir(ROOT / path))) if (ROOT / path).is_dir() else "absent"
        if path in files:
            return files[path]
        for f, art in self.fixtures.items():
            if path == f or path.startswith(f.rstrip("/") + "/"):
                return f"{art}:{path[len(f):]}"
        q = ROOT / path
        if not q.is_file():
            return "absent"
        h = hashlib.sha256()
        with open(q, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return "sha256:" + h.hexdigest()

    def _traced(self, digest: str) -> dict[str, str] | None:
        try:
            reads = json.loads((self.dir / TRACED / f"{digest}.json").read_text())["reads"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return reads if isinstance(reads, dict) and _digest(reads) == digest else None

    def traced_holds(self, digest: str) -> bool:
        """Whether every read of the traced read set ``digest`` finds what it found when the set was kept."""
        if digest not in self._held:
            reads = self._traced(digest)
            self._held[digest] = reads is not None and all(self._now(i) == v for i, v in reads.items())
        return self._held[digest]

    def trace(self, items: set[str]) -> str:
        """Keep a run's traced reads, the suite's own test modules left out (`keep` and `reads` hold those), as a read set; its
        digest."""
        return self.keep_reads({"\t".join((k, self._rel(p))): self._now(i) for i in sorted(items)
                                if not (i.startswith("F\t") and self.mods.narrowable(i[2:])) for k, p in [i.split("\t", 1)]})

    def keep_reads(self, reads: dict[str, str]) -> str:
        """Keep ``reads`` ({traced read: what it found}) as a read set under its digest; the digest."""
        d = _digest(reads)
        f = self.dir / TRACED / f"{d}.json"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f".{f.name}.tmp")
            tmp.write_text(json.dumps({"reads": reads}, indent=0, sort_keys=True) + "\n")
            os.replace(tmp, f)
        self._held[d] = True
        return d

    def reusable(self, lock: dict[str, dict], provided: set[str]) -> dict[str, str]:
        """{node id: the module entry it passed in}, for every kept pass whose module's key and recorded reads are as they were:
        a traced entry's (`traced_key`) with its read set as it was, or else an entry keyed on the suite's inputs whole."""
        out = {}
        for m, keys in self.tkeys.items():
            for k in keys:
                p = self._entry(m, k)
                if p.exists():
                    e = json.loads(p.read_text())
                    if e.get("traced") and self._holds(e, lock, provided) and self.traced_holds(e["traced"]):
                        out.update(dict.fromkeys((n for n in e["passed"] if n not in e.get("failed", {})), p.name))
        for m, keys in self.keys.items():
            for k in keys:
                p = self._entry(m, k)
                if p.exists():
                    e = json.loads(p.read_text())
                    if self._holds(e, lock, provided):
                        out.update(dict.fromkeys((n for n in e["passed"] if n not in e.get("failed", {})), p.name))
        return out

    def keep(self, r: "Result", head: str, used_now: dict, min_s: float) -> int:
        """Record the outcomes of a run the guard vouches for (a report, no read outside the inputs, every worker's record), each
        under its module's key, beside those recorded before under the same key and reads: every failure, and every pass of at
        least ``min_s`` seconds. A test with both under one key is `nondeterministic`: its inputs were identical. The number of
        passes kept."""
        if not r.reported or r.outside or r.unguarded or not (r.passed or r.failed_tests):
            return 0
        base = "" if self.suite.path == "." else self.suite.path + "/"
        when = time.strftime("%Y-%m-%d %H:%M")
        by_module: dict[str, tuple[dict, dict]] = {}
        for node, s in (r.passed or {}).items():
            if s >= min_s:
                by_module.setdefault(base + node.split("::", 1)[0], ({}, {}))[0][node] = s
        for node in r.failed_tests or []:
            by_module.setdefault(base + node.split("::", 1)[0], ({}, {}))[1][node] = when
        self.dir.mkdir(parents=True, exist_ok=True)
        n = 0
        blind = r.child_outside or {}
        untraced = r.untraced or {}
        run_reads = self.trace(r.traced) if self.narrow and r.traced is not None and "*" not in untraced else None
        for m, (passed, failed) in sorted(by_module.items()):
            if m not in self.keys or m in blind or "*" in blind:     # a child read what no key covers: nothing vouches for it
                continue
            keep = self.keeps[m]
            reads = {self._rel(f): self.own.get(f, "absent") for f in [*(r.reads or {}).get(m, []), *(r.reads or {}).get("*", [])]
                     if keep is not None and f not in keep and self.mods.narrowable(f)}
            traced = run_reads if m not in untraced else None       # a program no tracer sees into: the suite's inputs whole
            k = self.keys[m][0] if traced is None else self.tkeys[m][0]
            p = self._entry(m, k)
            old = json.loads(p.read_text()) if p.exists() else {}
            n += len(passed)
            if traced is None and (old.get("reads", {}) == reads or not old):
                passed, failed = {**old.get("passed", {}), **passed}, {**old.get("failed", {}), **failed}
            elif traced is not None and old.get("reads", {}) == reads and self.traced_holds(old.get("traced") or ""):
                # the kept passes answer for their own read set, which still holds: the entry holds both sets from now on
                passed, failed = {**old.get("passed", {}), **passed}, {**old.get("failed", {}), **failed}
                traced = self.keep_reads({**self._traced(old["traced"]), **self._traced(traced)})
            self.nondeterministic += sorted(set(passed) & set(failed))
            doc = {"suite": self.suite.name, "path": self.suite.path, "module": m, "key": k, "commit": head, "when": when,
                   "passed": dict(sorted(passed.items())), "failed": dict(sorted(failed.items())), "reads": reads,
                   "narrowed": keep is not None, "components": {**self.components, "keep": self._keep(keep)},
                   **({"traced": traced} if traced is not None else {}), **used_now}
            tmp = p.with_name(f".{p.name}.tmp")
            tmp.write_text(json.dumps(doc, indent=1) + "\n")
            os.replace(tmp, p)
            prefix = f"{self._stem(m)}--"
            older = sorted((q for q in self.dir.iterdir() if q.name.startswith(prefix) and q.suffix == ".json"),
                           key=lambda q: q.stat().st_mtime, reverse=True)[KEEP_PER_MODULE:]
            for q in older:
                q.unlink(missing_ok=True)
        self._collect()
        return n

    def whole(self, lock: dict[str, dict], provided: set[str]) -> dict | None:
        """The suite's traced pass (`keep_whole`) under one of this run's keys whose read set and what it used beside its key hold."""
        for k in self.wkeys:
            p = self.dir / f"{WHOLE}--{k}.json"
            if p.exists():
                e = json.loads(p.read_text())
                if e.get("traced") and holds(e, self.tree, lock, provided) and self.traced_holds(e["traced"]):
                    return e
        return None

    def keep_whole(self, r: "Result", head: str, used_now: dict, run: str | None) -> bool:
        """Keep a passing run as the suite's traced pass, when the guard traced all of it: every test ran (none reused), and none
        started a program no tracer sees into. Kept under `traced_key` with every test module in it, so it answers for the whole
        suite, in any order, which a suite that doesn't declare `parallel` needs."""
        if r.status != "passed" or r.reused or r.traced is None or r.untraced or r.child_outside or not r.reported:
            return False
        k = self.wkeys[0]
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / f"{WHOLE}--{k}.json"
        doc = {"suite": self.suite.name, "path": self.suite.path, "key": k, "commit": head, "run": run,
               "when": time.strftime("%Y-%m-%d %H:%M"), "counts": r.counts, "seconds": round(r.seconds, 1),
               "traced": self.trace(r.traced), **used_now}
        tmp = p.with_name(f".{p.name}.tmp")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, p)
        for q in sorted(self.dir.glob(f"{WHOLE}--*.json"), key=lambda q: q.stat().st_mtime, reverse=True)[KEEP_PER_MODULE:]:
            q.unlink(missing_ok=True)
        self._collect()
        return True

    def _collect(self) -> None:
        """Delete the read sets no entry here names any more."""
        if not (self.dir / TRACED).is_dir():
            return
        named = set()
        for q in self.dir.glob("*.json"):
            try:
                named.add(json.loads(q.read_text()).get("traced"))
            except (OSError, ValueError, AttributeError):
                continue
        for q in (self.dir / TRACED).glob("*.json"):
            if q.stem not in named:
                q.unlink(missing_ok=True)

    def explain(self, lock: dict[str, dict], provided: set[str]) -> list[str]:
        """Which modules' kept passes a run would reuse now, and for each other module what moved since its newest entry."""
        reusable = self.reusable(lock, provided)
        mods = {n.split("::", 1)[0] for n in reusable}
        base = "" if self.suite.path == "." else self.suite.path + "/"
        out = [f"per-test: {len(reusable)} tests of {len(mods)} of {len(self.keys)} test modules would be reused"
               + ("" if self.narrow else " (none: the suite doesn't declare parallel = true, so outcomes are recorded, not reused)")]
        for m in sorted(self.keys):
            if m[len(base):] in mods or not self.narrow:
                continue
            prefix = f"{self._stem(m)}--"
            entries = sorted((q for q in self.dir.glob(f"{prefix}*.json")), key=lambda q: q.stat().st_mtime, reverse=True) if self.dir.is_dir() else []
            if not entries:
                out.append(f"  {m}: nothing kept")
                continue
            e = json.loads(entries[0].read_text())
            out.append(f"  {m}: " + "; ".join(self._moved(m, e, lock, provided)))
        return out

    def _moved(self, m: str, e: dict, lock: dict[str, dict], provided: set[str]) -> list[str]:
        was, now = e.get("components") or {}, {**self.components, "keep": sorted(self.keeps[m]) if self.keeps[m] is not None else None}
        if not was:
            return ["its entry predates recorded components"]
        wide = [p for p in sorted(set(was.get("outside", {})) | set(now["outside"]))
                if not (e.get("traced") and (ROOT / self._abs(p)).is_dir())]
        why = [f"input {self._abs(p)} changed" for p in wide if was.get("outside", {}).get(p) != now["outside"].get(p)]
        if e.get("traced"):
            reads = self._traced(e["traced"])
            moved = sorted(self._abs(i.split("\t", 1)[1]) for i, v in (reads or {}).items() if self._now(i) != v)
            why += (["its read set isn't kept here"] if reads is None else [f"it read {f}, which changed" for f in moved[:5]]
                    + ([f"and {len(moved) - 5} more"] if len(moved) > 5 else []))
        why += [f"{k} changed" for k in ("depth", "artifacts", "environment", "resolved", "pytest") if was.get(k) != now[k]]
        if (was.get("keep") is None) != (now["keep"] is None):
            why.append("narrowing " + ("is now off: it reaches a computed import" if now["keep"] is None else "is now on"))
        if was.get("suite_tree") != now["suite_tree"]:
            kept = set(now["keep"] or ())
            files = _changed_files(was.get("suite_tree"), now["suite_tree"])
            if files is not None and self.suite.path != ".":
                files = [f"{self.suite.path}/{f}" for f in files]      # the subtree's own paths, as the root's
            if files is not None and not markdown(self.suite):
                files = [f for f in files if not f.endswith(".md")]
            if files is None:
                why.append(f"a file under {self.suite.path} changed (its old tree isn't here to list which)")
            else:
                hit = [f for f in files if (now["keep"] is None or f in kept or not self.mods.narrowable(f))
                       and not (e.get("traced") and not self.mods.narrowable(f))]       # a traced entry's other files: its read set
                why += [f"it reaches {f}, which changed" for f in hit[:5]] + ([f"and {len(hit) - 5} more"] if len(hit) > 5 else [])
        if not self._holds(e, lock, provided) and not why:
            why.append("what its tests read beside the key moved (a test module they opened, or a metadata file or distribution)")
        return why or ["nothing its key reads moved; its entry holds no pass it could reuse (only failures, or a nondeterministic test)"]

    def contradicted(self, r: "Result", lock: dict[str, dict], provided: set[str]) -> list[str]:
        """The tests a fresh run failed that a kept pass, still valid by its key and reads, says passed: a key that misses an input,
        or a test whose outcome isn't a function of its inputs."""
        kept = self.reusable(lock, provided)
        return sorted(n for n in (r.failed_tests or []) if n in kept)


def _changed_files(old: str | None, new: str | None) -> list[str] | None:
    """The files that differ between two of git's tree objects (root-relative paths are not needed: the caller's are the tree's
    own), None when either isn't a tree in this repository (a pass a machine kept before its trees were checkouts is keyed on
    digests, not git's ids)."""
    if not (old and new) or old == new:
        return [] if old == new else None
    r = subprocess.run(["git", "diff-tree", "-r", "--name-only", old, new], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.split() if r.returncode == 0 else None


def explain_changes(was: dict, now: dict, own: str = ".") -> list[str]:
    """What differs between two suites' key components (`components`), one line each, a changed directory with its files, by
    their paths from the root for the suite at ``own``."""
    out = []
    wi, ni = was.get("inputs", {}), now["inputs"]
    for q in sorted(set(wi) | set(ni)):
        p = _abs(own, q)
        if q not in wi:
            out.append(f"input {p}: new ({ni[q][:12]})")
        elif q not in ni:
            out.append(f"input {p}: no longer an input")
        elif wi[q] != ni[q]:
            wt, nt = was.get("trees", {}).get(q, wi[q]), now.get("trees", {}).get(q, ni[q])
            files = _changed_files(wt, nt)
            if files is not None and q in now.get("trees", {}):
                files = [f for f in files if not f.endswith(".md")]
            where = "" if files is None else f": {', '.join(f'{p.rstrip(chr(47))}/{f}' if p != '.' else f for f in files[:8])}" + \
                (f" and {len(files) - 8} more" if len(files) > 8 else "")
            out.append(f"input {p} changed ({wi[q][:12]} -> {ni[q][:12]}){where}")
    if was.get("depth") != now.get("depth"):
        out.append(f"depth: {was.get('depth')} -> {now.get('depth')} (its tests find the root by counting parents)")
    for k in sorted(set(was.get("artifacts", {})) | set(now["artifacts"])):
        if was.get("artifacts", {}).get(k) != now["artifacts"].get(k):
            out.append(f"build output {_abs(own, k)}: {was.get('artifacts', {}).get(k, 'absent')} -> {now['artifacts'].get(k, 'absent')}")
    wr, nr = was.get("resolved", {}), now["resolved"]
    for d in sorted(set(wr) | set(nr)):
        if wr.get(d) != nr.get(d):
            a, b = wr.get(d) or [None, None], nr.get(d) or [None, None]
            out.append(f"dependency {d}: " + (f"{a[1]} -> {b[1]}" if a[1] != b[1] else "its uv.lock entry changed"))
    we, ne = was.get("environment", {}), now["environment"]
    for k in ("python", "system", "machine"):
        if we.get(k) != ne.get(k):
            out.append(f"{k}: {we.get(k)} -> {ne.get(k)}")
    for t in sorted(set(we.get("tools", {})) | set(ne.get("tools", {}))):
        if we.get("tools", {}).get(t) != ne.get("tools", {}).get(t):
            out.append(f"tool {t}: {'on the PATH' if we.get('tools', {}).get(t) else 'absent'} -> {'on the PATH' if ne.get('tools', {}).get(t) else 'absent'}")
    wv, nv = we.get("variables", {}), ne.get("variables", {})
    for v in sorted(set(wv) | set(nv)):
        if wv.get(v) != nv.get(v):
            out.append(f"variable {v}: {wv.get(v, '(unset)')!r} -> {nv.get(v, '(unset)')!r}")
    if was.get("pytest") != now["pytest"]:
        out.append(f"pytest arguments: {was.get('pytest')} -> {now['pytest']}")
    return out


def explain(s: Suite, now: dict, keys: list[str], hit: dict | None, per: "PerTest | None", tree: str, lock: dict[str, dict],
            provided: set[str]) -> list[str]:
    """Why ``s`` would run: its key against the nearest pass kept here (fewest differing components), and its per-test reuse."""
    if hit:
        return [f"{s.name}: cached: a pass under this key, from {_since(hit)}"]
    if per and (w := per.whole(lock, provided)):
        return [f"{s.name}: cached: a traced pass of the whole suite, from {_since(w)}, every file its run read as it was"]
    out = []
    held = next((c for c in map(cached, keys) if c), None)
    if held:
        out.append(f"{s.name}: a pass under this key exists but no longer holds: {HELD}")
    near = nearest(kept_passes().get(s.name, []), now, s.path)
    if near is None and not held:
        out.append(f"{s.name}: missed; {NO_PASS}")
    elif near is not None:
        out.append(f"{s.name}: missed; the nearest pass kept here ({_since(near[0])}) differs in:")
        out += [f"  {line}" for line in near[1]] or [f"  {SAME_KEY}"]
    return out + (per.explain(lock, provided) if per else [])


HELD = ("what its tests read beside the key moved (a metadata file they opened, or a distribution they imported outside their "
        "resolved dependencies)")
NO_PASS = "no pass of it kept here records its key's components (passes kept before `--explain` don't)"
SAME_KEY = "nothing its key reads (a different pytest selection?)"


def kept_passes() -> dict[str, list[dict]]:
    """Every pass kept here that records its key's components (`components`), by suite name."""
    out: dict[str, list[dict]] = {}
    for f in CACHE.glob("*.json") if CACHE.is_dir() else ():
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get("suite"), str) and "components" in d:
            out.setdefault(d["suite"], []).append(d)
    return out


def nearest(passes: list[dict], now: dict, own: str) -> tuple[dict, list[str]] | None:
    """The pass of ``passes`` whose components differ least from ``now``, with what differs (`explain_changes`)."""
    if not passes:
        return None
    diffs = [(explain_changes(d["components"], now, own), d) for d in passes]
    lines, best = min(diffs, key=lambda x: (len(x[0]), x[1].get("when", "")))
    return best, lines


def forced_by(s: Suite, now: dict, passes: list[dict], held: bool, fresh: bool) -> dict:
    """What makes ``s`` run rather than reuse a pass, for `--json` and check's reuse report: the inputs of its key that differ
    from the nearest pass kept here (``nearest``: that pass's run and commit), or why there is none to compare."""
    if fresh:
        return {"why": ["--fresh"]}
    if held:
        return {"why": [f"a pass under this key no longer holds: {HELD}"]}
    near = nearest(passes, now, s.path)
    if near is None:
        return {"why": [NO_PASS]}
    best, lines = near
    return {"why": lines or [SAME_KEY], "nearest": {k: best.get(k) for k in ("run", "commit", "when")}}


def _durations() -> dict[str, float]:
    f = CACHE / "durations.json"
    return json.loads(f.read_text()) if f.exists() else {}


def longest(suite: Suite, n: int) -> list[str]:
    """The ``n`` tests of ``suite`` whose setup and call took longest when their passes were last kept (`PerTest`), longest first: what
    the guard starts at the heads of the workers' shares (`heavy_first`). An order only, which no key or verdict reads."""
    d = per_test_dir(suite)
    last: dict[str, tuple[str, float]] = {}
    for p in d.glob("*.json") if d.is_dir() else ():
        try:
            e = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        for node, s in (e.get("passed") or {}).items():
            last[node] = max(last.get(node, ("", 0.0)), (e.get("when") or "", s))
    return sorted(last, key=lambda node: (-last[node][1], node))[:n]


@dataclass
class Result:
    suite: Suite
    status: str                  # passed, cached, failed
    seconds: float = 0.0
    counts: dict | None = None
    outside: dict | None = None  # file read outside the inputs (a cached module as its source) -> the test module that read it ("*": a conftest or session fixture)
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
    passed: dict | None = None       # node id -> setup and call seconds, for each test that passed
    failed_tests: list | None = None     # node ids of the tests that failed
    reads: dict | None = None        # test module (or "*") -> the suite's other test modules it opened
    reused: int = 0                  # tests left out because a kept pass answers for them (`PerTest`)
    contradicts: list | None = None  # in a fresh run: what failed that a cached pass answers for
    nondeterministic: list | None = None     # tests that both passed and failed under one key
    child_outside: dict | None = None    # test module (or "*") -> what a process its tests started read outside the inputs
    traced: set | None = None        # "F<tab>file" / "L<tab>directory" every process of the run read (None: the guard traced nothing)
    untraced: dict | None = None     # test module (or "*") -> programs its tests started that no tracer sees into
    ran_all: bool = False            # pytest itself passed: every test it ran passed
    child_reads: list | None = None  # (kind, file or why, tests): "O" a file a child read outside the inputs, "E" a child untraced
    signal: str = ""                 # the signal pytest died by ("" when it exited)
    peak_rss_gb: float = 0.0         # pytest's peak RSS, its reaped children's included
    oom: bool = False                # SIGKILL while the kernel's OOM-kill count rose
    leftover: int = 0                # processes still in the suite's group when pytest ended, killed
    timed_out: float = 0.0           # the bound (seconds) the run reached, its process group then killed (0: it ended by itself)


#: pytest with the guard loaded by name (`-p`), which xdist repeats in each worker, from `guard/`: appended to sys.path, and
#: holding that one module only, so it shadows nothing a suite imports.
_PYTEST = "import sys, pytest; sys.path.append(sys.argv.pop(1)); sys.exit(pytest.main(sys.argv[1:]))"
#: xdist's scheduling: an idle worker takes queued tests from a busy one, which evens out suites whose tests differ in length.
DIST = ["--dist", "worksteal"]
#: `--timeouts`: a suite's run is bounded by its `timeout_s`, else this. The longest passes on vy-nebius-1 took about an hour
#: (verity-circuit-check, which declares its own); a suite that hangs holds its check slot until the run's own timeout (45fe hung
#: 82 min in test_suites.py on a FIFO until it was killed by hand, 7 Oct).
SUITE_TIMEOUT_S = 3600.0
#: what `run` arms the bound with (a test injects one that fires at once)
Timer = threading.Timer
#: each suite's pytest `--basetemp`, `<tmp>/<BASETEMP>-<suites.py's pid>-*`, removed when the suite ends: pytest's own keeps
#: its last three runs, and a flock suite's is ~45 GB, which filled vy-cpu-1 and vy-cpu-2 three times on 11 Oct. The name is no
#: longer than pytest's own `pytest-of-<user>/pytest-<n>`: a test's AF_UNIX socket under tmp_path has at most 107 bytes of path
BASETEMP = "verity-suite"


def sweep_basetemps(tmp: Path | None = None) -> list[Path]:
    """Remove the `--basetemp`s whose suites.py is gone (killed before it could remove them); returns them."""
    gone = []
    for d in (tmp or Path(tempfile.gettempdir())).glob(f"{BASETEMP}-*"):
        pid = d.name.removeprefix(f"{BASETEMP}-").split("-", 1)[0]
        if pid.isdigit() and int(pid) != os.getpid() and not Path(f"/proc/{pid}").exists():
            shutil.rmtree(d, ignore_errors=True)
            gone.append(d)
    return gone


def run(suite: Suite, allowed: list[str], args: list[str], logs: Path, workers: int = 1, watch: list[str] = (),
        reuse: list[str] = (), timeout_s: float = 0.0) -> Result:
    slug = suite.path.replace("/", "_") if suite.path != "." else "repository"
    log, report = logs / f"{slug}.log", logs / f"{slug}.json"
    report.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(RUNNER_PREFIX) and k not in RUNNER_ENV}
    env.update(VERITY_SUITE_ROOT=str(ROOT), VERITY_SUITE_ALLOW=json.dumps(allowed), VERITY_SUITE_REPORT=str(report),
               VERITY_SUITE_WATCH=json.dumps(list(watch)), VERITY_SUITE_MARKDOWN="1" if markdown(suite) else "0",
               VERITY_SUITE_PROGRAMS=json.dumps(list(suite.artifacts)))
    env.pop("VERITY_SUITE_REUSE", None)
    env.pop("VERITY_SUITE_MODULE", None)
    (logs / f"{slug}.reuse.json").unlink(missing_ok=True)
    reads_log = logs / f"{slug}.reads"
    reads_log.unlink(missing_ok=True)
    env.update(VERITY_SUITE_READS=str(reads_log), VERITY_SUITE_DIR=str(ROOT / suite.path), PYTHONPATH=tests_pythonpath())
    if reuse:
        (logs / f"{slug}.reuse.json").write_text(json.dumps(sorted(reuse)))
        env["VERITY_SUITE_REUSE"] = str(logs / f"{slug}.reuse.json")
    env.pop("VERITY_SUITE_HEAVY", None)
    (logs / f"{slug}.heavy.json").unlink(missing_ok=True)
    if workers > 1 and (first := longest(suite, workers)):
        (logs / f"{slug}.heavy.json").write_text(json.dumps(first))
        env["VERITY_SUITE_HEAVY"] = str(logs / f"{slug}.heavy.json")
    spread = ["-n", str(workers), *DIST] if workers > 1 else []
    own = not any(a == "--basetemp" or a.startswith("--basetemp=") for a in args)
    basetemp = Path(tempfile.mkdtemp(prefix=f"{BASETEMP}-{os.getpid()}-")) if own else None
    start, ooms = time.monotonic(), oom_kills()
    with log.open("w") as out:
        p = subprocess.Popen([sys.executable, "-P", "-c", _PYTEST, str(GUARD), "-p", "verity_suite_guard", "-p", "no:cacheprovider",
                              *spread, *([f"--basetemp={basetemp}"] if basetemp else []), *args], cwd=ROOT / suite.path, env=env,
                             stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        with _GROUPS_LOCK:
            _GROUPS.add(p.pid)
        expired = threading.Event()

        def expire() -> None:
            expired.set()
            kill_group(p.pid)

        bound = Timer(timeout_s, expire) if timeout_s > 0 else None
        if bound:
            bound.daemon = True
            bound.start()
        try:
            status, usage = _wait4(p.pid)
        finally:
            if bound:
                bound.cancel()
            with _GROUPS_LOCK:
                _GROUPS.discard(p.pid)
            leftover = kill_group(p.pid)
            if basetemp:
                shutil.rmtree(basetemp, ignore_errors=True)
    rc = p.returncode = os.waitstatus_to_exitcode(status)
    seconds = time.monotonic() - start
    sig = signal.Signals(-rc).name if rc < 0 else ""
    timed_out = timeout_s if expired.is_set() else 0.0
    oom = sig == "SIGKILL" and not timed_out and ooms is not None and (oom_kills() or 0) > ooms
    rep = json.loads(report.read_text()) if report.exists() else None
    ran_all = rc == 0 or (rc == 5 and bool(reuse))           # 5: nothing left to run, every test's pass reused
    rep = rep or {}
    reported = bool(rep)
    reads, child_outside, by = {m: set(fs) for m, fs in (rep.get("reads") or {}).items()}, {}, {}
    traced = set(rep["traced"]) if isinstance(rep.get("traced"), list) else None
    untraced = {m: set(v) for m, v in (rep.get("untraced") or {}).items()}
    base = "" if suite.path == "." else suite.path + "/"
    for line in reads_log.read_text().splitlines() if reads_log.exists() else ():
        kind, module, rel, test = (line.split("\t", 3) + ["", "", ""])[:4]
        if kind == "R":
            reads.setdefault(module, set()).add(rel)
        elif kind in ("O", "E"):
            child_outside.setdefault(module, []).append(rel)
            who = test or ("a session fixture or conftest" if module == "*" else f"{module.removeprefix(base)} (on import)")
            by.setdefault((kind, rel), set()).add(who)
        elif kind in ("F", "L") and traced is not None:
            traced.add(f"{kind}\t{rel}")
        elif kind == "X":
            untraced.setdefault(module, set()).add(rel)
    rep["reads"] = {m: sorted(fs) for m, fs in sorted(reads.items())}
    out = rep.get("outside") or {}
    rep["outside"] = dict.fromkeys(out, "*") if isinstance(out, list) else out      # a report written before the attribution
    # a child's read outside the inputs is the test's own: no key covers it, so nothing vouches for a pass
    passed = ran_all and reported and not rep["outside"] and not rep["unguarded"] and not child_outside and not timed_out
    return Result(suite, "passed" if passed else "failed", seconds, rep.get("counts"), rep.get("outside"),
                  rep.get("unmarked_slow"), log, workers=workers, unguarded=rep.get("unguarded"), reported=reported,
                  watched=rep.get("watched"), distributions=rep.get("distributions"), missing=rep.get("missing"),
                  passed=rep.get("passed"), failed_tests=rep.get("failed"), reads=rep.get("reads"), reused=len(reuse),
                  child_outside={m: sorted(set(fs)) for m, fs in child_outside.items()}, traced=traced,
                  untraced={m: sorted(v) for m, v in untraced.items()}, ran_all=ran_all,
                  child_reads=[(k, rel, sorted(w)) for (k, rel), w in sorted(by.items())], signal=sig,
                  peak_rss_gb=usage.ru_maxrss / 2**20, oom=oom, leftover=leftover, timed_out=timed_out)


#: the process groups of the suites running now: a ^C reaches suites.py's group only, so `main` passes it on to these
_GROUPS: set[int] = set()
_GROUPS_LOCK = threading.Lock()


def _wait4(pid: int):
    while True:
        try:
            _, status, usage = os.wait4(pid, 0)
            return status, usage
        except InterruptedError:
            continue


def oom_kills() -> int | None:
    """The kernel's count of OOM kills since boot (`/proc/vmstat`), or None where it isn't there."""
    try:
        for line in Path("/proc/vmstat").read_text().splitlines():
            if line.startswith("oom_kill "):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def kill_group(pgid: int, sig: int = signal.SIGKILL) -> int:
    """Signal what is left in a suite's process group; how many processes that was (0 where /proc can't say)."""
    n = 0
    for d in Path("/proc").glob("[0-9]*"):
        try:
            if int((d / "stat").read_text().rsplit(")", 1)[1].split()[2]) == pgid:
                n += 1
        except (OSError, ValueError, IndexError):
            pass
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass
    return n


def memory_gb() -> float | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError):
        return None


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


def serial_slots(seconds: list[float | None], parallel: int, jobs: int) -> int:
    """How many of the ``jobs`` slots the suites that run in one process share (``seconds``: each one's last duration, None when
    none is recorded), the rest going to the ``parallel`` suites, which run one at a time over all of theirs. The fewest that let
    the one-process suites, longest first, finish about when their longest does, plus a slot for each with no recorded duration
    up to a quarter of the jobs (a fresh test cache, as the lander's `--fresh` runs have, times none): the parallel suites keep
    every slot the others couldn't use."""
    if not parallel:
        return jobs
    if not seconds:
        return 0
    timed = [s for s in seconds if s is not None]
    need = (-(-sum(timed) // max(timed)) if timed and max(timed) > 0 else len(timed)) + min(len(seconds) - len(timed), jobs // 4)
    return max(1, min(int(need), jobs - 1))


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


def _since(d: dict | None) -> str:
    """The commit and time a kept pass records, or `?` for what it lacks."""
    d = d or {}
    return f"{(d.get('commit') or '?')[:10]} {d.get('when') or '?'}" + (f" in run {d['run']}" if d.get("run") else "")


def _count(r: Result) -> str:
    c = r.counts or {}
    c = {**c, "reused": r.reused}
    return ", ".join(f"{c[k]} {k}" for k in ("passed", "reused", "failed", "error", "skipped") if c.get(k)) or "no tests"


def failing_path(out: Path) -> Path:
    """Beside `--json`'s ``out``: the suites failed so far, rewritten as each one fails, so a check that keeps going is known to
    have failed while it runs (`research queue drive` reads it through `research status --also`)."""
    return out.with_name(f"{out.stem}.failed.json")


def failing(out: Path, results: list[Result]) -> None:
    doc = [{"name": r.suite.name, "suite": r.suite.path, "failed_tests": (r.failed_tests or [])[:10],
            "outside": r.outside or {}, "child_outside": r.child_outside or {},
            "log": r.log.name if r.log else None} for r in results if r.status == "failed"]
    tmp = failing_path(out).with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(failing_path(out))


def main(argv: list[str] | None = None) -> int:
    # a manual stop (`research cancel`) sends SIGTERM to the run's process group, which the suites' own groups aren't in: as
    # SIGINT, it stops them and removes their `--basetemp`s (a cancelled check left a 42 GB one on vy-cpu-2, 11 Oct)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    argv = sys.argv[1:] if argv is None else argv
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(prog="suites", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="*", help="suites to run (default: all)")
    ap.add_argument("--quick", action="store_true", help="deselect the tests marked slow")
    ap.add_argument("--changed", metavar="REV", help="of those, only the suites the change from REV to the tree touches (`touched`)")
    ap.add_argument("--fresh", action="store_true", help="run even where a pass is cached")
    ap.add_argument("--timeouts", action="store_true",
                    help=f"kill a suite's process group once it has run its [tool.verity.tests] timeout_s (default {SUITE_TIMEOUT_S:.0f}) "
                         "and fail it as timed out (`check` passes this)")
    ap.add_argument("--list", action="store_true", help="list the suites, their inputs, and whether a pass is cached")
    ap.add_argument("--skip", action="append", default=[], metavar="SUITE", help="leave this suite out (repeatable)")
    ap.add_argument("--jobs", "-j", type=int, default=default_jobs(),
                    help="test processes at once, a parallel suite's xdist workers included (default: the CPUs this process may use, "
                         f"at most one per {TEST_PROCESS_GB} GB of available memory)")
    ap.add_argument("--logs", type=Path, help="write each suite's log and report here (default: under the cache)")
    ap.add_argument("--json", type=Path, help="write the outcome of every selected suite here")
    ap.add_argument("--run", metavar="RUN", help="the recorded run this is (check passes its own): kept with each pass, so a reuse of it "
                    "on any machine names the run that earned it")
    ap.add_argument("--per-test-min-s", type=float, default=None, metavar="S",
                    help=f"keep per-test passes only of tests whose call took at least S seconds (default ${MIN_S_ENV}, else 0)")
    ap.add_argument("--no-per-test", action="store_true", help="neither reuse nor keep per-test passes")
    ap.add_argument("--explain", metavar="SUITE", help="say why SUITE would run: its key against the nearest pass kept here, and its "
                    "per-test reuse; runs nothing")
    ap.add_argument("--lean-built", action="store_true", help=f"the Lean verifier is built and `lake` is on the PATH (check's "
                    f"lean-suites step): don't run `lake build` before {', '.join(AFTER_LEAN)}")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    if not in_git():
        print(f"suites: {ROOT} is not a git checkout, and suites are keyed from git's view of the tree: run them in a checkout "
              f"(`research run --on` ships one; a tree the archive stream or an older launcher shipped isn't)", file=sys.stderr)
        return 2

    pkgs, suites = members(), discover()
    if not (a.list or a.explain) and (away := foreign(pkgs, dict(os.environ, PYTHONPATH=tests_pythonpath()))):
        print(f"suites: {sys.executable} would run this tree's tests against another checkout's code ({away}); run them with "
              f"{ROOT}'s own environment: `uv run tools/verity/check/suites.py` there", file=sys.stderr)
        return 2
    skipped = select(suites, a.skip) if a.skip else []
    chosen = [s for s in select(suites, [a.explain] if a.explain else a.suites) if s not in skipped]
    args = (QUICK if a.quick else []) + extra
    tree = worktree()
    if a.changed:
        chosen = touched(chosen, pkgs, a.changed, tree, locked())
        print(f"suites: the change from {a.changed[:12]} touches {len(chosen)}: {', '.join(s.name for s in chosen) or 'none'}")
    build_lean = (not (a.list or a.explain or a.lean_built) and any(s.name in AFTER_LEAN for s in chosen)
                  and (flock_lean() / "lakefile.toml").exists())
    if build_lean:
        lean_on_path()
    env = environment()
    inputs = {s: inputs_of(s, pkgs) for s in chosen}
    fixtures = fetched_fixtures(tree, {p for ps in inputs.values() for p in ps})
    logs = a.logs or CACHE / "logs" / f"{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    logs.mkdir(parents=True, exist_ok=True)
    if not (a.list or a.explain):
        sweep_basetemps()
    # the repository's own suite keys on the whole tree but reads tracked files only, so it alone never makes a fetch
    needed = {p: art for p, art in fixtures.items() if any(_under(p, [x for x in ps if x != "."]) for ps in inputs.values())}
    unfetched = set() if a.list or a.explain else fetch_fixtures(needed, logs)
    plan, ids_of = [], {}
    meta, lock, provided = metadata() + list(STORE_READS), locked(), set(importlib.metadata.packages_distributions())
    for s in chosen:
        paths = inputs[s]
        ids = input_ids(tree, sorted(set(paths) | set(RUNNER)), markdown(s))
        ids.update({f"fixture:{p}": art for p, art in fixtures.items() if _under(p, paths)})
        deps = resolved(roots(s, pkgs), lock, s.path)
        keys = [key(s, ids, env, args, deps)] + ([key(s, ids, env, extra, deps)] if a.quick else [])   # a full pass also covers --quick
        hit = next((c for c in map(cached, keys) if c and holds(c, tree, lock, provided)), None)
        plan.append((s, paths, keys, hit, deps))
        ids_of[s] = ids

    if a.explain:
        [(s, _paths, keys, hit, deps)] = plan
        ids = ids_of[s]
        now = components(s, ids, env, args, deps)
        per = None if a.no_per_test else PerTest(s, ids, env, [args] + ([extra] if a.quick else []), deps, tree)
        print("\n".join(explain(s, now, keys, hit, per, tree, lock, provided)))
        return 0

    if a.list:
        for s, paths, keys, hit, _deps in plan:
            state = f"cached from {_since(hit)}" if hit else "not cached"
            print(f"{s.name:28} {s.path:28} {state}\n    inputs: {', '.join(paths)}")
        return 0

    head = _git("rev-parse", "HEAD").strip()
    dirty = bool(_git("status", "--porcelain").strip())
    results: list[Result] = [Result(s, "cached", since=hit, key=hit["key"]) for s, _, _, hit, _ in plan if hit and not a.fresh]
    blocked = [s for s, paths, _, hit, _ in plan if (a.fresh or not hit) and any(_under(p, [x for x in paths if x != "."]) for p in unfetched)]
    for s in blocked:
        print(f"failed  {s.name:28} its fixtures could not be fetched: "
              f"{', '.join(sorted(p for p in unfetched if _under(p, inputs[s])))[:300]} (log: {logs / 'fetch-fixtures.log'})")
        results.append(Result(s, "failed", log=logs / "fetch-fixtures.log"))
    todo = [(s, paths, keys, deps) for s, paths, keys, hit, deps in plan if (a.fresh or not hit) and s not in blocked]
    per = {s: PerTest(s, ids_of[s], env, [args] + ([extra] if a.quick else []), deps, tree)
           for s, _paths, _keys, deps in todo if not a.no_per_test}
    whole = {s: w for s, pt in per.items() if not a.fresh and (w := pt.whole(lock, provided))}
    results += [Result(s, "cached", since=w, key=w["key"]) for s, w in whole.items()]
    todo = [t for t in todo if t[0] not in whole]
    per = {s: pt for s, pt in per.items() if s not in whole}
    if build_lean and any(s.name in AFTER_LEAN for s, *_ in todo) and (why := lean_build(logs / "lean-build.log")):
        for s, *_ in todo:
            if s.name in AFTER_LEAN:
                print(f"failed  {s.name:28} {why} (log: {logs / 'lean-build.log'})")
                results.append(Result(s, "failed", log=logs / "lean-build.log"))
        todo = [t for t in todo if t[0].name not in AFTER_LEAN]
    here = kept_passes() if todo else {}
    forced = {s: forced_by(s, components(s, ids_of[s], env, args, deps), here.get(s.name, []),
                           any(map(cached, keys)), a.fresh) for s, _paths, keys, deps in todo}
    known = _durations()
    todo.sort(key=lambda t: -known.get(t[0].name, float("inf")))
    for r in results:
        if r.status == "cached":
            print(f"cached  {r.suite.name:28} passed at {_since(r.since)}"
                  + (" (traced: every file its run read is as it was)" if (r.since or {}).get("traced") else ""))
    for s in skipped:
        print(f"skipped {s.name:28} (--skip)")
    jobs = max(1, a.jobs)
    spread = jobs if XDIST else 1
    if spread == 1 and jobs > 1 and any(t[0].parallel for t in todo):
        print("pytest-xdist is not installed: each parallel suite runs in one process")
    wide = [t[0] for t in todo if t[0].parallel and spread > 1]
    one = [t[0] for t in todo if t[0] not in wide]
    r = serial_slots([known.get(s.name) for s in one], len(wide), jobs)
    lanes = {False: Slots(r), True: Slots(jobs - r)}
    turns = {s: lane.index(s) for lane in (wide, one) for s in lane}
    if wide and one:
        print(f"suites: {len(wide)} parallel suites one at a time on {jobs - r} processes, beside {len(one)} others on {r}")
    min_s = a.per_test_min_s if a.per_test_min_s is not None else float(os.environ.get(MIN_S_ENV) or 0)
    per = {s: pt for s, pt in per.items() if any(t[0] is s for t in todo)}
    reuse = {s: [] if a.fresh or not s.parallel else sorted(pt.reusable(lock, provided)) for s, pt in per.items()}
    hit_of = {s: hit for s, _paths, _keys, hit, _deps in plan}
    stopping = threading.Event()

    def start(s: Suite, paths: list[str]) -> Result:
        n = jobs - r if s in wide else 1
        with lanes[s in wide].take(turns[s], n):
            if stopping.is_set():                  # interrupted: start no suite
                return Result(s, "failed", reported=False)
            return run(s, paths + meta + list(RUNNER), args, logs, n, [m for m in meta if not _under(m, paths)], reuse.get(s, []),
                       (s.timeout_s or SUITE_TIMEOUT_S) if a.timeouts else 0.0)

    def finish(fut) -> None:
        s, k, paths, deps = futures[fut]
        r = fut.result()
        r.key = k
        results.append(r)
        if a.json and r.status == "failed":
            failing(a.json, results)
        known[s.name] = r.seconds
        u = used(r, paths, deps, tree, lock)
        if a.fresh and r.status == "failed":
            r.contradicts = (["(the suite)"] if hit_of.get(s) else []) + (per[s].contradicted(r, lock, provided) if s in per else [])
        kept = per[s].keep(r, head, u, min_s) if s in per else 0
        if s in per:
            per[s].keep_whole(r, head, u, a.run)
        r.nondeterministic = sorted(set(per[s].nondeterministic)) if s in per else []
        print(f"{r.status:7} {s.name:28} {_count(r)} in {r.seconds:.0f} s" + (f" on {r.workers} workers" if r.workers > 1 else "")
              + (f"; {kept} per-test passes kept" if kept and per[s].narrow else f"; {kept} outcomes recorded" if kept else ""))
        if s in forced:
            print(f"        ran because: {'; '.join(forced[s]['why'][:3])}" + (" ..." if len(forced[s]["why"]) > 3 else ""))
        if r.timed_out:
            print(f"        TIMED OUT: still running after {r.timed_out:g} s, so its process group was killed; a suite that needs "
                  f"longer declares [tool.verity.tests] timeout_s in {s.path}/pyproject.toml")
        elif r.signal:
            mem = memory_gb()
            print(f"        {'OOM-KILLED' if r.oom else 'killed by ' + r.signal}: pytest died by {r.signal} at a peak RSS of "
                  f"{r.peak_rss_gb:.1f} GB" + (f" on a machine with {mem:.0f} GB" if mem else "")
                  + ("; run it in a check slot (`research queue ready N --on vy-nebius-1`, or `check.py --on`)" if r.oom else ""))
        if r.leftover:
            print(f"        killed {r.leftover} processes its tests left in the suite's process group")
        if r.child_reads:
            def who(w: list[str]) -> str:
                return " and ".join(w) if len(w) < 3 else f"{w[0]}, {w[1]} and {len(w) - 2} more"
            files = [f"{rel}: read by a process that {who(w)} started" for k, rel, w in r.child_reads if k == "O"]
            blind = [f"a Python that {who(w)} started could not start its read guard ({why.removeprefix('the tracer did not start: ')})"
                     for k, why, w in r.child_reads if k == "E"]
            lines = files + blind
            print(f"        processes its tests started read outside its inputs, so nothing vouches for its pass:\n          "
                  + "\n          ".join(lines[:10]) + (f"\n          and {len(lines) - 10} more" if len(lines) > 10 else "")
                  + (f"\n        list each file under [tool.verity.tests] inputs in {s.path}/pyproject.toml with a comment saying why, "
                     "or give the test a stand-in for it" if files else ""))
        if r.nondeterministic:
            print(f"        NONDETERMINISTIC: {', '.join(r.nondeterministic[:5])}{' ...' if len(r.nondeterministic) > 5 else ''} both passed "
                  "and failed under one key: their inputs were identical, so each is flaky; delete it, don't rerun or fix it "
                  "(Daniel, 6 Oct)")
        if r.contradicts:
            print(f"        CONTRADICTS a cached pass: {', '.join(r.contradicts[:5])}{' ...' if len(r.contradicts) > 5 else ''}"
                  " -- a key misses an input, or a test's outcome isn't a function of its inputs")
        if r.status == "passed":
            CACHE.mkdir(parents=True, exist_ok=True)
            (CACHE / f"{k}.json").write_text(json.dumps({"suite": s.name, "path": s.path, "key": k, "commit": head, "dirty": dirty,
                                                          "run": a.run, "when": time.strftime("%Y-%m-%d %H:%M"), "counts": r.counts,
                                                          "reused": r.reused, "seconds": round(r.seconds, 1), **u,
                                                          "components": components(s, ids_of[s], env, args, deps)}, indent=1) + "\n")
        elif r.outside:
            few = [p + (f" (read by {m})" if m != "*" else " (read by a conftest or session fixture)")
                   for p, m in list(r.outside.items())[:3]]
            print(f"        its tests read {len(r.outside)} files outside its inputs, for example {', '.join(few)}:"
                  f" depend on the package, or list the path under [tool.verity.tests] inputs in {s.path}/pyproject.toml"
                  f" (a file of a dependency that lists `exports`: under those, if its dependents read it)")
        elif r.unguarded:
            print(f"        xdist workers {', '.join(r.unguarded)} sent the guard no record of what they read\n        log: {r.log}")
        elif r.log is not None and not (r.ran_all and r.child_reads):     # failed on its children's reads alone: said above
            if not r.reported:
                print("        the guard wrote no report, so nothing vouches for what the tests read")
            print("        " + "\n        ".join(r.log.read_text().splitlines()[-25:]) + f"\n        log: {r.log}")

    if a.json and any(r.status == "failed" for r in results):
        failing(a.json, results)
    elif a.json:
        failing_path(a.json).unlink(missing_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(todo))) as pool:
        futures = {pool.submit(start, s, paths): (s, keys[0], paths, deps) for s, paths, keys, deps in todo}
        handled = set()
        try:
            for fut in concurrent.futures.as_completed(futures):
                handled.add(fut)
                finish(fut)
        except KeyboardInterrupt:
            # a stop (check.py's, or ^C) reaches the suites' pytest processes too: keep what each finished run's guard vouches for
            stopping.set()
            with _GROUPS_LOCK:
                for g in list(_GROUPS):
                    kill_group(g, signal.SIGINT)
            for fut in futures:
                if fut not in handled:
                    finish(fut)
            raise
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "durations.json").write_text(json.dumps(known, indent=1, sort_keys=True) + "\n")
    written = _git("diff-tree", "-r", "--name-only", tree, worktree()).split() if todo else []
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
                                       "failed_tests": r.failed_tests, "log": str(r.log) if r.log else None,
                                       "outside": r.outside, "child_outside": r.child_outside, "unguarded": r.unguarded,
                                       "since": r.since, "reused": r.reused,
                                       "contradicts": r.contradicts, "nondeterministic": r.nondeterministic,
                                       "timed_out": r.timed_out, "forced_by": forced.get(r.suite)} for r in results]
                                     + [{"suite": s.path, "name": s.name, "status": "skipped"} for s in skipped], indent=1) + "\n")
    return 1 if failed or written else 0


if __name__ == "__main__":
    sys.exit(main())
