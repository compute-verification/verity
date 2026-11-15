#!/usr/bin/env python3
"""`tools/move/tests_root.py`: every test in the tree under the root's `tests/`, mirrored by path, from the tree it is run on.

    python3 tools/move/tests_root.py inventory          each tests/ directory, its suite, and what the rule can't move as stated
    python3 tools/move/tests_root.py split [NAME...]    each distribution's suite to a virtual member `NAME-tests` in its tests/
    python3 tools/move/tests_root.py maps               write `tests_root_map.toml`: every test under the root's tests/
    python3 tools/move/tests_root.py suites-map         after it moved them, `tests_root_suites_map.toml`: each suite's pyproject
    python3 tools/move/tests_root.py anchors BASE [--write]   each moved file's anchor that no longer names what it did at BASE

The rule: a file at `X/tests/REST` goes to `tests/X/REST`, for each `X` under `AREAS`; the root suite's own files stay at the
top of `tests/`. A suite's pyproject goes beside its tests (`tests/X/pyproject.toml`), so its directory is its tests' and
`tools/check/suites.py` names it as `X` named it (`suite_name`). Stdlib and git only, like the movers; run from the root.
Where the rule can't hold as stated, the smallest change that keeps every suite running:

- **A `tests/` that is a package its files import as `tests`** (`from tests.lint import ...`, the vLLM and torch adapters):
  it goes to `tests/X/tests/`, one level kept, so its modules keep their names (`KEPT`). Moved as stated, `tests/X/` would be
  a package named for `X`'s last part (`vllm`), which shadows the distribution of that name.
- **A suite whose pyproject is in its own `tests/`** (`verity/tests/`, `verity/protocols/tests/`): it moves whole, to
  `tests/X/`, the inner `tests/` with it; the pyproject needs no row.
- **A suite whose pyproject builds a distribution**: no row can split a file, so `split` moves the pytest and test tables
  to a virtual member `NAME-tests` in `X/tests/` (testpaths `.`, the lists it writes from the suite's directory without
  `tests/`, `pythonpath` from there), first, as a commit of its own; `exports`, what the distribution's dependents key on,
  stays, and a suite whose inputs replace its dependencies (`inputs_only`) or that `exports` narrows reads `X` by name; the files map then moves it with the tests. Its
  `conftest.py` beside the pyproject (the vLLM adapter's) is the suite's and goes into `X/tests/` too: pytest reads none
  above the suite's directory. A dependency group the distribution keeps for its own tests is the member's too.
- **A virtual member that stays a member where it is** (`MEMBERS_KEPT`, with why): `split` treats it as a distribution's
  suite, but the member `NAME-tests` has its dependencies, not a dependency on it (nothing installs a virtual member),
  and reads `X` by name.
- **Every other suite's pyproject**: the suites map moves it to `tests/X/pyproject.toml` (`testpaths` `tests` to `.`,
  `strip_suite_relative`, `pythonpath_from`), with `X` added to its inputs where nothing else of its inputs covers it: its
  own directory did.

Each map is written from the tree as it is, so the sequence reruns on `main` after any later move: `split`, `maps`,
`layout.py --map tests_root_map.toml`, `suites-map`, `layout.py --map tests_root_suites_map.toml`, `anchors BASE --write`
(BASE the commit before the first move: `layout.py` recomputes a `Path(__file__)` anchor it reads as joins, but not one
through a name bound to it, `HERE.parent`). A map's `[modules]` is
what `layout.py --modules` derives, and its `[split]` rows are what its dry run finds: a reference to a directory whose
tests or suite's pyproject alone move means what stays (a pythonpath entry, which `pythonpath_from` writes again, too)."""
from __future__ import annotations

import argparse
import io
import json
import posixpath
import re
import subprocess
import sys
import tempfile
import tokenize
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from layout import _add_inputs, _strip_suite_relative  # noqa: E402
#: the top-level directories whose tests move under the root's `tests/`
AREAS = ("verity", "adapters", "benchmarks", "tools", "infra")
#: the root's directory that mirrors the tree by path (`tools/check/suites.py`'s `MIRROR`)
MIRROR = "tests"
FILES_MAP, SUITES_MAP = "tools/move/tests_root_map.toml", "tools/move/tests_root_suites_map.toml"
#: a module of a `tests/` package imports its siblings as `tests.X`: the package keeps its name (`KEPT`)
IMPORTS_TESTS = re.compile(r"^\s*(?:from|import)\s+tests(?:\.|\s|$)", re.M)
#: why a test's reference to a directory only whose tests move means the rest of it (`split_rows`)
REQUIREMENT = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
SPLIT_WHY = "the code beside the tests, which stays"
#: never rewritten: records of what was, files whose bytes a digest pins, and every Lean input; as the earlier maps'
EXCLUDE = ["fixtures/**", "**/fixtures/**", "**/*vectors*.json", "**/*vectors*.txt.gz", "**/CHANGELOG.md", "*.patch",
           "**/TRUSTED.sha256", "**/lake-manifest.json.lock", "tools/move/**", "archive/**",
           "**/numerical/tests/bench/data/**", "**/price_twins.json", "verity/circuits/silicon/captures/**",
           "verity/circuits/silicon/hawkeye/**", "tools/research/src/research/store/moves.json",
           "**/network_e2e/tests/data/**", "adapters/vllm/manifests/**", "adapters/vllm/workloads/**",
           "adapters/vllm/data/**", "adapters/vllm/verity_vllm/pins/**", "adapters/vllm/tools/migrate/superseded_quarantine.py",
           "**/migrate/test_superseded_quarantine.py", "**/*.lean", "**/lakefile.toml", "tests/tree.toml", "**/*.cu",
           "**/*.cuh", "**/*.cpp", "**/*.h", "**/*.c", "benchmarks/pouw/pearl_c_sm90/**"]
#: the mover's own pyproject is the suite's, which the suites map rewrites, though `tools/move/**` is never rewritten
INCLUDE = ["**/fixtures/README.md", "**/fixtures/**/README.md", "tools/move/pyproject.toml"]
#: {art: id: path} of every replicated fixture, whose bytes its id pins
ARTIFACTS = "artifacts.json"
#: `from __future__ import ...`, which must open a module
FUTURE = re.compile(r"^from __future__ import [^\n]*\n", re.M)
#: the tables a suite's pyproject holds for pytest and the check, which `split` moves
SUITE_TABLES = ("tool.pytest", "tool.verity.tests")
#: virtual workspace members that stay members where they are, their suite split off as a distribution's is: {dir: why}
MEMBERS_KEPT = {"tools/check": "ci's rule: a cold `uv run tools/check/*.py` syncs `verity-check`'s project, the whole "
                               "workspace; with no project there it syncs only the root's default groups"}


def git(*args: str, root: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout


def under(p: str, d: str) -> bool:
    return d in ("", ".") or p == d or p.startswith(d.rstrip("/") + "/")


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class Tree:
    """The tracked files of the worktree, and its suites."""

    def __init__(self, root: Path = ROOT):
        self.root = root
        self.files = [f for f in subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, text=True,
                                                check=True).stdout.split("\0") if f]
        self.tracked = set(self.files)
        self.pyprojects = {}
        for f in self.files:
            if posixpath.basename(f) == "pyproject.toml":
                self.pyprojects[posixpath.dirname(f) or "."] = tomllib.loads((root / f).read_text())
        #: {suite directory: its pytest testpaths, root-relative}
        self.suites = {d: [posixpath.normpath(posixpath.join(d, t)) for t in
                           data["tool"]["pytest"].get("ini_options", {}).get("testpaths", ["."])]
                       for d, data in self.pyprojects.items() if "pytest" in data.get("tool", {})}
        root_data = self.pyprojects["."]
        self.members = list(root_data.get("tool", {}).get("uv", {}).get("workspace", {}).get("members", []))
        self.sources = set(root_data.get("tool", {}).get("uv", {}).get("sources", {}))
        self.names = {norm(self.pyprojects[d]["project"]["name"]): d for d in self.members if d in self.pyprojects}

    def read(self, p: str) -> str:
        try:
            return (self.root / p).read_text()
        except (UnicodeDecodeError, FileNotFoundError):
            return ""

    def distribution(self, d: str) -> bool:
        """Whether the pyproject at `d` builds a workspace distribution: a member's `[project]` with a build system, not
        virtual. One no member lists is a suite's pyproject like any other: nothing installs it, its suite imports its code by
        `pythonpath`, and it moves whole."""
        data = self.pyprojects[d]
        return (d in self.members and bool(data.get("project")) and "build-system" in data
                and data.get("tool", {}).get("uv", {}).get("package") is not False)

    def splits(self, d: str) -> bool:
        """Whether `split` moves the suite at `d` to a member in its tests/: a distribution's, or a member's that stays one."""
        return self.distribution(d) or (d in self.members and d in MEMBERS_KEPT)

    def test_dirs(self) -> dict[str, list[str]]:
        """{`X/tests`: its tracked files}, the outermost `tests` of each path under `AREAS`."""
        out = defaultdict(list)
        for f in self.files:
            parts = f.split("/")
            if parts[0] in AREAS and "tests" in parts[:-1]:
                out["/".join(parts[:parts.index("tests") + 1])].append(f)
        return dict(sorted(out.items()))

    def owner(self, d: str) -> str | None:
        """The suite whose testpaths hold directory `d`, the innermost."""
        held = [s for s, tps in self.suites.items() if s != "." and any(under(d, t) or under(t, d) for t in tps)]
        return max(held, key=len, default=None)

    def kept(self, d: str) -> bool:
        """Whether `d` (`X/tests`) is a package its suite's modules import as `tests` (`KEPT`)."""
        if f"{d}/__init__.py" not in self.tracked:
            return False
        suite = self.owner(d) or d
        return any(f.endswith(".py") and under(f, suite) and IMPORTS_TESTS.search(self.read(f)) for f in self.files)

    def destination(self, d: str) -> str:
        x = d.removesuffix("/tests")
        return f"{MIRROR}/{x}/tests/" if self.kept(d) else f"{MIRROR}/{x}/"


# inventory ---------------------------------------------------------------------------------------------------------------

def inventory(t: Tree) -> dict:
    dirs = t.test_dirs()
    rows, snags = [], defaultdict(list)
    for d, files in dirs.items():
        suite = t.owner(d)
        py = sum(f.endswith(".py") for f in files)
        lean = sum(f.endswith(".lean") for f in files)
        lake = [f for f in files if posixpath.basename(f) in ("lakefile.toml", "lakefile.lean", "lake-manifest.json")]
        kind = ("none" if suite is None else "inside" if suite == d else "distribution" if t.distribution(suite)
                else "virtual" if t.pyprojects[suite].get("project") else "suite-only")
        rows.append({"dir": d, "area": d.split("/")[0], "files": len(files), "py": py, "lean": lean, "suite": suite,
                     "kind": kind, "kept": t.kept(d), "lands": t.destination(d)})
        if suite is None:
            snags["a tests/ directory no suite collects"].append(d)
        if kind == "inside":
            snags["a suite whose pyproject is in its own tests/ (moves whole)"].append(d)
        if kind == "distribution":
            snags["a suite whose pyproject builds a distribution (split first)"].append(f"{d} ({t.pyprojects[suite]['project']['name']})")
        elif suite and t.splits(suite):
            snags["a suite of a member that stays one (split first)"].append(f"{d} ({t.pyprojects[suite]['project']['name']})")
        if t.kept(d):
            snags["a tests/ package its modules import as `tests` (one level kept)"].append(d)
        elif f"{d}/__init__.py" in t.tracked:
            snags["a tests/ package no module imports as `tests` (moved as stated)"].append(d)
        if lake:
            snags["a Lake package among the test files (moved as data)"].append(f"{d}: {len(lake)} Lake files")
        if lean:
            snags["Lean files among the test files (standalone, outside every Lake build)"].append(f"{d}: {lean}")
    for f in t.files:
        parts = f.split("/")
        base = parts[-1]
        if parts[0] in AREAS and "tests" not in parts[:-1] and (base == "conftest.py" or re.match(r"test_.*\.py$", base)):
            snags["a test-like file outside every tests/ (conftest beside a suite's pyproject, or a module named test_*)"].append(f)
    for s in t.suites:
        if s != "." and s.split("/")[0] in AREAS and not any(under(d, tp) or under(tp, d) for d in dirs for tp in t.suites[s]):
            snags["a suite with no tests/ directory"].append(s)
    by_area = defaultdict(Counter)
    for r in rows:
        by_area[r["area"]].update({"tests dirs": 1, "files": r["files"], "py": r["py"]})
    suites = Counter(s.split("/")[0] for s in t.suites if s.split("/")[0] in AREAS)
    kinds = Counter((r["area"], r["kind"]) for r in rows)
    return {"rows": rows, "snags": dict(snags), "areas": {a: {**by_area[a], "suites": suites[a],
                                                              **{k: n for (ar, k), n in kinds.items() if ar == a}}
                                                          for a in AREAS}}


# split -------------------------------------------------------------------------------------------------------------------

def _tables(text: str) -> list[tuple[str | None, str]]:
    """[(the table's name, `[[x]]` as `x`, or None for the preamble; its text from its header line)]"""
    out, name, buf = [], None, []
    for line in text.splitlines(keepends=True):
        m = re.match(r"^\[\[?\s*([^\[\]]+?)\s*\]\]?\s*(?:#.*)?$", line.rstrip("\n"))
        if m:
            out.append((name, "".join(buf)))
            name, buf = m.group(1), [line]
        else:
            buf.append(line)
    out.append((name, "".join(buf)))
    for i in range(len(out) - 1):
        lines = out[i][1].splitlines(keepends=True)
        cut = len(lines)
        while cut and (lines[cut - 1].startswith("#") or (not lines[cut - 1].strip() and cut < len(lines))):
            cut -= 1
        while cut < len(lines) and not lines[cut].strip():
            cut += 1
        out[i], out[i + 1] = (out[i][0], "".join(lines[:cut])), (out[i + 1][0], "".join(lines[cut:]) + out[i + 1][1])
    return [(n, b) for n, b in out if n is not None or b.strip()]


def _from_tests(body: str) -> str:
    """The pytest table of a suite whose directory becomes its `tests/`: `testpaths` `tests` is `.`, and each `pythonpath` entry,
    written from the old directory, is written from the new one."""
    body = re.sub(r'(^testpaths\s*=\s*\[[^\]]*)"tests"', r'\1"."', body, flags=re.M)

    def entry(m: re.Match) -> str:
        items = [json.dumps(posixpath.normpath(posixpath.join("..", x))) for x in json.loads("[" + m.group(2) + "]")]
        return m.group(1) + ", ".join(items) + "]"
    return re.sub(r'^(pythonpath\s*=\s*\[)([^\]]*)\]', entry, body, flags=re.M)


def _exports(body: str) -> tuple[str, str] | None:
    """(`exports` with the comment lines above it, the table without them) of a `[tool.verity.tests]` table, or None: what its
    dependents key on (`suites.py`'s `shipped`) is the distribution's, and stays in its pyproject."""
    lines = body.splitlines(keepends=True)
    start = next((i for i, x in enumerate(lines) if re.match(r"^exports\s*=\s*\[", x)), None)
    if start is None:
        return None
    end = start
    while "]" not in lines[end].split("#", 1)[0]:
        end += 1
    while start > 1 and lines[start - 1].startswith("#"):
        start -= 1
    return "".join(lines[start:end + 1]).rstrip("\n"), "".join(lines[:start] + lines[end + 1:])


def split(t: Tree, d: str) -> dict[str, str]:
    """The files `split` writes for the distribution, or the member that stays one, at `d`: its pyproject without the
    suite's tables, the member's in `d/tests/`, the root's with the member; and the conftest beside it moved in. {path:
    text, or None to delete}."""
    text = t.read(f"{d}/pyproject.toml")
    data = t.pyprojects[d]
    name, project = data["project"]["name"], data["project"]
    virtual = not t.distribution(d)
    keep, moved, sources = [], [], ""
    for table, body in _tables(text):
        if table == "tool.uv.sources":
            sources = body
        if table == "tool.verity.tests" and (kept := _exports(body)):
            keep.append("[tool.verity.tests]\n" + kept[0] + "\n")
            body = kept[1]
        if table and table.startswith(SUITE_TABLES):
            moved.append(_from_tests(body) if table.startswith("tool.pytest") else body)
        else:
            keep.append(body)
    keep_text = "".join(keep).rstrip("\n") + "\n"
    # the extras that name a workspace package, which the suite's key reads; another (torch) would change the environment
    extras = sorted(e for e, reqs in project.get("optional-dependencies", {}).items()
                    if any((m := REQUIREMENT.match(r)) and norm(m.group(1)) in t.names for r in reqs))
    deps = project.get("dependencies", []) if virtual else [name + (f"[{','.join(extras)}]" if extras else "")]
    head = [f"# The suite of the distribution `{name}`: a virtual member that depends on it, so the suite keys on what the\n"
            "# distribution's dependents read, and `tools/check/suites.py` names it as the distribution.\n" if not virtual else
            f"# The suite of the workspace member `{name}`, which stays a member where it is: a virtual member with its\n"
            "# dependencies, which `tools/check/suites.py` names as the member.\n",
            f'[project]\nname = "{norm(name)}-tests"\nversion = "0.0.0"\n'
            f'requires-python = "{project.get("requires-python", ">=3.12")}"\ndependencies = {json.dumps(deps)}\n\n']
    if groups := data.get("dependency-groups"):
        head.append("[dependency-groups]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in groups.items()) + "\n")
    head.append("# Not an installable package: a workspace member so that its tests run in the workspace environment.\n"
                "[tool.uv]\npackage = false\n\n")
    if virtual and sources:
        head.append(sources.rstrip("\n") + "\n\n")
    body = "".join(moved)
    member = "".join(head) + _strip_suite_relative(body, "tests/").rstrip("\n") + "\n"
    tests = data.get("tool", {}).get("verity", {}).get("tests", {})
    if virtual or tests.get("inputs_only") or "exports" in tests:
        member = _add_inputs(member, [d], first=True)
    out = {f"{d}/pyproject.toml": keep_text, f"{d}/tests/pyproject.toml": member}
    rootp = t.read("pyproject.toml")
    if f"{d}/tests" not in t.members:
        m = re.search(r"^members\s*=\s*\[[^\]]*\]", rootp, re.M)
        listed = m.group(0).replace(f'"{d}"', f'"{d}", "{d}/tests"', 1)
        rootp = rootp[:m.start()] + listed + rootp[m.end():]
    if not virtual and norm(name) not in {norm(s) for s in t.sources}:
        # a member's own `workspace = true` source for a member whose directory holds it is refused by uv: the root names it
        m = re.search(r"^\[tool\.uv\.sources\]\n(?:[^\[\n].*\n)*", rootp, re.M)
        rootp = rootp[:m.end()] + f"{name} = {{ workspace = true }}\n" + rootp[m.end():]
    if rootp != t.read("pyproject.toml"):
        out["pyproject.toml"] = rootp
    if f"{d}/conftest.py" in t.tracked:
        out[f"{d}/conftest.py"] = None
        inner = t.read(f"{d}/tests/conftest.py")
        out[f"{d}/tests/conftest.py"] = _merge_conftests(t.read(f"{d}/conftest.py"), inner)
    return out


def _merge_conftests(outer: str, inner: str) -> str:
    """One conftest from the one beside a distribution's pyproject and the one in its `tests/`: the outer first, as pytest
    loaded it first. A file with none inside is the outer one. The inner one's `from __future__` imports join the outer's,
    which open the module."""
    if not inner.strip():
        return outer
    futures = [f for f in FUTURE.findall(inner) if f not in outer]
    if futures:
        at = max((m.end() for m in FUTURE.finditer(outer)), default=0)
        outer = outer[:at] + "".join(futures) + outer[at:]
    return outer.rstrip("\n") + "\n\n\n# The conftest of the suite's tests/ before its tables moved there.\n" + FUTURE.sub("", inner)


def run_split(t: Tree, names: list[str]) -> list[str]:
    todo = [s for s in t.suites if s != "." and s.split("/")[0] in AREAS and t.splits(s)
            and (not names or norm(t.pyprojects[s]["project"]["name"]) in {norm(n) for n in names})]
    done = []
    for d in sorted(todo):
        if f"{d}/tests/pyproject.toml" in t.tracked:
            continue
        for p, text in split(t, d).items():
            if text is None:
                git("rm", "-q", p, root=t.root)
            else:
                (t.root / p).write_text(text)
                git("add", p, root=t.root)
        t = Tree(t.root)
        done.append(d)
    return done


# maps --------------------------------------------------------------------------------------------------------------------

def roots(t: Tree) -> dict[str, str]:
    """{import name: its directory} of each package a member ships (hatch's `packages`, else `src/NAME` or `NAME`) and the
    root's `verity`."""
    out = {"verity": "verity"}
    for d in t.members:
        data = t.pyprojects.get(d, {})
        if not t.distribution(d):
            continue
        pkgs = data.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {}).get("packages")
        if not pkgs:
            n = norm(data["project"]["name"]).replace("-", "_")
            pkgs = [p for p in (f"src/{n}", n) if f"{d}/{p}/__init__.py" in t.tracked][:1]
        for p in pkgs:
            if f"{d}/{p}/__init__.py" in t.tracked:
                out.setdefault(posixpath.basename(p), f"{d}/{p}")
    return dict(sorted(out.items()))


def _toml_list(items: list[str], indent: int = 4) -> str:
    return "[\n" + "".join(f"{' ' * indent}{json.dumps(i)},\n" for i in items) + "]"


def _registered(t: Tree) -> list[str]:
    """The tracked replicated fixtures (`artifacts.json`): rewritten, a fixture no longer hashes to its id."""
    if not (t.root / ARTIFACTS).is_file():
        return []
    paths = json.loads((t.root / ARTIFACTS).read_text()).values()
    return sorted(p for p in paths if isinstance(p, str) and p in t.tracked)


def files_map(t: Tree) -> str:
    dirs = t.test_dirs()
    rows = {f"{d}/": t.destination(d) for d in dirs}
    r = roots(t)
    inline = "{ " + ", ".join(f'{k} = "{v}"' for k, v in r.items()) + " }"
    out = ["# Every test in the tree to the root's `tests/`, mirrored by path: `X/tests/REST` becomes `tests/X/REST` (a `tests/`\n"
           "# package its modules import as `tests` keeps its level: `tests/X/tests/REST`). Written by `tools/move/tests_root.py\n"
           "# maps` from the tree it ran on; `tools/move/layout.py --map tools/move/tests_root_map.toml` moves and rewrites.\n\n",
           f"[roots]\nold = {inline}\nnew = {inline}\n\n\n[paths]\n",
           "".join(f"{json.dumps(o)} = {json.dumps(n)}\n" for o, n in rows.items()),
           "\n[retired]\n# the paths this map empties\npaths = " + _toml_list(list(rows)) + "\n\n",
           "[rewrite]\n# Never rewritten: records of what was, files whose bytes a digest pins, and every Lean input.\n",
           "exclude = " + _toml_list(EXCLUDE + _registered(t)) + "\n", "include = " + _toml_list(INCLUDE) + "\n",
           "# a test's relative paths name a toy's files; its anchors are `tests_root.py anchors`\n",
           "keep_relative = " + _toml_list([f"{d}/*.py" for d in dirs]) + "\n"]
    for d in dirs:
        if f"{d}/pyproject.toml" in t.tracked and _pythonpath(t, d):
            out.append(f"\n[pyproject.{json.dumps(t.destination(d) + 'pyproject.toml')}]\npythonpath_from = {json.dumps(d)}\n")
    return "".join(out)


def split_rows(path: str, root: Path = ROOT) -> str:
    """The `[split]` table for the map at `path`: each reference `layout.py` finds to a directory the map splits, which is a
    directory of the tree whose tests alone move, so the reference means what stays (`SPLIT_WHY`)."""
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.json"
        r = subprocess.run([sys.executable, str(HERE / "layout.py"), "--root", str(root), "--map", path, "--dry-run", "--no-lock", "--allow-dirty",
                            "--report-splits", "--report", str(report)], cwd=root, capture_output=True, text=True)
        if r.returncode:
            raise SystemExit(f"tests_root: layout.py --dry-run on {path}: {r.stderr.strip()[-2000:]}")
        notes = json.loads(report.read_text())["notes"].get("references to a split directory [split] doesn't acknowledge", [])
    rows = sorted({m.group(1) for x in notes if (m := re.match(r"^(\S+: \S+) \(", x))})
    return "".join(f"{json.dumps(k)} = {{ to = {json.dumps(k.split(': ', 1)[1])}, why = {json.dumps(SPLIT_WHY)} }}\n" for k in rows)


def _pythonpath(t: Tree, s: str) -> list[str]:
    return t.pyprojects[s].get("tool", {}).get("pytest", {}).get("ini_options", {}).get("pythonpath", [])


def _covered(t: Tree, s: str) -> bool:
    """Whether suite `s`'s inputs other than its own directory cover `s`: a declared input, or a workspace dependency's
    directory (the root's `verity` for a path under `verity/`)."""
    data = t.pyprojects[s]
    tests = data.get("tool", {}).get("verity", {}).get("tests", {})
    ins = [i for i in tests.get("inputs", []) if isinstance(i, str)]
    deps = {norm(re.split(r"[\s\[<>=!~;@]", r, maxsplit=1)[0]) for r in data.get("project", {}).get("dependencies", [])}
    for g in data.get("dependency-groups", {}).values():
        deps |= {norm(re.split(r"[\s\[<>=!~;@]", r, maxsplit=1)[0]) for r in g if isinstance(r, str)}
    named = [t.names.get(norm(i), i) for i in ins] + [t.names[n] for n in deps if n in t.names]
    return any(under(s, p) for p in named + (["verity"] if "verity" in deps else []))


def suites_map(t: Tree) -> str:
    rows, tables = {}, []
    for s, tps in sorted(t.suites.items()):
        if s == "." or s.split("/")[0] not in AREAS or t.splits(s) or s.endswith("/tests") or tps != [f"{s}/tests"]:
            continue
        kept = t.kept(f"{s}/tests")
        new = f"{MIRROR}/{s}/{'tests/' if kept else ''}pyproject.toml"
        rows[f"{s}/pyproject.toml"] = new
        conf = [] if kept else ['testpaths_replace = { "tests" = "." }', 'strip_suite_relative = "tests/"']
        if _pythonpath(t, s):
            conf.append(f"pythonpath_from = {json.dumps(s)}")
        if any(not under(f, f"{s}/tests") and f != f"{s}/pyproject.toml" for f in t.files if under(f, s)) and not _covered(t, s):
            conf.append(f"add_inputs = [{json.dumps(s)}]")
        if conf:
            tables.append(f"[pyproject.{json.dumps(new)}]\n" + "\n".join(conf) + "\n")
    r = roots(t)
    inline = "{ " + ", ".join(f'{k} = "{v}"' for k, v in r.items()) + " }"
    return "".join([
        "# Each suite's pyproject beside its tests, after `tests_root_map.toml`: `X/pyproject.toml` becomes\n"
        "# `tests/X/pyproject.toml`, its `testpaths` the suite's own directory and its suite-relative lists without `tests/`, and\n"
        "# `X` one of its inputs where nothing else covers it. A distribution's suite moved with its tests (`tests_root.py split`).\n"
        "# Written by `tools/move/tests_root.py maps`.\n\n",
        f"[roots]\nold = {inline}\nnew = {inline}\n\n\n[paths]\n",
        "".join(f"{json.dumps(o)} = {json.dumps(n)}\n" for o, n in rows.items()), "\n",
        "[rewrite]\n# as tests_root_map.toml's\n", "exclude = " + _toml_list(EXCLUDE + _registered(t)) + "\n",
        "include = " + _toml_list(INCLUDE) + "\n\n",
        "\n".join(tables)])


# anchors -----------------------------------------------------------------------------------------------------------------

#: `Path(__file__)` resolved, then `.parent`s or `.parents[N]`: a directory found from a file's own place
ANCHOR = r"(?:pathlib\.)?Path\(__file__\)(?:\.resolve\(\)|\.absolute\(\))?(?P<up>(?:\.parent\b)+|\.parents\[\d+\](?:\.parent\b)*)"
#: literal joins after an anchor
JOINS = r"(?P<joins>(?:\s*/\s*(?:\"[^\"\n]*\"|'[^'\n]*'))*)"
NAMED = re.compile(r"^(?P<name>[A-Z_][A-Z0-9_]*)\s*=\s*(?P<open>\(?)" + ANCHOR + JOINS + r"(?P<close>\)?)\s*$", re.M)


def _levels(up: str) -> int:
    """How many directories above the file `up` (`.parent.parent`, `.parents[1]`) names: the file's directory is 1."""
    m = re.match(r"\.parents\[(\d+)\]", up)
    return (int(m.group(1)) + 1 if m else 0) + up.count(".parent") - (1 if m else 0)


def _ancestor(path: str, n: int) -> str:
    for _ in range(n):
        path = posixpath.dirname(path)
    return path


def _segments(joins: str) -> list[str]:
    return [x for seg in re.findall(r"[\"']([^\"'\n]*)[\"']", joins) for x in seg.split("/") if x]


def _strings(text: str) -> list[tuple[int, int]]:
    """The spans of a module's string literals (a toy file's text a test writes holds anchors that aren't the test's)."""
    starts = [0]
    for line in text.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.STRING or tok.type == getattr(tokenize, "FSTRING_START", -1):
                (l0, c0), (l1, c1) = tok.start, tok.end
                if tok.type == tokenize.STRING:
                    out.append((starts[l0 - 1] + c0, starts[l1 - 1] + c1))
    except (tokenize.TokenError, IndentationError, SyntaxError) as e:
        print(f"tests_root.py: a text that doesn't tokenize keeps the strings before {e}", file=sys.stderr)
    return out


def _names(text: str) -> dict[str, tuple[int, list[str]]]:
    """Each module constant bound to an anchor, or to another such name joined (`LIVE = REPO / "x"`): {name: (the levels
    above the file it starts from, its literal joins, `..` for a name's `.parent`)}."""
    names = {m.group("name"): (_levels(m.group("up")), _segments(m.group("joins"))) for m in NAMED.finditer(text)
             if bool(m.group("open")) == bool(m.group("close"))}
    while names:
        derived = re.compile(r"^(?P<name>[A-Z_][A-Z0-9_]*)\s*=\s*(?P<open>\(?)(?P<base>" + "|".join(map(re.escape, names))
                             + r")\b(?P<up>\.parents\[\d+\](?:\.parent\b)*|(?:\.parent\b)*)" + JOINS + r"(?P<close>\)?)\s*$", re.M)
        more = {m.group("name"): (names[m.group("base")][0],
                                  names[m.group("base")][1] + [".."] * _levels(m.group("up")) + _segments(m.group("joins")))
                for m in derived.finditer(text) if m.group("name") not in names and bool(m.group("open")) == bool(m.group("close"))}
        if not more:
            break
        names.update(more)
    return names


def _anchors(text: str) -> list[tuple[int, int, str | None, int, list[str]]]:
    """Each anchor of a file outside its string literals: (start, end, the name it starts from or None for
    `Path(__file__)`, the levels above its start it names, its literal joins, those after a parenthesis around it too). A
    name is a module constant bound to an anchor; `NAME.parent... / "x"` counts from the directory it names, and a use
    of the name joined to nothing is one too, but where it is bound."""
    names = _names(text)
    found = [(m, None, _levels(m.group("up"))) for m in re.finditer(ANCHOR + JOINS, text)]
    if names:
        named = (r"\b(?P<name>" + "|".join(map(re.escape, names)) + r")\b(?!\s*=[^=])(?P<up>\.parents\[\d+\](?:\.parent\b)*|(?:\.parent\b)*)"
                 + JOINS)
        found += [(m, m.group("name"), _levels(m.group("up"))) for m in re.finditer(named, text)
                  if text[m.end():m.end() + 1] != "(" and not text[:m.start()].rstrip().endswith(".")]
    strings = _strings(text)
    out = []
    for m, name, levels in found:
        a, b, segs = m.start(), m.end(), _segments(m.group("joins"))
        if any(x <= a < y for x, y in strings):
            continue
        if a and text[a - 1] == "(" and text[b:b + 1] == ")" and (more := re.match(JOINS, text[b + 1:])) and more.group("joins"):
            a, b, segs = a - 1, b + 1 + more.end(), segs + _segments(more.group("joins"))
        out.append((a, b, name, levels, segs))
    return sorted(out)


def _rows(root: Path) -> list[tuple[str, str]]:
    rows = []
    for path in (FILES_MAP, SUITES_MAP):
        if (root / path).is_file():
            rows += list(tomllib.loads((root / path).read_text()).get("paths", {}).items())
    return rows


def _mapped(p: str, rows: list[tuple[str, str]]) -> str:
    for o, n in rows:
        if o.endswith("/") and (p + "/").startswith(o):
            return posixpath.normpath(n + p[len(o):]) if p + "/" != o else n.rstrip("/")
        if p == o:
            return n
    return p


def _expression(name: str | None, levels: int, segs: list[str], style: str) -> str:
    """An anchor naming the directory `levels` above the file (for `name`, the levels above it), then `segs`."""
    if name is None:
        head = style + (".parent" if levels == 1 else f".parents[{levels - 1}]")
    else:
        head = name if levels == 0 else f"{name}.parent" if levels == 1 else f"{name}.parents[{levels - 1}]"
    return head + "".join(f' / "{x}"' for x in segs)


def _start(path: str, name: str | None, names: dict[str, tuple[int, list[str]]]) -> str:
    """The directory an anchor of file `path` counts from: the file's, or the one `name` is bound to."""
    if name is None:
        return posixpath.dirname(path)
    levels, segs = names[name]
    return posixpath.normpath(posixpath.join(_ancestor(path, levels), *segs))


def _target(start: str, name: str | None, levels: int, segs: list[str]) -> str:
    """What an anchor names from `start`, `_start`'s: `levels` above the file (its directory is 1), or above `name`'s
    directory, then `segs`."""
    return posixpath.normpath(posixpath.join(_ancestor(start, levels - (name is None)), *segs))


def anchors(t: Tree, base: str, write: bool) -> dict[str, list[str]]:
    """Each anchor of a Python file, moved or not, that no longer names what it named at `base`, the commit before the moves
    (the mover recomputes the anchors it reads as joins; a name bound to one, `NAME.parent` or `NAME / "x"`, and an anchor
    of a file that stays naming a test that moved, it can't). With `write`, each is rewritten to name it again.
    {file: [what changed]}."""
    rows = _rows(t.root)
    out = {}
    for old in git("ls-tree", "-r", "--name-only", base, root=t.root).split("\n"):
        new = _mapped(old, rows)
        if not new.endswith(".py") or not (t.root / new).is_file() or "__file__" not in (after := (t.root / new).read_text()):
            continue
        before = git("show", f"{base}:{old}", root=t.root)
        if before.count("\n") != after.count("\n"):
            out[new] = [f"{new}: its lines differ from {old}'s, so its anchors are not checked"]
            continue
        line_of = lambda text, at: text.count("\n", 0, at)  # noqa: E731
        pairs = defaultdict(lambda: ([], []))
        for k, text in ((0, before), (1, after)):
            for x in _anchors(text):
                pairs[line_of(text, x[0])][k].append(x)
        names_old, names_new = _names(before), _names(after)
        # a use of a name counts from what the name will name once its own anchor is rewritten: what it named at `base`
        meant = {n: _mapped(_start(old, n, names_old), rows) for n in names_new if n in names_old}
        edits = []
        for line, (ax, ay) in sorted(pairs.items()):
            if len(ax) != len(ay):
                continue
            for (_, _, n0, l0, g0), (s1, e1, n1, l1, g1) in zip(ax, ay):
                if n0 != n1:
                    continue
                start = meant.get(n1) or _start(new, n1, names_new) if n1 else posixpath.dirname(new)
                want = _mapped(_target(_start(old, n0, names_old), n0, l0, g0), rows)
                have = _target(start, n1, l1, g1)
                if want == have:
                    continue
                common = posixpath.commonpath([start, want]) if start and want and not want.startswith("..") else ""
                up = (start.count("/") + 1 if start else 0) - (common.count("/") + 1 if common else 0)
                rel = posixpath.relpath(want or ".", common or ".")
                style = "" if n1 else re.match(r"\(?((?:pathlib\.)?Path\(__file__\)(?:\.resolve\(\)|\.absolute\(\))?)", after[s1:]).group(1)
                expr = _expression(n1, up + (0 if n1 else 1), [] if rel == "." else rel.split("/"), style)
                if " / " in expr and after[e1:e1 + 1] in (".", "["):
                    expr = f"({expr})"
                edits.append((s1, e1, expr, f"{new}:{line + 1}: {after[s1:e1].strip()} -> {expr} ({want})"))
        if edits:
            out[new] = [e[3] for e in edits]
            for s1, e1, expr, _ in sorted(edits, reverse=True):
                after = after[:s1] + expr + after[e1:]
            if write:
                (t.root / new).write_text(after)
    return out


def modules(path: str, root: Path = ROOT) -> str:
    """The `[modules]` table `layout.py --modules` derives for the map at `path`, as TOML."""
    r = subprocess.run([sys.executable, str(HERE / "layout.py"), "--root", str(root), "--map", path, "--modules"],
                       cwd=root, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"tests_root: layout.py --modules on {path}: {r.stderr.strip()[-500:]}")
    return r.stdout


def write_map(t: Tree, path: str, text: str) -> str:
    """The map at `path` with the `[modules]` its rows imply and the `[split]` rows its dry run finds."""
    (t.root / path).parent.mkdir(parents=True, exist_ok=True)
    (t.root / path).write_text(text)
    if (derived := modules(path, t.root)).strip():
        body = derived if derived.lstrip().startswith("[modules]") else "[modules]\n" + derived
        text = text.rstrip("\n") + "\n\n\n# the rename [paths] implies (`layout.py --modules`)\n" + body
        (t.root / path).write_text(text)
    if rows := split_rows(path, t.root):
        (t.root / path).write_text(text.rstrip("\n") + "\n\n[split]\n" + rows)
    return path


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    inv = sub.add_parser("inventory", help="each tests/ directory, its suite, and the layout's snags")
    inv.add_argument("--json", type=Path, help="write the whole inventory here")
    sp = sub.add_parser("split", help="each distribution's suite to a virtual member in its tests/")
    sp.add_argument("names", nargs="*", help="only these distributions")
    sub.add_parser("maps", help=f"write {FILES_MAP}")
    sub.add_parser("suites-map", help=f"write {SUITES_MAP}, once {FILES_MAP} has moved the tests")
    an = sub.add_parser("anchors", help="each moved file's anchor that no longer names what it named at BASE")
    an.add_argument("base", help="the commit before the moves")
    an.add_argument("--write", action="store_true", help="rewrite each to name it again")
    a = ap.parse_args(argv)
    t = Tree(root)
    if a.cmd == "inventory":
        r = inventory(t)
        if a.json:
            a.json.write_text(json.dumps(r, indent=1) + "\n")
        for area, c in r["areas"].items():
            print(f"{area}: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
        for snag, items in sorted(r["snags"].items()):
            print(f"{snag}: {len(items)}")
            for i in items:
                print(f"  {i}")
        return 0
    if a.cmd == "split":
        done = run_split(t, a.names)
        print(f"tests_root: split {len(done)} distribution(s)' suites: {', '.join(done) or 'none'}")
        return 0
    if a.cmd == "anchors":
        found = anchors(t, a.base, a.write)
        for f, items in sorted(found.items()):
            print("\n".join(items))
        n = sum(" -> " in x for items in found.values() for x in items)
        print(f"tests_root: {n} anchor(s){' rewritten' if a.write else ''}, in {len(found)} file(s)")
        return 0
    if a.cmd == "maps":
        print(f"tests_root: wrote {write_map(t, FILES_MAP, files_map(t))}")
    else:
        print(f"tests_root: wrote {write_map(t, SUITES_MAP, suites_map(t))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
