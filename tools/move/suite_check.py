#!/usr/bin/env python3
"""`tools/move/suite_check.py MAP... [--ref REF] [--json FILE]`: no map splits a suite.

Reads the maps' rows (`layout.py` maps, applied in the order given) and REF's tree (default `origin/main`), moves
nothing, and exits 1, naming the paths, when after the maps:

1. **a test file is in no suite**: `tests/test_repository.py`'s `test_every_test_file_is_in_a_suite` on the moved tree.
   A suite is a directory whose `pyproject.toml` configures pytest, and its `testpaths` are read as written, relative to
   where the pyproject lands.
2. **a suite has parted from a test file it collects**: the pyproject and each test file under its testpaths today (the
   innermost suite's, `conftest.py` included) keep their places relative to each other. A test file that lands in
   another suite's testpaths passes 1 and fails here: splitting a suite is its owner's change after the moves.
3. **a suite's inputs no longer hold a file its tests import**: a module the suite's test files import, directly or
   through the modules they import, that its inputs cover today (`tools/verity/check/suites.py`'s `inputs_of`: its
   directory, its workspace dependencies' directories, the paths `[tool.verity.tests] inputs` declares) and that lands
   outside them. An input is read as `layout.py` rewrites it: a file's row, a directory's row or `[tokens]` row, else
   unchanged. A module that goes to `archive/` is counted apart and does not fail: the move retires it.

With no map it checks REF's tree as it is, which passes all three. Imports are read from the source (`import`, `from`),
resolved under the first map's `[roots] old`; a module loaded by path or by name at run time is not seen."""
from __future__ import annotations

import argparse
import ast
import json
import posixpath
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout as L
from moved import Moved, under

#: what pytest collects from a module (`tests/test_repository.py`'s `_DEFINES_A_TEST`)
DEFINES_A_TEST = re.compile(r"^(?:async\s+)?def\s+test|^class\s+Test", re.M)
#: test files that were never part of `check` (`tests/test_repository.py`'s `UNSUITED_TEST_DIRS`), as REF names them
UNSUITED = re.compile(r"^UNSUITED_TEST_DIRS\s*=\s*\(([^)]*)\)", re.M)
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def is_test(p: str) -> bool:
    b = posixpath.basename(p)
    return p.endswith(".py") and (b.startswith("test_") or b.endswith("_test.py"))


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class Imports:
    """The tracked modules a file imports, by the source."""

    def __init__(self, m: Moved):
        self.m = m
        self.roots = m.layouts[0].old_roots if m.layouts else {}
        self.file = {}
        for p in m.files:
            if p.endswith(".py") and (name := L.module_name(p, self.roots)):
                self.file.setdefault(name, p)
        self.memo: dict[str, set[str]] = {}

    def resolve(self, name: str) -> str | None:
        while name:
            if name in self.file:
                return self.file[name]
            name = name.rpartition(".")[0]
        return None

    def direct(self, f: str) -> set[str]:
        if f in self.memo:
            return self.memo[f]
        out: set[str] = set()
        self.memo[f] = out
        try:
            tree = ast.parse(self.m.read(f) or "")
        except (SyntaxError, ValueError):
            return out
        me = L.module_name(f, self.roots) or ""
        pkg = (me if f.endswith("/__init__.py") else me.rpartition(".")[0]).split(".")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join([*pkg[: len(pkg) - node.level + 1], *([base] if base else [])])
                names = [f"{base}.{a.name}" for a in node.names] + [base]
            else:
                continue
            out |= {r for n in names if (r := self.resolve(n)) and r != f}
        return out

    def reach(self, files: list[str]) -> dict[str, str]:
        """{module file: the file that imports it} for everything `files` import, directly or through an import."""
        seen: dict[str, str] = {}
        todo = [(f, g) for f in files for g in self.direct(f)]
        while todo:
            f, g = todo.pop()
            if g in seen:
                continue
            seen[g] = f
            todo += [(g, h) for h in self.direct(g)]
        return seen


def check(m: Moved) -> dict:
    pyprojects = {p: tomllib.loads(m.read(p) or "") for p in m.files if posixpath.basename(p) == "pyproject.toml"}
    suites = {p: ini.get("testpaths", ["."]) for p, d in pyprojects.items()
              if (ini := d.get("tool", {}).get("pytest", {}).get("ini_options")) is not None}
    pkgs = {norm(d["project"]["name"]): p for p, d in pyprojects.items() if d.get("project", {}).get("name")}

    def paths(pyproject: str, where: str) -> list[str]:
        return [posixpath.normpath(posixpath.join(posixpath.dirname(where), t)) for t in suites[pyproject]]

    tests = [p for p in m.files if is_test(p) or posixpath.basename(p) == "conftest.py"]
    unsuited = [d.strip(" \"'") for d in (UNSUITED.search(m.read("tests/test_repository.py") or "") or [None, ""])[1].split(",")
                if d.strip(" \"'")]
    unsuited = [(m.directory(d.rstrip("/")) or d.rstrip("/")) + "/" for d in unsuited]

    orphans = []
    after = [t for p in suites for t in paths(p, m.new[p])]
    for f in tests:
        new = m.new[f]
        if (is_test(f) and not any(under(new, t) for t in after) and not new.startswith(tuple(unsuited))
                and DEFINES_A_TEST.search(m.read(f) or "")):
            orphans.append({"test": f, "lands": new})

    parted, collects = [], defaultdict(list)
    for f in tests:
        owners = [p for p in suites if any(under(f, t) for t in paths(p, p))]
        if not owners:
            continue
        p = max(owners, key=lambda q: len(posixpath.dirname(q)))        # the innermost suite collects it
        collects[p].append(f)
        want = posixpath.normpath(posixpath.join(posixpath.dirname(m.new[p]), posixpath.relpath(f, posixpath.dirname(p) or ".")))
        if m.new[f] != want:
            parted.append({"suite": p, "suite_lands": m.new[p], "test": f, "lands": m.new[f], "with_its_suite": want})

    def requires(p: str, groups: bool) -> set[str]:
        d = pyprojects[p]
        reqs = list(d.get("project", {}).get("dependencies", []))
        for extra in d.get("project", {}).get("optional-dependencies", {}).values():
            reqs += extra
        for group in d.get("dependency-groups", {}).values() if groups else ():
            reqs += [r for r in group if isinstance(r, str)]
        return {norm(x.group()) for r in reqs if (x := NAME.match(r))} & set(pkgs)

    def shipped(p: str) -> list[str]:
        d = pyprojects[p]
        if (exports := d.get("tool", {}).get("verity", {}).get("tests", {}).get("exports")) is not None:
            return list(exports)
        if posixpath.dirname(p):
            return [posixpath.dirname(p)]
        return list(d.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {}).get("packages", []))

    def inputs_of(p: str) -> list[tuple[str, str]]:
        """(what covers, the path it names today) for each of the suite's inputs."""
        t = pyprojects[p].get("tool", {}).get("verity", {}).get("tests", {})
        out = [("its directory", posixpath.dirname(p) or ".")]
        if t.get("inputs_only") is not True:
            seen, todo = set(), list(requires(p, True))
            while todo:
                n = todo.pop()
                if n not in seen:
                    seen.add(n)
                    todo += requires(pkgs[n], False)
            out += [(f"its dependency {n}", s) for n in sorted(seen) if pkgs[n] != p for s in shipped(pkgs[n])]
        for i in t.get("inputs", []):
            if isinstance(i, str):
                out += ([(f"its input {i}", s) for s in shipped(pkgs[norm(i)])] if norm(i) in pkgs and i not in m.old_dirs
                        else [(f"its input {i}", i.rstrip("/"))])
        return out

    def reads(path: str) -> str:
        if path == ".":
            return "."
        if path in m.tracked:
            return m.new[path]
        if path + "/pyproject.toml" in m.tracked and m.directory(path) is None:     # a package: where its pyproject lands
            return posixpath.dirname(m.new[path + "/pyproject.toml"]) or "."
        return m.directory(path) or path

    imports, lost, archived = Imports(m), [], 0
    for p in sorted(suites) if m.layouts else ():
        inputs = inputs_of(p)
        now = [(posixpath.dirname(m.new[p]) or "." if why == "its directory" else reads(path)) for why, path in inputs]
        for f, importer in sorted(imports.reach(collects[p]).items()):
            why, path = next(((w, path) for w, path in inputs if under(f, path)), (None, None))
            if why is None or any(under(m.new[f], n) for n in now):
                continue
            if m.new[f].startswith("archive/"):
                archived += 1
                continue
            rest = f[len(path):] if path != "." else "/" + f
            lost.append({"suite": p, "suite_lands": m.new[p], "module": f, "lands": m.new[f], "imported_by": importer,
                         "covered_today_by": why,
                         "holds_it_after": m.new[f][: -len(rest)] if m.new[f].endswith(rest) and len(rest) < len(m.new[f])
                         else posixpath.dirname(m.new[f])})
    return {"orphans": orphans, "parted": parted, "inputs": lost, "archived_imports": archived}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("maps", nargs="*", help="layout.py maps, in the order they run")
    ap.add_argument("--ref", default="origin/main", help="the commit whose tree the maps move (default: origin/main)")
    ap.add_argument("--json", type=Path, metavar="FILE", help="write every finding here")
    ap.add_argument("--all", action="store_true", help="print every path (default: five a group; --json has them all)")
    a = ap.parse_args(argv)
    m = Moved(a.maps, a.ref)
    try:
        r = check(m)
    finally:
        m.close()
    if a.json:
        a.json.write_text(json.dumps({"ref": m.ref, "maps": a.maps, **r}, indent=1) + "\n")
    suites_hit = {x["suite"] for x in r["inputs"]}
    print(f"suite check of {len(a.maps)} map(s) on {m.ref[:12]}: {len(r['orphans'])} test files in no suite, "
          f"{len(r['parted'])} parted from their suite, {len(r['inputs'])} imported modules outside the inputs of "
          f"{len(suites_hit)} suites ({r['archived_imports']} more imports go to archive/)")

    def show(title: str, groups: dict[str, list[str]]) -> None:
        for head, lines in sorted(groups.items()):
            print(f"  {title} {head}: {len(lines)}")
            for line in lines if a.all else lines[:5]:
                print(f"    {line}")
            if not a.all and len(lines) > 5:
                print(f"    and {len(lines) - 5} more (--all or --json lists them)")

    by = defaultdict(list)
    for o in r["orphans"]:
        by[posixpath.dirname(o["lands"])].append(f"{o['lands']} (today {o['test']})")
    show("in no suite, under", by)
    by = defaultdict(list)
    for x in r["parted"]:
        by[f"{x['suite']} -> {x['suite_lands']}"].append(f"{x['test']} -> {x['lands']} (with its suite: {x['with_its_suite']})")
    show("parted from suite", by)
    by = defaultdict(list)
    for x in r["inputs"]:
        by[f"{x['suite']} -> {x['suite_lands']}"].append(f"{x['module']} -> {x['lands']} (imported by {x['imported_by']}; today "
                                                         f"under {x['covered_today_by']})")
    show("outside the inputs of", by)
    needs = defaultdict(set)
    for x in r["inputs"]:
        needs[x["suite_lands"]].add(x["holds_it_after"])
    for suite, dirs in sorted(needs.items()):
        print(f"  {suite}: its inputs would also need {', '.join(sorted(dirs))}")
    return 1 if r["orphans"] or r["parted"] or r["inputs"] else 0


if __name__ == "__main__":
    sys.exit(main())
