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
   unchanged.

**Archive wins.** A test file goes to `archive/` with the code it tests, and a suite that stays never loads `archive/`.
So a test file the maps archive is in no suite and parted from none (`tests/test_repository.py` exempts `archive/`
once the move's pull request adds it to `UNSUITED_TEST_DIRS`), provided the maps archive code it tests: a module it
imports, or one of its suite's whose name it holds. One archived alone fails, under 2. And under 3, a module that goes
to `archive/` while a test that stays still imports it fails, named once with the file that imports it.

**A test goes with the code it tests.** On the same condition, a test file the maps send into another suite's testpaths
with code it tests (a module that lands in that suite's directory from outside it) is parted from none. It is then that
suite's: under 3 what it imports must land in that suite's inputs.

Rows decide 1 and 2. 3 is decided by rows only where one new directory holds all of an input's files; elsewhere the
suite's inputs need an edit after the move, and the check prints the directories they would need.

With no map it checks REF's tree as it is, which passes all three. Imports are read from the source (`import`, `from`),
resolved under the first map's `[roots] old`; a module loaded by path or by name at run time is not seen, and one imported
where no test runs is. A file that doesn't parse is printed with the parser's reason, and its imports are not followed. So 3 is wider than the read guard, and each finding says whether the import runs when a test file is
imported (`at_import`: outside every function and `if TYPE_CHECKING:`), which the guard is sure to see. The last line
says how many imports are outside their suite's inputs on REF already, which 3 counts none of."""
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
ARCHIVE = "archive/"


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
        self.memo: dict[tuple[str, bool], set[str]] = {}
        #: {a file that doesn't parse: why}; its imports are not followed
        self.unparsed: dict[str, str] = {}

    def resolve(self, name: str) -> str | None:
        while name:
            if name in self.file:
                return self.file[name]
            name = name.rpartition(".")[0]
        return None

    def direct(self, f: str, at_import: bool = False) -> set[str]:
        """The modules `f` imports; `at_import`: only by a statement that runs when `f` is imported (outside every function
        and every `if TYPE_CHECKING:`)."""
        if (f, at_import) in self.memo:
            return self.memo[f, at_import]
        out: set[str] = set()
        self.memo[f, at_import] = out
        try:
            tree = ast.parse(self.m.read(f) or "")
        except (SyntaxError, ValueError) as e:
            self.unparsed[f] = f"{type(e).__name__}: {e}"
            return out
        me = L.module_name(f, self.roots) or ""
        pkg = (me if f.endswith("/__init__.py") else me.rpartition(".")[0]).split(".")
        for node in _runs_at_import(tree) if at_import else ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join([*pkg[: len(pkg) - node.level + 1], *([base] if base else [])])
                names = [f"{base}.{a.name}" for a in node.names]       # a name that is no module resolves to `base`
            else:
                continue
            out |= {r for n in names if (r := self.resolve(n)) and r != f}
        return out

    def reach(self, files: list[str], at_import: bool = False) -> dict[str, str]:
        """{module file: the file that imports it} for everything `files` import, directly or through an import."""
        seen: dict[str, str] = {}
        todo = [(f, g) for f in files for g in self.direct(f, at_import)]
        while todo:
            f, g = todo.pop()
            if g in seen:
                continue
            seen[g] = f
            todo += [(g, h) for h in self.direct(g, at_import)]
        return seen


def _runs_at_import(node: ast.AST):
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(child, ast.If) and "TYPE_CHECKING" in ast.dump(child.test):
            continue
        yield child
        yield from _runs_at_import(child)


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

    imports = Imports(m)
    archived = {f for f in m.files if m.new[f].startswith(ARCHIVE) and not f.startswith(ARCHIVE)}
    code = {f for f in archived if f.endswith(".py") and not is_test(f) and posixpath.basename(f) != "conftest.py"}

    def with_its_code(test: str, suite: str, code: set[str]) -> bool:
        """Whether `code`, what the maps send where this test file goes, holds code it tests: a module it imports, or one
        of its suite's whose name it holds."""
        if imports.direct(test) & code:
            return True
        text, base = m.read(test) or "", posixpath.dirname(suite) or "."
        return any(under(g, base) and re.search(rf"(?<![\w.]){re.escape(posixpath.basename(g)[:-3])}(?!\w)", text) for g in code)

    def sent_into(q: str) -> set[str]:
        """The code the maps send into suite `q`'s directory from outside it."""
        if q not in into:
            old, new = posixpath.dirname(q) or ".", posixpath.dirname(m.new[q]) or "."
            into[q] = {f for f in m.files if f.endswith(".py") and not is_test(f) and posixpath.basename(f) != "conftest.py"
                       and m.new[f] != f and under(m.new[f], new) and not under(f, old)}
        return into[q]

    orphans = []
    after = [t for p in suites for t in paths(p, m.new[p])]
    for f in tests:
        new = m.new[f]
        if (is_test(f) and not any(under(new, t) for t in after) and not new.startswith((ARCHIVE, *unsuited))
                and DEFINES_A_TEST.search(m.read(f) or "")):
            orphans.append({"test": f, "lands": new})

    parted, collects, with_code = [], defaultdict(list), []
    into: dict[str, set[str]] = {}
    sent, with_suite = defaultdict(list), []            # {the suite a test file goes into: [(the file, its suite today)]}
    for f in tests:
        owners = [p for p in suites if any(under(f, t) for t in paths(p, p))]
        if not owners:
            continue
        p = max(owners, key=lambda q: len(posixpath.dirname(q)))        # the innermost suite collects it
        want = posixpath.normpath(posixpath.join(posixpath.dirname(m.new[p]), posixpath.relpath(f, posixpath.dirname(p) or ".")))
        if f in archived and (p in archived or with_its_code(f, p, code)):
            with_code.append({"suite": p, "test": f, "lands": m.new[f]})
            continue
        if m.new[f] != want and f not in archived:
            others = [q for q in suites if q != p and q not in archived and any(under(m.new[f], t) for t in paths(q, m.new[q]))]
            q = max(others, key=lambda q: len(posixpath.dirname(m.new[q])), default=None)       # the innermost collects it
            if q is not None and with_its_code(f, p, sent_into(q)):
                sent[q].append((f, p))
                with_suite.append({"suite": p, "test": f, "lands": m.new[f], "into": q, "into_lands": m.new[q]})
                continue
        collects[p].append(f)
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

    lost, loads, reached, outside = {}, {}, set(), {}
    for p in sorted(suites) if m.layouts else ():
        if p in archived:
            continue
        own = inputs_of(p)
        now = [(posixpath.dirname(m.new[p]) or "." if why == "its directory" else reads(path)) for why, path in own]
        # a test file sent into this suite: what its suite's inputs cover today must land in this suite's
        for files, inputs in [(collects[p], own), *(([f], inputs_of(q)) for f, q in sent[p])]:
            opened = imports.reach(files, at_import=True)       # what importing the test files imports
            for f, importer in sorted(imports.reach(files).items()):
                if f in archived:
                    if importer not in archived:            # an archived module's own imports are not a suite's
                        x = loads.setdefault(f, {"module": f, "lands": m.new[f], "imported_by": importer, "suites": []})
                        x["suites"] += [p] if p not in x["suites"] else []
                    continue
                why, path = next(((w, path) for w, path in inputs if under(f, path)), (None, None))
                reached.add((p, f))
                if why is None:                             # outside the inputs today already: not the maps' doing
                    outside.setdefault((p, f), {"suite": p, "module": f, "imported_by": importer, "at_import": f in opened})
                if why is None or any(under(m.new[f], n) for n in now):
                    continue
                rest = f[len(path):] if path != "." else "/" + f
                lost.setdefault((p, f), {
                    "suite": p, "suite_lands": m.new[p], "module": f, "lands": m.new[f], "imported_by": importer, "covered_today_by": why,
                    "at_import": f in opened,
                    "holds_it_after": m.new[f][: -len(rest)] if m.new[f].endswith(rest) and len(rest) < len(m.new[f])
                    else posixpath.dirname(m.new[f])})
    return {"orphans": orphans, "parted": parted, "inputs": list(lost.values()),
            "loads_archive": sorted(loads.values(), key=lambda x: x["module"]), "archived_with_code": with_code,
            "sent_with_code": with_suite, "reached": len(reached), "outside_today": list(outside.values()),
            "unparsed": dict(sorted(imports.unparsed.items()))}


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
          f"{len(suites_hit)} suites, {len(r['loads_archive'])} archived modules a test that stays imports "
          f"({len(r['archived_with_code'])} test files go to archive/ with their code, {len(r['sent_with_code'])} into "
          f"another suite with theirs)")
    at_import = [x for x in r["inputs"] if x["at_import"]]
    print(f"  of the {len(r['inputs'])}, {len(at_import)} in {len({x['suite'] for x in at_import})} suites are imported when a test "
          f"file is: the read guard refuses those; the rest are imported inside a function, which a test may never reach")

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
    for x in r["loads_archive"]:
        print(f"  goes to archive/ and is still imported: {x['module']} (by {x['imported_by']}; the tests of {len(x['suites'])} "
              f"suite{'s' if len(x['suites']) != 1 else ''} reach it)")
    needs = defaultdict(set)
    for x in r["inputs"]:
        needs[x["suite_lands"]].add(x["holds_it_after"])
    for suite, dirs in sorted(needs.items()):
        print(f"  {suite}: its inputs would also need {', '.join(sorted(dirs))}")
    for f, why in r["unparsed"].items():
        print(f"  doesn't parse, so its imports are not followed: {f} ({why})")
    if r["reached"]:
        today = r["outside_today"]
        print(f"  not counted: {len(today)} of the {r['reached']} imports (a suite and a module its tests reach) are outside the "
              f"suite's inputs on {m.ref[:12]} as it is, {sum(x['at_import'] for x in today)} of them imported when a test file is")
    return 1 if r["orphans"] or r["parted"] or r["inputs"] or r["loads_archive"] else 0


if __name__ == "__main__":
    sys.exit(main())
