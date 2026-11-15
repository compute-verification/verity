#!/usr/bin/env python3
"""What each ratchet lint finds and what its list allows, by file, on a tree of one commit.

    python benchmarks/tree/lint_counts.py REF [--lists NAME,NAME,...]      # from inside the repository's checkout

A ratchet lint holds what it finds in the tree to a committed list: ``tests/lint/allowlists/NAME.json`` for the
repository's lints and ``adapters/vllm/tests/lint/allowlists/NAME.json`` for the vLLM adapter's.  A list bounds the
findings only where its lint fails on a finding past its entry.  Where the lint warns, the tree may hold more than the
list says, so a count of entries reads low.  This prints the lints' own count beside the lists'.

It checks ``REF`` out in a clone of its own, which borrows the repository's objects and is removed at the end, so it
writes nothing into the repository.  It calls each lint's own code there; nothing here scans a file.  It prints one line
for each file that a lint finds a site in or that a list has an entry for, sorted, tab-separated, to stdout:

    lint    file    found    allowed

``lint`` is the list's name, ``file`` the file's path from the repository's root, ``found`` the sites the lint finds in
the file and ``allowed`` the sites the list's entries allow it.  A list that could not be counted has one line,
``NAME  -  -  -``; its reason goes to stderr, where each counted list has a line of totals, and the exit is 1.  Exit 0:
every list asked for was counted.  Exit 2: no such commit, list or lint.

How a list is read:

* A repository lint (``tests/lint/test_NAME.py``): its ``found()`` and ``allowed()``.  The environment-reads lint's
  ``allowed()`` returns the list's registry marks too, which its ``found`` takes.  The ruff lint has no ``found()``:
  its findings are its ``scan()``'s, as its own ``--prune`` reads them.  A list whose module has no ``allowed()`` is
  not a ratchet's and is not counted: the Python verdicts' list is a reading's, and its lint finds nothing outside it.
* ``lean_retired_terms``, which ``tests/test_retired_terms.py`` holds: the Lean files' sites of its ``offenders`` and
  its ``lean_allowed()``, as its own ``--prune`` reads them.
* ``superseded_versions``, which circuit-check's suite holds (``tools/verity/circuit_check/tests/``): the files of
  the superseded versions its ``superseded(registered())`` finds, and of the versions its list's entries name and the
  superseded ones its ``held`` lines name, each one site.
* An adapter lint (the ``test_*.py`` whose ``NAME``, or ``LIST``, is the list's name): its ``scan()``'s violations
  (mypy's ``violations()``) and ``_ratchet.load_allowlist(NAME)``, which are what its tests compare.  Its keys name a
  file from ``adapters/vllm`` where the file is under it; a line here names every file from the repository's root.
* A lint with two readings is counted by the one its test of new findings reads (the typed-records lint).
* P10's ``count`` is a size in lines, not a number of sites.  Its line counts one for each module or function past the
  limit and one for each entry.

A line sums a file's keys.  Where a list is keyed finer than the file (by rule, or by kind, symbol and detail), a key
past its entry can stand beside one under its own in the same file; the totals on stderr count by key.

What a count needs of the Python that runs this:

* The ruff lists (``RUFF``): its ruff is the version the tree's root ``dev`` group pins, since another ruff finds other
  things.  Otherwise they are not counted.
* mypy's list: the environment the list records (Python, mypy, pydantic, torch and vLLM).  The lint itself refuses any
  other, and its refusal is the reason printed.  ``uv sync --all-packages --extra torch-cpu`` makes both.
* ``superseded_versions``: the installed ``circuit_check`` and ``verity``, whose registries it loads, so its count is
  of ``REF`` only where the environment was synced from a checkout of ``REF``.
* Every other list needs the standard library alone: pytest's marks get a stand-in where pytest is absent.

Each lint runs in an isolated Python (``-I``) without ``PYTHONPATH`` and ``MYPYPATH``, so a count is of the tree's files
and the installed packages, whatever launched this.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections import Counter
from pathlib import Path

ROOT_LINT = "tests/lint"
ADAPTER = "adapters/vllm"
ADAPTER_LINT = f"{ADAPTER}/tests/lint"
#: a repository list that no ``tests/lint/test_NAME.py`` holds: (how it is read, the module that holds it)
ELSEWHERE = {"lean_retired_terms": ("lean_terms", "tests/test_retired_terms.py"),
             "superseded_versions": ("superseded", "tools/verity/circuit_check/tests/test_superseded_versions.py")}
#: the lists whose lint runs ruff
RUFF = ("repo_ruff", "ruff", "ruff_vllm")
#: what would have a lint import, or mypy follow, another checkout's code
DROPPED = ("PYTHONPATH", "MYPYPATH")

#: Run as ``python -I -c CHILD TREE OUT JOBS`` (JOBS a JSON list of [list, how it is read, its lint module]): writes to OUT
#: a JSON list with, for each job, ``{"list", "found", "allowed", "keys"}`` (each file's count, as the lint names the
#: file, and the counts by key) or ``{"list", "refused"}``.
CHILD = r'''
import importlib, importlib.util, json, os, sys, traceback, types
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
tree, out, jobs = Path(sys.argv[1]), Path(sys.argv[2]), json.loads(sys.argv[3])

try:
    import pytest
except ImportError:
    class Mark:
        def __getattr__(self, name):
            def mark(*args, **kwargs):
                return args[0] if len(args) == 1 and callable(args[0]) and not kwargs else (lambda f: f)
            return mark

    pytest = types.ModuleType("pytest")
    pytest.mark = Mark()
    pytest.fixture = pytest.mark.fixture
    sys.modules["pytest"] = pytest


def root(name, module):
    path = tree / module
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    assert hasattr(mod, "allowed"), f"{module} has no allowed(): its list is not a ratchet's, which this counts"
    have, *marks = have if isinstance(have := mod.allowed(), tuple) else (have,)
    return (mod.found(*marks) if hasattr(mod, "found") else Counter(k for k, *_ in mod.scan())), have


def lean_terms(name, module):
    path = tree / module
    sys.path.insert(0, str(path.parent))                 # a root test imports its neighbours by name
    mod = importlib.import_module(path.stem)
    return Counter(f for f, *_ in mod.offenders(mod._tracked())[0] if f.endswith(".lean")), mod.lean_allowed()


def superseded(name, module):
    path = tree / module
    sys.path.insert(0, str(path.parent))
    mod = importlib.import_module(path.stem)
    os.chdir(out.parent)                                 # else its loader imports verity from the tree, whose files the
    try:                                                 # registries don't count as authored
        now = mod.superseded(mod.registered())
    finally:
        os.chdir(tree)
    entries, held = mod.listed()
    return Counter(f for f, _ in now.values()), Counter([*entries.values(), *(now[i][0] for i in held if i in now)])


def adapter(name, module):
    path = tree / module
    if "lints" not in sys.modules:                       # the lint directory as a package of its own
        sys.modules["lints"] = types.ModuleType("lints")
        sys.modules["lints"].__path__ = [str(path.parent)]
    have = importlib.import_module("lints._ratchet").load_allowlist(name)
    mod = importlib.import_module(f"lints.{path.stem}")
    if hasattr(mod, "sizes"):                            # P10: a count is a size in lines
        return Counter(dict.fromkeys(mod.sizes(), 1)), Counter(dict.fromkeys(have, 1))
    got = mod.violations() if hasattr(mod, "violations") else mod.scan()
    if len(got) == 2 and isinstance(got[1], dict):       # P13: the violations, and each one's other copies
        got = got[0]
    return Counter(v.key for v in got), have


def by_file(counts):
    files = Counter()
    for k, n in counts.items():
        files[k if isinstance(k, str) else k[0]] += n
    return dict(files)


results = []
for name, how, module in jobs:
    try:
        got, have = {"root": root, "lean_terms": lean_terms, "superseded": superseded, "adapter": adapter}[how](name, module)
        over, under = got - have, have - got
        results.append({"list": name, "found": by_file(got), "allowed": by_file(have),
                        "keys": [len(over), sum(over.values()), len(under), sum(under.values())]})
    except AssertionError as e:
        results.append({"list": name, "refused": str(e)})
    except (Exception, SystemExit):
        results.append({"list": name, "refused": traceback.format_exc(limit=3)})
out.write_text(json.dumps(results), encoding="utf-8")
'''


def stop(message: str) -> None:
    print(f"lint_counts: {message}", file=sys.stderr)
    sys.exit(2)


def git(*args: str, cwd: Path) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        stop(f"git {' '.join(args)}: {r.stderr.strip()[-300:]}")
    return r.stdout.strip()


def named(module: Path) -> str | None:
    """The list a lint module names (its ``NAME``, or ``LIST``), read without running it."""
    for node in ast.parse(module.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("NAME", "LIST") for t in node.targets) \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


def lists(tree: Path) -> dict[str, tuple[str, str] | None]:
    """{each list's name: (how it is read, its lint module), or None where no lint holds it}, in both lint directories."""
    out: dict[str, tuple[str, str] | None] = {}
    lints = {named(f): f"{ADAPTER_LINT}/{f.name}" for f in sorted((tree / ADAPTER_LINT).glob("test_*.py"))}
    for directory in (ROOT_LINT, ADAPTER_LINT):
        for path in sorted((tree / directory / "allowlists").glob("*.json")):
            name = path.stem
            if name in out:
                stop(f"two lists are named {name}: {ROOT_LINT}/allowlists and {ADAPTER_LINT}/allowlists")
            how, module = ("adapter", lints.get(name)) if directory == ADAPTER_LINT else \
                ELSEWHERE.get(name, ("root", f"{ROOT_LINT}/test_{name}.py"))
            out[name] = (how, module) if module and (tree / module).is_file() else None
    return out


def ruff_refusal(tree: Path, env: dict[str, str]) -> str | None:
    """Why this Python can't count the ruff lists, or None: its ruff is the one the tree's root ``dev`` group pins."""
    try:
        doc = tomllib.loads((tree / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        return f"the tree's root pyproject.toml, which pins ruff, can't be read: {e}"
    dev = doc.get("dependency-groups", {}).get("dev", [])
    pin = next((d for d in dev if isinstance(d, str) and re.split(r"[\s=<>!~;\[]", d, maxsplit=1)[0] == "ruff"), None)
    r = subprocess.run([sys.executable, "-m", "ruff", "--version"], capture_output=True, text=True, env=env)
    have = f"ruff=={r.stdout.split()[-1]}" if r.returncode == 0 and r.stdout.split() else "no ruff"
    if pin is None or not pin.startswith("ruff=="):
        return f"the tree's root `dev` group pins no one ruff ({pin}), so no ruff is known to find what its lists hold"
    if have != pin:
        return (f"the tree pins {pin} and {sys.executable} has {have}: run this with the tree's environment "
                f"(`uv sync --all-packages`, then `uv run python benchmarks/tree/lint_counts.py REF`)")
    return None


def checkout(repo: Path, commit: str, tree: Path) -> None:
    """``commit`` checked out at ``tree``, in a clone that borrows the objects of ``repo``, a checkout of any kind (a clone, a
    worktree), and so copies none: git answers there as in a checkout, and nothing is written into ``repo``."""
    common = (repo / git("rev-parse", "--git-common-dir", cwd=repo)).resolve()
    git("clone", "--quiet", "--shared", "--no-checkout", str(common), str(tree), cwd=repo)
    git("-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", commit, cwd=tree)


def from_root(tree: Path, file: str) -> str:
    """An adapter list's file, from the repository's root.  Its lints name a file from ``adapters/vllm`` where it is under
    it and from the root elsewhere; a file that is gone was the adapter's where its first directory is the adapter's."""
    if (tree / ADAPTER / file).exists() or not (tree / file).exists() and (tree / ADAPTER / file.split("/")[0]).exists():
        return f"{ADAPTER}/{file}"
    return file


def count(tree: Path, jobs: list[tuple[str, str, str]], env: dict[str, str]) -> list[dict]:
    """The child's result for each of ``jobs``, in a tree where it wrote nothing."""
    with tempfile.TemporaryDirectory(prefix="lint-counts-out-") as d:
        out = Path(d) / "counts.json"
        r = subprocess.run([sys.executable, "-I", "-c", CHILD, str(tree), str(out), json.dumps(jobs)], cwd=tree, env=env,
                           capture_output=True, text=True)
        if r.returncode or not out.is_file():
            why = f"the lints' process ended {r.returncode}: {r.stderr.strip()[-600:]}"
            return [{"list": name, "refused": why} for name, *_ in jobs]
        return json.loads(out.read_text(encoding="utf-8"))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="lint_counts.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", metavar="REF", help="the commit whose tree is counted")
    ap.add_argument("--lists", default="", help="the lists to count, by name, comma-separated (default: every list)")
    args = ap.parse_args(argv)
    repo = Path(git("rev-parse", "--show-toplevel", cwd=Path.cwd()))
    commit = git("rev-parse", "--verify", f"{args.ref}^{{commit}}", cwd=repo)
    env = {k: v for k, v in os.environ.items() if k not in DROPPED}
    scratch = Path(tempfile.mkdtemp(prefix="lint-counts-"))
    tree = scratch / "tree"
    try:
        checkout(repo, commit, tree)
        there = lists(tree)
        if not there:
            stop(f"{args.ref} has no list under {ROOT_LINT}/allowlists or {ADAPTER_LINT}/allowlists")
        asked = [n for n in args.lists.split(",") if n] or sorted(there)
        if unknown := sorted(set(asked) - set(there)):
            stop(f"no list named {', '.join(unknown)} at {args.ref}; the lists there: {', '.join(sorted(there))}")
        read = {n: lint for n in asked if (lint := there[n])}
        refused = {n: f"no lint under {ROOT_LINT} or {ADAPTER_LINT} names it, and lint_counts.py's ELSEWHERE has no row for it"
                   for n in asked if n not in read}
        if set(asked) & set(RUFF) and (why := ruff_refusal(tree, env)):
            refused.update({n: why for n in asked if n in RUFF})
        jobs = [(n, *read[n]) for n in asked if n not in refused]
        results = {r["list"]: r for r in count(tree, jobs, env)} if jobs else {}
        rows: list[tuple[str, str, str, str]] = []
        total: Counter[str] = Counter()
        for name in asked:
            r = results.get(name, {})
            if why := refused.get(name) or r.get("refused"):
                rows.append((name, "-", "-", "-"))
                print(f"{name}: not counted: {why.strip()}", file=sys.stderr)
                continue
            found, allowed = (Counter({from_root(tree, f) if read[name][0] == "adapter" else f: n for f, n in r[side].items()})
                              for side in ("found", "allowed"))
            rows += [(name, f, str(found[f]), str(allowed[f])) for f in sorted(found.keys() | allowed.keys())]
            over_keys, over, under_keys, under = r["keys"]
            total.update(found=sum(found.values()), allowed=sum(allowed.values()), over=over, under=under, counted=1)
            print(f"{name}: {sum(found.values())} found in {len(found)} file(s), {sum(allowed.values())} allowed in "
                  f"{len(allowed)}; {over_keys} key(s) hold {over} more than their entries, {under_keys} entr(y/ies) allow "
                  f"{under} more than is found", file=sys.stderr)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    sys.stdout.write("".join("\t".join(row) + "\n" for row in sorted(rows)))
    missed = len(asked) - total["counted"]
    print(f"lint_counts: {commit[:12]}: {total['counted']} list(s) counted, {missed} not counted; {total['found']} found, "
          f"{total['allowed']} allowed; {total['over']} past their entries by key, {total['under']} allowed past what is found",
          file=sys.stderr)
    return 1 if missed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
