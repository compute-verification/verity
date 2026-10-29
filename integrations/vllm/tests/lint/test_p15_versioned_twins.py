"""P15: one implementation of each thing, so no versioned twins.

A twin is a name carrying a version token (``_v2``, ``_fast``, ``_new``, ``_old`` or ``_impl`` as a whole
``_``-separated token) beside a sibling with the same base name: ``global_match_fast.py`` beside ``global_match.py``,
``_acquire_fast`` and ``_acquire_impl`` beside ``_acquire``, ``variance_v1`` beside ``variance_v2``.  The older one is
dead weight or a second path the checks don't hold to the first.  Checked today, against
``allowlists/p15_versioned_twins.json``:

* ``module-twin``: a module or package anywhere in ``verity_vllm`` whose base name another module or package has
  (``check/match/global_match_fast.py`` beside ``check/match/global_match.py``).
* ``def-twin`` / ``class-twin``: a function or class whose base name another function or class in the same scope has.

A tokened name with no sibling is P11's (``module-name``, ``def-name``), not a twin.
"""
from __future__ import annotations

import ast
import re
from collections import defaultdict
from functools import lru_cache

from . import _ratchet as R

NAME = "p15_versioned_twins"
RULE = "P15 (one implementation, no versioned twins)"
TOKEN = re.compile(r"_(?:v\d+|fast|new|old|impl)(?=_|$)")


def base(name: str) -> str:
    return TOKEN.sub("", name).lstrip("_")


def twins(names: list[str]) -> list[tuple[str, str]]:
    """(name, base) for each tokened name whose base another of ``names`` shares."""
    groups: dict[str, set[str]] = defaultdict(set)
    for n in names:
        groups[base(n)].add(n)
    return sorted((n, b) for b, ns in groups.items() if len(ns) > 1 for n in ns if TOKEN.search(n))


def file_violations(src: str, rel: str) -> list[R.Violation]:
    tree = R.parse_src(src)
    defs: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for node, symbol in R.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs[("def-twin", symbol)].setdefault(node.name, node.lineno)
        elif isinstance(node, ast.ClassDef):
            defs[("class-twin", symbol)].setdefault(node.name, node.lineno)
    return [R.Violation(rel, kind, symbol, f"{n} ~ {b}", names[n])
            for (kind, symbol), names in sorted(defs.items()) for n, b in twins(list(names))]


def module_violations(files: tuple[str, ...]) -> list[R.Violation]:
    """``files``: paths relative to ``integrations/vllm``; a package is named by its directory."""
    paths: dict[str, list[str]] = defaultdict(list)
    for f in files:
        parts = f.removesuffix(".py").split("/")
        package = parts[-1] == "__init__"
        if package:
            parts.pop()
        if len(parts) > 1:
            paths[parts[-1]].append("/".join(parts) + ("" if package else ".py"))
    return [R.Violation(p, "module-twin", "<module>", f"{n} ~ {b}") for n, b in twins(list(paths)) for p in paths[n]]


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    files = R.library_files()
    out = module_violations(tuple(R.rel(p) for p in files))
    for p in files:
        out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Keep one implementation: delete the twin the served path doesn't use, or fold it into the "
                    "survivor, and give the survivor the plain name. An alternative being measured goes behind a "
                    "parameter of the one function, or under benchmarks/.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    assert [base(n) for n in ("_acquire_fast", "_acquire_impl", "_fast_read", "variance_v2", "_serve_v3_binder",
                              "build_new", "load_old")] == \
        ["acquire", "acquire", "read", "variance", "serve_binder", "build", "load"]
    assert [n for n in ("fasttree", "newton", "impl_table", "v2x", "breakfast", "old_style") if TOKEN.search(n)] == []
    src = (
        "def read(x): return x\n"
        "def _fast_read(x): return x\n"
        "def variance_v1(z): return z\n"
        "def variance_v2(z): return z\n"
        "def build_v3(p): return p\n"
        "class Committer:\n"
        "    def _acquire(self): self._acquire_impl()\n"
        "    def _acquire_impl(self): pass\n"
        "    def read(self): pass\n"
        "class Committer_old: pass\n"
    )
    got = [(v.kind, v.symbol, v.detail) for v in file_violations(src, "verity_vllm/x.py")]
    assert got == [("class-twin", "<module>", "Committer_old ~ Committer"),
                   ("def-twin", "<module>", "_fast_read ~ read"),
                   ("def-twin", "<module>", "variance_v1 ~ variance"),
                   ("def-twin", "<module>", "variance_v2 ~ variance"),
                   ("def-twin", "Committer", "_acquire_impl ~ acquire")], got
    mods = ("verity_vllm/engine/capture.py", "verity_vllm/observe/capture_v1.py", "verity_vllm/check/match/__init__.py",
            "verity_vllm/check/match_fast/__init__.py", "verity_vllm/acquire/apply_v8.py", "verity_vllm/__init__.py")
    assert [(v.file, v.detail) for v in module_violations(mods)] == [
        ("verity_vllm/observe/capture_v1.py", "capture_v1 ~ capture"),
        ("verity_vllm/check/match_fast", "match_fast ~ match")]
