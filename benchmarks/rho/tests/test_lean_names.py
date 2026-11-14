"""Every Lean name the calculator cites is one a lock lists.

A citation is a string of the calculator's Python outside its tests that reads as a Lean declaration (dotted, each part
capitalized, under a namespace some lock's names are under), as a literal or as a module's value once built (a name its
module composes, `min_cut.HE_CHARGED`). A lock lists a name where `nci/arguments.py` reads one from a lock (a
guarantee, an assumption, an open premise or a domain hypothesis) or `rho.lean` does (a guarantee's statement, a
definition an assumptions module holds), in either lock `rho.lean` reads. A namespace (`min_cut.AREA`) is no
citation. A name no lock lists is in `ALLOWED`, which only shrinks, with why."""

import ast
import inspect
import json
import pkgutil
import re
from importlib import import_module
from pathlib import Path

import rho
from rho import lean, min_cut

CALCULATOR = Path(rho.__file__).resolve().parents[1]
#: a dotted name, each part starting with a capital, as a Lean declaration reads
NAME = re.compile(r"(?:[A-Z][A-Za-z0-9_']*\.)+[A-Z][A-Za-z0-9_']*")
#: {a cited Lean name no lock lists: why}
ALLOWED = {
    min_cut.HE_CHARGED: "stated beside HE and frozen unstated in the lock's shape: no theorem proves it until the count "
                        "over a set of blocks it follows from is proved",
}


def listed() -> set[str]:
    """Every name the locks `rho.lean` reads list."""
    out = set()
    for path in lean.LOCKS:
        lock = json.loads(path.read_text())
        out |= set(lock["guarantees"]) | set(lock["assumptions"]) | set(lock.get("open") or {}) | set(
            lock.get("domain") or {})
        out |= {s.rsplit(" : ", 1)[-1] for e in lock["guarantees"].values() for s in e.get("signature", ())}
        out |= {d for m, g in (lock.get("reads") or {}).items() for d in (g.get("definitions") or {})}
    return out


def _strings(v, depth: int = 0):
    if isinstance(v, str):
        yield v
    elif isinstance(v, (tuple, list, set, frozenset)) and depth < 3:
        for x in v:
            yield from _strings(x, depth + 1)
    elif isinstance(v, dict) and depth < 3:
        for x in v.values():
            yield from _strings(x, depth + 1)


def cited(sources: dict[str, str], values: dict[str, object], names: set[str]) -> dict[str, str]:
    """{each name ``sources`` (path -> text) and ``values`` (where -> a module's value) cite: where}, under the
    namespaces of ``names`` and none of them a namespace of one."""
    roots = {n.split(".")[0] for n in names}
    found: dict[str, str] = {}
    for rel, text in sources.items():
        for n in ast.walk(ast.parse(text, filename=rel)):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and NAME.fullmatch(n.value):
                found.setdefault(n.value, f"{rel}:{n.lineno}")
    for where, v in values.items():
        for s in _strings(v):
            if NAME.fullmatch(s):
                found.setdefault(s, where)
    prefixes = {n.rsplit(".", i)[0] for n in names for i in range(1, n.count(".") + 1)}
    return {n: w for n, w in found.items() if n.split(".")[0] in roots and n not in prefixes}


def calculator_cites() -> dict[str, str]:
    sources = {p.relative_to(CALCULATOR).as_posix(): p.read_text() for p in sorted(CALCULATOR.rglob("*.py"))
               if not {"tests", ".venv", "__pycache__"} & set(p.relative_to(CALCULATOR).parts)}
    values = {}
    for m in pkgutil.iter_modules(rho.__path__):
        mod = import_module(f"rho.{m.name}")
        for k, v in vars(mod).items():
            if inspect.isclass(v) and v.__module__ == mod.__name__:
                values |= {f"rho.{m.name}.{k}.{a}": x for a, x in vars(v).items() if a.isupper()}
            elif k.isupper() or isinstance(v, str):
                values[f"rho.{m.name}.{k}"] = v
    return cited(sources, values, listed())


def test_every_lean_name_the_calculator_cites_is_in_a_lock():
    names = listed()
    missing = {n: w for n, w in calculator_cites().items() if n not in names and n not in ALLOWED}
    assert not missing, ("the calculator cites Lean names no lock lists: name the statement a lock lists, or "
                         "add the name to ALLOWED in tests/test_lean_names.py with why (it only shrinks):\n"
                         + "\n".join(f"{w}: {n}" for n, w in sorted(missing.items())))


def test_allowed_only_shrinks():
    names, cites = listed(), calculator_cites()
    stale = sorted(n for n in ALLOWED if n in names or n not in cites)
    assert not stale, f"ALLOWED names a lock now lists or the calculator no longer cites; remove them: {stale}"


def test_a_name_no_lock_lists_is_found():
    """The check isn't vacuous: a literal and a composed name no lock lists are found, a listed name, a namespace and
    a dotted string outside the locks' namespaces are not."""
    names = {"Area.Part.Guarantees.Stated", "Area.Assumptions.Named"}
    src = {"rho/x.py": "A = 'Area.Guarantees.Nowhere'\nB = 'Area.Part.Guarantees.Stated'\nC = 'Area.Part'\n"
                       "D = 'Other.Name'\n"}
    got = cited(src, {"rho.x.E": ("Area.Part.Guarantees.Composed", "draft:x")}, names)
    assert got == {"Area.Guarantees.Nowhere": "rho/x.py:1", "Area.Part.Guarantees.Stated": "rho/x.py:2",
                   "Area.Part.Guarantees.Composed": "rho.x.E"}
    assert {n for n in got if n not in names} == {"Area.Guarantees.Nowhere", "Area.Part.Guarantees.Composed"}
