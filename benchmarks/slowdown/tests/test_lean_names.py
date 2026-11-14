"""Every string of the calculator that is wholly a Lean name, as a literal or as a module's value, is one a lock lists.

A citation is a string of the calculator's Python outside its tests that reads as a Lean declaration (dotted, each part
but the last capitalized, under a namespace some lock's names are under), as a literal or as a module's value once
built (a name its module composes, `min_cut.HE_CHARGED`). A name built inside a function, or one inside prose, is not
seen. A lock lists a name `slowdown.lean.read` takes from the locks it reads (a guarantee's statement, an assumption, an
open premise, a domain hypothesis, or a definition an assumptions module holds) or a guarantee those locks key, as
`nci/arguments.py` reads them. A namespace (`min_cut.AREA`) is no citation. A name no lock lists is in `ALLOWED`,
which only shrinks, with why."""

import ast
import inspect
import json
import pkgutil
import re
from importlib import import_module
from pathlib import Path

import slowdown
from slowdown import lean, min_cut

CALCULATOR = Path(rho.__file__).resolve().parents[1]
#: a dotted name, each namespace starting with a capital, as a Lean declaration reads
NAME = re.compile(r"(?:[A-Z][A-Za-z0-9_']*\.)+[A-Za-z_][A-Za-z0-9_']*")
#: {a cited Lean name no lock lists: why}
ALLOWED = {
    min_cut.HE_CHARGED: "stated beside HE and frozen unstated in the lock's shape: no theorem proves it until the count "
                        "over a set of blocks it follows from is proved",
}


def listed(locks: tuple[Path, ...] = lean.LOCKS) -> set[str]:
    """Every name ``locks`` list: what `slowdown.lean.read` takes from them, and their guarantees' keys."""
    proved, assumed = lean.read(locks)
    return set(proved | assumed).union(*(json.loads(p.read_text())["guarantees"] for p in locks))


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


def test_a_name_no_lock_lists_is_found(tmp_path):
    """The check isn't vacuous. Against a toy lock, a literal and a composed name no lock lists are found, and so are
    a name whose last part is lower case and a definition a module other than an assumptions module reads. A
    guarantee's statement and key, a premise, an assumptions module's definition, a namespace and a dotted string
    outside the locks' namespaces are not."""
    lock = tmp_path / "lean-audit.json"
    lock.write_text(json.dumps({
        "guarantees": {"Area.Proofs.stated_lower": {"signature": ["Area.Proofs.stated_lower : Area.Guarantees.Stated"]}},
        "assumptions": {"Area.Assumptions.Named": {}},
        "reads": {"Area.Assumptions": {"definitions": {"Area.Assumptions.Held": {}}},
                  "Area.Part": {"definitions": {"Area.Part.Read": {}}}},
    }))
    names = listed((lock,))
    src = {"rho/x.py": "A = 'Area.Guarantees.Nowhere'\nB = 'Area.Guarantees.Stated'\nC = 'Area.Guarantees'\n"
                       "D = 'Other.Name'\nF = 'Area.Part.Read'\nG = 'Area.Guarantees.noSuch'\n"
                       "H = 'Area.Proofs.stated_lower'\nI = 'Area.Assumptions.Named'\nJ = 'Area.Assumptions.Held'\n"}
    got = cited(src, {"rho.x.E": ("Area.Guarantees.Composed", "draft:x")}, names)
    assert set(got) == {"Area.Guarantees.Nowhere", "Area.Guarantees.Stated", "Area.Part.Read", "Area.Guarantees.noSuch",
                        "Area.Proofs.stated_lower", "Area.Assumptions.Named", "Area.Assumptions.Held",
                        "Area.Guarantees.Composed"}
    assert got["Area.Guarantees.Composed"] == "rho.x.E" and got["Area.Part.Read"] == "rho/x.py:5"
    assert {n for n in got if n not in names} == {"Area.Guarantees.Nowhere", "Area.Part.Read", "Area.Guarantees.noSuch",
                                                  "Area.Guarantees.Composed"}
