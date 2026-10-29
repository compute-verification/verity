"""Adversarial probes of `circuit_check.cache`: edits that change a target's verdict while its key stays the same.

Each test states the property a sound cache keeps (an edit that changes the verdict reruns the target), on the synthetic tree
of `test_cache`: a cold run, the edit, a run under a fresh cache (the verdict as it is now), and a run under the first cache.
A test fails where the cached run reuses the stale entry."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from test_cache import FILES, HARNESS, REPO, _edit, _entry

#: the real correspondence check (the reference, `kernels` and `self_check`), not `test_cache`'s stand-in
CHECK_CORR = """
def check_definition(fn, opts, *, memo, pins, as_call=None, **kw):
    from circuit_check import checks as CK
    from verity.ml.circuits.evaluation.reference import resolve

    F = CK.Findings()
    corr, _arrays, _ref = CK.correspondence(resolve(fn), opts, F)
    return {"failures": F.failures, "warnings": F.warnings, "realizations": corr["realizations"]}
"""

#: a kernel module of the kind `verity.ml.definitions.numerics.rows` is: imported with the registries (`load_registries`), before
#: any check, registering kernels for Definitions defined elsewhere
KERN = """
import numpy as np

from verity.ml.circuits.evaluation import register_kernel

register_kernel("SynAdd_v1", lambda a, b: ((a.astype(np.uint64) + b) & 0xFFFF, None))
"""


def _harness(registries: str, check: str) -> str:
    h = HARNESS.replace("import syn.extra, syn.idle, syn.kdef, syn.other, syn.top", f"import {registries}")
    h = h.replace("from syn import catalog, check\n", f"from syn import catalog\nfrom syn import {check} as check\n")
    return h.replace('print(json.dumps({"stats"',
                     'print(json.dumps({"targets": {t["target"]: {k: t.get(k) for k in ("failures", "value")} for t in rep["targets"]}, "stats"')


def _tree(tmp: Path, extra: dict[str, str]) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp)], check=True)
    for rel, text in {**FILES, **extra}.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text.lstrip("\n"))
    return tmp


def _run(root: Path, harness: str, cache: str = "cache") -> dict:
    """``cache`` other than the tree's own: a directory beside the tree, which no listing of the tree sees."""
    where = 'root / "cache"' if cache == "cache" else f'Path({str(root.parent / f"{root.name}-{cache}")!r})'
    h = harness.replace('cache=root / "cache"', f"cache={where}")
    r = subprocess.run([sys.executable, "-c", h, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert not out["stats"]["untraced"] and not out["stats"]["errors"] and not out["stats"]["changed"], out
    return out


def _codes(fails: list[dict]) -> list[str]:
    return sorted(f"{f['check']}/{f['code']}" for f in fails)


def test_a_kernel_registered_in_a_module_no_check_ran_reruns_its_target(tmp_path):
    """`kernels(t)` finds nothing for SynTop_v1, so its check runs none of `syn/kern.py` and its key holds no file of it.  A
    kernel registered there afterwards (as `rows.py`'s `_row(...)` or a new `ir_rows.FAMILIES` entry registers one) is
    `self_check`ed on a fresh run, and a wrong one fails `correspondence/kernel-mismatch`."""
    root = _tree(tmp_path, {"syn/kern.py": KERN, "syn/check2.py": CHECK_CORR})
    h = _harness("syn.extra, syn.idle, syn.kdef, syn.other, syn.top, syn.kern", "check2")
    cold = _run(root, h)
    assert cold["stats"]["stored"] == 2 and cold["targets"]["SynTop_v1"]["failures"] == []
    assert "syn/kern.py" not in _entry(root, "SynTop_v1")["reads"]["files"]
    (root / "syn/kern.py").write_text((root / "syn/kern.py").read_text()
                                      + 'register_kernel("SynTop_v1", lambda a, b: ((a.astype(np.uint64) - b) & 0xFFFF, None))\n')
    fresh = _run(root, h, cache="fresh")
    assert _codes(fresh["targets"]["SynTop_v1"]["failures"]) == ["correspondence/kernel-mismatch"]
    warm = _run(root, h)
    assert "SynTop_v1" not in warm["hits"], (
        f"SynTop_v1 reused its entry (failures {warm['targets']['SynTop_v1']['failures']}) for a correspondence/kernel-mismatch")


#: `targets.py` with two `_model_roots` entries of one label
DUP_TARGETS = """
def _model_roots():
    return [
        ("SynOther_v1", lambda: "SynOther_v1"),
        ("Dup_v1", lambda: "first"),
        ("Dup_v1", lambda: "second"),
    ]
"""


def test_an_edit_to_either_of_two_same_label_entries_changes_a_unit():
    """`_units_of` keys each `_model_roots` entry by its label, so of two entries with one label only the last is digested,
    though `catalog` runs both (each under that one unit, `_produce`)."""
    from circuit_check import cache as CC

    units = CC._units_of(DUP_TARGETS.lstrip("\n"))[0]
    edited = CC._units_of(DUP_TARGETS.lstrip("\n").replace('lambda: "first"', 'lambda: "edited"'))[0]
    assert edited != units, "an edit to the first Dup_v1 entry changed no unit: every target that holds it keeps its key"


#: a Definition whose body depends on what the binding passed, not on its id (`make(swap)`), over a primitive that does not commute
MINUS = """
from verity.ml.circuits import Value, composite, primitive

V = Value(16)


@primitive("SynMinus", 1, [("a", V), ("b", V)], V)
def SynMinus(a, b):
    return (a - b) & 0xFFFF


def make(swap):
    @composite("SynMade", 1, [], lambda S: ((("a", V), ("b", V)), V))
    def SynMade(B, S, a, b):
        return B.call(SynMinus, b, a) if swap else B.call(SynMinus, a, b)
    return SynMade
"""

#: the catalog's `_model_roots`, two entries labelled alike, run by the harness as `targets.catalog` runs them
DUP_CATALOG_TARGETS = """
def _model_roots():
    from verity.ml.circuits.defs import bind
    from syn import minus, other
    return [
        ("Dup_v1", lambda: bind(minus.make(False))),
        ("Dup_v1", lambda: bind(other.SynOther)),
    ]
"""

CHECK_EVAL = """
def check_definition(fn, opts, *, memo, pins, as_call=None, **kw):
    from verity.ml.circuits.evaluation import evaluate
    return {"failures": [], "warnings": [], "value": list(evaluate(fn, 5, 3))}
"""

RUN_ENTRIES = """
import runpy
T = runpy.run_path(str(root / "targets.py"))
ENTRIES = [((f"_model_roots:{label}",), (lambda make=make: [make()])) for label, make in T["_model_roots"]()]
for units, make in ENTRIES:"""


def test_an_edit_to_the_first_of_two_same_label_entries_reruns_what_it_made(tmp_path):
    """The first `Dup_v1` entry makes SynMade_v1 with ``swap=False``.  Flipping it to ``True`` changes SynMade's body (its
    value at (5, 3): 2 -> 0xfffe), not its id, and changes only the first entry, which no unit digests."""
    root = _tree(tmp_path, {"syn/minus.py": MINUS, "syn/check3.py": CHECK_EVAL, "targets.py": DUP_CATALOG_TARGETS})
    h = _harness("syn.other", "check3").replace("for units, make in catalog.BINDINGS:", RUN_ENTRIES)
    h = h.replace("if TG.family(fn) in catalog.FAMILIES:", 'if TG.family(fn) in ("SynMade_v1", "SynOther_v1"):')
    cold = _run(root, h)
    assert cold["stats"]["stored"] == 2 and cold["targets"]["SynMade_v1"]["value"] == [2]
    assert "targets.py#_model_roots:Dup_v1" in _entry(root, "SynMade_v1")["reads"]["units"]
    _edit(root, "targets.py", "minus.make(False)", "minus.make(True)")
    assert _run(root, h, cache="fresh")["targets"]["SynMade_v1"]["value"] == [0xFFFE]
    warm = _run(root, h)
    assert "SynMade_v1" not in warm["hits"], (
        f"SynMade_v1 reused its entry (value {warm['targets']['SynMade_v1']['value']}) where a fresh run computes [65534]")
