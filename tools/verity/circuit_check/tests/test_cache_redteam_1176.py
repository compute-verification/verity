"""Red-team probes of `circuit_check.cache` v3 (PR #1176): edits that change a target's verdict while its entry is still reused.

Each end-to-end test runs the synthetic tree of `test_cache`: a cold run, the edit, a run under a fresh cache (the verdict as
it is now), and a run under the first cache.  A test fails where the cached run reuses the stale entry.  The fresh caches sit
beside the tree, so no listing of the tree sees them."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_cache import FILES, HARNESS, KERNEL_FILES, REPO, _bindings, _edit, _entry, _move


def _tree(tmp: Path, extra: dict[str, str]) -> Path:
    subprocess.run(["git", "init", "-q", str(tmp)], check=True)
    for rel, text in {**FILES, **extra}.items():
        p = tmp / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text.lstrip("\n"))
    return tmp


def _harness(check: str = "check") -> str:
    h = HARNESS if check == "check" else \
        HARNESS.replace("from syn import catalog, check\n", f"from syn import catalog\nfrom syn import {check} as check\n")
    return h.replace('print(json.dumps({"stats"',
                     'print(json.dumps({"targets": {t["target"]: {k: t.get(k) for k in ("failures", "value", "regkind")}'
                     ' for t in rep["targets"]}, "stats"')


def _run(root: Path, harness: str, cache: str = "cache") -> dict:
    where = 'root / "cache"' if cache == "cache" else f'Path({str(root.parent / f"{root.name}-{cache}")!r})'
    h = harness.replace('cache=root / "cache"', f"cache={where}")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run([sys.executable, "-c", h, str(root), str(REPO)], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert not out["stats"]["untraced"] and not out["stats"]["errors"] and not out["stats"]["changed"], out
    return out


def _codes(fails: list[dict]) -> list[str]:
    return sorted(f"{f['check']}/{f['code']}" for f in fails)


def _add_binding(root: Path, name: str, module: str, attr: str, family: str) -> None:
    _edit(root, "syn/catalog.py", "\nBINDINGS = [",
          f"\ndef {name}():\n    from syn import {module}\n    return [bind({module}.{attr})]\n\n\nBINDINGS = [")
    _bindings(root, f'(("_{name}_roots",), {name})', families=(family,))


#: a package no enumerator lists, and a primitive whose evaluator reads its constant through a plain ``import lib.consts``
PLAIN = {
    "lib/__init__.py": "",
    "lib/consts.py": "MASK = 0xFFFF\nBITS = 16\n",
    "syn/pkern.py": """
import lib.consts
from verity.ml.circuits import primitive

from syn.prims import V


@primitive("SynPx", 1, [("a", V), ("b", V)], V)
def SynPx(a, b):
    return (a ^ b) & lib.consts.MASK
""",
    "syn/pdef.py": """
from verity.ml.circuits import composite

from syn.pkern import SynPx
from syn.prims import V


@composite("SynP", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynP(B, S, a, b):
    return B.call(SynPx, a, b)
""",
}


def test_a_module_a_plain_import_binds_is_held_at_the_name_it_binds(tmp_path):
    """``import lib.consts`` records no import site (`_Norm` keeps it by name), so `_moved` finds a changed `lib.consts` as
    "the one loaded module of its old content".  Moving `lib/consts.py` into `lib/fam/` and leaving at the old name a module
    that takes the moved one and narrows ``MASK`` maps `lib.consts` to `lib.fam.consts`, while `syn.pkern` still binds and
    reads `lib.consts`: a fresh run computes the narrowed value."""
    root = _tree(tmp_path, PLAIN)
    _add_binding(root, "pxor", "pdef", "SynP", "SynP_v1")
    h = _harness()
    cold = _run(root, h)
    assert cold["targets"]["SynP_v1"]["value"] == [(3 ^ 0x0105) & 0xFFFF]
    assert "lib.consts" in _entry(root, "SynP_v1")["reads"]["modules"]
    _move(root, "lib/consts.py", "lib/fam/consts.py")
    (root / "lib/consts.py").write_text("from lib.fam.consts import *  # noqa: F403\n\nMASK = 0x00FF\n")
    assert _run(root, h, cache="fresh")["targets"]["SynP_v1"]["value"] == [(3 ^ 0x0105) & 0xFF]
    warm = _run(root, h)
    assert "SynP_v1" not in warm["hits"], (
        f"SynP_v1 reused its entry (value {warm['targets']['SynP_v1']['value']}) where a fresh run computes [6]")


#: a module the enumeration imports, which registers into a table another module owns (as a kernel module registers into
#: `_KERNELS`): no check runs any of it.  The check reports what it looks up as ``regkind`` (the report's own ``kind`` is the
#: target's)
PLUG = {
    "syn/plug.py": 'from syn import reg\n\nreg.KINDS["SynOther_v1"] = "plugged"\n',
    "syn/pcheck.py": """
def check_definition(fn, opts, *, memo, pins, as_call=None, **kw):
    from syn import reg
    return {"failures": [], "warnings": [], "regkind": reg.KINDS.get(fn.id)}
""",
}


@pytest.mark.xfail(strict=True, reason="not covered (cache.py docstring): a value a module only the registries import"
                   " leaves in another module's state at import; an enumerated module is held by its name in the listing")
def test_an_edit_to_a_module_an_enumerated_package_lists_is_no_move(tmp_path):
    """SynOther's check reads ``reg.KINDS``, which `syn/plug.py` fills at import; SynOther holds `syn.plug` only through the
    enumerated package `syn` (`enums`, each module by its content).  An edit of `syn/plug.py` changes that record, and
    `_moved` accepts it: a module listed both before and now is not compared.  Pre-existing (v2 held no enumerated module) and
    deliberate (`test_editing_a_registry_module_no_target_resolves_reuses_every_target`), so the fix is the docstring's: say
    the listing is held, not each module's content, and list this edit as not covered."""
    root = _tree(tmp_path, PLUG)
    h = _harness("pcheck")
    cold = _run(root, h)
    assert cold["targets"]["SynOther_v1"]["regkind"] == "plugged"
    rec = _entry(root, "SynOther_v1")["reads"]
    assert "syn.plug" not in rec["modules"] and "plug" in rec["enums"]["syn"]
    _edit(root, "syn/plug.py", '"plugged"', '"replugged"')
    assert _run(root, h, cache="fresh")["targets"]["SynOther_v1"]["regkind"] == "replugged"
    warm = _run(root, h)
    assert "SynOther_v1" not in warm["hits"], (
        f"SynOther_v1 reused its entry (regkind {warm['targets']['SynOther_v1']['regkind']!r}) where a fresh run reads"
        " 'replugged'")


#: `kernel_registry.register`'s shape: core holds a wrapper (`_batch.<locals>.run`) defined in the registry module, closing over
#: the function registered, which `syn/krows.py` names from `syn/kimpl.py`
WRAPPED = {
    "syn/kreg.py": """
from verity.ml.circuits.evaluation import register_kernel


def _batch(fn):
    def run(*arrays, **statics):
        return fn(*arrays, **statics), None
    return run


def register(target, fn, **kw):
    return register_kernel(target, _batch(fn), **kw)
""",
    "syn/kimpl.py": """
import numpy as np


def add(a, b):
    return (a.astype(np.uint64) + b) & 0xFFFF


def sub(a, b):
    return (a.astype(np.uint64) - b) & 0xFFFF
""",
    "syn/krows.py": 'from syn import kimpl, kreg\n\nkreg.register("SynTop_v1", kimpl.add)\n',
    "syn/corr.py": KERNEL_FILES["syn/corr.py"],
}


def test_a_kernel_registered_through_a_wrapper_reruns_its_target_when_the_function_registered_changes(tmp_path):
    """SynTop's kernel is ``run``, a closure in `syn/kreg.py`; `_where` names it by that module's content, its qualname and its
    definition, the same whichever function it wraps.  `syn/krows.py`, which chose ``kimpl.add``, ran only at import, so no
    check holds it: registering ``kimpl.sub`` instead keeps every read SynTop holds, and a fresh run fails
    `correspondence/kernel-mismatch`."""
    root = _tree(tmp_path, WRAPPED)
    h = _harness("corr")
    cold = _run(root, h)
    assert cold["targets"]["SynTop_v1"]["failures"] == []
    rec = _entry(root, "SynTop_v1")["reads"]
    assert list(rec["kernels"]) == ["SynTop_v1"]
    _edit(root, "syn/krows.py", "kimpl.add", "kimpl.sub")
    assert _codes(_run(root, h, cache="fresh")["targets"]["SynTop_v1"]["failures"]) == ["correspondence/kernel-mismatch"]
    warm = _run(root, h)
    assert "SynTop_v1" not in warm["hits"], (
        f"SynTop_v1 reused its entry (failures {warm['targets']['SynTop_v1']['failures']}) for a correspondence/kernel-mismatch")


#: a constant in a package no enumerator lists, read by a primitive's evaluator through a ``from`` import
MOVED = {
    "lib/__init__.py": "",
    "lib/lconsts.py": "MASK = 0xFFFF\nBITS = 16\n",
    "syn/lkern.py": """
from verity.ml.circuits import primitive

from lib.lconsts import MASK
from syn.prims import V


@primitive("SynLx", 1, [("a", V), ("b", V)], V)
def SynLx(a, b):
    return (a ^ b) & MASK
""",
    "syn/ldef.py": """
from verity.ml.circuits import composite

from syn.lkern import SynLx
from syn.prims import V


@composite("SynL", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynL(B, S, a, b):
    return B.call(SynLx, a, b)
""",
}


def test_an_entry_stored_again_after_a_move_holds_the_package_the_module_moved_into(tmp_path):
    """`lib/lconsts.py` moves into a new package `lib/fam/` (its ``__init__.py`` a docstring) and its one import is rewritten:
    SynL is reused, and stored again under the new names (`lookup`).  A fresh run would hold `lib.fam`, the moved module's
    parent, whose ``__init__.py`` runs before it; the entry stored again does not, so an edit there that sets ``MASK``
    before `syn.lkern` reads it is reused."""
    root = _tree(tmp_path, MOVED)
    _add_binding(root, "lxor", "ldef", "SynL", "SynL_v1")
    h = _harness()
    assert _run(root, h)["targets"]["SynL_v1"]["value"] == [(3 ^ 0x0105) & 0xFFFF]
    _move(root, "lib/lconsts.py", "lib/fam/lconsts.py", ("syn/lkern.py", "from lib.lconsts import", "from lib.fam.lconsts import"))
    assert "SynL_v1" in _run(root, h)["hits"]
    assert "SynL_v1" in _run(root, h)["hits"]
    init = root / "lib/fam/__init__.py"
    init.write_text(init.read_text() + "\nimport lib.fam.lconsts as _c\n\n_c.MASK = 0x00FF\n")
    assert _run(root, h, cache="fresh")["targets"]["SynL_v1"]["value"] == [(3 ^ 0x0105) & 0xFF]
    warm = _run(root, h)
    assert "SynL_v1" not in warm["hits"], (
        f"SynL_v1 reused its entry (value {warm['targets']['SynL_v1']['value']}) where a fresh run computes [6]")


#: an optional backend imported in a ``try`` whose fallback does not rebind every name the body imports
GUARDED = {
    "syn/wide.py": "MASK = 0xFFFF\nWIDE = True\n",
    "syn/okern.py": """
from verity.ml.circuits import primitive

from syn.prims import V

MASK = 0x00FF
try:
    from syn.wide import MASK
    import syn_absent_backend as fast
except ImportError:
    fast = None


@primitive("SynOx", 1, [("a", V), ("b", V)], V)
def SynOx(a, b):
    return (a ^ b) & MASK
""",
    "syn/odef.py": """
from verity.ml.circuits import composite

from syn.okern import SynOx
from syn.prims import V


@composite("SynO", 1, [], lambda S: ((("a", V), ("b", V)), V))
def SynO(B, S, a, b):
    return B.call(SynOx, a, b)
""",
}


def test_imports_in_a_try_body_do_not_commute(tmp_path):
    """`_commuted` orders the run of imports in the ``try`` body by the names they bind, so the order an import sorter writes
    (``import syn_absent_backend as fast`` before ``from syn.wide import MASK``) digests as the original does; but the failing
    import now comes first, ``MASK`` keeps its fallback, and a fresh run computes the narrowed value."""
    root = _tree(tmp_path, GUARDED)
    _add_binding(root, "oxor", "odef", "SynO", "SynO_v1")
    h = _harness()
    assert _run(root, h)["targets"]["SynO_v1"]["value"] == [(3 ^ 0x0105) & 0xFFFF]
    _edit(root, "syn/okern.py", "    from syn.wide import MASK\n    import syn_absent_backend as fast\n",
          "    import syn_absent_backend as fast\n    from syn.wide import MASK\n")
    assert _run(root, h, cache="fresh")["targets"]["SynO_v1"]["value"] == [(3 ^ 0x0105) & 0xFF]
    warm = _run(root, h)
    assert "SynO_v1" not in warm["hits"], (
        f"SynO_v1 reused its entry (value {warm['targets']['SynO_v1']['value']}) where a fresh run computes [6]")


def test_a_future_import_keeps_its_place():
    """``from __future__`` must come first, or the module does not compile; `_commuted` orders it among the run by the name it
    binds, so a module with it moved below another import digests as the valid one does."""
    from circuit_check import cache as CC

    valid = "from __future__ import annotations\nfrom abc import ABC\n"
    broken = "from abc import ABC\nfrom __future__ import annotations\n"
    assert CC._normalized(ast.parse(valid))[:2] != CC._normalized(ast.parse(broken))[:2], \
        "a module whose __future__ import no longer comes first (it does not compile) digests as the valid one"


@pytest.mark.xfail(strict=True, reason="not covered (cache.py docstring): a module edited after this process imported it"
                   " and before the check ran")
def test_a_module_edited_after_it_was_imported_and_before_the_check_ran_stores_nothing(tmp_path):
    """The registries import `syn.prims` before any check; an edit of its evaluator that lands after that import and before
    SynTop's check (an editor saving during a long ``--all`` run) leaves the process running the old code.  The worker's
    `seen` and `store` both digest the file after the edit, so they agree, and the old code's value is stored under the new
    source's digest: every later run reuses it.  (Pre-existing: v2 digests at the same two points.)"""
    root = _tree(tmp_path, {})
    h = _harness().replace("TG.load_registries = lambda: None\n",
                           "TG.load_registries = lambda: None\n"
                           "_p = root / 'syn/prims.py'\n"
                           "_p.write_text(_p.read_text().replace('(a + b) & 0xFFFF', '(a + b) & 0xFF'))\n", 1)
    raced = _run(root, h)
    assert raced["targets"]["SynTop_v1"]["value"] == [(3 + 0x0105) & 0xFFFF], "the run computed with the code it imported"
    plain = _harness()
    assert _run(root, plain, cache="fresh")["targets"]["SynTop_v1"]["value"] == [(3 + 0x0105) & 0xFF]
    warm = _run(root, plain)
    assert "SynTop_v1" not in warm["hits"], (
        f"SynTop_v1 reused its entry (value {warm['targets']['SynTop_v1']['value']}), computed by code the tree no longer has,"
        " where a fresh run computes [8]")
