"""Adversarial probes of `circuit_check.cache` through imports, moves and kernel registrations: edits that change a target's
verdict while its entry is still reused.

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
from test_cache import HARNESS, REPO, _edit, _entry, _tree


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
