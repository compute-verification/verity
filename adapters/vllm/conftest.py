"""Suite-wide isolation guard: no test may leave a loaded `verity` module swapped for another copy.

A re-executed `verity.circuits.defs` is a second module with its own empty REGISTRY; every registry module first imported against it registers
into that orphan for the rest of the process, so the failure surfaces in whichever later file shares the worker, never in the test that
caused it.  The guard fails the causing test by name and puts the original modules back.

Every process of the suite runs under the engine profile's fixed Inductor switches (`engine.env.FIXED`), as an engine's process
does: torch reads them once, when `torch._inductor.config` is imported, so this file pins them and imports that module itself,
before any test.  No test is then the first to import it, whatever that test has cleared from the environment, and a test that
builds engine arguments (`engine.env.require_inductor`) doesn't depend on which test ran before it.  A second guard fails, by
name, a test that leaves the loaded module reading a fixed switch otherwise, and puts the pinned value back; it looks once the
test's teardown has finalized every fixture, so a change the test's `monkeypatch` undoes is not one it left."""
from __future__ import annotations

import importlib.util
import sys

import pytest

from verity_vllm.engine import env as _engine_env

_engine_env.pin({k: _engine_env.ENGINE[k] for k in _engine_env.FIXED})
if importlib.util.find_spec("torch") is not None:  # torch is an extra of this package: where it is absent, nothing reads the switches
    import torch._inductor.config  # noqa: F401

import verity.circuits.defs  # noqa: E402,F401  (loaded before the first test, so the guard's snapshot always holds it)


_MISSING = object()


def _core(name: str) -> bool:
    """`name` is core as it was before the platform move: under `verity`, except what moves since put there from outside it
    (the catalog, the kernels, Flock, verity_experimental's multiproofs, private circuits and ports), with core's own
    protocols and claims, which are under `verity.core` now."""
    def under(prefix: str) -> bool:
        return name == prefix or name.startswith(prefix + ".")
    moved_in = ("verity.core", "verity.commitments.multiproof", "verity.definitions.private_circuits", "verity.definitions.ops",
                "verity.definitions.silicon", "verity.definitions.circuits.targets", "verity.definitions.kernels", "verity.ml.flock")
    return under("verity") and (not any(map(under, moved_in))
                                or any(map(under, ("verity.core.protocols", "verity.claims"))))


def _verity_modules() -> dict[str, tuple[object, object, object]]:
    """{name: (module, parent package, the parent's attribute of that name)} -- the attribute is snapshotted, not assumed to be the
    module: a package may re-export a function under its submodule's name (`verity.circuits.query`)."""
    snap = {}
    for n, m in list(sys.modules.items()):
        if m is None or not _core(n):
            continue
        parent, _, child = n.rpartition(".")
        pkg = sys.modules.get(parent) if parent else None
        snap[n] = (m, pkg, _MISSING if pkg is None else getattr(pkg, child, _MISSING))
    return snap


@pytest.fixture(autouse=True)
def _verity_modules_not_swapped():
    before = _verity_modules()
    yield
    swapped = []
    for n, (m, pkg, attr) in before.items():
        child = n.rpartition(".")[2]
        if sys.modules.get(n) is not m or (pkg is not None and getattr(pkg, child, _MISSING) is not attr):
            swapped.append(n)
            sys.modules[n] = m
            if pkg is not None and attr is not _MISSING:
                setattr(pkg, child, attr)
    if swapped:
        pytest.fail(f"test left loaded verity module(s) swapped for another copy (sys.modules or package attribute): {sorted(swapped)}; "
                    f"a re-executed verity.circuits.defs carries its own REGISTRY and orphans every registry module imported against it",
                    pytrace=False)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item):
    """The Inductor guard, after the test's every fixture is finalized (the hook's own work), not as a fixture among them: an
    autouse fixture is torn down before `monkeypatch` undoes the test's change wherever `monkeypatch` was set up first, as under a
    plugin whose own autouse fixture takes it, and would read what the test was about to put back."""
    try:
        result = yield
    finally:
        cfg, pinned = sys.modules.get("torch._inductor.config"), dict(_engine_env.FIXED.values())
        left = {attr: getattr(cfg, attr) for attr, want in pinned.items() if cfg is not None and getattr(cfg, attr, want) != want}
        for attr in left:
            setattr(cfg, attr, pinned[attr])
    if left:
        pytest.fail(f"test left torch._inductor.config reading {left}, where every process of the suite reads {pinned} "
                    f"(engine.env.FIXED): it set the attribute, or unloaded the module and imported it again without the pin in its "
                    f"environment, and every later engine build in this process would be refused (engine.env.require_inductor); undo "
                    f"what the test changed (monkeypatch.setattr, monkeypatch.delitem)", pytrace=False)
    return result
