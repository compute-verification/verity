"""Suite-wide isolation guard: no test may leave a loaded `verity` module swapped for another copy.

A re-executed `verity.ml.circuits.defs` is a second module with its own empty REGISTRY; every registry module first imported against it registers
into that orphan for the rest of the process, so the failure surfaces in whichever later file shares the worker, never in the test that
caused it.  The guard fails the causing test by name and puts the original modules back."""
from __future__ import annotations

import sys

import pytest

import verity.ml.circuits.defs  # noqa: F401  (loaded before the first test, so the guard's snapshot always holds it)


_MISSING = object()


def _core(name: str) -> bool:
    """`name` is core as it was before the platform move: under `verity`, except what moves since put there from outside it
    (the catalog, the kernels, Flock, verity_experimental's multiproofs, private circuits and ports), with core's own
    protocols and claims, which are under `verity.core` now."""
    def under(prefix: str) -> bool:
        return name == prefix or name.startswith(prefix + ".")
    moved_in = ("verity.core", "verity.ml.commitments.multiproof", "verity.ml.private_circuits", "verity.ml.circuits.ops",
                "verity.ml.circuits.silicon", "verity.ml.circuits.targets", "verity.ml.kernels", "verity.ml.flock")
    return under("verity") and (not any(map(under, moved_in))
                                or any(map(under, ("verity.core.protocols", "verity.core.claims"))))


def _verity_modules() -> dict[str, tuple[object, object, object]]:
    """{name: (module, parent package, the parent's attribute of that name)} -- the attribute is snapshotted, not assumed to be the
    module: a package may re-export a function under its submodule's name (`verity.ml.circuits.query`)."""
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
                    f"a re-executed verity.ml.circuits.defs carries its own REGISTRY and orphans every registry module imported against it",
                    pytrace=False)
