"""`targets.pinned`: a specialization a pin is held for is a target whatever its size, so ``--all`` and its sidecar compare its
pin.  `targets.select` dropped one over `targets.MAX_GATES` when its family had a smaller one, which is how
`ChargeCount_v1{N=14}`'s stale pin went unreported while ``{N=2}`` was checked; the catalog records (`units`) keep it the same
way, and one stored before its pin is `units.Stale`."""
from __future__ import annotations

from pathlib import Path

import pytest

from verity.ml.circuits import Array, bind
from verity.ml.circuits.defs import CompositeDefinition
from verity.ml.circuits.ops.scalar import BIT, F32, F32Eq
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Tuple

from circuit_check import checks as CK
from circuit_check import targets as TG
from circuit_check import units as UN

REPO = Path(__file__).resolve().parents[4]
SIZES = (2, 8, 14)

Leaf = CompositeDefinition("PinnedLeaf", 1, ("N",), lambda S: ((("x", Array(S.N, F32)), ("y", Array(S.N, F32))), Array(S.N, BIT)),
                           lambda B, S, x, y: B.batch(F32Eq, x, y, axes=(0, 0)), register=False)
Parent = CompositeDefinition(
    "PinnedParent", 1, (),
    lambda S: (tuple((f"{v}{n}", Array(n, F32)) for n in SIZES for v in "xy"), Tuple(*(Array(n, BIT) for n in SIZES))),
    lambda B, S, *xs: tuple_of(*(B.call(bind(Leaf, N=n), xs[2 * i], xs[2 * i + 1]) for i, n in enumerate(SIZES))),
    register=False)


def test_a_pinned_specialization_over_max_gates_is_kept_whatever_its_size(monkeypatch):
    """A family reached at three sizes, the smallest at `MAX_GATES`: its smallest is kept, and of the two over it only the one
    pinned, also when every one is over it.  The records describe the pinned one, `units._select` and `units.derive` keep what
    `targets.select` keeps, and a record stored before the pin, which has no descriptor for it, is stale."""
    g = {n: bind(Leaf, N=n).gates for n in SIZES}
    assert g[2] < g[8] < g[14]
    monkeypatch.setattr(TG, "MAX_GATES", g[2])
    root = bind(Parent)
    small, mid, big = (bind(Leaf, N=n).id for n in SIZES)
    pinned = frozenset({big})
    found = TG.reachable([root])
    plain = set(TG.select(found, max_gates=g[2]))
    assert small in plain and not {mid, big} & plain
    kept = set(TG.select(found, max_gates=g[2], pinned=pinned))
    assert kept == plain | {big}
    over = {fid: fn for fid, fn in found.items() if fid != small}
    assert set(TG.select(over, max_gates=g[2], pinned=pinned)) == set(TG.select(over, max_gates=g[2])) | {big}

    nodes, _ = UN._reach([root], pinned=pinned)
    assert "d" in nodes[big] and "d" not in nodes[mid]
    assert UN._select(nodes, pinned) == kept
    assert set(UN.derive([UN.Entry(("t",), "t", "t", {}, [root], nodes)], pinned=pinned).ids) == kept
    before, _ = UN._reach([root], pinned=frozenset())
    with pytest.raises(UN.Stale, match="no descriptor"):
        UN.derive([UN.Entry(("t",), "t", "t", {}, [root], before)], pinned=pinned)


def test_the_pinned_ids_are_every_pin_held_for_a_definition(monkeypatch):
    monkeypatch.setattr(CK, "load_pins", lambda: {"definitions": {"A_v1": {}}, "digests": {"B_v1": {}}, "pieces": {"C_v1": {}},
                                                  "units": {"unit/u": {}}})
    assert TG.pinned() == {"A_v1", "B_v1", "C_v1"}


@pytest.mark.parametrize("sidecar, family", [("arith_encode", "ChargeCount_v1"), ("charge_chain", "ChargeLink_v1")])
def test_the_certifiers_pinned_n14_bindings_are_their_sidecars_targets(sidecar, family):
    """``{N=14}`` of the certifier's charge units is over `MAX_GATES` beside ``{N=2}`` under it; both are pinned, so both are
    their sidecar's targets, which ``--all`` keeps the same way (`select`)."""
    path = str(REPO / f"verity/core/protocols/certifier/{sidecar}.circuit-check.py")
    n2, n14 = f"{family}{{N=2}}", f"{family}{{N=14}}"
    assert {n2, n14} <= TG.pinned()
    found, _ = TG.sidecar_targets(path)
    assert {n2, n14} <= set(found) and found[n2].gates <= TG.MAX_GATES < found[n14].gates
    assert n14 not in TG.sidecar_dropped(path)
