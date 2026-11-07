"""What `ChargeCount_v2`'s charge doesn't see of its witness, which the slice that wires it must pin. `WindowCharge.Window`
keeps each field's values at the slots that hold one, in order, so the guarantees count advice values, not slot
arrangements: the wiring takes `hen` and `oen` from the replay, as `en` is, and reads no disabled slot's value
(`protocol/charge-chain.md`, "The slice")."""
from __future__ import annotations

import itertools
import json
from math import comb
from pathlib import Path

from verity.ml.protocols.certifier import window_charge as WC

FIXTURE = Path(__file__).parent / "fixtures" / "arith_encode_vectors.json"


def _book():
    return next(s for s in json.loads(FIXTURE.read_text())["streams"] if "window" in s)["decisions"]["book"]


def test_the_hit_and_order_slots_arrangement_and_disabled_values_are_not_charged():
    """Two hits in three hit slots and one order entry in two order slots: every arrangement of the enabled slots,
    with the disabled slots holding values past every cap, is one charge and one carry. So a slice that lets the
    prover choose `hen` or `oen`, or reads a disabled slot, passes log2(C(H, h) · C(O, o)) bits per window, and the
    disabled slots' values, that no guarantee counts: the wiring must take them from the scan or read only the held
    values."""
    N, H, O = 1, 3, 2
    wb, carry = WC.window_book(_book(), N, H, O)
    hits, entry = [5, 70], (9, 3)
    cases = []
    for held in itertools.combinations(range(H), 2):
        for at in range(O):
            vals = iter(hits)
            hen = [int(i in held) for i in range(H)]
            short = [next(vals) if e else (1 << 17) - 1 for e in hen]
            oen = [int(i == at) for i in range(O)]
            p = [entry[0] if e else (1 << 32) - 1 for e in oen]
            r = [entry[1] if e else (1 << 25) - 1 for e in oen]
            cases.append((wb, carry, [1], [0], hen, short, oen, p, r))
    got = WC.evaluate_windows(N, H, O, cases)
    want = WC.reference_window(N, H, O, *cases[0])
    assert want[0] == 1
    assert len(cases) == comb(H, 2) * comb(O, 1) == 6
    assert all(g == want for g in got)
