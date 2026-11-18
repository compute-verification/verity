"""The scan state's rows a window reads (`network.prover.scan_state_rows`) on a window of two buckets and two slots; the
scan of served traffic is `tests/adapters/vllm/tests/program/predict/test_window_rows.py`'s, at the adapter's classes."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from verity.protocols.resource_accounting.network.prover import ingress_scan as IS
from verity.protocols.resource_accounting.network.prover import scan_state_rows as SR

SCAN = IS.Class(B=512, r=4, lanes=2, C=47, IDB=17, EVERY=16, WITHIN=12, D=27, KO=2, KI=4)


@dataclass(frozen=True)
class Window:
    scan: IS.Class = SCAN
    buckets: int = 2
    slots: int = 2

    def anchor(self, k: int) -> int:
        return 1

    def lane(self, k: int) -> int:
        return k


def test_the_standin_rows_hold_each_slots_row_at_its_anchor_and_lane_and_nothing_else():
    rows = SR.standin_rows(Window(), [([1, 3, 17, 5, 9, 0], 2), ([0, 0, 0, 0, 0, 0], 0)])
    assert len(rows) == 2 and all(len(r) == IS.state_bits(SCAN) for r in rows)
    assert rows[0] == IS.encode(SCAN, IS.empty(SCAN))
    assert IS.token_row(SCAN, IS.decode(SCAN, rows[1])) == [(1, 3, (17, 5, 9)), (0, 0, ())]


def test_a_window_past_the_ingress_rows_given_is_refused_by_name():
    with pytest.raises(ValueError, match="give the ingress rows to it, 3, not 1"):
        SR.scan_state(Window(), [b""], {}, 2)
