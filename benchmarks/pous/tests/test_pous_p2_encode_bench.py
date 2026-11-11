"""`p2_v1/encode_bench.py`'s time budget, which reads no GPU: how long a pass may take, from the passes before it."""
from __future__ import annotations

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]


def test_a_pass_is_estimated_from_passes_on_as_many_blocks_not_from_a_rate_at_one_block():
    sys.path.insert(0, str(BENCH / "p2_v1"))
    import encode_bench as E

    assert E.at_most({}, 1024) is None
    # one block alone takes a root's whole chain, about as long as a thousand side by side
    assert E.at_most({1: 1.1}, 1024) == 1.1 * 1024
    assert E.at_most({1: 1.1, 2112: 3.2, 65536: 98.8}, 1024) == 3.2
    assert E.at_most({1: 1.1, 2112: 3.2, 65536: 98.8}, 65536) == 98.8
    assert E.at_most({2112: 3.2}, 65536) == 3.2 * 65536 / 2112
