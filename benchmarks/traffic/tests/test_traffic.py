from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import flops  # noqa: E402
import hashtrace  # noqa: E402
import measure  # noqa: E402

B = hashtrace.BLOCK


def _req(t, n, out, h):
    return {"timestamp": t, "input_length": n, "output_length": out, "hash_ids": h}


def test_a_prompt_many_sessions_share_is_registered_and_a_session_history_is_not():
    # ten sessions share a two-block system prompt (ids 0, 1); session 7 continues for a second turn
    rows = [_req(i, 4 * B + 10, 100, [0, 1, 10 * i + 2, 10 * i + 3, 10 * i + 4]) for i in range(10)]
    rows.append(_req(20, 7 * B + 5, 50, [0, 1, 72, 73, 74, 900, 901, 902]))
    a = hashtrace.analyze(hashtrace.read_rows(rows), fork_min=4)
    assert list(a["reg"]) == [2 * B] * 11
    assert a["sess"][10] == a["sess"][7] and a["turn"][10] == 1
    assert a["new"][10] == (7 * B + 5) - (4 * B + 10) - 100
    assert a["new"][0] == 4 * B + 10 - 2 * B
    assert a["stats"]["sessions"] == 10


def test_a_reply_the_next_turn_does_not_hold_counts_its_whole_growth():
    rows = [_req(0, 2 * B, 5000, [0, 1]), _req(1, 3 * B + 1, 10, [0, 1, 2, 3])]
    a = hashtrace.analyze(hashtrace.read_rows(rows), fork_min=4)
    assert a["new"][1] == B + 1 and a["stats"]["reply_not_held"] == 1


def test_per_token_flops_are_two_per_active_parameter_plus_attention():
    assert flops.call_flops("dense7b", 0, 1, 0) == pytest.approx(1.4e10, rel=1e-3)
    assert flops.call_flops("dsv4", 0, 1, 0) == pytest.approx(9.8e10, rel=1e-3)
    # full attention grows linearly with position; DSV4's compressed attention much more slowly
    d = [flops.call_flops("dense7b", t, t + 1, 0) - 1.4e10 for t in (1e4, 1e5)]
    assert d[1] / d[0] == pytest.approx(10, rel=1e-3)
    s = [flops.call_flops("dsv4", t, t + 1, 0) - 9.8e10 for t in (1e4, 1e5)]
    assert s[1] / s[0] < 6


def test_rule_three_counts_a_session_once_and_recomputes_nothing():
    d = {"sess": np.array([0, 0]), "turn": np.array([0, 1]), "ctx": np.array([1000, 1600]),
         "reg": np.array([200, 200]), "new": np.array([800, 400]), "out": np.array([200, 100])}
    r1, r2, r3 = (measure.per_unit(d, r) for r in ("R1", "R2", "R3"))
    assert list(r1["tok"]) == [1000, 1600] and list(r2["tok"]) == [800, 1400] and list(r3["tok"]) == [1200]
    assert r3["dense7b"][0] == pytest.approx(flops.call_flops("dense7b", 200, 1000, 200)
                                             + flops.call_flops("dense7b", 1200, 1600, 100))
    assert r3["dense7b"][0] < r2["dense7b"].sum() < r1["dense7b"].sum()
