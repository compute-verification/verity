"""F6's pinned menu: each bucket's B is the cheapest at its public session count among those that are in range, fit one
node and stay in the law's fitted D, with the outer proof's cost when given and on the inner alone without it."""
from __future__ import annotations

import math

import pytest

from verity_numerical.bench import unit_costs_menu as M
from verity_numerical.bench import unit_costs_padding as P

#: one second per session at any size; RSS 1 GiB + 2 GiB per 2^30 bits of D
COEF = {"rss_gib": (1.0, 2.0, 0.0, 0.0), "prove_s": (1.0, 0.0, 0.0, 0.0), "wall_s": (0.0, 0.0, 0.0, 0.0)}
TINY = {"bytes_per_nnz": 4, "glue": 0}


def _bucket(units, Z):
    return {"part": "body", "class": "c", "Z": Z, "units": units, "pieces": False}


def test_the_cheapest_public_b_that_fits_one_node():
    """N = 100, a type of 60 units, K = 50 drawn: its public count q needs ceil(q / B) one-second sessions, fewer at larger B.
    A 2^22-bit slot gives RSS 1 + B / 128 GiB: within 1.2 GiB, B = 16 is the largest that fits and so the cheapest."""
    b = _bucket([P.Unit("a", "t", 60, 3, 3)], 3)
    opts = {B: M.inner(100, 50, b, B, {}, COEF, **TINY) for B in (1, 4, 16, 64)}
    q = P.upper_q(100, 60, 50)
    assert [opts[B]["sessions_public"] for B in (1, 4, 16, 64)] == [q, -(-q // 4), -(-q // 16), -(-q // 64)]
    assert [opts[B]["peak_gib"] for B in (1, 4, 16, 64)] == pytest.approx([1 + B / 128 for B in (1, 4, 16, 64)])
    assert M.pick(opts, gib=math.inf) == 64 and M.pick(opts, gib=1.2) == 16
    assert M.excluded(opts[64], gib=1.2) == "inner 2 GiB > 1"


def test_a_slot_past_k_log_27_runs_at_no_b():
    """2^26 nonzeros at one per row: k_log 29, out of range at every B."""
    b = _bucket([P.Unit("big", "t", 60, 2.0 ** 26, 2.0 ** 26)], 2.0 ** 26)
    opts = {B: M.inner(100, 50, b, B, {"big": 1.0}, COEF, **TINY) for B in (1, 16)}
    assert {o["k_log"] for o in opts.values()} == {29}
    with pytest.raises(ValueError, match="k_log 29 > 27"):
        M.pick(opts)


def test_pick_on_inner_plus_outer():
    o = {"in_range": True, "k_log": 22, "peak_gib": 1.0, "inner_m": 30}
    opts = {1: dict(o, inner_h_public=4.0), 4: dict(o, inner_h_public=2.0), 16: dict(o, inner_h_public=1.0, in_range=False),
            64: dict(o, inner_h_public=1.0, inner_m=36), 256: dict(o, inner_h_public=2.0)}
    assert M.pick(opts) == 4
    assert M.excluded(opts[64]) == "D 2^36 > 2^35"
    opts[4]["outer_h_public"], opts[256]["outer_h_public"] = 3.0, 0.5
    assert M.pick(opts) == 256
    opts[256]["outer_rss_gib"] = 2000.0
    assert M.pick(opts) == 1 and M.excluded(opts[256]) == "outer 2000 GiB > 1716"


def test_the_menu_with_and_without_the_outer_input():
    b = _bucket([P.Unit("a", "t", 60, 3, 3)], 3)
    b["by_B"] = {B: M.inner(100, 50, b, B, {}, COEF, **TINY) for B in (1, 16)}
    pl = [{"workload": "w", "N": 100, "cap": 2.0 ** 29, "buckets": [b]}]
    alone = M.menu(pl, 50)
    e = alone["models"][0]["entries"][0]
    assert "inner hours alone" in alone["chosen_on"] and e["slots"] == 16 and e["outer_h_public"] is None
    assert alone["models"][0]["totals"]["outer_h_public"] is None
    outer = {"buckets": [{"workload": M.row_id("w", 0, 1), "prove_s": 1.0, "rss_gib": 1.0},
                         {"workload": M.row_id("w", 0, 16), "prove_s": 3600.0 * 100, "rss_gib": 1.0, "refused": ["x"]}]}
    m = M.menu(pl, 50, outer)
    e = m["models"][0]["entries"][0]
    assert e["slots"] == 1 and e["outer_h_public"] == pytest.approx(e["sessions_public"] / 3600)
    assert e["total_h_public"] == pytest.approx(e["inner_h_public"] + e["outer_h_public"])
    assert m["models"][0]["totals"]["refused_entries"] == 0 and isinstance(e["options"]["16"], float)
    assert M.render(m) == M.render(M.menu(pl, 50, outer))
