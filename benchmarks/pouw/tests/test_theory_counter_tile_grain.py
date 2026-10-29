"""counter_tile_grain.py, `pouw-tile-grain`'s falsification test, on small calls of the lead: its copy term is
counter_patch's lane model, every measured skip form saves nothing, and a form W1 alone allows is caught."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))
import counter_patch as cpatch
import counter_tile_grain as tg


@pytest.fixture(scope="module")
def calls():
    cpatch.PARTIAL = True
    cpatch.use_head(tg.HEAD)
    out = {}
    for m, n, k, pattern in ((16, 16, 200, "zero"), (64, 16, 200, "zero"), (64, 32, 300, "a2c")):
        cpatch.COLUMNS = cpatch.spread_columns(n, 4096)
        w = cpatch.instance(m, n, k, 1, pattern)
        out[(m, n, k, pattern)] = (w, tg.histograms(w), m * n * k)
    yield out
    cpatch.COLUMNS = None
    cpatch.PARTIAL = False


def test_share_is_the_lane_model(calls):
    """The histogram copy term equals counter_patch's per-tile (`r1`) bound at every grid price, both source rules."""
    for (_, _, k, _), (w, h, stock) in calls.items():
        curve = cpatch.lanes(w, k)["floor_curve"]
        for src in ("uniform", "own"):
            for p in cpatch.GRID:
                assert tg.share(h, stock, p, 0.0, src) == pytest.approx(curve[f"{src}/p{p:g}/r1"], abs=1e-12)


def test_measured_forms_save_nothing(calls):
    """Each reading issues the patch arm it would skip, so its copy term is 0 at every source rule."""
    for _, h, stock in calls.values():
        for r in tg.READINGS:
            assert r.c == tg.H32
            assert all(tg.share(h, stock, r.p, r.c, src) == 0.0 for src in ("uniform", "own+load", "own"))


def test_w1_alone_is_falsified(calls):
    """A free decision with free arms (what W1 prices) makes copy-and-patch save more than `coinBranchGap`."""
    assert max(tg.share(h, st, 0.0, 0.0, "own+load") for _, h, st in calls.values()) > tg.COIN_BRANCH


def test_thresholds_are_least(calls):
    """Each threshold fits its budget and a price 0.01 under it doesn't, at both budgets."""
    for _, h, stock in calls.values():
        th = tg.thresholds(h, stock)
        for b, cap in tg.BUDGETS.items():
            p = th[b]["p_star/uniform"]
            assert tg.share(h, stock, p, 0.0, "uniform") <= cap
            if p > 0:
                assert tg.share(h, stock, p - 0.01, 0.0, "uniform") > cap
            c = th[b]["c_star/own+load/p13.5"]
            assert tg.share(h, stock, 13.5, c, "own+load") <= cap
            if c > 0:
                assert tg.share(h, stock, 13.5, c - 0.01, "own+load") > cap
            x = th[b]["b_star"]
            assert tg.register_share(h, stock, x) <= cap
            if x > 0:
                assert tg.register_share(h, stock, x - 0.01) > cap


def test_slack_is_the_tighter_budget(calls):
    for _, h, stock in calls.values():
        th = tg.thresholds(h, stock)
        assert th["coinSlack"]["p_star/uniform"] >= th["coinBranchGap"]["p_star/uniform"]
        assert th["coinSlack"]["c_star/uniform/p13.5"] >= th["coinBranchGap"]["c_star/uniform/p13.5"]


def test_verdict(calls):
    """The verdict holds at the readings and fails for a form whose arm doesn't issue, priced under `p_star`."""
    res = {key: {"m": key[0], "n": key[1], "k": key[2], "stock": st, "thresholds": tg.thresholds(h, st),
                 "judge": tg.judge(h, st)} for key, (_, h, st) in calls.items()}
    v = tg.verdict(res)
    assert v["holds"] and all(b["warp_branch"]["holds"] for b in v["by_budget"].values())
    assert all(f["holds"] for b in v["by_budget"].values() for f in b["forms"].values())
    free_arm = (tg.Reading("free-arm", 0.0, 0.0, "a skip whose arm doesn't issue, with a free decision", "test"),)
    res = {key: r | {"thresholds": tg.thresholds(calls[key][1], r["stock"], free_arm),
                     "judge": tg.judge(calls[key][1], r["stock"], free_arm)} for key, r in res.items()}
    v = tg.verdict(res, free_arm)
    assert not v["holds"] and not v["by_budget"]["coinSlack"]["forms"]["free-arm"]["holds"]
