"""PoUS design C alone: the certified-fraction curve and the CPU reference challenger against simulated reads."""

from fractions import Fraction
from math import log

import numpy as np
import pytest

from verity.proofs import IntegrityProfile
from verity_pous import Challenger, DesignC, Response, respond, useful_kv

CFG = DesignC(sectors=16)
W = CFG.sector_words
REGIONS = {"weights": np.arange(2 * W, dtype=np.uint16).reshape(2, W), "kv": np.zeros((8, W), np.uint16)}


@pytest.fixture
def ch():
    return Challenger(CFG, b"key" * 11, "gpu0", regions={k: v.copy() for k, v in REGIONS.items()}, filler_sectors=32)


def device(ch, regions, offloaded=(), dropped=()):
    def read(region, s):
        words = np.zeros(W, np.uint16) if (region, s) in dropped else regions[region][s]
        return words, ch.sector_bytes / (CFG.b_alt if (region, s) in offloaded else CFG.b_hbm)
    return read


def memory(ch):
    return {**{k: v.copy() for k, v in REGIONS.items()}, "filler": ch.filler}


def run(ch, read, n=20, waited=0.0):
    out = []
    for i in range(n):
        c = ch.challenge(i, 0.0)
        r = respond(CFG, read, c, waited)
        out.append((ch.check(c, r), r.elapsed))
    return out


def test_the_curve_reproduces_the_research_figure_and_falls_smoothly():
    cfg = DesignC()
    assert cfg.tau("filler", cfg.audit_bytes / cfg.b_hbm + 5e-4) == pytest.approx(0.0246, abs=5e-4)  # 2.5% at 0.5 ms slack
    curve = [cfg.certified("filler", L) for L in (1e-6, 16.4e-6, 1e-4, 1e-3, 1e-2, 1e-1)]
    assert all(a > b for a, b in zip(curve, curve[1:])) and curve[3] > 0.96 and curve[-1] == 0
    assert cfg.certified("filler", 1e-3) - cfg.certified("filler", 2e-3) == pytest.approx(cfg.b_alt * 1e-3 / cfg.audit_bytes)


def test_rebuild_bandwidth_is_charged_per_region():
    cfg = DesignC(b_rec={"weights": 100e9})
    assert cfg.certified("weights", 1e-4) < cfg.certified("filler", 1e-4) == DesignC().certified("filler", 1e-4)


def test_honest_device_folds_to_sampling_plus_timing(ch):
    records = run(ch, device(ch, memory(ch)))
    assert {o for o, _ in records} == {"accepted"}
    window = max(w for _, w in records)
    assert window == pytest.approx(CFG.audit_bytes / CFG.b_hbm)
    expected = log(1 / CFG.delta) / (20 * CFG.sectors) + CFG.tau("filler", window)
    assert all(s.deficit == pytest.approx(expected) and s.window == window for s in ch.fold(records).values())


def test_a_slow_engine_costs_certified_fraction_instead_of_failing(ch):
    prompt = ch.fold(run(ch, device(ch, memory(ch))))["filler"]
    slow = ch.fold(run(ch, device(ch, memory(ch)), waited=1e-3))["filler"]
    assert 0 < slow.accounted < prompt.accounted
    assert prompt.deficit - slow.deficit == pytest.approx(-CFG.b_alt * 1e-3 / CFG.audit_bytes)


def test_offloading_shows_up_as_a_slower_response_and_a_smaller_certificate(ch):
    honest = ch.fold(run(ch, device(ch, memory(ch))))["filler"]
    offloaded = ch.fold(run(ch, device(ch, memory(ch), offloaded={("filler", s) for s in range(32)})))["filler"]
    assert offloaded.window > 10 * honest.window and offloaded.accounted < honest.accounted


def test_kv_writes_must_reach_the_challenger_through_the_tap(ch):
    mem = memory(ch)
    mem["kv"][3] = 7
    assert "rejected" in {o for o, _ in run(ch, device(ch, mem))}
    ch.tap("kv", 3, mem["kv"][3])
    assert {o for o, _ in run(ch, device(ch, mem))} == {"accepted"}


def test_dropped_sectors_fail_the_mac_and_leave_nothing_certified(ch):
    records = run(ch, device(ch, memory(ch), dropped={("filler", s) for s in range(16)}))
    assert "rejected" in {o for o, _ in records}
    assert all(s.accounted == 0 for s in ch.fold(records).values())


def test_a_wrong_aggregate_is_rejected(ch):
    c = ch.challenge(0, 0.0)
    good = respond(CFG, device(ch, memory(ch)), c)
    bad = Response({k: (v[0] ^ 1, *v[1:]) for k, v in good.mu.items()}, good.elapsed)
    assert ch.check(c, good) == "accepted" and ch.check(c, bad) == "rejected"


def test_useful_kv_reads_linkage_and_touched_units_off_the_profile(ch):
    kv = ch.fold(run(ch, device(ch, memory(ch))))["kv"]
    page = ch.sector_bytes
    full = IntegrityProfile((), frozenset(), True, 100, 4, Fraction(1), 4, 0.01, (0, 0))
    loose = IntegrityProfile((), frozenset(), True, 100, 4, Fraction(1, 2), 1, 0.01, (0, 0))
    unlinked = IntegrityProfile((), frozenset(), False, 100, 4, Fraction(1), 4, 0.01, (0, 0))
    assert useful_kv(kv, 6 * page, page, full).accounted == 6 * page
    assert useful_kv(kv, 6 * page, page, loose).accounted == 0
    assert useful_kv(kv, 6 * page, page, unlinked).accounted == 0
