"""PoUS design C alone: the CPU reference challenger against simulated reads."""

from fractions import Fraction
from math import log

import numpy as np
import pytest

from verity.proofs import IntegrityProfile
from verity_pous import Challenger, DesignC, Response, respond, useful_kv

CFG = DesignC(sectors=16)
W = CFG.sector_words
REGIONS = {"weights": np.arange(2 * W, dtype=np.uint16).reshape(2, W), "kv": np.zeros((8, W), np.uint16)}


def device(regions, offloaded=(), dropped=()):
    def read(region, s):
        words = np.zeros(W, np.uint16) if (region, s) in dropped else regions[region][s]
        return words, words.nbytes / (CFG.b_alt if (region, s) in offloaded else CFG.b_hbm)
    return read


@pytest.fixture
def ch():
    return Challenger(CFG, b"key" * 11, "gpu0", regions={k: v.copy() for k, v in REGIONS.items()}, filler_sectors=32)


def memory(ch):
    return {**{k: v.copy() for k, v in REGIONS.items()}, "filler": ch.filler}


def run(ch, read, n=20, waited=0.0):
    return [ch.check(c, respond(CFG, read, c, waited)) for c in (ch.challenge(i, 0.0) for i in range(n))]


def test_honest_device_passes_and_folds_to_design_c_deficit(ch):
    outcomes = run(ch, device(memory(ch)))
    assert set(outcomes) == {"accepted"}
    eps = log(1 / CFG.delta) / (20 * CFG.sectors) + CFG.tau(ch.audit_bytes)
    assert all(abs(s.deficit - eps) < 1e-12 for s in ch.fold(outcomes).values())


def test_the_engine_may_take_preempt_bound_to_yield_and_no_longer(ch):
    assert set(run(ch, device(memory(ch)), waited=CFG.preempt_bound)) == {"accepted"}
    assert set(run(ch, device(memory(ch)), waited=CFG.deadline(ch.audit_bytes))) == {"late"}
    assert CFG.tau(ch.audit_bytes) > DesignC(sectors=16, preempt_bound=0).tau(ch.audit_bytes)


def test_kv_writes_must_reach_the_challenger_through_the_tap(ch):
    mem = memory(ch)
    mem["kv"][3] = 7
    assert "rejected" in run(ch, device(mem))
    ch.tap("kv", 3, mem["kv"][3])
    assert set(run(ch, device(mem))) == {"accepted"}


def test_dropped_sectors_fail_the_mac_and_offloaded_ones_miss_the_deadline(ch):
    assert "rejected" in run(ch, device(memory(ch), dropped={("filler", s) for s in range(16)}))
    assert set(run(ch, device(memory(ch), offloaded={("weights", 0), ("weights", 1)}))) == {"late"}
    assert all(s.accounted == 0 for s in ch.fold(["accepted", "late"]).values())


def test_a_wrong_aggregate_is_rejected(ch):
    c = ch.challenge(0, 0.0)
    good = respond(CFG, device(memory(ch)), c)
    bad = Response({k: (v[0] ^ 1, *v[1:]) for k, v in good.mu.items()}, good.elapsed)
    assert ch.check(c, good) == "accepted" and ch.check(c, bad) == "rejected"


def test_useful_kv_reads_linkage_and_touched_units_off_the_profile(ch):
    kv = ch.fold(run(ch, device(memory(ch))))["kv"]
    page = 2 * W
    full = IntegrityProfile((), frozenset(), True, 100, 4, Fraction(1), 4, 0.01, (0, 0))
    loose = IntegrityProfile((), frozenset(), True, 100, 4, Fraction(1, 2), 1, 0.01, (0, 0))
    assert useful_kv(kv, 6 * page, page, full).accounted == 6 * page
    assert useful_kv(kv, 6 * page, page, loose).accounted == 0
    unlinked = IntegrityProfile((), frozenset(), False, 100, 4, Fraction(1), 4, 0.01, (0, 0))
    assert useful_kv(kv, 6 * page, page, unlinked).accounted == 0
