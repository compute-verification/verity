"""PoUS design C alone: the CPU reference challenger against simulated devices."""

from math import log

import numpy as np
import pytest

from verity_pous import Challenger, DesignC, Response, respond, sectors

CFG = DesignC()
WEIGHTS = np.arange(300, dtype=np.uint16)


def device(regions, offloaded=(), dropped=()):
    def read(region, s):
        words = np.zeros(CFG.sector_words, np.uint16) if (region, s) in dropped else regions[region][s]
        return words, words.nbytes / (CFG.b_alt if (region, s) in offloaded else CFG.b_hbm)
    return read


@pytest.fixture(scope="module")
def ch():
    return Challenger(CFG, b"key" * 11, "gpu0", weights=WEIGHTS, filler_sectors=64)


def regions(ch):
    return {"weights": sectors(WEIGHTS, CFG.sector_words), "filler": ch.filler}


def run(ch, read, duration=0.5):
    return [ch.check(c, respond(CFG, read, c)) for c in (ch.challenge(i, t) for i, t in enumerate(ch.times(duration)))]


def test_honest_device_passes_and_folds_to_design_c_deficit(ch):
    outcomes = run(ch, device(regions(ch)))
    assert set(outcomes) == {"accepted"} and 30 < len(outcomes) < 80
    spaces = ch.fold(outcomes)
    eps = log(1 / CFG.delta) / (len(outcomes) * CFG.sectors) + CFG.tau
    assert {s.region for s in spaces} == {"weights", "filler"}
    assert all(abs(s.deficit - eps) < 1e-12 and s.accounted == int(s.declared * (1 - eps)) for s in spaces)


def test_filler_depends_on_the_key_only_the_challenger_holds(ch):
    other = Challenger(CFG, b"other key", "gpu0", weights=WEIGHTS, filler_sectors=64)
    assert not np.array_equal(ch.filler, other.filler)


def test_dropped_sectors_fail_the_mac(ch):
    dropped = {("filler", s) for s in range(32)}
    assert "rejected" in run(ch, device(regions(ch), dropped=dropped))
    assert all(s.accounted == 0 for s in ch.fold(run(ch, device(regions(ch), dropped=dropped))))


def test_offloaded_sectors_miss_the_deadline(ch):
    offloaded = {("weights", s) for s in range(len(regions(ch)["weights"]))}
    assert set(run(ch, device(regions(ch), offloaded=offloaded))) == {"late"}


def test_a_wrong_aggregate_is_rejected(ch):
    c = ch.challenge(0, 0.0)
    good = respond(CFG, device(regions(ch)), c)
    bad = Response({k: (v[0] ^ 1, *v[1:]) for k, v in good.mu.items()}, good.elapsed)
    assert ch.check(c, good) == "accepted" and ch.check(c, bad) == "rejected"
