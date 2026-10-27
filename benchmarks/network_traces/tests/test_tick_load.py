"""tick_load's secret-bit stub without the clock: its slots on a grid the test gives (bit 1's lead before the bucket's
boundary, bit 0's half a bucket more), each token's bucket the first slot at or after it, and `distinguish` on offsets
the test makes: offsets that follow the bit are told apart, offsets that don't are not."""

from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import tick_load as tl  # noqa: E402

TAU = 0.1


def _bits() -> tl.SecretBits:
    bits = tl.SecretBits(Path("unused"), bytes(range(64)))
    bits.grid = (5.0, TAU)
    return bits


def test_a_slot_is_its_lead_before_its_boundary_or_half_a_bucket_more():
    bits = _bits()
    seen = set()
    for t in range(64):
        bit, lead = bits.draw(t)
        seen.add(bit)
        assert 0 <= lead < tl.LEAD_MS / 1000
        assert abs(bits.slot(t) - (5.0 + t * TAU - lead - (0 if bit else TAU / 2))) < 1e-12
    assert seen == {0, 1}
    assert [bits.slot(t) for t in range(64)] == sorted(bits.slot(t) for t in range(64))


def test_a_token_waits_for_the_first_slot_after_it():
    bits = _bits()
    for k in range(400):
        due = 5.0 + k * 0.0137
        t = bits.bucket(due)
        assert bits.slot(t) >= due
        assert t == 0 or bits.slot(t - 1) < due
        assert bits.slot(t) - due < 1.5 * TAU + tl.LEAD_MS / 1000


def test_offsets_that_follow_the_bit_are_told_apart():
    rng = random.Random(1)
    pairs = [(rng.gauss(1.0 + 0.8 * b, 0.3), b) for b in (rng.randrange(2) for _ in range(400))]
    d = tl.distinguish(pairs)
    assert d["mann_whitney"]["z"] > 10 and d["mann_whitney"]["p"] < 1e-6
    assert d["decoder"]["side"] == 1 and d["decoder"]["accuracy"] > 0.85
    assert d["information"]["p"] == 1 / (1 + tl.SHUFFLES)


def test_offsets_that_ignore_the_bit_are_not():
    rng = random.Random(2)
    pairs = [(rng.gauss(1.0, 0.3), rng.randrange(2)) for _ in range(400)]
    d = tl.distinguish(pairs)
    assert abs(d["mann_whitney"]["z"]) < 3
    assert d["decoder"]["accuracy"] < 0.62
    assert d["information"]["p"] > 0.01


def test_too_few_of_a_bit_reports_counts_alone():
    d = tl.distinguish([(1.0, 0), (2.0, 0), (3.0, 1)])
    assert d["by_bit"]["1"]["n"] == 1 and "decoder" not in d
