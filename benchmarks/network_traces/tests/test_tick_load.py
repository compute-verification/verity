"""tick_load's secret-bit stub without the clock: its slots on a grid the test gives (bit 1's lead before the bucket's
boundary, bit 0's half a bucket more), each token's bucket the first slot at or after it, and `distinguish` on offsets
the test makes: offsets that follow the bit are told apart, offsets that don't are not."""

from __future__ import annotations

import gzip
import json
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


def test_jitter_reads_the_pacers_write_starts_and_pairs_each_offset_with_the_next_bit(tmp_path):
    """A paced run's pacer reports are each bucket's write start, and `next` pairs each bucket's client offset with
    the next bucket's bit over the buckets whose next was written in: offsets that follow the next bit are decoded."""
    bits, n = _bits(), 96
    bits.writes = [[t, 0.0] for t in range(n) if t % 3]
    (tmp_path / "client").mkdir()
    lag = [50.0 + bits.draw(t + 1)[0] for t in range(n)]
    (tmp_path / "client" / "peer.json").write_text(json.dumps({"bucket_lag_ms": lag}))
    log = {"lateness_ms": {"egress": [0.1] * n},
           "paced": {"ms": 50.0, "releases": [[t, -40.0, t / 100, 0.2, True] for t in range(n)]}}
    with gzip.open(tmp_path / "proxy.json.gz", "wt") as fh:
        json.dump(log, fh)
    j = tl.jitter(tmp_path, bits)
    assert j["paced_ms"] == 50.0 and [r[6] for r in j["buckets"]] == [t / 100 for t in range(n)]
    assert j["next"]["client"]["n"] == sum(1 for t in range(n - 1) if (t + 1) % 3)
    assert j["next"]["client"]["decoder"]["accuracy"] == 1.0 and "release" in j["written"]


def test_summary_lists_each_late_paced_bucket_with_its_end_and_the_previous_ones(tmp_path):
    """A paced run's summary lists the buckets that reached the pacer after their time, started past the slack or
    weren't written, each with its write's end and the previous bucket's, and the ends' tail."""
    releases = [[t, -40.0, 0.0, 0.1, True] for t in range(8)]
    releases[3], releases[5], releases[6] = [3, 2.0, 2.0, 2.1, True], [5, -40.0, 30.0, 30.1, True], \
        [6, -40.0, 0.0, 0.1, False]
    releases[2][3] = 61.0
    log = {"slack_ms": 25.0, "lateness_ms": {"egress": [0.1] * 8}, "paced": {"ms": 50.0, "releases": releases},
           "forwards": [[0, -3.0], [1, 4.5], [1, None]]}
    with gzip.open(tmp_path / "proxy.json.gz", "wt") as fh:
        json.dump(log, fh)
    (tmp_path / "calibration.json").write_text(json.dumps({"grid": {"window_buckets": 4}}))
    s = tl.summary(tmp_path)
    p = s["paced"]
    assert p["late"] == [[3, 0, 3, 2.0, 2.0, 2.1, 61.0], [5, 1, 1, -40.0, 30.0, 30.1, 0.1],
                         [6, 1, 2, -40.0, 0.0, 0.1, 30.1]]
    assert p["end_max_ms"] == 61.0 and p["buckets"] == 8
    assert s["forwards_late"] == [[1, 4.5], [1, None]]
