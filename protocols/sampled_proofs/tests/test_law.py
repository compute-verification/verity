"""The two-stage law: exact stages, beacon-keyed draws addressed by replay-unit index, and the integrity profile it implies."""

from __future__ import annotations

from fractions import Fraction

import pytest

from verity.proofs.profile import Bernoulli, IntegrityProfile, Subset
from verity_sampled_proofs.law import (CHECK, REPLAY, OwnRandomness, ReplayUnit, Stage, TwoStageLaw, check_probability, ru_key,
                                       select_replay_units, select_verification_units, vu_key)

BIND = {"program": b"\x01" * 32, "boundary_root": b"\x02" * 32}
LAW = TwoStageLaw({"decode": Stage(Fraction(1, 4), 2), "prefill": Stage(Fraction(1), None), "off": Stage(Fraction(0), 3)})


def _units(n: int, cls: str = "decode", n_v: int = 6) -> list[ReplayUnit]:
    return [ReplayUnit(i, cls, n_v) for i in range(n)]


def test_stages_check_exactly_k_or_everything():
    assert Stage(Fraction(1, 2), 3).checked(10) == 3 and Stage(Fraction(1, 2), 3).checked(2) == 2
    assert Stage(Fraction(1, 2)).checked(7) == 7
    for bad in (dict(p=Fraction(-1)), dict(p=Fraction(2)), dict(p=Fraction(1), k=-1)):
        with pytest.raises(ValueError):
            Stage(**bad)


def test_first_stage_endpoints_and_rate():
    key = ru_key(b"round-1", BIND)
    assert select_replay_units(key, LAW, _units(50, "prefill")) == _units(50, "prefill")
    assert select_replay_units(key, LAW, _units(50, "off")) == []
    picked = select_replay_units(key, LAW, _units(8000))
    assert abs(len(picked) - 2000) < 5 * (8000 * 3 / 16) ** 0.5


def test_second_stage_checks_exactly_k_distinct_positions():
    key = vu_key(b"round-2", {**BIND, "interiors_root": b"\x03" * 32})
    units = [ReplayUnit(0, "decode", 6), ReplayUnit(1, "decode", 1), ReplayUnit(2, "prefill", 5), ReplayUnit(3, "off", 2)]
    vus = select_verification_units(key, LAW, units)
    assert len(vus[0]) == 2 and len(set(vus[0])) == 2 and all(0 <= v < 6 for v in vus[0])
    assert vus[1] == (0,) and vus[2] == (0, 1, 2, 3, 4) and vus[3] == (0, 1)


def test_draws_are_sharded_by_replay_unit_index():
    key = ru_key(b"round-1", BIND)
    units = _units(1000)
    whole = {u.index for u in select_replay_units(key, LAW, units)}
    shards = [{u.index for u in select_replay_units(key, LAW, units[s::3])} for s in range(3)]
    assert set().union(*shards) == whole
    vkey = vu_key(b"round-2", BIND)
    assert select_verification_units(vkey, LAW, units[:10])[7] == select_verification_units(vkey, LAW, units[7:8])[7]


def test_draws_follow_their_bindings_and_rounds():
    units = _units(200)
    base = select_replay_units(ru_key(b"round-1", BIND), LAW, units)
    assert base != select_replay_units(ru_key(b"round-1", {**BIND, "boundary_root": b"\x09" * 32}), LAW, units)
    assert base != select_replay_units(ru_key(b"round-9", BIND), LAW, units)
    assert ru_key(b"r", BIND) != vu_key(b"r", BIND)
    for make in (ru_key, vu_key):
        with pytest.raises(ValueError):
            make(b"r", {})


def test_indices_must_be_distinct_and_classes_known():
    key = ru_key(b"r", BIND)
    with pytest.raises(ValueError):
        select_replay_units(key, LAW, [ReplayUnit(0, "decode", 2), ReplayUnit(0, "decode", 2)])
    with pytest.raises(KeyError, match="no stage"):
        select_replay_units(key, LAW, [ReplayUnit(0, "nope", 2)])


def test_detection_matches_the_profile_law():
    """Empirically, an RU with m incorrect VUs escapes both stages at the rate IntegrityProfile.accept(m) states."""
    law = TwoStageLaw({"u": Stage(Fraction(1, 2), 2)})
    profile = law.profile("u", n_r=1, n_v=4, delta=0.01)
    unit = [ReplayUnit(0, "u", 4)]
    trials = 4000
    for m in range(5):
        escaped = 0
        for t in range(trials):
            bind = {"root": t}
            if not select_replay_units(ru_key(b"ru", bind), law, unit):
                escaped += 1
            elif not set(select_verification_units(vu_key(b"vu", bind), law, unit)[0]) & set(range(m)):
                escaped += 1
        a = float(profile.accept(m))
        assert abs(escaped / trials - a) < 5 * (a * (1 - a) / trials) ** 0.5 + 1e-9


def test_bernoulli_is_the_single_unit_law():
    law = TwoStageLaw.bernoulli({"x": Fraction(1, 10)})
    assert law.stage("x") == Stage(Fraction(1, 10), 1)
    assert check_probability(law, "x", 1) == Fraction(1, 10)
    assert check_probability(LAW, "decode", 6) == Fraction(1, 4) * Fraction(2, 6)
    a = law.profile("x", n_r=100, n_v=1, delta=0.01)
    assert (a.sizes[1], a.draws[CHECK].k, a.draws[REPLAY].p) == (1, 1, Fraction(1, 10))


def test_profile_carries_the_class_law_and_covered_families():
    a = LAW.profile("decode", n_r=512, n_v=6, delta=0.01, covered=["Input"])
    assert (*a.sizes, a.draws[REPLAY].p, a.draws[CHECK].k, a.delta, a.covered) == (512, 6, Fraction(1, 4), 2, 0.01,
                                                                                  frozenset({"Input"}))
    assert LAW.profile("prefill", n_r=3, n_v=5, delta=0.01).draws[CHECK].k == 5
    assert LAW.to_json()["decode"] == {"p": "1/4", "k": 2}


def test_the_profile_is_two_named_levels_and_keeps_the_prototype_numbers():
    """The protocol prototype's composed toy through the law: 26.6 incorrect VUs for PoUW, 16.0 touched RUs for KV."""
    a = TwoStageLaw({"u": Stage(Fraction(1, 2), 2)}).profile("u", n_r=512, n_v=4, delta=0.01, drawn={REPLAY: 3, CHECK: 6})
    assert a == IntegrityProfile((REPLAY, CHECK), (512, 4), {REPLAY: Bernoulli(Fraction(1, 2)), CHECK: Subset(2)}, 0.01,
                                 drawn={REPLAY: 3, CHECK: 6})
    assert round(a.worst_case(lambda m: m), 1) == 26.6 and round(a.worst_case(lambda m: 1), 1) == 16.0


def _stream(b: bytes):
    at = [0]

    def read(n: int) -> bytes:
        out = b[at[0]:at[0] + n]
        at[0] += n
        return out
    return read


def _ref_uniform(n, stream, pos):
    """The Lean verifier's reference on a raw stream (backends/flock/tests/test_lean_verifier.py)."""
    w = (n.bit_length() + 64 + 7) // 8
    limit = (1 << 8 * w) - (1 << 8 * w) % n
    while pos + w <= len(stream):
        x = int.from_bytes(stream[pos:pos + w], "big")
        pos += w
        if x < limit:
            return x % n, pos
    return None


def test_own_randomness_is_the_lean_verifiers_samplers_on_raw_bytes():
    import random
    rng = random.Random(7)
    for case in range(200):
        n, k = rng.randint(1, 5000), None
        k = rng.randint(0, min(n, 20))
        stream = bytearray(rng.randbytes(2000))
        if case % 4 == 0:
            stream[:16] = b"\xff" * 16
        swapped, pos = {}, 0
        for i in range(k):
            j, pos = _ref_uniform(n - i, bytes(stream), pos)
            j += i
            swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)
        want = tuple(sorted(swapped.get(i, i) for i in range(k)))
        assert OwnRandomness(_stream(bytes(stream))).subset(k, n) == want
        p = Fraction(rng.randint(1, 9), 10)
        got = OwnRandomness(_stream(bytes(stream))).bernoulli(p)
        assert got == (_ref_uniform(p.denominator, bytes(stream), 0)[0] < p.numerator)
    assert OwnRandomness(_stream(b"")).bernoulli(Fraction(1)) is True
    with pytest.raises(ValueError):
        OwnRandomness(_stream(b"\x00" * 3)).uniform(10)


def test_the_stages_draw_from_the_verifiers_own_randomness():
    units = _units(8000)
    picked = select_replay_units(OwnRandomness(), LAW, units)
    assert abs(len(picked) - 2000) < 5 * (8000 * 3 / 16) ** 0.5
    vus = select_verification_units(OwnRandomness(), LAW, [ReplayUnit(0, "decode", 6), ReplayUnit(1, "prefill", 5)])
    assert len(vus[0]) == 2 and len(set(vus[0])) == 2 and vus[1] == (0, 1, 2, 3, 4)
