"""The two-stage law: exact stages, draws from the verifier's own randomness (and LEGACY's derived keys) addressed by
replay-unit index, and the integrity profile it implies."""

from __future__ import annotations

from fractions import Fraction
from math import comb

import pytest

from verity.primitives.circuits.partition import Level
from verity.protocols.profile import Bernoulli, IntegrityProfile
from verity_experimental.sampled_proofs.law import (CHECK, REPLAY, OwnRandomness, ReplayUnit, Stage, TwoStageLaw, check_probability, ru_key,
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


def test_detection_matches_the_per_ru_law_and_the_profile_bounds_it():
    """Empirically, an RU with m incorrect VUs escapes both stages at 1 - p + p C(n_v - m, k) / C(n_v, k); a wrong RU
    (m >= 1) escapes at most at the profile's accept(1) = 1 - p k / n_v, with equality at m = 1."""
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
        a = float(1 - Fraction(1, 2) + Fraction(1, 2) * Fraction(comb(4 - m, 2), comb(4, 2)))
        assert abs(escaped / trials - a) < 5 * (a * (1 - a) / trials) ** 0.5 + 1e-9
        if m:
            assert a <= float(profile.accept(1)) and (m > 1 or a == float(profile.accept(1)))


def test_bernoulli_is_the_single_unit_law():
    law = TwoStageLaw.bernoulli({"x": Fraction(1, 10)})
    assert law.stage("x") == Stage(Fraction(1, 10), 1)
    assert check_probability(law, "x", 1) == Fraction(1, 10)
    assert check_probability(LAW, "decode", 6) == Fraction(1, 4) * Fraction(2, 6)
    a = law.profile("x", n_r=100, n_v=1, delta=0.01)
    assert (a.levels, a.sizes, a.draws[REPLAY].p) == ((Level(REPLAY),), (100,), Fraction(1, 10))


def test_profile_carries_the_class_law_and_covered_families():
    a = LAW.profile("decode", n_r=512, n_v=6, delta=0.01, covered=["Input"])
    assert (*a.sizes, a.draws[REPLAY].p, a.delta, a.covered) == (512, Fraction(1, 4) * Fraction(2, 6), 0.01, frozenset({"Input"}))
    assert LAW.profile("prefill", n_r=3, n_v=5, delta=0.01).draws[REPLAY].p == 1
    assert LAW.to_json()["decode"] == {"p": "1/4", "k": 2}


TOY = TwoStageLaw({"u": Stage(Fraction(1, 2), 2)})


def test_toy_attack_is_admitted():
    """docs/audit-protocols.md §2.9 and §3.4 row 6, at the prototype's TOY (512 RUs of 4 VUs, p = 1/2, k = 2, delta = 0.01).
    The interiors are committed after the RU draw, so a prover skipping 16 whole RUs (64 VUs of work) replays each drawn
    one honestly but for one wrong VU: each skipped RU escapes with probability 1 - p k / n_v = 3/4, and all 16 with
    0.75^16 >= 0.01.  The profile must give a fully skipped RU at least that escape, and certify the 64 skipped VUs."""
    a = TOY.profile("u", n_r=512, n_v=4, delta=0.01)
    achieved = 1 - Fraction(1, 2) * Fraction(2, 4)
    assert achieved ** 16 >= Fraction(1, 100)
    skipped_ru = a.accept(a.count(a.levels[-1].name, a.levels[0].name))   # every checked member of one RU wrong
    assert skipped_ru >= achieved
    assert a.worst_case(lambda m: 4 * m) >= 64


def test_the_profile_is_over_replay_units():
    """TOY's profile: one level of RUs at rate p k / n_v = 1/4; at delta = 0.01 it admits 16.0 wrong (and touched) RUs,
    so 64 VUs of skipped work, where the old two-level profile certified 26.6."""
    a = TOY.profile("u", n_r=512, n_v=4, delta=0.01, drawn={REPLAY: 3})
    assert a == IntegrityProfile((REPLAY,), (512,), {REPLAY: Bernoulli(Fraction(1, 4))}, 0.01, drawn={REPLAY: 3})
    assert round(a.worst_case(lambda m: m), 1) == 16.0 and round(a.worst_case(lambda m: 1), 1) == 16.0
    assert round(a.worst_case(lambda m: 4 * m), 1) == 64.0
    with pytest.raises(ValueError):
        TOY.profile("u", n_r=512, n_v=4, delta=0.01, drawn={REPLAY: 3, CHECK: 6})
    with pytest.raises(ValueError):
        TOY.profile("u", n_r=512, n_v=0, delta=0.01)


def test_a_mixed_class_is_profiled_at_its_largest_replay_unit():
    """red-team-flock-3's C1 on #121: p = 1/2, k = 2, delta = 0.01, a class of 4-VU and 8-VU RUs.  An 8-VU RU escapes at
    1 - 1/8, so an adaptive prover leaves 34 of them wrong and passes with probability (7/8)^34 >= 0.01; profiled at
    n_v = 4 the class would admit only 16.  From its RUs the profile takes n_v = 8 and certifies 34.49."""
    units = [ReplayUnit(i, "u", 4) for i in range(200)] + [ReplayUnit(200 + i, "u", 8) for i in range(200)]
    units.append(ReplayUnit(1000, "other", 64))
    assert (1 - Fraction(1, 2) * Fraction(2, 8)) ** 34 >= Fraction(1, 100)
    a = TOY.profile("u", units, delta=0.01)
    assert a.sizes == (400,) and a.draws[REPLAY].p == Fraction(1, 8)
    assert a.worst_case(lambda m: m) >= 34 and round(a.worst_case(lambda m: m), 2) == 34.49
    assert a == TOY.profile("u", units, n_r=400, n_v=8, delta=0.01)
    assert TOY.profile("u", units, n_v=16, delta=0.01).draws[REPLAY].p == Fraction(1, 16)
    for bad in (dict(n_v=4), dict(n_r=399)):
        with pytest.raises(ValueError):
            TOY.profile("u", units, delta=0.01, **bad)
    with pytest.raises(ValueError):
        TOY.profile("u", [ReplayUnit(0, "other", 4)], delta=0.01)
    with pytest.raises(ValueError):
        TOY.profile("u", delta=0.01)


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
