"""The two-stage law: exact stages, beacon-keyed draws addressed by replay-unit index, and the allowance it implies."""

from __future__ import annotations

from fractions import Fraction

import pytest

from verity_sampled_proofs.law import (ReplayUnit, Stage, TwoStageLaw, check_probability, ru_key, select_replay_units,
                                       select_verification_units, vu_key)

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


def test_detection_matches_the_allowance_law():
    """Empirically, an RU with m incorrect VUs escapes both stages at the rate IntegrityAllowance.accept(m) states."""
    law = TwoStageLaw({"u": Stage(Fraction(1, 2), 2)})
    allowance = law.allowance("u", n_r=1, n_v=4, delta=0.01)
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
        a = float(allowance.accept(m))
        assert abs(escaped / trials - a) < 5 * (a * (1 - a) / trials) ** 0.5 + 1e-9


def test_bernoulli_is_the_single_unit_law():
    law = TwoStageLaw.bernoulli({"x": Fraction(1, 10)})
    assert law.stage("x") == Stage(Fraction(1, 10), 1)
    assert check_probability(law, "x", 1) == Fraction(1, 10)
    assert check_probability(LAW, "decode", 6) == Fraction(1, 4) * Fraction(2, 6)
    a = law.allowance("x", n_r=100, n_v=1, delta=0.01)
    assert (a.n_v, a.k, a.p) == (1, 1, Fraction(1, 10))


def test_allowance_carries_the_class_law():
    a = LAW.allowance("decode", n_r=512, n_v=6, delta=0.01)
    assert (a.n_r, a.n_v, a.p, a.k, a.delta) == (512, 6, Fraction(1, 4), 2, 0.01)
    assert LAW.allowance("prefill", n_r=3, n_v=5, delta=0.01).k == 5
    assert LAW.to_json()["decode"] == {"p": "1/4", "k": 2}
