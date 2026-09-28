"""The two-stage law: exact stages, a first stage that is a probability per class or any one-stage law over the replay
units, draws from the verifier's own randomness, and the integrity profile it implies."""

from __future__ import annotations

import random
from fractions import Fraction
from math import comb

import pytest

from verity.ml.circuits.partition import Level
from verity.core.protocols.profile import Bernoulli, IntegrityProfile, Stratified, Stratum, Subset
from verity.core.protocols.sampled_proofs.one_stage import audit, draw as D
from verity.core.protocols.sampled_proofs.two_stage import (REPLAY, OwnRandomness, ReplayUnit, Stage, TwoStageLaw,
                                                                    check_probability, closure_of, draw_first,
                                                                    select_proof_units, select_replay_units)

LAW = TwoStageLaw({"decode": Stage(Fraction(1, 4), 2), "prefill": Stage(Fraction(1), None), "off": Stage(Fraction(0), 3)})


def _units(n: int, cls: str = "decode", n_v: int = 6) -> list[ReplayUnit]:
    return [ReplayUnit(i, cls, n_v) for i in range(n)]


def _seeded(seed: int) -> OwnRandomness:
    return OwnRandomness(random.Random(seed).randbytes)


def _stream(b: bytes):
    at = [0]

    def read(n: int) -> bytes:
        out = b[at[0]:at[0] + n]
        at[0] += n
        return out
    return read


def _stratified(k: int = 4) -> dict:
    """A stratified law over 10 RUs: stratum A the RUs [0, 4) and [7, 10), stratum B [4, 7)."""
    strata = [("A", [[0, 4], [7, 10]]), ("B", [[4, 7]])]
    return {"law": "stratified", "k": k,
            "strata": [{"template": t, "units": rs, "k": D.stratum_k(k, sum(h - l for l, h in rs), 10)} for t, rs in strata]}


def _work(k: int = 4) -> dict:
    """A work law over the same strata: A's RUs do work 3, B's none (drawn at their floor)."""
    strata = [("A", [[0, 4], [7, 10]], 3), ("B", [[4, 7]], 0)]
    total = sum(w * sum(h - l for l, h in rs) for _, rs, w in strata)
    return {"law": "work", "k": k, "strata": [{"template": t, "units": rs, "work": w, "floor": 1,
                                               "k": D.work_k(k, w, sum(h - l for l, h in rs), total, 1)}
                                              for t, rs, w in strata]}


def test_stages_check_exactly_k_or_everything():
    assert Stage(Fraction(1, 2), 3).checked(10) == 3 and Stage(Fraction(1, 2), 3).checked(2) == 2
    assert Stage(Fraction(1, 2)).checked(7) == 7
    for bad in (dict(first=Fraction(-1)), dict(first=Fraction(2)), dict(first=Fraction(1), k=-1)):
        with pytest.raises(ValueError):
            Stage(**bad)


def test_a_first_stage_is_any_one_stage_law_and_checks_all_unless_bernoulli():
    """A subset, stratified or work first stage takes k = None: a second draw thinning it would leave an escape (the
    expectation of (1 - k / n_v)^X over the first draw's count X) that no profile law states.  A bernoulli law thinned stays
    Bernoulli, so it takes any k."""
    for law in ({"law": "subset", "k": 3}, _stratified(), _work()):
        assert Stage(law).first == law and Stage(law).p is None and Stage(law).to_json() == {"law": law, "k": None}
        with pytest.raises(ValueError, match="checks every proof unit"):
            Stage(law, 2)
    assert Stage({"law": "bernoulli", "p": "2/8"}, 2).p == Fraction(1, 4)
    for bad in ({"law": "subset"}, {"law": "subset", "k": -1}, {"law": "subset", "k": True}, {"law": "subset", "k": 1, "p": "1/2"},
                {"law": "bernoulli", "p": "3/2"}, {"law": "bernoulli", "p": "1"}, {"law": "stratified", "k": 4},
                {"law": "work", "k": 4, "strata": "A"}, {"law": "uniform", "k": 1}):
        with pytest.raises(ValueError):
            Stage(bad)


def test_first_stage_endpoints_and_rate():
    assert select_replay_units(_seeded(1), LAW, _units(50, "prefill")) == _units(50, "prefill")
    assert select_replay_units(OwnRandomness(_stream(b"")), LAW, _units(50, "off")) == []      # p = 0 reads nothing
    picked = select_replay_units(_seeded(2), LAW, _units(8000))
    assert abs(len(picked) - 2000) < 5 * (8000 * 3 / 16) ** 0.5


def test_second_stage_checks_exactly_k_distinct_positions_and_all_draws_nothing():
    units = [ReplayUnit(0, "decode", 6), ReplayUnit(1, "decode", 1), ReplayUnit(2, "prefill", 5), ReplayUnit(3, "off", 2)]
    picks = select_proof_units(_seeded(3), LAW, units)
    assert len(picks[0]) == 2 and len(set(picks[0])) == 2 and all(0 <= v < 6 for v in picks[0])
    assert picks[1] == (0,) and picks[2] == (0, 1, 2, 3, 4) and picks[3] == (0, 1)
    assert select_proof_units(OwnRandomness(_stream(b"")), LAW, units[1:]) == {1: (0,), 2: (0, 1, 2, 3, 4), 3: (0, 1)}


def test_indices_must_be_distinct_and_classes_known():
    with pytest.raises(ValueError):
        select_replay_units(_seeded(4), LAW, [ReplayUnit(0, "decode", 2), ReplayUnit(0, "decode", 2)])
    with pytest.raises(KeyError, match="no stage"):
        select_replay_units(_seeded(4), LAW, [ReplayUnit(0, "nope", 2)])


def test_detection_matches_the_per_ru_law_and_the_profile_bounds_it():
    """An RU with m incorrect proof units escapes both stages at 1 - p + p C(n_v - m, k) / C(n_v, k), measured over the
    verifier's own draws from a seeded source; a wrong RU (m >= 1) escapes at most at the profile's accept(1) =
    1 - p k / n_v, with equality at m = 1."""
    law = TwoStageLaw({"u": Stage(Fraction(1, 2), 2)})
    profile = law.profile("u", n_r=1, n_v=4, delta=0.01)
    unit = [ReplayUnit(0, "u", 4)]
    trials, rng = 4000, _seeded(5)
    for m in range(5):
        escaped = 0
        for _ in range(trials):
            if not select_replay_units(rng, law, unit) or not set(select_proof_units(rng, law, unit)[0]) & set(range(m)):
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
    with pytest.raises(ValueError, match="depends on the population"):
        check_probability(TwoStageLaw({"t": Stage({"law": "subset", "k": 2})}), "t", 4)


def test_profile_carries_the_class_law_and_covered_families():
    a = LAW.profile("decode", n_r=512, n_v=6, delta=0.01, covered=["Input"])
    assert (*a.sizes, a.draws[REPLAY].p, a.delta, a.covered) == (512, Fraction(1, 4) * Fraction(2, 6), 0.01, frozenset({"Input"}))
    assert LAW.profile("prefill", n_r=3, n_v=5, delta=0.01).draws[REPLAY].p == 1
    assert LAW.to_json()["decode"] == {"p": "1/4", "k": 2}


TOY = TwoStageLaw({"u": Stage(Fraction(1, 2), 2)})


def test_toy_attack_is_admitted():
    """docs/audit-protocols.md §2.9 and §3.4 row 6, at the prototype's TOY (512 RUs of 4 proof units, p = 1/2, k = 2,
    delta = 0.01).  The interiors are committed after the RU draw, so a prover skipping 16 whole RUs (64 proof units of
    work) replays each drawn one honestly but for one wrong proof unit: each skipped RU escapes with probability
    1 - p k / n_v = 3/4, and all 16 with 0.75^16 >= 0.01.  The profile must give a fully skipped RU at least that escape,
    and certify the 64 skipped proof units."""
    a = TOY.profile("u", n_r=512, n_v=4, delta=0.01)
    achieved = 1 - Fraction(1, 2) * Fraction(2, 4)
    assert achieved ** 16 >= Fraction(1, 100)
    skipped_ru = a.accept(a.count(a.levels[-1].name, a.levels[0].name))   # every checked member of one RU wrong
    assert skipped_ru >= achieved
    assert a.worst_case(lambda m: 4 * m) >= 64


def test_the_profile_is_over_replay_units():
    """TOY's profile: one level of RUs at rate p k / n_v = 1/4; at delta = 0.01 it admits 16.0 wrong (and touched) RUs,
    so 64 proof units of skipped work, where the old two-level profile certified 26.6."""
    a = TOY.profile("u", n_r=512, n_v=4, delta=0.01, drawn={REPLAY: 3})
    assert a == IntegrityProfile((REPLAY,), (512,), {REPLAY: Bernoulli(Fraction(1, 4))}, 0.01, drawn={REPLAY: 3})
    assert round(a.worst_case(lambda m: m), 1) == 16.0 and round(a.worst_case(lambda m: 1), 1) == 16.0
    assert round(a.worst_case(lambda m: 4 * m), 1) == 64.0
    with pytest.raises(ValueError):
        TOY.profile("u", n_r=512, n_v=4, delta=0.01, drawn={REPLAY: 3, "check": 6})
    with pytest.raises(ValueError):
        TOY.profile("u", n_r=512, n_v=0, delta=0.01)


def test_a_mixed_class_is_profiled_at_its_largest_replay_unit():
    """red-team-flock-3's C1 on #121: p = 1/2, k = 2, delta = 0.01, a class of 4- and 8-proof-unit RUs.  An 8-unit RU
    escapes at 1 - 1/8, so an adaptive prover leaves 34 of them wrong and passes with probability (7/8)^34 >= 0.01;
    profiled at n_v = 4 the class would admit only 16.  From its RUs the profile takes n_v = 8 and certifies 34.49."""
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


def test_a_bernoulli_law_first_stage_is_the_probability_form():
    """A `bernoulli` law object first stage profiles as the probability it states, at the class's largest RU."""
    units = [ReplayUnit(i, "u", 4) for i in range(10)] + [ReplayUnit(10, "u", 8)]
    law = TwoStageLaw({"u": Stage({"law": "bernoulli", "p": "1/2"}, 2)})
    assert law.profile("u", units, delta=0.01) == TOY.profile("u", units, delta=0.01)


def test_a_one_stage_first_stage_profiles_as_the_one_stage_law_over_replay_units():
    """With every proof unit of a drawn RU checked, a wrong RU escapes exactly when stage 1 misses it (Lean
    `effEscape_full`), so the profile is the one-stage audit's for the same draw, over the RUs: `Subset(k)`, and for a
    stratified or work law core's `Stratified` over the strata at the drawn sizes."""
    reg = {"population": 10, "partition": "pp" * 64, "program": "aa" * 64}
    for law, want in (({"law": "subset", "k": 3}, Subset(3)),
                      (_stratified(), Stratified((Stratum("A", 7, 3), Stratum("B", 3, 1)))),
                      (_work(), Stratified((Stratum("A", 7, 4), Stratum("B", 3, 1))))):
        two = TwoStageLaw({"ru": Stage(law)})
        got = two.profile("ru", n_r=10, delta=0.01, drawn={REPLAY: 4})
        d = draw_first(_seeded(6), law, 10)
        one = audit.profile(reg, d, 0.01)
        assert got.draws[REPLAY] == want == one.draws[audit.LEVEL] and got.sizes == one.sizes == (10,)
        assert got.bound() == one.bound()
        assert two.profile("ru", _units(10, "ru", 5), delta=0.01).draws[REPLAY] == want      # n_v isn't read
    with pytest.raises(ValueError, match="exceeds"):
        TwoStageLaw({"ru": Stage({"law": "subset", "k": 11})}).profile("ru", n_r=10, delta=0.01)
    with pytest.raises(ValueError, match="tile"):
        TwoStageLaw({"ru": Stage(_stratified())}).profile("ru", n_r=11, delta=0.01)
    with pytest.raises(ValueError, match="n_r"):
        TwoStageLaw({"ru": Stage(_stratified())}).profile("ru", delta=0.01)


def test_a_subset_first_stage_escapes_at_its_profile():
    """Subset(3) of 10 RUs, every proof unit checked: m wrong RUs escape at C(10 - m, 3) / C(10, 3), the profile's
    accept(m), over the verifier's draws from a seeded source."""
    law = TwoStageLaw({"ru": Stage({"law": "subset", "k": 3})})
    a = law.profile("ru", n_r=10, delta=0.01)
    units, rng, trials = _units(10, "ru", 3), _seeded(7), 3000
    for m in (1, 2, 4):
        escaped = sum(not {u.index for u in select_replay_units(rng, law, units)} & set(range(m)) for _ in range(trials))
        want = float(a.accept(m))
        assert want == comb(10 - m, 3) / comb(10, 3)
        assert abs(escaped / trials - want) < 5 * (want * (1 - want) / trials) ** 0.5


def _uniform_on(n, src, pos):
    """The Lean verifier's `Flock.Draw.uniform` on a raw stream."""
    w = (n.bit_length() + 64 + 7) // 8
    limit = (1 << 8 * w) - (1 << 8 * w) % n
    while pos + w <= len(src):
        x = int.from_bytes(src[pos:pos + w], "big")
        pos += w
        if x < limit:
            return x % n, pos
    return None


def _subset_on(k, n, src):
    swapped, pos = {}, 0
    for i in range(k):
        r = _uniform_on(n - i, src, pos)
        if r is None:
            return None
        j, pos = i + r[0], r[1]
        swapped[i], swapped[j] = swapped.get(j, j), swapped.get(i, i)
    return sorted(swapped.get(i, i) for i in range(k))


def _draw_on(law, n, src):
    """The Lean verifier's `Flock.Draw.drawOn`: subset and bernoulli on the one stream; stratified and work on consecutive
    segments, stratum s's the next 2 k_s width(n_s) + 64 bytes (`verity/ml/flock/tests/stratified_agree.py`)."""
    if law["law"] == "subset":
        return _subset_on(law["k"], n, src)
    if law["law"] == "bernoulli":
        num, den = map(int, law["p"].split("/"))
        if num == 0:
            return []
        if num == den:
            return list(range(n))
        out, pos = [], 0
        for u in range(n):
            r = _uniform_on(den, src, pos)
            if r is None:
                return None
            out += [u] if r[0] < num else []
            pos = r[1]
        return out
    out, at = [], 0
    for s in law["strata"]:
        n_s = sum(hi - lo for lo, hi in s["units"])
        need = 2 * s["k"] * ((n_s.bit_length() + 64 + 7) // 8) + 64
        local = _subset_on(s["k"], n_s, src[at:at + need])
        if local is None:
            return None
        flat = [u for lo, hi in s["units"] for u in range(lo, hi)]
        out += [flat[j] for j in local]
        at += need
    return sorted(out)


def test_own_randomness_is_the_lean_verifiers_samplers_on_raw_bytes():
    rng = random.Random(7)
    for case in range(200):
        n = rng.randint(1, 5000)
        k = rng.randint(0, min(n, 20))
        stream = bytearray(rng.randbytes(2000))
        if case % 4 == 0:
            stream[:16] = b"\xff" * 16
        assert OwnRandomness(_stream(bytes(stream))).subset(k, n) == tuple(_subset_on(k, n, bytes(stream)))
        p = Fraction(rng.randint(1, 9), 10)
        got = OwnRandomness(_stream(bytes(stream))).bernoulli(p)
        assert got == (_uniform_on(p.denominator, bytes(stream), 0)[0] < p.numerator)
    assert OwnRandomness(_stream(b"")).bernoulli(Fraction(1)) is True
    with pytest.raises(ValueError):
        OwnRandomness(_stream(b"\x00" * 3)).uniform(10)


def test_draw_first_is_the_lean_verifiers_draw_on_the_same_bytes():
    """`draw_first` reads the verifier's bytes as `Flock.Draw.drawWith` does: on a stream that doesn't run out, its units
    are `drawOn`'s on the same bytes, for every law (a bernoulli law at its NUM/DEN as written, unreduced), and
    `one_stage.draw.check_draw` accepts the draw against the law and the population."""
    rng = random.Random(8)
    laws = [{"law": "subset", "k": 0}, {"law": "subset", "k": 4}, {"law": "subset", "k": 10}, {"law": "bernoulli", "p": "2/6"},
            {"law": "bernoulli", "p": "0/3"}, {"law": "bernoulli", "p": "3/3"}, _stratified(1), _stratified(4), _stratified(10),
            _work(1), _work(4), _work(10)]
    for case in range(120):
        law = laws[case % len(laws)]
        src = rng.randbytes(4096)
        d = draw_first(OwnRandomness(_stream(src)), law, 10)
        assert d["units"] == _draw_on(law, 10, src), (case, law)
        assert d == {**law, "population": 10, "units": d["units"]} and D.check_draw(d, {"population": 10}, law).ok
    big = {"law": "subset", "k": 300}
    src = rng.randbytes(8192)
    assert draw_first(OwnRandomness(_stream(src)), big, 1_000_000)["units"] == _draw_on(big, 1_000_000, src)


def test_a_stream_that_runs_out_is_extended_as_lean_does():
    """Lean's `onWith`: when the first `need` bytes can't finish the draw (here every word is rejected), `need` more are
    read and the sampler reruns from the start of the longer stream; a source that ends is refused."""
    need = 2 * 3 * 9 + 64
    src = b"\xff" * need + random.Random(9).randbytes(need)
    assert draw_first(OwnRandomness(_stream(src)), {"law": "subset", "k": 3}, 10)["units"] == _subset_on(3, 10, src)
    with pytest.raises(ValueError, match="ran out"):
        draw_first(OwnRandomness(_stream(b"\xff" * need)), {"law": "subset", "k": 3}, 10)


def test_a_work_draw_carries_its_closure_and_the_profile_counts_the_drawn_rus():
    """PoUW's shape: stage 1 a work law over tiles with the verifier's closure map (a tile's RU is the tile and what its
    credit reads), stage 2 all.  The draw carries `closure_of` (Lean `closureOf`: what drawn RUs' closures hold beyond
    the drawn RUs, ascending, omitted when empty), `one_stage.draw.proved` is drawn and closure, and the profile counts
    the drawn RUs only."""
    closure = {0: [1, 4], 2: [3], 4: [5, 0], 7: [8, 9], 9: [7]}
    assert closure_of(closure, [0, 4]) == [1, 5] and closure_of(closure, [7, 9]) == [8] and closure_of(closure, [6]) == []
    law = _work()
    for seed in range(20):
        d = draw_first(_seeded(seed), law, 10, closure)
        want = closure_of(closure, d["units"])
        assert d.get("closure", []) == want and ("closure" in d) == bool(want)
        assert D.check_draw(d, {"population": 10}, law).ok and D.proved(d) == sorted(d["units"] + want)
    two = TwoStageLaw({"tile": Stage(law)})
    a = two.profile("tile", n_r=10, delta=0.01, drawn={REPLAY: len(d["units"])})
    assert a.drawn[REPLAY] == 5 and a.draws[REPLAY] == audit.core_law(law, 10)
    for other in ({"law": "subset", "k": 3}, _stratified()):
        with pytest.raises(ValueError, match="work draw only"):
            draw_first(_seeded(0), other, 10, closure)


def test_a_one_stage_class_is_drawn_over_its_replay_units_in_index_order():
    """`select_replay_units` draws a one-stage class's RUs together, its law's positions the class's RUs in index order;
    a probability class beside it keeps its per-RU coins."""
    law = TwoStageLaw({"tile": Stage({"law": "subset", "k": 3}), "rest": Stage(Fraction(1))})
    tiles = [ReplayUnit(10 + 2 * i, "tile", 4) for i in range(8)]
    units = [x for pair in zip(tiles, _units(8, "rest", 1)) for x in pair]
    src = random.Random(10).randbytes(4096)
    got = select_replay_units(OwnRandomness(_stream(src)), law, units)
    assert [u for u in got if u.cls == "rest"] == _units(8, "rest", 1)
    assert [u.index for u in got if u.cls == "tile"] == [tiles[j].index for j in _subset_on(3, 8, src)]
    with pytest.raises(ValueError, match="exceeds"):
        select_replay_units(_seeded(0), law, tiles[:2])


def test_draw_first_refuses_what_lean_refuses():
    for law, n in (({"law": "subset", "k": 4}, 3), (_stratified(), 9), ({**_stratified(), "k": 5}, 10),
                   ({"law": "bernoulli", "p": "1/0"}, 10), ({"law": "uniform", "k": 1}, 10)):
        with pytest.raises(ValueError):
            draw_first(_seeded(0), law, n)


def test_the_stages_draw_from_the_verifiers_own_randomness():
    units = _units(8000)
    picked = select_replay_units(OwnRandomness(), LAW, units)
    assert abs(len(picked) - 2000) < 5 * (8000 * 3 / 16) ** 0.5
    picks = select_proof_units(OwnRandomness(), LAW, [ReplayUnit(0, "decode", 6), ReplayUnit(1, "prefill", 5)])
    assert len(picks[0]) == 2 and len(set(picks[0])) == 2 and picks[1] == (0, 1, 2, 3, 4)
    d = draw_first(OwnRandomness(), _work(), 10)
    assert D.check_draw(d, {"population": 10}, _work()).ok
