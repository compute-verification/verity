"""`draw.meets_rate` and `draw.cheapest_for`: today's served rules meet the rate at amount 1, a passing design's exact miss
probability is within the rate's bound for every set of proof blocks, the cheapest strata are the least that pass, and
how padding moves the verdict."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from math import comb
from pathlib import Path

import pytest

from verity.commitments.rate import rate_count
from verity.protocols.proofs.spot.verifier.one_stage import draw as DR

CARD = Path(__file__).resolve().parents[6] / "benchmarks/spot_figures/cards/workload.glm-5.3.json"


def population(sizes: list[int]) -> list[tuple[str, list[list[int]]]]:
    out, at = [], 0
    for i, n in enumerate(sizes):
        out.append((f"t{i}", [[at, at + n]]))
        at += n
    return out


def stratified(pop, ks: list[int], law: str = "stratified") -> dict:
    return {"law": law, "k": 1, "strata": [{"template": t, "units": rs, "k": k} for (t, rs), k in zip(pop, ks)]}


def passes(num: int, den: int, table: dict, pop, design: dict) -> bool:
    try:
        DR.meets_rate(num, den, DR.work_table(table), pop, design)
    except ValueError:
        return False
    return True


def served_strata() -> list[int]:
    card = json.loads(CARD.read_text())["value"]
    return [round(s["per_token"] * card["tokens"]) for s in card["strata"]]


@pytest.mark.parametrize("rule", ["stratified-rate:1/2097152", "rate:1/3481648"])
def test_todays_served_rules_meet_their_rate_at_amount_one(rule):
    sizes = served_strata()
    pop, n = population(sizes), sum(sizes)
    table = {t: 1 for t, _ in pop}
    if rule.startswith("stratified-rate:"):
        k = DR.strat_k(1, 2097152, sizes, n)
        design, num, den = stratified(pop, [DR.stratum_k(k, s, n) for s in sizes]), 1, 2097152
    else:
        design, num, den = {"law": "subset", "k": rate_count(1, 3481648, 1, n)}, 1, 3481648
    DR.meets_rate(num, den, DR.work_table(table), pop, design)
    with pytest.raises(ValueError, match="misses the rate"):
        DR.meets_rate(num, den, DR.work_table({t: 2 for t in table}), pop, design)


def test_a_stratified_law_is_agreed_only_at_its_strata_sum():
    pop = population([3, 3])
    mine = stratified(pop, [2, 2])

    def agreed(ks: list[int], k: int) -> dict:
        law = {**stratified(pop, ks), "k": k}
        return DR.rate_agreed(1, 2, pop, mine, law)

    assert agreed([2, 2], 4)["k"] == 4 and agreed([3, 2], 5)["k"] == 5
    for ks, k in (([2, 2], 5), ([2, 2], 3), ([3, 2], 6), ([3, 2], 4)):
        with pytest.raises(ValueError) as refused:
            agreed(ks, k)
        assert str(refused.value) == f"rate: the stratified law's k = {k} is not the sum of its strata's k_s = {sum(ks)}"


def _miss(kind: str, sizes, ks, p: Fraction, ts) -> Fraction:
    """The exact probability that the design samples none of a set with `ts[s]` proof blocks in stratum `s`."""
    if kind == "stratified":
        out = Fraction(1)
        for n, k, t in zip(sizes, ks, ts):
            out *= Fraction(comb(n - t, k), comb(n, k))
        return out
    if kind == "subset":
        return Fraction(comb(sum(sizes) - sum(ts), ks), comb(sum(sizes), ks))
    return (1 - p) ** sum(ts)


def test_a_passing_design_misses_every_set_within_the_rates_bound():
    """For every set of proof blocks of every passing design, miss <= prod_s (1 - c_s NUM/DEN)^t_s, the rational form of
    exp(-amount NUM/DEN); a stratum drawn whole contributes 0 to a set that meets it."""
    rng = random.Random(20261119)
    checked = {"stratified": 0, "subset": 0, "bernoulli": 0}
    for _ in range(400):
        sizes = [rng.randint(1, 4) for _ in range(rng.randint(1, 3))]
        amounts = [rng.randint(0, 4) for _ in sizes]
        num, den = rng.randint(1, 3), rng.randint(1, 12)
        pop = population(sizes)
        table = {t: c for (t, _), c in zip(pop, amounts)}
        kind = rng.choice(list(checked))
        ks, p = [rng.randint(0, n) for n in sizes], Fraction(0)
        if kind == "stratified":
            design = stratified(pop, ks)
        elif kind == "subset":
            ks = rng.randint(0, sum(sizes))
            design = {"law": "subset", "k": ks}
        else:
            b = rng.randint(1, 8)
            p = Fraction(rng.randint(0, b), b)
            design = {"law": "bernoulli", "p": f"{p.numerator}/{p.denominator}"}
        if not passes(num, den, table, pop, design):
            continue
        checked[kind] += 1
        q = Fraction(num, den)
        units = [s for s, n in enumerate(sizes) for _ in range(n)]
        for mask in range(1 << len(units)):
            ts = [0] * len(sizes)
            for i, s in enumerate(units):
                if mask >> i & 1:
                    ts[s] += 1
            bound = Fraction(1)
            for s, t in enumerate(ts):
                whole = (kind == "stratified" and ks[s] == sizes[s] or kind == "subset" and ks == sum(sizes)
                         or kind == "bernoulli" and p == 1)
                bound *= (0 if t else 1) if whole else (1 - amounts[s] * q) ** t
            assert _miss(kind, sizes, ks, p, ts) <= bound, (kind, sizes, amounts, num, den, ks, p, ts)
    assert min(checked.values()) >= 30, checked


def test_the_cheapest_strata_pass_and_one_fewer_fails():
    rng = random.Random(7)
    for _ in range(300):
        sizes = [rng.randint(1, 30) for _ in range(rng.randint(1, 4))]
        pop = population(sizes)
        table = {t: rng.randint(0, 6) for t, _ in pop}
        num, den = rng.randint(0, 5), rng.randint(1, 40)
        ss = DR.cheapest_for(num, den, DR.work_table(table), pop)
        ks = [s["k"] for s in ss]
        assert passes(num, den, table, pop, stratified(pop, ks)), (sizes, table, num, den, ks)
        assert passes(num, den, table, pop, stratified(pop, ks, "work"))
        for i, k in enumerate(ks):
            if k:
                assert not passes(num, den, table, pop, stratified(pop, [*ks[:i], k - 1, *ks[i + 1:]])), (sizes, table, i)


def test_padding_with_amount_zero_keeps_a_stratified_pass():
    rng = random.Random(11)
    for _ in range(200):
        sizes = [rng.randint(1, 20) for _ in range(rng.randint(1, 3))]
        pop = population(sizes)
        table = {t: rng.randint(0, 5) for t, _ in pop}
        num, den = rng.randint(1, 4), rng.randint(1, 30)
        ks = [s["k"] for s in DR.cheapest_for(num, den, DR.work_table(table), pop)]
        ks = [min(n, k + rng.randint(0, 2)) for n, k in zip(sizes, ks)]
        assert passes(num, den, table, pop, stratified(pop, ks))
        padded = population([*sizes, rng.randint(1, 10 ** 6)])
        assert passes(num, den, {**table, padded[-1][0]: 0}, padded, stratified(padded, [*ks, 0]))
        zero = [i for i, (t, _) in enumerate(pop) if table[t] == 0]
        if zero:
            grown = population([n + (10 ** 6 if i == zero[0] else 0) for i, n in enumerate(sizes)])
            assert passes(num, den, table, grown, stratified(grown, ks))


def test_padding_a_stratum_of_positive_amount_fails_once_its_k_falls_below_the_rate():
    rng = random.Random(13)
    for _ in range(100):
        sizes = [rng.randint(1, 10) for _ in range(rng.randint(1, 3))]
        pop = population(sizes)
        table = {t: rng.randint(1, 5) for t, _ in pop}
        num, den = rng.randint(1, 4), rng.randint(1, 30)
        ks = [s["k"] for s in DR.cheapest_for(num, den, DR.work_table(table), pop)]
        s = rng.randrange(len(sizes))
        c, failed = table[f"t{s}"], False
        for extra in range(1, 200):
            grown = [n + (extra if i == s else 0) for i, n in enumerate(sizes)]
            ok = passes(num, den, table, population(grown), stratified(population(grown), ks))
            assert ok == (ks[s] * den >= grown[s] * c * num), (sizes, table, num, den, ks, s, extra)
            failed |= not ok
        assert failed


def test_a_pooled_subset_fails_once_amount_zero_padding_dilutes_it():
    """The pooled subset is one stratum at the largest amount, so blocks of amount 0 still lower each block's inclusion
    k/N: the check holds it to its padded N, which is why a fixed count is no rule."""
    pop = population([8, 8])
    table, k = {"t0": 1, "t1": 0}, 4
    assert passes(1, 4, table, pop, {"law": "subset", "k": k})
    padded = population([8, 8, 9])
    assert not passes(1, 4, {**table, "t2": 0}, padded, {"law": "subset", "k": k})


@pytest.mark.parametrize("design, table, num, den, why", [
    ({"law": "stratified", "strata": [{"template": "t0", "units": [[0, 8]], "k": 8}]}, {"t0": 9}, 1, 1, None),
    ({"law": "stratified", "strata": [{"template": "t0", "units": [[0, 8]], "k": 0}]}, {"t0": 0}, 1, 1, None),
    ({"law": "stratified", "strata": [{"template": "t0", "units": [[0, 8]], "k": 0}]}, {"t0": 9}, 0, 1, None),
    ({"law": "subset", "k": 8}, {"t0": 1}, 1, 0, "a rate needs DEN > 0, not 1/0"),
    ({"law": "subset", "k": 8}, {}, 1, 1, "the amount table names no amount for template 't0'"),
    ({"law": "subset", "k": 9}, {"t0": 1}, 1, 1, "a subset design of 9 exceeds the population of 8"),
    ({"law": "bernoulli", "p": "3/2"}, {"t0": 1}, 1, 1, "a bernoulli design's p = 3/2 is not a probability"),
    ({"law": "work", "strata": [{"template": "t1", "units": [[0, 8]], "k": 8}]}, {"t0": 1}, 1, 1,
     "the design's stratum 't1' is no template of the population"),
    ({"law": "work", "strata": [{"template": "t0", "units": [[0, 7]], "k": 7}]}, {"t0": 1}, 1, 1,
     "stratum 't0': its units are not the population's units of its template"),
    ({"law": "stratified", "strata": [{"template": "t0", "units": [[0, 8]], "k": 9}]}, {"t0": 1}, 1, 1,
     "stratum 't0': k = 9 exceeds its 8 proof blocks"),
])
def test_the_refusals_name_what_fails(design, table, num, den, why):
    pop = population([8])
    if why is None:
        DR.meets_rate(num, den, DR.work_table(table), pop, design)
    else:
        with pytest.raises(ValueError) as e:
            DR.meets_rate(num, den, DR.work_table(table), pop, design)
        assert str(e.value) == why
