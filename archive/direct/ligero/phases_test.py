"""The phase accounting of the bench runners (``phases.py``): pipelined sub-batch timings -> rep sums -> the reported rep, with
the contract's own phase-sum check (``contract.validate_measurements``: buckets <= t.total within 1 %).  Torch-free:

    python -m pytest archive/direct/ligero/phases_test.py
"""
from __future__ import annotations

import random
import statistics
from types import SimpleNamespace

import pytest
from verity_numerical.bench import contract

from . import phases

#: the laps ``protocol._prove_stages`` records on the fused v3 / v3x4 path (hints_fn -> ``hints_host``, the commit graph ->
#: ``witness``, ...), plus a host trace mark that must stay out of the attribution
LAPS = ("statement", "hints_host", "witness", "zk_masks", "encode", "transfer", "merkle", "tests_w", "tests_quad",
        "tests_chain", "openings")


def _sub_batch(weights: dict[str, float], wall_hint: float) -> SimpleNamespace:
    """One sub-batch's ``timings`` as ``protocol.prove`` returns them (event laps, ``tests`` derived, ``total`` = clock wall
    minus ``hints_host``) before ``pipeline.prove_many`` rescales them."""
    tm = {k: weights[k] * wall_hint for k in LAPS}
    tm["tests"] = tm["tests_w"] + tm["tests_quad"] + tm["tests_chain"]
    tm["host_wait_root"] = 0.3 * wall_hint
    tm["total"] = 1.05 * sum(tm[k] for k in LAPS) - tm["hints_host"]
    return SimpleNamespace(timings=tm)


def _weights(rng: random.Random, heavy: str | None = None) -> dict[str, float]:
    w = {k: rng.uniform(0.02, 0.2) for k in LAPS}
    if heavy is not None:
        w[heavy] *= 8.0
    return w


def _rep(rng: random.Random, wall: float, n_sub: int, heavy: str | None = None) -> dict[str, float]:
    """A pipelined rep: ``n_sub`` sub-batches whose laps overlap (their sum is ~3x the pass wall), rescaled to the wall
    and summed over the sub-batches as the runners' ``agg``."""
    subs = [_sub_batch(_weights(rng, heavy), wall) for _ in range(n_sub)]
    phases.rescale(subs, wall, "total")
    agg: dict[str, float] = {}
    for s in subs:
        for k, v in s.timings.items():
            agg[k] = agg.get(k, 0.0) + v
    return agg


def _phase_sum_problems(tm: dict[str, float], total: float) -> list[str]:
    meas = [contract.measurement(n, v) for n, v in phases.buckets(tm).items()] + [contract.measurement("t.total", total)]
    return [p for p in contract.validate_measurements(meas) if "double-counted" in p]


def test_rescale_buckets_are_the_share():
    rng = random.Random(1)
    wall, n = 0.6, 3
    subs = [_sub_batch(_weights(rng), wall) for _ in range(n)]
    measured = [dict(s.timings) for s in subs]
    phases.rescale(subs, wall, "total")
    for s, before in zip(subs, measured):
        tm = s.timings
        assert phases.prover_seconds(tm) == pytest.approx(wall / n)
        assert sum(phases.buckets(tm).values()) == pytest.approx(wall / n)
        assert tm["tests"] == pytest.approx(tm["tests_w"] + tm["tests_quad"] + tm["tests_chain"])
        assert tm["host_wait_root"] == before["host_wait_root"]
        assert all(tm["measured_" + k] == before[k] for k in LAPS)


def test_median_rep_keeps_buckets_within_total():
    """Two slow timed passes (graph captures) and three fast ones, each with another dominant phase: the per-key median of
    every lap is then the largest lap of the fast reps, and the old per-key aggregation overshoots t.total."""
    rng = random.Random(7)
    walls = [0.3386, 0.3349, 0.0993, 0.0950, 0.0932]
    heavy = [None, None, "encode", "tests_w", "witness"]
    runs = [(_rep(rng, w, 13, h), 0.2) for w, h in zip(walls, heavy)]

    keys = runs[0][0].keys()
    per_key = {k: statistics.median(r[0][k] for r in runs) for k in keys}
    old_total = per_key["total"] + per_key["hints_host"]
    assert sum(phases.buckets(per_key).values()) > old_total * (1 + contract.TOLERANCE)
    assert _phase_sum_problems(per_key, old_total)

    i = phases.median_rep(runs)
    med = runs[i][0]
    total = phases.prover_seconds(med)
    assert total == pytest.approx(statistics.median(walls))
    assert sum(phases.buckets(med).values()) == pytest.approx(total)
    assert not _phase_sum_problems(med, total)


def test_median_rep_even_count_is_upper_middle():
    rng = random.Random(3)
    walls = [0.21, 0.19, 0.25, 0.20]
    runs = [(_rep(rng, w, 7), 0.1) for w in walls]
    i = phases.median_rep(runs)
    assert walls[i] == 0.21
    assert not _phase_sum_problems(runs[i][0], phases.prover_seconds(runs[i][0]))


def test_sequential_rep_has_slack():
    """Without --pipeline (no rescale) the laps are a subset of the clock's wall: the buckets stay below t.total."""
    rng = random.Random(5)
    subs = [_sub_batch(_weights(rng), 0.05) for _ in range(4)]
    agg: dict[str, float] = {}
    for s in subs:
        for k, v in s.timings.items():
            agg[k] = agg.get(k, 0.0) + v
    assert sum(phases.buckets(agg).values()) < phases.prover_seconds(agg)


def test_serialization_extra():
    tm = {k: 1.0 for k in LAPS} | {"tests": 3.0, "auth_openings": 0.5, "total": 10.0}
    assert phases.buckets(tm)["t.serialization"] == 2.0
    assert phases.buckets(tm, serialization_extra=("auth_openings",))["t.serialization"] == 2.5
