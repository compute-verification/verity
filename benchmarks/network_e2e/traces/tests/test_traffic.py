from __future__ import annotations

import random

import pytest


def docs(n: int = 3000, seed: int = 7) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(1000, 120000) for _ in range(rng.choice((40, 300, 700, 1300, 1500)))] for _ in range(n)]


def test_plan_is_seeded_and_in_range(mod):
    traffic = mod("traffic")
    d = docs()
    a, b = traffic.plan(d, 400, seed=3), traffic.plan(d, 400, seed=3)
    assert a == b
    assert a != traffic.plan(d, 400, seed=4)
    for r in a:
        assert 16 <= r["prompt_len"] <= 1024 and len(r["prompt_ids"]) == r["prompt_len"]
        assert r["prompt_ids"][0] == traffic.BOS
        assert r["max_tokens"] in traffic.CAPS
        body = r["prompt_ids"][r["prefix_len"] or 1:]
        doc = d[r["doc"]]
        end = len(doc) - r["gap"]
        assert body == doc[end - len(body):end]
    assert len({r["doc"] for r in a}) == len(a)
    assert {r["max_tokens"] for r in a} == set(traffic.CAPS)


def test_shared_prefixes_are_block_aligned_and_shared(mod):
    traffic = mod("traffic")
    d = docs()
    heads = traffic.prefixes(d)
    reqs = traffic.plan(d, 400, seed=0)
    grouped = [r for r in reqs if r["prefix"] is not None]
    assert 0.45 < len(grouped) / len(reqs) < 0.75
    for r in grouped:
        p = r["prefix_len"]
        assert p % traffic.BLOCK == 0 and r["prompt_ids"][:p] == heads[r["prefix"]]
        assert r["prompt_len"] - p >= traffic.MIN_BODY
    assert max(sum(1 for r in grouped if r["prefix"] == g) for g in range(len(traffic.PREFIXES))) > 20


def test_arrivals_scale_with_rate(mod):
    traffic = mod("traffic")
    reqs = traffic.plan(docs(), 200, seed=1)
    slow, fast = traffic.arrivals(reqs, 1.0), traffic.arrivals(reqs, 4.0)
    assert all(x == pytest.approx(4 * y) for x, y in zip(slow, fast))
    assert all(a < b for a, b in zip(slow, slow[1:]))
    assert 150 < slow[-1] < 250


def test_plan_refuses_when_documents_run_out(mod):
    traffic = mod("traffic")
    with pytest.raises(ValueError, match="no unused document"):
        traffic.plan(docs(n=30), 200, seed=0)
