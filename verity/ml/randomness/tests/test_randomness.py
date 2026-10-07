"""verity.ml.randomness: known-answer vectors pin the byte formats; property tests check the samplers are exact."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import pytest

from verity.ml.randomness import Key, derive, frame, uniform_from_words

VECTORS = json.loads((Path(__file__).with_name("vectors.json")).read_text())


def _context(doc: dict) -> dict:
    return {n: bytes.fromhex(v["bytes"]) if "bytes" in v else v.get("str", v.get("int")) for n, v in doc.items()}


def _index(x):
    return tuple(x) if isinstance(x, list) else x


@pytest.mark.parametrize("value, hexed", VECTORS["frame"])
def test_frame_vectors(value, hexed):
    assert frame(ast.literal_eval(value)).hex() == hexed


@pytest.mark.parametrize("case", VECTORS["cases"], ids=lambda c: c["domain"])
def test_known_answer_vectors(case):
    key = derive(bytes.fromhex(case["source"]), case["domain"], _context(case["context"]))
    assert key.value.hex() == case["key"]
    s = case["stream"]
    assert key.stream(_index(s["index"]), len(s["bytes"]) // 2).hex() == s["bytes"]
    assert key.shard(_index(case["shard"]["index"])).value.hex() == case["shard"]["key"]
    assert [key.uniform(n, i) for n, i, _ in case["uniform"]] == [x for *_, x in case["uniform"]]
    assert [key.bernoulli(Fraction(p), i) for p, i, _ in case["bernoulli"]] == [x for *_, x in case["bernoulli"]]
    assert [list(key.subset(k, n, i)) for k, n, i, _ in case["subset"]] == [x for *_, x in case["subset"]]


def test_the_vectors_follow_the_documented_construction():
    """An independent restatement of the module docstring's formats, so the vectors are not only self-consistent."""
    case = VECTORS["cases"][1]
    ctx = _context(case["context"])
    fr = lambda tag, body: tag + len(body).to_bytes(8, "big") + body  # noqa: E731
    enc = lambda x: fr(b"b", x) if isinstance(x, bytes) else fr(b"s", x.encode()) if isinstance(x, str) else fr(b"i", x.to_bytes(max(1, (x.bit_length() + 7) // 8), "big"))  # noqa: E731
    body = b"".join(enc(k) + enc(ctx[k]) for k in sorted(ctx))
    key = hashlib.sha512(b"verity/randomness/derive/v2\0" + enc(bytes.fromhex(case["source"])) + enc(case["domain"]) + enc(body)).digest()[:32]
    assert key.hex() == case["key"]
    index = enc(enc("s") + enc(3))
    stream = b"".join(hashlib.sha512(b"verity/randomness/stream/v2\0" + key + index + c.to_bytes(8, "big")).digest() for c in range(2))
    assert stream[:80].hex() == case["stream"]["bytes"]
    shard = hashlib.sha512(b"verity/randomness/shard/v2\0" + key + enc(enc(5))).digest()[:32]
    assert shard.hex() == case["shard"]["key"]
    n, i, want = case["uniform"][2]
    width = (n.bit_length() + 64 + 7) // 8
    u_stream = b"".join(hashlib.sha512(b"verity/randomness/stream/v2\0" + key + enc(enc("u") + enc(i)) + c.to_bytes(8, "big")).digest()
                        for c in range(2))
    words = [int.from_bytes(u_stream[j:j + width], "big") for j in range(0, len(u_stream) - width + 1, width)]
    limit = 2 ** (8 * width) - 2 ** (8 * width) % n
    assert next(w % n for w in words if w < limit) == want


def test_the_vectors_are_the_generators():
    spec = importlib.util.spec_from_file_location("randomness_vectors", Path(__file__).with_name("vectors.py"))
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    assert gen.build() == VECTORS


def test_derive_separates_source_domain_and_context():
    base = derive(b"round", "verity/sampled-proofs/ru-sample/v1", {"root": b"r", "epoch": 3})
    assert base == derive(b"round", "verity/sampled-proofs/ru-sample/v1", {"epoch": 3, "root": b"r"})
    others = {
        derive(b"other", "verity/sampled-proofs/ru-sample/v1", {"root": b"r", "epoch": 3}),
        derive(b"round", "verity/pouw/noise-seed/v1", {"root": b"r", "epoch": 3}),
        derive(b"round", "verity/sampled-proofs/ru-sample/v1", {"root": b"s", "epoch": 3}),
        derive(b"round", "verity/sampled-proofs/ru-sample/v1", {"root": b"r", "epoch": "3"}),
        derive(b"round", "verity/sampled-proofs/ru-sample/v1", {"root": b"r"}),
        derive(b"round", "verity/sampled-proofs/ru-sample/v1", {"root": b"r", "epoch": 3, "x": b""}),
    }
    assert base not in others and len(others) == 6


@pytest.mark.parametrize("domain", ["pouw/noise/v1", "verity/pouw/v1", "verity/PoUW/noise/v1", "verity/pouw/noise", "verity/a/b/v1/"])
def test_domain_shape_is_enforced(domain):
    with pytest.raises(ValueError):
        derive(b"k", domain, {})


@pytest.mark.parametrize("source, context", [(b"", {}), ("text", {}), (b"k", {"x": -1}), (b"k", {"x": True}), (b"k", {"x": 1.0}),
                                             (b"k", {1: b""})])
def test_bad_sources_and_context_parts_are_refused(source, context):
    with pytest.raises(TypeError):
        derive(source, "verity/test/refuse/v1", context)


def test_key_value_is_not_in_repr():
    key = derive(b"secret", "verity/test/repr/v1", {})
    assert key.value.hex() not in repr(key)


KEY = derive(b"property", "verity/test/properties/v1", {})


@pytest.mark.parametrize("width, n", [(3, 1), (3, 3), (3, 5), (4, 7), (4, 16), (5, 31)])
def test_the_reduction_is_exact(width, n):
    """Over every word of the width, each residue is returned exactly as often as every other (no modulo bias)."""
    counts = Counter()
    for w in range(2 ** width):
        try:
            counts[uniform_from_words(n, iter([w]), width)] += 1
        except AssertionError:
            pass
    assert set(counts) == set(range(n)) and len(set(counts.values())) == 1


def test_streams_are_prefix_consistent_and_index_separated():
    assert KEY.stream("a", 100)[:33] == KEY.stream("a", 33)
    assert KEY.stream("a", 32) != KEY.stream("b", 32) and KEY.stream(("a", 1), 32) != KEY.stream(("a", 2), 32)
    assert KEY.stream(("a", "b"), 32) != KEY.stream("ab", 32)


def test_sharded_draws_equal_unsharded_draws():
    """Draws are addressed by index, so each shard reproduces its slice of the population on its own."""
    population = range(1000)
    whole = [KEY.bernoulli(Fraction(1, 5), ("ru", i)) for i in population]
    shards = [[KEY.bernoulli(Fraction(1, 5), ("ru", i)) for i in population[s::4]] for s in range(4)]
    assert all(whole[s::4] == shards[s] for s in range(4))
    assert KEY.shard(0) != KEY.shard(1) and KEY.shard(0) == KEY.shard(0) and isinstance(KEY.shard(0), Key)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 10])
def test_uniform_is_in_range_and_balanced(n):
    draws = Counter(KEY.uniform(n, i) for i in range(2000 * n))
    assert set(draws) == set(range(n))
    assert all(abs(c - 2000) < 5 * 2000 ** 0.5 for c in draws.values())


def test_uniform_handles_large_ranges():
    for n in (2**64, 2**128 + 1, 10**30, 2**300 - 1):
        assert all(0 <= KEY.uniform(n, i) < n for i in range(50))


@pytest.mark.parametrize("p", [Fraction(1, 4), Fraction(2, 3), Fraction(1, 1000)])
def test_bernoulli_rate(p):
    trials = 20000
    hits = sum(KEY.bernoulli(p, i) for i in range(trials))
    mean, sd = trials * p, (trials * p * (1 - p)) ** 0.5
    assert abs(hits - mean) < 5 * sd + 1


def test_bernoulli_endpoints_and_domain():
    assert not any(KEY.bernoulli(Fraction(0), i) for i in range(50)) and all(KEY.bernoulli(Fraction(1), i) for i in range(50))
    with pytest.raises(ValueError):
        KEY.bernoulli(Fraction(3, 2))


@pytest.mark.parametrize("k, n", [(0, 0), (0, 5), (1, 1), (3, 10), (10, 10), (5, 10**15)])
def test_subset_shape(k, n):
    s = KEY.subset(k, n, ("shape", k, n))
    assert len(s) == k == len(set(s)) and list(s) == sorted(s) and all(0 <= x < n for x in s)
    assert s == KEY.subset(k, n, ("shape", k, n))


def test_subset_is_uniform_over_subsets():
    draws = Counter(KEY.subset(2, 5, i) for i in range(10000))
    assert set(draws) == set(combinations(range(5), 2))
    assert all(abs(c - 1000) < 5 * 1000 ** 0.5 for c in draws.values())


def test_subset_refuses_bad_sizes():
    for k, n in ((-1, 3), (4, 3)):
        with pytest.raises(ValueError):
            KEY.subset(k, n)
