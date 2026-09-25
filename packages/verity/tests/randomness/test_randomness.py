from fractions import Fraction

import pytest

from verity.randomness import derive


def test_derive_is_deterministic_and_separates_source_domain_and_context():
    base = derive(b"round", "verity/sampled-proofs/ru-sample/v1", {"root": b"r", "epoch": 3})
    assert base == derive(b"round", "verity/sampled-proofs/ru-sample/v1", {"epoch": 3, "root": b"r"})
    others = {
        derive(b"other", "verity/sampled-proofs/ru-sample/v1", {"root": b"r", "epoch": 3}),
        derive(b"round", "verity/pouw/noise-seed/v1", {"root": b"r", "epoch": 3}),
        derive(b"round", "verity/sampled-proofs/ru-sample/v1", {"root": b"s", "epoch": 3}),
        derive(b"round", "verity/sampled-proofs/ru-sample/v1", {"root": b"r", "epoch": "3"}),
    }
    assert base not in others and len(others) == 4


@pytest.mark.parametrize("domain", ["pouw/noise/v1", "verity/pouw/v1", "verity/PoUW/noise/v1", "verity/pouw/noise"])
def test_domain_shape_is_enforced(domain):
    with pytest.raises(ValueError):
        derive(b"k", domain, {})


def test_samplers_are_exact_and_in_range():
    key = derive(b"k", "verity/test/samplers/v1", {})
    assert all(0 <= key.below(7, i) < 7 for i in range(200))
    assert key.subset(3, 10, "a") == key.subset(3, 10, "a")
    assert len(set(key.subset(10, 10, "b"))) == 10
    hits = sum(key.bernoulli(Fraction(1, 4), i) for i in range(4000))
    assert 850 < hits < 1150
    assert not any(key.bernoulli(Fraction(0), i) for i in range(50))
    assert all(key.bernoulli(Fraction(1), i) for i in range(50))
