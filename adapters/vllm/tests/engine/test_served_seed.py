"""`engine.served_seed`: a client's seed is kept, a greedy request without one carries the public default, and a stochastic
one is sampled under the seed derived from its window's seed coin and its certified id, a hash of those two alone, or refused
without an id; the caller's params are never changed, and a fresh coin is flipped only for an idle engine."""
from __future__ import annotations

import copy
import hashlib
import random

import pytest

from verity_vllm.engine import served_seed

COIN, OTHER = bytes(range(64)), bytes(range(1, 65))


class Params:
    def __init__(self, seed=None, temperature=0.0, **kw):
        self.seed, self.temperature = seed, temperature
        self.__dict__.update(kw)

    def clone(self):
        return copy.copy(self)


def test_the_default_is_the_hash_of_its_name():
    want = int.from_bytes(hashlib.sha512(b"verity-vllm/served-default-seed/v1").digest()[:8], "big") >> 1
    assert served_seed.DEFAULT_SEED == want and 0 <= want < 1 << 63


def test_a_derived_seed_is_the_hash_of_its_domain_the_coin_and_the_id_each_length_prefixed():
    coin = bytes(range(64))
    data = served_seed.DERIVED + (64).to_bytes(4, "big") + coin + (2).to_bytes(4, "big") + b"q1"
    want = int.from_bytes(hashlib.sha512(data).digest()[:8], "big") >> 1
    assert served_seed.derived(coin, "q1") == want and 0 <= want < 1 << 63 and want != served_seed.DEFAULT_SEED
    assert served_seed.derived(coin + b"q", "1") != served_seed.derived(coin, "q1")
    for bad in (b"", "coin", None):
        with pytest.raises(ValueError, match="a window's verifier coin is non-empty bytes"):
            served_seed.derived(bad, "q1")
    with pytest.raises(ValueError, match="a request's id is a string"):
        served_seed.derived(coin, 1)


@pytest.mark.parametrize("params", [Params, dict])
def test_served_keeps_a_clients_seed_defaults_a_greedy_one_and_derives_a_stochastic_one_without_changing_the_callers(params):
    mine, greedy, hot = params(seed=7, temperature=1.0), params(seed=None, temperature=0.0), params(seed=None, temperature=0.8)
    assert served_seed.served(mine, "q1", COIN) == (mine, "request")
    for p, seed, rule in ((greedy, served_seed.DEFAULT_SEED, "default"), (hot, served_seed.derived(COIN, "q1"), "derived")):
        got, why = served_seed.served(p, "q1", COIN)
        assert (served_seed.param_of(got, "seed"), why) == (seed, rule) and got is not p and served_seed.param_of(p, "seed") is None
    assert served_seed.param_of(served_seed.served(greedy, None, None)[0], "seed") == served_seed.DEFAULT_SEED
    if params is dict:
        assert served_seed.rule_of({"seed": None}) == "derived"          # no temperature: vLLM's default, 1


@pytest.mark.parametrize("params", [Params(temperature=0.8), {"temperature": 1.0, "seed": None}])
def test_an_unseeded_stochastic_request_without_an_id_is_refused_by_name(params):
    with pytest.raises(ValueError, match="sampled under one derived from its window's seed coin and its certified id, the body's `user`"):
        served_seed.served(params, None, COIN)


def test_a_served_seed_is_a_function_of_the_coin_and_the_id_alone():
    """Every other sampling field, the params' kind and the order requests come in leave a derived seed as it is; another coin or
    another id moves it."""
    rng = random.Random(5)
    for i in range(50):
        rid = f"user-{rng.randrange(1000)}"
        kw = {"temperature": rng.choice((0.3, 0.8, 1.0)), "top_p": rng.choice((1.0, 0.9)), "top_k": rng.randrange(60),
              "max_tokens": rng.randrange(1, 64), "min_tokens": rng.randrange(4), "stop": ["\n"] * rng.randrange(2)}
        a, _ = served_seed.served(Params(**kw), rid, COIN)
        b, _ = served_seed.served({**kw, "temperature": 0.5, "max_tokens": 1, "seed": None}, rid, COIN)
        assert a.seed == b["seed"] == served_seed.derived(COIN, rid)
        assert served_seed.served(Params(**kw), rid, OTHER)[0].seed == served_seed.derived(OTHER, rid) != a.seed
        assert served_seed.served(Params(**kw), rid + "x", COIN)[0].seed != a.seed


def test_seed_coins_flip_a_fresh_coin_only_for_an_idle_engine_and_name_each():
    """A request coming into an idle engine with no window open flips a fresh coin; one joining a busy engine or an open window
    keeps the current one, so each window holds the coin its requests were seeded under."""
    flips = iter([(COIN, "test"), (OTHER, "test")])
    coins = served_seed.SeedCoins(lambda: next(flips))
    assert coins.record() is None
    assert coins.admitting(idle=False) == COIN                          # the first flips, whatever it joins
    assert coins.admitting(idle=False) == COIN and coins.record() == {"index": 0, "coin": COIN.hex(), "source": "test"}
    assert coins.admitting(idle=True) == OTHER and coins.record() == {"index": 1, "coin": OTHER.hex(), "source": "test"}
    got = served_seed.SeedCoins().admitting(idle=True)
    assert type(got) is bytes and len(got) == 64 and served_seed.SeedCoins().record() is None
    with pytest.raises(ValueError, match="a seed coin is non-empty bytes"):
        served_seed.SeedCoins(lambda: (b"", "test")).admitting(idle=True)
