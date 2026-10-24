"""`engine.served_seed`: a request admitted without a seed is sampled under the public default, a client's seed is kept, and
the caller's params are never changed."""
from __future__ import annotations

import copy
import hashlib

from verity_vllm.engine import served_seed


class Params:
    def __init__(self, seed=None, temperature=0.0):
        self.seed, self.temperature = seed, temperature

    def clone(self):
        return copy.copy(self)


class Engine:
    def __init__(self):
        self.admitted = []

    def add_request(self, request_id, prompt, params=None, *rest, **kw):
        self.admitted.append((request_id, params, rest, kw))


def test_the_default_is_the_hash_of_its_name():
    want = int.from_bytes(hashlib.sha512(b"verity-vllm/served-default-seed/v1").digest()[:8], "big") >> 1
    assert served_seed.DEFAULT_SEED == want and 0 <= want < 1 << 63


def test_install_seeds_unseeded_requests_only_and_copies():
    eng, mine, theirs = Engine(), Params(), Params(seed=7, temperature=1.0)
    hooks = served_seed.install(eng)
    eng.add_request("a", {"prompt_token_ids": [1]}, mine, 3, arrival_time=None)
    eng.add_request("b", {"prompt_token_ids": [1]}, params=theirs)
    (_, pa, rest, kw), (_, pb, _, _) = eng.admitted
    assert pa.seed == served_seed.DEFAULT_SEED and pa is not mine and mine.seed is None
    assert (rest, kw) == ((3,), {"arrival_time": None})
    assert pb is theirs and pb.seed == 7
    hooks.uninstall()
    eng.add_request("c", {"prompt_token_ids": [1]}, Params())
    assert eng.admitted[-1][1].seed is None


def test_seeded_falls_back_to_copy_and_leaves_paramless_requests():
    class Bare:
        seed = None
    got = served_seed.seeded(b := Bare())
    assert got.seed == served_seed.DEFAULT_SEED and b.seed is None
    assert served_seed.seeded(None) is None
    assert served_seed.seeded(object) is object
