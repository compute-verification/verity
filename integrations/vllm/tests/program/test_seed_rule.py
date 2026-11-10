"""`frontend.seed_rule`: `ServedSeed_v1` is the body's seed where it names one and `served_seed.derived` of the window's coin and
the request's id where it doesn't, at a message of one SHA-512 block and of two (one `Sha512Compress_v1` Call a block but the last,
which `SeedSelect_v1` compresses); `SeedCheck_v1`, from a body's bytes through ingress schema v1's parse
(`boolean_io_shell.IngressSampled_v1`), passes the rule's seed and fails every forged one; and `SeedCheck_v2`, the check a
stochastic request Program returns, does so on the sampler's i64 seed word."""
from __future__ import annotations

import numpy as np
import pytest
from verity_vllm.engine import served_seed
from verity_vllm.program.frontend import seed_rule as SR

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.definitions.registry import boolean_io_shell as BX
from verity.ml.definitions.registry import io_shell as X


def _col(xs) -> np.ndarray:
    return np.array([[int(x)] for x in xs], dtype=np.uint64)


def _rows(xs) -> np.ndarray:
    return np.array([list(x) for x in xs], dtype=np.uint32)


def _eval(fn, *args) -> list[int]:
    return [int(v) for v in np.asarray(evaluate_words(fn, *args), dtype=np.uint64).reshape(-1)]


@pytest.mark.slow
@pytest.mark.parametrize("COIN, RID, n", [(16, 4, 1), (64, 16, 2)], ids=["one-block", "two-block"])
def test_the_rule_is_the_bodys_seed_else_derived_from_the_coin_and_the_id(COIN, RID, n):
    fn = bind(SR.ServedSeed, COIN=COIN, RID=RID)
    assert SR.blocks(COIN, RID) == n == 1 + sum(c.fn.id == "Sha512Compress_v1" for c in fn.body.nodes)
    assert fn.body.nodes[-1].fn.id == "SeedSelect_v1"
    rng = np.random.default_rng(COIN)
    coins = [bytes(rng.integers(0, 256, COIN, dtype=np.uint8)) for _ in range(6)]
    users = [bytes(rng.integers(0x20, 0x7F, RID, dtype=np.uint8)) for _ in range(6)]
    seeds = [0, (1 << 63) - 1, *[int(rng.integers(0, 1 << 63, dtype=np.uint64)) for _ in range(4)]]
    seeded = [1, 1, 0, 0, 1, 0]
    got = _eval(fn, _col(seeded), _col(seeds), _rows(coins), _rows(users))
    assert got == [SR.reference(*a) for a in zip(seeded, seeds, coins, users)]
    assert [g for g, s in zip(got, seeded) if not s] == [served_seed.derived(c, u.decode()) for c, u, s in zip(coins, users, seeded)
                                                         if not s]
    assert [g for g, s in zip(got, seeded) if s] == [x for x, s in zip(seeds, seeded) if s]


@pytest.mark.slow
def test_a_forged_seed_fails_the_check():
    """From two bodies' bytes, one naming no seed and one naming 12345, through `IngressSampled_v1` to `SeedCheck_v1` under the
    window's coin: the seed the rule gives passes, and every other a server might sample under fails: the derived seed with a
    bit flipped, the public default, the seed derived under another coin or for another id, the null field's digit sum, and
    for the seeded body the derived seed and a neighbour of its own."""
    st, COIN = {"MODEL": "m", "LP": 2, "V": 1000, "RID": 4}, 16
    coin, other = bytes(range(COIN)), bytes(range(1, COIN + 1))
    bodies = [X.sampled_encode("m", [5, 7], 9, 1000, temperature=0.7, top_p=0.9, seed=s, user=u)
              for s, u in ((None, "r001"), (12345, "r002"))]
    arr = np.frombuffer(b"".join(bodies), dtype=np.uint8).reshape(2, -1).astype(np.uint32)
    rows = np.asarray(evaluate_words(bind(BX.IngressSampledV1, **st), arr), dtype=np.uint64).reshape(2, -1)
    at = 1 + st["LP"] + 3
    parsed = [(int(r[0]), int(r[at]), int(r[at + 1]), bytes(int(c) for c in r[at + 2:])) for r in rows]
    assert [p[0] for p in parsed] == [1, 1] and [p[1] for p in parsed] == [0, 1] and parsed[1][2] == 12345
    derived = served_seed.derived(coin, "r001")
    cases = [(0, derived, 1), *[(0, derived ^ 1 << k, 0) for k in (0, 31, 62)], (0, served_seed.DEFAULT_SEED, 0),
             (0, served_seed.derived(other, "r001"), 0), (0, served_seed.derived(coin, "r002"), 0), (0, parsed[0][2], 0),
             (1, 12345, 1), (1, served_seed.derived(coin, "r002"), 0), (1, 12346, 0)]
    served = [s for _, s, _ in cases]
    seeded, seed, user = ([parsed[i][k] for i, _, _ in cases] for k in (1, 2, 3))
    got = _eval(bind(SR.SeedCheck, COIN=COIN, RID=st["RID"]), _col(served), _col(seeded), _col(seed), _rows([coin] * len(cases)),
                _rows(user))
    assert got == [w for _, _, w in cases]
    assert got == [SR.check_reference(*a) for a in zip(served, seeded, seed, [coin] * len(cases), user)]


@pytest.mark.slow
def test_the_word_check_accepts_the_rules_seed_alone():
    """At the window's 64-byte seed salt and a request's `values`, the rule's seed of an unseeded request (derived from the salt
    and its id) and of a seeded one (its body's) passes, and fails with any one of its 64 bits flipped (`SeedCheck_v1`) or with
    the other request's seed: `SeedCheck_v2`, the check the Program returns, agrees on the sampler's i64 word at its ends' bits, so
    a Program's sampler passes it with one seed alone."""
    from verity.ml.circuits.evaluation import evaluate_batch

    coin, uid = bytes(range(SR.COIN)), "r001"
    rule = [(None, served_seed.derived(coin, uid)), (12345, 12345)]

    def cases(flips):
        return [(body, served, int(served == good)) for i, (body, good) in enumerate(rule)
                for served in (good, *(good ^ 1 << k for k in flips), rule[1 - i][1])]

    def ins(cs):
        return [np.stack(col) for col in zip(*(SR.values(body, coin, uid) for body, _, _ in cs))]

    every, ends = cases(range(64)), cases((0, 1, 62, 63))
    bits = _eval(bind(SR.SeedCheck, COIN=SR.COIN, RID=SR.rid_of(uid)), _col(s for _, s, _ in every),
                 _col(b is not None for b, _, _ in every), _col(b or 0 for b, _, _ in every), _rows([coin] * len(every)),
                 _rows([uid.encode()] * len(every)))
    word = evaluate_batch(bind(SR.SeedCheckWord, COIN=SR.COIN, RID=SR.rid_of(uid)), np.array([s for _, s, _ in ends], dtype=np.uint64),
                          *ins(ends), kernel="reference")
    for cs, got in ((every, bits), (ends, [int(v) for v in np.asarray(word).reshape(-1)])):
        assert got == [w for _, _, w in cs] == [SR.check_reference(s, b is not None, b or 0, coin, uid.encode()) for b, s, _ in cs]
        assert sum(got) == 2


def test_the_wrappers_inputs_are_the_programs_parameters_and_the_check_refuses_any_other():
    """The traced request wrapper's example inputs (`inputs`, bool tensors) type as the predicted Program's `params` at the
    policy's RID, every request-local group the workload Program and the batch rule hold its request's own; a row whose ids
    share no length has no RID, and a policy without one and a seed or inputs of other types are refused by name."""
    from verity.ml.circuits.defs import Builder
    from verity.ml.circuits.refs import Affine, Coll
    from verity.ml.circuits.types import Array, Value
    from verity_vllm.program import workload as W
    from verity_vllm.program.frontend.value_types import type_of_tensor
    from verity_vllm.program.predict import batch

    assert [type_of_tensor(t) for t in SR.inputs({"rid": 36})] == [t for _, t in SR.params(36)]
    assert set(SR.PARAMS) <= set(W._REQUEST_LOCAL_GROUPS) and set(SR.PARAMS) <= set(batch.OWN_GROUPS)
    assert (SR.rid_of_ids(["r001", "r002"]), SR.rid_of_ids(["r1", "r002"]), SR.rid_of_ids([])) == (4, None, None)
    with pytest.raises(ValueError, match="its sampling policy states as `rid`.*this one states None"):
        SR.rid_in({"kind": "stochastic"})
    params = (("seed", Array(1, Value(64))), *SR.params(4))
    B = Builder(params, Value(1))
    ins = [Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params)]
    bit, what = SR.check(B, ins[0], ins[1:])
    assert bit.type == Value(1) and B.nodes[-1].fn.id == "SeedCheck_v2{COIN=64,RID=4}" and what.startswith("the seed rule")
    with pytest.raises(ValueError, match="the sampler's seed, one i64"):
        SR.check(B, ins[1], ins[1:])
    with pytest.raises(ValueError, match="the seed rule reads seeded, body_seed, seed_salt, user"):
        SR.check(B, ins[0], ins[2:])


@pytest.mark.parametrize("kw", [{"COIN": 0, "RID": 4}, {"COIN": 16, "RID": 0}, {"COIN": 16, "RID": True}])
def test_lengths_of_no_bytes_are_refused(kw):
    for d in (SR.ServedSeed, SR.SeedCheck, SR.SeedCheckWord):
        with pytest.raises(ValueError, match="refuses this binding"):
            bind(d, **kw)
