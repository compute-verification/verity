"""`predict.seed_rule`: `ServedSeed_v1` is the body's seed where it names one and `served_seed.derived` of the window's coin and
the request's id where it doesn't, at a message of one SHA-512 block and of two (one `Sha512Compress_v1` Call a block); and
`SeedCheck_v1`, from a body's bytes through ingress schema v1's parse (`boolean_io_shell.IngressSampled_v1`), passes the rule's
seed and fails every forged one."""
from __future__ import annotations

import numpy as np
import pytest
from verity_vllm.engine import served_seed
from verity_vllm.program.predict import seed_rule as SR

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
    assert SR.blocks(COIN, RID) == n == sum(c.fn.id == "Sha512Compress_v1" for c in fn.body.nodes)
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


@pytest.mark.parametrize("kw", [{"COIN": 0, "RID": 4}, {"COIN": 16, "RID": 0}, {"COIN": 16, "RID": True}])
def test_lengths_of_no_bytes_are_refused(kw):
    for d in (SR.ServedSeed, SR.SeedCheck):
        with pytest.raises(ValueError, match="refuses this binding"):
            bind(d, **kw)
