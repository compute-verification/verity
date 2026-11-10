"""`boolean_dsa.DsaMqaLogits_v4`, the indexer's logits on bits as a composite over the keys (`DsaMqaLogitsQ_v1` once a row,
`DsaMqaLogitsKey_v1` batched over the keys):

* at small N its circuit is the flat trace it replaced (`DsaMqaLogit_v4` batched over the keys, traced as one circuit): the same
  AND, XOR and NOT counts and the same outputs on random bits, so the cut at q and w keeps every gate that trace shared;
* its description is the same two nodes at 2 keys and at 131,072, and it is `boolean_version`'s Boolean `DsaMqaLogits_v3` at
  every N;
* through its word view it is `DsaMqaLogits_v3` on finite words.
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation import evaluate_batch
from verity.definitions.ops.boolean import elementwise as EW
from verity.definitions.boolean import boolean_version
from verity.definitions.registry import boolean_dsa as BD
from verity.definitions.registry import dsa as D

STATICS = ["N", "H", "D"]


def _flat_body(B, S, q, w, k, ks):
    return B.batch(bind(BD.DsaMqaLogitV4, H=S.H, D=S.D), q, w, k, ks, axes=(None, None, 0, 0))


#: the logits as one traced circuit a binding, as `boolean_dsa.flat` made them
FLAT = EW.specialized("DsaMqaLogitsFlat", 1, STATICS,
                      BD.replayed(EW.on_bits("DsaMqaLogitsFlat", 1, STATICS, BD._logits_sig, _flat_body, D.DsaMqaLogitsV3,
                                             register=False), STATICS, BD._logits_sig),
                      BD._logits_sig, register=False)


@pytest.mark.slow
@pytest.mark.parametrize("H,N", [(16, 1), (16, 2), (8, 3)])
def test_the_composite_is_the_flat_trace(H: int, N: int) -> None:
    """At H = 16 each lane sums two heads, so the contracted first FFMA is in the circuit; H = 8 reaches three keys cheaply."""
    new, old = bind(BD.DsaMqaLogitsV4, N=N, H=H, D=32), bind(FLAT, N=N, H=H, D=32)
    assert BS.compile(new).counts == BS.compile(old).counts
    rng = np.random.default_rng(N)
    ins = [rng.integers(0, 2, size=(128, *s), dtype=np.uint8) for s in ((H, 32, 8), (H, 32), (N, 32, 8), (N, 32))]
    assert (BS.evaluate_bits(new, *ins) == BS.evaluate_bits(old, *ins)).all()


@pytest.mark.slow
def test_the_description_does_not_grow_with_the_keys() -> None:
    q, key = bind(BD.DsaMqaLogitsQ, H=8, D=32), bind(BD.DsaMqaLogitsKey, H=8, D=32)
    for n in (2, 32_533, 131_072):
        v4 = bind(BD.DsaMqaLogitsV4, N=n, H=8, D=32)
        assert [(nd.form, nd.fn.id, nd.n, nd.axes) for nd in v4.body.nodes] == [
            ("call", q.id, 1, ()), ("batch", key.id, n, (None, 0, 0, None))]
        assert boolean_version(bind(D.DsaMqaLogitsV3, N=n, H=8, D=32)).id == v4.id
    two = BS.compile(bind(BD.DsaMqaLogitsV4, N=2, H=8, D=32)).counts
    assert two == {k: BS.compile(q).counts[k] + 2 * BS.compile(key).counts[k] for k in two}


def _finite_words(rng, n: int, H: int, Dd: int, N: int) -> list[np.ndarray]:
    """q and k e4m3 bytes without NaN (0x7F, 0xFF), w normal f32 words of either sign, ks positive powers of two."""
    def e4m3(*shape):
        b = rng.integers(0, 256, size=shape, dtype=np.uint8)
        return np.where((b & 0x7F) == 0x7F, b ^ 1, b).astype(np.uint8)

    w = rng.standard_normal((n, H)).astype(np.float32).view(np.uint32)
    ks = (np.float32(2.0) ** rng.integers(-12, 4, size=(n, N))).astype(np.float32).view(np.uint32)
    return [e4m3(n, H, Dd), w, e4m3(n, N, Dd), ks]


@pytest.mark.slow
def test_its_word_view_is_v3_on_finite_words() -> None:
    H, Dd, N = 16, 32, 3
    words = _finite_words(np.random.default_rng(5), 64, H, Dd, N)
    got = np.asarray(BS.evaluate_words(bind(BD.DsaMqaLogitsV4, N=N, H=H, D=Dd), *words), dtype=np.uint64).reshape(64, N)
    want = np.asarray(evaluate_batch(bind(D.DsaMqaLogitsV3, N=N, H=H, D=Dd), *words), dtype=np.uint64).reshape(64, N)
    assert (got == want).all()
