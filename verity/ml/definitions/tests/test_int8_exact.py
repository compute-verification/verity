"""`registry.int8_exact`'s `Int8RowQuant_v3{K,R}` is `Int8RowQuant_v1{K}` with x as R rows of K/R values: word for word at K = 2048
with R = 1 and R = 32 on spread rows, a zero row (the floored scale) and a row of edge words, the same body as v1's, and `bind`
refuses an R that does not divide K.  Its Boolean version is `boolean_int8`'s `Int8RowQuant_v4{K,R}`, σ of v3, which is v3 on
bits."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation import evaluate, evaluate_batch
from verity.ml.circuits.types import Array
from verity.ml.definitions.boolean import boolean_version
from verity.ml.definitions.registry import boolean_int8 as BX
from verity.ml.definitions.registry import int8_exact as X
from verity.ml.definitions.registry.prims import BF16

K = 2048


def _rows() -> list[list[int]]:
    """bf16 words, every one finite: rows at three spreads, a zero row, and a row of subnormals, -0, +-1 and bf16's largest
    finite."""
    g = np.random.default_rng(K)
    out = []
    for spread in (0.5, 4.0, 30.0):
        f = (g.standard_normal(K) * np.exp(g.standard_normal(K) * spread / 4)).astype(np.float32)
        out.append((f.view(np.uint32) >> 16).astype(int).tolist())
    out.append([0] * K)
    edge = [0x0001, 0x8001, 0x8000, 0x3F80, 0xBF80, 0x007F] * (-(-K // 6))
    edge[K // 2] = 0x7F7F
    out.append(edge[:K])
    return out


ROWS = _rows()


@pytest.mark.parametrize("R", [1, 32])
def test_v3_is_v1_word_for_word(R):
    v1, v3 = bind(X.Int8RowQuant, K=K), bind(X.Int8RowQuantV3, K=K, R=R)
    for i, x in enumerate(ROWS):
        assert evaluate(v3, x) == evaluate(v1, x), i


@pytest.mark.parametrize("R", [1, 32])
def test_v3_is_v1s_body_with_x_as_R_rows(R):
    v1, v3 = bind(X.Int8RowQuant, K=K), bind(X.Int8RowQuantV3, K=K, R=R)
    (_, x), = v3.signature()[0]
    assert x == Array(R, Array(K // R, BF16)) and v3.signature()[1] == v1.signature()[1]

    def shape(d):
        b = d.body
        return ([(n.form, n.fn.id, [(a.type, a.refs.to_json()) for a in n.args], n.out, n.axes) for n in b.nodes],
                b.ret.to_json(), b.offsets)

    assert shape(v3) == shape(v1)


@pytest.mark.parametrize("statics", [dict(K=2048, R=3), dict(K=64, R=0), dict(K=0, R=1), dict(K=64, R=True), dict(K=64, R=2.0)])
@pytest.mark.parametrize("defn", [X.Int8RowQuantV3, BX.Int8RowQuantV4], ids=["v3", "v4"])
def test_bind_refuses_x_that_is_not_R_rows_of_K_over_R(defn, statics):
    with pytest.raises(ValueError, match="refuses this binding"):
        bind(defn, **statics)


@pytest.mark.parametrize("R", [1, 32])
def test_v3s_boolean_version_is_v4_its_sigma(R):
    word = bind(X.Int8RowQuantV3, K=K, R=R)
    bits = boolean_version(word)
    assert bits.id == f"Int8RowQuant_v4{{K={K},R={R}}}" and BL.is_boolean(bits) and TR.word_of(bits).id == word.id


@pytest.mark.parametrize("k, R", [(4, 2), (64, 32)])
def test_v4_is_v3_on_bits(k, R):
    word, bits = bind(X.Int8RowQuantV3, K=k, R=R), bind(BX.Int8RowQuantV4, K=k, R=R)
    x = np.array([r[:k] for r in ROWS], dtype=np.uint64).reshape(len(ROWS), R, k // R)
    want = np.asarray(evaluate_batch(word, x, kernel="reference"), dtype=np.uint64).reshape(len(ROWS), -1)
    assert BS.evaluate_words(bits, x).reshape(len(ROWS), -1).tolist() == want.tolist()
