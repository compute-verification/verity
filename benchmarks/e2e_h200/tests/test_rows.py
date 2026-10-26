"""The row units on the words where a careless reference goes wrong: a NaN in a row and a tied argmax. The replay's rows
(`units`, the replay lane's templates through `replay.template`) and the driver's ops (`ops`, torch on the CPU) agree with
the Definitions' semantics: TokenSelect never takes a NaN and takes the first of equal maxima (+0 = -0), and a quantized NaN
is code 0 under a scale of 1."""
from __future__ import annotations

import numpy as np
import torch

from e2e_h200 import ops as O
from e2e_h200 import units as U

V = 256
NAN, NEG_NAN, NEG_INF = 0x7FC0, 0xFFC1, 0xFF80


def bf16_words(values: list[float]) -> np.ndarray:
    return torch.tensor(values, dtype=torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def logits(n: int = 6) -> tuple[np.ndarray, list[int]]:
    """Rows of V logits and each row's TokenSelect index."""
    rows = np.tile(bf16_words([-1.0]), (n, V))
    want = []
    rows[0, [3, 7]] = bf16_words([2.0, 2.0])                    # a tie: the first
    want.append(3)
    rows[1, :] = bf16_words([-2.0])
    rows[1, [2, 5]] = [0x8000, 0x0000]                          # -0 then +0: equal, the first
    want.append(2)
    rows[2, 4], rows[2, 9] = NAN, bf16_words([3.0])[0]           # a NaN before the greatest: never taken
    want.append(9)
    rows[3, 0], rows[3, 9] = NEG_NAN, bf16_words([3.0])[0]       # a NaN first: TokenSelect starts from it and keeps 0
    want.append(0)
    rows[4, :] = NEG_INF                                        # nothing above -inf
    want.append(0)
    rows[5, [1, 200, 250]] = [NAN, bf16_words([5.0])[0], bf16_words([5.0])[0]]
    want.append(200)
    return rows, want


def test_a_nan_and_a_tied_argmax_agree_with_the_definition():
    rows, want = logits()
    assert U.argmax_row(rows).tolist() == want
    assert O.argmax(torch.from_numpy(rows.view(np.int16)).view(torch.bfloat16)).tolist() == want
    t = U.tpl("ArgmaxCombine_v1", V=V, C=V)
    assert t.key == f"ArgmaxCombine_v1{{V={V},C={V}}}"


def test_a_nan_in_a_quantized_row_agrees_with_the_definition():
    """A NaN makes the row's max |x| a NaN: the scale is 1 and the NaN's code 0; the other codes are RNE(x), saturated."""
    for relu in (False, True):
        x = np.stack([bf16_words([1.5, -2.5, 300.0, -0.0, 0.5, 7.0, -130.0, 2.5]) for _ in range(2)])
        x[0, 3] = NAN
        x[1, 1] = NEG_NAN
        codes, scale = U.quant_row(x, relu)
        q = O.quantize(torch.from_numpy(x.view(np.int16)).view(torch.bfloat16), relu)
        assert scale.tolist() == [0x3F800000] * 2
        assert q.scale.view(torch.int32).tolist() == [0x3F800000] * 2
        assert np.array_equal(q.codes.numpy().view(np.uint8), codes), relu
        assert codes[0, 3] == 0 and codes[1, 1] == 0
        assert codes.view(np.int8)[0].tolist() == ([2, 0, 127, 0, 0, 7, 0, 2] if relu else [2, -2, 127, 0, 0, 7, -128, 2])


def test_a_softmax_row_with_a_nan_or_a_tie_takes_its_max_as_the_definition():
    x = np.tile(bf16_words([0.25, 1.0, 1.0, -3.0]), (2, 1))
    x[1, 2] = NAN
    m = U.tpl("SoftmaxMax_v1", N=4).evaluate(x).reshape(-1)
    assert m.tolist() == [bf16_words([1.0])[0], 0x7FFF]
    p = U.softmax_row(x)
    assert p[0, 1] == p[0, 2] and p.shape == x.shape
