"""torch's flash attention forward on cc 12.0 (`fa2_train`): `train_sdpa` binds the unsplit head at each captured shape the
kernel heuristic leaves unsplit and the four-split head at the one it splits, and refuses the rest.  Each row Definition gives
the capture's O and LSE words (`fixtures/fa2_train_sm120.npz`, cut from the captures `cc12_fa2_forward_capture` cites) on rows
that reach each part of the kernels: a diagonal block of one key, of a partial and a full lane set and of every key, two blocks
(the rescale), one and two of the four splits and the combine; on heads whose words tell each flush-to-zero apart from none: the
f32 FADD / FMUL / FFMA (``tinyv``: subnormal outputs), MUFU.EX2 (``ex2sub``: subnormal probabilities) and the combine's expf
(``expsub``).  The diagonal block and the combine on bits (`boolean_fa2_train`) give their words on random inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import evaluate
from verity.definitions.registry import boolean_fa2_train as BX
from verity.definitions.registry import fa2_train as W

FIXTURE = Path(__file__).parent / "fixtures" / "fa2_train_sm120.npz"


def test_train_sdpa_routes_each_captured_shape() -> None:
    assert W.train_sdpa(2, 8, 1024, 64).id == "Fa2TrainSplitHead_v1{T=1024,D=64,BN=256}"
    for H in (12, 16, 24):
        assert W.train_sdpa(2, H, 1024, 64).id == "Fa2TrainHead_v1{T=1024,D=64,BN=128}"
    for shape in ((2, 4, 1024, 64), (2, 12, 512, 64), (2, 12, 1024, 128)):
        with pytest.raises(ValueError, match="is not captured"):
            W.train_sdpa(*shape)
    assert "whole blocks" in W.Fa2TrainHead.refuse({"T": 100, "BN": 128})
    assert "4 splits" in W.Fa2TrainSplitHead.refuse({"T": 1024, "BN": 128})
    assert "1 to BN" in W.Fa2TrainFirst.refuse({"BN": 16, "NVIS": 17})
    assert W.Fa2TrainRow.refuse({"D": 64, "BN": 128, "NB": 8, "NVIS": 128}) is None


def _rows():
    d = np.load(FIXTURE)
    for case in sorted({k.rsplit("/", 1)[0] for k in d.files}):
        BN = 256 if case.startswith("s/") else 128
        k, v = d[f"{case}/k"], d[f"{case}/v"]
        for i, r in enumerate(d[f"{case}/rows"].tolist()):
            n = (r // BN + 1) * BN
            pad = np.zeros((max(0, n - len(k)), k.shape[1]), np.uint16)
            yield (case, r, BN, d[f"{case}/q"][i], np.concatenate([k, pad])[:n], np.concatenate([v, pad])[:n],
                   d[f"{case}/o"][i].tolist(), int(d[f"{case}/lse"][i]))


def test_each_row_is_the_captured_kernels_words() -> None:
    """Keys past a row's diagonal are zeros here: none is read as a key, and a masked key's (finite) V row meets a probability
    of +0, so its product is a zero the accumulator absorbs, as the capture's own rows were."""
    seen = set()
    for case, r, BN, q, k, v, o, lse in _rows():
        D = q.shape[0]
        if case.startswith("s/"):
            row = bind(W.Fa2TrainSplitRow, D=D, BN=BN, NS=r // BN + 1, NVIS=r % BN + 1)
        else:
            row = bind(W.Fa2TrainRow, D=D, BN=BN, NB=r // BN + 1, NVIS=r % BN + 1)
        out = [int(x) for x in evaluate(row, [int(x) for x in q], [int(x) for x in k.ravel()], [int(x) for x in v.ravel()])]
        assert out[:D] == o, (case, r)
        assert out[D] == lse, (case, r, hex(out[D]), hex(lse))
        seen.add(case)
    assert seen == {"m/ex2sub", "m/tinyv", "m/wide", "s/expsub", "s/tinyv"}


def _bf16(rng, shape, scale=3.0):
    x = (rng.standard_normal(shape) * scale).astype(np.float32)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def test_the_block_and_the_combine_on_bits_are_their_words() -> None:
    rng = np.random.default_rng(4207)
    N, D, BN = 6, 16, 16
    q, k, v = _bf16(rng, (N, D)), _bf16(rng, (N, BN, D)), _bf16(rng, (N, BN, D))
    first = bind(W.Fa2TrainFirst, D=D, BN=BN, NVIS=5)
    bits = BS.evaluate_words(bind(BX.Fa2TrainFirst, D=D, BN=BN, NVIS=5), q, k, v)
    for i in range(N):
        want = evaluate(first, [int(x) for x in q[i]], [int(x) for x in k[i].ravel()], [int(x) for x in v[i].ravel()])
        assert [int(x) for x in bits[i]] == [int(x) for x in want], i
    o = (rng.standard_normal((N, 4, D)).astype(np.float32)).view(np.uint32)
    lse = (rng.standard_normal((N, 4)).astype(np.float32) * 4).view(np.uint32)
    lse[0, 2:] = 0xFF800000
    lse[1, :] = 0xFF800000
    combine = bind(W.Fa2TrainCombine, D=D)
    bits = BS.evaluate_words(bind(BX.Fa2TrainCombine, D=D), o, lse)
    for i in range(N):
        want = evaluate(combine, [int(x) for x in o[i].ravel()], [int(x) for x in lse[i]])
        assert [int(x) for x in bits[i]] == [int(x) for x in want], i
