"""torch's flash attention backward on cc 12.0 (`fa2_train_bwd`): `train_sdpa_backward` binds the head at each captured shape,
with the kernel's dQ buffers at that B and H, and refuses the rest.  The Definitions give the capture's dQ, dK and dV words
(`fixtures/fa2_train_bwd_sm120.npz`, cut by `fixtures/fa2_train_bwd_sm120.py` from the capture `cc12_fa2_backward_capture`
cites) on rows and keys that reach each part of the kernel: rows that see one, a few and every key of their diagonal block, a
row over two key blocks, a row over five whose word tells the kernel's buffers from the sum in the key blocks' order, and a key
block through its two query blocks, the one on the diagonal last.  A small head is its rows and keys in the kernel's order, the
head at a stand-in shape's statics keeps its signature and builds, and each Definition on bits (`boolean_fa2_train_bwd`) gives
its words on random inputs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import evaluate
from verity.circuits.types import Array, Tuple
from verity.definitions.registry import boolean_fa2_train_bwd as BX
from verity.definitions.registry import fa2_train_bwd as W
from verity.definitions.registry.prims import BF16, F32

FIXTURE = Path(__file__).parent / "fixtures" / "fa2_train_bwd_sm120.npz"
BM, BN = 64, 128


def _ints(a) -> list[int]:
    return [int(x) for x in np.asarray(a).ravel()]


def test_train_sdpa_backward_binds_each_captured_shape() -> None:
    for H, NS in ((8, 12), (12, 8), (16, 6), (24, 4)):
        assert W.train_sdpa_backward(2, H, 1024, 64).id == f"Fa2TrainBwdHead_v1{{T=1024,D=64,BM=64,BN=128,NS={NS}}}"
    for shape in ((2, 4, 1024, 64), (2, 12, 512, 64), (2, 12, 1024, 128)):
        with pytest.raises(ValueError, match="is not captured"):
            W.train_sdpa_backward(*shape)
    assert "whole key blocks" in W.Fa2TrainBwdHead.refuse({"T": 100, "D": 64, "BM": 64, "BN": 128, "NS": 4})
    assert "a multiple of BM" in W.Fa2TrainBwdHead.refuse({"T": 1024, "D": 64, "BM": 48, "BN": 128, "NS": 4})
    assert "multiple of 32" in W.Fa2TrainBwdPre.refuse({"D": 16})
    assert "1 to BN" in W.Fa2TrainBwdRow.refuse({"D": 64, "BN": 128, "NVIS": 129})
    assert "k16 steps" in W.Fa2TrainBwdKey.refuse({"D": 64, "BM": 40})
    assert W.Fa2TrainBwdRow.refuse({"D": 64, "BN": 128, "NVIS": 128}) is None


def _loop(q, k, v, do, o, lse, bm: int, bn: int):
    """The kernel's loop over rows q, do, o, lse against keys k, v, the first key at the first row's key block's start and the
    last key block ending at the last row: each row's dQ parts in the key blocks' order, and each key's (dK, dV)."""
    T, D = q.shape
    first = T - len(k)
    pre = [evaluate(bind(W.Fa2TrainBwdPre, D=D), _ints(do[r]), _ints(o[r]), [int(lse[r])]) for r in range(T)]
    parts: list[list[int]] = [[] for _ in range(T)]
    dk, dv = [], []
    for n in range(len(k) // bn):
        keys = slice(n * bn, (n + 1) * bn)
        acc: list[list[int]] = []
        for m in range(T // bm - 1, (first + n * bn) // bm - 1, -1):
            p, ds = [], []
            for r in range(m * bm, (m + 1) * bm):
                row = bind(W.Fa2TrainBwdRow, D=D, BN=bn, NVIS=min(bn, r - first - n * bn + 1))
                out = _ints(evaluate(row, _ints(q[r]), _ints(k[keys]), _ints(v[keys]), _ints(do[r]), [pre[r][0]], [pre[r][1]]))
                p.append(out[:bn])
                ds.append(out[bn:2 * bn])
                parts[r] += out[2 * bn:]
            block = (_ints(do[m * bm:(m + 1) * bm]), _ints(q[m * bm:(m + 1) * bm]))
            for c in range(bn):
                col = ([x[c] for x in p], [x[c] for x in ds])
                if len(acc) < bn:
                    acc.append(_ints(evaluate(bind(W.Fa2TrainBwdKeyFirst, D=D, BM=bm), *col, *block)))
                else:
                    acc[c] = _ints(evaluate(bind(W.Fa2TrainBwdKey, D=D, BM=bm), *col, *block, acc[c]))
        for c in range(bn):
            out = _ints(evaluate(bind(W.Fa2TrainBwdDkDv, D=D), acc[c]))
            dk.append(out[:D])
            dv.append(out[D:])
    return parts, dk, dv


def _parts(d, case: str, i: int, r: int) -> list[int]:
    """Row r's dQ parts of the key blocks it sees, in the key blocks' order."""
    D = d[f"{case}/q"].shape[1]
    q, do = _ints(d[f"{case}/q"][i]), _ints(d[f"{case}/do"][i])
    pre = evaluate(bind(W.Fa2TrainBwdPre, D=D), do, _ints(d[f"{case}/o"][i]), [int(d[f"{case}/lse"][i])])
    parts: list[int] = []
    for b in range(r // BN + 1):
        keys = slice(b * BN, (b + 1) * BN)
        row = bind(W.Fa2TrainBwdRow, D=D, BN=BN, NVIS=min(BN, r - b * BN + 1))
        out = evaluate(row, q, _ints(d[f"{case}/k"][keys]), _ints(d[f"{case}/v"][keys]), do, [pre[0]], [pre[1]])
        parts += _ints(out[2 * BN:])
    return parts


def test_each_row_s_dq_is_the_captured_kernels_words() -> None:
    d = np.load(FIXTURE)
    seen = []
    for case in sorted({k.rsplit("/", 1)[0] for k in d.files if k.startswith("rows/")}):
        D, NS = d[f"{case}/q"].shape[1], int(d[f"{case}/ns"])
        for i, r in enumerate(d[f"{case}/rows"].tolist()):
            NP = r // BN + 1
            parts = _parts(d, case, i, r)
            assert _ints(evaluate(bind(W.Fa2TrainBwdDq, D=D, NP=NP, NS=NS), parts)) == _ints(d[f"{case}/dq"][i]), (case, r)
            if NP > NS:
                in_order = _ints(evaluate(bind(W.Fa2TrainBwdDq, D=D, NP=NP, NS=NP), parts))
                assert in_order != _ints(d[f"{case}/dq"][i]), (case, r)
            seen.append((case, r, NP, NS))
    assert seen == [("rows/xl/wide", 0, 1, 4), ("rows/xl/wide", 5, 1, 4), ("rows/xl/wide", 127, 1, 4),
                    ("rows/xl/wide", 130, 2, 4), ("rows/xl/wide", 621, 5, 4)]


@pytest.mark.slow
def test_a_key_block_s_dk_and_dv_are_the_captured_kernels_words() -> None:
    d = np.load(FIXTURE)
    case = "block/l16/unit"
    _, dk, dv = _loop(*(d[f"{case}/{f}"] for f in ("q", "k", "v", "do", "o", "lse")), BM, BN)
    assert dk == [_ints(x) for x in d[f"{case}/dk"]]
    assert dv == [_ints(x) for x in d[f"{case}/dv"]]


def _bf16(rng, shape, scale=1.0):
    x = (rng.standard_normal(shape) * scale).astype(np.float32)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def _f32(rng, shape, scale=1.0):
    return (rng.standard_normal(shape) * scale).astype(np.float32).view(np.uint32)


def test_a_head_is_its_rows_and_keys_in_the_kernels_order() -> None:
    """A head of 64 rows at D = 32, BM = 16, BN = 32 and two dQ buffers, against the kernel's loop written out here (`_loop`):
    key block n's query blocks from the last down to its own first, in each the rows and then the keys, then the keys' ends;
    a row's parts in the key blocks' order into its buffers."""
    T, D, bm, bn, NS = 64, 32, 16, 32, 2
    rng = np.random.default_rng(4207)
    do, q, k, v, o = (_bf16(rng, (T, D)) for _ in range(5))
    lse = _f32(rng, (T,), 2.0)
    head = _ints(evaluate(bind(W.Fa2TrainBwdHead, T=T, D=D, BM=bm, BN=bn, NS=NS), *map(_ints, (do, q, k, v, o, lse))))
    parts, dk, dv = _loop(q, k, v, do, o, lse, bm, bn)
    dq = [x for r in range(T) for x in _ints(evaluate(bind(W.Fa2TrainBwdDq, D=D, NP=r // bn + 1, NS=NS), parts[r]))]
    assert head[:T * D] == dq
    assert head[T * D:2 * T * D] == [x for row in dk for x in row]
    assert head[2 * T * D:] == [x for row in dv for x in row]


def test_the_head_at_a_stand_in_shape_keeps_its_signature_and_builds(monkeypatch) -> None:
    """At T = 16, D = 32 and blocks of 16, the statics a training step's frontend binds the head at when a test makes its shape a
    route: a stand-in entry of `TRAIN_SDPA_BWD` binds through `train_sdpa_backward`, the head takes (dO, q, k, v, O, LSE) and
    returns (dQ, dK, dV) in the op's order, it builds on words and on bits, and on words it is the kernel's loop at those statics
    with one dQ buffer as with the stand-in's."""
    T, D, bm = 16, 32, 16
    monkeypatch.setitem(W.TRAIN_SDPA_BWD, (2, 2, T, D), (bm, bm))
    stand_in = W.train_sdpa_backward(2, 2, T, D)
    one = bind(W.Fa2TrainBwdHead, T=T, D=D, BM=bm, BN=bm, NS=1)
    assert stand_in.id == f"Fa2TrainBwdHead_v1{{T={T},D={D},BM={bm},BN={bm},NS=47}}"
    rows = Array(T, Array(D, BF16))
    for head in (stand_in, one):
        assert head.signature() == (tuple((n, rows) for n in ("do", "q", "k", "v", "o")) + (("lse", Array(T, F32)),),
                                    Tuple(rows, rows, rows))
    assert bind(BX.Fa2TrainBwdHead, T=T, D=D, BM=bm, BN=bm, NS=1).gates > one.gates > 0
    rng = np.random.default_rng(4207)
    do, q, k, v, o = (_bf16(rng, (T, D)) for _ in range(5))
    lse = _f32(rng, (T,), 2.0)
    parts, dk, dv = _loop(q, k, v, do, o, lse, bm, bm)
    want = [x for r in range(T) for x in _ints(evaluate(bind(W.Fa2TrainBwdDq, D=D, NP=1, NS=1), parts[r]))]
    want += [x for row in dk + dv for x in row]
    for head in (stand_in, one):
        assert _ints(evaluate(head, *map(_ints, (do, q, k, v, o, lse)))) == want, head.id


def test_on_bits_each_is_its_words() -> None:
    rng = np.random.default_rng(4207)
    N, D, bm, bn = 3, 32, 16, 16
    do, o, q = _bf16(rng, (N, D)), _bf16(rng, (N, D)), _bf16(rng, (N, D), 3.0)
    lse = _f32(rng, (N,), 4.0)
    lse[0] = 0xFF800000
    k, v = _bf16(rng, (N, bn, D), 3.0), _bf16(rng, (N, bn, D))
    dpsum, ms = _f32(rng, (N,)), _f32(rng, (N,), 4.0)
    p, ds = _bf16(rng, (N, bm)), _bf16(rng, (N, bm))
    dob, qb = _bf16(rng, (N, bm, D)), _bf16(rng, (N, bm, D))
    acc = _f32(rng, (N, 2, D))
    parts = _f32(rng, (N, 5, D))
    for word, on_bits, args in (
            (bind(W.Fa2TrainBwdPre, D=D), bind(BX.Fa2TrainBwdPre, D=D), (do, o, lse)),
            (bind(W.Fa2TrainBwdRow, D=D, BN=bn, NVIS=5), bind(BX.Fa2TrainBwdRow, D=D, BN=bn, NVIS=5), (q, k, v, do, dpsum, ms)),
            (bind(W.Fa2TrainBwdKeyFirst, D=D, BM=bm), bind(BX.Fa2TrainBwdKeyFirst, D=D, BM=bm), (p, ds, dob, qb)),
            (bind(W.Fa2TrainBwdKey, D=D, BM=bm), bind(BX.Fa2TrainBwdKey, D=D, BM=bm), (p, ds, dob, qb, acc)),
            (bind(W.Fa2TrainBwdDkDv, D=D), bind(BX.Fa2TrainBwdDkDv, D=D), (acc,)),
            (bind(W.Fa2TrainBwdDq, D=D, NP=5, NS=2), bind(BX.Fa2TrainBwdDq, D=D, NP=5, NS=2), (parts,))):
        bits = BS.evaluate_words(on_bits, *args)
        for i in range(N):
            assert _ints(bits[i]) == _ints(evaluate(word, *(_ints(a[i]) for a in args))), (word.id, i)
