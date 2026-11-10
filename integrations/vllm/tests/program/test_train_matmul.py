"""`rules.training_matmul.Matmul`'s products beside the mm: torch's mv as one row of its mm without a cc and as F.linear's one-row
GEMM at one, a bf16 GEMM whose reduction length is not a multiple of 16 padded with +0 words to whole k16 steps, an f32 mm under
allow_tf32 as cuBLAS's TF32 GEMM where the TF32 capture covers it, and the products whose order is cuBLAS's and no capture's
refused with the capture each needs."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from tests.program.test_train_step import CC, _bits, _data, _meta, _program
from verity.circuits.boolean.trace import word_of
from verity.definitions.numerics.serve3_reference import materialize
from verity.definitions.registry import bf16_gemm_check as BC
from verity.definitions.registry import cublas_bf16 as CB
from verity_vllm.program.frontend.derive_graph import derive_graph
from verity_vllm.program.frontend.outcomes import Unsupported
from verity_vllm.program.frontend.rules.training import training_rules

GPU = training_rules(cc=CC)
NOSPLIT = {"kernel": "nosplit"}


def _out(prog, *xs) -> np.ndarray:
    tab = materialize(prog, [_bits(x) for x in xs])
    return np.array([tab[g] for g in prog.circuit.root_outputs()], dtype=np.uint64)


def _u16(t: torch.Tensor) -> np.ndarray:
    return t.contiguous().view(torch.int16).numpy().astype(np.uint16)


def _pad(a: np.ndarray) -> np.ndarray:
    return np.pad(a, ((0, 0), (0, -a.shape[1] % 16)))


def _refused(f, *xs, rules=GPU) -> Unsupported:
    gm = make_fx(torch.func.functionalize(f, remove="mutations"), tracing_mode="fake")(*xs)
    u = derive_graph(gm, xs, mode="reference", rules=rules, name="T").unsupported
    assert isinstance(u, Unsupported) and u.reason == "numerics_unregistered", u
    return u


@pytest.mark.parametrize("N, K", [(24, 40), (3, 67), (7, 5)])
def test_mv_is_torch_cpus(N: int, K: int) -> None:
    """Without a cc, a bf16 `mv(A, x)` (and `A @ x`) is `LinearRef_v1` of x against A's rows, torch CPU's mv bit for bit, A
    row-major or a transposed view; an f32 one is refused, torch CPU's being its BLAS's sgemv, which isn't its mm's row there."""
    A, x, At = (_data(*s, seed=3 + i).to(torch.bfloat16) for i, s in enumerate(((N, K), (K,), (K, N))))
    for f in (lambda a, v: (torch.mv(a, v),), lambda a, v: (a @ v,)):
        prog = _program(f, A, x)
        assert [nd.fn.base_id for nd in prog.root.body.nodes[2:]] == ["LinearRef_v1"]
        assert _out(prog, A, x).tolist() == _bits(torch.mv(A, x))
    assert _out(_program(lambda b, v: (torch.mv(b.t(), v),), At, x), At, x).tolist() == _bits(torch.mv(At.t(), x))
    assert "BLAS's (sgemv)" in _refused(lambda a, v: (torch.mv(a, v),), A.float(), x.float(), rules=training_rules()).hint


def test_gpu_mv_is_a_one_row_gemm(monkeypatch) -> None:
    """At cc 12.0 a bf16 `mv(A, x)` is F.linear's GEMM at one row, x[1, K] @ A^T (torch's CUDA gemv is cuBLAS's GEMM with one
    column): a row-major A the forward GEMM against A's rows, a column-major one dX's, each its stand-in route's model bit for
    bit on cancelling words, which the two routes round apart."""
    N, K = 24, 48
    routes = {(1, K, N, "fwd"): NOSPLIT, (1, N, K, "dX"): {"kernel": "serialk", "SPLIT_K": 3}}
    for key, route in routes.items():
        monkeypatch.setitem(CB.TRAIN_ROUTES, key, route)
    xw, ww = BC.route_probe_rows(1, K, N, np.random.default_rng(117))[1]
    A, x = (torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16) for a in (ww, xw[0]))
    prog = _program(lambda a, v: (torch.mv(a, v),), A, x, rules=GPU)
    assert [nd.fn.id for nd in prog.root.body.nodes[2:]] == [word_of(CB.train_linear("fwd", 1, K, N)).id]
    assert np.array_equal(_out(prog, A, x).astype(np.uint16), BC.cublas_route_rows(NOSPLIT, _u16(x)[None], _u16(A))[0])
    At = A.t().contiguous()
    prog = _program(lambda b, v: (torch.mv(b.t(), v),), At, x, rules=GPU)
    assert word_of(CB.train_linear("dX", 1, N, K)).id in [nd.fn.id for nd in prog.root.body.nodes]
    want = BC.cublas_route_rows(routes[(1, N, K, "dX")], _u16(x)[None], _u16(A))[0]
    assert np.array_equal(_out(prog, At, x).astype(np.uint16), want)
    assert not np.array_equal(want, BC.cublas_route_rows(NOSPLIT, _u16(x)[None], _u16(A))[0]), "the two routes differ here"


def test_gpu_gemm_pads_k_to_whole_k16_steps(monkeypatch) -> None:
    """A bf16 GEMM whose row GEMM reduces over 40 words, at an unsplit stand-in route, is `GemmPad_v1{T=40, K=48}`: the route's
    model on the operands padded with +0 words at the back, for an mm at one row (a combination of directions, coef @ Z, whose
    reduction is over the directions) and at three, and for an mv."""
    K, N = 40, 24
    for key in ((1, N, K, "dX"), (3, N, K, "dX"), (1, K, N, "fwd")):
        monkeypatch.setitem(CB.TRAIN_ROUTES, key, NOSPLIT)
    pad = f"GemmPad_v1{{T={K},K=48,N={N},DOT=HopperBF16WgmmaDot16_v1}}"
    for M in (1, 3):
        c, Z = _data(M, K, seed=40).to(torch.bfloat16), _data(K, N, seed=41).to(torch.bfloat16)
        prog = _program(lambda a, b: (a @ b,), c, Z, rules=GPU)
        assert [nd.fn.id for nd in prog.root.body.nodes[2:]] == [pad]
        want = BC.cublas_route_rows(NOSPLIT, _pad(_u16(c)), _pad(_u16(Z).T))
        assert np.array_equal(_out(prog, c, Z).astype(np.uint16), want.reshape(-1)), M
    A, x = _data(N, K, seed=42).to(torch.bfloat16), _data(K, seed=43).to(torch.bfloat16)
    prog = _program(lambda a, v: (torch.mv(a, v),), A, x, rules=GPU)
    assert [nd.fn.id for nd in prog.root.body.nodes[2:]] == [pad]
    assert np.array_equal(_out(prog, A, x).astype(np.uint16), BC.cublas_route_rows(NOSPLIT, _pad(_u16(x)[None]), _pad(_u16(A)))[0])


def test_one_row_products_at_uncaptured_shapes_wait_for_their_capture() -> None:
    """A zeroth-order step's two bf16 products over 1,000 directions of 6,291,456 weights are one-row GEMMs no capture covers:
    coef @ Z, dX's GEMM at x (1, 6291456), W (1000, 6291456), whose row GEMM reduces over the 1,000 directions, and Z @ g, the
    forward GEMM at the same shape."""
    u = _refused(lambda c, z: (c @ z,), *_meta((1, 1000), (1000, 6291456)))
    assert "dX GEMM at x (1, 6291456), W (1000, 6291456)" in u.hint and "is not captured" in u.hint, u.hint
    u = _refused(lambda z, g: (torch.mv(z, g),), *_meta((1000, 6291456), (6291456,)))
    assert "aten.mv.default" in u.hint and "fwd GEMM at x (1, 6291456), W (1000, 6291456)" in u.hint, u.hint


def test_gpu_order_products_are_refused_with_their_capture() -> None:
    """At cc 12.0 a product whose order is cuBLAS's and no capture's is refused with the capture it needs: an f32 bmm with allow_tf32
    (`training_rules(tf32=True)`; an eigenvalue estimate's Gram bmm) or an f32 mm with it at a shape the TF32 capture doesn't
    cover, an f32 dot (a power iteration's),
    a bf16 dot and an f32 mv.  Without tf32 an f32 bmm is `LinearRef_v1` as without a cc, and without a cc a dot is refused too:
    torch CPU's f32 dot is its BLAS's, not its mm's row."""
    tf32 = training_rules(cc=CC, tf32=True)

    def f32(*shapes):
        return [torch.empty(*s, device="meta") for s in shapes]

    u = _refused(lambda a, b: (torch.bmm(a, b),), *f32((16, 512, 4096), (16, 4096, 512)), rules=tf32)
    assert "TF32 GEMM" in u.hint and "allow_tf32 at [16, 512, 4096] @ [16, 4096, 512]" in u.hint, u.hint
    assert "torch.mm's kernels" in _refused(lambda a, b: (a @ b,), *f32((64, 32), (32, 16)), rules=tf32).hint
    u = _refused(lambda a, b: (a @ b,), *f32((1000,), (1000,)))
    assert "cuBLAS's dot kernel" in u.hint and "torch.dot's words on cc 12.0 at [1000] @ [1000]" in u.hint, u.hint
    assert "a bfloat16 dot" in _refused(lambda a, b: (torch.dot(a, b),), *_meta((64,), (64,))).hint
    assert "cuBLAS's sgemv" in _refused(lambda a, v: (torch.mv(a, v),), *f32((24, 40), (40,))).hint
    assert "BLAS's (sdot)" in _refused(lambda a, b: (torch.dot(a, b),), *f32((64,), (64,)), rules=training_rules()).hint
    a, b = _data(2, 4, 8, seed=6), _data(2, 8, 3, seed=7)
    ids = [[nd.fn.id for nd in _program(lambda x, y: (torch.bmm(x, y),), a, b, rules=r).root.body.nodes[2:]]
           for r in (GPU, training_rules())]
    assert ids[0] == ids[1] and {i.split("{")[0] for i in ids[0]} == {"LinearRef_v1"}, ids
    assert tf32.name == "training-cc12.0-tf32" and GPU.name == "training-cc12.0"
    with pytest.raises(ValueError, match="tf32 is a GPU's"):
        training_rules(tf32=True)


def test_gpu_tf32_mm_is_cublas_tf32_gemm(monkeypatch) -> None:
    """With allow_tf32 an f32 mm at a shape the TF32 capture covers (`cublas_bf16.TRAIN_TF32`) is one batch of `GemmTf32_v1{K, N}`
    over a's rows, b row- or column-major (a rotation codec's y @ R and y @ R^T), and each word is the TF32 step's chain from +0
    (`mhc_twin.tf32_dot8`; checked at a stand-in shape, the captured ones being thousands of rows).  An operand that is not
    dense is refused with the capture it needs."""
    from verity.definitions.numerics import mhc_twin as MT

    tf32 = training_rules(cc=CC, tf32=True)
    meta = [torch.empty(4096, 64, device="meta"), torch.empty(64, 64, device="meta")]
    for f in (lambda a, r: (a @ r,), lambda a, r: (a @ r.t(),)):
        ids = {nd.fn.id for nd in _program(f, *meta, rules=tf32).root.body.nodes}
        assert "GemmTf32_v1{K=64,N=64}" in ids and not any(i.startswith("LinearRef") for i in ids), ids
    u = _refused(lambda a, r: (a[:, :64] @ r,), torch.empty(4096, 128, device="meta"), meta[1], rules=tf32)
    assert "TF32 GEMM" in u.hint and "no capture covers here" in u.hint, u.hint

    monkeypatch.setattr(CB, "TRAIN_TF32", CB.TRAIN_TF32 + ((3, 16, 4),))
    a, b = _data(3, 16, seed=8), _data(16, 4, seed=9)
    A, B = (np.asarray(_bits(t), np.uint32).reshape(t.shape) for t in (a, b))
    acc = np.zeros(12, np.uint32)
    for k in range(0, 16, 8):
        acc = MT.tf32_dot8(acc, np.repeat(A[:, k:k + 8], 4, axis=0), np.tile(B[k:k + 8].T, (3, 1)))
    for bm in (b, b.t().contiguous().t()):
        got = _out(_program(lambda x, y: (x @ y,), a, bm, rules=tf32), a, bm)
        assert got.astype(np.uint32).tolist() == acc.tolist()
