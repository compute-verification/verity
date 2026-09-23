"""The device prover (``pcs.prove``) against the prototype prover (``pcs.prove_reference``): same coins -> same proof bytes, plus
the device kernels against numpy.  GPU only (``pytest backends/direct/ligerito/proto -q`` on the pod)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from . import claims as cl
from . import ext
from .dims import Dims
from .ext import D, P
from .pcs import Proof, prove, prove_reference, verify
from .transcript import LocalCoins

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="device prover")
DEV = "cuda"
RNG = np.random.default_rng(1729)


def _inst(n):
    f = torch.as_tensor(RNG.integers(0, P, size=1 << n, dtype=np.int64).astype(np.int32), device=DEV)
    z = RNG.integers(0, P, size=(n, D), dtype=np.int64)
    return f, z


@pytest.mark.parametrize("n,kprime,q,rates", [(12, [4, 3, 3], [12, 12, 12], 2), (14, [5, 4, 3], [24, 20, 16], [1, 2, 2]),
                                              (16, [4, 3, 3, 3], [30, 17, 17, 18], [2, 4, 4, 4]), (11, [5], [9], 2),
                                              (18, [6, 4, 4], [64, 48, 40], [2, 2, 2])])
def test_fast_matches_reference_bytes(n, kprime, q, rates):
    dims = Dims(n, kprime, q, rates)
    f, z = _inst(n)
    a = prove(f, z, dims)
    b = prove_reference(f, z, dims)
    assert a.to_bytes() == b.to_bytes()
    ok, why = verify(Proof.from_bytes(a.to_bytes()), z)
    assert ok, why
    # the same with non-Fiat-Shamir coins (the call order is the protocol)
    a = prove(f, z, dims, coins=LocalCoins(7))
    b = prove_reference(f, z, dims, coins=LocalCoins(7))
    assert a.to_bytes() == b.to_bytes()


@pytest.mark.parametrize("n,kprime,q,rates", [(14, [5, 4, 3], [24, 20, 16], [1, 2, 2]), (15, [6, 4], [30, 17], 2),
                                              (12, [4], [9], 2), (18, [7, 4, 4], [64, 48, 24], [2, 2, 2])])
def test_streamed_commit_matches_reference(n, kprime, q, rates, monkeypatch):
    monkeypatch.setenv("LIGERITO_STREAM", "1")
    dims = Dims(n, kprime, q, rates)
    f, z = _inst(n)
    a = prove(f, z, dims)
    monkeypatch.setenv("LIGERITO_STREAM", "0")
    b = prove_reference(f, z, dims)
    assert a.to_bytes() == b.to_bytes()


def _mle(f: torch.Tensor, z: np.ndarray) -> np.ndarray:
    e = cl.eq_vector(z)                                                   # (N, 6), x little-endian
    fv = f.to(torch.int64).cpu().numpy()
    return ext.sum_ext(ext.mul(e, fv[:, None] * np.array([1, 0, 0, 0, 0, 0])), axis=0)


def _relation_coins(root: bytes):
    from .transcript import FiatShamirCoins
    tr = FiatShamirCoins()
    tr.absorb(b"statement", b"relation statement")
    tr.absorb(b"root", root)
    tr.absorb(b"relation", b"zero-check messages")
    tr.challenge(b"tau", 3)
    return tr


@pytest.mark.parametrize("m,stream", [(1, False), (4, False), (3, True), (2, None)])
def test_commit_first_multi_claim(m, stream):
    from .pcs import commit, evaluate, open as pcs_open, verify_open
    n = 14
    dims = Dims(n, [5, 4, 3], [24, 20, 16], [1, 2, 2])
    f, _ = _inst(n)
    zs = RNG.integers(0, P, size=(m, n, D), dtype=np.int64)
    # the relation's view: commit, bind the root and its own rounds, then ONE evaluation proof for all claims
    com = commit(f, dims, stream=stream)
    assert (com.U1 is None) == bool(stream)
    values = evaluate(com, zs)
    for j in range(m):
        assert np.array_equal(values[j], _mle(f, zs[j]))
    pf = pcs_open(com, zs, values, _relation_coins(com.root))
    pf_nocache = pcs_open(com, zs, None, _relation_coins(com.root))           # without evaluate's cached contractions
    assert pf_nocache.to_bytes() == pf.to_bytes()
    assert pf.v.shape == (m, D) and np.array_equal(pf.v, values)
    pf2 = Proof.from_bytes(pf.to_bytes())
    assert pf2.to_bytes() == pf.to_bytes() and pf2.order == "commit-first"
    ok, why = verify_open(pf2, zs, values, _relation_coins(pf2.roots[0]))
    assert ok, why
    # negatives: a wrong value, wrong points, a different relation transcript, the standalone prefix, the legacy verifier path
    bad = values.copy()
    bad[m - 1, 0] = (bad[m - 1, 0] + 1) % P
    assert not verify_open(pf2, zs, bad, _relation_coins(pf2.roots[0]))[0]
    assert not verify_open(pf2, zs[::-1] if m > 1 else (zs + 1) % P, values, _relation_coins(pf2.roots[0]))[0]
    assert not verify_open(pf2, zs, values, _relation_coins(b"\0" * 32))[0]
    assert not verify(pf2, zs)[0]


@pytest.mark.parametrize("m", [1, 3])
def test_commit_first_standalone(m):
    from .pcs import prove_commit_first
    n = 12
    dims = Dims(n, [4, 3, 3], [12, 12, 12], 2)
    f, _ = _inst(n)
    zs = RNG.integers(0, P, size=(m, n, D), dtype=np.int64)
    pf = Proof.from_bytes(prove_commit_first(f, zs, dims).to_bytes())
    ok, why = verify(pf, zs)
    assert ok, why
    assert pf.v.shape == (m, D)
    assert not verify(pf, zs, v=(pf.v + 1) % P)[0]


def test_stream_group_plan():
    from .pcs import _stream_group
    big = torch.cuda.get_device_properties(0).total_memory > (40 << 30)
    assert _stream_group(Dims(29, [7, 4, 4, 4], [192] * 4, 2), DEV) == 0
    if not big:     # 24 GB parts: S at 2^30 streams in 32-column groups, default at 2^30 in 128-column groups
        assert _stream_group(Dims(30, [6, 4, 4, 3, 3], [191, 143, 143, 143, 144], [2, 4, 4, 4, 4]), DEV) == 32
        assert _stream_group(Dims(30, [8, 4, 4, 4], [192] * 4, 2), DEV) == 128
        assert _stream_group(Dims(30, [6, 4, 4, 3, 3], [312, 191, 193, 194, 195], [1, 2, 2, 2, 2]), DEV) == 0


@pytest.mark.parametrize("ns,nc,n1,B", [(128, 4, 64, 3), (256, 4, 256, 2), (512, 2, 1024, 2), (1024, 4, 1024, 2),
                                        (1024, 2, 2048, 1), (2048, 2, 2048, 1), (2048, 4, 512, 2), (1024, 16, 128, 2)])
def test_pass_a_matches_generic(ns, nc, n1, B):
    from .ntt_gpu import ntt_axis, ntt_pass_a, post_twiddles
    n2 = ns * nc
    x = torch.randint(0, P, (B, ns, n1), dtype=torch.int64, device=DEV).to(torch.int32)
    ref = ntt_axis(x, n2, out=torch.empty((B, n2, n1), dtype=torch.int32, device=DEV), post_tw=post_twiddles(n1, n2, DEV))
    got = ntt_pass_a(x, n2, n1 * n2, out=torch.empty((B, n2, n1), dtype=torch.int32, device=DEV))
    assert torch.equal(got, ref)


def test_contract_geom_and_planes():
    from . import device as dv
    for Pn, R, T in ((96, 1 << 12, 70), (192, 1 << 10, 130), (48, 1 << 9, 5), (128, 1 << 13, 64)):
        A = torch.randint(0, P, (Pn, R), dtype=torch.int64, device=DEV).to(torch.int32)
        etas = RNG.integers(1, P, size=T, dtype=np.int64)
        got = (dv.contract_geom(A, etas) % P).cpu().numpy()
        G = np.stack([cl.powers(int(e), R) for e in etas], axis=1)          # (R, T)
        from .linalg import matmul_mod_p_np
        ref = matmul_mod_p_np(A.to(torch.int64).cpu().numpy(), G)
        assert np.array_equal(got, ref), (Pn, R, T)
        for W in (6, 12, 18, 24):
            Bp = torch.randint(0, P, (W, R), dtype=torch.int64, device=DEV).to(torch.int32)
            got = (dv.contract_planes(A, Bp) % P).cpu().numpy()
            ref = matmul_mod_p_np(A.to(torch.int64).cpu().numpy(), Bp.to(torch.int64).cpu().numpy().T)
            assert np.array_equal(got, ref), (Pn, R, W)


def test_contract_eq_multi():
    from . import device as dv
    from .linalg import matmul_mod_p_np
    for Pn, k, m in ((96, 12, 1), (40, 11, 3), (128, 9, 5), (16, 13, 9)):
        A = torch.randint(0, P, (Pn, 1 << k), dtype=torch.int64, device=DEV).to(torch.int32)
        zs = RNG.integers(0, P, size=(m, k, D), dtype=np.int64)
        got = (dv.contract_eq(A, zs) % P).cpu().numpy()
        An = A.to(torch.int64).cpu().numpy()
        for j in range(m):
            ref = matmul_mod_p_np(An, cl.eq_vector(zs[j]))                   # (P, 6)
            assert np.array_equal(got[j], ref), (Pn, k, m, j)


def test_fold_ext_and_eq_table():
    from . import device as dv
    from .linalg import contract_rows_ext, eq_planes
    C, R = 16, 1 << 11
    M = torch.randint(0, P, (D, C, R), dtype=torch.int64, device=DEV).to(torch.int32)
    rb = RNG.integers(0, P, size=(C, D), dtype=np.int64)
    assert torch.equal(dv.fold_ext(M, rb), contract_rows_ext(M, rb))
    for k in (5, 11, 14):
        z = RNG.integers(0, P, size=(k, D), dtype=np.int64)
        assert torch.equal(dv.eq_table(z, DEV).to(torch.int64), eq_planes(z, DEV).to(torch.int64))


def test_device_sumcheck_matches_host():
    from . import device as dv
    k, kp, Tg = 4, 5, 9
    C = 1 << kp
    z = RNG.integers(0, P, size=(k + kp, D), dtype=np.int64)
    etas = RNG.integers(1, P, size=Tg, dtype=np.int64)
    coef = RNG.integers(0, P, size=(Tg + 1, D), dtype=np.int64)
    ue = RNG.integers(0, P, size=(D * C, D), dtype=np.int64)
    ug = RNG.integers(0, P, size=(D * C, Tg), dtype=np.int64)
    # the host prover's view: u_eq[c] = combine(ue), u_geom[t][c]_a = ug[a C + c, t]
    u_eq = ext.zero((C,))
    for a in range(D):
        for b in range(D):
            kk = a + b
            fac = ext.BETA if kk >= D else 1
            u_eq[:, kk % D] = (u_eq[:, kk % D] + fac * ue[a * C:(a + 1) * C, b]) % P
    u_g = np.stack([np.stack([ug[a * C:(a + 1) * C, t] for a in range(D)], axis=1) for t in range(Tg)], axis=0)
    terms = [cl.Term(coef[0], "eq", z=z, nvars=k + kp)] + [cl.Term(coef[1 + t], "geom", eta=int(etas[t]), nvars=k + kp)
                                                           for t in range(Tg)]
    host = cl.ColumnSumcheckProver(terms, [u_eq] + list(u_g), k, kp)
    colg = cl.geom_colparts(etas, k, kp)[:, :, 0]
    cole = terms[0].colpart(k, kp)
    dsc = dv.DeviceSumcheck(torch.as_tensor(ug, device=DEV), torch.as_tensor(ue, device=DEV), coef, colg, cole, C, DEV)
    for _ in range(kp):
        g_d = dsc.message()
        g_h = host.round_message()
        assert np.array_equal(g_d, g_h)
        r = RNG.integers(0, P, size=D, dtype=np.int64)
        dsc.fold(r)
        host.fold(r)
