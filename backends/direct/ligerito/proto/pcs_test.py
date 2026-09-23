"""Ligerito PCS prototype: correctness at toy sizes (CPU or GPU), extension arithmetic, the tensor-claim sumcheck,
the 4-step encoder against ``field.ntt`` (GPU only), and the negatives (tampered opened row, wrong final vector,
wrong evaluation, tampered sumcheck message, tampered root).  ``pytest backends/direct/ligerito/proto -q``."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from . import claims as cl
from . import ext
from .dims import Dims, choose
from .ext import D, P
from .pcs import Proof, prove, verify

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RNG = np.random.default_rng(20260923)


def rand_ext(shape=()):
    return RNG.integers(0, P, size=tuple(shape) + (D,), dtype=np.int64)


def eval_multilinear(f: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Reference f(z) by folding (bit j of the index <-> z_j)."""
    cur = ext.from_base(f)
    for j in range(z.shape[0]):
        h = cur.shape[0] // 2
        lo, hi = cur[:h], cur[h:]
        # bit j is the LOW bit at this stage: interleave
        cur = cur.reshape(h, 2, D)
        lo, hi = cur[:, 0], cur[:, 1]
        one_minus = (ext.one() - z[j]) % P
        cur = ext.add(ext.mul(lo, one_minus), ext.mul(hi, z[j]))
    return cur[0]


# ------------------------------------------------------------------------------------------------ ext field

def test_ext_field_axioms():
    a, b, c = rand_ext(), rand_ext(), rand_ext()
    assert np.array_equal(ext.mul(a, b), ext.mul(b, a))
    assert np.array_equal(ext.mul(ext.mul(a, b), c), ext.mul(a, ext.mul(b, c)))
    assert np.array_equal(ext.mul(a, ext.add(b, c)), ext.add(ext.mul(a, b), ext.mul(a, c)))
    assert np.array_equal(ext.mul(a, ext.one()), a)
    assert np.array_equal(ext.mul(a, ext.inv(a)), ext.one())


def test_x6_minus_beta_irreducible():
    # 31 is a generator: neither a square nor a cube in F_p (6 | p - 1)
    assert pow(ext.BETA, (P - 1) // 2, P) != 1
    assert pow(ext.BETA, (P - 1) // 3, P) != 1
    x = ext.zero()
    x[1] = 1
    assert np.array_equal(ext.pow_int(x, P**6 - 1), ext.one())        # x is a unit of the field


# ------------------------------------------------------------------------------------------------ claims

def test_geom_term_is_powers():
    eta = 12345
    t = cl.Term(ext.one(), "geom", eta=eta, nvars=6)
    vec = t.colpart(0, 6)
    assert np.array_equal(vec[:, 0], cl.powers(eta, 64))
    assert not vec[:, 1:].any()


def test_eq_term_matches_eq_vector():
    z = rand_ext((5,))
    t = cl.Term(ext.one(), "eq", z=z, nvars=5)
    assert np.array_equal(t.colpart(0, 5), cl.eq_vector(z))
    r = rand_ext((2,))
    # colpart_at over the top two variables equals the tensor contracted with rbar
    top = t.colpart(3, 2)
    assert np.array_equal(t.colpart_at(3, r), ext.sum_ext(ext.mul(top, cl.rbar(r)), axis=0))


def test_column_sumcheck_roundtrip():
    k, kp, T = 3, 4, 3
    terms = [cl.Term(rand_ext(), "eq", z=rand_ext((k + kp,)), nvars=k + kp),
             cl.Term(rand_ext(), "geom", eta=777, nvars=k + kp),
             cl.Term(rand_ext(), "geom", eta=99991, nvars=k + kp)]
    us = [rand_ext((1 << kp,)) for _ in range(T)]
    sc = cl.ColumnSumcheckProver(terms, us, k, kp)
    claim = sc.claim()
    msgs, rs = [], []
    for _ in range(kp):
        g = sc.round_message()
        r = rand_ext()
        sc.fold(r)
        msgs.append(g)
        rs.append(r)
    ok, final = cl.verify_column_rounds(claim, msgs, rs)
    assert ok
    r = sc.challenges()
    rb = cl.rbar(r)
    expect = ext.zero()
    for t, u in zip(terms, us):
        expect = ext.add(expect, ext.mul(ext.mul(t.coef, t.colpart_at(k, r)), ext.sum_ext(ext.mul(u, rb), axis=0)))
    assert np.array_equal(expect, final)
    assert np.array_equal(sc.claim(), final)


# ------------------------------------------------------------------------------------------------ PCS

def _instance(n: int, dims: Dims):
    f_np = RNG.integers(0, P, size=1 << n, dtype=np.int64)
    z = rand_ext((n,))
    f = torch.as_tensor(f_np.astype(np.int32), device=DEV)
    return f_np, f, z


@pytest.mark.parametrize("n,kprime,q", [(10, [4, 3], 8), (12, [4, 3, 3], 12), (9, [4], 8)])
def test_prove_verify_small(n, kprime, q):
    dims = Dims(n, kprime, [q] * len(kprime))
    f_np, f, z = _instance(n, dims)
    pf = prove(f, z, dims)
    assert np.array_equal(pf.v, eval_multilinear(f_np, z))
    ok, why = verify(pf, z)
    assert ok, why
    # bytes round trip
    buf = pf.to_bytes()
    pf2 = Proof.from_bytes(buf)
    ok, why = verify(pf2, z)
    assert ok, why
    assert len(buf) == len(pf2.to_bytes())


def test_negatives():
    n = 12
    dims = Dims(n, [4, 3, 3], [12, 12, 12])
    f_np, f, z = _instance(n, dims)
    pf = prove(f, z, dims)
    base = pf.to_bytes()
    assert verify(Proof.from_bytes(base), z)[0]
    # wrong evaluation
    ok, why = verify(Proof.from_bytes(base), z, v=(pf.v + 1) % P)
    assert not ok
    p2 = Proof.from_bytes(base)
    p2.v = (p2.v + 1) % P
    assert not verify(p2, z)[0]
    # tampered opened row (round 1, one word)
    p3 = Proof.from_bytes(base)
    p3.opens[0].rows[3, 1] = (p3.opens[0].rows[3, 1] + 1) % P
    ok, why = verify(p3, z)
    assert not ok and "Merkle" in why, why
    # tampered opened row of the LAST round
    p4 = Proof.from_bytes(base)
    p4.opens[-1].rows[0, 0] = (p4.opens[-1].rows[0, 0] + 1) % P
    assert not verify(p4, z)[0]
    # wrong final vector
    p5 = Proof.from_bytes(base)
    p5.final_y[1, 2] = (p5.final_y[1, 2] + 1) % P
    ok, why = verify(p5, z)
    assert not ok, why           # (y is absorbed before S_L is drawn, so the Merkle check fires first)
    # tampered sumcheck message
    p6 = Proof.from_bytes(base)
    p6.sumcheck[1][0][1, 0] = (p6.sumcheck[1][0][1, 0] + 1) % P
    assert not verify(p6, z)[0]
    # tampered root
    p7 = Proof.from_bytes(base)
    p7.roots[1] = bytes(32)
    assert not verify(p7, z)[0]
    # a proof for another polynomial
    f2 = torch.as_tensor(RNG.integers(0, P, size=1 << n, dtype=np.int64).astype(np.int32), device=DEV)
    p8 = prove(f2, z, dims)
    p8.v = pf.v
    assert not verify(p8, z)[0]


def test_dims_choose_reproduces_constraints():
    for n in (20, 26, 28, 29, 30):
        d = choose(n, 192)
        assert sum(d.kprime) + d.final_log == n
        assert d.codeword_log(0) <= 24


@pytest.mark.skipif(DEV != "cuda", reason="GPU encoder")
def test_encoder_bit_exact_against_field_ntt():
    from .ntt_gpu import Encoder, reference_encode
    for R in (1 << 8, 1 << 10, 1 << 11, 1 << 13, 1 << 15):
        enc = Encoder(R, DEV)
        msg = torch.randint(0, P, (5, R), dtype=torch.int64, device=DEV)
        got = enc.encode(msg)
        ref = reference_encode(msg, enc.n, enc)
        assert torch.equal(got, ref), R


@pytest.mark.skipif(DEV != "cuda", reason="GPU kernels")
def test_contractions_against_torch():
    from .kernels import col_contract, row_contract
    from .linalg import eq_planes
    C, R = 16, 1 << 12
    f = torch.randint(0, P, (C, R), dtype=torch.int64, device=DEV).to(torch.int32)
    z = rand_ext((12,))
    zr = eq_planes(z, DEV)
    u = col_contract(f, zr)
    f64 = f.to(torch.int64)
    ref = torch.stack([((f64 * zr[d]) % P).sum(1) % P for d in range(D)], dim=1).cpu().numpy()
    assert np.array_equal(u, ref)
    rb = rand_ext((C,))
    y = row_contract(f, rb).to(torch.int64).cpu().numpy()
    rbt = torch.as_tensor(rb, device=DEV)
    ref = torch.stack([((f64 * rbt[:, d:d + 1]) % P).sum(0) % P for d in range(D)], dim=0).cpu().numpy()
    assert np.array_equal(y, ref)


def test_vectorised_terms_match_scalar():
    k, kp = 5, 3
    terms = [cl.Term(rand_ext(), "eq", z=rand_ext((k + kp,)), nvars=k + kp)] + \
            [cl.Term(rand_ext(), "geom", eta=int(e), nvars=k + kp) for e in RNG.integers(1, P, size=7)]
    B = cl.colparts(terms, k, kp)
    for i, t in enumerate(terms):
        assert np.array_equal(B[i], t.colpart(k, kp))
    r = rand_ext((kp,))
    got = cl.restrict_terms(terms, k, r)
    for t, g in zip(terms, got):
        s = t.restrict(k, r)
        assert np.array_equal(g.coef, s.coef) and g.kind == s.kind and g.eta == s.eta and g.nvars == s.nvars
