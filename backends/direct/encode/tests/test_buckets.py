"""buckets.py kernels vs buckets_ref (GPU tests skip without CUDA; the reference self-checks run anywhere)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from encode import babybear as bb
from encode import buckets_ref as ref


def test_field_and_reference():
    assert ref.is_irreducible_binomial()
    rng = np.random.default_rng(1)
    a = [int(v) for v in rng.integers(0, bb.P, 6)]
    assert ref.fc_mul(a, ref.fc_inv(a)) == [1, 0, 0, 0, 0, 0]
    # linear test == IRS combination when rA rows are constant (a scalar per row broadcast over the k payload positions)
    # only in the systematic half: check the systematic columns of the quadratic/linear references against direct sums
    x = bb.fill_rows(3, seed=2)
    r = rng.integers(0, bb.P, (3, 6))
    q = ref.quadratic_test(r, x, x, x)
    direct = ref.coords_times_rows(r, (x.astype(np.uint64) ** 2 + (bb.P - x.astype(np.uint64))) % bb.P)
    assert np.array_equal(q[:, : bb.K_CODE], direct)


try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    HAS_CUDA = True
except Exception:                          # noqa: BLE001 - pragma: no cover - laptop
    HAS_CUDA = False

if HAS_CUDA:
    from encode import buckets as bk
    from encode.encode_ntt_simt import RSEncoderSIMT

gpu = pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")


@gpu
def test_irs_row_combination():
    rng = np.random.default_rng(3)
    rows = 300
    W = bb.fill_rows(rows, seed=4)
    r = rng.integers(0, bb.P, (rows, 6)).astype(np.uint64)
    cb = bk.Combiner(chunk_rows=32)
    acc = cb.acc(rows, bb.K_CODE)
    cb.scalar(cp.asarray(W), cp.asarray(bk.mont(r)), acc)
    got = cp.asnumpy(cb.reduce(acc)).astype(np.uint64)
    assert np.array_equal(got, ref.irs_row_combination(r, W))
    # streamed in two slices into the same accumulator
    acc2 = cb.acc(150, bb.K_CODE)
    Wd, rd = cp.asarray(W), cp.asarray(bk.mont(r))
    cb.scalar(Wd[:150], rd[:150], acc2)
    cb.scalar(Wd[150:], rd[150:], acc2)
    assert np.array_equal(cp.asnumpy(cb.reduce(acc2)).astype(np.uint64), ref.irs_row_combination(r, W))


@gpu
def test_quadratic_test():
    rng = np.random.default_rng(5)
    rows = 70
    x, y, z = (bb.fill_rows(rows, seed=s) for s in (6, 7, 8))
    r = rng.integers(0, bb.P, (rows, 6)).astype(np.uint64)
    enc = RSEncoderSIMT()
    xd, yd, zd = (cp.asarray(a) for a in (x, y, z))
    c1 = [enc(a, cp.empty((rows, bb.K_CODE), dtype=cp.uint32), write_systematic=False, ncosets=1) for a in (xd, yd, zd)]
    cb = bk.Combiner(chunk_rows=16)
    acc = cb.acc(rows, 2 * bb.K_CODE)
    r12 = cp.asarray(np.concatenate([bk.mont(r), bk.mont2(r)], axis=1))
    cb.quadratic(xd, yd, zd, *c1, r12, acc)
    got = cp.asnumpy(cb.reduce(acc)).astype(np.uint64)
    assert np.array_equal(got, ref.quadratic_test(r, x, y, z))


@gpu
def test_linear_test():
    rows = 40
    u = bb.fill_rows(rows, seed=9)
    rA = np.stack([bb.fill_rows(rows, seed=10 + c) for c in range(6)])          # (6, rows, k)
    enc = RSEncoderSIMT()
    ud = cp.asarray(u)
    rAd = cp.asarray(rA)
    u_c1 = enc(ud, cp.empty((rows, bb.K_CODE), dtype=cp.uint32), write_systematic=False, ncosets=1)
    rA_c1 = cp.empty((6, rows, bb.K_CODE), dtype=cp.uint32)
    for c in range(6):
        enc(rAd[c], rA_c1[c], write_systematic=False, ncosets=1)
    cb = bk.Combiner(chunk_rows=8)
    acc = cb.acc(rows, 2 * bb.K_CODE)
    cb.linear(ud, u_c1, rAd, rA_c1, acc)
    got = cp.asnumpy(cb.reduce(acc)).astype(np.uint64)
    assert np.array_equal(got, ref.linear_test(rA, u))


@gpu
@pytest.mark.parametrize("N,T", [(1000, 64), (64 * 7, 64), (5, 64)])
def test_batch_inverse(N, T):
    rng = np.random.default_rng(11)
    d = rng.integers(1, bb.P, (6, N)).astype(np.uint64)
    inv = bk.BatchInverter(threads_total=T, block=32)
    got = cp.asnumpy(inv(cp.asarray(bk.mont(d)))).astype(np.uint64)
    got_plain = (got * np.uint64(pow(bb.R_MOD_P, -1, bb.P))) % np.uint64(bb.P)
    assert np.array_equal(got_plain, ref.batch_inverse(d))


@gpu
def test_gather_and_encode_gather():
    rows = 200
    W = bb.fill_rows(rows, seed=12)
    code = bb.encode_rows(W, bb.N_CODE, layout="coset")
    cols = np.array([0, 5, 4095, 4096, 4097, 9000, 16383, 12288, 777], dtype=np.int64)
    Wd = cp.asarray(W)
    # from a kept cosets store
    store = RSEncoderSIMT()(Wd, cp.empty((rows, 3 * bb.K_CODE), dtype=cp.uint32), write_systematic=False)
    g = bk.ColumnGather()
    sel_cos = cp.asarray((cols[cols >= bb.K_CODE] - bb.K_CODE).astype(np.int32))
    sel_sys = cp.asarray(cols[cols < bb.K_CODE].astype(np.int32))
    got_cos = cp.asnumpy(g(store, sel_cos)).astype(np.uint64)
    got_sys = cp.asnumpy(g(Wd, sel_sys)).astype(np.uint64)
    assert np.array_equal(got_cos, code[:, cols[cols >= bb.K_CODE]].T)
    assert np.array_equal(got_sys, code[:, cols[cols < bb.K_CODE]].T)
    # re-encode with the gather-mode encoder
    eg = RSEncoderSIMT(gather=True)
    got = cp.asnumpy(eg.gather(Wd, cols)).astype(np.uint64)
    assert np.array_equal(got, code[:, cols].T)
