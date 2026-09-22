"""Reference checks (run anywhere, seconds) and GPU-vs-reference bit-exactness (skipped without CUDA)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from encode import babybear as bb  # noqa: E402
from encode import matdft_ref  # noqa: E402


def _cuda():
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
        return cp
    except Exception:  # noqa: BLE001
        return None


def test_constants():
    assert bb.P == 2**31 - 2**27 + 1
    assert (bb.P * bb.P_INV_MOD_R) % 2**32 == 1
    assert (bb.P * bb.NEG_P_INV_MOD_R + 1) % 2**32 == 0
    w = bb.root_of_unity(2**27)
    assert pow(w, 2**27, bb.P) == 1 and pow(w, 2**26, bb.P) != 1
    assert pow(bb.GENERATOR, (bb.P - 1) // 2, bb.P) != 1 and pow(bb.GENERATOR, (bb.P - 1) // 3, bb.P) != 1
    assert pow(bb.GENERATOR, (bb.P - 1) // 5, bb.P) != 1


def test_ntt_roundtrip_and_naive():
    rng = np.random.default_rng(1)
    x = rng.integers(0, bb.P, size=(3, 64), dtype=np.int64)
    w = bb.root_of_unity(64)
    X = bb.ntt_rows(x, w)
    naive = np.array([[sum(int(x[r, j]) * pow(w, j * k, bb.P) for j in range(64)) % bb.P for k in range(64)] for r in range(3)], dtype=np.uint64)
    assert np.array_equal(X, naive)
    assert np.array_equal(bb.intt_rows(X, w), x.astype(np.uint64))


def test_encode_matches_naive_and_is_systematic():
    rng = np.random.default_rng(2)
    k, n = 8, 32
    rows = rng.integers(0, bb.P, size=(2, k), dtype=np.int64)
    ref_nat = bb.encode_rows_naive(rows, n)
    got_nat = bb.encode_rows(rows, n, layout="natural")
    got_cos = bb.encode_rows(rows, n, layout="coset")
    assert np.array_equal(got_nat, ref_nat)
    assert np.array_equal(got_cos[:, bb.coset_to_natural(n, k)], got_nat)
    assert np.array_equal(got_cos[:, :k], rows.astype(np.uint64))          # systematic: coset 0 is the message
    assert np.array_equal(got_nat[:, ::4], rows.astype(np.uint64))


def test_twiddle_tables_shapes():
    from encode.encode_ntt_simt import twiddle_tables

    t = twiddle_tables(64, 256)
    assert t["tw_fwd"].shape == (64,) and t["coset"].shape == (3 * 64,)
    # stage m=1 uses omega_2^0 = 1; stage m=2, t=1 holds omega_4 (a square root of -1); all in Montgomery form
    assert int(t["tw_fwd"][1]) == bb.to_mont(1)
    w4 = (int(t["tw_fwd"][3]) * pow(bb.R_MOD_P, -1, bb.P)) % bb.P
    assert (w4 * w4) % bb.P == bb.P - 1


def test_matdft_reference_matches_encoder():
    rng = np.random.default_rng(3)
    k, n = 64, 256
    rows = rng.integers(0, bb.P, size=(3, k), dtype=np.int64)
    ref = bb.encode_rows(rows, n, layout="coset")
    got = matdft_ref.encode_rows_matdft(rows, n)
    assert np.array_equal(got, ref)
    # limb decomposition round-trips and stays in int8
    limbs = matdft_ref.balanced_limbs(np.array([0, 1, bb.P - 1, 2**31 - 2**27], dtype=np.int64))
    assert limbs.dtype == np.int8
    assert np.array_equal(matdft_ref.from_limbs(limbs.astype(np.int64)), np.array([0, 1, bb.P - 1, 2**31 - 2**27]))


@pytest.mark.skipif(_cuda() is None, reason="no CUDA device")
@pytest.mark.parametrize("k,threads", [(512, 256), (4096, 256), (4096, 512)])
def test_simt_kernel_bit_exact(k, threads):
    cp = _cuda()
    from encode.encode_ntt_simt import RSEncoderSIMT

    rows = bb.fill_rows(37, k, seed=k)
    ref = bb.encode_rows(rows, 4 * k, layout="coset")
    enc = RSEncoderSIMT(k, 4 * k, threads=threads)
    out = cp.asnumpy(enc(cp.asarray(rows)))
    assert np.array_equal(out.astype(np.uint64), ref)
    out2 = cp.asnumpy(enc(cp.asarray(rows), write_systematic=False))
    assert np.array_equal(out2[:, k:].astype(np.uint64), ref[:, k:])


@pytest.mark.skipif(_cuda() is None, reason="no CUDA device")
def test_int8_matdft_bit_exact():
    import torch

    from encode.encode_matdft_int8 import RSEncoderMatDFT

    rows = bb.fill_rows(129, 4096, seed=9)
    ref = bb.encode_rows(rows, 16384, layout="coset")
    for variant in RSEncoderMatDFT.available():
        enc = RSEncoderMatDFT(variant=variant)
        out = enc(torch.from_numpy(rows.view(np.int32)).cuda()).cpu().numpy()
        assert np.array_equal(out.view(np.uint32).astype(np.uint64), ref), variant
