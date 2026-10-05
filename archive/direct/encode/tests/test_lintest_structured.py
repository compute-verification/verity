"""lintest_structured: the structure of r^T A under the powers combination and the shortcut's bit-exactness (numpy on the
laptop; the GPU checks run in the bench on the pod and skip here without CUDA)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from encode import babybear as bb
from encode import buckets_ref as ref
from encode import lintest_structured as ls


def test_fc_mul_np_matches_python_ints():
    rng = np.random.default_rng(0)
    a = rng.integers(0, bb.P, (5, 6)).astype(np.uint64)
    b = rng.integers(0, bb.P, (5, 6)).astype(np.uint64)
    got = ls.fc_mul_np(a, b)
    for i in range(5):
        assert got[i].tolist() == ref.fc_mul([int(v) for v in a[i]], [int(v) for v in b[i]])


def test_toy_chain_system_rta_is_scalar_times_fixed_row():
    """r^T A of the unit-pattern system with r = powers of rho equals s_i * E on every regular row (units with a
    predecessor and a successor); only the rows holding unit 0 and the last unit are exceptional."""
    rng = np.random.default_rng(1)
    for seed, S in ((1, 2), (2, 3)):
        sysm = ls.UnitPatternSystem.random(U=12, w=7, C=5, transition=(3, 4), nb=2, S=S, k=32, l=21, seed=seed)
        rho = [int(v) for v in rng.integers(0, bb.P, 6)]
        st = sysm.structured(rho)
        dense = sysm.dense_rta(rho)
        assert st.M == 1
        assert sorted(st.exceptional) == [0, (sysm.U - 1) // S]
        assert np.array_equal(st.expand(), dense)
        # and the regular rows really are NOT captured by a single fixed row without the neighbour term (sanity: the structure is not trivial)
        assert dense[:, 1].any()


def test_numpy_shortcut_equals_general_linear_test():
    rows, M, X = 12, 2, 2
    st = ls.random_structured(rows, M, X, k=bb.K_CODE, seed=3)
    W = bb.fill_rows(rows, seed=4)
    assert np.array_equal(ls.structured_linear_test_ref(st, W), ref.linear_test(st.expand(), W))


def test_int8_limb_recombination_identity():
    """The exact-integer identity behind combine_int8_tc, on the host: sum_i s_i W_i via bytes shifted by 128."""
    rng = np.random.default_rng(5)
    rows, k = 9, 8
    W = rng.integers(0, bb.P, (rows, k)).astype(np.int64)
    s = rng.integers(0, bb.P, (rows, 6)).astype(np.int64)
    W8 = np.stack([((W >> (8 * b)) & 0xFF) - 128 for b in range(4)], axis=-1).reshape(rows, 4 * k)      # bytes ^ 0x80
    A = np.zeros((32, rows), dtype=np.int64)
    A[:24] = np.stack([((s >> (8 * a)) & 0xFF) for a in range(4)], axis=1).reshape(rows, 24).T - 128
    A[24] = 1
    C = A @ W8
    ssum = (A + 128).sum(axis=1)
    out = np.zeros((6, k), dtype=np.int64)
    for a in range(4):
        for b in range(4):
            coef = pow(2, 8 * (a + b), bb.P)
            for c in range(6):
                row = a * 6 + c
                t = C[row, b::4] + 128 * C[24, b::4] + 128 * ssum[row]
                out[c] = (out[c] + (t % bb.P) * coef) % bb.P
    assert np.array_equal(out.astype(np.uint64), ref.coords_times_rows(s.astype(np.uint64), W.astype(np.uint64)))


try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    HAS_CUDA = True
except Exception:                          # noqa: BLE001 - pragma: no cover - laptop
    HAS_CUDA = False


@pytest.mark.skipif(not HAS_CUDA, reason="no CUDA device")
def test_gpu_checks():
    from encode.encode_ntt_simt import RSEncoderSIMT

    out = ls.checks(RSEncoderSIMT())
    assert all(v is True for k, v in out.items() if k != "toy_exceptional_rows"), out
