"""`PouwCallCheck_v1`'s word view (`segment.call_check`) is `edited_gemm/cgemm.cu`'s served H1 receipt: on a random
128 x 128 tile it accepts the receipts `cgemm_ref` derives (`checked_strip.co32c_words`, then `floored_gemm/derive.py`'s
`tile_receipts` over T_1 .. T_last, O', D_1 .. D_(m-1); for fx1, over those words times `fx1_key`), thread by thread
from the thread's leaves, coin and sample, and refuses each with one bit changed."""
from __future__ import annotations

import checked_strip as cs
import numpy as np
import pytest
from conftest import load

from verity.protocols.pouw.prover.circuit import segment as SG

derive = load("floored_gemm/derive.py")


def _thread_args(V, W, b, c, U, z, ks, ctid):
    """Thread ``ctid``'s `PouwCallCheck_v1` operand and coin ports (receipt rows are V's, columns W's)."""
    warp, lane = divmod(ctid, 32)
    wm, wn, g, q = warp // 4, warp % 4, lane >> 2, lane & 3
    rows = [32 * wm + g + 8 * r for r in range(4)]
    cols = [32 * wn + 2 * q + 8 * (k >> 1) + (k & 1) for k in range(8)]

    def enc(a) -> list:
        return (np.asarray(a) & 0xFF).tolist()

    def seg(M, i: int, s: int) -> list:
        return M[i, 32 * s:32 * s + 32].tolist()

    return ([enc(V[64 * (wm // 2):64 * (wm // 2) + 64, 32 * s:32 * s + 32]) for s in range(ks)],
            [int(i == 8 * (wm % 2) + g) for i in range(16)],
            [[seg(b, i, s) for i in rows] for s in range(ks)],
            [enc(W[64 * (wn // 2):64 * (wn // 2) + 64, 32 * s:32 * s + 32]) for s in range(ks)],
            [int(i == 4 * (wn % 2) + q) for i in range(8)] + [0] * 8,
            [[seg(c, j, s) for j in cols] for s in range(ks)],
            [enc(U[j]) for j in cols],
            [[z[d, :, i].tolist() for i in rows] for d in range(z.shape[0])])


@pytest.mark.parametrize("ks,fold", [(2, "duo"), (4, "duo"), (4, "fx1"), pytest.param(130, "duo", marks=pytest.mark.slow)])
def test_call_check_accepts_cgemm_refs_receipts(ks, fold):
    rng = np.random.default_rng(ks)
    k, T = 32 * ks, 128
    V, W = rng.integers(-127, 127, (T, k)), rng.integers(-60, 61, (T, k))
    b, c = rng.integers(0, 2, (T, k)), rng.integers(0, 2, (T, k))
    U = rng.integers(-64, 65, (T, 32))
    m = cs.n_digits(ks * -128, ks * 127)
    assert m == SG.n_digits(ks)
    z = rng.integers(0, 2, (m - 1, 32, T))
    Ts, D, Op = cs.co32c_words(W, V.T, U, c, b.T, z, 32, 32, m)
    words = [t.T for t in Ts] + [Op.T] + [d.T for d in D]
    if fold == "fx1":
        K = np.zeros((T, T), np.uint64)
        for ctid in range(SG.NCT):
            warp, lane = divmod(ctid, 32)
            wm, wn, g, qq = warp // 4, warp % 4, lane >> 2, lane & 3
            for e in range(32):
                cc = SG.col_slot(e)
                K[32 * wm + g + 8 * SG.row_slot(e), 32 * wn + 2 * qq + 8 * (cc >> 1) + (cc & 1)] = SG.fx1_key(e)
        words = [((w.astype(np.uint64) * K) & 0xFFFFFFFF).astype(np.uint32) for w in words]
    seed, tile = 0x5EED0001 + ks, 3
    want = derive.tile_receipts(seed, words, tile)
    for ctid in (0, 37, 200, 511):
        y, zz = SG.init(seed, tile, ctid, 0, 1)
        rc = [0] * SG.RW
        slot = ctid % 64
        rc[8 * slot:8 * slot + 8] = [int(v) for v in want[ctid]]
        args = (*_thread_args(V, W, b, c, U, z, ks, ctid), y + zz)
        rsel = [int(s == slot) for s in range(64)]
        assert SG.call_check(*args, rc, rsel, fold=fold) == 1, ctid
        rc[8 * slot + ctid % 8] ^= 1 << (ctid % 32)
        assert SG.call_check(*args, rc, rsel, fold=fold) == 0, ctid
