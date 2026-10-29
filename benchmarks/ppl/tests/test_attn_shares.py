import math

import torch

import attn_shares


def test_row_counts_are_the_kernels_loop():
    """The vectorized counts equal a per-row walk of the blocks: the running max over keys 1.. at each block, y in log2
    units, the u8 P rounded against the block's largest V scale."""
    g = torch.Generator().manual_seed(0)
    H, G, s, d, block = 4, 2, 70, 8, 16
    q = torch.randn(H, s, d, generator=g) * 2
    k = torch.randn(G, s, d, generator=g) * 2
    v = torch.randn(G, s, d, generator=g)
    counts, keys = attn_shares.row_counts(q, k, v, 0.5, block, 24)
    sv = v.abs().amax(-1) / 127
    for h in range(H):
        kg = h // (H // G)
        for i in range(1, s):
            sc = (q[h, i] @ k[kg].T) * 0.5
            want = {c: 0 for c in counts}
            m = -math.inf
            for a in range(0, s, block):
                lo, hi = max(a, 1), min(a + block, i + 1)
                if lo >= hi:
                    continue
                m = max(m, float(sc[lo:hi].max()))
                top = float(sv[kg, max(a, 1):a + block].max())
                for j in range(lo, hi):
                    y = (float(sc[j]) - m) * math.log2(math.e)
                    for c in attn_shares.CUTS:
                        want[c] += y < c
                    want["u8_zero"] += round(255 * 2 ** y * float(sv[kg, j]) / top) == 0
            assert {c: int(counts[c][h, i - 1]) for c in counts} == want, (h, i)
            assert int(keys[h, i - 1]) == i
