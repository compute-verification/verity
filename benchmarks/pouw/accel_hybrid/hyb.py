"""HYB (BET-HYBRID, CMP-120): Dither/H1 with wide periodic noise on both operands, 8-bit data, and its wraps corrected.

Exact CPU reference on small shapes: honest prover and a verifier that recomputes drawn cells (in the clear; in the
protocol that recomputation is a zero-knowledge proof). `python hyb.py` runs every check and prints one line each.

~~~text
coins    after A's and B's rows are committed, from the live seed: b (m x k), c (n x k) bits; V (m x L) uniform on ±uX,
         U (n x L) uniform on ±uW, each repeated every segment (V_rep[i, l] = V[i, l mod L]); L = 32, k = S·L
operands X  = wrap8(A + b + V_rep) = X* − 256 wX,   W' = wrap8(B + c + U_rep) = W* − 256 wW   (wX, wW ∈ {−1, 0, 1})
words    T_s = Σ_{s' <= s} X_{s'} W'_{s'}ᵀ mod 2^32, one per segment and output cell (the s8 IMMA coordinate, as H1's)
fold     per cell (i, j): duo-style chain over its S words (u = y ^ w ^ K; y = u·KA + z; z = rotl(u, 13) + y)
leaf     per row of cells: SHA-512(le32 receipts), Merkle root over rows (SHA-512 of the two children)
served   O = (A + b)(B + c)ᵀ, exactly H1's served product, from the last word:
         O = T_last + 256 (wX W'ᵀ + X wWᵀ + 256 wX wWᵀ) − Y_X Uᵀ − V Y_Wᵀ − S·V Uᵀ,
         Y_X[i, ρ] = Σ_s (A + b)[i, sL + ρ],  Y_W[j, ρ] = Σ_s (B + c)[j, sL + ρ]
cost     peels: 3 products of inner size L (3L/k per MAC); wraps: one add per (nonzero of wW, row of X) and per
         (nonzero of wX, row of W'), i.e. p_W + p_X adds per MAC
~~~
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np

L = 32
M32 = (1 << 32) - 1
KA, KK = 0x9E3779B1, 0x7F4A7C15


@dataclass(frozen=True)
class Params:
    uX: int = 64   # activation-side noise half-width (theta_X = uX / 128)
    uW: int = 96   # weight-side noise half-width (theta_W = uW / 128)


def wrap8(v):
    return ((v + 128) & 0xFF) - 128


def stream(seed: bytes, label: str, n: int) -> np.ndarray:
    """n uint32 from SHA-512 in counter mode (the stand-in for the challenger's live coins)."""
    out, ctr = bytearray(), 0
    while len(out) < 4 * n:
        out += hashlib.sha512(seed + label.encode() + struct.pack("<Q", ctr)).digest()
        ctr += 1
    return np.frombuffer(bytes(out[:4 * n]), dtype="<u4").astype(np.int64)


def uniform(seed, label, shape, u):
    return (stream(seed, label, int(np.prod(shape))) % (2 * u + 1) - u).reshape(shape)


def coins(seed: bytes, m: int, n: int, k: int, p: Params):
    return dict(b=stream(seed, "b", m * k).reshape(m, k) & 1, c=stream(seed, "c", n * k).reshape(n, k) & 1,
                V=uniform(seed, "V", (m, L), p.uX), U=uniform(seed, "U", (n, L), p.uW))


def rep(Z, k):
    return np.tile(Z, (1, k // L))


def operands(A, B, cn):
    k = A.shape[1]
    Xs = A + cn["b"] + rep(cn["V"], k)
    Ws = B + cn["c"] + rep(cn["U"], k)
    X, W = wrap8(Xs), wrap8(Ws)
    return X, W, (Xs - X) // 256, (Ws - W) // 256


def words(X, W):
    """T[s] (S x m x n): prefix sums over segments, mod 2^32."""
    m, k = X.shape
    seg = np.stack([X[:, s * L:(s + 1) * L] @ W[:, s * L:(s + 1) * L].T for s in range(k // L)])
    return np.cumsum(seg, axis=0) & M32


def rotl(x, r):
    return ((x << r) | (x >> (32 - r))) & M32


def fold(ws) -> tuple[int, int]:
    y, z = 0, 0
    for w in ws:
        u = y ^ int(w) ^ KK
        y = (u * KA + z) & M32
        z = (rotl(u, 13) + y) & M32
    return y, z


def h(*parts: bytes) -> bytes:
    return hashlib.sha512(b"".join(parts)).digest()


def merkle(leaves):
    levels = [leaves]
    while len(levels[-1]) > 1:
        lv = levels[-1]
        levels.append([h(lv[i], lv[i + 1] if i + 1 < len(lv) else lv[i]) for i in range(0, len(lv), 2)])
    return levels


def path(levels, i):
    out = []
    for lv in levels[:-1]:
        j = i ^ 1
        out.append(lv[j] if j < len(lv) else lv[i])
        i //= 2
    return out


def root_from(leaf, i, pth):
    x = leaf
    for sib in pth:
        x = h(x, sib) if i % 2 == 0 else h(sib, x)
        i //= 2
    return x


def s32(v):
    return ((v + (1 << 31)) & M32) - (1 << 31)


def serve(A, B, cn, X, W, wX, wW, T):
    """The served product from the last word, the peels and the sparse wrap corrections; returns (O, adds)."""
    k = A.shape[1]
    S = k // L
    YX = (A + cn["b"]).reshape(A.shape[0], S, L).sum(1)
    YW = (B + cn["c"]).reshape(B.shape[0], S, L).sum(1)
    corr = np.zeros((A.shape[0], B.shape[0]), dtype=np.int64)
    for j, l in zip(*np.nonzero(wW)):            # one add per row of X: column j gets 256 wW[j,l] X[:, l]
        corr[:, j] += wW[j, l] * X[:, l]
    for i, l in zip(*np.nonzero(wX)):            # one add per row of W'
        corr[i, :] += wX[i, l] * W[:, l]
    both = wX @ wW.T                             # doubly sparse: nnz(wX) nnz(wW) / k pairs
    O = s32(T[-1]) + 256 * corr + 65536 * both - YX @ cn["U"].T - cn["V"] @ YW.T - S * (cn["V"] @ cn["U"].T)
    adds = int(np.count_nonzero(wW)) * A.shape[0] + int(np.count_nonzero(wX)) * B.shape[0]
    return O, adds


def prove(A, B, seed, p: Params):
    cn = coins(seed, A.shape[0], B.shape[0], A.shape[1], p)
    X, W, wX, wW = operands(A, B, cn)
    T = words(X, W)
    rc = [[fold(T[:, i, j]) for j in range(B.shape[0])] for i in range(A.shape[0])]
    leaves = [h(b"".join(struct.pack("<II", *r) for r in row)) for row in rc]
    lv = merkle(leaves)
    O, adds = serve(A, B, cn, X, W, wX, wW, T)
    return dict(cn=cn, T=T, rc=rc, levels=lv, root=lv[-1][0], O=O, adds=adds, wX=wX, wW=wW)


def verify_cell(A_i, B_j, i, j, seed, p, m, n, root, row_rc, pth, O_ij) -> bool:
    """Recompute cell (i, j) from its committed rows and the coins; check its receipt's row leaf and the served value."""
    k = A_i.shape[0]
    cn = coins(seed, m, n, k, p)
    X = wrap8(A_i + cn["b"][i] + rep(cn["V"][i:i + 1], k)[0])
    W = wrap8(B_j + cn["c"][j] + rep(cn["U"][j:j + 1], k)[0])
    ws = np.cumsum([int(X[s * L:(s + 1) * L] @ W[s * L:(s + 1) * L]) for s in range(k // L)]) & M32
    ok = fold(ws) == tuple(row_rc[j])
    leaf = h(b"".join(struct.pack("<II", *r) for r in row_rc))
    ok &= root_from(leaf, i, pth) == root
    return bool(ok and int((A_i + cn["b"][i]) @ (B_j + cn["c"][j])) == int(O_ij))


def word_of_bits(A_i, B_j, V_i, U_j, bb, cc, s):
    """Cell (i, j)'s segment-s contribution as a function of its coin bits there (other coins fixed)."""
    x = wrap8(A_i[s * L:(s + 1) * L] + bb + V_i)
    w = wrap8(B_j[s * L:(s + 1) * L] + cc + U_j)
    return int(x @ w) & M32


def rank_f2(M) -> int:
    M = M.copy() % 2
    r = 0
    for col in range(M.shape[1]):
        piv = next((i for i in range(r, M.shape[0]) if M[i, col]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, col]:
                M[i] ^= M[r]
        r += 1
    return r


def p2_rank(A_i, B_j, V_i, U_j, s, base_b, base_c) -> int:
    """rank over F2 of the second differences of one word in its 2L coin bits (Theorem K's P2), at a base point."""
    def f(z):
        return word_of_bits(A_i, B_j, V_i, U_j, (base_b + z[:L]) & 1, (base_c + z[L:]) & 1, s)
    e = np.eye(2 * L, dtype=np.int64)
    f0, f1 = f(np.zeros(2 * L, dtype=np.int64)), [f(e[u]) for u in range(2 * L)]
    P = np.array([[(f(e[u] + e[v]) - f1[u] - f1[v] + f0) & 1 if u != v else 0 for v in range(2 * L)]
                  for u in range(2 * L)])
    return rank_f2(P)


def main() -> None:
    p = Params()
    rng = np.random.default_rng(7)
    m, n, k = 16, 16, 256
    t = rng.standard_t(4, (m, k)), rng.standard_t(4, (n, k))
    cases = {
        "heavy-tailed 8-bit A, B": tuple(np.round(127 * z / np.abs(z).max(1, keepdims=True)).astype(np.int64) for z in t),
        "A = B = 0": (np.zeros((m, k), np.int64), np.zeros((n, k), np.int64)),
        "A = B = 127 (dense wraps)": (np.full((m, k), 127, np.int64), np.full((n, k), 127, np.int64)),
        "uniform 8-bit A, B": (rng.integers(-128, 128, (m, k)), rng.integers(-128, 128, (n, k))),
    }
    for name, (A, B) in cases.items():
        seed = hashlib.sha512(name.encode()).digest()
        pr = prove(A, B, seed, p)
        exact = bool(np.array_equal(pr["O"], (A + pr["cn"]["b"]) @ (B + pr["cn"]["c"]).T))
        ok = all(verify_cell(A[i], B[j], i, j, seed, p, m, n, pr["root"], pr["rc"][i], path(pr["levels"], i),
                             pr["O"][i, j]) for i in range(m) for j in range(n))
        bad = [row[:] for row in pr["rc"]]
        bad[3][5] = (bad[3][5][0] ^ 1, bad[3][5][1])
        caught = not verify_cell(A[3], B[5], 3, 5, seed, p, m, n, pr["root"], bad[3], path(pr["levels"], 3),
                                 pr["O"][3, 5])
        pW, pX = np.count_nonzero(pr["wW"]) / pr["wW"].size, np.count_nonzero(pr["wX"]) / pr["wX"].size
        print(f"{name:28s} served exact {exact}; all {m * n} cells verify {ok}; flipped receipt caught {caught}; "
              f"wraps W' {pW:.4f} X {pX:.4f}; correction adds per MAC {pr['adds'] / (m * n * k):.4f}")
        cn = pr["cn"]
        rk = p2_rank(A[2], B[9], cn["V"][2], cn["U"][9], 1, cn["b"][2, L:2 * L], cn["c"][9, L:2 * L])
        print(f"{'':28s} rank_F2 P2 of word (2, 9, s = 1) in its {2 * L} coin bits: {rk} (Theorem K: >= {2 * L} "
              f"needs |S| >= {L} products)")


if __name__ == "__main__":
    main()
