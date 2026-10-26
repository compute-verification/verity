"""Pearl-I8 (BET-PEARL-I8, CMP-121): Pearl-C's noise inside an INT8 quantization, as an exact CPU reference.

Honest prover and verifier on small shapes, in exact integers (Python ints where a value can pass 2^63).

~~~text
coins    after A's and B's rows are committed: E_A (m x r), E_B (n x r), F_A, F_B (k x r), int8 each, from the live seed
forming  per row x (int16 data, length k):  a = (q_d << s) // max(max|x|, 1);  z = a·x + N_row,  N = E·Fᵀ (exact);
         X̃ = clamp((z + 2^(s-1)) >> s, ±127)            (the noise is inside the rounding: ρ = z − 2^s X̃ depends on x and N)
chain    C̃ = X̃_A·X̃_Bᵀ in s32 (exact: |C̃| <= k·127²); prefix words T_l = Σ_{t < L(l+1)} X̃_A,it X̃_B,jt mod 2^32
peel     D = 2^s X̃ − N = a·x − ρ, and exactly  D_A·D_Bᵀ = 2^(2s) C̃ + [E_A ‖ 2^s H_A]·[J_B − 2^s K_B ‖ −E_B]ᵀ
         with H_A = X̃_A F_B, K_B = X̃_B F_A, G = F_Aᵀ F_B, J_B = E_B Gᵀ   (`peel_exact`, needs s64)
         pinned P16 (`peel_p16`, the served form): U = C̃ + E_A·M̂_Bᵀ − Ĥ_A·E_Bᵀ, Ĥ_A = rnd(H_A / 2^s),
         M̂_B = rnd((J_B − 2^s K_B) / 2^(2s)), rnd = floor(v + 1/2); int8 x int16 operands, s32 accumulation
fold     per output cell (i, j): duo-style chain over its k/L prefix words (u = y ^ w ^ K; y = u·KA + z; z = rotl(u, 13) + y)
leaf     per tile: SHA-512(le32 receipts ‖ le32 U), Merkle root over tiles (SHA-512 of the two children)
useful   y_ij = U_ij · 2^(2s) / (a_A,i · a_B,j) ≈ x_A,i · x_B,j
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
    r: int = 32            # noise rank
    s: int = 11            # residual bits below the int8 grid
    q_d: int = 67          # data budget in grid units (noise takes the rest of ±127)
    tile: int = 8          # output tile edge for leaves


def coins(seed: bytes, tag: str, rows: int, cols: int) -> np.ndarray:
    """rows x cols int8 from SHA-512 in counter mode over (seed, tag)."""
    need, out, ctr = rows * cols, b"", 0
    while len(out) < need:
        out += hashlib.sha512(seed + tag.encode() + ctr.to_bytes(8, "little")).digest()
        ctr += 1
    return np.frombuffer(out[:need], dtype=np.int8).astype(np.int64).reshape(rows, cols)


def draw(seed: bytes, m: int, n: int, k: int, p: Params) -> dict[str, np.ndarray]:
    return {"EA": coins(seed, "EA", m, p.r), "EB": coins(seed, "EB", n, p.r),
            "FA": coins(seed, "FA", k, p.r), "FB": coins(seed, "FB", k, p.r)}


def form(x: np.ndarray, e: np.ndarray, f: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X̃, a, N) for data rows x (int16 values), noise rows e and basis f."""
    mx = np.maximum(np.abs(x).max(axis=1), 1)
    a = (p.q_d << p.s) // mx
    n = e @ f.T
    z = a[:, None] * x + n
    xt = np.clip((z + (1 << (p.s - 1))) >> p.s, -127, 127)
    return xt, a, n


def prefix_words(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:
    """T[i, j, l] = Σ_{t < L(l+1)} xa[i, t]·xb[j, t] mod 2^32, as uint32."""
    k = xa.shape[1]
    segs = np.stack([xa[:, l:l + L] @ xb[:, l:l + L].T for l in range(0, k, L)], axis=2)
    return (np.cumsum(segs, axis=2) & M32).astype(np.uint64)


def rotl(x: int, r: int) -> int:
    return ((x << r) | (x >> (32 - r))) & M32


def fold(words) -> int:
    y, z = 0x6A09E667, 0xBB67AE85
    for w in words:
        u = y ^ int(w) ^ KK
        y = (u * KA + z) & M32
        z = (rotl(u, 13) + y) & M32
    return y ^ rotl(z, 16)


def peel_exact(xa, xb, fa, fb, ea, eb, p: Params):
    """D_A·D_Bᵀ exactly, by the rank-2r identity (Python ints)."""
    o = lambda v: np.asarray(v, dtype=object)
    xa, xb, fa, fb, ea, eb = map(o, (xa, xb, fa, fb, ea, eb))
    c = xa.dot(xb.T)
    h_a, k_b, g = xa.dot(fb), xb.dot(fa), fa.T.dot(fb)
    j_b = eb.dot(g.T)
    pa = np.concatenate([ea, (1 << p.s) * h_a], axis=1)
    pb = np.concatenate([j_b - (1 << p.s) * k_b, -eb], axis=1)
    return (1 << (2 * p.s)) * c + pa.dot(pb.T)


def rnd_shift(v: np.ndarray, sh: int) -> np.ndarray:
    return (v + (1 << (sh - 1))) >> sh


def peel_p16(c, xa, xb, fa, fb, ea, eb, p: Params):
    """The pinned peel: U = C̃ + E_A·M̂_Bᵀ − Ĥ_A·E_Bᵀ, and its operands' widths."""
    h_a, k_b, g = xa @ fb, xb @ fa, fa.T @ fb
    j_b = eb @ g.T
    hh = rnd_shift(h_a, p.s)
    mm = rnd_shift(j_b - (k_b << p.s), 2 * p.s)
    u = c + ea @ mm.T - hh @ eb.T
    return u, {"H_hat": int(np.abs(hh).max()), "M_hat": int(np.abs(mm).max()), "U": int(np.abs(u).max())}


def sha(b: bytes) -> bytes:
    return hashlib.sha512(b).digest()


def merkle(leaves: list[bytes]) -> bytes:
    lvl = leaves
    while len(lvl) > 1:
        lvl = [sha(lvl[i] + (lvl[i + 1] if i + 1 < len(lvl) else lvl[i])) for i in range(0, len(lvl), 2)]
    return lvl[0]


def tile_leaf(rec: np.ndarray, u: np.ndarray, ti: int, tj: int, p: Params) -> bytes:
    sl = (slice(ti * p.tile, (ti + 1) * p.tile), slice(tj * p.tile, (tj + 1) * p.tile))
    return sha(struct.pack(f"<{rec[sl].size}I", *rec[sl].ravel().tolist())
               + struct.pack(f"<{u[sl].size}i", *u[sl].ravel().tolist()))


def prove(xa_data, xb_data, seed: bytes, p: Params):
    m, k = xa_data.shape
    n = xb_data.shape[0]
    cn = draw(seed, m, n, k, p)
    xa, aa, na = form(xa_data, cn["EA"], cn["FA"], p)
    xb, ab, nb = form(xb_data, cn["EB"], cn["FB"], p)
    t = prefix_words(xa, xb)
    c = xa @ xb.T
    assert np.array_equal(t[:, :, -1].astype(np.int64), c & M32)
    u, widths = peel_p16(c, xa, xb, cn["FA"], cn["FB"], cn["EA"], cn["EB"], p)
    rec = np.array([[fold(t[i, j]) for j in range(n)] for i in range(m)], dtype=np.uint64)
    leaves = [tile_leaf(rec, u, ti, tj, p) for ti in range(m // p.tile) for tj in range(n // p.tile)]
    return {"root": merkle(leaves), "leaves": leaves, "rec": rec, "u": u, "xa": xa, "xb": xb, "aa": aa, "ab": ab,
            "na": na, "nb": nb, "coins": cn, "widths": widths}


def verify_cell(xa_data, xb_data, seed: bytes, p: Params, out, i: int, j: int) -> bool:
    """Recompute cell (i, j) from its two committed data rows and the coins, and check it against the opened leaf.
    (In the protocol this is a zero-knowledge proof; here the opening is in the clear.)"""
    k = xa_data.shape[1]
    cn = draw(seed, xa_data.shape[0], xb_data.shape[0], k, p)
    xa, _, _ = form(xa_data[i:i + 1], cn["EA"][i:i + 1], cn["FA"], p)
    xb, _, _ = form(xb_data[j:j + 1], cn["EB"][j:j + 1], cn["FB"], p)
    t = prefix_words(xa, xb)[0, 0]
    c = xa @ xb.T
    u, _ = peel_p16(c, xa, xb, cn["FA"], cn["FB"], cn["EA"][i:i + 1], cn["EB"][j:j + 1], p)
    ok = fold(t) == int(out["rec"][i, j]) and int(u[0, 0]) == int(out["u"][i, j])
    ti, tj = i // p.tile, j // p.tile
    return ok and tile_leaf(out["rec"], out["u"], ti, tj, p) == out["leaves"][ti * (xb_data.shape[0] // p.tile) + tj]


def selfcheck(m=16, n=16, k=256, p: Params = Params(), seed=b"pearl-i8/selfcheck") -> dict:
    rng = np.random.default_rng(1)
    xa_data = rng.integers(-3000, 3000, size=(m, k))
    xb_data = rng.integers(-20000, 20000, size=(n, k))
    xa_data[0] = 0                     # degenerate rows are admissible
    xb_data[1] = 0
    out = prove(xa_data, xb_data, seed, p)
    cn = out["coins"]
    d_a = (out["xa"] << p.s) - out["na"]
    d_b = (out["xb"] << p.s) - out["nb"]
    exact = peel_exact(out["xa"], out["xb"], cn["FA"], cn["FB"], cn["EA"], cn["EB"], p)
    assert (exact == np.asarray(d_a, dtype=object).dot(np.asarray(d_b, dtype=object).T)).all(), "exact peel identity"
    true_u = np.asarray(d_a, dtype=np.float64) @ np.asarray(d_b, dtype=np.float64).T / 2.0 ** (2 * p.s)
    peel_err = float(np.abs(out["u"] - true_u).max())
    ok = all(verify_cell(xa_data, xb_data, seed, p, out, i, j) for i in range(m) for j in range(n))
    bad = {**out, "rec": out["rec"].copy()}
    bad["rec"][3, 5] ^= 1
    caught = not verify_cell(xa_data, xb_data, seed, p, bad, 3, 5)
    y = out["u"] * 2.0 ** (2 * p.s) / np.outer(out["aa"], out["ab"])
    ref = xa_data.astype(np.float64) @ xb_data.T.astype(np.float64)
    rel = float(np.linalg.norm((y - ref)[1:, :]) / np.linalg.norm(ref[1:, :]))
    clamp = float(np.mean(np.abs(np.concatenate([out["xa"].ravel(), out["xb"].ravel()])) == 127))
    return {"all_cells_verify": ok, "tamper_caught": caught, "exact_identity": True, "p16_peel_max_err_grid2": peel_err,
            "useful_rel_err": rel, "clamp_rate": clamp, "widths": out["widths"]}


if __name__ == "__main__":
    for prm in (Params(), Params(r=64, s=12, q_d=84)):
        print(prm, selfcheck(p=prm))
