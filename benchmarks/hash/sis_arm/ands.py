"""In-circuit cost of opening a Module-SIS row in C-Flock: ANDs (XOR free) to recompute h = A x mod 2^32 from x's bits.

    python3 ands.py OUT.json            # numpy only; a minute on one core

A is public, so a product a_ij x_j is linear in x's bits: no AND. All the ANDs are in the sum over j, reduced mod 2^32.
Each row of A is one column-compression: column k holds the addend bits of weight 2^k, a full adder (3 bits in, sum
and carry out) or half adder (2 in) costs one AND, carries move to column k + 1, and column 31's carries fall off mod
2^32, so column 31 is XOR only. A column of b bits costs ceil((b - 1) / 2) cells and sends that many carries on. This is
an exact count for one seeded A; it is a construction's count, not a lower bound.

Three ways to turn a_ij x_j into addend bits, x_j an int8 in two's complement (bit 7 weighs -128):
- `pp-bin`: one addend per set bit of each constant c_t = a_ij w_t mod 2^32, t = 0..7 (the hash lane's count).
- `pp-naf`: the constants in non-adjacent form; a -1 digit at k adds the complemented bit (free) and a public -2^k,
  folded into one constant per row whose set bits are added once.
- `anf-g`: x's bits in groups of g; every AND-monomial of a group is computed once (2^g - g - 1 ANDs, shared by all n
  rows), and each row's term for the group is one 32-bit word, each bit an XOR of monomials: free in ANDs, but each such
  bit is a linear combination of up to 2^g columns, so it shows as R1CS nonzeros (`xor_fanin`), which ANDs don't count.

A row's opening also hashes the digest: x = SHA-512(P || digest) with the digest 4n bytes, then hm96's private
compressions (`row_ands` in verity/core/protocols/sampled_proofs/optimize.py: 58,120 ANDs a compression, 116,240 for
hm96).
"""
from __future__ import annotations

import json
import sys
from math import ceil

import numpy as np

SHA512_ANDS = 58_120
HM96_ANDS = 116_240
SHA512_PER_BYTE = SHA512_ANDS / 128
MASK = (1 << 32) - 1
WEIGHTS_INT8 = [1 << t for t in range(7)] + [(-128) & MASK]


def compress(native: np.ndarray, width: int = 32) -> np.ndarray:
    """ANDs to sum one row's addend bits mod 2^width; native[..., k] is the count of bits of weight 2^k."""
    native = np.concatenate([native, np.zeros((*native.shape[:-1], max(0, width - native.shape[-1])), np.int64)], -1)
    ands = np.zeros(native.shape[:-1], dtype=np.int64)
    carry = np.zeros_like(ands)
    for k in range(width):
        cells = (native[..., k] + carry) // 2
        if k < width - 1:
            ands += cells
        carry = cells
    return ands


def bit_counts(c: np.ndarray) -> np.ndarray:
    """c: uint64 array (..., j); returns (..., 32) counts of set bits per position summed over j."""
    return np.stack([((c >> np.uint64(k)) & np.uint64(1)).sum(axis=-1) for k in range(32)], axis=-1).astype(np.int64)


def naf(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Non-adjacent form of c mod 2^32: (pos, neg) bit masks with pos - neg = c mod 2^32, no two digits adjacent."""
    xh = c >> 1
    x3 = c + xh
    d = xh ^ x3
    pos, neg = x3 & d, xh & d
    assert np.all(((pos - neg) & MASK) == (c & MASK)) and np.all((pos & neg) == 0)
    assert np.all(((pos | neg) & ((pos | neg) >> 1)) == 0)
    return pos & MASK, neg & MASK


def products(a: np.ndarray, weights: list[int]) -> list[np.ndarray]:
    return [(a * np.uint64(w)) & np.uint64(MASK) for w in weights]


def pp_bin(a: np.ndarray, weights: list[int]) -> np.ndarray:
    return compress(sum(bit_counts(c) for c in products(a, weights)))


def pp_naf(a: np.ndarray, weights: list[int]) -> np.ndarray:
    native = 0
    const = np.zeros(a.shape[:-1], dtype=np.uint64)
    for c in products(a, weights):
        pos, neg = naf(c)
        native = native + bit_counts(pos) + bit_counts(neg)
        const = (const - neg.sum(axis=-1, dtype=np.uint64)) & np.uint64(MASK)
    return compress(native + bit_counts(const[..., None]))


def anf_weight(table: np.ndarray, g: int) -> np.ndarray:
    """Möbius transform over the last axis (length 2^g, 0/1 truth tables): the ANF coefficients."""
    t = table.copy()
    for i in range(g):
        step = 1 << i
        t = t.reshape(*t.shape[:-1], -1, 2 * step)
        t[..., step:] ^= t[..., :step]
        t = t.reshape(*t.shape[:-2], -1)
    return t


def anf(a: np.ndarray, g: int, sample: int = 64) -> tuple[np.ndarray, float, float]:
    """x_j's 8 bits in groups of g (g divides 8): one word per row and group. Returns per-row ANDs (without the shared
    monomials), the shared monomial ANDs per byte, and the mean XOR fan-in of a word bit."""
    groups = 8 // g
    words, fanins = [], []
    vals = np.arange(1 << g, dtype=np.int64)
    for gi in range(groups):
        w = [WEIGHTS_INT8[gi * g + t] for t in range(g)]
        coef = np.array([sum(((v >> t) & 1) * w[t] for t in range(g)) & MASK for v in vals], dtype=np.uint64)
        word = (a[..., None] * coef) & np.uint64(MASK)
        nonconst = np.stack([((word >> np.uint64(k)) & np.uint64(1)).max(axis=-1) for k in range(32)], axis=-1)
        words.append(nonconst.sum(axis=-2).astype(np.int64))
        tt = np.stack([((word[..., :sample, :] >> np.uint64(k)) & np.uint64(1)).astype(np.uint8) for k in range(32)],
                      axis=-2)
        fanins.append(anf_weight(tt, g).sum(axis=-1).mean())
    ands = compress(sum(words))
    return ands, groups * ((1 << g) - g - 1), float(np.mean(fanins))


def projection(m: int, p_bits: int = 61, rng=None) -> dict:
    """The live-coin projection check, per byte: with D = A x over Z committed at serve time and a coin rho drawn after
    the opening's witness is fixed, the circuit checks sum_j c_j x_j = rho . D mod p with c = rho^T A mod p public.
    `const`: c as gate constants (a circuit per coin), NAF partial products. `var`: c as public input words, so each bit
    of x ANDs each bit of c. Both sum into one integer of p_bits + 8 + log2 m + 1 bits; the reduction mod p and the
    rho . D side are per row, not per byte, and `opening_projection` adds them."""
    rng = rng or np.random.default_rng(61)
    width = p_bits + 8 + int(np.ceil(np.log2(m))) + 2
    c = rng.integers(0, 1 << p_bits, size=(1, m), dtype=np.uint64)
    xh = c >> np.uint64(1)
    x3 = c + xh
    d = xh ^ x3
    digits = sum(np.stack([((part >> np.uint64(k)) & np.uint64(1)).sum(axis=-1) for k in range(64)], -1)[0]
                 for part in (x3 & d, xh & d)).astype(np.int64)
    native = np.zeros((1, width), dtype=np.int64)
    for t in range(8):
        native[0, t:t + 64] += digits[:width - t]
    const = float(compress(native, width)[0]) / m + width / m
    pp = 8 * p_bits
    native_var = np.zeros((1, width), dtype=np.int64)
    for t in range(8):
        native_var[0, t:t + p_bits] += m
    var = (m * pp + float(compress(native_var, width)[0])) / m
    return {"p_bits": p_bits, "m": m, "width": width, "const_ands_per_byte": round(const, 1),
            "var_ands_per_byte": round(var, 1)}


def opening_projection(n: int, m: int, h: float, p_bits: int = 61) -> dict:
    """A projected opening of one row: h m, plus per row rho . D over the n committed 54-bit integers D_i (NAF partial
    products of 61-bit constants, about p_bits / 3 addends a bit), the reduction mod p, SHA-512 over the 7n-byte D and
    hm96."""
    rho_d = round(n * 54 * p_bits / 3)
    digest = SHA512_ANDS * ceil((7 * n + 17) / 128)
    total = h * m + rho_d + 4 * p_bits + digest + HM96_ANDS
    sha = HM96_ANDS + SHA512_ANDS * ceil((m + 17) / 128)
    return {"n": n, "m": m, "per_byte_ands": round(h * m), "rho_d_ands": rho_d, "digest_sha_ands": digest,
            "total": round(total), "sha512_row_ands": sha, "ratio_vs_sha512_row": round(total / sha, 2)}


def check_naf_circuit(trials: int = 200, m: int = 24) -> int:
    """Builds pp-naf's addend bits for one row and runs its full and half adders on bits, counting ANDs; asserts the
    result is a x mod 2^32 for random int8 x and the count is `compress`'s. Returns the trials run."""
    rng = np.random.default_rng(7)
    for _ in range(trials):
        a = rng.integers(0, 1 << 32, size=(1, m), dtype=np.uint64)
        x = rng.integers(-128, 128, size=m)
        xb = [[(int(v) >> t) & 1 for t in range(8)] for v in x]
        cols: list[list[int]] = [[] for _ in range(32)]
        const = 0
        for t, c in enumerate(products(a, WEIGHTS_INT8)):
            pos, neg = naf(c)
            for j in range(m):
                for k in range(32):
                    if (int(pos[0, j]) >> k) & 1:
                        cols[k].append(xb[j][t])
                    if (int(neg[0, j]) >> k) & 1:
                        cols[k].append(1 - xb[j][t])
                        const -= 1 << k
        for k in range(32):
            if ((const & MASK) >> k) & 1:
                cols[k].append(1)
        ands, out = 0, 0
        for k in range(32):
            col = cols[k]
            while len(col) > 1:
                if len(col) >= 3:
                    p, q, r = col.pop(), col.pop(), col.pop()
                    s, carry = p ^ q ^ r, ((p ^ r) & (q ^ r)) ^ r
                else:
                    p, q = col.pop(), col.pop()
                    s, carry = p ^ q, p & q
                col.insert(0, s)
                if k < 31:
                    ands += 1
                    cols[k + 1].append(carry)
            out |= (col[0] if col else 0) << k
        want = int(sum(int(a[0, j]) * int(x[j]) for j in range(m))) & MASK
        assert out == want, (out, want)
        assert ands == int(pp_naf(a, WEIGHTS_INT8)[0]), (ands, int(pp_naf(a, WEIGHTS_INT8)[0]))
    return trials


def opening(n: int, m: int, h: float) -> dict:
    """ANDs to open one row of m bytes under a digest of n words: the SIS recompute, SHA-512 over the 4n-byte digest,
    hm96."""
    digest = SHA512_ANDS * ceil((4 * n + 17) / 128)
    total = h * m + digest + HM96_ANDS
    sha = HM96_ANDS + SHA512_ANDS * ceil((m + 17) / 128)
    return {"n": n, "m": m, "sis_ands": round(h * m), "digest_sha_ands": digest, "hm96_ands": HM96_ANDS,
            "total": round(total), "sha512_row_ands": sha, "ratio_vs_sha512_row": round(total / sha, 1)}


def main() -> None:
    out = sys.argv[1]
    rng = np.random.default_rng(20261028)
    m_sample = 2048
    rows = []
    for n in (96, 128, 160, 192, 208, 224, 256):
        a = rng.integers(0, 1 << 32, size=(n, m_sample), dtype=np.uint64)
        for name, f in (("pp-bin", lambda a: pp_bin(a, WEIGHTS_INT8)), ("pp-naf", lambda a: pp_naf(a, WEIGHTS_INT8))):
            per_row = f(a)
            h = float(per_row.sum()) / m_sample
            rows.append({"encoding": "int8", "construction": name, "n": n, "ands_per_byte": round(h, 1),
                         "per_element_row": round(h / n, 2), "xor_fanin": 1, "x_sha512": round(h / SHA512_PER_BYTE, 1)})
        for g in (4, 8):
            per_row, shared, fanin = anf(a[:, :256] if g == 8 else a, g)
            mm = 256 if g == 8 else m_sample
            h = float(per_row.sum()) / mm + shared
            rows.append({"encoding": "int8", "construction": f"anf-{g}", "n": n, "ands_per_byte": round(h, 1),
                         "per_element_row": round((h - shared) / n, 2), "xor_fanin": round(fanin, 1),
                         "x_sha512": round(h / SHA512_PER_BYTE, 1)})
    for enc, bits, ns in (("nibble", 4, (48, 64, 80, 96)), ("bit", 1, (16, 24, 32, 40, 48))):
        for n in ns:
            a = rng.integers(0, 1 << 32, size=(n, m_sample * 8 // bits), dtype=np.uint64)
            c = a if bits == 1 else np.concatenate(products(a, [1, 2, 4, 8]), axis=-1)
            h = float(pp_naf(c, [1]).sum()) / m_sample
            rows.append({"encoding": enc, "construction": "pp-naf", "n": n, "ands_per_byte": round(h, 1),
                         "per_element_row": round(h / n, 2), "xor_fanin": 1, "x_sha512": round(h / SHA512_PER_BYTE, 1)})
    checked = check_naf_circuit()
    openings = [{"encoding": r["encoding"], "construction": r["construction"], **opening(r["n"], m, r["ands_per_byte"])}
                for r in rows for m in (4096, 16384)]
    proj = [projection(m) for m in (4096, 16384)]
    proj_open = [{"variant": v, **opening_projection(n, p["m"], p[f"{v}_ands_per_byte"])}
                 for p in proj for v in ("const", "var") for n in (192, 208, 224, 256)]
    json.dump({"seed": 20261028, "m_sample": m_sample, "sha512_ands_per_byte": SHA512_PER_BYTE,
               "naf_circuit_checked": checked, "rows": rows, "openings": openings, "projection": proj,
               "projection_openings": proj_open}, open(out, "w"), indent=1)
    for p in proj:
        print(f"projection m={p['m']}: const {p['const_ands_per_byte']} ANDs/B, var {p['var_ands_per_byte']} ANDs/B")
    print(f"pp-naf circuit evaluated on bits against a x mod 2^32, and its AND count against compress(): {checked} trials")
    for r in rows:
        print(f"{r['encoding']:6} {r['construction']:7} n={r['n']:4}  {r['ands_per_byte']:9.1f} ANDs/B"
              f"  ({r['per_element_row']:6.2f} per element-row, x{r['x_sha512']} SHA-512, XOR fan-in {r['xor_fanin']})")


if __name__ == "__main__":
    main()
