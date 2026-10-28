"""Rows and nonzeros as C-Flock pays them, for SHA-512 and the SIS openings, on Verity's own circuit builder:

    uv run python benchmarks/hash/sis_arm/rows.py OUT.json

`ands.py` counts ANDs with XOR free, the model of hm96 PROTOCOL §5's 58,120 a SHA-512 compression. A C-Flock row is one
committed bit, `A.z * B.z = z_i` over F2. An XOR is free in rows, but its linear form is the row's nonzeros, and a sum's
form compounds wherever it is reused. So `compress_gadget` also commits copies (one row each, `A = form, B = 1`) that
keep forms short. Its circuit-check pin, `Sha512Compress_v1`, is 81,402 rows: 58,120 ANDs plus copies.

This builds both on `verity.ml.circuits.boolean.forms.Circuit` and tallies each row's nonzeros, |A| + |B| (the
constant 1 counts as a column).
- SHA-512: one `compress_gadget` on private chaining value and message.
- SIS: one row of A (n = 1) over `m` int8 bytes, the addend bits of `ands.py`'s constructions, reduced mod 2^32 by a FIFO
  carry-save of full adders (`maj`, one AND) with column 31 XOR-only. A sum or carry whose form exceeds `lmax` terms is
  committed. lmax = 6 is about what `compress_gadget` allows (a ripple carry is committed every CARRY_EVERY = 5 bits);
  lmax = None commits nothing, so the ANDs are `ands.py`'s and the nonzeros grow.
- The projection check: sum_j c_j x_j over Z for 61-bit public c_j, one row of the data (not times n).
Per-byte figures scale by n (the rows of A are independent sums over the same x bits); anf-g's monomials are shared.
"""
from __future__ import annotations

import json
import sys
from collections import deque

import numpy as np

from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, maj
from verity.ml.commitments.gates.sha512 import compress_gadget

MASK = (1 << 32) - 1
WEIGHTS_INT8 = [1 << t for t in range(7)] + [(-128) & MASK]


def tally(C: Circuit) -> dict:
    ands = copies = nnz = 0
    for kind, data in zip(C.kind, C.data):
        if kind != "and":
            continue
        a, b = data
        nnz += len(a.s) + a.c + len(b.s) + b.c
        if b.is_const():
            copies += 1
        else:
            ands += 1
    return {"ands": ands, "copies": copies, "rows": ands + copies, "nnz": nnz}


def fit(C: Circuit, f, lmax):
    return C.commit([f])[0] if lmax is not None and len(f.s) > lmax else f


def compress_mod(C: Circuit, cols: list[list], width: int, lmax) -> list:
    cols = [deque(c) for c in cols] + [deque() for _ in range(width - len(cols))]
    out = []
    for k in range(width):
        q = cols[k]
        while len(q) > 1:
            if len(q) >= 3:
                x, y, z = sorted((q.popleft(), q.popleft(), q.popleft()), key=lambda f: -len(f.s))
                if k < width - 1:
                    cols[k + 1].append(fit(C, maj(C, x, y, z), lmax))
                q.append(fit(C, x ^ y ^ z, lmax))
            else:
                x, y = q.popleft(), q.popleft()
                if k < width - 1:
                    cols[k + 1].append(C.AND(x, y))
                q.append(fit(C, x ^ y, lmax))
        out.append(q[0] if q else ZERO)
    return out


def naf(c: int) -> tuple[int, int]:
    xh = c >> 1
    x3 = c + xh
    d = xh ^ x3
    return x3 & d, xh & d


def sis_row_naf(a: np.ndarray, lmax) -> dict:
    """One row of A over m bytes, pp-naf addends."""
    m = len(a)
    C = Circuit()
    x = [C.input(f"x{j}", 8) for j in range(m)]
    cols: list[list] = [[] for _ in range(32)]
    const = 0
    for j in range(m):
        for t, w in enumerate(WEIGHTS_INT8):
            pos, neg = naf((int(a[j]) * w) & MASK)
            for k in range(32):
                if pos >> k & 1:
                    cols[k].append(x[j][t])
                if neg >> k & 1:
                    cols[k].append(x[j][t] ^ ONE)
                    const -= 1 << k
    for k in range(32):
        if (const & MASK) >> k & 1:
            cols[k].append(ONE)
    compress_mod(C, cols, 32, lmax)
    return tally(C)


def sis_row_anf(a: np.ndarray, g: int, lmax) -> dict:
    """One row of A over m bytes, anf-g: each byte's g-bit groups' monomials once, each word bit an XOR of monomials. The
    monomials' ANDs are shared by every row of A, so they are reported apart (`shared_ands`)."""
    m = len(a)
    C = Circuit()
    x = [C.input(f"x{j}", 8) for j in range(m)]
    cols: list[list] = [[] for _ in range(32)]
    shared = 0
    for j in range(m):
        for lo in range(0, 8, g):
            bits = x[j][lo:lo + g]
            mono = {0: ONE}
            for s in range(1, 1 << g):
                top = s.bit_length() - 1
                rest = s & ~(1 << top)
                if rest == 0:
                    mono[s] = bits[top]
                else:
                    mono[s] = C.AND(mono[rest], bits[top])
                    shared += 1
            table = []
            for v in range(1 << g):
                val = sum(WEIGHTS_INT8[lo + i] for i in range(g) if v >> i & 1)
                table.append((int(a[j]) * val) & MASK)
            for k in range(32):
                tt = [table[v] >> k & 1 for v in range(1 << g)]
                for i in range(g):
                    for v in range(1 << g):
                        if v >> i & 1:
                            tt[v] ^= tt[v ^ (1 << i)]
                f = ZERO
                for s in range(1 << g):
                    if tt[s]:
                        f = f ^ mono[s]
                if not f.is_const() or f.c:
                    cols[k].append(fit(C, f, lmax))
    t = tally(C)
    compress_mod(C, cols, 32, lmax)
    total = tally(C)
    own = {"ands": total["ands"] - shared, "copies": total["copies"], "rows": total["rows"] - shared,
           "nnz": total["nnz"] - 2 * shared}
    return {**own, "shared_ands": shared, "word_bit_copies": t["copies"]}


def projection_row(m: int, lmax, p_bits: int = 61, seed: int = 61) -> dict:
    """sum_j c_j x_j over Z for random p_bits-bit public c_j, NAF addends, into width bits."""
    rng = np.random.default_rng(seed)
    width = p_bits + 8 + int(np.ceil(np.log2(m))) + 2
    full = (1 << width) - 1
    C = Circuit()
    x = [C.input(f"x{j}", 8) for j in range(m)]
    cols: list[list] = [[] for _ in range(width)]
    const = 0
    for j in range(m):
        c = int(rng.integers(0, 1 << p_bits, dtype=np.uint64))
        for t in range(8):
            w = (c << t) if t < 7 else ((-(c << 7)) & full)
            pos, neg = naf(w)
            for k in range(width):
                if pos >> k & 1:
                    cols[k].append(x[j][t])
                if neg >> k & 1:
                    cols[k].append(x[j][t] ^ ONE)
                    const -= 1 << k
    for k in range(width):
        if (const & full) >> k & 1:
            cols[k].append(ONE)
    compress_mod(C, cols, width, lmax)
    return {**tally(C), "width": width}


def sha512() -> dict:
    C = Circuit()
    cv = [C.input(f"cv{i}", 64) for i in range(8)]
    msg = [C.input(f"w{i}", 64) for i in range(16)]
    compress_gadget(C, cv, msg)
    return tally(C)


def main() -> None:
    out = sys.argv[1]
    rng = np.random.default_rng(20261028)
    sha = sha512()
    res = {"sha512_compression": sha, "sha512_per_byte": {k: round(v / 128, 1) for k, v in sha.items()}, "sis": []}
    print(f"sha512 compression: {sha}  per byte {res['sha512_per_byte']}", flush=True)
    for m in (128, 512):
        a = rng.integers(0, 1 << 32, size=m, dtype=np.uint64)
        for lmax in (None, 6):
            rows = [("pp-naf", sis_row_naf(a, lmax)), ("anf-8", sis_row_anf(a, 8, lmax)),
                    ("proj-const", projection_row(m, lmax))]
            for name, r in rows:
                per_byte = {k: round(v / m, 2) for k, v in r.items() if k in ("ands", "copies", "rows", "nnz")}
                row = {"construction": name, "m": m, "lmax": lmax, **r, "per_byte_per_A_row": per_byte}
                res["sis"].append(row)
                print(json.dumps(row), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
