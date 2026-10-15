"""CPU reference for mgemm.cu's gate dump: every formed operand, product and receipt, bit for bit, from the dumped values.

python3 mref.py DIR [--no-receipts] prints one JSON line of checks and exits 0 only if every check passes.

Recomputed independently of the device, per format (int8, E4M3's integer grid): the full-rank operands [A+R | R | A] and
[B+S | -(B+S) | -S] (R and S hashed as the kernel hashes them), the low-rank operands [A~ | -w_l U | -Q_l] and
[B~ | P_l | w_l W] (P = B~ V, Q = A~ Z - U G in the kernel's digits), the peel identity A' B'^T = A B^T in int64, every
arm's product against A B^T, and the receipts (duo, TB4C at 32 and 64) over the masked product's running sums.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "floored_gemm"))
from ref import key_mul, key_xor, mix32, rot_of, rotl, tbc_receipts  # noqa: E402

RANKS = [8, 16, 32, 64, 128]
FMTS = ["i8", "e4"]


def e4m3_table():
    t = np.zeros(256, dtype=np.float64)
    for b in range(256):
        s, e, m = b >> 7, (b >> 3) & 15, b & 7
        v = np.nan if (e == 15 and m == 7) else (m / 8 * 2.0**-6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))
        t[b] = -v if s else v
    return t


E4 = e4m3_table()


def decode(fmt, raw):
    if fmt == "i8":
        return raw.view(np.int8).astype(np.int64)
    v = E4[raw]
    if np.isnan(v).any() or (v != np.round(v)).any():
        return None
    return v.astype(np.int64)


def fdiv(a, b):
    return np.floor_divide(a, b)


def digits(fmt, y, L):
    """The kernel's digits of y (rows x r) in L limbs and their weights, and whether anything is left over."""
    ds, ws = [], []
    if fmt == "i8":
        d = y - 127 * fdiv(y + 63, 127)
        q = (y - d) // 127
        ds.append(d), ws.append(1)
        for _ in range(1, L):
            d = np.clip(q, -127, 127)
            q = q - d
            ds.append(d), ws.append(127)
    else:
        d = y - 16 * fdiv(y + 8, 16)
        q = (y - d) // 16
        ds.append(d), ws.append(1)
        for l in range(1, L):
            if l == 1:
                d = q - 16 * fdiv(q + 8, 16)
                q = (q - d) // 16
                ws.append(16)
            else:
                d = np.clip(q, -16, 16)
                q = q - d
                ws.append(256)
            ds.append(d)
    return ds, ws, bool((q != 0).any())


def full_masks(rows, k, seed, fmt):
    word = (np.arange(rows, dtype=np.uint64)[:, None] * (k // 4) + np.arange(k // 4, dtype=np.uint64)[None, :])
    h = mix32(np.uint32(seed) ^ mix32((word & 0xFFFFFFFF).astype(np.uint32)))
    hb = np.stack([(h >> np.uint32(8 * i)) & np.uint32(0xFF) for i in range(4)], axis=-1).reshape(rows, k).astype(np.int64)
    return (hb & 0x7F) - 64 if fmt == "i8" else (hb & 0xF) - 8


def duo_receipts(seed, Ts, rs, TM, BN, WM, NCT, tiles_m, tiles_n):
    """ref.py's duo over the boundary words Ts (one every rs columns), [tile, ctid, 8]."""
    KA = np.array([[key_mul(ln, t) for t in range(16)] for ln in range(2)], dtype=np.uint32)
    KK = np.array([[key_xor(ln, t) for t in range(16)] for ln in range(2)], dtype=np.uint32)
    warps_n, MI, NI, nch = BN // 32, WM // 16, 4, 8
    tile = np.arange(tiles_m * tiles_n)
    tm, tn = tile // tiles_n, tile % tiles_n
    ctid = np.arange(NCT)
    warp, lane = ctid // 32, ctid % 32
    wm, wn, g, q = warp // warps_n, warp % warps_n, lane >> 2, lane & 3
    e = np.arange(4)
    rows = (tm[:, None, None, None] * TM + wm[None, :, None, None] * WM + 16 * np.arange(MI)[None, None, None, :]
            + g[None, :, None, None] + 8 * (e >> 1)[None, None, :, None])
    cols = (tn[:, None, None, None] * BN + wn[None, :, None, None] * 32 + 8 * np.arange(NI)[None, None, None, :]
            + 2 * q[None, :, None, None] + (e & 1)[None, None, :, None])
    idx = ((tile[:, None, None] * NCT + ctid[None, :, None]) * nch + np.arange(nch)[None, None, :]).astype(np.uint32)
    y = mix32(np.uint32(seed) ^ mix32(idx))
    z = mix32(y + np.uint32(0x9E3779B9))
    with np.errstate(over="ignore"):
        for Tw in Ts:
            for mi in range(MI):
                for ni in range(NI):
                    t = mi * NI + ni
                    p = Tw[rows[:, :, :, mi], cols[:, :, :, ni]]
                    sl = slice(4 * (t % 2), 4 * (t % 2) + 4)
                    u = y[:, :, sl] ^ p ^ KK[0, t]
                    y[:, :, sl] = u * KA[0, t] + z[:, :, sl]
                    z[:, :, sl] = rotl(u, rot_of(False, 0, t)) + y[:, :, sl]
        return y ^ rotl(z, 16)


def words(fmt, T):
    if fmt == "i8":
        assert np.abs(T).max() < 2**31
        return (T & 0xFFFFFFFF).astype(np.uint32)
    assert np.abs(T).max() < 2**24
    return T.astype(np.float32).view(np.uint32)


def main(d: Path, receipts: bool) -> int:
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, seed = meta["m"], meta["n"], meta["k"], meta["seed"]
    TM, BN, WM, NCT = meta["TM"], meta["BN"], meta["WM"], meta["NCT"]
    tiles_m, tiles_n = m // TM, n // BN
    arms = set(meta["arms"])
    checks = {"digit_overflow_flag": meta["digit_overflow"] == 0}
    seen = set()

    def ld(name, dt, shape):
        return np.fromfile(d / name, dtype=dt).reshape(shape)

    def check_product(name, fmt, want):
        seen.add(name)
        if fmt == "i8":
            got = ld(f"U_{name}.bin", np.int32, (m, n)).astype(np.int64)
        else:
            got = ld(f"U_{name}.bin", np.float32, (m, n)).astype(np.float64)
        checks[f"U_{name}"] = bool((got == want).all())

    def check_receipts(tag, fmt, XA, XB):
        """The receipt arms <rc>_<tag>_<fmt> over the running sums of the first k columns of XA XB^T."""
        if not receipts:
            return
        T = np.zeros((m, n), dtype=np.int64)
        T32, T64 = [], []
        for s in range(k // 32):
            T = T + XA[:, 32 * s : 32 * s + 32] @ XB[:, 32 * s : 32 * s + 32].T
            w = words(fmt, T)
            T32.append(w)
            if s % 2 == 1:
                T64.append(w)
        for rc, Ts, kind in [("cduo32", T32, "duo"), ("cduo64", T64, "duo"), ("ct4c32", T32, 8), ("ct4c64", T64, 8)]:
            name = f"{rc}_{tag}_{fmt}"
            seen.add(name)
            if kind == "duo":
                want = duo_receipts(seed, Ts, 0, TM, BN, WM, NCT, tiles_m, tiles_n)
            else:
                want = tbc_receipts(kind, Ts, TM, BN, WM, NCT, tiles_m, tiles_n)
            got = ld(f"R_{name}.bin", np.uint32, (tiles_m * tiles_n, NCT, 8))
            checks[f"R_{name}"] = bool((got == want).all())
            if kind != "duo":
                checks[f"R_{name}_nonzero"] = bool((want[:, :, :4] != 0).mean() > 0.99)

    for fmt in FMTS:
        A = ld(f"A_{fmt}.bin", np.int8, (m, k)).astype(np.int64)
        B = ld(f"B_{fmt}.bin", np.int8, (n, k)).astype(np.int64)
        AB = A @ B.T
        checks[f"AB_{fmt}_exact_range"] = bool(np.abs(AB).max() < (2**31 if fmt == "i8" else 2**24))
        for name in [f"plain_{fmt}", f"full_{fmt}", f"cg_full_{fmt}"]:
            check_product(name, fmt, AB)
        for rc in ["cduo32", "cduo64", "ct4c32", "ct4c64"]:
            check_product(f"{rc}_full_{fmt}", fmt, AB)

        R = full_masks(m, k, seed, fmt)
        S = full_masks(n, k, seed ^ 0x5A5A5A5A, fmt)
        XA = np.concatenate([A + R, R, A], axis=1)
        XB = np.concatenate([B + S, -(B + S), -S], axis=1)
        lim = 127 if fmt == "i8" else 16
        checks[f"full_{fmt}_grid"] = bool(max(np.abs(XA).max(), np.abs(XB).max()) <= lim)
        checks[f"full_{fmt}_peel"] = bool((XA @ XB.T == AB).all())
        gA = decode(fmt, ld(f"XA_fullform_{fmt}.bin", np.uint8, (m, 3 * k)))
        gB = decode(fmt, ld(f"XB_fullform_{fmt}.bin", np.uint8, (n, 3 * k)))
        seen.add(f"fullform_{fmt}")
        checks[f"XA_fullform_{fmt}"] = gA is not None and bool((gA == XA).all())
        checks[f"XB_fullform_{fmt}"] = gB is not None and bool((gB == XB).all())
        check_receipts("full", fmt, XA, XB)

        for ri, r in enumerate(RANKS):
            lw = meta["low"][str(r)]
            fi = FMTS.index(fmt)
            L, Kc = lw["L"][fi], lw["Kc"][fi]
            U = ld(f"UA{r}.bin", np.int8, (m, r)).astype(np.int64)
            W = ld(f"UB{r}.bin", np.int8, (n, r)).astype(np.int64)
            V = ld(f"V{r}.bin", np.int32, (k, r)).astype(np.int64)
            Z = ld(f"Z{r}.bin", np.int32, (k, r)).astype(np.int64)
            G = V.T @ Z
            At, Bt = A + U @ V.T, B + W @ Z.T
            P = Bt @ V
            Q = At @ Z - U @ G
            dP, wts, oP = digits(fmt, P, L)
            dQ, _, oQ = digits(fmt, Q, L)
            checks[f"low{r}_{fmt}_digits_fit"] = not (oP or oQ)
            XA = np.zeros((m, Kc), dtype=np.int64)
            XB = np.zeros((n, Kc), dtype=np.int64)
            XA[:, :k], XB[:, :k] = At, Bt
            for l in range(L):
                XA[:, k + l * r : k + (l + 1) * r] = -wts[l] * U
                XA[:, k + (L + l) * r : k + (L + l + 1) * r] = -dQ[l]
                XB[:, k + l * r : k + (l + 1) * r] = dP[l]
                XB[:, k + (L + l) * r : k + (L + l + 1) * r] = wts[l] * W
            checks[f"low{r}_{fmt}_grid"] = bool(max(np.abs(XA).max(), np.abs(XB).max()) <= lim)
            checks[f"low{r}_{fmt}_peel"] = bool((XA @ XB.T == AB).all())
            for form in ["lowform", "lowtcform"]:
                gA = decode(fmt, ld(f"XA_{form}{r}_{fmt}.bin", np.uint8, (m, Kc)))
                gB = decode(fmt, ld(f"XB_{form}{r}_{fmt}.bin", np.uint8, (n, Kc)))
                seen.add(f"{form}{r}_{fmt}")
                checks[f"XA_{form}{r}_{fmt}"] = gA is not None and bool((gA == XA).all())
                checks[f"XB_{form}{r}_{fmt}"] = gB is not None and bool((gB == XB).all())
            for name in [f"low{r}_{fmt}", f"lowsep{r}_{fmt}", f"lowtc{r}_{fmt}", f"cg_low{r}_{fmt}"]:
                check_product(name, fmt, AB)
            for rc in ["cduo32", "cduo64", "ct4c32", "ct4c64"]:
                check_product(f"{rc}_low{r}_{fmt}", fmt, AB)
            check_receipts(f"low{r}", fmt, XA, XB)

    missing = sorted(arms - seen)
    checks["every_arm_checked"] = not missing
    ok = all(v for v in checks.values() if isinstance(v, bool))
    failed = [key for key, v in checks.items() if v is False]
    print(json.dumps({"dir": d.name, "m": m, "n": n, "k": k, "ok": ok, "receipts": receipts, "failed": failed,
                      "unchecked_arms": missing, "n_checks": len(checks), **checks}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), "--no-receipts" not in sys.argv[2:]))
