"""CPU reference for fgemm.cu's gate dump: every output and receipt, bit for bit, from the dumped inputs alone.

python3 ref.py DIR prints one JSON line of checks and exits 0 only if every check passes.

Recomputed independently of the device: the noised operands A~ = Q(A) + E_A F_A^T and B~, the tail rows (the wide factors
W = B~ F_A and V = A~ F_B - E_A G in two s8 limbs), U = Q(A) Q(B)^T exactly (int64, then checked to fit s32), and every
receipt chain over the running sums T_s = A~[:, :32 s] B~[:, :32 s]^T in the kernel's canonical order (fgemm.cu's header).
"""

import json
import sys
from pathlib import Path

import numpy as np

M32 = np.uint32(0xFFFFFFFF)


def mix32(x):
    x = np.asarray(x, dtype=np.uint32)
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint32(16))
        x = x * np.uint32(0x7FEB352D)
        x = x ^ (x >> np.uint32(15))
        x = x * np.uint32(0x846CA68B)
        return x ^ (x >> np.uint32(16))


R1 = [15, 18, 20, 5, 29, 25, 4, 9, 1, 6, 12, 28, 8, 7, 31, 19, 10, 17, 3, 11, 27, 14, 2, 26, 16, 30, 13, 23, 21, 24, 22]
R2 = [6, 23, 7, 19, 4, 14, 10, 12, 9, 28, 11, 20, 26, 2, 22, 17, 5, 16, 15, 3, 25, 8, 13, 29, 30, 27, 21, 24, 18]


def rot_of(duo2, lane, t):
    """fgemm.cu's rot_of: duo's rotation, else duo2's R1 (lane 0) and R2 (lane 1)."""
    if not duo2:
        return ((t * 7 + 5) % 31) + 1
    return R1[t % 31] if lane == 0 else R2[t % 29]


def key_mul(lane, t):
    return int(mix32(np.uint32(0xA0000000 + 2 * (lane * 16 + t)))) | 1


def key_xor(lane, t):
    return int(mix32(np.uint32(0xB0000000 + 2 * (lane * 16 + t) + 1)))


def rotl(x, r):
    return (x << np.uint32(r)) | (x >> np.uint32(32 - r))


def f_matrix(own, k):
    """F (k x 32) from the own-side descriptors."""
    f = np.zeros((k, 32), dtype=np.int64)
    cols = np.arange(k)
    p1, p2 = own & 31, (own >> 8) & 31
    s1 = np.where((own >> 16) & 1, -1, 1)
    s2 = np.where((own >> 17) & 1, -1, 1)
    np.add.at(f, (cols, p1), s1)
    np.add.at(f, (cols, p2), s2)
    return f


def limbs(y):
    lo = ((y & 0xFF) ^ 0x80) - 0x80
    hi = (y - lo) >> 8
    return hi, lo


def main(d: Path) -> int:
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, seed = meta["m"], meta["n"], meta["k"], meta["seed"]
    TM, BN, WM, NCT = meta["TM"], meta["BN"], meta["WM"], meta["NCT"]

    def ld(name, dt, shape):
        return np.fromfile(d / name, dtype=dt).reshape(shape)

    QA = ld("QA.bin", np.int8, (m, k)).astype(np.int64)
    QB = ld("QB.bin", np.int8, (n, k)).astype(np.int64)
    EA = ld("EA.bin", np.int8, (m, 32)).astype(np.int64)
    EB = ld("EB.bin", np.int8, (n, 32)).astype(np.int64)
    ownA = np.fromfile(d / "ownA.bin", dtype=np.uint32).astype(np.int64)
    ownB = np.fromfile(d / "ownB.bin", dtype=np.uint32).astype(np.int64)
    KA = np.array([[key_mul(ln, t) for t in range(16)] for ln in range(2)], dtype=np.uint32)
    KK = np.array([[key_xor(ln, t) for t in range(16)] for ln in range(2)], dtype=np.uint32)
    FA, FB = f_matrix(ownA, k), f_matrix(ownB, k)
    G = FA.T @ FB
    XA = QA + EA @ FA.T
    XB = QB + EB @ FB.T
    checks = {"limb_overflow_flag": meta["limb_overflow"] == 0}
    checks["XA_in_s8"] = bool(np.abs(XA).max() <= 127)
    checks["XB_in_s8"] = bool(np.abs(XB).max() <= 127)
    checks["XA"] = bool((ld("XA.bin", np.int8, (m, k)) == XA).all())
    checks["XB"] = bool((ld("XB.bin", np.int8, (n, k)) == XB).all())
    W = XB @ FA
    V = XA @ FB - EA @ G
    Whi, Wlo = limbs(W)
    Vhi, Vlo = limbs(V)
    checks["limbs_in_s8"] = bool(max(np.abs(Whi).max(), np.abs(Vhi).max()) <= 127)
    PA = np.concatenate([EA, Vhi, EA, Vlo], axis=1)
    PB = np.concatenate([Whi, EB, Wlo, EB], axis=1)
    checks["PA"] = bool((ld("PA.bin", np.int8, (m, 128)) == PA).all())
    checks["PB"] = bool((ld("PB.bin", np.int8, (n, 128)) == PB).all())
    U = QA @ QB.T
    checks["U_fits_s32"] = bool(np.abs(U).max() < 2**31)
    peel = XA @ XB.T - EA @ W.T - V @ EB.T
    checks["peel_identity"] = bool((peel == U).all())
    U32 = U.astype(np.int32)
    for name in ["lt", "plain", "g0", "duo32", "duo64", "duo2_32", "duo2_64"]:
        checks[f"U_{name}"] = bool((ld(f"U_{name}.bin", np.int32, (m, n)) == U32).all())

    # running sums at every 32 columns, as the kernel's s32 words
    T = np.zeros((m, n), dtype=np.int64)
    Ts = []
    for s in range(k // 32):
        T = T + XA[:, 32 * s : 32 * s + 32] @ XB[:, 32 * s : 32 * s + 32].T
        assert np.abs(T).max() < 2**31
        Ts.append((T & 0xFFFFFFFF).astype(np.uint32))
    tiles_m, tiles_n = m // TM, n // BN
    warps_n = BN // 32
    MI, NI = WM // 16, 4
    tile = np.arange(tiles_m * tiles_n)
    tm, tn = tile // tiles_n, tile % tiles_n
    ctid = np.arange(NCT)
    warp, lane = ctid // 32, ctid % 32
    wm, wn, g, q = warp // warps_n, warp % warps_n, lane >> 2, lane & 3
    e = np.arange(4)
    # rows[tile, ctid, e, mi], cols[tile, ctid, e, ni]
    rows = (tm[:, None, None, None] * TM + wm[None, :, None, None] * WM + 16 * np.arange(MI)[None, None, None, :]
            + g[None, :, None, None] + 8 * (e >> 1)[None, None, :, None])
    cols = (tn[:, None, None, None] * BN + wn[None, :, None, None] * 32 + 8 * np.arange(NI)[None, None, None, :]
            + 2 * q[None, :, None, None] + (e & 1)[None, None, :, None])
    for name, rs, nch in [("duo32", 32, 4), ("duo64", 64, 4), ("duo2_32", 32, 8), ("duo2_64", 64, 8)]:
        idx = ((tile[:, None, None] * NCT + ctid[None, :, None]) * nch + np.arange(nch)[None, None, :]).astype(np.uint32)
        y = mix32(np.uint32(seed) ^ mix32(idx))
        z = mix32(y + np.uint32(0x9E3779B9))
        bounds = range(1, k // 32 + 1) if rs == 32 else range(2, k // 32 + 1, 2)
        with np.errstate(over="ignore"):
            for s in bounds:
                Tw = Ts[s - 1]
                for mi in range(MI):
                    for ni in range(NI):
                        t = mi * NI + ni
                        p = Tw[rows[:, :, :, mi], cols[:, :, :, ni]]   # [tile, ctid, e]
                        for ln in range(nch // 4):
                            sl = slice(4 * ln, 4 * ln + 4)
                            u = y[:, :, sl] ^ p ^ KK[ln, t]
                            y[:, :, sl] = u * KA[ln, t] + z[:, :, sl]
                            z[:, :, sl] = rotl(u, rot_of(nch == 8, ln, t)) + y[:, :, sl]
            fin = y[:, :, :4] ^ rotl(z[:, :, :4], 16)
            if nch == 8:
                fin = fin ^ rotl(y[:, :, 4:] ^ rotl(z[:, :, 4:], 16), 8)
        got = np.fromfile(d / f"R_{name}.bin", dtype=np.uint32).reshape(fin.shape)
        checks[f"R_{name}"] = bool((got == fin).all())
        checks[f"R_{name}_words"] = int(fin.size)
    ok = all(v for v in checks.values() if isinstance(v, bool))
    print(json.dumps({"dir": d.name, "m": m, "n": n, "k": k, "TM": TM, "BN": BN, "grid": meta["grid"], "ok": ok, **checks}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
