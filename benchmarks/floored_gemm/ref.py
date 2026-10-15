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


TB = {"tb4_32": 5, "tb2_32": 6, "tb1_32": 7}


def key_tb(kind, t, lane, j):
    """fgemm.cu's key_tb: register j of key t's B fragment in lane `lane`."""
    return int(mix32(np.uint32((0xC0000000 + ((kind * 64 + t) * 32 + lane) * 2 + j) & 0xFFFFFFFF)))


def tb_key(kind, t):
    """K_t as an s8 matrix (k x 8): B fragment register j of lane 4 n + q holds K[16 j + 4 q + i, n] as byte i."""
    kb = 2 if kind == 5 else 1
    K = np.zeros((16 * kb, 8), dtype=np.int64)
    for n in range(8):
        for qq in range(4):
            for jr in range(kb):
                w = key_tb(kind, t, 4 * n + qq, jr)
                for i in range(4):
                    b = (w >> (8 * i)) & 0xFF
                    K[16 * jr + 4 * qq + i, n] = b - 256 if b >= 128 else b
    return K


def tb_operand(kind, blk, mi, ni):
    """The receipt IMMA's A operand (16 x k, u8) for accumulator tile (mi, ni) of a warp tile's bytes blk [WM, 32, 4]."""
    b = blk[16 * mi : 16 * mi + 16, 8 * ni : 8 * ni + 8]
    if kind == 5:   # A row g: words (g, 2q) at k 4q.., (g + 8, 2q) at 16 + 4q..; A row g + 8: the same for column 2q + 1
        top = np.concatenate([b[:8, 0::2, :].reshape(8, 16), b[8:, 0::2, :].reshape(8, 16)], axis=1)
        bot = np.concatenate([b[:8, 1::2, :].reshape(8, 16), b[8:, 1::2, :].reshape(8, 16)], axis=1)
        return np.concatenate([top, bot], axis=0)
    if kind == 6:   # word (r, c) bytes 0, 1 at k 2c, 2c + 1
        return b[:, :, :2].reshape(16, 16)
    x = blk[16 * mi : 16 * mi + 16, 8 * (ni - 1) : 8 * ni, 1].reshape(16, 4, 2)
    y = b[:, :, 1].reshape(16, 4, 2)
    return np.stack([x, y], axis=2).reshape(16, 16)   # k 4q + c: tile ni - 1's word (r, 2q + c); 4q + 2 + c: tile ni's


def tb_receipts(kind, SB, TM, BN, WM, NCT, tiles_m, tiles_n):
    """Every thread's R fragment, [tile, ctid, 8] (R0..R3, then four zeros), from SB = the bytes of every word summed over
    the checkpoints: R = sum over checkpoints and positions t of A_t K_t, and K_t doesn't depend on the checkpoint."""
    MI, NI, warps_n = WM // 16, 4, BN // 32
    keys = {t: tb_key(kind, t) for t in range(MI * NI)}
    out = np.zeros((tiles_m * tiles_n, NCT, 8), dtype=np.uint32)
    lane = np.arange(32)
    g, q = lane >> 2, lane & 3
    for ti in range(tiles_m * tiles_n):
        tm, tn = ti // tiles_n, ti % tiles_n
        for w in range(NCT // 32):
            r0, c0 = tm * TM + (w // warps_n) * WM, tn * BN + (w % warps_n) * 32
            blk = SB[r0 : r0 + WM, c0 : c0 + 32]
            R = np.zeros((16, 8), dtype=np.int64)
            for mi in range(MI):
                for ni in range(NI):
                    if kind == 7 and ni % 2 == 0:
                        continue
                    t = mi * NI + ni
                    R += tb_operand(kind, blk, mi, ni) @ keys[t // 2 if kind == 7 else t]
            R = (R & 0xFFFFFFFF).astype(np.uint32)
            s = slice(32 * w, 32 * w + 32)
            out[ti, s, 0], out[ti, s, 1] = R[g, 2 * q], R[g, 2 * q + 1]
            out[ti, s, 2], out[ti, s, 3] = R[g + 8, 2 * q], R[g + 8, 2 * q + 1]
    return out


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
    for name in ["lt", "plain", "g0", "duo32", "duo64", "duo2_32", "duo2_64", "add3_32", "mul2_32", *TB]:
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
    for name, rs, nch in [("duo32", 32, 8), ("duo64", 64, 8), ("duo2_32", 32, 16), ("duo2_64", 64, 16), ("add3_32", 32, 8), ("mul2_32", 32, 8)]:
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
                        for ln in range(nch // 8):
                            sl = slice(8 * ln + 4 * (t % 2), 8 * ln + 4 * (t % 2) + 4)
                            u = y[:, :, sl] ^ p ^ KK[ln, t]
                            y[:, :, sl] = (u + KA[ln, t] if name == "add3_32" else u * KA[ln, t]) + z[:, :, sl]
                            if name == "mul2_32":
                                z[:, :, sl] = u * np.uint32(key_mul(ln + 2, t)) + y[:, :, sl]
                            else:
                                z[:, :, sl] = rotl(u, rot_of(nch == 16, ln, t)) + y[:, :, sl]
            fin = y[:, :, :8] ^ rotl(z[:, :, :8], 16)
            if nch == 16:
                fin = fin ^ rotl(y[:, :, 8:] ^ rotl(z[:, :, 8:], 16), 8)
        got = np.fromfile(d / f"R_{name}.bin", dtype=np.uint32).reshape(fin.shape)
        checks[f"R_{name}"] = bool((got == fin).all())
        checks[f"R_{name}_words"] = int(fin.size)
    SB = np.zeros((m, n, 4), dtype=np.int64)
    for Tw in Ts:
        SB += np.stack([(Tw >> np.uint32(8 * j)) & np.uint32(0xFF) for j in range(4)], axis=-1).astype(np.int64)
    for name, kind in TB.items():
        want = tb_receipts(kind, SB, TM, BN, WM, NCT, tiles_m, tiles_n)
        got = np.fromfile(d / f"R_{name}.bin", dtype=np.uint32).reshape(want.shape)
        checks[f"R_{name}"] = bool((got == want).all())
        checks[f"R_{name}_nonzero"] = bool((want[:, :, :4] != 0).mean() > 0.99)
    ok = all(v for v in checks.values() if isinstance(v, bool))
    print(json.dumps({"dir": d.name, "m": m, "n": n, "k": k, "TM": TM, "BN": BN, "grid": meta["grid"], "ok": ok, **checks}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
