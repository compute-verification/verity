"""CPU reference for lpgemm.cu's gate dump: U and F from hashlib, every operand, digit, product and receipt bit for bit.

python3 lpref.py DIR [--no-receipts] prints one JSON line of checks and exits 0 only if every check passes.

From the dumped A and B (stacked over the batch) and the seed alone: sigma (sref.py's), U (m x 32, shared by the batch)
from SHA-512 blocks with tag 6 (block j = rows 4j..4j+3, entry e = (row % 4) 32 + l digest nibble e, the low nibble
first, u = (v & 7) + 1 negated if v & 8), F (sref.py's pm1 noise with B's tag, fresh for every problem), A' = A + U W^T,
B' = B + F, S = W^T B'^T, P_A = [U U U U] and P_B = [S_h1 | S_h2 | S_l | 0] (the tail's peel), the extension's columns
(A's -U and -15 U; B's b0 and Sh's mx parts, from the definition (Sh + c - 1) // mx), the served product A B'^T (with
A' B'^T - U S^T and the extended product checked equal to it), the duo receipts of A' B'^T at L = 32, and the
generation-only kernel's folded words of F. A component arm is checked through the route that runs it (meta's cover).
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "floored_gemm"))
from mref import duo_receipts, words  # noqa: E402
from sref import fold, noise, sigma  # noqa: E402

U_TAG = 6


def u_noise(key, m):
    U = np.zeros((m, 32), np.int64)
    for j in range(m // 4):
        b = np.frombuffer(hashlib.sha512(key + struct.pack(">Q", U_TAG << 56 | j)).digest(), np.uint8).astype(np.int64)
        v = np.stack([b & 15, b >> 4], axis=1).reshape(128)
        u = (v & 7) + 1
        U[4 * j : 4 * j + 4] = np.where(v & 8, -u, u).reshape(4, 32)
    return U


def main(d: Path, receipts: bool) -> int:
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, batch, seed, Kp, mx = (meta[x] for x in ("m", "n", "k", "batch", "seed", "Kp", "mx"))
    TM, BN, WM, NCT = meta["TM"], meta["BN"], meta["WM"], meta["NCT"]
    tiles_m, tiles_n = batch * m // TM, n // BN
    arms, skipped, cover = set(meta["arms"]), set(meta.get("skipped", [])), meta["cover"]
    sg = sigma(seed)
    key = b"".join(struct.pack(">Q", w) for w in sg)
    checks = {"sigma": [f"{w:016x}" for w in sg] == meta["sigma"], "Kp": Kp == k + 32 * (mx + 1)}
    seen = set()

    def ld(name, dt, shape):
        return np.fromfile(d / name, dtype=dt).reshape(shape)

    def product(X, Y):
        return np.matmul(X.reshape(batch, m, -1), Y.reshape(batch, n, -1).transpose(0, 2, 1)).reshape(batch * m, n)

    A = ld("A.bin", np.int8, (batch * m, k)).astype(np.int64)
    B = ld("B.bin", np.int8, (batch * n, k)).astype(np.int64)
    checks["inputs_range"] = bool(max(np.abs(A).max(), np.abs(B).max()) <= 119)
    checks["inputs_saturated_rows"] = bool((A.reshape(batch, m, k)[:, 0] == 119).all() and (B.reshape(batch, n, k)[:, 1] == -119).all())

    U = u_noise(key, m)
    F, pt = noise(batch * n, k, key, 1, 1)
    vals = np.array([v for v in range(-8, 9) if v])
    checks["U_in_S"] = bool(np.isin(U, vals).all())
    checks["U_uniform"] = bool(max(abs((U == v).mean() - 1 / 16) for v in vals) < 0.25 / 16 + 3 / np.sqrt(U.size))
    checks["F_pm1"] = bool(np.isin(F, [-1, 1]).all() and abs(F.mean()) < 0.05)
    Ub = np.tile(U, (batch, 1))
    Ap = A + np.tile(Ub, (1, k // 32))
    Bp = B + F
    S = Bp.reshape(batch * n, k // 32, 32).sum(axis=1)
    checks["Ap_Bp_range"] = bool(120 <= np.abs(Ap).max() <= 127 and np.abs(Bp).max() == 120)
    checks["S_range"] = bool(np.abs(S).max() <= 120 * (k // 32))
    PA = np.tile(U, (1, 4))
    lo = (S + 128) % 256 - 128
    hi = (S - lo) // 256
    h1 = hi // 2
    PB = np.concatenate([h1, hi - h1, lo, np.zeros_like(S)], axis=1)
    checks["PB_digits"] = bool(np.abs(PB).max() <= 128 and PB.max() <= 127 and (256 * (PB[:, :32] + PB[:, 32:64]) + PB[:, 64:96] == S).all())
    Sh = (S + 7) // 15
    b0 = S - 15 * Sh
    bc = [(Sh + c - 1) // mx for c in range(1, mx + 1)]
    checks["ext_digits"] = bool(np.abs(b0).max() <= 7 and max(np.abs(b).max() for b in bc) <= 127 and (b0 + 15 * sum(bc) == S).all())
    XAx = np.concatenate([Ap, -Ub] + [-15 * Ub] * mx, axis=1)
    XBx = np.concatenate([Bp, b0] + bc, axis=1)
    want = product(A, Bp)
    checks["AB_exact_range"] = bool(np.abs(product(Ap, Bp)).max() < 2**31)
    checks["peel_identity"] = bool((product(Ap, Bp) - product(Ub, S) == want).all())
    checks["ext_identity"] = bool((product(XAx, XBx) == want).all())

    def check_r(name, X, Y):
        if not receipts:
            return
        T = np.zeros((batch * m, n), dtype=np.int64)
        Ts = []
        for s in range(k // 32):
            T = T + product(X[:, 32 * s : 32 * s + 32], Y[:, 32 * s : 32 * s + 32])
            Ts.append(words("i8", T))
        checks[f"R_{name}"] = bool((ld(f"R_{name}.bin", np.uint32, (tiles_m * tiles_n, NCT, 8)) ==
                                    duo_receipts(seed, Ts, 0, TM, BN, WM, NCT, tiles_m, tiles_n)).all())

    def got(pre, name, dt, shape):
        return ld(f"{pre}_{name}.bin", dt, shape).astype(np.int64)

    for name in ("plain", "cplain", "cduo"):
        seen.add(name)
        if name in arms:
            checks[f"U_{name}"] = bool((got("U", name, np.int32, (batch * m, n)) == product(A, B)).all())
    if "cduo" in arms:
        check_r("cduo", A, B)
    for name in ("genU", "genF"):
        seen.add(name)
    if "genU" in arms:
        checks["PA_genU"] = bool((got("PA", "genU", np.int8, (m, 128)) == PA).all())
    if "genF" in arms:
        checks["G_genF"] = bool((ld("G_genF.bin", np.uint32, (len(pt),)) == fold(pt)).all())
    for name in sorted(arms):
        route = name.split("_")[0]
        if route not in ("lt", "t", "td", "f", "fd", "o", "od"):
            continue
        seen.add(name)
        checks[f"U_{name}"] = bool((got("U", name, np.int32, (batch * m, n)) == want).all())
        checks[f"PA_{name}"] = bool((got("PA", name, np.int8, (m, 128)) == PA).all())
        XB = got("XB", name, np.int8, (batch * n, Kp))
        checks[f"XB_{name}"] = bool((XB == XBx).all()) if route == "lt" else bool((XB[:, :k] == Bp).all())
        if route != "lt":
            checks[f"PB_{name}"] = bool((got("PB", name, np.int8, (batch * n, 128)) == PB).all())
        if route in ("lt", "t", "td"):
            XA = got("XA", name, np.int8, (batch * m, Kp))
            checks[f"XA_{name}"] = bool((XA == XAx).all()) if route == "lt" else bool((XA[:, :k] == Ap).all())
        if route in ("td", "fd", "od"):
            check_r(name, Ap, Bp)
    for comp, by in cover.items():
        seen.add(comp)
        checks[f"cover_{comp}"] = by in arms and all(v for key_, v in checks.items() if key_.endswith(f"_{by}"))

    missing = sorted(arms - seen)
    checks["every_arm_checked"] = not missing
    ok = all(v for v in checks.values() if isinstance(v, bool))
    failed = [key_ for key_, v in checks.items() if v is False]
    print(json.dumps({"dir": d.name, "m": m, "n": n, "k": k, "batch": batch, "Kp": Kp, "mx": mx, "ok": ok, "receipts": receipts,
                      "failed": failed, "skipped": sorted(skipped), "unchecked_arms": missing, "n_checks": len(checks), **checks}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), "--no-receipts" not in sys.argv[2:]))
