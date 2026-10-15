"""CPU reference for sgemm.cu's gate dump: the SHA-512 noise, every edited operand, product, receipt and generation word,
bit for bit.

python3 sref.py DIR [--no-receipts] prints one JSON line of checks and exits 0 only if every check passes.

Recomputed independently of the device, from the dumped A and B (stacked over the batch) and the seed alone: sigma
(splitmix64 from the seed), the noise of every patch (4 rows x 128 columns) with hashlib's SHA-512 in counter mode, as
sha_noise.cuh states it, for each alphabet (pm1: 1 - 2 bit; u17: base-17 digits of the accepted 64-bit words, minus 8;
u17w: of the accepted 32-bit words),
A' = A + E_A and B' = B + E_B within the format's exact range, each batch's A B^T and A' B'^T, the duo receipts at L = 32,
and the generation-only kernel's folded words.
"""

import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "floored_gemm"))
from mref import decode, duo_receipts, words  # noqa: E402

FMTS = ["i8", "e4"]
ALS = {"pm1": 1, "u17": 2, "u17w": 3}
M64 = (1 << 64) - 1
P17_15 = 17**15
U17_LIM = 6 * P17_15
U17W_LIM = 10 * 17**7


def sigma(seed, inst=0):
    x, out = (seed << 32 | inst) & M64, []
    for _ in range(4):
        x = (x + 0x9E3779B97F4A7C15) & M64
        z = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        out.append(z ^ (z >> 31))
    return out


def patch_noise(key, ns, side, p):
    """Patch p's 512 noise values, entry e = (row % 4) 128 + col % 128."""
    tag = 2 * (ns - 1) + side
    if ns == 1:
        d = hashlib.sha512(key + struct.pack(">Q", tag << 56 | p)).digest()
        bits = np.unpackbits(np.frombuffer(d, np.uint8), bitorder="little").astype(np.int64)
        return 1 - 2 * bits
    digits, r = [], 0
    while len(digits) < 512:
        assert r < 8, f"patch {p} exhausted 8 blocks"
        d = hashlib.sha512(key + struct.pack(">Q", tag << 56 | (8 * p + r))).digest()
        r += 1
        wb = 8 if ns == 2 else 4
        for i in range(64 // wb):
            w = int.from_bytes(d[wb * i : wb * i + wb], "big")
            if w >= (U17_LIM if ns == 2 else U17W_LIM):
                continue
            v = w % (P17_15 if ns == 2 else 17**7)
            for _ in range(15 if ns == 2 else 7):
                digits.append(v % 17)
                v //= 17
            if len(digits) >= 512:
                break
    return np.array(digits[:512], dtype=np.int64) - 8


def noise(rows, k, key, ns, side):
    """The patches of a rows x k operand, laid out: E (rows x k) and the patches (rows / 4 * k / 128, 512)."""
    kb = k // 128
    pt = np.stack([patch_noise(key, ns, side, p) for p in range(rows // 4 * kb)])
    return pt.reshape(rows // 4, kb, 4, 128).transpose(0, 2, 1, 3).reshape(rows, k), pt


def fold(pt):
    """sha_gen's word per patch: xor over chunks c of rotl(n0 ^ n1 ^ n2 ^ n3, c)."""
    w = pt.astype(np.int8).view(np.uint8).reshape(len(pt), 32, 16).copy().view("<u4").astype(np.uint64)   # (patches, 32, 4)
    x = (w[:, :, 0] ^ w[:, :, 1] ^ w[:, :, 2] ^ w[:, :, 3])
    c = np.arange(32, dtype=np.uint64)[None, :]
    rot = ((x << c) | (x >> ((32 - c) % 32))) & 0xFFFFFFFF
    out = np.zeros(len(pt), dtype=np.uint64)
    for j in range(32):
        out ^= rot[:, j]
    return out.astype(np.uint32)


def main(d: Path, receipts: bool) -> int:
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, batch, seed = meta["m"], meta["n"], meta["k"], meta["batch"], meta["seed"]
    TM, BN, WM, NCT = meta["TM"], meta["BN"], meta["WM"], meta["NCT"]
    tiles_m, tiles_n = batch * m // TM, n // BN
    arms, skipped = set(meta["arms"]), set(meta.get("skipped", []))
    sg = sigma(seed)
    checks = {"sigma": [f"{w:016x}" for w in sg] == meta["sigma"]}
    key = b"".join(struct.pack(">Q", w) for w in sg)
    seen = set()

    def ld(name, dt, shape):
        return np.fromfile(d / name, dtype=dt).reshape(shape)

    def product(X, Y):
        return np.matmul(X.reshape(batch, m, -1), Y.reshape(batch, n, -1).transpose(0, 2, 1)).reshape(batch * m, n)

    def check_product(name, fmt, want):
        seen.add(name)
        if name in skipped:
            return
        if fmt == "i8":
            got = ld(f"U_{name}.bin", np.int32, (batch * m, n)).astype(np.int64)
        else:
            got = ld(f"U_{name}.bin", np.float32, (batch * m, n)).astype(np.float64)
        checks[f"U_{name}"] = bool((got == want).all())

    def check_duo(name, fmt, X, Y):
        if name in skipped or not receipts:
            return
        T = np.zeros((batch * m, n), dtype=np.int64)
        Ts = []
        for s in range(k // 32):
            T = T + product(X[:, 32 * s : 32 * s + 32], Y[:, 32 * s : 32 * s + 32])
            Ts.append(words(fmt, T))
        want = duo_receipts(seed, Ts, 0, TM, BN, WM, NCT, tiles_m, tiles_n)
        checks[f"R_{name}"] = bool((ld(f"R_{name}.bin", np.uint32, (tiles_m * tiles_n, NCT, 8)) == want).all())

    ops = {f: (decode(f, ld(f"A_{f}.bin", np.uint8, (batch * m, k))), decode(f, ld(f"B_{f}.bin", np.uint8, (batch * n, k))))
           for f in FMTS}
    for f, (A, B) in ops.items():
        checks[f"AB_{f}_range"] = A is not None and B is not None and bool(max(np.abs(A).max(), np.abs(B).max()) <= (119 if f == "i8" else 8))
        check_product(f"plain_{f}", f, product(A, B))
    for al, ns in ALS.items():
        EA, pa = noise(batch * m, k, key, ns, 0)
        EB, pb = noise(batch * n, k, key, ns, 1)
        vals = np.array([-1, 1]) if ns == 1 else np.arange(-8, 9)
        allE = np.concatenate([EA.ravel(), EB.ravel()])
        checks[f"noise_{al}_in_S"] = bool(np.isin(allE, vals).all())
        checks[f"noise_{al}_uniform"] = bool(max(abs((allE == v).mean() - 1 / len(vals)) for v in vals) < 0.25 / len(vals))
        name = f"gen_{al}"
        seen.add(name)
        if name not in skipped:
            checks[f"G_{name}"] = bool((ld(f"G_{name}.bin", np.uint32, (len(pa) + len(pb),)) == np.concatenate([fold(pa), fold(pb)])).all())
        for f, (A, B) in ops.items():
            Ae, Be = A + EA, B + EB
            lim = 127 if f == "i8" else 16
            checks[f"AeBe_{al}_{f}_range"] = bool(max(np.abs(Ae).max(), np.abs(Be).max()) <= lim)
            AeBe = product(Ae, Be)
            checks[f"AeBe_{al}_{f}_exact_range"] = bool(np.abs(AeBe).max() < (2**31 if f == "i8" else 2**24))
            name = f"sform_{al}_{f}"
            seen.add(name)
            if name not in skipped:
                gA = decode(f, ld(f"XA_{name}.bin", np.uint8, (batch * m, k)))
                gB = decode(f, ld(f"XB_{name}.bin", np.uint8, (batch * n, k)))
                checks[f"XA_{name}"] = gA is not None and bool((gA == Ae).all())
                checks[f"XB_{name}"] = gB is not None and bool((gB == Be).all())
            arms_f = [f"slt_{al}_{f}", f"sduo_{al}_{f}"]
            if f == "i8" and al != "u17w":
                arms_f += [f"fus_{al}_i8", f"fusd_{al}_i8", f"fov_{al}_i8", f"fovd_{al}_i8"]
            for a in arms_f:
                check_product(a, f, AeBe)
            for a in arms_f:
                if a.split("_")[0] in ("sduo", "fusd", "fovd"):
                    check_duo(a, f, Ae, Be)

    missing = sorted(arms - seen)
    checks["every_arm_checked"] = not missing
    ok = all(v for v in checks.values() if isinstance(v, bool))
    failed = [key for key, v in checks.items() if v is False]
    print(json.dumps({"dir": d.name, "m": m, "n": n, "k": k, "batch": batch, "ok": ok, "receipts": receipts,
                      "failed": failed, "skipped": sorted(skipped), "unchecked_arms": missing, "n_checks": len(checks),
                      **checks}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), "--no-receipts" not in sys.argv[2:]))
