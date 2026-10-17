"""CPU reference for egemm.cu's gate dump: the noise, every edited operand, product and receipt, bit for bit.

python3 eref.py DIR [--no-receipts] prints one JSON line of checks and exits 0 only if every check passes.

Recomputed independently of the device, per format (int8, E4M3's integer grid), from the dumped A and B (stacked over the
batch) and the seed alone: the noise keys, the noise E (each 16-byte chunk of a stacked row from
mix32(key ^ mix32(row k/16 + chunk)), element j's index into S = {5, 7, -5, -7} the 2 bits at 4 (j % 8) + 2 (j / 8)),
A' = A + E_A and B' = B + E_B within the format's exact range, each batch's A B^T (the unedited arms) and A' B'^T (the
edited ones: separate pass, fused, fused and overlapped), and the receipts (duo at 32 and 64, TB4C at 32) over the running sums of all of K.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "floored_gemm"))
from mref import decode, duo_receipts, words  # noqa: E402
from ref import mix32, tbc_receipts  # noqa: E402

FMTS = ["i8", "e4"]
S = np.array([5, 7, -5, -7], dtype=np.int64)
RCS = [("cduo32", "duo", 32), ("cduo64", "duo", 64), ("ct4c32", 8, 32)]


def keys(seed, inst=0):
    return (int(mix32(np.uint32(seed) ^ mix32(np.uint32(2 * inst + 0x0A0A0A0A)))),
            int(mix32(np.uint32(seed) ^ mix32(np.uint32(2 * inst + 1 + 0x0A0A0A0A)))))


def noise(rows, k, key):
    chunk = (np.arange(rows, dtype=np.uint64)[:, None] * (k // 16) + np.arange(k // 16, dtype=np.uint64)[None, :])
    h = mix32(np.uint32(key) ^ mix32((chunk & 0xFFFFFFFF).astype(np.uint32)))
    j = np.arange(16, dtype=np.uint32)
    idx = (h[:, :, None] >> (4 * (j % 8) + 2 * (j // 8))) & np.uint32(3)
    return S[idx.reshape(rows, k)]


def main(d: Path, receipts: bool) -> int:
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, batch, seed = meta["m"], meta["n"], meta["k"], meta["batch"], meta["seed"]
    TM, BN, WM, NCT = meta["TM"], meta["BN"], meta["WM"], meta["NCT"]
    tiles_m, tiles_n = batch * m // TM, n // BN   # the stacked tile index: (b tiles_m + tm) tiles_n + tn
    arms, skipped = set(meta["arms"]), set(meta.get("skipped", []))
    ka, kb = keys(seed)
    checks = {"noise_keys": (meta["nzA"], meta["nzB"]) == (ka, kb)}
    seen = set()

    def ld(name, dt, shape):
        return np.fromfile(d / name, dtype=dt).reshape(shape)

    def product(X, Y):
        """Each batch's X Y^T, stacked: (batch m) x n."""
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

    def check_receipts(prefix, fmt, X, Y):
        T = np.zeros((batch * m, n), dtype=np.int64)
        T32, T64 = [], []
        for s in range(k // 32):
            T = T + product(X[:, 32 * s : 32 * s + 32], Y[:, 32 * s : 32 * s + 32])
            w = words(fmt, T)
            T32.append(w)
            if s % 2 == 1:
                T64.append(w)
        for rc, kind, L in RCS:
            name = f"{prefix}{rc}_{fmt}"
            if name in skipped or not receipts:
                continue
            Ts = T32 if L == 32 else T64
            if kind == "duo":
                want = duo_receipts(seed, Ts, 0, TM, BN, WM, NCT, tiles_m, tiles_n)
            else:
                want = tbc_receipts(kind, Ts, TM, BN, WM, NCT, tiles_m, tiles_n)
            got = ld(f"R_{name}.bin", np.uint32, (tiles_m * tiles_n, NCT, 8))
            checks[f"R_{name}"] = bool((got == want).all())
            if kind != "duo":
                checks[f"R_{name}_nonzero"] = bool((want[:, :, :4] != 0).mean() > 0.99)

    EA, EB = noise(batch * m, k, ka), noise(batch * n, k, kb)
    checks["noise_uniform_on_S"] = bool(all(abs((EA == s).mean() - 0.25) < 0.02 for s in S))
    for fmt in FMTS:
        A = decode(fmt, ld(f"A_{fmt}.bin", np.uint8, (batch * m, k)))
        B = decode(fmt, ld(f"B_{fmt}.bin", np.uint8, (batch * n, k)))
        if A is None or B is None:
            checks[f"AB_{fmt}_decode"] = False
            continue
        lo, lim = (120, 127) if fmt == "i8" else (9, 16)
        Ae, Be = A + EA, B + EB
        checks[f"AB_{fmt}_range"] = bool(max(np.abs(A).max(), np.abs(B).max()) <= lo)
        checks[f"AeBe_{fmt}_range"] = bool(max(np.abs(Ae).max(), np.abs(Be).max()) <= lim)
        AB, AeBe = product(A, B), product(Ae, Be)
        checks[f"AeBe_{fmt}_exact_range"] = bool(np.abs(AeBe).max() < (2**31 if fmt == "i8" else 2**24))
        check_product(f"plain_{fmt}", fmt, AB)
        check_product(f"sep_lt_{fmt}", fmt, AeBe)
        seen.add(f"sepform_{fmt}")
        if f"sepform_{fmt}" not in skipped:
            gA = decode(fmt, ld(f"XA_sepform_{fmt}.bin", np.uint8, (batch * m, k)))
            gB = decode(fmt, ld(f"XB_sepform_{fmt}.bin", np.uint8, (batch * n, k)))
            checks[f"XA_sepform_{fmt}"] = gA is not None and bool((gA == Ae).all())
            checks[f"XB_sepform_{fmt}"] = gB is not None and bool((gB == Be).all())
        for rc in ["cg", "cduo32", "cduo64", "ct4c32"]:
            check_product(f"{rc}_{fmt}", fmt, AB)
            check_product(f"sep_{rc}_{fmt}", fmt, AeBe)
            check_product(f"fus_{rc}_{fmt}", fmt, AeBe)
            check_product(f"fov_{rc}_{fmt}", fmt, AeBe)
        check_receipts("", fmt, A, B)
        for prefix in ["sep_", "fus_", "fov_"]:
            check_receipts(prefix, fmt, Ae, Be)

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
