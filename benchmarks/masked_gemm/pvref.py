"""CPU reference for pvgemm.cu's gate dump: every noise, edited operand, product and receipt bit for bit against hashlib.

python3 pvref.py DIR prints one JSON line of checks and exits 0 only if every check passes.

From the dumped P, V^T, Q, K and the seed alone (sref.py's sigma and patch noise, for pm1, u17, u17w, u17p, u17wp): the shared E
(s x s, A's tag) against genP's buffer; V's fresh noise (H 128 x s, B's tag) against editV's V' and genV's folded words;
P + E broadcast over the heads (sepS's P'); every P V product, edited or not, per head, and the duo-32 receipts; S's row
maxima (prodmax), exact; the producer's P~ (prodN) in [0, 119] and within 1 of a float64 softmax step; and the producer's
noised outputs exactly P~ + E (prodS, shared) and P~ + the unshared noise (prodU, A's tag over the stacked H s x s), with
the routes' products (P~ + noise) V'^T.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "floored_gemm"))
from mref import duo_receipts, words  # noqa: E402
from sref import fold, noise, sigma  # noqa: E402

ALS = {"pm1": 1, "u17": 2, "u17w": 3, "u17p": 2, "u17wp": 3}   # u17p, u17wp: the noise of u17, u17w, five lanes a patch
HD = 128


def main(d: Path) -> int:
    meta = json.loads((d / "meta.json").read_text())
    s, H, seed, c = meta["s"], meta["H"], meta["seed"], meta["c"]
    TM, BN, WM, NCT = meta["TM"], meta["BN"], meta["WM"], meta["NCT"]
    tiles_m, tiles_n = H * s // TM, HD // BN
    arms, skipped = set(meta["arms"]), set(meta.get("skipped", []))
    sg = sigma(seed)
    key = b"".join(struct.pack(">Q", w) for w in sg)
    checks = {"sigma": [f"{w:016x}" for w in sg] == meta["sigma"]}
    seen = set()

    def ld(name, dt, shape):
        return np.fromfile(d / name, dtype=dt).reshape(shape)

    P = ld("P.bin", np.int8, (H * s, s)).astype(np.int64)
    Vt = ld("Vt.bin", np.int8, (H * HD, s)).astype(np.int64)
    Q = ld("Q.bin", np.int8, (H * s, HD)).astype(np.int64)
    K = ld("K.bin", np.int8, (H * s, HD)).astype(np.int64)
    checks["inputs_range"] = bool(P.min() >= 0 and P.max() <= 119 and np.abs(Vt).max() <= 119 and np.abs(Q).max() <= 119
                                  and np.abs(K).max() <= 119)

    def product(X, Y):
        return np.matmul(X.reshape(H, s, s), Y.reshape(H, HD, s).transpose(0, 2, 1)).reshape(H * s, HD)

    def check_u(name, X, Y, duo=False):
        if name not in arms:
            return
        seen.add(name)
        if name in skipped:
            return
        checks[f"U_{name}"] = bool((ld(f"U_{name}.bin", np.int32, (H * s, HD)).astype(np.int64) == product(X, Y)).all())
        if duo:
            T = np.zeros((H * s, HD), dtype=np.int64)
            Ts = []
            X3, Y3 = X.reshape(H, s, s), Y.reshape(H, HD, s)
            for k in range(s // 32):
                T = T + np.matmul(X3[:, :, 32 * k : 32 * k + 32], Y3[:, :, 32 * k : 32 * k + 32].transpose(0, 2, 1)).reshape(H * s, HD)
                Ts.append(words("i8", T))
            want = duo_receipts(seed, Ts, 0, TM, BN, WM, NCT, tiles_m, tiles_n)
            checks[f"R_{name}"] = bool((ld(f"R_{name}.bin", np.uint32, (tiles_m * tiles_n, NCT, 8)) == want).all())

    S = np.matmul(Q.reshape(H, s, HD), K.reshape(H, s, HD).transpose(0, 2, 1)).reshape(H * s, s)
    mx = S.max(axis=1)
    checks["mx_setup"] = bool((ld("mx.bin", np.int32, (H * s,)) == mx).all())
    seen.add("prodmax")
    if "prodmax" not in skipped:
        checks["mx_prodmax"] = bool((ld("mx_prodmax.bin", np.int32, (H * s,)) == mx).all())
    seen.add("prodN")
    Pt = ld("Pt.bin", np.int8, (H * s, s)).astype(np.int64)
    want_pt = np.minimum(119, np.rint(119.0 * np.exp2((S - mx[:, None]) * c)))
    checks["Pt_range"] = bool(Pt.min() >= 0 and Pt.max() <= 119)
    checks["Pt_softmax_within_1"] = bool(np.abs(Pt - want_pt).max() <= 1)
    checks["Pt_softmax_nonzero"] = bool((Pt > 0).mean() > 0.05)
    check_u("plain", P, Vt)
    check_u("cduo", P, Vt, duo=True)

    by_ns = {}
    for al, ns in ALS.items():
        if ns not in by_ns:
            by_ns[ns] = noise(s, s, key, ns, 0), noise(H * HD, s, key, ns, 1), noise(H * s, s, key, ns, 0)
        (E, pe), (EV, pv), (EU, _) = by_ns[ns]
        vals = np.array([-1, 1]) if ns == 1 else np.arange(-8, 9)
        checks[f"noise_{al}_in_S"] = bool(np.isin(np.concatenate([E.ravel(), EV.ravel(), EU.ravel()]), vals).all())
        Eb = np.tile(E, (H, 1))
        Vp = Vt + EV
        for name, want, dt, shape in [(f"genP_{al}", E, np.int8, (s, s)), (f"editV_{al}", Vp, np.int8, (H * HD, s))]:
            seen.add(name)
            if name not in skipped:
                pre = "E" if name.startswith("genP") else "XB"
                checks[f"{pre}_{name}"] = bool((ld(f"{pre}_{name}.bin", dt, shape).astype(np.int64) == want).all())
        name = f"genV_{al}"
        seen.add(name)
        if name not in skipped:
            checks[f"G_{name}"] = bool((ld(f"G_{name}.bin", np.uint32, (len(pv),)) == fold(pv)).all())
        Ps = P + Eb
        checks[f"Ps_{al}_range"] = bool(Ps.max() <= 127 and Vp.min() >= -127 and Vp.max() <= 127)
        name = f"sepS_{al}"
        if name not in skipped:
            checks[f"XA_{name}"] = bool((ld(f"XA_{name}.bin", np.int8, (H * s, s)).astype(np.int64) == Ps).all())
        check_u(name, Ps, Vp)
        check_u(f"sepSd_{al}", Ps, Vp, duo=True)
        check_u(f"fusS_{al}", Ps, Vp)
        check_u(f"fovS_{al}", Ps, Vp)
        check_u(f"fusSd_{al}", Ps, Vp, duo=True)
        for tag, En in [("S", Eb), ("U", EU)]:
            name = f"prod{tag}_{al}"
            if name not in arms:
                continue
            seen.add(name)
            if name not in skipped:
                got = ld(f"XA_{name}.bin", np.int8, (H * s, s)).astype(np.int64)
                checks[f"XA_{name}"] = bool((got - Pt == En).all())
            check_u(f"prod{tag}lt_{al}", Pt + En, Vp)
            check_u(f"prod{tag}d_{al}", Pt + En, Vp, duo=True)

    missing = sorted(arms - seen)
    checks["every_arm_checked"] = not missing
    ok = all(v for v in checks.values() if isinstance(v, bool))
    failed = [k for k, v in checks.items() if v is False]
    print(json.dumps({"dir": d.name, "s": s, "H": H, "ok": ok, "failed": failed, "skipped": sorted(skipped),
                      "unchecked_arms": missing, "n_checks": len(checks), **checks}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
