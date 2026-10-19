"""fattn.cu's gate judged on the CPU: for each dumped query column, the softmax recomputed from the device's X (the keys,
hquant's) and W^ (the served query row, phat's) alone.

S = X W^ in int64 must equal h_l7qs's stored column (the unfused QK^T); m = max S over keys 0 .. pos must equal hm_l7qs's
column max (the fused pass 1); e = LUT[min((m - S) A >> 32, 255)] over keys 0 .. pos with LUT[t] = rint(127 2^(-t/16)) (the
dumped LUT must equal it), P_prob = e at keys 1 .. pos and 0 at key 0 and past pos (on the bytes apv reads, keys below
128 (pos // 128 + 1)), e(0) = E0, s_q = float32(1) / float32(L) and z_q = float32(e(0)) / float32(L) bitwise, L = sum e.
One JSON line a gate directory; exit 1 on any difference, on a device mismatch or on a control the device didn't reject.
    uv run --no-project --with numpy python3 benchmarks/pouw/edited_gemm/fattn_ref.py GATE_DIR...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def check(d: Path) -> dict:
    meta = json.loads((d / "meta.json").read_text())
    s, rep, A = meta["s"], meta["rep"], meta["A"]
    lut = np.rint(127.0 * np.exp2(-np.arange(256) / 16.0)).astype(np.int64)
    res = {"gate": str(d), "s": s, "H": meta["H"], "rep": rep, "lut": bool((np.array(meta["lut"]) == lut).all()),
           "device_ok": meta["device_ok"], "control_rejected": meta["control_rejected"], "cols": []}
    ok = res["lut"] and meta["device_ok"] and meta["control_rejected"]
    xs: dict[int, np.ndarray] = {}
    nz = []
    for k, col, sp, sz, m_dev, e0_dev in meta["cols"]:
        if k not in xs:
            xs[k] = np.fromfile(d / f"X_{k}.bin", np.int8).astype(np.int64).reshape(s, 128)
        sfx = f"_{k}_{col}.bin"
        wh = np.fromfile(d / f"Wh{sfx}", np.int8).astype(np.int64)
        S = xs[k] @ wh
        pos = col % s
        m = int(S[: pos + 1].max())
        dd = (m - S[: pos + 1]).astype(np.uint64)
        t = np.minimum((dd * np.uint64(A)) >> np.uint64(32), 255).astype(np.int64)
        e = lut[t]
        L = int(e.sum())
        nk = (pos | 127) + 1
        P = np.zeros(nk, np.int64)
        P[1: pos + 1] = e[1:]
        Pd = np.fromfile(d / f"P{sfx}", np.int8).astype(np.int64)[:nk]
        Sd = np.fromfile(d / f"S{sfx}", np.int32).astype(np.int64)
        f32 = np.float32
        spr, szr = f32(1) / f32(L), f32(e[0]) / f32(L)
        r = {"kvh": k, "col": col, "pos": pos, "S": bool((Sd == S).all()), "m": m == m_dev, "P": bool((Pd == P).all()),
             "e0": int(e[0]) == e0_dev, "sp": f32(sp).tobytes() == spr.tobytes(), "sz": f32(sz).tobytes() == szr.tobytes(),
             "L": L, "nonzero": int((e > 0).sum())}
        nz.append(r["nonzero"] / (pos + 1))
        ok &= all(r[x] for x in ("S", "m", "P", "e0", "sp", "sz"))
        res["cols"].append(r)
    res["ok"] = bool(ok)
    res["nonzero_frac_mean"] = float(np.mean(nz)) if nz else None
    return res


def main() -> int:
    ok = True
    for a in sys.argv[1:]:
        r = check(Path(a))
        ok &= r["ok"]
        print(json.dumps(r))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
