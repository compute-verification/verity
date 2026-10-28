"""Aggregation of 8-bit cell words (GD-NOISE's fresh-R two-GEMM commits T = wrap8(A + d_A)·Rᵀ mod 2^8 a cell): f maps a
byte to Z/2^16 and the group's word is sum f(T) mod 2^16. For z uniform in Z/2^8, psi in Z/2^16 \\ 0 and alpha in Z/2^8,
c8(psi, alpha) = E_z e(psi f(z)/2^16 + alpha z/2^8). Reported: rho8 = max |c8| over psi != 0 and every alpha, rho8_0 the same
at alpha = 0 (f can't be a bijection onto 2^16 values, so it is not 0), and the lemma's figures at 8-bit cells:
Pr[W = w | r linear statistics mod 2^8] <= 2^-16 + 2^(4 r) rho8^(g - r), so r_max = (x g - 16)/(4 + x), x = -log2 rho8;
and with nothing known, 2^-16 + rho8_0^g."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_f as F  # noqa: E402

N = F.N
B = 256
BYTES = np.arange(B, dtype=np.uint32)
_TABLE = np.random.default_rng(20261028 + 8).integers(0, N, size=B).astype(np.uint32)

CANDIDATES8 = {
    "lin": (lambda z: (np.asarray(z, dtype=np.uint64) * 0x9E37 & F.M16).astype(np.uint32), "z·0x9E37 mod 2^16: linear"),
    "mix4": (F.mix4, "mix4 on the zero-extended byte"),
    "m32a": (F.m32a, "m32a on the zero-extended byte"),
    "table": (lambda z: _TABLE[np.asarray(z, dtype=np.uint32) & 0xFF], "a random 256-entry table of 16-bit words"),
}


def spectrum8(f) -> dict:
    fz = f(BYTES).astype(np.int64)
    tw = np.exp((2j * np.pi / N) * np.arange(N)).astype(np.complex64)
    rho, rho0, arg = 0.0, 0.0, None
    for p0 in range(1, N, 4096):
        ps = np.arange(p0, min(N, p0 + 4096))
        sp = np.abs(np.fft.fft(tw[np.outer(ps, fz) % N], axis=1)) / B
        k = int(np.argmax(sp))
        if sp.flat[k] > rho:
            rho, arg = float(sp.flat[k]), (int(ps[k // B]), int((-(k % B)) % B))
        rho0 = max(rho0, float(sp[:, 0].max()))
    x = -np.log2(rho)
    return {"distinct_values": int(len(np.unique(fz))), "rho8": rho, "arg": arg, "rho8_0": rho0,
            "r_max": {g: float(max(0.0, (x * g - 16) / (4 + x))) for g in (8, 16, 64, 128, 256)},
            "log2_guess_excess": {g: float(g * np.log2(rho0)) for g in (2, 4, 8, 16)}}


def main(out: str) -> None:
    res = {}
    for name, (f, what) in CANDIDATES8.items():
        s = spectrum8(f)
        s["what"] = what
        res[name] = s
        print(name, json.dumps(s), flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "spec8.json")
