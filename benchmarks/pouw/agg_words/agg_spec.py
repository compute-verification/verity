"""The additive spectrum of each candidate f over Z/2^16, which bounds what cheap statistics of a group's cells say about
its aggregate A = sum_j f(z_j) mod 2^16.

For psi != 0 and alpha in Z/N, c(psi, alpha) = E_z e((psi f(z) + alpha z)/N), z uniform. Reported:
- rho_lin = max over psi != 0 and every alpha of |c|; rho_lin0 the same at alpha = 0 only (0 for a bijection);
- rho_res[c] = max over psi != 0, alpha and r of |E[e((psi f(z) + alpha z)/N) | z = r mod 2^c]|, c = 1..3: the bias that
  knowing each cell's low c bits (computable from operands mod 2^c) leaves; rho_res0[c] the same at alpha = 0.
Lemma (cells i.i.d. uniform, f a bijection): an attacker who knows r linear statistics s = Mz of the g cells mod N (M of
rank r) predicts A with probability at most 1/N + N^(r/2) rho_lin^(g-r) (expand 1{Mz = s} in characters; on an information
set of r columns of M bound the coefficients' product by Cauchy-Schwarz and Parseval, sum_alpha |c(psi, alpha)|^2 = 1,
and the other g - r by rho_lin), and one who knows every cell's residue mod 2^c, at most 1/N + rho_res0[c]^g. So the
statistics a group can leak while the word stays within 2/N of uniform number r_max = (x g - 16)/(8 + x), x = -log2 rho_lin."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_f as F  # noqa: E402

N = F.N
BATCH = 128


def spectrum(f, cs=(1, 2, 3)) -> dict:
    fz = f(F.Z).astype(np.int64)
    bij = len(np.unique(fz)) == N
    tw = np.exp((2j * np.pi / N) * np.arange(N)).astype(np.complex64)
    out = {"bijective": bool(bij), "rho_lin": 0.0, "arg_lin": None, "rho_lin0": 0.0,
           "rho_res": {c: 0.0 for c in cs}, "rho_res0": {c: 0.0 for c in cs}}
    psis = np.arange(1, N // 2 + 1)
    for b0 in range(0, len(psis), BATCH):
        ps = psis[b0:b0 + BATCH]
        u = tw[np.outer(ps, fz) % N]
        spec = np.abs(sfft.fft(u, axis=1, workers=4)) / N
        k = int(np.argmax(spec))
        if spec.flat[k] > out["rho_lin"]:
            out["rho_lin"] = float(spec.flat[k])
            out["arg_lin"] = (int(ps[k // N]), int((-(k % N)) % N))
        out["rho_lin0"] = max(out["rho_lin0"], float(spec[:, 0].max()))
        for c in cs:
            m = 1 << c
            sub = u.reshape(len(ps), N // m, m).transpose(0, 2, 1)
            sp = np.abs(sfft.fft(sub, axis=2, workers=4)) / (N // m)
            out["rho_res"][c] = max(out["rho_res"][c], float(sp.max()))
            out["rho_res0"][c] = max(out["rho_res0"][c], float(sp[:, :, 0].max()))
    return out


def bounds(s: dict) -> dict:
    """The lemma's two bounds, as log2 of (bound − 1/N) where that is below 1, at a few g, and r* (the most linear
    statistics with N^r rho_lin^g <= 1/N) as a share of g."""
    x = -np.log2(max(s["rho_lin"], 1e-300))
    out = {"log2_rho_lin": float(x and -x), "r_star_share": float(x / 16) if s["bijective"] else 0.0,
           "r_max": {g: float(max(0.0, (x * g - 16) / (8 + x))) for g in (8, 16, 64, 128, 256)}, "res": {}}
    for c, r0 in s["rho_res0"].items():
        out["res"][c] = {g: float(g * np.log2(r0)) if r0 > 0 else None for g in (2, 8, 64)}
    return out


def main(out: str) -> None:
    res = {}
    for name, (f, what) in F.CANDIDATES.items():
        t = time.time()
        s = spectrum(f)
        s["what"] = what
        s["bounds"] = bounds(s)
        s["seconds"] = round(time.time() - t, 1)
        res[name] = s
        print(name, json.dumps({k: s[k] for k in ("bijective", "rho_lin", "arg_lin", "rho_lin0", "rho_res", "rho_res0")}),
              s["seconds"], flush=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/agg/spec.json")
