"""Simulator-style test of the HVZK masking (PROTOCOL.md section 8b as implemented in ``protocol.py``).

What the honest verifier sees of the witness: the t opened columns of U (every committed row at t points off H), and
the messages w (proximity), v (linear), h (quadratic), q (chain).  Under the masking each of these is uniform (w, v, h)
or uniform subject to the public constraint (v on H = beta; sum over H of q = r^T b), independent of the witness --
so a simulator that samples them uniformly and programs the random oracle is perfect up to the 2^-31 sampling bias.
This module checks the *distributional* consequence on real proofs, against the unmasked prover as the control:

  1. real vs simulated: a proof of real VUs and a proof of the trivial VU (all-zero operands; what a simulator that
     knows no witness can produce) at the same layout -- two-sample chi-square over 64 value bins on the opened
     columns and on every message.  Masked: indistinguishable (p > 0.01).  Unmasked: they differ wildly (the trivial
     witness has thousands of constant rows).
  2. uniformity of the opened columns (chi-square against uniform, 64 bins).
  3. payload in the clear: opened values equal to a payload value of their own row (non-ZK: every constant row is
     opened in the clear at every column -- ~35% of all opened values are 0; ZK: the uniform-chance count M t |payload| / p).

    python -m backends.direct.ligero.redteam_zk --device cuda --out results/redteam_zk.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from verity_numerical.bench import instances

from .compile import P
from .protocol import mask_rows
from .vu import STEP, STEPS, ChainRunner, _vu_data
from .witness import build_witness

BINS = 64


def _chi2_uniform(x: np.ndarray) -> tuple[float, float]:
    """Chi-square statistic and p-value (Wilson-Hilferty normal approximation) of x against uniform on [0, p)."""
    counts = np.bincount((x.astype(np.int64).ravel() * BINS // P), minlength=BINS)
    e = x.size / BINS
    stat = float(((counts - e) ** 2 / e).sum())
    return stat, _chi2_sf(stat, BINS - 1)


def _chi2_two_sample(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    cx = np.bincount((x.astype(np.int64).ravel() * BINS // P), minlength=BINS).astype(float)
    cy = np.bincount((y.astype(np.int64).ravel() * BINS // P), minlength=BINS).astype(float)
    kx, ky = math.sqrt(cy.sum() / cx.sum()), math.sqrt(cx.sum() / cy.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (kx * cx - ky * cy) ** 2 / (cx + cy)
    stat = float(np.nansum(terms))
    return stat, _chi2_sf(stat, BINS - 1)


def _chi2_sf(stat: float, dof: int) -> float:
    z = ((stat / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _payload_in_clear(W: torch.Tensor, opened: np.ndarray) -> dict:
    """Opened values equal to a payload value of their own row, and the constant rows opened in the clear."""
    Wn = W.cpu().numpy().astype(np.int64)
    m = Wn.shape[0]
    hits, const_rows, const_leaked, distinct = 0, 0, 0, 0
    for j in range(m):
        vals = np.unique(Wn[j])
        distinct += vals.size
        h = np.isin(opened[j].astype(np.int64), vals).sum()
        hits += int(h)
        if vals.size == 1:
            const_rows += 1
            const_leaked += int((opened[j] == vals[0]).all())
    t = opened.shape[1]
    return {"opened_values_equal_to_own_payload": hits, "of": m * t, "expected_if_uniform": distinct * t / P,
            "constant_rows": const_rows, "constant_rows_opened_in_the_clear": const_leaked}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--root", default=str(instances.DEFAULT_ROOT))
    ap.add_argument("--vus", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    vu = instances.load("vu-k1536", Path(args.root))
    real = [_vu_data(vu, i) for i in range(args.vus)]
    trivial = [([0] * (STEP * STEPS), [0] * (STEP * STEPS), [0] * STEPS, 0)] * args.vus
    report = {}
    for zk in (False, True):
        R = ChainRunner(args.device, -128.0, zk=zk)
        pr, pubs_r, lay = R.prove_vus(real, check=False)
        ps, pubs_s, _ = R.prove_vus(trivial, check=False)
        okr, whyr = R.verify_vus(pr, pubs_r, lay)
        oks, whys = R.verify_vus(ps, pubs_s, lay)
        assert okr and oks, (whyr, whys)
        m = R.sys.m
        # the witness of the real proof, rebuilt for the payload test (the test knows the witness; the verifier does not)
        from .hints_torch import hints_torch
        from .unit import public_vectors
        a, b, _ = pubs_r
        units = []
        for (x, w, accs, y) in real:
            c = 0
            for s in range(STEPS):
                units.append((c, x[16 * s:16 * s + 16], w[16 * s:16 * s + 16], accs[s]))
                c = accs[s]
        while len(units) < lay.l:
            units.append((0, [0] * STEP, [0] * STEP, 0))
        c_in = np.asarray([u[0] for u in units]); y_out = np.asarray([u[3] for u in units])
        pub = public_vectors(R.p, None, a, b, None, device=R.device)
        hints = hints_torch(R.p, R.sys, c_in, a, b, y_out, R.device)
        W = build_witness(R.sys, pub, hints, R.device)
        rep = {"config": str(R.cfg(lay.l)), "committed_rows": int(pr.opened.shape[0]), "mask_rows": {k: [s.start, s.stop] for k, s in mask_rows(R.cfg(lay.l), m, True).items()}}
        stat, pv = _chi2_uniform(pr.opened[:m])
        rep["opened_uniform"] = {"chi2": stat, "p": pv, "n": int(pr.opened[:m].size)}
        rep["real_vs_simulated"] = {}
        for name, xr, xs in (("opened", pr.opened, ps.opened), ("w", pr.w, ps.w), ("h", pr.h, ps.h), ("q", pr.q, ps.q),
                             ("v", pr.v, ps.v)):
            if xr is None:
                continue
            stat, pv = _chi2_two_sample(xr, xs)
            rep["real_vs_simulated"][name] = {"chi2": stat, "p": pv, "n": int(xr.size)}
        rep["payload"] = _payload_in_clear(W, pr.opened[:m])
        rep["zeros_among_opened"] = int((pr.opened == 0).sum())
        report["zk" if zk else "non_zk"] = rep
        print(("ZK   " if zk else "plain") + f" {rep['config']}\n   opened uniform: chi2 {rep['opened_uniform']['chi2']:.1f} p {rep['opened_uniform']['p']:.3g}")
        for k_, v_ in rep["real_vs_simulated"].items():
            print(f"   real vs simulated {k_:6s}: chi2 {v_['chi2']:.1f} p {v_['p']:.3g}")
        print(f"   payload: {rep['payload']}\n   zeros among opened values: {rep['zeros_among_opened']}")
    zk = report["zk"]
    verdict = (zk["opened_uniform"]["p"] > 0.001 and all(v["p"] > 0.001 for v in zk["real_vs_simulated"].values())
               and zk["payload"]["constant_rows_opened_in_the_clear"] == 0
               and zk["payload"]["opened_values_equal_to_own_payload"] < 20 * (1 + zk["payload"]["expected_if_uniform"]))
    report["verdict"] = "PASS" if verdict else "FAIL"
    print("verdict:", report["verdict"])
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1))
    return 0 if verdict else 2


if __name__ == "__main__":
    raise SystemExit(main())
