"""Re-audit finding G1: the simulator's DUMMY run (the run that extracts the coins) sends ``v0, q0`` uniform WITHOUT
the public constraints ``v|_H = beta`` and ``sum_H q = rhs`` that every honest prover's messages satisfy.  A V* that
aborts at slot 2 iff those public checks fail (a legal, witness-free verifier policy) never aborts on the real
prover and aborts on the simulator's dummy run with probability ~1 -- the simulator outputs an abort transcript the
real interaction never produces.  Not a witness leak (the predicate sees no witness); a simulator bug: the dummy run
must sample its messages from the honest-verifier-observable distribution (both constraints are public).

    python -m backends.direct.ligero.redteam.zk2_dummy --device cpu --vus 2 --n 12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from verity_numerical.bench import instances

from .. import protocol
from ..chain import chain_coefs
from ..compile import P
from ..protocol import CoinOpeningError, Coins
from ..unit import public_vectors
from ..vu import ChainRunner
from . import simulator
from .run import _load_real


class PublicCheckVerifier(Coins):
    """Honest coins; refuses to open slot 2 iff the prover's ``v`` or ``q`` violates its PUBLIC constraint."""

    def __init__(self, R, cfg, lay, y16, a, b, device):
        h = Coins.sample()
        super().__init__(h.r1, h.s1, h.r2, h.s2)
        y16_t = torch.as_tensor(y16, dtype=torch.int64)
        stmt = protocol.statement_digest(R.sys, cfg, a, b, layout=lay, y16=y16_t)
        M = protocol.n_rows(cfg, R.sys.m, True)
        _, rho_lin, _, rc = protocol._challenge1(stmt, b"", cfg.D, M, R.tb.L, R.tb.Q, device, cfg.l, self.commitments(), self.r1)
        pub = public_vectors(R.p, None, a, b, None, device=device)
        _, beta = protocol._alpha_beta(R.tb, rho_lin, pub, R.sys.m, cfg.l, device)
        object.__setattr__(self, "beta", (beta % P).cpu())
        object.__setattr__(self, "rhs", chain_coefs(R.sys, lay, rc, y16_t).rhs.cpu())
        object.__setattr__(self, "l", cfg.l)
        object.__setattr__(self, "aborted", [])

    def open2(self, root, w, h, q, v):
        vH = protocol._fold_to_H(torch.from_numpy(v.astype(np.int64)), self.l)
        qs = protocol._sum_over_H(torch.from_numpy(q.astype(np.int64)), self.l)
        if not torch.equal(vH, self.beta) or not torch.equal(qs, self.rhs):
            self.aborted.append(2)
            return b"\0" * protocol.COIN_BYTES, b"\0" * protocol.COIN_BYTES
        return self.r2, self.s2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--root", default=str(instances.DEFAULT_ROOT))
    ap.add_argument("--vus", type=int, default=2)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    dev = args.device
    real = _load_real(Path(args.root), args.vus)
    R = ChainRunner(dev, -128.0, n_proofs=1, zk=True, mode=protocol.MODE_INTERACTIVE)
    lay = R.layout(len(real))
    cfg = R.cfg(lay.l)
    pr, (a, b, y16), _ = R.prove_vus(real, check=False)
    ab_real = ab_sim = 0
    for _ in range(args.n):
        V = PublicCheckVerifier(R, cfg, lay, y16, a, b, dev)
        try:
            p2, _, _ = R.prove_vus(real, check=False, coins=V)
            ok, why = R.verify_vus(p2, (a, b, y16), lay, coins=V)
            assert ok, why
        except CoinOpeningError:
            ab_real += 1
        V2 = PublicCheckVerifier(R, cfg, lay, y16, a, b, dev)
        out = simulator.simulate(V2, R.p, R.sys, R.tb, cfg, lay, y16, a, b, dev, None)
        if isinstance(out, simulator.AbortTranscript):
            ab_sim += 1
    rep = {"config": {"l": cfg.l, "t": cfg.t, "D": cfg.D}, "n": args.n, "real_abort_rate": ab_real / args.n,
           "sim_abort_rate": ab_sim / args.n, "advantage": abs(ab_real - ab_sim) / args.n,
           "verdict": "SIMULATOR BUG: dummy-run messages off-distribution" if ab_sim > ab_real else "no separation"}
    print(json.dumps(rep))
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
