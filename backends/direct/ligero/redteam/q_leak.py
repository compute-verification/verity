"""The finding: the chain-test message ``q`` is not masked on H.

``q_d = Q_d + Z_H s_chain_d`` (protocol.py, "ZK masking"): the mask ``Z_H(x) s_chain(x)`` VANISHES at every point of
``H`` (not merely in its sum over H, which is what the verifier's scalar check uses), so ``q_d|_H = Q_d|_H`` exactly,
and ``Q_d(h_j) = sum_i coef[d, i, j] W[rows[i]][j]`` is the per-column share of the chain constraints -- a
witness-dependent linear function the verifier reads off the transcript with ``_fold_to_H(q, l)``.

For the honest witness (``c_in[j+1] = c_out[j]``) it is, with the verifier's own coefficients ``rc``,

    Q_d(h_j) = sum_x rl_x[d, j-1] c_x[j] + sum_x rs_x[d, j] c_x[j] - sum_x rl_x[d, j] (c_x[j+1] - Yc_x) + re[d, j] (y16_j - Y16c)

(``rl`` = link, ``rs`` = start, ``re`` = end coefficients; ``c_x`` the accumulator components ``acc.s, acc.t, acc.f``
at column ``j``; ``Yc_x``, ``Y16c`` the constant terms of the output expressions).  ``c[start] = 0`` is public, so
walking each VU forward gives ``D = 6`` equations in the 3 unknown components of the next accumulator at every
step: the whole accumulator chain of every VU, exactly, from the public transcript and the verifier's coins.

Under this prover's statement the operands are public, so the accumulators are computable anyway (the ZK claim
is vacuous for this statement); as a Ligero ZK *construction* -- the thing the label is about once operands are
committed rather than published -- the message leaks ``l - 1`` witness-dependent field elements per coordinate.
"""
from __future__ import annotations

import numpy as np
import torch

from .. import protocol
from ..compile import P, System
from ..protocol import Config, Proof, _challenge1, _fold_to_H, n_rows, statement_digest
from .linalg import solve_overdetermined


def q_on_H(proof: Proof, cfg: Config) -> np.ndarray:
    """(D, l) evaluations of the sent chain polynomial on H -- public data."""
    q = torch.from_numpy(proof.q.astype(np.int64))
    return _fold_to_H(q, cfg.l).numpy()


def chain_challenge(proof: Proof, sys: System, tb, cfg: Config, layout, y16, a, b, device) -> torch.Tensor:
    """The verifier's chain coefficients ``rc`` (D, 7, l) recomputed exactly as ``verify`` does."""
    stmt = statement_digest(sys, cfg, a, b, layout=layout, y16=torch.as_tensor(y16, dtype=torch.int64))
    M = n_rows(cfg, sys.m, True)
    ccom = tuple(proof.coin_commit) if (protocol._COMMIT_COINS and proof.coin_commit is not None) else None
    r1 = proof.coins.r1 if (protocol._COMMIT_COINS and proof.coins is not None) else None
    _, _, _, rc = _challenge1(stmt, proof.root, cfg.D, M, tb.L, tb.Q, device, cfg.l, ccom, r1)
    return rc.cpu()


def recover_accumulators(proof: Proof, sys: System, tb, cfg: Config, layout, y16, a, b, device) -> dict:
    """Recover ``(acc.s, acc.t, acc.f)`` of every accumulator of every VU from ``q`` alone (plus the public
    statement and the verifier's coins).  Returns the recovered components per column and the consistency flags."""
    D, l, steps, n_vus = cfg.D, cfg.l, layout.steps, layout.n_vus
    QH = q_on_H(proof, cfg)                                     # (D, l)
    rc = chain_challenge(proof, sys, tb, cfg, layout, y16, a, b, device).numpy()   # (D, 7, l)
    start, link, end = (m.cpu().numpy() for m in layout.masks("cpu"))
    y16 = np.asarray(y16, dtype=np.int64)
    Yc = [int(e.c) % P for e in sys.chain["y"]]
    Y16c = int(sys.chain["y16"].c) % P
    rl = (rc[:, 0:3, :] * link[None, None, :]) % P             # (D, 3, l)
    rs = (rc[:, 3:6, :] * start[None, None, :]) % P
    re = (rc[:, 6, :] * end[None, :]) % P
    rec = np.zeros((l, 3), dtype=np.int64)                      # c_x[j]: accumulator entering column j
    solved = np.zeros(l, dtype=bool)
    consistent = np.zeros(l, dtype=bool)
    end_ok = []
    for v in range(n_vus):
        j0 = steps * v
        c_prev = [0, 0, 0]                                      # public: c_0 = 0 at every chain start
        solved[j0] = consistent[j0] = True
        for i in range(steps):
            j = j0 + i
            # known part of Q_d(h_j)
            known = []
            for d in range(D):
                k = 0
                for x in range(3):
                    k += int(rl[d, x, j - 1] if j >= 1 else 0) * c_prev[x] + int(rs[d, x, j]) * c_prev[x]
                    k += int(rl[d, x, j]) * Yc[x]
                k += int(re[d, j]) * ((int(y16[j]) - Y16c) % P)
                known.append(k % P)
            if i < steps - 1:
                # sum_x rl_x[d, j] c_x[j+1] = known_d - Q_d(h_j)
                A = [[int(rl[d, x, j]) for x in range(3)] for d in range(D)]
                bvec = [(known[d] - int(QH[d, j])) % P for d in range(D)]
                sol, ok = solve_overdetermined(A, bvec)
                if sol is None:
                    break
                rec[j + 1] = sol
                solved[j + 1] = True
                consistent[j + 1] = ok
                c_prev = sol
            else:
                end_ok.append(all((known[d] - int(QH[d, j])) % P == 0 for d in range(D)))
    return {"components": rec, "solved": solved, "consistent": consistent, "end_positions_consistent": end_ok}


def decode_fp32(word: int) -> tuple[int, int, int]:
    return (word >> 31) & 1, (word >> 23) & 0xFF, word & 0x7FFFFF


def compare_with_recorded(rec: dict, vus: list[tuple], steps: int) -> dict:
    """``vus``: (x, w, accs, y16) per VU as the runner receives them; the recovered accumulator entering column
    ``j0 + i + 1`` must equal the recorded ``accs[i]``."""
    comp = rec["components"]
    total = match = 0
    mismatches = []
    for v, (_, _, accs, _) in enumerate(vus):
        j0 = steps * v
        for i in range(steps - 1):
            want = decode_fp32(int(accs[i]))
            got = tuple(int(x) for x in comp[j0 + i + 1])
            total += 1
            if got == want:
                match += 1
            elif len(mismatches) < 5:
                mismatches.append({"vu": v, "step": i, "recovered": got, "recorded": want})
    solved = rec["solved"]
    return {"accumulators_recovered_exactly": match, "of": total,
            "all_equations_consistent": bool(rec["consistent"][solved].all()), "columns_solved": int(solved.sum()),
            "end_positions_consistent": rec["end_positions_consistent"], "mismatches": mismatches}


def witness_free_distinguisher(proof: Proof, sys: System, tb, cfg: Config, layout, y16, a, b, device) -> dict:
    """A verifier-side test that needs no witness: the 6 x 3 systems above are consistent for a REAL transcript
    (every accumulator exists) and inconsistent (probability 1 - p^-3 per step) for a transcript whose ``q`` was
    sampled uniformly subject to the public sum -- what any simulator without the witness produces.  The fraction
    of consistent steps is the distinguisher's statistic."""
    rec = recover_accumulators(proof, sys, tb, cfg, layout, y16, a, b, device)
    steps = layout.steps
    idx = [steps * v + i + 1 for v in range(layout.n_vus) for i in range(steps - 1)]
    cons = rec["consistent"][idx]
    return {"steps": len(idx), "consistent_steps": int(cons.sum()), "fraction": float(cons.mean()) if len(idx) else None}
