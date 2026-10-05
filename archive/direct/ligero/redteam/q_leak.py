"""The finding (red team F1) -- FIXED on lane b-zk-fix: the chain-test message ``q`` was not masked on H.
Since the fix ``q_d = Q_d + u_d`` with ``u_d = u_ca + x^k u_cb`` uniform of degree < l + k - 1 subject to
``sum_H u = 0``, so ``q|_H`` is uniform on the affine hyperplane ``sum = rhs`` and the recovery below must FAIL
(``compare_with_recorded`` -> 0 accumulators, ``witness_free_distinguisher`` -> fraction ~ p^-3).  The module is
kept as the regression test.  The text below describes the pre-fix protocol.


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
The recovery itself is torch-free (``verity_numerical.redteam.ligero_q_leak``, shared with the fixture test).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from verity_numerical.redteam import ligero_q_leak as core

from .. import protocol
from ..compile import P, System
from ..protocol import Config, Proof, _challenge1, _fold_to_H, n_rows, statement_digest

decode_fp32 = core.decode_fp32


def q_on_H(proof: Proof, cfg: Config) -> np.ndarray:
    """(D, l) evaluations of the sent chain polynomial on H -- public data."""
    q = torch.from_numpy(proof.q.astype(np.int64))
    return _fold_to_H(q, cfg.l).numpy()


def chain_challenge(proof: Proof, sys: System, tb, cfg: Config, layout, y16, a, b, device) -> torch.Tensor:
    """The verifier's chain coefficients ``rc`` (D, 7, l) recomputed exactly as ``verify`` does."""
    stmt = statement_digest(sys, cfg, a, b, layout=layout, y16=torch.as_tensor(y16, dtype=torch.int64))
    M = n_rows(cfg, sys.m, True)
    ccom = tuple(proof.coin_commit) if (cfg.interactive and proof.coin_commit is not None) else None
    r1 = proof.coins.r1 if (cfg.interactive and proof.coins is not None) else None
    _, _, _, rc, _ = _challenge1(stmt, proof.root, cfg.D, M, tb.L, tb.Q, device, cfg.l, ccom, r1)
    return rc.cpu()


def _chain_constants(sys: System) -> tuple[list[int], int]:
    return [int(e.c) % P for e in sys.chain["y"]], int(sys.chain["y16"].c) % P


def recover_accumulators(proof: Proof, sys: System, tb, cfg: Config, layout, y16, a, b, device) -> dict:
    """Recover ``(acc.s, acc.t, acc.f)`` of every accumulator of every VU from ``q`` alone (plus the public
    statement and the verifier's coins).  Returns the recovered components per column and the consistency flags."""
    QH = q_on_H(proof, cfg).tolist()
    rc = chain_challenge(proof, sys, tb, cfg, layout, y16, a, b, device).numpy().tolist()
    Yc, Y16c = _chain_constants(sys)
    rec = core.recover(QH, rc, [int(v) for v in np.asarray(y16)], Yc, Y16c, cfg.l, layout.n_vus, layout.steps)
    return {"components": np.asarray(rec["components"], dtype=np.int64), "solved": np.asarray(rec["solved"]),
            "consistent": np.asarray(rec["consistent"]), "end_positions_consistent": rec["end_positions_consistent"]}


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


def export_fixture(path: Path, proof: Proof, sys: System, tb, cfg: Config, layout, vus: list[tuple], a, b, y16, device,
                   source: str) -> dict:
    """Write the torch-free reproduction data: q, the verifier's rc, the chain constants, the layout, the recorded
    accumulators, and what the recovery gave.  The loader test replays ``ligero_q_leak.recover`` on it."""
    rc = chain_challenge(proof, sys, tb, cfg, layout, y16, a, b, device).numpy()
    Yc, Y16c = _chain_constants(sys)
    rec = recover_accumulators(proof, sys, tb, cfg, layout, y16, a, b, device)
    cmp_ = compare_with_recorded(rec, vus, layout.steps)
    doc = {
        "schema": "redteam-zk/q-leak/v1", "source": source,
        "config": {"l": cfg.l, "n": cfg.n, "k": cfg.k, "t": cfg.t, "t_pad": cfg.t_pad, "D": cfg.D, "m": sys.m},
        "layout": {"l": layout.l, "n_vus": layout.n_vus, "steps": layout.steps},
        "transcript": {"root": proof.root.hex(), "coin_commit": [c.hex() for c in proof.coin_commit],
                       "r1": proof.coins.r1.hex(), "s1": proof.coins.s1.hex(), "q": proof.q.astype(np.int64).tolist()},
        "verifier_rc": rc.astype(np.int64).tolist(),
        "chain_constants": {"Yc": Yc, "Y16c": Y16c},
        "y16": [int(v) for v in np.asarray(y16)],
        "recorded_accumulators": [[int(x) for x in accs] for (_, _, accs, _) in vus],
        "recovered_components": rec["components"].tolist(),
        "recovered_matches": cmp_,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return doc
