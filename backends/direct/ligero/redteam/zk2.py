"""Re-audit (lane red-team-zk-2) of the b-zk-fix repairs.

The F1 repair masks ``q = Q + u_ca + x^k u_cb`` and commits ``u_ca`` (deg < k) and ``u_cb`` (deg < l - 1) as rows,
OPENED at the ``t`` columns.  The verifier's column check ``q(eta_c) = Q(eta_c) + u_ca(eta_c) + eta_c^k u_cb(eta_c)``
fixes only the sum ``u(eta_c)``; the separately opened ``u_cb(eta_c)`` is EXTRA information:

    u_cb(eta_c) = (q - Q)[k:](eta_c)      =>      Q[k:](eta_c) = q[k:](eta_c) - u_cb(eta_c)   for every opened c,

``t`` linear functionals of the HIGH coefficients (indices >= k) of ``Q = sum_i R_i p_i`` -- and ``Q`` has degree up
to ``2l - 2 >= k`` at the real layout, so those coefficients carry the witness.  They are hidden iff the residual
row-mask randomness covers them: ``p_i = P_i + Z_H s_i`` with ``s_i`` of ``t_pad`` coefficients, of which the
openings fix ``t`` (``s_i(eta_c)``); the residual is ``s_i = (fixed) + Z_cols(x) r_i(x)`` with ``deg r_i < t_pad - t``.
So the map

    (r_i)_{i in rows(Q)}  ->  ( [R_i Z_H Z_cols r_i][k:](eta_c) )_c        in F^t

must have rank ``min(t, l - 1)`` (the high part has ``l - 1`` coefficients) for every challenge coordinate ``d`` (the ``u_d`` are independent).  b-zk-fix's simulator
samples ``u_cb`` uniform and fits ``u_ca``: it is exactly correct iff these ranks are full.  ``u_high_rank`` computes
them on the real challenge ``rc`` and the real opened columns.

Same structure for the quadratic test (unchanged since 8b): ``s_qb(eta_c)`` opened separately gives
``(P0 / Z_H)[k:](eta_c)``; there the high part is ``sum_q rho_q [Z_H s_a s_b][k:]`` (``P_a s_b`` has degree < k), a
function of the masks only -- witness-free by degree, checked empirically across two witnesses (``qb_two_witness``).
"""
from __future__ import annotations

import numpy as np
import torch

from .. import protocol
from ..chain import chain_coefs
from ..compile import P
from ..field import coset_points, encode, eval_on_coset, intt, zh_on_coset
from ..protocol import Config, Proof
from .linalg import rank_mod_p


def _coefs_from_coset(vals: torch.Tensor, n: int) -> torch.Tensor:
    """(R, n) evaluations on g<w_n> -> (R, n) coefficients (the prover's own recipe)."""
    return (intt(vals % P) * protocol._ginv_powers(n, vals.device)) % P


def _high_part_at_cols(f_on_coset: torch.Tensor, n: int, k: int, cols_t: torch.Tensor, deg_bound: int) -> torch.Tensor:
    """For polynomials given by their coset evaluations (R, n) with degree < deg_bound <= n: the value at the opened
    columns of the part with indices >= k, i.e. f(eta_c) - f[:k](eta_c)."""
    cf = _coefs_from_coset(f_on_coset, n)
    assert int((cf[:, deg_bound:] != 0).sum()) == 0, "degree overflow"
    low_at = eval_on_coset(cf[:, :k], n)[:, cols_t]
    return (f_on_coset[:, cols_t] - low_at) % P


def u_high_rank(proof: Proof, sys, tb, cfg: Config, layout, y16, a, b, device, max_rows: int | None = None) -> dict:
    """Rank, per challenge coordinate d, of the map from the residual row masks to ``Q_d[k:](eta_c)`` at the proof's
    opened columns.  ``max_rows``: use only that many involved rows (the rank is monotone in the rows)."""
    l, n, k, t, tp, D = cfg.l, cfg.n, cfg.k, cfg.t, cfg.t_pad, cfg.D
    slack = tp - t
    y16_t = torch.as_tensor(y16, dtype=torch.int64)
    stmt = protocol.statement_digest(sys, cfg, a, b, layout=layout, y16=y16_t)
    M = protocol.n_rows(cfg, sys.m, True)
    cc = tuple(proof.coin_commit) if cfg.interactive else None
    r1 = proof.coins.r1 if cfg.interactive else None
    _, _, _, rc = protocol._challenge1(stmt, proof.root, D, M, tb.L, tb.Q, device, l, cc, r1)
    ccf = chain_coefs(sys, layout, rc, y16_t)
    R = ccf.coef.shape[1]
    rows_used = R if max_rows is None else min(R, max_rows)
    cols_t = torch.tensor(proof.cols, dtype=torch.int64, device=device)
    xs = coset_points(n, torch.arange(n, dtype=torch.int64)).to(device)                 # every coset point
    eta = xs[cols_t]
    zc = torch.ones(n, dtype=torch.int64, device=device)
    for e in eta.tolist():
        zc = (zc * ((xs - e) % P)) % P                                                  # Z_cols on the coset
    base = (zh_on_coset(n, l, device) * zc) % P                                         # Z_H Z_cols on the coset
    xp = torch.empty((slack, n), dtype=torch.int64, device=device)                      # x^j, j < slack
    xp[0] = 1
    for j in range(1, slack):
        xp[j] = (xp[j - 1] * xs) % P
    Renc = encode(ccf.coef.reshape(D * R, l), n).reshape(D, R, n)                       # R_i on the coset
    out = {"config": {"l": l, "k": k, "t": t, "t_pad": tp, "D": D, "n": n, "mode": cfg.mode}, "involved_rows": int(R),
           "rows_used": rows_used, "residual_dims_per_row": slack, "total_residual_dims": rows_used * slack,
           "needed_rank": min(t, l - 1), "high_coefficients": l - 1, "Q_degree_bound": 2 * l - 1, "Q_has_high_part": 2 * l - 2 >= k, "per_d": []}
    for d in range(D):
        blocks = []
        for i in range(rows_used):
            f = ((Renc[d, i] * base) % P)[None, :] * xp % P                             # (slack, n)
            blocks.append(_high_part_at_cols(f, n, k, cols_t, l + k - 1).T.cpu().numpy())   # (t, slack)
        Mx = np.concatenate(blocks, 1) % P
        rk = rank_mod_p(Mx)
        out["per_d"].append({"d": d, "rank": rk, "full": rk == min(t, l - 1)})
    out["all_full"] = all(x["full"] for x in out["per_d"])
    out["verdict"] = ("HOLDS: the opened u_cb values are uniform given the rest of the transcript" if out["all_full"]
                      else "LEAK: Q[k:](eta_c) not covered by the residual row masks")
    return out


def disclosed_Q_high(proof: Proof, m: int, cfg: Config, device) -> np.ndarray:
    """The ``t`` values the transcript discloses per coordinate: ``Q[k:](eta_c) / eta_c^k = q[k:](eta_c) - u_cb(eta_c)``
    (``q[k:]`` read as the polynomial with the high coefficients shifted down by ``k``).  (D, t)."""
    mr = protocol.mask_rows(cfg, m, True)
    k, n = cfg.k, cfg.n
    cols_t = torch.tensor(proof.cols, dtype=torch.int64, device=device)
    q = torch.from_numpy(proof.q.astype(np.int64)).to(device)
    q_high_at = eval_on_coset(q[:, k:], n)[:, cols_t]
    ucb_at = torch.from_numpy(proof.opened[mr["chain_b"]].astype(np.int64)).to(device)
    return ((q_high_at - ucb_at) % P).cpu().numpy()


def pooled_uniform_chi2(x: np.ndarray, bins: int = 64) -> dict:
    from .simulator import _chi2_sf
    counts = np.bincount((np.asarray(x).astype(np.int64).ravel() * bins // P), minlength=bins)
    e = x.size / bins
    stat = float(((counts - e) ** 2 / e).sum())
    return {"chi2": stat, "p": float(_chi2_sf(np.array([stat]), bins - 1)[0]), "n": int(x.size)}


def qb_two_witness(proofs_a: list[Proof], proofs_b: list[Proof], cfg: Config, m: int) -> dict:
    """Opened ``s_qb`` and ``u_cb`` rows across two witnesses: per-position chi2 (marginal); and the disclosed
    ``Q[k:](eta_c)`` values across two witnesses (must be at chance iff ``u_high_rank`` is full)."""
    from .simulator import per_position_chi2
    mr = protocol.mask_rows(cfg, m, True)
    out = {}
    for name in ("qb", "chain_b", "chain_a", "lin", "qa", "prox"):
        X = np.stack([pr.opened[mr[name]] for pr in proofs_a])
        Y = np.stack([pr.opened[mr[name]] for pr in proofs_b])
        out[f"opened_mask_{name}"] = per_position_chi2(X, Y)
    return out
