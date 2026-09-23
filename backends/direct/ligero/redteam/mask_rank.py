"""Class 1: exact dimension counts of the section-8b masks against every disclosure of the transcript.

For each disclosed object the *admissible space* (what the verifier's public checks allow) is compared with the
*mask span* (the image of the fresh randomness the prover adds).  The object is witness-independent iff the mask
span covers the admissible space (then the honest message is uniform on it whatever the witness).  Ranks are
computed on explicit matrices at small parameters and stated by formula at the real layout (the matrices are
structured: identities, shifts, and ``Z_H`` multiplication, whose ranks do not depend on the size).

  opened columns of a witness row   admissible F^t, mask span = image of s_j (deg < t_pad) under
                                    s -> (Z_H(eta_c) s(eta_c))_c : rank t iff t <= t_pad (Vandermonde at distinct
                                    points, Z_H != 0 on the coset)                              -> covered
  w_d  (k coefficients)             mask s_prox_d, deg < k: identity                              -> covered
  v_d  (k coefficients, v|_H = beta) admissible dim k - l = t_pad; mask Z_H s_lin_d, deg s < t_pad: rank t_pad -> covered
  h_d  (2k-1-l coefficients)        mask s_qa (deg < k) + x^k s_qb (deg < t_pad - 1): rank k + t_pad - 1 = 2k-1-l -> covered
  q_d  (l+k-1 coefficients, sum_H q = rhs)  admissible dim l + k - 2; PRE-FIX mask Z_H s_chain (deg < k - 1): rank k - 1
                                    -> DEFICIENT by l - 1 (red team F1).  b-zk-fix: mask u_ca + x^k u_cb (deg < k, < l-1)
                                    with sum_H u = 0: rank l + k - 2 = admissible dim               -> covered
"""
from __future__ import annotations

import numpy as np

from ..compile import P
from ..field import GEN, root_of_unity
from ..protocol import Config
from .linalg import rank_mod_p


def opened_columns_rank(cfg: Config, cols: list[int]) -> dict:
    """Rank of the map s_j -> (Z_H(eta_c) s_j(eta_c))_{c in cols} for the actual opened columns."""
    w = root_of_unity(cfg.n)
    tp = cfg.t_pad
    M = np.zeros((len(cols), tp), dtype=np.int64)
    for i, c in enumerate(cols):
        eta = GEN * pow(w, c, P) % P
        zh = (pow(eta, cfg.l, P) - 1) % P
        pw = zh
        for e in range(tp):
            M[i, e] = pw
            pw = pw * eta % P
    r = rank_mod_p(M)
    return {"t": len(cols), "t_pad": tp, "rank": r, "covered": r == len(cols)}


def _zh_times_matrix(l: int, deg_s: int, out_len: int) -> np.ndarray:
    """Matrix of s -> Z_H s = x^l s - s as (out_len x deg_s)."""
    M = np.zeros((out_len, deg_s), dtype=np.int64)
    for i in range(deg_s):
        M[i, i] = (M[i, i] - 1) % P
        M[i + l, i] = (M[i + l, i] + 1) % P
    return M


def message_ranks(l: int, t_pad: int) -> dict:
    """Explicit matrices at (l, t_pad): admissible dimension vs mask rank per message."""
    k = l + t_pad
    out = {}
    # w: mask is the identity on F^k
    out["w"] = {"admissible_dim": k, "mask_rank": k, "covered": True}
    # v: admissible = interp(beta) + {Z_H u : deg u < t_pad}; mask Z_H s_lin with deg s_lin < t_pad
    Mv = _zh_times_matrix(l, t_pad, k)
    rv = rank_mod_p(Mv)
    out["v"] = {"admissible_dim": k - l, "mask_rank": rv, "covered": rv == k - l}
    # h: 2k-1-l coefficients; mask [I_k | x^k I_{t_pad-1}]
    hl = 2 * k - 1 - l
    Mh = np.zeros((hl, k + t_pad - 1), dtype=np.int64)
    for i in range(k):
        Mh[i, i] = 1
    for i in range(t_pad - 1):
        Mh[k + i, k + i] = 1
    rh = rank_mod_p(Mh)
    out["h"] = {"admissible_dim": hl, "mask_rank": rh, "covered": rh == hl}
    # q: l + k - 1 coefficients with one public linear condition.  Pre-fix mask Z_H s_chain, deg s_chain < k - 1:
    ql = l + k - 1
    Mq = _zh_times_matrix(l, k - 1, ql)
    rq = rank_mod_p(Mq)
    out["q_prefix_Z_H_s_chain"] = {"admissible_dim": ql - 1, "mask_rank": rq, "covered": rq == ql - 1, "deficiency": ql - 1 - rq,
                                   "expected_deficiency_formula_l_minus_1": l - 1}
    # b-zk-fix mask u_ca + x^k u_cb (identity on all ql coefficients) restricted to the zero-sum subspace: the map
    # (u_ca, u_cb) -> u projected along the sum functional has rank ql - 1 = admissible dim
    Mu = np.eye(ql, dtype=np.int64)
    Mu = np.concatenate([Mu[:, :k], Mu[:, k:k + l - 1]], 1)            # (ql, k + l - 1) = identity
    # quotient by the sum-over-H functional: append the functional as a row constraint and count the kernel dimension
    f = np.zeros((1, ql), dtype=np.int64)
    f[0, ::l] = l % P
    span = rank_mod_p(Mu)                                              # ql
    ker_sum = span - rank_mod_p(f @ Mu % P)                            # ql - 1 mask vectors with zero sum
    out["q"] = {"admissible_dim": ql - 1, "mask_rank": ker_sum, "covered": ker_sum == ql - 1, "deficiency": ql - 1 - ker_sum}
    return out


def real_layout_counts(cfg: Config) -> dict:
    """The same counts by formula at the real configuration."""
    l, k, tp, t = cfg.l, cfg.k, cfg.t_pad, cfg.t
    return {
        "config": {"l": l, "k": k, "t_pad": tp, "t": t, "D": cfg.D},
        "opened_columns": {"admissible": t, "mask_free_coefficients": tp, "spare": tp - t, "covered": tp >= t},
        "w": {"admissible_dim": k, "mask_rank": k, "covered": True},
        "v": {"admissible_dim": k - l, "mask_rank": tp, "covered": tp == k - l},
        "h": {"admissible_dim": 2 * k - 1 - l, "mask_rank": k + tp - 1, "covered": k + tp - 1 == 2 * k - 1 - l},
        "q": {"admissible_dim": l + k - 2, "mask_rank": l + k - 2, "covered": True, "deficiency": 0,
              "field_elements_in_the_clear_per_proof": 0,
              "prefix": {"mask_rank": k - 1, "deficiency": l - 1, "field_elements_in_the_clear_per_proof": cfg.D * (l - 1)}},
        "linear_disclosures_of_s_j_per_witness_row": {"openings": t, "through_messages": 0,
                                                       "note": "w, v, h are fully masked by their own mask rows; q on H does not involve s_j (p_j = W_j on H)"},
    }
