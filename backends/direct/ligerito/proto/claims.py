"""Tensor claims and the partial sumcheck over the column variables (host side, numpy).

A *claim* on a committed vector y in F^{2^m} (m = k + k', column-major matrix M~ with 2^k rows, 2^{k'} columns:
x = row + 2^k col, so the column bits are the HIGH k' variables) is  < sum_t w_t , y > = v  where every term
w_t = coef_t . (x)_j (a_tj, b_tj)  is a tensor over the m variables:

* ``eq`` term (the evaluation claim and its descendants): pairs (1 - z_j, z_j), coef in F_{p^6};
* ``geom`` term (an opened codeword row s of the previous round): pairs (1, eta_s^{2^j}), i.e. w[x] = coef . eta_s^x,
  the row of the coefficient-basis Reed-Solomon generator; eta_s is a BASE element, so the row part
  (eta_s^{row})_{row < 2^k} is a base vector and the prover's contraction u_t = M~^T rowpart_t is base x ext.

Partial sumcheck over the k' column variables, highest bit first (paper section 4): the prover holds, per term,
u_t = M~^T rowpart_t (2^{k'} extension values) and colpart_t (2^{k'} tensor values); round j sends
g_j(X) = sum_t coef_t sum_x A_t(X, x) B_t(X, x) at X = 0, 1, 2 (degree 2), receives r_j, folds both tables.
After k' rounds the claim on y' = M~ r-bar is  sum_t coef_t colpart_t(r) < rowpart_t , y' > = g_{k'}(r_{k'}),
i.e. the same terms with coef_t *= colpart_t(r) and the k' column variables removed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import ext
from .ext import D, P

INV2 = pow(2, P - 2, P)


@dataclass
class Term:
    coef: np.ndarray                 # (6,)
    kind: str                        # "eq" | "geom"
    z: np.ndarray | None = None      # eq: (nvars, 6)
    eta: int = 0                     # geom: base element
    nvars: int = 0

    def pair(self, j: int) -> np.ndarray:
        """(2, 6): the factor (a_j, b_j) of variable j (bit j of the index)."""
        if self.kind == "eq":
            return ext.eq_pairs(self.z[j:j + 1])[0]
        out = ext.zero((2,))
        out[0, 0] = 1
        out[1, 0] = pow(self.eta, 1 << j, P)
        return out

    def colpart(self, k: int, kp: int) -> np.ndarray:
        """(2^{kp}, 6): the tensor over variables k .. k + kp - 1 (the column index bits, bit 0 = variable k)."""
        vec = ext.one((1,))
        for j in range(kp):
            pr = self.pair(k + j)                       # (2, 6)
            lo = ext.mul(vec, pr[0])
            hi = ext.mul(vec, pr[1])
            vec = np.concatenate((lo, hi), axis=0)      # bit j of the column index is the NEW high bit
        return vec

    def colpart_at(self, k: int, r: np.ndarray) -> np.ndarray:
        """(6,): prod_j (a_{k+j} (1 - r_j) + b_{k+j} r_j) for the challenges r (kp, 6), r_j <-> variable k + j."""
        acc = ext.one()
        for j in range(r.shape[0]):
            pr = self.pair(k + j)
            one_minus = (ext.one() - r[j]) % P
            acc = ext.mul(acc, ext.add(ext.mul(pr[0], one_minus), ext.mul(pr[1], r[j])))
        return acc

    def restrict(self, k: int, r: np.ndarray, alpha=None) -> "Term":
        """The term after the k' = r.shape[0] top variables are fixed at r: coef *= colpart(r) [* alpha]."""
        coef = ext.mul(self.coef, self.colpart_at(k, r))
        if alpha is not None:
            coef = ext.mul(coef, alpha)
        if self.kind == "eq":
            return Term(coef, "eq", z=self.z[:k], nvars=k)
        return Term(coef, "geom", eta=self.eta, nvars=k)

    def rowpart_base(self, k: int) -> np.ndarray:
        """geom only: (2^k,) base vector eta^{row}."""
        assert self.kind == "geom"
        return powers(self.eta, 1 << k)

    def rowpart_eq(self, k: int) -> np.ndarray:
        """eq only: (2^k, 6) the tensor of the low k variables."""
        assert self.kind == "eq"
        return eq_vector(self.z[:k])


def powers(eta: int, count: int) -> np.ndarray:
    out = np.empty(count, dtype=np.int64)
    out[0] = 1
    step, cur = 1, eta % P
    while step < count:
        m = min(step, count - step)
        out[step:step + m] = (out[:m] * cur) % P
        step *= 2
        cur = cur * cur % P
    return out


def eq_vector(z: np.ndarray) -> np.ndarray:
    """(2^nvars, 6): eq(x, z) over x in {0,1}^nvars, bit j of x <-> z_j (numpy; for the small sizes)."""
    vec = ext.one((1,))
    pairs = ext.eq_pairs(z)
    for j in range(z.shape[0]):
        vec = np.concatenate((ext.mul(vec, pairs[j, 0]), ext.mul(vec, pairs[j, 1])), axis=0)
    return vec


def interpolate_deg2(g0, g1, g2, r):
    """Degree-2 polynomial from its values at 0, 1, 2, evaluated at the extension point r (all (6,))."""
    c2 = ext.mul_base(ext.sub(ext.add(g2, g0), ext.mul_base(g1, 2)), INV2)
    c1 = ext.sub(ext.sub(g1, g0), c2)
    return ext.add(g0, ext.mul(r, ext.add(c1, ext.mul(c2, r))))


class ColumnSumcheckProver:
    """Runs the k' column rounds for a list of terms with their contractions u_t ((2^{k'}, 6) each)."""

    def __init__(self, terms: list[Term], us: list[np.ndarray], k: int, kp: int):
        assert len(terms) == len(us)
        self.terms, self.k, self.kp = terms, k, kp
        self.A = np.stack(us, axis=0) % P                                      # (T, C, 6)
        self.B = np.stack([t.colpart(k, kp) for t in terms], axis=0)           # (T, C, 6)
        self.coef = np.stack([t.coef for t in terms], axis=0)                  # (T, 6)
        self.rs: list[np.ndarray] = []

    def claim(self) -> np.ndarray:
        """sum_t coef_t < A_t, B_t > -- the current running claim (6,)."""
        return ext.sum_ext(ext.mul(self.coef, ext.sum_ext(ext.mul(self.A, self.B), axis=1)), axis=0)

    def round_message(self) -> np.ndarray:
        """(3, 6): g(0), g(1), g(2) for the current top variable."""
        C = self.A.shape[1]
        h = C // 2
        Alo, Ahi = self.A[:, :h], self.A[:, h:]
        Blo, Bhi = self.B[:, :h], self.B[:, h:]
        A2 = (2 * Ahi - Alo) % P
        B2 = (2 * Bhi - Blo) % P
        out = []
        for a, b in ((Alo, Blo), (Ahi, Bhi), (A2, B2)):
            per_t = ext.sum_ext(ext.mul(a, b), axis=1)                          # (T, 6)
            out.append(ext.sum_ext(ext.mul(self.coef, per_t), axis=0))
        return np.stack(out, axis=0)

    def fold(self, r: np.ndarray) -> None:
        C = self.A.shape[1]
        h = C // 2
        one_minus = (ext.one() - r) % P
        self.A = ext.add(ext.mul(self.A[:, :h], one_minus), ext.mul(self.A[:, h:], r))
        self.B = ext.add(ext.mul(self.B[:, :h], one_minus), ext.mul(self.B[:, h:], r))
        self.rs.append(r)

    def challenges(self) -> np.ndarray:
        """(kp, 6) with r[j] <-> variable k + j (the rounds go highest bit first, so reverse the draw order)."""
        return np.stack(self.rs[::-1], axis=0)


def rbar(r: np.ndarray) -> np.ndarray:
    """(2^{kp}, 6): the tensor prod_j ((1 - r_j), r_j) over the column index (bit j <-> r[j])."""
    return eq_vector(r)


def verify_column_rounds(claim: np.ndarray, messages: list[np.ndarray], rs: list[np.ndarray]) -> tuple[bool, np.ndarray]:
    """Check g_j(0) + g_j(1) == claim, claim <- g_j(r_j).  ``rs`` in draw order (highest bit first)."""
    cur = claim % P
    for g, r in zip(messages, rs):
        if not np.array_equal(ext.add(g[0], g[1]), cur):
            return False, cur
        cur = interpolate_deg2(g[0], g[1], g[2], r)
    return True, cur
