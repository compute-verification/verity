"""The chain test (PROTOCOL.md section "Chain"): Ligero's *general* linear test for the cross-column constraints of
the verification unit.  Columns j = 96 v + i hold unit i of VU v; the constraints

    link_j  : W[c.x][j + 1] - Y_x(W)[j] = 0     x in (s, t, f), for every j whose successor is in the same VU
    start_j : W[c.x][j] = 0                     for every j that starts a VU (and every pad column)
    end_j   : Y16(W)[j] = y16_pub[j]            for every j that ends a VU

are one linear system A w = b over the flattened witness.  The verifier draws one uniform coefficient per constraint
(D coordinates); r^T A is a coefficient vector per (row, column), interpolated per row to R_i(X) (degree < l); the
prover sends q(X) = sum_i R_i(X) p_i(X) (degree < 2l - 1); the verifier checks sum_{H} q = r^T b, i.e.
l (q_0 + q_l) = r^T b, and q(eta_c) = sum_i R_i(eta_c) U_i[c] at the opened columns.  Nothing is rotated or opened
beyond the t columns; the soundness term is the linear test's ((k + l)/n)^t with k = l (Appendix C).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .compile import P, System

__all__ = ["N_FAMILIES", "ChainCoefs", "Layout", "chain_coefs"]

N_FAMILIES = 7        # link x3, start x3, end


@dataclass(frozen=True)
class Layout:
    """l columns; VU v occupies columns [steps v, steps (v + 1)); columns >= steps * n_vus are pad units (a start
    with no end: c_in = 0, operands 0, output unchecked)."""
    l: int
    n_vus: int
    steps: int

    def __post_init__(self):
        if self.steps * self.n_vus > self.l:
            raise ValueError("layout does not fit")

    def masks(self, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        j = torch.arange(self.l, device=device)
        in_vu = j < self.steps * self.n_vus
        i = j % self.steps
        start = (~in_vu) | (i == 0)
        end = in_vu & (i == self.steps - 1)
        link = in_vu & (i != self.steps - 1)                 # j + 1 is in the same VU
        return start.to(torch.int64), link.to(torch.int64), end.to(torch.int64)


@dataclass
class ChainCoefs:
    rows: torch.Tensor          # (R,) involved witness row indices
    coef: torch.Tensor          # (D, R, l) coefficient of W[rows[r]][j] in the combination
    rhs: torch.Tensor           # (D,) r^T b


def chain_coefs(sys: System, layout: Layout, rc: torch.Tensor, y16_pub: torch.Tensor) -> ChainCoefs:
    """``rc``: (D, N_FAMILIES, l) uniform challenge coefficients (one per constraint; entries at masked-out
    positions are ignored); ``y16_pub``: (l,) the claimed BF16 words at the chain ends (0 elsewhere)."""
    ch = sys.chain
    device = rc.device
    D, _, l = rc.shape
    start, link, end = layout.masks(device)
    involved: dict[int, int] = {}
    for idx in ch["c"]:
        involved.setdefault(idx, len(involved))
    for e in ch["y"] + [ch["y16"]]:
        for idx in e.lin:
            involved.setdefault(idx, len(involved))
    rows = torch.tensor(list(involved.keys()), dtype=torch.int64, device=device)
    coef = torch.zeros((D, len(involved), l), dtype=torch.int64, device=device)
    rhs = torch.zeros((D,), dtype=torch.int64, device=device)
    y16 = y16_pub.to(device) % P
    for x in range(3):
        c_idx, Y = ch["c"][x], ch["y"][x]
        r_link = (rc[:, x, :] * link[None, :]) % P            # (D, l), zero off the linked positions
        r_start = (rc[:, 3 + x, :] * start[None, :]) % P
        # link_j: + W[c][j+1] - sum_i kappa_i W[i][j] - Y.c = 0
        ci = involved[c_idx]
        coef[:, ci, 1:] = (coef[:, ci, 1:] + r_link[:, :-1]) % P
        for idx, kappa in Y.lin.items():
            coef[:, involved[idx], :] = (coef[:, involved[idx], :] + r_link * ((-kappa) % P)) % P
        rhs = (rhs + (r_link.sum(1) % P) * (Y.c % P)) % P                    # b = +Y.c
        # start_j: W[c][j] = 0
        coef[:, ci, :] = (coef[:, ci, :] + r_start) % P
    # end_j: sum_i kappa_i W[i][j] + Y16.c = y16_pub[j]
    Y16 = ch["y16"]
    r_end = (rc[:, 6, :] * end[None, :]) % P
    for idx, kappa in Y16.lin.items():
        coef[:, involved[idx], :] = (coef[:, involved[idx], :] + r_end * (kappa % P)) % P
    rhs = (rhs + ((r_end * ((y16 - Y16.c) % P)[None, :]) % P).sum(1)) % P
    return ChainCoefs(rows, coef, rhs)
