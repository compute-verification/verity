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
    tabs = _chain_tables(sys, layout, device)
    rows, T = tabs["rows"], tabs["terms"]
    # rm[:, f, :] = the family-f coefficients masked to their constraint positions (link x3, start x3, end)
    rm = (rc * tabs["maskmat"][None, :, :]) % P                                    # (D, 7, l)
    # every (row, family, kappa) term of r^T A at once: coef[:, row, j] += kappa rc[f][j] (masked); the link terms'
    # W[c][j + 1] coefficient is the same sum shifted one column right.  Exact integer sums of < 2^31 residues (a few
    # hundred terms per entry: no overflow), reduced once -- the same residues as the per-term reduction, in ~10 kernels
    # instead of one per term (lane hostphase).
    coef = torch.zeros((D, len(rows), l), dtype=torch.int64, device=device)
    coef.index_add_(1, T["row"], (rm.index_select(1, T["fam"]) * T["kappa"][None, :, None]) % P)
    coef[:, :, 1:].index_add_(1, T["shift_row"], rm[:, 0:3, :-1])
    coef %= P
    rhs = torch.zeros((D,), dtype=torch.int64, device=device)
    y16 = y16_pub.to(device) % P
    for x in range(3):
        Y = ch["y"][x]
        r_link = rm[:, x, :]                                                       # zero off the linked positions
        rhs = (rhs + (r_link.sum(1) % P) * (Y.c % P)) % P                          # b = +Y.c
    # end_j: sum_i kappa_i W[i][j] + Y16.c = y16_pub[j]
    Y16 = ch["y16"]
    r_end = rm[:, 6, :]
    rhs = (rhs + ((r_end * ((y16 - Y16.c) % P)[None, :]) % P).sum(1)) % P
    return ChainCoefs(rows, coef, rhs)


_TABLES: dict = {}


def _chain_tables(sys: System, layout: Layout, device) -> dict:
    """The constant part of ``chain_coefs`` for (system, layout, device): the involved rows, the (row, family, kappa)
    term table, the shifted link rows and the (7, l) family mask matrix."""
    key = (id(sys), layout.l, layout.n_vus, layout.steps, str(device))
    hit = _TABLES.get(key)
    if hit is not None and hit[0] is sys:
        return hit[1]
    ch = sys.chain
    involved: dict[int, int] = {}
    for idx in ch["c"]:
        involved.setdefault(idx, len(involved))
    for e in ch["y"] + [ch["y16"]]:
        for idx in e.lin:
            involved.setdefault(idx, len(involved))
    row, fam, kappa, shift_row = [], [], [], []
    for x in range(3):
        c_idx, Y = ch["c"][x], ch["y"][x]
        ci = involved[c_idx]
        shift_row.append(ci)                                   # link_j: + W[c][j+1]  (family x, shifted)
        for idx, kap in Y.lin.items():                         #         - sum_i kappa_i W[i][j]
            row.append(involved[idx]); fam.append(x); kappa.append((-kap) % P)
        row.append(ci); fam.append(3 + x); kappa.append(1)     # start_j: W[c][j] = 0
    for idx, kap in ch["y16"].lin.items():                     # end_j: sum_i kappa_i W[i][j] + Y16.c = y16_pub[j]
        row.append(involved[idx]); fam.append(6); kappa.append(kap % P)
    start, link, end = layout.masks(device)
    maskmat = torch.stack([link, link, link, start, start, start, end])           # (7, l) 0/1
    t = lambda v: torch.tensor(v, dtype=torch.int64, device=device)  # noqa: E731
    tabs = {"rows": t(list(involved.keys())), "maskmat": maskmat,
            "terms": {"row": t(row), "fam": t(fam), "kappa": t(kappa), "shift_row": t(shift_row)}}
    _TABLES[key] = (sys, tabs)
    return tabs
