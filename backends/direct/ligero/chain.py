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

Chain ends in components (``sys.chain["y_end"]``, the FP4 relation).  A public word of more than 30 bits cannot be
compared to a single field element (2^32 > p: the word w and w + p are the same residue), so a relation whose claimed
word is a full FP32 may end its chain on up to three bounded COMPONENTS instead: ``y_end = [(Y_x, lo_x, bits_x)]``,
one constraint ``Y_x(W)[j] = (y_pub[j] >> lo_x) & (2^bits_x - 1)`` per component at every end column j.  Each
component is < 2^31 < p, so equality mod p is integer equality once the relation range-constrains the row.  The
coefficients: component 0 uses the end family (6) and components 1, 2 the link families (0, 1), which are unused at
end columns (an end column has no successor in its VU: the link mask is 0 there), so every constraint still has its
own uniform coefficient and nothing is sampled beyond the seven families.  Single-``y16`` systems (BF16, FP8) keep
their transcripts byte for byte.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .compile import P, System

__all__ = ["N_FAMILIES", "END_FAMILIES", "ChainCoefs", "Layout", "chain_coefs", "end_terms"]

N_FAMILIES = 7        # link x3, start x3, end
END_FAMILIES = (6, 0, 1)   # coefficient family of end component x (see the module comment)


def end_terms(ch: dict) -> list[tuple[object, int, int, int]]:
    """The chain-end constraints of a compiled chain as ``(Expr, family, lo, bits)``: one ``(y16, 6, 0, 0)`` for a
    single-word end (``bits = 0``: the whole public word, compared as a residue), else one per component."""
    if ch.get("y16") is not None:
        return [(ch["y16"], END_FAMILIES[0], 0, 0)]
    comps = ch["y_end"]
    if not 1 <= len(comps) <= len(END_FAMILIES):
        raise ValueError(f"a chain end has 1..{len(END_FAMILIES)} components, got {len(comps)}")
    return [(Y, END_FAMILIES[x], int(lo), int(bits)) for x, (Y, lo, bits) in enumerate(comps)]


def end_values(ch: dict, y_pub: torch.Tensor) -> list[torch.Tensor]:
    """The public value each end constraint compares against, per column: ``y_pub`` (l,) as a residue for a
    single-word end, else the bit field ``(y_pub >> lo) & (2^bits - 1)`` of every component."""
    out = []
    for _, _, lo, bits in end_terms(ch):
        out.append(y_pub % P if bits == 0 else (y_pub >> lo) & (2 ** bits - 1))
    return out


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
    ends = end_terms(ch)
    involved: dict[int, int] = {}
    for idx in ch["c"]:
        involved.setdefault(idx, len(involved))
    for e in ch["y"] + [Y for Y, _, _, _ in ends]:
        for idx in e.lin:
            involved.setdefault(idx, len(involved))
    rows = torch.tensor(list(involved.keys()), dtype=torch.int64, device=device)
    coef = torch.zeros((D, len(involved), l), dtype=torch.int64, device=device)
    rhs = torch.zeros((D,), dtype=torch.int64, device=device)
    y_vals = end_values(ch, y16_pub.to(device))
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
    # end_j: sum_i kappa_i W[i][j] + Y.c = y_pub[j] (one constraint per end component)
    for (Y, fam, _, _), yv in zip(ends, y_vals):
        r_end = (rc[:, fam, :] * end[None, :]) % P
        for idx, kappa in Y.lin.items():
            coef[:, involved[idx], :] = (coef[:, involved[idx], :] + r_end * (kappa % P)) % P
        rhs = (rhs + ((r_end * ((yv - Y.c) % P)[None, :]) % P).sum(1)) % P
    return ChainCoefs(rows, coef, rhs)
