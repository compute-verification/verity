"""``chain.chain_coefs`` (one index_add over a cached term table, lane hostphase) equals the per-term reference (main's
loop, kept here verbatim) residue for residue -- on a single-``y16`` chain (bf16-hopper) and on a component-ended chain
(the FP4 relation: ``y_end`` in three bit fields under families 6, 0, 1).  CPU; torch required."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from .chain import Layout, chain_coefs, end_terms, end_values  # noqa: E402
from .compile import P  # noqa: E402


def _reference(sys, layout, rc, y16_pub):
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
        r_link = (rc[:, x, :] * link[None, :]) % P
        r_start = (rc[:, 3 + x, :] * start[None, :]) % P
        ci = involved[c_idx]
        coef[:, ci, 1:] = (coef[:, ci, 1:] + r_link[:, :-1]) % P
        for idx, kappa in Y.lin.items():
            coef[:, involved[idx], :] = (coef[:, involved[idx], :] + r_link * ((-kappa) % P)) % P
        rhs = (rhs + (r_link.sum(1) % P) * (Y.c % P)) % P
        coef[:, ci, :] = (coef[:, ci, :] + r_start) % P
    for (Y, fam, _, _), yv in zip(ends, y_vals):
        r_end = (rc[:, fam, :] * end[None, :]) % P
        for idx, kappa in Y.lin.items():
            coef[:, involved[idx], :] = (coef[:, involved[idx], :] + r_end * (kappa % P)) % P
        rhs = (rhs + ((r_end * ((yv - Y.c) % P)[None, :]) % P).sum(1)) % P
    return rows, coef, rhs


def _check(sys, steps, n_vus, l, y_bits, seed=3):
    g = torch.Generator().manual_seed(seed)
    lay = Layout(l, n_vus, steps)
    rc = torch.randint(0, P, (6, 7, l), dtype=torch.int64, generator=g)
    y = torch.zeros(l, dtype=torch.int64)
    for v in range(n_vus):
        y[steps * v + steps - 1] = int(torch.randint(0, 2 ** y_bits, (1,), generator=g))
    rows, coef, rhs = _reference(sys, lay, rc, y)
    got = chain_coefs(sys, lay, rc, y)
    assert torch.equal(got.rows, rows)
    assert torch.equal(got.coef, coef)
    assert torch.equal(got.rhs, rhs)


def test_single_word_end_bf16_hopper():
    from .relations import relation
    rel = relation("bf16-hopper")
    sys = rel.compile(rel.params, True)
    assert sys.chain.get("y16") is not None
    _check(sys, 96, 2, 256, 16)


def test_component_ends_fp4():
    from .fp4.relation import NVFP4, STEPS, compile_fp4_unit
    sys = compile_fp4_unit(NVFP4, chain=True)
    assert sys.chain.get("y16") is None and len(sys.chain["y_end"]) == 3
    _check(sys, STEPS, 2, 64, 32)
    _check(sys, STEPS, 5, 128, 32, seed=4)
