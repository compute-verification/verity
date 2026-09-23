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


def test_check_component_is_the_rust_rule():
    """1 <= bits <= 30, lo + bits <= 32 (ligero-verify's parser): the boundaries, both sides."""
    from .chain import MAX_COMPONENT_BITS, WORD_BITS, check_component
    assert (MAX_COMPONENT_BITS, WORD_BITS) == (30, 32) and 2 ** 30 < P < 2 ** 31
    for lo, bits in ((0, 30), (2, 30), (31, 1), (0, 1), (23, 8), (0, 23), (1, 31 - 1)):
        assert check_component(0, lo, bits) == (lo, bits)
    for lo, bits in ((0, 31), (3, 30), (32, 1), (-1, 1), (0, 0), (31, 2), (0, 32)):
        with pytest.raises(ValueError, match="chain end component 1"):
            check_component(1, lo, bits)


def test_component_ends_fp4():
    from .fp4.relation import NVFP4, STEPS, compile_fp4_unit
    sys = compile_fp4_unit(NVFP4, chain=True)
    assert sys.chain.get("y16") is None and len(sys.chain["y_end"]) == 3
    _check(sys, STEPS, 2, 64, 32)
    _check(sys, STEPS, 5, 128, 32, seed=4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")
def test_fused_fp_lanes_and_chain_quotient_are_the_torch_sums() -> None:
    """tests_fused.fp_lanes / chain_quotient equal the torch expressions of chain_coefs / protocol._tests_compute residue
    for residue, at P - 1 everywhere and at random, C = 8 and 10, int32 and int64 codewords."""
    from . import tests_fused
    D, dev = 6, torch.device("cuda")
    ft = tests_fused.fused_for(D, dev)
    assert ft is not None
    g = torch.Generator().manual_seed(11)
    for C, n_op, n_slots, steps, extreme in ((8, 2, 341, 48, False), (10, 2, 85, 48, True), (8, 1, 18, 48, False)):
        R, l = 50, n_slots * steps + 7
        u = torch.randint(0, P, (D, n_op, C, n_slots), generator=g, dtype=torch.int64)
        pw = torch.randint(0, P, (C, steps, 16), generator=g, dtype=torch.int64)
        coef = torch.randint(0, P, (D, R, l), generator=g, dtype=torch.int64)
        if extreme:
            u.fill_(P - 1); pw.fill_(P - 1); coef.fill_(P - 1)
        lane_pos = torch.randperm(R, generator=g)[:n_op * 16]
        prod = (u[:, :, :, :, None, None] * pw[None, None, :, None, :, :]) % P
        lane = (prod.sum(2) % P).permute(0, 1, 4, 2, 3).reshape(D, n_op * 16, n_slots * steps)
        ref = coef.clone()
        ref[:, lane_pos, :n_slots * steps] = (ref[:, lane_pos, :n_slots * steps] + lane) % P
        got = coef.to(dev)
        ft.fp_lanes(u.to(dev), pw.to(dev), lane_pos.to(dev), got, steps)
        assert torch.equal(got.cpu(), ref)
    for R, M, n, dt, extreme in ((42, 60, 4096, torch.int32, False), (5, 9, 1000, torch.int64, False), (17, 17, 512, torch.int32, True)):
        Renc = torch.randint(0, P, (D, R, n), generator=g, dtype=torch.int64)
        U = torch.randint(0, P, (M, n), generator=g, dtype=torch.int64)
        if extreme:
            Renc.fill_(P - 1); U.fill_(P - 1)
        rows = torch.randint(0, M, (R,), generator=g)
        ref = ((Renc * U.index_select(0, rows)[None]) % P).sum(1) % P
        got = ft.chain_quotient(Renc.to(dt).to(dev), U.to(torch.int32).to(dev), rows.to(dev))
        assert torch.equal(got.cpu(), ref)
