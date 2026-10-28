"""orbit_gpu's host side on the CPU: B's carries folded into s8orbit_bfix's lists (G over the pair rows), read the way the
kernel reads them, plus the wrapped forms' orbit, are X·Wᵀ exactly where A's forms don't carry, and with `a_terms` where
they do; s8orbit_pfix's lists give the same sums mod 4."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orbit_gpu as G  # noqa: E402

SHAPES = ((4, 8, 1024), (6, 12, 2048), (10, 16, 3072), (6, 40, 256), (34, 72, 192))


def wrapped_orbit(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    wa, _ = G.forms(x, G.A_FORMS, True)
    wb, _ = G.forms(w, G.B_FORMS, False)
    return G.assemble([(wa[j].double() @ wb[j].double().t()).long() for j in range(7)])


def list_sums(x: torch.Tensor, n: int, ptr: torch.Tensor, ent: torch.Tensor) -> torch.Tensor:
    """X~·Gᵀ [M/2, 2N] from the lists as the kernels read them: group u's lane i takes ent[32 e + i], e in [ptr[u],
    ptr[u + 1]), each entry o << 13 | i' << 8 | g of list 32 u + i', its runs as even as 32 lanes allow."""
    m, k = x.shape
    lists = 2 * n
    pair = G.pair_rows(x).long()
    assert ptr.numel() == G.cdiv(lists, 32) + 1 and bool((torch.diff(ptr) % 4 == 0).all())
    s = torch.zeros((m // 2, lists), dtype=torch.int64)
    for u in range(ptr.numel() - 1):
        v = ent[32 * int(ptr[u]):32 * int(ptr[u + 1])].long().view(-1, 32)
        o, li, g = v >> 13, (v >> 8) & 31, (v & 0xFF).to(torch.uint8).view(torch.int8).long()
        real = g != 0
        most = int(real.sum(0).max())
        assert most <= v.shape[0] < most + 4
        assert bool((o[~real] == 0).all()) and bool((o < 2 * k).all())
        l = 32 * u + li[real]
        assert bool((l < lists).all())
        s.index_add_(1, l, pair[:, o[real]] * g[real])
    return s


def bfix(x: torch.Tensor, n: int, ptr: torch.Tensor, ent: torch.Tensor) -> torch.Tensor:
    """256 X~·Gᵀ assembled into O's quadrants, list q (N/2) + c to column c of quadrant q."""
    m = x.shape[0]
    hm, hn = m // 2, n // 2
    s = list_sums(x, n, ptr, ent)
    out = torch.zeros((m, n), dtype=torch.int64)
    for q in range(4):
        out[(q >> 1) * hm:(q >> 1) * hm + hm, (q & 1) * hn:(q & 1) * hn + hn] += 256 * s[:, q * hn:(q + 1) * hn]
    return out


@pytest.mark.parametrize("m,n,k", SHAPES)
def test_b_carries_exact(m: int, n: int, k: int) -> None:
    gen = torch.Generator().manual_seed(m * n + k)
    x = torch.randint(-63, 64, (m, k), generator=gen, dtype=torch.int8)
    w = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8)
    _, cb = G.forms(w, G.B_FORMS, False)
    ptr, ent, entries = G.fix_lists(cb)
    assert entries == int(torch.count_nonzero(ent & 0xFF)) > 0
    got = wrapped_orbit(x, w) + bfix(x, n, ptr, ent)
    assert torch.equal(got, x.long() @ w.long().t())


@pytest.mark.parametrize("m,n,k", SHAPES)
def test_both_sides_exact(m: int, n: int, k: int) -> None:
    gen = torch.Generator().manual_seed(m + n * k)
    x = torch.randint(-128, 128, (m, k), generator=gen, dtype=torch.int8)
    w = torch.randint(-128, 128, (n, k), generator=gen, dtype=torch.int8)
    wb, cb = G.forms(w, G.B_FORMS, False)
    ptr, ent, _ = G.fix_lists(cb)
    got = wrapped_orbit(x, w) + bfix(x, n, ptr, ent) + G.a_terms(x, wb)
    assert torch.equal(got, x.long() @ w.long().t())


def test_no_carries_no_entries() -> None:
    w = torch.randint(-63, 64, (8, 1024), generator=torch.Generator().manual_seed(5), dtype=torch.int8)
    _, cb = G.forms(w, G.B_FORMS, False)
    ptr, ent, entries = G.fix_lists(cb)
    assert entries == 0 and int(ptr.abs().sum()) == 0 and ent.numel() == 1


@pytest.mark.parametrize("m,n,k", SHAPES)
def test_pack_lists_mod4(m: int, n: int, k: int) -> None:
    gen = torch.Generator().manual_seed(3 * m + n + k)
    x = torch.randint(-128, 128, (m, k), generator=gen, dtype=torch.int8)
    w = torch.randint(-128, 128, (n, k), generator=gen, dtype=torch.int8)
    _, cb = G.forms(w, G.B_FORMS, False)
    ptr, ent, entries = G.fix_lists(cb)
    pptr, pent, pentries = G.pack_lists(cb)
    g = (ent & 0xFF).to(torch.uint8).view(torch.int8)
    pg = pent & 0xFF
    assert pentries == int(((g != 0) & ((g & 3) != 0)).sum()) == int((pg != 0).sum()) and int(pg.max()) <= 3
    assert torch.equal(list_sums(x, n, pptr, pent) & 3, list_sums(x, n, ptr, ent) & 3)


@pytest.mark.parametrize("m,n,k", SHAPES)
def test_lists_dense(m: int, n: int, k: int) -> None:
    w = torch.randint(-128, 128, (n, k), generator=torch.Generator().manual_seed(n + k), dtype=torch.int8)
    _, cb = G.forms(w, G.B_FORMS, False)
    key, gv, lists, k2 = G.fix_terms(cb)
    want = torch.zeros((lists, k2), dtype=torch.int8)
    want.view(-1)[key] = gv.to(torch.int8)
    ptr, ent = G.chunk_lists(key, gv, lists, k2)
    assert torch.equal(G.lists_dense(ptr, ent, lists, k2), want)


def test_unpack() -> None:
    out = torch.randint(-2**31, 2**31, (4, 6, 2), generator=torch.Generator().manual_seed(1), dtype=torch.int64).to(torch.int32)
    s = G.unpack(out, 100)
    r, l = torch.meshgrid(torch.arange(100), torch.arange(6), indexing="ij")
    w = out.long()[r // 32, l]
    assert torch.equal(s, ((w[..., 0] >> (r % 32)) & 1) + 2 * ((w[..., 1] >> (r % 32)) & 1))


def test_fix_rows() -> None:
    assert G.fix_rows(4096) == 16 and G.fix_rows(14336) == 8 and G.fix_rows(16384) is None
    assert G.pack_words(4096) == 2 and G.pack_words(14336) == 1 and G.pack_words(16384) is None
    assert G.pfix_smem(14336, 1) <= G.SMEM_LIMIT
