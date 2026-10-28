"""orbit_gpu's host side on the CPU: B's carries folded into s8orbit_bfix's lists (G over the pair rows), read the way the
kernel reads them, plus the wrapped forms' orbit, are X·Wᵀ exactly where A's forms don't carry, and with `a_terms` where
they do."""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import orbit_gpu as G  # noqa: E402

SHAPES = ((4, 8, 1024), (6, 12, 2048), (10, 16, 3072))


def wrapped_orbit(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    wa, _ = G.forms(x, G.A_FORMS, True)
    wb, _ = G.forms(w, G.B_FORMS, False)
    return G.assemble([(wa[j].double() @ wb[j].double().t()).long() for j in range(7)])


def bfix(x: torch.Tensor, n: int, ptr: torch.Tensor, ent: torch.Tensor) -> torch.Tensor:
    """256 X~·Gᵀ from the lists, a chunk at a time, as s8orbit_bfix reads them."""
    m, k = x.shape
    hm, hn, h = m // 2, n // 2, k // 1024
    pair = torch.cat([x[:hm], x[hm:]], 1).long()
    o, g = (ent >> 8).long(), (ent & 0xFF).to(torch.uint8).view(torch.int8).long()
    ptr = ptr.view(4 * hn, h + 1)
    out = torch.zeros((m, n), dtype=torch.int64)
    for li in range(4 * hn):
        q, c = divmod(li, hn)
        for ch in range(h):
            b, e = int(ptr[li, ch]), int(ptr[li, ch + 1])
            assert ((o[b:e] >= G.FCHUNK * ch) & (o[b:e] < G.FCHUNK * (ch + 1))).all()
            col = (pair[:, o[b:e]] * g[b:e]).sum(1)
            out[(q >> 1) * hm:(q >> 1) * hm + hm, (q & 1) * hn + c] += 256 * col
    return out


@pytest.mark.parametrize("m,n,k", SHAPES)
def test_b_carries_exact(m: int, n: int, k: int) -> None:
    gen = torch.Generator().manual_seed(m * n + k)
    x = torch.randint(-63, 64, (m, k), generator=gen, dtype=torch.int8)
    w = torch.randint(-127, 128, (n, k), generator=gen, dtype=torch.int8)
    _, cb = G.forms(w, G.B_FORMS, False)
    ptr, ent, entries = G.fix_lists(cb)
    assert entries == ent.numel() > 0
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
