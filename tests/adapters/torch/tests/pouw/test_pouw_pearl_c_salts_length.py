"""The adapter's two callers that hand A's salts to the Pearl-C kernels by address state the buffer's length, CPU only, on
the sm_90a kernels' `run.py` with no device: a buffer a byte short of its rows' salts is refused by name (`salts_at`)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from verity.protocols.resource_accounting.compute.pouw.prover.kernels.pearl_c_sm90 import run as R
from verity_torch.pouw.pearl_c import grouped as PG
from verity_torch.pouw.pearl_c.graphs import GraphCalls

torch = pytest.importorskip("torch")


def test_a_graphed_calls_salts_short_of_its_rows_are_refused():
    run = SimpleNamespace(kernel=SimpleNamespace(run=R))
    sh = SimpleNamespace(m16=16, buf=lambda name: torch.zeros(R.HM_SALT * 16, dtype=torch.uint8))
    assert GraphCalls.salts_a(run, sh, None).numel() == R.HM_SALT * 16
    short = SimpleNamespace(salts_a=torch.zeros(R.HM_SALT * 16 - 1, dtype=torch.uint8))
    with pytest.raises(ValueError, match=rf"^salts_a: {R.HM_SALT * 16 - 1} bytes for 16 leaves"):
        GraphCalls.salts_a(run, sh, short)


def test_a_grouped_calls_salts_short_of_its_rows_are_refused_before_its_kernels_take_them(monkeypatch):
    Rs = SimpleNamespace(PD_MAX_ROWS=64, BM=16, HM_SALT=R.HM_SALT, pow2=R.pow2, salts_at=R.salts_at,
                         Pipeline=SimpleNamespace(sizes=lambda bm, n, k, bn: {}))
    G = SimpleNamespace(STACKED=(), SEG=16, sizes=lambda R_, rows, count, n, k: {x: 8 for x in PG.ROTATED},
                        Grouped=lambda *a: pytest.fail("the kernels took a short salts buffer"))
    run = SimpleNamespace(entry="e", weights={0: SimpleNamespace(n=64, k=256, b={})}, kernel=SimpleNamespace(run=Rs),
                          device=torch.device("cpu"), driver=None, nb=64, join_sides=lambda: None)
    g = PG._Group(run, G, (0,))
    zeros = PG._zeros
    monkeypatch.setattr(PG, "_zeros", lambda s, d: zeros(s - (s == R.HM_SALT * PG.MIN_ROWS), d))
    with pytest.raises(ValueError, match=rf"^salts_a: {R.HM_SALT * PG.MIN_ROWS - 1} bytes for {PG.MIN_ROWS} leaves"):
        g.fit(run, 1)
