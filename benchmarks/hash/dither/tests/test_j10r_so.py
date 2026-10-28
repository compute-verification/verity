"""j10r_so.Refs: the gate's J10-R reference over a capture in host memory (a prefill gate's) and B on the device."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

DITHER = Path(__file__).resolve().parents[1]


def _module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    return m


@pytest.fixture
def JS(monkeypatch):
    """j10r_so with dither_a, dither_one and h1r stood in: `Refs.receipts` reads only the window, the capture and
    j10r_units (stood in by each test)."""
    monkeypatch.syspath_prepend(str(DITHER))
    for m in (_module("dither_a"), _module("dither_one", h1v3_itself=None), _module("h1r", H1R=type("H1R", (), {}))):
        monkeypatch.setitem(sys.modules, m.__name__, m)
    monkeypatch.delitem(sys.modules, "j10r_so", raising=False)
    import j10r_so
    return j10r_so


def test_receipts_take_a_host_capture_to_b_device(JS, monkeypatch):
    """B on another device than the host-memory capture (`meta` standing in for cuda:0): call_receipts gets A on B's
    device, and a skipped dead row tile's units the zero receipt."""
    seen = []

    def call_receipts(a, b, salt, base, root_a, root_b, only):
        if a.device != b.device:
            raise RuntimeError(f"Expected all tensors to be on the same device, {b.device} and {a.device}")
        seen.append((tuple(a.shape), a.device.type, base, list(only)))
        return np.full((len(only), 2), 7, dtype=np.uint32)

    monkeypatch.setitem(sys.modules, "j10r_units", _module("j10r_units", call_receipts=call_receipts))
    T, N, K = 64, 64, 128
    win = types.SimpleNamespace(nk=[(N, K)], t=T, wt=[torch.zeros(N, K, dtype=torch.int8, device="meta")], u_step=10,
                                offs=[0])
    g = {"s": 2, "m": 20, "a": [torch.ones(T + 8, K, dtype=torch.int8)]}
    rec = JS.Refs(win, g, T, skip=True).receipts(0, b"salt", b"ra", b"rb")
    assert seen == [((T, K), "meta", 20, [0, 1])]
    assert rec.tolist() == [[7, 7], [7, 7], [0, 0], [0, 0]]
