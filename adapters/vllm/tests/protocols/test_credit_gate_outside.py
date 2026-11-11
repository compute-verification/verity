"""`credit_gate.outside`: the ops inside it are not the trace's, and H1's hook records a call whose z the arm handed over
outside the trace with the same verdicts, tables and records as inside it, while a call without z stays in it."""
from __future__ import annotations

import importlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
TorchDispatchMode = importlib.import_module("torch.utils._python_dispatch").TorchDispatchMode
OPS = importlib.import_module("verity_vllm.engine.int8_ops")
CG = importlib.import_module("verity_vllm.protocols.pouw.credit_gate")
PH = importlib.import_module("verity_vllm.protocols.pouw.h1")
PHK = importlib.import_module("verity_vllm.protocols.pouw.h1.hooks")
SV = importlib.import_module("verity_vllm.protocols.pouw.h1.served")
T = importlib.import_module("tests.protocols.test_pouw_h1_served")


def test_ops_outside_the_trace_are_not_its():
    x = torch.ones(3)
    with CG.Trace("q") as t:
        a = x + 1
        with CG.outside():
            b = x * 2
        c = x - 1
    assert [i.name for i in t.invocations] == ["aten::add", "aten::sub"]
    assert t.producer(a).name == "aten::add" and t.producer(b) is None and t.producer(c).name == "aten::sub"


class _Count(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.ops = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops += 1
        return func(*args, **(kwargs or {}))


def test_outside_leaves_any_other_innermost_mode_and_no_mode_as_they_are():
    with CG.outside():
        assert torch.equal(torch.ones(2) + 1, torch.full((2,), 2.0))
    with CG.Trace("q") as t, _Count() as count:
        with CG.outside():
            torch.ones(2) + 1
    assert count.ops == 2 and [i.name for i in t.invocations] == ["aten::ones", "aten::add"]


def _window(no_z: int) -> str:
    """Two steps of `T`'s three linears through H1's hooks (`traced` apply, `observer` observe), linear ``no_z`` handing
    over no z, with a retaining run: its manifest, call tables, commits, served entries and digest, and verdicts as JSON."""
    rng, g = np.random.default_rng(0), torch.Generator().manual_seed(0)
    run = PH.H1Run("t", T.SALT, T.MODULES, T.SHAPES, T.T, T.STEPS, retain=1, key=bytes([1]) * 32,
                   served=SV.Served(T.TILE, T._unit(), T.BLOCKS, window=T.W1, weights=T._period(T._weights())))
    ws = T._weights()

    def arm(w):
        observe = PHK.observer(run, w, lambda *a: None)

        def apply(layer, x, bias=None):
            a, s_a = x
            y, zy = OPS.scaled_mm_zy(a, s_a, layer.weight, layer.weight_scale, T.T)
            observe(layer, a, s_a, y) if w == no_z else observe(layer, a, s_a, y, zy)
            return y
        return PHK.traced(run, w, apply)

    applies = [arm(w) for w in range(len(T.SHAPES))]
    for _s in range(T.STEPS):
        xs = {}
        for b, (_n, k) in zip(T.BLOCKS, T.SHAPES):
            xs[b] = xs[b] if b in xs else T._codes(rng, T.SERVED, k)
        for w, b in enumerate(T.BLOCKS):
            applies[w](SimpleNamespace(weight=ws[w][0], weight_scale=ws[w][1]), (xs[b], torch.rand(T.SERVED, generator=g) + 0.5))
    run.close()
    return json.dumps({"manifest": run.manifest(), "tables": [run.tables[c.index] for c in run.calls],
                       "commits": run.commits, "entries": run.served.entries(), "digest": run.served.digest(),
                       "verdicts": {str(c): v for c, v in run.verdicts.items()}},
                      sort_keys=True, default=lambda o: o.hex() if isinstance(o, bytes) else repr(o))


def test_a_record_given_z_outside_the_trace_records_and_judges_as_inside_it(monkeypatch):
    """Linear 2 hands over no z, so its retaining record runs ``int_mm`` in the trace and its verdict lists it; q and k
    hand theirs over and are credited. Every byte is as it is with each record inside the trace."""
    outside = _window(no_z=2)
    verdicts = json.loads(outside)["verdicts"]
    assert [v["credit"] for v in verdicts.values()] == [True, True, False] * T.STEPS
    assert all("aten::_int_mm" in v["gemms"] for v in list(verdicts.values())[2::3])
    monkeypatch.setattr(CG, "outside", nullcontext)
    assert _window(no_z=2) == outside
