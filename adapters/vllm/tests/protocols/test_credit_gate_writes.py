"""`credit_gate._written` reads each op's schema once (`_writes`): over in-place, ``out=``, multi-result, view and fresh
ops it records the tensors a walk of the schema at every call does, so a trace's writers, and its verdict, are the same."""
from __future__ import annotations

import pytest

from verity_vllm.protocols.pouw import credit_gate as CG

torch = pytest.importorskip("torch")


def _walked(func, args, kwargs, out):
    s, w = func._schema, []
    for i, a in enumerate(s.arguments):
        if a.alias_info is not None and a.alias_info.is_write:
            w += CG._tensors(kwargs.get(a.name) if a.name in kwargs else args[i] if i < len(args) else None)
    rets = out if isinstance(out, (tuple, list)) else (out,)
    for r, t in zip(s.returns, rets):
        if r.alias_info is None:
            w += CG._tensors(t)
    return w


def test_an_ops_writes_read_once_are_its_schemas_at_every_call(monkeypatch):
    seen = []
    written = CG._written

    def both(func, args, kwargs, out):
        got = written(func, args, kwargs, out)
        seen.append((func._schema.name, [CG.key(t) for t in got], [CG.key(t) for t in _walked(func, args, kwargs, out)]))
        return got

    monkeypatch.setattr(CG, "_written", both)
    x, out = torch.arange(12, dtype=torch.float32).reshape(3, 4), torch.empty(3, 4)
    with CG.Trace("l0") as t:
        for _ in range(2):
            y = x.mm(x.t())
            x.add_(1)
            torch.add(x, 1, out=out)
            vals, idx = x.max(dim=1)
            v = x.view(12)[2:5]
            z = torch.cat([v, vals]).clone()
            torch.sort(z, out=(torch.empty(7), torch.empty(7, dtype=torch.long)))
    names = {n for n, _g, _w in seen}
    assert {"aten::mm", "aten::add_", "aten::add", "aten::max", "aten::view", "aten::cat", "aten::clone"} <= names
    assert all(g == w for _n, g, w in seen) and any(g for _n, g, _w in seen if _n == "aten::sort")
    assert CG._writes.cache_info().hits >= len(seen) // 2
    assert t.producer(y).name == "aten::mm" and t.producer(out).name == "aten::add" and t.producer(idx).name == "aten::max"
    assert CG.verdict(t, y, y)["credit"] is False and CG.verdict(t, y, y)["gemms"] == ["aten::mm", "aten::mm"]
