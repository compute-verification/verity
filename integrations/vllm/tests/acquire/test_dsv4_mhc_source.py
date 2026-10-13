"""The mHC source on CPU (`verity_vllm/acquire/sources/mhc_source.py`) against the DeepSeek-V4-Pro capture map: the stand-in of
`tests/acquire/dsv4_model.py` runs the pinned vLLM's decoder layers and model forward, which call the four mHC functions by their
call-time globals, on the steps of `STEPS` (two of them above 16 tokens, on the TF32 path) on and off the target, and a real
`NativeHostCommitter`, whose `_acquire` records a copy on CPU, commits what the source binds.

It catches a name the map lists that a step does not bind on its hc route, or binds off it or off the target; a value from the wrong
call (another layer's, the other side's, another return); a call attributed to no side; and a step the route source admits without
its mHC names."""
from __future__ import annotations

import sys
import types
from collections import Counter
from typing import Any

import pytest
import torch

from tests.acquire.dsv4_model import CMAP, ENV, FMA_CALLS, H, HC, L, STEPS, dsv4_model
from tests.acquire.test_dsv4_route_source import Run, map_names
from verity_vllm.acquire.sources import mhc_source as MS
from verity_vllm.acquire.sources import route_source as RS
from verity_vllm.commit import capture_map as CM

OTHERS = tuple(b for b in RS.BINDERS if b != MS.BINDER)
TAPS_ = [t for t in CMAP["taps"] if (t.get("commit") or {}).get("binder") == MS.BINDER]
#: committed name -> (the value it is: `<module>/<side>`, which return)
VALUE_OF = {CM.at_layer(t["commit"]["name"], i): (f"{CM.at_layer(t['module'], i)}/{t['op'].rsplit('.', 1)[-1]}", int(t["site"]["output"]))
            for t in TAPS_ for i in (CM.tap_layers(CMAP, t, ENV) or [None])}


class MhcRun(Run):
    """`Run` with the mHC source attached, recording each step's bound copies, the committer's acquire/flush order and every value
    the stand-in's mHC calls returned."""

    def __init__(self, target: bool = True) -> None:
        super().__init__(target)
        self.mhc = MS.MhcSource(self.com, self.src, self.model)
        self.mhc.install()
        self.rec: dict[int, list[tuple[str, torch.Tensor]]] = {}
        self.log: list[tuple[str, str]] = []
        self.seen: dict[int, dict[str, Any]] = {}
        self.com._acquire = self._record
        self.com.flush_after_plane = self._flush

    def _record(self, name: str, t: torch.Tensor, job: Any) -> None:
        self.rec.setdefault(int(job.step), []).append((name, t.clone()))
        self.log.append(("acquire", name))

    def _flush(self, what: str) -> bool:
        self.log.append(("flush", what))
        return False

    def step(self, k: int, md: Any, skip: tuple[int, ...] = ()) -> None:
        super().step(k, md, skip)
        layers, model = self.model.model.layers, self.model.model
        self.seen[k] = {f"model.layers.{i}/{side}": layers[i].seen[side] for i in range(L) for side in ("hc_attn", "hc_ffn")}
        self.seen[k].update({"model/hc_post": model.seen["hc_post"], "model/hc_head": model.seen["hc_head"]})

    def names(self, k: int) -> list[str]:
        return [n for n, _t in self.rec.get(k, [])]

    def close(self) -> None:
        self.mhc.uninstall()
        self.src.uninstall()


@pytest.fixture(scope="module")
def runs() -> dict[bool, MhcRun]:
    out = {}
    for target in (True, False):
        r = MhcRun(target)
        try:
            for k, (md, _routes, _phase) in enumerate(STEPS):
                r.step(k, md)
        finally:
            r.close()
        out[target] = r
    return out


def test_the_source_wraps_the_four_functions_in_the_model_module():
    r = MhcRun()
    home = sys.modules["tests.acquire.dsv4_model"]
    try:
        assert r.mhc.homes == [home] and all(getattr(home, f).__name__ == "call" for f in MS.FUNCTIONS)
        assert len(r.mhc.owners) == 2 * L + 1 and r.mhc.unowned == {MS.POST: ("hc_post", None)}
    finally:
        r.close()
    assert all(getattr(home, f).__name__ == f for f in MS.FUNCTIONS)


def test_every_step_binds_the_map_names_of_its_hc_route_once(runs, capsys):
    for target, r in runs.items():
        for k, (_md, routes, phase) in enumerate(STEPS):
            want = CM.committer_names(CMAP, ENV, phase=phase, routes=routes, rank=3, binders=(MS.BINDER,), on_target=target)
            assert Counter(r.names(k)) == Counter(set(want)), (target, k)
            if target:
                assert sorted(r.names(k)) == sorted(map_names(capsys, routes, phase, binders=(MS.BINDER,)))
    split = [k for k, (_md, routes, _p) in enumerate(STEPS) if "hc-split" in routes]
    assert split == [2, 4] and {m for f, m, fma in FMA_CALLS if f == MS.POST_PRE and not fma} == {20}
    on, off = runs[True], runs[False]
    assert len(on.names(0)) == len(on.names(2)) == 4 * 2 * L + 2 and sum(n.endswith("_split") for n in on.names(2)) == 4 * (2 * L - 1)
    first = {f"model.layers.0/hc_attn_{v}" for v in ("residual", "post_mix", "comb_mix", "x")}
    assert sorted(off.names(2)) == sorted(first | {"model/hc_post", "model/hc_head"}), "off the target no TF32 return is bound"


def test_each_bound_value_is_the_return_it_names(runs):
    for r in runs.values():
        for k in range(len(STEPS)):
            for name, t in r.rec[k]:
                key, port = VALUE_OF[name]
                v = r.seen[k][key]
                v = v[port] if isinstance(v, tuple) else v
                assert t.dtype == v.dtype and torch.equal(t, v), (k, name)


def test_each_call_flushes_after_its_returns(runs):
    r = runs[True]
    assert r.mhc.stats["calls"] == len(STEPS) * (2 * L + 2) and r.mhc.stats["unattributed"] == 0
    assert sum(1 for e in r.log if e == ("flush", MS.BINDER)) == r.mhc.stats["calls"] and r.log[-1] == ("flush", MS.BINDER)


def test_a_call_with_no_side_s_fn_fails_the_step():
    r = MhcRun()
    try:
        home = sys.modules["tests.acquire.dsv4_model"]
        x, res = torch.zeros(4, H), torch.zeros(4, HC, H)
        r.com._cur = types.SimpleNamespace(step=0)
        before = r.com.x03_expected
        home.mhc_fused_post_pre_tilelang(x, res, torch.zeros(4, HC, 1), torch.zeros(4, HC, HC), torch.zeros(3), None, None, 1e-6, 1e-6,
                                         1e-6, 2.0, 20)
        r.com._cur = None
        assert r.mhc.stats["unattributed"] == 1 and r.rec == {} and r.com.x03_expected == before + 1
        assert r.com.errors[-1].startswith("mhc: mhc_fused_post_pre_tilelang called with an fn")
    finally:
        r.close()


def test_the_route_source_admits_a_step_only_with_its_mhc_names(runs, capsys):
    r = runs[True]
    steps = {k: map_names(capsys, routes, phase, binders=OTHERS) + r.names(k) for k, (_md, routes, phase) in enumerate(STEPS)}
    got = r.check(steps)
    assert got["ok"] is True, got["violations"][:2]
    gone = "model.layers.7/hc_ffn_x"
    got = r.check({**steps, 0: [n for n in steps[0] if n != gone]})
    assert got["ok"] is False and got["violations"][0]["routes_bad"] == {gone: (0, 1)}
    fused = [n.removesuffix("_split") for n in steps[2]]
    got = r.check({**steps, 2: fused})
    assert got["ok"] is False and got["violations"][0]["routes_bad_n"] == 2 * 4 * (2 * L - 1)


def test_the_arena_bound_counts_one_hc_route():
    side = HC * ENV["H"] * 2 + HC * 4 + HC * HC * 4 + ENV["H"] * 2
    for target in (True, False):
        r = MhcRun(target)
        try:
            got = r.mhc.extra_step_bytes({}, r.model)
        finally:
            r.close()
        assert got == {"per_token_bytes": 2 * L * side + (HC + 1) * ENV["H"] * 2, "per_seq_bytes": 0, "tensors_per_step": 8 * L + 2}


def test_attach_attaches_the_mhc_source_beside_the_route_source(monkeypatch):
    class Com:
        def __init__(self) -> None:
            self.sources: list[Any] = []

        def attach_source(self, s: Any) -> None:
            self.sources.append(s)

    assert MS.BINDER in RS.BINDERS and MS.BINDER not in RS.unbuilt(CMAP)
    monkeypatch.setattr(RS, "OWN_RULES", (*RS.OWN_RULES, *RS.unbuilt(CMAP)))
    com = Com()
    src = RS.attach(com, dsv4_model(), rank=3)
    assert [s.short_name for s in com.sources] == ["routes", "mhc"] and com.sources[0] is src
