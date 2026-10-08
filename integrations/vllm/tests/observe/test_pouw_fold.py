"""Match's record of a PoUW call: the executor's own op under the real observer, folded by `observe.fold.patterns.pouw`.

A model's two linear layers, qkv with its bias and o without, have the shapes of `test_pouw_lowering`'s. They run PoUW's
`ncp-v2` executor installed as a Match capture installs it (`protocol_options.at_capture`: the target's executors, sampled proofs
left out, nothing committed).

- Under the real `Observer` each call is one `verity_pouw.ncp_linear` dispatch, with x, W, b and y as views and nothing of the
  host work inside it.
- `resolve_step` with `PouwLinear` in the profile (`with_protocols`, from the capture's header) resolves each dispatch to the
  row the Build binds the same linear to on a real `torch.export` (`test_pouw_lowering`'s derive): `NcpLinearRowBias_v1` and
  `NcpLinearRow_v1`, one instance per row of x, reading x per row and the parameters shared.
- The capture serves the committed y, word for word.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from verity.core.protocols.pouw.coins import Challenger
from verity_vllm import protocol_options as PO
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Root, RunHeader, StepEnd, StepStart
from verity_vllm.observe.fold.patterns.pouw import OP, PouwLinear, with_protocols
from verity_vllm.observe.fold.resolver import Compute, PerRow, Profile, Shared, StepContext, resolve_step
from verity_vllm.observe.fold.tree import step_events
from verity_vllm.observe.observer import Observer, Recorder
from verity_vllm.observe.storage import StorageAdapter
from verity_vllm.program.frontend.rules.vllm_bindings import triton_launches as TL
from verity_vllm.protocol_options import pouw
from tests.program import test_pouw_lowering as LW

M, K, NQ = LW.M, LW.K, LW.NQ
BF = torch.bfloat16


class Apply:
    def apply(self, layer, x, bias=None):
        raise AssertionError("the PoUW executor replaces this apply")


class Linear(torch.nn.Module):
    """A linear layer as vLLM's `LinearBase` is to PoUW: a weight, maybe a bias, a quant method whose ``apply`` it replaces."""

    def __init__(self, k: int, n: int, bias: bool, g: torch.Generator) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter((torch.randn(n, k, generator=g) * 0.3).to(BF), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.randn(n, generator=g).to(BF), requires_grad=False) if bias else None
        self.quant_method = Apply()

    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(3)
        self.qkv, self.o = Linear(K, NQ, True, g), Linear(NQ, K, False, g)

    def forward(self, x):
        return self.o(self.qkv(x))


def _install(model, stage: str):
    settings = {"pouw": {"coins": Challenger(0)}}
    if stage == "capture":
        return PO.at_capture(LW.POUW, model, settings)
    return PO.install(model, PO.Context(stage=stage, enabled=("pouw:ncp-v2",), settings=settings))


@pytest.fixture
def linears(monkeypatch):
    monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))


@pytest.fixture
def observed(linears):
    """The model's forward under the real observer, with PoUW installed as a Match capture installs it."""
    model, x = Model(), (torch.randn(M, K, generator=torch.Generator().manual_seed(11)) * 0.8).to(BF)
    comp = _install(model, "capture")
    events, storage = [], StorageAdapter()
    rec = Recorder(storage, events.append)
    rec.active, rec.step, rec.phase = True, 0, "forward"
    try:
        with Observer(rec):
            y = model(x)
    finally:
        comp.close()
    return model, x, y, comp, events, storage


def test_the_capture_installs_the_executor_alone_and_commits_nothing(observed):
    _model, _x, _y, comp, _events, _storage = observed
    assert tuple(comp.entries) == ("pouw:ncp-v2",)
    run, _sites = pouw.circuit_run("pouw:ncp-v2")
    assert run.commit is False and run.records == [] and len(run.seconds) == 2


def test_each_call_is_one_opaque_dispatch_of_the_executors_op(observed):
    model, x, y, _comp, events, storage = observed
    enters = [e for e in events if isinstance(e, DispatchEnter)]
    assert [e.op for e in enters] == [f"{OP}.default"] * 2 and all(e.parent is None for e in enters)
    qkv, o = enters
    assert qkv.args[:3] == [storage.view_of(x), storage.view_of(model.qkv.weight), storage.view_of(model.qkv.bias)]
    assert o.args[2] is None and qkv.args[3:] == ["pouw:ncp-v2", 0] and o.args[3:] == ["pouw:ncp-v2", 1]
    exits = {e.id: e for e in events if isinstance(e, DispatchExit)}
    assert exits[o.id].outs == storage.view_of(y) and o.args[0] == exits[qkv.id].outs


def test_the_fold_resolves_each_call_to_the_row_the_build_binds_the_same_linear_to(observed, monkeypatch):
    model, x, _y, _comp, events, storage = observed
    header = RunHeader("verity-capture/log/v1", "2026-09-29T00:00:00Z", "test", {}, {}, {"requests": [], "sampling": {}},
                       notes={"protocols": ["pouw:ncp-v2"]})
    profile = with_protocols(Profile("test", [], {}, lambda name: None), header)
    assert [type(p) for p in profile.patterns] == [PouwLinear]
    names = {"qkv.weight": model.qkv.weight, "qkv.bias": model.qkv.bias, "o.weight": model.o.weight}
    roots: dict[int, list[Root]] = {}
    for n, t in names.items():
        v = storage.view_of(t)
        roots.setdefault(v.alloc, []).append(Root(v, "weight", n))
    start = StepStart(0, [("r0", i) for i in range(M)], list(range(M)), list(range(M)), {"r0": M}, {"r0": 0}, {"r0": [0]}, 16,
                      list(range(M)), [("r0", M - 1)], [M - 1], 1, M, M)
    se = step_events([start, *[e for e in events if isinstance(e, (DispatchEnter, DispatchExit))], StepEnd(0)])
    res = resolve_step(se, StepContext(header, start, None, [], roots, {}, profile))
    assert all(isinstance(r, Compute) for r in res), res
    qkv, o = res
    assert qkv.rows == o.rows == M and qkv.pattern == o.pattern == "pouw.ncp_linear"
    assert qkv.operands == (PerRow(storage.view_of(x)), Shared(storage.view_of(model.qkv.weight)), Shared(storage.view_of(model.qkv.bias)))
    assert o.operands[1:] == (Shared(storage.view_of(model.o.weight)),) and qkv.roots == (None, "qkv.weight", "qkv.bias")
    # the Build's rows for the same linears (a real torch.export, `test_pouw_lowering`)
    monkeypatch.setattr(TL, "linear_types", lambda: (LW.LinearBase,))
    build = LW._derive(LW.Tiny(), LW.POUW)
    built = sorted({n.fn.id for n in build.program.fn.body.nodes if n.fn.id.startswith("NcpLinearRow")})
    assert sorted({qkv.spec.id, o.spec.id}) == built == [f"NcpLinearRowBias_v1{{K={K},N={NQ}}}", f"NcpLinearRow_v1{{K={NQ},N={K}}}"]


def test_without_a_pouw_executor_in_the_header_the_profile_is_unchanged():
    header = RunHeader("verity-capture/log/v1", "t", "test", {}, {}, {"requests": [], "sampling": {}}, notes={"protocols": []})
    profile = Profile("test", [object()], {}, lambda name: None)
    assert with_protocols(profile, header) is profile and with_protocols(profile, None) is profile


def test_the_capture_serves_the_committed_y_word_for_word(observed, linears):
    model, x, y, _comp, _events, _storage = observed
    comp = _install(model, "commit")
    try:
        committed = model(x)
    finally:
        comp.close()
    run, _sites = pouw.circuit_run("pouw:ncp-v2")
    assert run.commit is True and len(run.records) == 2
    assert torch.equal(committed.view(torch.int16), y.view(torch.int16))


def test_the_capture_context_leaves_sampled_proofs_out_and_is_none_without_an_executor():
    ctx = PO.capture_context(LW.POUW, settings={"pouw": {"coins": Challenger(0)}, "sampled-proofs": {}})
    assert ctx.stage == "capture" and ctx.enabled == ("pouw:ncp-v2",) and set(ctx.settings) == {"pouw"}
    assert PO.capture_context(LW.DEFAULT_TARGET) is None and PO.capture_context(None) is None


def test_the_served_y_alone_is_the_commits(linears):
    from verity_vllm.protocol_options import pouw_circuit as PC
    from verity_vllm.protocol_options import pouw_native as PN
    rng = np.random.default_rng(2)
    b = rng.integers(-63, 64, (48, 64)).astype(np.int8)
    t = (np.abs(rng.standard_normal(48)).astype(np.float32) * 0.01 + 1e-3).view(np.uint32)
    call = PC.Call(rng.integers(-63, 64, (32, 64)).astype(np.int8), (np.abs(rng.standard_normal(32)).astype(np.float32) + 1e-3).view(np.uint32),
                   PC.Quantized(b, t), (rng.standard_normal(48).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16), bytes(32),
                   forward=0, layer=0, site=0, weight=0, rowids=np.arange(32, dtype=np.uint32), colids=np.arange(48, dtype=np.uint32))
    assert np.array_equal(PN.served(call, block=16), PN.commit_call(0, "ncp-v2", call).y)


def test_the_match_capture_installs_its_workloads_target_executors(linears):
    """`m1_capture` installs what the workload's target declares, from its `protocols_config` with the epoch's challenger
    its own (`pouw.challenged`), and records the entries in the header."""
    from types import SimpleNamespace

    from verity_vllm.pipeline import m1_capture
    wl = {"target": {"protocols": ["pouw:ncp-v2", "sampled-proofs:vllm-v1"]}, "protocols_config": {"pouw": {}}}
    model, args = Model(), SimpleNamespace(target=None)
    protocols, entries = m1_capture.capture_protocols(wl, args, model)
    try:
        model((torch.randn(M, K, generator=torch.Generator().manual_seed(5)) * 0.8).to(BF))
    finally:
        protocols.close()
    run, _sites = pouw.circuit_run("pouw:ncp-v2")
    assert entries == ["pouw:ncp-v2"] and run.commit is False and len(run.seconds) == 2
    assert m1_capture.capture_protocols({"protocols_config": wl["protocols_config"]}, args, model) == (None, [])


def test_a_workloads_protocols_config_reaches_the_capture_and_the_commit(tmp_path):
    """`load_workload` carries the workload's `protocols_config` (the capture and the Commit install from it), and the Commit's
    `--protocols-config` names another file for one run (a negative control's settings)."""
    import json

    from verity_vllm.engine import vllm_adapter as va
    from verity_vllm.pipeline import cli
    doc = json.loads(open("workloads/workload_qwen05_32x16_1req.json").read())
    doc["protocols_config"] = {"pouw": {"draws": 8}}
    (tmp_path / "wl.json").write_text(json.dumps(doc))
    assert va.load_workload(tmp_path / "wl.json")["protocols_config"] == {"pouw": {"draws": 8}}
    del doc["protocols_config"]
    (tmp_path / "wl.json").write_text(json.dumps(doc))
    assert va.load_workload(tmp_path / "wl.json")["protocols_config"] is None
    assert cli.parse("commit", ["--workload", "w.json", "--out", "o", "--protocols-config", "tamper.json"]).protocols_config == "tamper.json"
