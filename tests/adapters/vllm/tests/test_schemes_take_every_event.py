"""Every declared scheme, attached on an exchange as the engine attaches it (`messages.attach_listed` with
`scheme_plugin.attach`'s context, then the build's answer), takes every event a served run emits in an honest run's
order without raising, on a processor with no engine: a case names the scheme and the event, each emitted after the
events an honest run emits before it. Pearl-C's device code is a stand-in (`serve`) that nothing hands PoUW until the
serve's path does, so Pearl+ fails as on a card where nothing hands its device run in."""

from __future__ import annotations

import dataclasses
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch

from verity import hooks as H
from verity.channel import Refused
from verity.protocols.run import schemes as declarations
from tests.pipeline.test_serve_learn import SHAPE, parse, stand_ins
from verity_vllm import messages as M
from verity_vllm.pipeline import scheme_plugin as SP
from verity_vllm.pipeline import serve_learn as sl

#: the least settings each scheme needs to attach, as an application gives them (Pearl+'s passes retained besides, as
#: `serve_openai.listed` holds them)
SETTINGS = {"pearl-plus": {"coins": "in-process", "kernel": "stand-in"},
            "network-attester": {"bucket_us": 1000, "sat_us": 100, "free_us": 10}}

ROOT = bytes(64)


class _Run:
    """Pearl-C's device run off a card: its passes counted, each block's output its input."""

    def __init__(self, *args, **kwargs) -> None:
        self.commits, self.details, self.passes = {"coins": {}}, {"passes": []}, 0
        self.cur = SimpleNamespace(matmuls=[])

    def begin(self) -> None:
        pass

    def end_pass(self, step: bool = False) -> None:
        self.passes += 1

    def finish(self) -> None:
        pass

    def gemm(self, key, x, weight, bias):
        return x


@pytest.fixture
def serve(monkeypatch):
    """The serve's exchange module imported afresh, as a serve's process imports it, over Pearl-C's device code
    (`verity_torch.pouw`'s run, weight roots and step limit) as stand-ins and its kernel loading no cubin, with nothing
    handed to PoUW (`tensors.use_pearl_c`) before that import."""
    import verity_torch
    import verity_vllm
    from verity.protocols.accounting.compute.pouw.prover.serving import tensors
    from verity.protocols.accounting.compute.pouw.prover.serving.pearl_c import kernel
    from verity_torch.pouw.pearl_c import passes, run_device

    monkeypatch.setattr(run_device, "PearlCDeviceRun", _Run)
    monkeypatch.setattr(run_device, "weight_roots", lambda k, weights, scheme, *, weight_salts: {w: ROOT for w in weights})
    monkeypatch.setattr(passes, "step_tokens", lambda run, calls: 8)
    monkeypatch.setattr(kernel.Kernel, "load", classmethod(lambda cls, directory, cubin=None: SimpleNamespace()))
    monkeypatch.setattr(tensors, "_pearl_c", None)
    for package, name in ((verity_torch, "pouw"), (verity_vllm, "messages")):
        monkeypatch.delitem(sys.modules, f"{package.__name__}.{name}")
        monkeypatch.setattr(package, name, getattr(package, name))
    return importlib.import_module("verity_vllm.messages")


def _honest() -> list[object]:
    """One input's step in a phase, every event of `H.EVENTS` but `H.Declared` (the answer's), in the order a run
    emits them."""
    w, x, kv = torch.zeros(64, 256, dtype=torch.int8), torch.zeros(1, 256), torch.zeros(2, 3, 16, 256, dtype=torch.uint8)
    return [H.RegisteredValues((("w", w),), operands={"w": 0}),
            H.RangesDeclared(3, 16, 256, 2, (kv,)),
            H.InputArrived("r0", 3, 0, {}, "POST /v1/completions HTTP/1.1"),
            H.RangeAllocated((0,), "r0"),
            H.PhaseOpened(0, 0),
            H.StepBegan(0, ("r0",), 3),
            H.BlockDue("b", 0, {"m": 1, "n": 64, "k": 256}, "bf16", {"x": x, "weight": w}, 0, {"weight": "w"}),
            H.ValueProduced("b", "o0", 0, x),
            H.RangeWritten(0, 0, 3),
            H.StepComputed(0),
            H.StepEnded(0),
            H.OutputsComplete("r0", 0),
            H.RangeFreed((0,)),
            H.PhaseClosed(0, 0, ROOT),
            H.PhaseCommitted(0, ROOT)]


ORDER = [H.Declared, *(type(e) for e in _honest())]


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(model="m", enforce_eager=True, dtype="torch.bfloat16", quantization=None),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1,
                                        distributed_executor_backend="uni"),
        cache_config=SimpleNamespace(enable_prefix_caching=False, num_gpu_blocks_override=None),
        scheduler_config=SimpleNamespace(async_scheduling=False))


def _attached(M, name: str, tmp_path):
    """``name`` attached on a fresh exchange and the exchange answered, every property it requires declared and every
    reader a process beside it names offered (never started); skipped, with why, where it can't attach off a card."""
    ex = M.Exchange()
    try:
        given = {**SETTINGS.get(name, {}), **({"transcript_dir": str(tmp_path / "pearl")} if name == "pearl-plus" else {})}
        M.attach_listed(ex, [(declarations.get(name), given)], private=tmp_path, process=H.ENGINE,
                        feed=str(tmp_path / "feed"))
    except (Refused, LookupError, ValueError, RuntimeError) as e:
        pytest.skip(f"{name} does not attach off a card: {type(e).__name__}: {e}")
    d = SP.declared(_config(), ex.attached)
    required = {r.property: r.value for _, r in ex.requests if isinstance(r, H.RequireProperty)}
    ex.answer(H.Declared(d.card, {**d.properties, **required}, d.schemes, d.settings))
    return ex


@pytest.mark.parametrize("event", ORDER, ids=[t.__name__ for t in ORDER])
@pytest.mark.parametrize("name", sorted(declarations.DECLARATIONS))
def test_a_scheme_takes_every_event(name, event, serve, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = _attached(serve, name, tmp_path)
    for e in _honest()[:ORDER.index(event)]:
        if isinstance(e, H.BlockDue):
            ex.compute(e, lambda inputs: e.inputs["x"])
        else:
            ex.emit(e)


@pytest.mark.parametrize("name", sorted(declarations.DECLARATIONS))
def test_a_scheme_takes_the_serves_warm_up_before_its_first_phase(name, serve, tmp_path, monkeypatch):
    """The serve's order under its Commit: the learn-only warm-up (`serve_learn.warmed`) over a stand-in engine whose
    steps emit on the current exchange as a served step does, from no input and outside any phase, then the first
    window's phase (`serve_commit`) and its step."""
    monkeypatch.chdir(tmp_path)
    _through_the_warm_up(serve, _attached(serve, name, tmp_path))


def test_pearl_plus_and_the_attester_in_release_order_take_the_serves_warm_up(serve, tmp_path, monkeypatch):
    """Pearl+ attached as the engine attaches it, then, after the warm-up, the network attester under release order as
    `serve_openai.network_admission` attaches it (the Pearl+ arm of `h100_smoke.sh` and of `served_verdicts.sh`):
    `weight_copy` allows the pair, the attester files no gate, and both take the first window's events."""
    monkeypatch.chdir(tmp_path)
    ex = _attached(serve, "pearl-plus", tmp_path)

    def attester() -> None:
        got = serve.current().attach(serve.providing("admission"), {"order": "release-order"}, source="--protocols-config",
                                     submit=lambda adm: None, abort=lambda key: None, take=lambda rec: None)
        assert got.gate is None and got.front is not None

    _through_the_warm_up(serve, ex, attester)
    assert sorted(ex.attached) == ["network-attester", "pearl-plus"]
    assert not any(isinstance(r, H.StartOnBoundary) for _, r in ex.requests)


def _through_the_warm_up(serve, ex, after=None) -> None:
    honest = _honest()
    due = next(e for e in honest if isinstance(e, H.BlockDue))
    for e in honest[:2]:
        ex.emit(e)
    opts = parse(*SHAPE)
    engine, com, llm = stand_ins(opts)
    step = engine.step

    def emitting() -> object:
        on, s = serve.current(), serve.current().next_step
        on.emit(H.StepBegan(s, ("learn-warmup-0",), 3))
        on.compute(dataclasses.replace(due, step=s), lambda inputs: due.inputs["x"])
        on.emit(H.StepComputed(s))
        on.emit(H.StepEnded(s))
        return step()

    vars(engine)["step"] = emitting
    with serve.using(ex):
        sl.warmed(com, llm, opts)
        if after is not None:
            after()
    shift = ex.next_step
    for e in honest[2:]:
        if "step" in {f.name for f in dataclasses.fields(e)}:
            e = dataclasses.replace(e, step=e.step + shift)
        if isinstance(e, H.BlockDue):
            ex.compute(e, lambda inputs: due.inputs["x"])
        else:
            ex.emit(e)


def test_a_handler_takes_only_the_events_it_names():
    ex, got = M.Exchange(), []

    def registered(event: H.RegisteredValues) -> None:
        got.append(event.values)

    ex.subscribe("s", "memory", registered)
    ex.subscribe("s", "memory", lambda e: got.append(type(e).__name__))
    for e in _honest()[:2]:
        ex.emit(e)
    assert [type(g).__name__ for g in got] == ["tuple", "str", "str"] and got[1:] == ["RegisteredValues", "RangesDeclared"]

    def stepped(event: H.StepBegan) -> None: ...

    with pytest.raises(Refused, match="StepBegan"):
        ex.subscribe("s", "memory", stepped)
