"""Every declared scheme, attached on an exchange as the engine attaches it (`messages.attach_listed` with
`scheme_plugin.attach`'s context, then the build's answer), takes every event a served run emits in an honest run's
order without raising, on a processor with no engine: a case names the scheme and the event, each emitted after the
events an honest run emits before it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from verity import hooks as H
from verity.channel import Refused
from verity.protocols.compose import schemes as declarations
from verity.protocols.resource_accounting.network.prover import admission as NA
from verity_vllm import messages as M
from verity_vllm.pipeline import scheme_plugin as SP

#: the least settings each scheme needs to attach, as an application gives them
SETTINGS = {"pearl-plus": {"coins": "in-process", "kernel": "stand-in"},
            "network-attester": {"bucket_us": 1000, "sat_us": 100, "free_us": 10}}

#: the event at which a scheme starts its device run, from which on its cases need a card
DEVICE = {"pearl-plus": (H.PhaseOpened, "its epoch's device run opens at a phase's opening (pouw.scheme.Attached.open, "
                                        "pearl_c_device)")}
ROOT = bytes(64)


def _honest() -> list[object]:
    """One input's step in a phase, every event of `H.EVENTS` but `H.Declared` (the answer's), in the order a run
    emits them."""
    w, x, kv = torch.zeros(4, 4, dtype=torch.int8), torch.zeros(1, 4), torch.zeros(2, 3, 16, 256, dtype=torch.uint8)
    return [H.RegisteredValues((("w", w),), operands={"w": 0}),
            H.RangesDeclared(3, 16, 256, 2, (kv,)),
            H.InputArrived("r0", 3, 0, {}, "POST /v1/completions HTTP/1.1"),
            H.RangeAllocated((0,), "r0"),
            H.PhaseOpened(0, 0),
            H.StepBegan(0, ("r0",), 3),
            H.BlockDue("b", 0, {"m": 1, "n": 4, "k": 4}, "bf16", {"x": x}, 0, {"w": "w"}),
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


def _attached(name: str, tmp_path) -> M.Exchange:
    """``name`` attached on a fresh exchange and the exchange answered, every property it requires declared and every
    reader a process beside it names offered (never started); skipped, with why, where it can't attach off a card."""
    ex = M.Exchange()
    try:
        M.attach_listed(ex, [(declarations.get(name), SETTINGS.get(name, {}))], private=tmp_path, process=H.ENGINE,
                        feed=str(tmp_path / "feed"))
    except (Refused, LookupError, ValueError, RuntimeError) as e:
        pytest.skip(f"{name} does not attach off a card: {type(e).__name__}: {e}")
    ex.readers = {r: lambda *a, **k: None for _, q in ex.requests if isinstance(q, H.StartBeside) for r in q.readers}
    d = SP.declared(_config(), ex.attached)
    required = {r.property: r.value for _, r in ex.requests if isinstance(r, H.RequireProperty)}
    ex.answer(H.Declared(d.card, {**d.properties, **required}, d.schemes, d.settings))
    return ex


@pytest.mark.parametrize("event", ORDER, ids=[t.__name__ for t in ORDER])
@pytest.mark.parametrize("name", sorted(declarations.DECLARATIONS))
def test_a_scheme_takes_every_event(name, event, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if name in DEVICE and ORDER.index(event) >= ORDER.index(DEVICE[name][0]):
        pytest.skip(f"{name} needs a card from {DEVICE[name][0].__name__} on: {DEVICE[name][1]}")
    ex = _attached(name, tmp_path)
    for e in _honest()[:ORDER.index(event)]:
        ex.emit(e)


#: the network attester as the served path configures it in one process (`serve_openai.network_admission`)
SERVED_TICK = {"order": "tick", "hold_ms": 1000, **SETTINGS["network-attester"]}


@pytest.mark.parametrize("event", ORDER[1:], ids=[t.__name__ for t in ORDER[1:]])
def test_the_network_attesters_served_attach_through_the_admission_handle_takes_every_event(event):
    """The served path's attach (`serve_openai.network_admission`): the serving loop's three calls as one
    `hooks.Admit`, each reaching the frontend, which then takes every event before ``event`` in an honest run's order."""
    calls: list[tuple[str, object]] = []
    admit = H.Admit(lambda adm: calls.append(("submit", adm)), lambda eid: calls.append(("abort", eid)),
                    lambda rec: calls.append(("take", rec)))
    ex = M.Exchange()
    got = ex.attach(declarations.get("network-attester"), SERVED_TICK, source="--protocols-config", admit=admit)
    assert type(got) is NA.Admitter and isinstance(got.front, NA.TickAdmit)
    assert (got.front.forward, got.front.abort, got.front.take) == (admit.submit, admit.abort, admit.take)
    assert [type(r) for _, r in ex.requests] == [H.StartOnBoundary, H.ServeRoute]
    for e in _honest()[:ORDER.index(event)]:
        ex.emit(e)


@pytest.mark.parametrize("given", [{"submit": print, "abort": print, "take": print}, {}],
                         ids=["loose-callables", "no-handle"])
def test_the_network_attesters_served_attach_without_the_admission_handle_is_refused_by_name(given):
    ex = M.Exchange()
    with pytest.raises(ValueError, match=r"network-attester: .*verity\.hooks\.Admit, ctx's admit"):
        ex.attach(declarations.get("network-attester"), SERVED_TICK, source="--protocols-config", **given)
    assert ex.requests == []


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
