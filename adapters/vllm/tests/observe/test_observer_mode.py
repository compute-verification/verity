"""`verity_vllm.observe.observer`'s mode around a call, on CPU ops: the registered body of a re-entered op and where it is looked
for, what a call hands `after_call`, a `Tensor[]` return with an undefined slot, a call that ends under a later open one, and the
module names `ModuleNamer` keeps for a call's hint."""
from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest
import torch

from verity_vllm.observe import observer as O
from verity_vllm.observe.events import DispatchEnter, DispatchExit, Note
from verity_vllm.observe.storage import StorageAdapter


def _rec() -> tuple[O.Recorder, list]:
    events: list = []
    rec = O.Recorder(StorageAdapter(), events.append)
    rec.active = True
    return rec, events


def _enters(events: list) -> list[DispatchEnter]:
    return [e for e in events if isinstance(e, DispatchEnter)]


@pytest.fixture
def reentry(monkeypatch):
    """The module's record of the implementations it has looked up and the notes it has made, empty for the test."""
    for name in ("_REENTER_IMPL", "_REENTER_MISSING"):
        monkeypatch.setattr(O, name, {})
    for name in ("_MISSING_IMPL_NOTED", "_UNDEFINED_RETURN_NOTED"):
        monkeypatch.setattr(O, name, set())
    return monkeypatch


def test_an_ops_name_is_its_namespace_its_name_and_its_overload_and_default_where_it_names_none():
    assert O.op_name(torch.ops.aten.add.Tensor) == "aten.add.Tensor" and O.op_name(torch.ops.aten.mm.default) == "aten.mm.default"
    schema = SimpleNamespace(name="vllm::unified_kv_cache_update")
    for unnamed in (SimpleNamespace(namespace="vllm", _schema=schema), SimpleNamespace(namespace="vllm", _schema=schema, _overloadname="")):
        assert O.op_name(unnamed) == "vllm.unified_kv_cache_update.default"


def test_the_ops_that_are_reentered_are_the_two_named_and_every_moe_forward():
    names = ["vllm.unified_attention_with_output", "vllm.unified_kv_cache_update", "vllm.moe_forward", "vllm.moe_forward_shared",
             "vllm.unified_attention", "aten.mm", "_C.moe_forward"]
    assert [O.reenters(n) for n in names] == [True, True, True, True, False, False, False]
    assert O.op_base_of("vllm.moe_forward_shared.default") == "vllm.moe_forward_shared"


def test_an_ops_implementation_is_the_first_of_its_paths_that_imports_and_is_looked_up_once(reentry):
    reentry.setitem(O.REENTER_IMPL_PATHS, "verity.op", [("verity_vllm.no_such_module", "impl"), ("verity_vllm.observe.observer", "op_base_of"),
                                                         ("verity_vllm.observe.observer", "op_name")])
    assert O.reenter_impl("verity.op") is O.op_base_of and "verity.op" not in O._REENTER_MISSING
    reentry.setitem(O.REENTER_IMPL_PATHS, "verity.op", [("verity_vllm.observe.observer", "op_name")])
    assert O.reenter_impl("verity.op") is O.op_base_of, "the implementation found is kept"
    assert O.reenter_impl_name("verity.op") == "verity_vllm.observe.observer.op_base_of"


def test_an_implementations_name_is_the_function_it_wraps_and_none_where_no_path_is_known(reentry):
    def body(*args):
        return None

    reentry.setitem(O._REENTER_IMPL, "verity.wrapped", functools.update_wrapper(body, O.op_name, assigned=()))
    assert body.__qualname__.endswith(".body") and O.reenter_impl_name("verity.wrapped") == "verity_vllm.observe.observer.op_name"
    assert O.reenter_impl_name("verity.unknown") is None
    assert O._REENTER_IMPL["verity.unknown"] is None and O._REENTER_MISSING["verity.unknown"] == []


def test_a_reentered_ops_registered_body_runs_in_its_place_and_its_calls_are_the_ops_children(reentry):
    rec, events = _rec()
    reentry.setattr(O, "REENTER_OPS", {"aten.mm", "aten.mul"})
    reentry.setitem(O._REENTER_IMPL, "aten.mm", lambda a, b: torch.mul(a, b))
    reentry.setitem(O._REENTER_IMPL, "aten.mul", lambda a, b: torch.add(a, b))
    a = torch.full((2, 2), 3.0)
    with O.Observer(rec):
        out = torch.mm(a, a)
    assert out.tolist() == [[6.0, 6.0], [6.0, 6.0]], "the innermost body's sum: a body's own re-entered op runs its body too"
    outer, middle, inner = _enters(events)
    assert (outer.op, middle.op, inner.op) == ("aten.mm.default", "aten.mul.Tensor", "aten.add.Tensor")
    assert (outer.parent, middle.parent, inner.parent) == (None, outer.id, middle.id), "a call's parent is the innermost open call"
    exits = [e for e in events if isinstance(e, DispatchExit)]
    assert [e.id for e in exits] == [inner.id, middle.id, outer.id] and all(e.outs == rec.storage.view_of(out) for e in exits)
    assert rec.counts == {"dispatch": 3, "triton": 0, "notes": 0, "reentered": 2} and rec.open == []


def test_a_mode_told_not_to_reenter_runs_the_op_itself(reentry):
    rec, events = _rec()
    reentry.setattr(O, "REENTER_OPS", {"aten.mm"})
    reentry.setitem(O._REENTER_IMPL, "aten.mm", lambda a, b: pytest.fail("the body ran"))
    a = torch.full((2, 2), 3.0)
    with O.Observer(rec, reenter=False):
        out = torch.mm(a, a)
    assert out.tolist() == [[18.0, 18.0], [18.0, 18.0]] and [e.op for e in _enters(events)] == ["aten.mm.default"]
    assert rec.counts["reentered"] == 0


def test_a_reentered_op_with_no_implementation_runs_itself_and_is_noted_once(reentry):
    rec, events = _rec()
    reentry.setattr(O, "REENTER_OPS", {"aten.mm"})
    reentry.setitem(O.REENTER_IMPL_PATHS, "aten.mm", [("verity_vllm.no_such_module", "impl")])
    a = torch.full((2, 2), 3.0)
    with O.Observer(rec):
        first, second = torch.mm(a, a), torch.mm(a, a)
    assert first.tolist() == second.tolist() == [[18.0, 18.0], [18.0, 18.0]]
    assert [type(e).__name__ for e in events if not type(e).__name__.startswith("Storage")] == [
        "DispatchEnter", "Note", "DispatchExit", "DispatchEnter", "DispatchExit"]
    [note] = [e for e in events if isinstance(e, Note)]
    assert note.text == "re-entry impl not found; op recorded without children" and set(note.data) == {"op", "tried"}
    assert note.data["op"] == "aten.mm.default" and len(note.data["tried"]) == 1
    assert note.data["tried"][0].startswith("verity_vllm.no_such_module.impl: ModuleNotFoundError(")
    assert rec.counts["reentered"] == 0 and rec.counts["notes"] == 1


def test_after_call_gets_a_calls_id_name_arguments_and_result_once_the_call_has_ended():
    rec, events = _rec()
    seen: list = []
    rec.after_call = lambda *call: seen.append((call, type(events[-1]).__name__, list(rec.open)))
    a, wide = torch.full((2, 2), 3.0), torch.zeros(2, 3)
    with O.Observer(rec):
        out = torch.add(a, a, alpha=2)
        with pytest.raises(RuntimeError):
            torch.mm(wide, wide)
    assert len(seen) == 1, "a call that raises is handed to no one"
    [((eid, name, args, kwargs, got), last, still_open)] = seen
    assert (eid, name, last, still_open) == (_enters(events)[0].id, "aten.add.Tensor", "DispatchExit", [])
    assert len(args) == 2 and args[0] is a and args[1] is a and kwargs == {"alpha": 2}
    assert got.tolist() == out.tolist() == [[9.0, 9.0], [9.0, 9.0]]


def test_an_undefined_slot_of_a_tensor_list_goes_back_as_an_empty_tensor_and_is_recorded_as_null(reentry):
    rec, events = _rec()
    reentry.setattr(O, "REENTER_OPS", {"aten.unbind"})
    reentry.setitem(O._REENTER_IMPL, "aten.unbind", lambda x, dim=0: [x[0], None])
    x = torch.arange(6.0).reshape(2, 3)
    with O.Observer(rec):
        first, second = torch.unbind(x), torch.unbind(x)
    for out in (first, second):
        assert len(out) == 2 and out[0].tolist() == [0.0, 1.0, 2.0] and (out[1].shape, out[1].device) == ((0,), x.device)
    unbinds = [e.id for e in _enters(events) if e.op == "aten.unbind.int"]
    assert [e.outs for e in events if isinstance(e, DispatchExit) and e.id in unbinds] == [[rec.storage.view_of(x[0]), None]] * 2
    [note] = [e for e in events if isinstance(e, Note)]
    assert note.text.startswith("undefined Tensor[] return elements replaced by empty placeholders")
    assert note.data == {"op": "aten.unbind.int", "undefined": [1]}, "noted once an op"


def test_a_return_with_no_undefined_slot_beside_a_tensor_goes_back_as_it_is():
    rec, events = _rec()
    t = torch.zeros(2)
    for out in ([t, t], [None, None], [], (t, None), t, None):
        assert O._defined_tensor_list(rec, "an.op", out) is out
    assert events == []


def test_a_call_that_ends_under_a_later_open_one_leaves_the_stack_by_its_id(reentry):
    rec, _ = _rec()
    reentry.setattr(O, "REENTER_OPS", {"aten.mm", "aten.mul"})

    def leaves_one_open(a, b):
        rec.open.append(900)
        return torch.add(a, b)

    def clears_the_stack(a, b):
        rec.open.clear()
        return torch.add(a, b)

    reentry.setitem(O._REENTER_IMPL, "aten.mm", leaves_one_open)
    reentry.setitem(O._REENTER_IMPL, "aten.mul", clears_the_stack)
    a = torch.full((2, 2), 3.0)
    with O.Observer(rec):
        torch.mm(a, a)
        assert rec.open == [900]
        torch.mul(a, a)
        assert rec.open == []


class _Attention(torch.nn.Module):
    def __init__(self, rotary: torch.nn.Module) -> None:
        super().__init__()
        self.rotary = rotary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rotary(x)


class _Layer(torch.nn.Module):
    def __init__(self, rotary: torch.nn.Module) -> None:
        super().__init__()
        self.attn = _Attention(rotary)
        self.rotary = rotary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rotary(self.attn(x)).neg()


class _Model(torch.nn.Module):
    """Two layers that share one rotary module, as vLLM's layers share a cached rotary embedding."""

    def __init__(self) -> None:
        super().__init__()
        rotary = torch.nn.Tanh()
        self.layers = torch.nn.ModuleList([_Layer(rotary), _Layer(rotary)])
        self.head = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.exp()
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _hints(rec: O.Recorder, events: list, model: torch.nn.Module) -> list[tuple[str, str | None]]:
    events.clear()
    with O.Observer(rec), torch.no_grad():
        model(torch.ones(1, 2))
    return [(e.op.split(".")[1], e.module) for e in _enters(events)]


def test_a_calls_module_is_the_innermost_submodule_running_and_a_shared_one_is_named_under_the_layer_calling_it():
    rec, events = _rec()
    model = _Model()
    assert [n for n, _ in model.named_modules()] == ["", "layers", "layers.0", "layers.0.attn", "layers.0.attn.rotary", "layers.1",
                                                     "layers.1.attn", "head"]
    namer = O.ModuleNamer(rec, model)
    assert len(namer.handles) == 14, "a pre hook and a post hook on every submodule, none on the model itself"
    assert _hints(rec, events, model) == [
        ("ones", None), ("exp", None),
        ("tanh", "layers.0.attn.rotary"), ("tanh", "layers.0.attn.rotary"), ("neg", "layers.0"),
        # called from inside layer 1's attention, the shared module is named under layer 1; called by the layer module itself, whose
        # name has nothing after its number, it keeps the name it was registered under
        ("tanh", "layers.1.attn.rotary"), ("tanh", "layers.0.attn.rotary"), ("neg", "layers.1"),
        ("sigmoid", "head")]
    assert rec.module_stack == []


def test_a_module_outside_the_layers_keeps_its_name_and_a_shared_one_is_named_under_the_outermost_layer_calling_it():
    rec, _ = _rec()
    namer = O.ModuleNamer(rec, torch.nn.Identity())
    assert namer.handles == [], "a model with no submodule has no hook"
    for caller, registered, named in (
            ("layers.1.attn", "head", "head"),
            ("head", "layers.0.attn.rotary", "layers.0.attn.rotary"),
            ("model.layers.0.attn", "model.layers.0.attn.rotary", "model.layers.0.attn.rotary"),
            ("model.layers.1.blocks.layers.3.attn", "model.layers.0.blocks.layers.0.rotary", "model.layers.1.blocks.layers.0.rotary")):
        rec.module_stack[:] = [caller]
        namer._pre(registered, None, ())
        assert rec.module_stack == [caller, named]


def test_a_removed_namer_leaves_no_hook_and_a_post_hook_on_an_empty_stack_does_nothing():
    rec, events = _rec()
    model = _Model()
    namer = O.ModuleNamer(rec, model)
    namer._post("layers.0", model.layers[0], (), None)
    assert rec.module_stack == []
    namer.remove()
    assert namer.handles == [] and {hint for _, hint in _hints(rec, events, model)} == {None}
    assert not any(m._forward_pre_hooks or m._forward_hooks for m in model.modules())
