"""install_from_plan on a TINY nn.Module tree installs exactly the plan's sites -- and nothing a class table would have added."""
from __future__ import annotations

import types
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import torch
    from torch import nn
else:
    torch = pytest.importorskip("torch")
    nn = torch.nn

from verity_vllm import messages
from verity_vllm.acquire import plan as P
from verity_vllm.acquire.install import (
    IncompletePlan,
    OccurrenceSink,
    apply_plan_to_committer,
    install_from_plan,
    plan_selection,
)
from verity_vllm.correspondence.reader_for_acquire import AcquireCorrespondence
from verity_vllm.check import capture_plan as CP
from verity_vllm.records.runtime_tree import runtime_tree_of

PD = "0" * 64


def observe(plan, model, runner, sink):
    """The plan installed, with ``sink`` subscribed to the values its sites emit."""
    inst = install_from_plan(plan, model, runner)
    messages.current().subscribe("acquisition", "values", lambda e: sink((e.block, e.port, e.slot), e.value))
    return inst
TABLES: dict[str, set[str]] = {"HAZARD_LEAVES": set(), "GATHER_FLUSH_LEAVES": set(), "PRE_FLUSH_LEAVES": set()}


class TwoOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(4, 4, bias=False)

    def forward(self, x, residual):
        y = self.w(x)
        return y, y + residual


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = TwoOut()
        self.proj = nn.Linear(4, 4, bias=False)
        self.extra = nn.Linear(4, 4, bias=False)     # an acquire-class module with no required Value: v1 hooks it, the plan must not

    def forward(self, x):
        h, r = self.norm(x, x)
        return self.extra(self.proj(h)) + r


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 4)
        self.blocks = nn.ModuleList([Block()])
        self.logits = nn.Linear(4, 16, bias=False)

    def forward(self, ids):
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.logits(x)


class SamplerOutput:
    def __init__(self, ids):
        self.sampled_token_ids = ids


class Runner:
    def __init__(self):
        self.sampler = nn.Identity()

    def sample(self, logits):
        return SamplerOutput(logits.argmax(-1))


def row(family, op, member, step=0, **extra):
    return {"family": family, "program_digest": PD, "request_id": "r0", "step": step, "invocation": 0, "op_path": op, "output_member": member,
            "element_range": [0, 4], "rank": 0, **extra}


def make(rows, model, runner):
    capture = {"input_token_ids": CP.INPUT_IDS, "instance_outputs": CP.MODULE_OUTPUT, "sampled_token_ids": CP.SAMPLED}
    classes = [{"op_path": r["op_path"], "member": r["output_member"], "capture": capture[r["family"]], "width": 8,
                "dtype": r.get("dtype", "bf16"), "definitions": []} for r in rows if r["family"] in capture]
    rt = runtime_tree_of(model, runner, source="tiny")
    return P.plan_of_capture(classes, AcquireCorrespondence(rt), tables=TABLES), rt


def base_rows():
    return [row("input_token_ids", "embed", "in", dtype="i32"), row("instance_outputs", "embed", "out"),
            row("instance_outputs", "blocks.0.norm", "0"), row("instance_outputs", "blocks.0.norm", "1"), row("instance_outputs", "blocks.0.proj", "out"),
            row("instance_outputs", "logits", "out"), row("sampled_token_ids", "runner.sampler", "sampled_token_ids", dtype="i64"),
            row("weights", "logits.weight", "weight", step="*", dtype="f32")]




def test_a_second_installation_on_the_runner_observes_its_own_sites_alone():
    model, runner = Tiny(), Runner()
    plan, _ = make(base_rows(), model, runner)
    observe(plan, model, runner, OccurrenceSink()).remove()
    sink = OccurrenceSink()
    inst = observe(plan, model, runner, sink)
    runner.sample(model(torch.tensor([[1, 2, 3]])))
    assert sink.counts[("runner.sampler", "runner_return", ".sampled_token_ids")] == 1 and set(sink.counts.values()) == {1}
    inst.remove()
    assert isinstance(runner.sample, types.MethodType) and runner.sample.__func__ is Runner.sample


def test_a_whole_return_site_of_a_module_returning_several_records_each_tensor():
    model, runner = Tiny(), Runner()
    rows = [r for r in base_rows() if r["op_path"] != "blocks.0.norm"] + [row("instance_outputs", "blocks.0.norm", "out")]
    plan, _ = make(rows, model, runner)
    sink = OccurrenceSink()
    inst = observe(plan, model, runner, sink)
    model(torch.tensor([[1, 2]]))
    key = ("blocks.0.norm", "producer_output", "out")
    assert sink.counts[key] == 1 and sink.shapes[key] == [[1, 2, 4]] and key not in sink.non_tensor
    sink(("linear", "producer_output", "out"), (torch.zeros(3, 2, dtype=torch.bfloat16), None))
    assert sink.shapes[("linear", "producer_output", "out")] == [[3, 2]] and sink.dtypes[("linear", "producer_output", "out")] == {"bfloat16"}
    inst.remove()


def test_refuses_a_plan_for_another_runtime():
    model, runner = Tiny(), Runner()
    plan, _ = make(base_rows(), model, runner)
    other = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(IncompletePlan, match="not in the built runtime"):
        install_from_plan(plan, other, runner)
    assert all(len(m._forward_hooks) == 0 and len(m._forward_pre_hooks) == 0 for m in other.modules())     # nothing left half-installed
    # a runner without sample(): the runner_return site cannot be installed
    with pytest.raises(IncompletePlan):
        install_from_plan(plan, model, object())


def test_refuses_incomplete_plan():
    model, runner = Tiny(), Runner()
    plan, _ = make(base_rows() + [row("instance_outputs", "blocks.0.gone", "out")], model, runner)
    assert not plan.complete
    with pytest.raises(IncompletePlan, match="unsupported"):
        install_from_plan(plan, model, runner)


def test_a_selection_refuses_what_an_install_refuses():
    model, runner = Tiny(), Runner()
    plan, _ = make(base_rows(), model, runner)
    assert plan_selection(plan, model) == (["embed", "blocks.0.norm", "blocks.0.proj", "logits"], ["embed"])
    with pytest.raises(IncompletePlan, match=r"^no model to install on$"):
        plan_selection(plan, None)
    with pytest.raises(IncompletePlan, match=r"^plan hook sites not in the built model: \['blocks.0.norm', 'blocks.0.proj', 'embed', 'logits', 'embed'\]$"):
        plan_selection(plan, nn.Sequential(nn.Linear(4, 4)))
    with pytest.raises(IncompletePlan, match=r"^the committer has no model to install on$"):
        apply_plan_to_committer(types.SimpleNamespace(), plan)
    gone, _ = make(base_rows() + [row("instance_outputs", "blocks.0.gone", "out")], model, runner)
    with pytest.raises(IncompletePlan, match=r"^1 unsupported Value\(s\): blocks.0.gone/out: no legal site: producer_output at blocks.0.gone/out: "):
        plan_selection(gone, model)


def test_an_occurrence_that_holds_no_tensor_is_counted_apart():
    sink, key = OccurrenceSink(), ("blocks.0.proj", "producer_output", "out")
    sink(key, None)
    sink(key, (None, 3))
    assert sink.counts == {key: 2} and sink.non_tensor == {key: 2} and sink.shapes == {} and sink.dtypes == {}
    assert sink.to_json()["sites"] == [{"module": key[0], "kind": key[1], "slot": key[2], "occurrences": 2, "shapes": [], "dtypes": [], "non_tensor": 2}]


def test_a_site_whose_slot_names_no_place_of_its_call_hands_the_sink_nothing():
    """A hook reads its site's slot at each call: `out` or the k-th member of what its module returns, `in[k]` of its arguments.
    A slot of another form, and one past what the call holds, is installed and hands the sink nothing."""
    model = Tiny()
    second = P.AcquisitionSite("producer_output", "blocks.0.norm", "1")
    sites = [P.AcquisitionSite("producer_output", "blocks.0.proj", "hidden"), P.AcquisitionSite("producer_output", "blocks.0.norm", "2"),
             P.AcquisitionSite("producer_output", "logits", "0"), P.AcquisitionSite("consumer_input", "embed", "out"),
             P.AcquisitionSite("consumer_input", "logits", "in[1]"), second]
    plan = P.AcquisitionPlan(PD, PD, "tiny", [P.ValuePlan(s.module, s.slot, "instance_outputs", 1, chosen=s, status="chosen") for s in sites])
    sink = OccurrenceSink()
    inst = observe(plan, model, None, sink)
    assert inst.keys() == sorted(s.key() for s in sites) and not inst.missing
    model(torch.tensor([[1, 2, 3]]))
    assert sink.counts == {second.key(): 1} and sink.shapes == {second.key(): [[1, 3, 4]]}
    inst.remove()


