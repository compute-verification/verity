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
from verity_vllm.query.manifest import format as RM
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




def base_rows():
    return [row("input_token_ids", "embed", "in", dtype="i32"), row("instance_outputs", "embed", "out"),
            row("instance_outputs", "blocks.0.norm", "0"), row("instance_outputs", "blocks.0.norm", "1"), row("instance_outputs", "blocks.0.proj", "out"),
            row("instance_outputs", "logits", "out"), row("sampled_token_ids", "runner.sampler", "sampled_token_ids", dtype="i64"),
            row("weights", "logits.weight", "weight", step="*", dtype="f32")]














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


