"""install_from_plan on a TINY nn.Module tree installs exactly the plan's sites -- and nothing a class table would have added."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from verity_capture.commit import required_manifest as RM  # noqa: E402
from verity_vllm.acquire import plan as P  # noqa: E402
from verity_vllm.acquire.correspondence import AnnotationCorrespondence, runtime_tree_of  # noqa: E402
from verity_vllm.acquire.install import IncompletePlan, OccurrenceSink, install_from_plan  # noqa: E402

PD = "0" * 64
TABLES = {"HAZARD_LEAVES": set(), "GATHER_FLUSH_LEAVES": set(), "PRE_FLUSH_LEAVES": set()}


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
    man = {"schema": RM.SCHEMA, "program_digest": PD, "identities": rows, "manifest_digest": RM.manifest_digest(rows), "wrapper_prefix": "", "op_path_aliases": {}}
    rt = runtime_tree_of(model, runner, source="tiny")
    return P.acquisition_plan(man, AnnotationCorrespondence(rt), tables=TABLES), rt


def base_rows():
    return [row("input_token_ids", "embed", "in", dtype="i32"), row("instance_outputs", "embed", "out"),
            row("instance_outputs", "blocks.0.norm", "0"), row("instance_outputs", "blocks.0.norm", "1"), row("instance_outputs", "blocks.0.proj", "out"),
            row("instance_outputs", "logits", "out"), row("sampled_token_ids", "runner.sampler", "sampled_token_ids", dtype="i64"),
            row("weights", "logits.weight", "weight", step="*", dtype="f32")]


def test_installs_exactly_the_sites_and_observes_them():
    torch.manual_seed(0)
    model, runner = Tiny(), Runner()
    plan, rt = make(base_rows(), model, runner)
    assert plan.complete, [v.why for v in plan.unsupported]
    sink = OccurrenceSink()
    inst = install_from_plan(plan, model, runner, sink=sink)
    expect = [("blocks.0.norm", "producer_output", "0"), ("blocks.0.norm", "producer_output", "1"), ("blocks.0.proj", "producer_output", "out"),
              ("embed", "consumer_input", "in[0]"), ("embed", "producer_output", "out"), ("logits", "producer_output", "out"),
              ("runner.sampler", "runner_return", ".sampled_token_ids")]
    assert inst.keys() == expect and inst.keys() == [h.key() for h in plan.hooks()]
    assert [s.key() for s in inst.deferred] == [("logits.weight", "registered_input", "weight")]
    # the class table would also hook `blocks.0.extra` (a Linear): the plan does not
    assert not any(s.module == "blocks.0.extra" for s in inst.installed)
    assert len(model.blocks[0].extra._forward_hooks) == 0 and len(model.blocks[0].proj._forward_hooks) == 1 and len(model.embed._forward_pre_hooks) == 1
    ids = torch.tensor([[1, 2, 3]])
    for _ in range(2):
        out = model(ids)
        runner.sample(out)
    assert {k: v for k, v in sink.counts.items()} == {k: 2 for k in expect}
    assert sink.shapes[("blocks.0.norm", "producer_output", "1")] == [[1, 3, 4]] and sink.shapes[("embed", "consumer_input", "in[0]")] == [[1, 3]]
    assert sink.shapes[("runner.sampler", "runner_return", ".sampled_token_ids")] == [[1, 3]] and sink.dtypes[("embed", "consumer_input", "in[0]")] == {"int64"}
    doc = sink.to_json()
    assert [(s["module"], s["kind"], s["slot"]) for s in doc["sites"]] == expect and all(s["occurrences"] == 2 for s in doc["sites"])
    inst.remove()
    assert len(model.blocks[0].proj._forward_hooks) == 0 and len(model.embed._forward_pre_hooks) == 0 and runner.sample.__func__ is Runner.sample
    model(ids); runner.sample(out)
    assert sink.counts[("logits", "producer_output", "out")] == 2          # nothing fires after remove()


def test_install_from_the_serialised_document(tmp_path):
    model, runner = Tiny(), Runner()
    plan, _ = make(base_rows(), model, runner)
    path = tmp_path / "plan.json"
    plan.save(path)
    doc = P.load_plan(path)
    inst = install_from_plan(doc, model, runner)
    assert inst.keys() == [h.key() for h in plan.hooks()]
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


def test_refuses_incomplete_plan_unless_told():
    model, runner = Tiny(), Runner()
    plan, _ = make(base_rows() + [row("instance_outputs", "blocks.0.gone", "out")], model, runner)
    assert not plan.complete
    with pytest.raises(IncompletePlan, match="unsupported"):
        install_from_plan(plan, model, runner)
    inst = install_from_plan(plan, model, runner, refuse_incomplete=False)
    assert len(inst.installed) == 7
    inst.remove()


def test_promoted_consumer_pre_hook_only():
    model, runner = Tiny(), Runner()
    rows = base_rows() + [row("instance_outputs", "blocks.0", RM.PROMOTED_MEMBER, **RM.promoted_annotations({"rows": 3, "width": 4, "dtype": "f32", "spec": "Add_v1", "consumers": ["logits"]}))]
    plan, _ = make(rows, model, runner)
    assert plan.complete
    sink = OccurrenceSink()
    inst = install_from_plan(plan, model, runner, sink=sink)
    assert ("logits", "consumer_input", "in[0]") in inst.keys() and not any(s.module == "blocks.0" for s in inst.installed)
    model(torch.tensor([[1, 2]]))
    assert sink.counts[("logits", "consumer_input", "in[0]")] == 1 and sink.shapes[("logits", "consumer_input", "in[0]")] == [[1, 2, 4]]
    inst.remove()
