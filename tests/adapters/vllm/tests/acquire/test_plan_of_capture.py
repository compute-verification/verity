"""The acquisition plan of a capture plan's classes (`plan.plan_of_capture`, `stage.run_plan`) on synthetic runtime trees: one site
per capture class, a shared instance's path acquired under its first path's hooks, and a class with no site unsupported."""
from __future__ import annotations

import pytest

from verity_vllm.acquire import stage as G
from verity_vllm.records import capture as CP
from verity_vllm.records.runtime_tree import RuntimeTree, runtime_tree_from_paths, runtime_tree_of

ROPE = "body.blocks.{}.attn.rotary_emb"


def tree() -> RuntimeTree:
    paths = {"": ["Wrap"], "body.embed": ["Embed"], "body.blocks.0.proj_a": ["Lin"], "body.blocks.0.attn": ["Attn"], "body.blocks.1.attn": ["Attn"],
             ROPE.format(0): ["Rope"], "logits": ["Logits"]}
    t = runtime_tree_from_paths(paths, runner={"sample": True}, source="synthetic")
    t.shared = {ROPE.format(1): [ROPE.format(0), 1]}
    return t


def cls(op, member, capture, definitions=("D_v1",)):
    return {"op_path": op, "member": member, "capture": capture, "width": 8, "dtype": "bf16", "definitions": list(definitions)}


CLASSES = [cls("body.embed", "in", CP.INPUT_IDS, ()), cls("body.embed", "out", CP.MODULE_OUTPUT),
           cls("body.blocks.0.proj_a", "0", CP.MODULE_OUTPUT), cls("body.blocks.0.proj_a/row_quant", "0", CP.CALL_BOUNDARY),
           cls(ROPE.format(0), "query", CP.MODULE_OUTPUT), cls(ROPE.format(1), "query", CP.MODULE_OUTPUT),
           cls("runner.sampler", "sampled_token_ids", CP.SAMPLED)]


def test_each_capture_class_has_its_one_site():
    plan = G.run_plan(CLASSES, tree())
    got = {(v.op_path, v.output_member): (v.chosen.kind, v.chosen.module, v.chosen.slot) for v in plan.values}
    assert got == {("body.embed", "in"): ("consumer_input", "body.embed", "in[0]"),
                   ("body.embed", "out"): ("producer_output", "body.embed", "out"),
                   ("body.blocks.0.proj_a", "0"): ("producer_output", "body.blocks.0.proj_a", "0"),
                   ("body.blocks.0.proj_a/row_quant", "0"): ("native_tap", "body.blocks.0.proj_a", "0"),
                   (ROPE.format(0), "query"): ("producer_output", ROPE.format(0), "out"),
                   (ROPE.format(1), "query"): ("producer_output", ROPE.format(0), "out"),
                   ("runner.sampler", "sampled_token_ids"): ("runner_return", "runner.sampler", ".sampled_token_ids")}
    assert not plan.unsupported and plan.digest == G.run_plan(CLASSES, tree()).digest


def test_a_class_at_no_module_of_the_tree_is_unsupported():
    plan = G.run_plan([cls("body.blocks.7.proj_a", "0", CP.MODULE_OUTPUT), cls("body.embed", "x", "hidden")], tree())
    assert [v.why for v in plan.unsupported] == [
        "no legal site: producer_output at body.blocks.7.proj_a/0: 'body.blocks.7.proj_a' is not a module of the runtime tree",
        "no legal site: ? at body.embed/x: capture class 'hidden' has no site rule"]


def test_the_runtime_tree_records_a_shared_instances_paths_in_order():
    torch = pytest.importorskip("torch")
    rope = torch.nn.Identity()
    layers = torch.nn.ModuleList()
    for _ in range(3):
        layer = torch.nn.Module()
        layer.rotary_emb = rope
        layers.append(layer)
    model = torch.nn.Module()
    model.layers = layers
    t = runtime_tree_of(model)
    assert "layers.1.rotary_emb" not in t.modules
    assert t.aliases() == {"layers.1.rotary_emb": ("layers.0.rotary_emb", 1), "layers.2.rotary_emb": ("layers.0.rotary_emb", 2)}
    assert RuntimeTree.from_json(t.to_json()).aliases() == t.aliases()
