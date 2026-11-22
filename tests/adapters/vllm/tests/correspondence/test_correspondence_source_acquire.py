"""The plan reads its owning modules off ONE reader -- the `runtime-correspondence/v1` record (`verity_torch.correspondence.runtime`) -- and
falls back to the manifest's v1-annotation reading only when a Build carries no record; either way the plan says which (`correspondence`
block), and the plan's CONTENT (sites, digest) is the same under both sources on the same manifest + tree."""
from __future__ import annotations

import pytest
from verity_torch.correspondence import runtime as RC

from tests.acquire.test_plan import tree
from verity_vllm.correspondence.reader_for_acquire import (
    AcquireCorrespondence,
    ReturnSlot,
    infer_wrapper_prefix,
)


def _record(modules_of_calls: list[str], *, prefix: str = "model.") -> dict:
    """A `runtime-correspondence/v1` document: Call k lowered from an occurrence whose owning module (Program vocabulary = prefix + collector path)."""
    occs: list[dict] = []
    calls, seen = [], {}
    for k, m in enumerate(modules_of_calls):
        pm = (prefix + m) if m else ""
        if pm not in seen:
            seen[pm] = len(occs)
            occs.append({"node": f"n{k}", "module": pm, "module_class": None, "op": "aten.op", "rule": None, "outputs": [{"shape": [8], "dtype": "bf16"}],
                         "written": [], "inputs": [], "lifetime": {}, "collective": None})
        calls.append({"occ": seen[pm], "member": None, "row": k, "hint": None, "returns": [{"port": 0, "output": 0, "leaves": [0, 8]}], "args": []})
    return {"schema": RC.SCHEMA, "body": "program_fn", "inputs": ["ids"], "weights": None, "rank": None, "occurrences": occs, "calls": calls}


MODULES = ["body.embed", "body.blocks.0.norm_in", "body.blocks.0.proj_a", "body.blocks.0.act", "body.blocks.0.norm_out", "logits", ""]








def test_a_path_that_is_the_trees_own_votes_for_no_prefix():
    """A record written in the tree's own paths (no export wrapper): each path that is a module of the tree votes for no prefix, so the
    record's modules read confirmed.  No prefix wins a tie with a longer one, and is outvoted by a prefix more paths meet the tree under."""
    rt = tree()
    own = AcquireCorrespondence(rt, wrapper_prefix="", tables={"p": RC.RuntimeCorrespondence.from_json(_record(MODULES, prefix=""))})
    assert own.wrapper_prefix == "" and own.wrapper_prefix_source == "runtime_tree"
    assert own.confirmed("body.blocks.0.act") is True and own.owned_modules() == sorted(MODULES)
    assert infer_wrapper_prefix(MODULES, rt) == ""
    both = tree(embed=["Embed", "Module"])                                         # `body.embed` is a module as it stands, and `embed` is one under `body.`
    assert infer_wrapper_prefix(["body.embed"], both) == ""                       # one vote each: the shorter
    assert infer_wrapper_prefix(["body.embed", "logits"], both) == ""             # two votes to one
    assert infer_wrapper_prefix(["model.body.embed", "model.logits", "body"], rt) == "model."


def test_owning_module_by_call_id_reads_the_record():
    corr = AcquireCorrespondence(tree(), wrapper_prefix="model.", tables={"p": RC.RuntimeCorrespondence.from_json(_record(MODULES))},
                                 aliases={"body.blocks.0.act": ["body.blocks.0.norm_in", 1]})
    assert corr.owning_module({"call": 2}) == "body.blocks.0.proj_a"
    assert corr.owning_module({"call": 3, "program": "p"}) == "body.blocks.0.norm_in"     # op_path_aliases apply to the record's path too
    assert corr.owning_module({"i": 6, "name": "argmax[0]"}) == "" and corr.owning_module({"name": "model.body.embed/e[0]"}) == "body.embed"
    assert corr.confirmed("body.embed") is True and corr.confirmed("runner.sampler") is False
    assert AcquireCorrespondence(tree()).confirmed("body.embed") is None


def test_a_named_port_off_the_runner_is_the_whole_return_and_a_call_that_names_no_module_is_refused():
    corr = AcquireCorrespondence(tree(), wrapper_prefix="model.")
    assert corr.return_slot({"op_path": "body.embed", "output_member": "hidden"}) == ReturnSlot()
    for call in ({}, {"i": 6}, {"call": 2}):                                         # a call id reads a record, and this reader has none
        with pytest.raises(KeyError, match="neither op_path, call nor name"):
            corr.owning_module(call)
