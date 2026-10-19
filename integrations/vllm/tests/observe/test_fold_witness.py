"""`WeightField` and `Witness` operands on the toy profile of `test_fold_synthetic`: the embedding table read as its weights field whole
is the same Program as its root read through memory; a witness read takes the next leaves of its request's parameter, which is sized
to the reads and recorded as `witness_layout`; a read the profile declares no parameter for, and a declared parameter no read takes,
are refused.  Under a batch, a launch several requests share takes the run's `launch_<name>` parameter and reserves each member's
linked words in its own, in issue order beside its keys."""

from __future__ import annotations

import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array, Value
from verity_vllm.engine.profiles.gen_marlin_moe_patterns import launch_members
from verity_vllm.observe.fold.fold import FoldError, fold
from verity_vllm.observe.fold.resolver import Compute, PerRow, Profile, Shared, WeightField, Witness
from verity_vllm.observe.fold.witnesses import WitnessInputs
from verity_vllm.program.registry import b1
from tests.observe import test_fold_synthetic as T
from tests.observe.test_fold_synthetic import memory  # noqa: F401  (the real memory model, or the stub while it is a skeleton)


def _embedding(ids, table):
    def match(node, ctx):
        if node.op != "aten.embedding.default":
            return None
        w, idx = node.args
        return Compute(bind(b1.Embedding, V=T.C["V"], H=T.C["H"]), idx.shape[0], (ids(idx, ctx), table(w)), (PerRow(node.outs),), (node.id,),
                       node.enter.module or "")
    return match


def _profile(embedding, witnesses=()):
    pats = [T.Pat(p.name, embedding if p.name == "embedding" else p.fn) for p in T.PATTERNS]
    return Profile("toy", pats, dict(T.C), T.weight_field, False, witnesses=witnesses)


def _schedule(idx, ctx):
    return Witness("schedule", tuple(req for req, _ in ctx.start.rows))


def test_a_weight_field_read_whole_is_the_program_of_its_root_read(memory):
    a = fold(T.toy_log(), _profile(_embedding(lambda idx, ctx: PerRow(idx), Shared)), strict=True)
    b = fold(T.toy_log(), _profile(_embedding(lambda idx, ctx: PerRow(idx), lambda w: WeightField(("embed",)))), strict=True)
    assert a.digest == b.digest == fold(T.toy_log(), T.toy_profile(), strict=True).digest


def test_a_weight_field_path_outside_the_weights_record_is_refused(memory):
    with pytest.raises(FoldError, match="not in the weights Record"):
        fold(T.toy_log(), _profile(_embedding(lambda idx, ctx: PerRow(idx), lambda w: WeightField(("nope",)))), strict=True, with_program=False)


def test_witness_reads_take_their_request_parameter_in_issue_order(memory):
    r = fold(T.toy_log(), _profile(_embedding(_schedule, Shared), ("schedule",)), strict=True, with_program=False)
    names = [n for n, _ in r.params]
    assert names[-2:] == ["schedule", "weights"] and r.params[-2][1] == Array(3, Value(32))
    [lay] = r.witness_layout
    assert (lay["param"], lay["name"], lay["witness"], lay["request"], lay["request_index"], lay["leaves"]) == (len(names) - 2, "schedule",
                                                                                                            "schedule", T.REQ, 0, 3)
    assert [(f["offset"], f["leaves"], f["step"]) for f in lay["fields"]] == [(0, 1, 0), (1, 1, 0), (2, 1, 1)]
    assert all(f["spec"].startswith("Embedding_v1") for f in lay["fields"])


def test_a_witness_read_without_its_parameter_is_refused(memory):
    with pytest.raises(FoldError, match="no 'schedule' witness parameter"):
        fold(T.toy_log(), _profile(_embedding(_schedule, Shared)), strict=True, with_program=False)


def test_a_witness_parameter_no_read_takes_is_refused(memory):
    with pytest.raises(FoldError, match="witness parameter, which would be empty"):
        fold(T.toy_log(), _profile(_embedding(lambda idx, ctx: PerRow(idx), Shared), ("schedule",)), strict=True, with_program=False)


def test_a_shared_launch_takes_the_launch_parameter_and_reserves_its_members_linked_words():
    W, params = WitnessInputs(), [("prompt_0", Array(2, Value(32))), ("prompt_1", Array(1, Value(32)))]
    W.declare(params, ("schedule",), 2, ("schedule",))
    assert [n for n, _ in params] == ["prompt_0", "prompt_1", "schedule_0", "schedule_1", "launch_schedule"]
    req, read = {"a": 0, "b": 1}, {"step": 3, "event": 7, "spec": "S", "hint": "h"}
    solo = W.leaves(Witness("schedule", ("a",)), 0, Array(2, Value(32)), "solo", req, read)
    shared = W.leaves(Witness("schedule", ("e3.0",), (("a", 0, 2), ("b", 2, 4))), 0, Array(6, Value(32)), "launch", req, read)
    assert [(r.idx, r.base, r.count) for r in (solo, shared)] == [(2, 0, 2), (4, 0, 6)]
    sized, empty = W.finalise(tuple(params), [{"request_id": "a"}, {"request_id": "b"}])
    assert not empty and [t for _, t in sized[2:]] == [Array(5, Value(32)), Array(5, Value(32)), Array(6, Value(32))]
    lay = {e["name"]: e for e in W.layout(sized, [{"request_id": "a"}, {"request_id": "b"}])}
    assert (lay["launch_schedule"]["request"], lay["launch_schedule"]["request_index"]) == (None, None)
    assert [(f["offset"], f["leaves"], f.get("form"), f.get("slot")) for f in lay["schedule_0"]["fields"]] == [(0, 2, None, None),
                                                                                                           (2, 3, "linked", 0)]
    assert [(f["offset"], f["leaves"], f["launch"], f["slot"], f["pairs"]) for f in lay["schedule_1"]["fields"]] == [(0, 5, "e3.0", 2, 4)]


def test_a_launch_member_without_its_parameter_is_refused():
    W, params = WitnessInputs(), []
    W.declare(params, ("schedule",), 2, ("schedule",))
    with pytest.raises(ValueError, match="launch member 'c' has no 'schedule'"):
        W.leaves(Witness("schedule", ("e0.0",), (("a", 0, 1), ("c", 1, 1))), 0, Array(2, Value(32)), "launch", {"a": 0, "b": 1}, {})


def test_launch_members_are_the_requests_in_launch_order_at_their_slots():
    assert launch_members([("b", 7), ("a", 3), ("a", 4), ("c", 0)], 6) == (("b", 0, 6), ("a", 6, 12), ("c", 18, 6))
    assert launch_members([("a", 3)], 6) == (("a", 0, 6),)
    assert "not contiguous" in launch_members([("a", 3), ("b", 7), ("a", 4)], 6)
