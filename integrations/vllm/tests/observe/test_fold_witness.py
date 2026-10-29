"""`WeightField` and `WitnessParam` operands on the toy profile of `test_fold_synthetic`: the embedding table read as its weights field whole
is the same Program as its root read through memory; a witness read takes the next leaves of its request's parameter, which is sized
to the reads and recorded as `witness_layout`; a read the profile declares no parameter for, and a declared parameter no read takes,
are refused."""

from __future__ import annotations

import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array, Value
from verity_vllm.observe.fold.fold import FoldError, fold
from verity_vllm.observe.fold.resolver import Compute, PerRow, Profile, Shared, WeightField, WitnessParam
from verity.ml.definitions.registry import b1
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
    return WitnessParam("schedule", tuple(req for req, _ in ctx.start.rows))


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
