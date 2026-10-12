"""Runtime correspondence (`runtime-correspondence/v1`): the frontend emits, per Call, the facts that resolve a logical Value to
a runtime / source occurrence, as data beside the Program (`annotations["correspondence"]`), digested apart from it.

    (a) every Call of the derived body has a record, with an occurrence (the fx node it was lowered from);
    (b) `correspondence_digest` is stable across derivations and across a descriptor JSON round trip;
    (c) the namespace is excluded from `program_digest`: adding / removing it leaves the digest byte-identical;
    (d) a rank Program of a tensor-parallel world carries the rank coordinate as integers, and its collective Calls carry the
        collective occurrence with operand ranks -- no string prefix is parsed anywhere;
    (e) the reader's lookups (`owning_module`, `return_slot`, `arg_slot`, `calls_of_module`) answer from the data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.program import Program
from verity_vllm.correspondence.runtime import (
    JSON_SCHEMA, NAMESPACE, SCHEMA, ArgSpan, RuntimeCorrespondence, correspondence_digest, load_correspondence, validate_document,
)
from verity_vllm.program.frontend import derive_examples as ex
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset

sys.path.insert(0, str(Path(__file__).parent))
import test_tp_collective_site as tp  # noqa: E402  (the synthetic tensor-parallel Program: one rank of a world of 2)

PROF = dict(mode="profile", profile="b1-eager")


def _meta(*shape, dtype=torch.bfloat16):
    return torch.empty(*shape, dtype=dtype, device="meta")


def _tp_rank_program(rank=(2, 0), n=3):
    tp.ensure_ops()
    rep = derive_report(tp.Model(), (tp._meta(n, tp.H), tp._meta(n, tp.Q), tp._meta(n, tp.Q), tp._meta(n, tp.H)), **tp.PROF,
                        rules=vllm_ruleset(), name="TpSite", rank=rank)
    assert rep.ok, rep.unsupported
    return rep


def _serve_program(**over):
    from verity_vllm.program.registry import b1
    C = dict(b1.B0_CONFIG)
    C.update(NL=2, V=64, MAX_POS=64)
    C.update(over)
    with torch.device("meta"):
        m = ex.GreedyServe(C, 2)
    rep = derive_report(m, (_meta(3, dtype=torch.int32),), **PROF, config=C, name="Serve")
    assert rep.ok, rep.unsupported
    return rep


# ---------------------------------------------------------------------------------------------------------------------------
# (a) every Call has a record
# ---------------------------------------------------------------------------------------------------------------------------


def test_every_call_has_a_record_with_an_occurrence():
    rep = _tp_rank_program()
    doc = rep.program.annotations[NAMESPACE]
    assert doc["schema"] == SCHEMA and doc["body"] == "program_fn"
    assert validate_document(doc) == []
    body = rep.program.fn.body
    assert len(doc["calls"]) == len(body.nodes)
    assert all(c["occ"] is not None for c in doc["calls"]), [k for k, c in enumerate(doc["calls"]) if c["occ"] is None]
    table = load_correspondence(rep.program)
    assert len(table) == len(body.nodes)
    for k in range(len(body.nodes)):
        rec = table.record(k)
        assert rec.node and rec.op and rec.module          # a source occurrence, an op and an owning module for every Call
        assert len(rec.returns) >= 1 and all(r.output is not None and r.leaves is not None for r in rec.returns)
        assert len(rec.args) == len(body.nodes[k].args)


def test_every_call_has_a_record_on_a_serve_program_with_builder_emitted_calls():
    # rules that emit through the Builder directly (not `Ctx.call`) are covered by `Ctx.note_emitted`: no Call is left without an occurrence
    rep = _serve_program(tie_word_embeddings=False)      # untied: no family claim, the flat body IS the Program's body
    assert not rep.family
    doc = rep.program.annotations[NAMESPACE]
    assert validate_document(doc) == []
    body = rep.program.fn.body
    assert len(doc["calls"]) == len(body.nodes) > 20
    missing = [k for k, c in enumerate(doc["calls"]) if c["occ"] is None]
    assert missing == [], f"Calls without a source occurrence: {missing[:10]} (of {len(missing)})"
    unresolved_returns = [k for k, c in enumerate(doc["calls"]) if any(r["leaves"] is None for r in c["returns"])]
    assert unresolved_returns == [], f"Calls whose return Value was not located in its occurrence's output: {unresolved_returns[:10]}"
    table = load_correspondence(rep.program)
    # in-place lowerings (`rms_norm(out, ...)`, `rotary_embedding(pos, q, k)`, `fused_add_rms_norm(x, residual, ...)`) land their Values in
    # the storage of a WRITTEN fx argument, not an fx output: the slot names that argument
    written = [r for k in range(len(table)) for r in table.record(k).returns if r.written_arg is not None]
    assert written and all(r.output is None and r.located and r.written_node for r in written)
    ops_written = {table.record(r.call).op for r in written}
    assert "vllm_iface.rotary_embedding.default" in ops_written and "vllm_iface.fused_add_rms_norm.default" in ops_written
    for k in range(len(table)):
        rec = table.record(k)
        if rec.op == "vllm_iface.fused_add_rms_norm.default":
            assert [r.written_arg for r in rec.returns] == [0, 1]         # (normed -> x's storage, residual -> residual's storage)
            assert rec.lifetime["in_place"] is True
        if rec.op == "aten.embedding.default":
            assert rec.returns[0].output == 0 and rec.returns[0].written_arg is None
    # the module-body partition covers every Call exactly once
    seen = sorted(k for m in table.modules() for k in table.calls_of_module(m))
    assert seen == list(range(len(body.nodes)))
    assert rep.correspondence_digest == correspondence_digest(rep.program)
    assert rep.program.annotations["derive"]["result"]["correspondence_digest"] == rep.correspondence_digest


def test_family_claim_replaces_the_flat_body_and_the_correspondence_is_not_attached():
    # a verified family claim swaps the flat body for the family's: the correspondence describes the FLAT Calls, so it is NOT attached to
    # the family Program (its Calls are not the flat ones); the derive provenance says so and carries the flat document's digest
    rep = _serve_program(tie_word_embeddings=True)
    assert rep.family and rep.family.get("matched")
    assert NAMESPACE not in rep.program.annotations and rep.correspondence_digest is None
    res = rep.program.annotations["derive"]["result"]
    assert res["correspondence_digest"] is None and res["correspondence"]["applies_to"].startswith("flat body only")
    assert res["correspondence"]["digest"] and res["correspondence"]["calls"] > 0


# ---------------------------------------------------------------------------------------------------------------------------
# (b) digests stable
# ---------------------------------------------------------------------------------------------------------------------------


def test_correspondence_digest_stable_across_derivations_and_json_round_trip():
    a, b = _tp_rank_program(), _tp_rank_program()
    assert a.correspondence_digest == b.correspondence_digest == correspondence_digest(a.program)
    assert a.digest == b.digest
    desc = json.loads(json.dumps(encode_program(a.program)))
    assert correspondence_digest(desc) == a.correspondence_digest
    assert correspondence_digest(desc["annotations"][NAMESPACE]) == a.correspondence_digest
    assert load_correspondence(desc).owning_module(0) == load_correspondence(a.program).owning_module(0)
    assert correspondence_digest({"schema": "other"}) is None and correspondence_digest(None) is None


# ---------------------------------------------------------------------------------------------------------------------------
# (c) namespace excluded from program_digest
# ---------------------------------------------------------------------------------------------------------------------------


def test_namespace_is_excluded_from_program_digest_add_and_remove():
    rep = _tp_rank_program()
    program = rep.program
    assert NAMESPACE in program.annotations
    with_ns = program_digest(program)
    stripped = Program(program.fn, annotations={k: v for k, v in program.annotations.items() if k != NAMESPACE})
    bare = Program(program.fn, annotations={})
    assert program_digest(stripped) == with_ns == program_digest(bare)
    # at the descriptor level too: the digest is a function of everything but `annotations`
    desc = encode_program(program)
    d2 = dict(desc)
    d2["annotations"] = {k: v for k, v in desc["annotations"].items() if k != NAMESPACE}
    assert program_digest(desc) == program_digest(d2) == with_ns
    d3 = dict(desc)
    d3["annotations"] = {**desc["annotations"], NAMESPACE: {"schema": SCHEMA, "body": "program_fn", "inputs": [], "occurrences": [], "calls": []}}
    assert program_digest(d3) == with_ns
    # ... while the correspondence digest DOES see the change
    assert correspondence_digest(d3) != correspondence_digest(desc)
    # and the rank coordinate changes the correspondence, never the Program
    other = _tp_rank_program(rank=(2, 1))
    assert other.digest == rep.digest
    assert other.correspondence_digest != rep.correspondence_digest


def test_rank_argument_is_validated():
    tp.ensure_ops()
    with pytest.raises(ValueError):
        derive_report(tp.Model(), (tp._meta(3, tp.H), tp._meta(3, tp.Q), tp._meta(3, tp.Q), tp._meta(3, tp.H)), **tp.PROF,
                      rules=vllm_ruleset(), name="TpSite", rank=(2, 2))


def test_world_above_two_is_refused():
    """A two-rank op (`verity_tp::all_gather2` / `all_reduce2`) on a rank of a larger world is refused, never emitted with one peer."""
    tp.ensure_ops()
    with pytest.raises(NotImplementedError, match="world 3 refused"):
        derive_report(tp.Model(), (tp._meta(3, tp.H), tp._meta(3, tp.Q), tp._meta(3, tp.Q), tp._meta(3, tp.H)), **tp.PROF,
                      rules=vllm_ruleset(), name="TpSite", rank=(3, 0))


# ---------------------------------------------------------------------------------------------------------------------------
# (d) TP rank coordinate on collective Calls
# ---------------------------------------------------------------------------------------------------------------------------


def test_tp_rank_coordinate_and_collective_occurrence_are_explicit_integers():
    rep = _tp_rank_program(rank=(2, 1))
    table = load_correspondence(rep.program)
    assert table.rank is not None and (table.rank.world, table.rank.rank) == (2, 1)
    coll = table.collective_calls()
    assert len(coll) == 9                         # 2 all_gather sites x 3 rows + 1 all_reduce site x 3 rows
    kinds = {}
    for k in coll:
        rec = table.record(k)
        assert rec.rank is not None and (rec.rank.world, rec.rank.rank) == (2, 1)
        c = rec.collective
        assert c is not None and isinstance(c.occurrence, int) and c.operand_ranks == (0, 1)
        assert c.own_operand == 0 and c.peer_operand == 1 and c.peer_input in ("peer_q", "peer_k", "peer_o")
        kinds.setdefault(c.kind, set()).add(c.occurrence)
        assert rec.module == "layers.0.self_attn" if c.kind == "all_gather" else rec.module == "layers.0.o_proj"
        assert rec.member in ("all_gather", "all_reduce") and rec.row in (0, 1, 2)
        # the collective's operand is `own ++ peer`: the argument spans two fx inputs, each piece located explicitly
        arg = table.arg_slot(k, 0)
        assert arg.arg is None and len(arg.spans) == 2
        own, peer = arg.spans
        assert isinstance(own, ArgSpan) and own.source == "call" and peer.source == "input" and peer.name == c.peer_input
        assert own.leaves[1] - own.leaves[0] == peer.leaves[1] - peer.leaves[0]
    assert kinds == {"all_gather": {0, 1}, "all_reduce": {2}}
    # non-collective Calls carry the rank coordinate too (they belong to the rank Program) but no collective occurrence
    lin = [k for k in range(len(table)) if k not in coll]
    assert lin and all(table.record(k).collective is None and table.record(k).rank.rank == 1 for k in lin)
    # a Program that is not one rank of a world has no rank coordinate
    solo = _tp_rank_program(rank=None)
    assert load_correspondence(solo.program).rank is None
    assert solo.digest == rep.digest


# ---------------------------------------------------------------------------------------------------------------------------
# (e) the reader's lookups
# ---------------------------------------------------------------------------------------------------------------------------


def test_reader_lookups_owning_module_return_slot_arg_slot():
    rep = _tp_rank_program()
    table = load_correspondence(rep.program)
    assert isinstance(table, RuntimeCorrespondence)
    assert table.modules() == ["layers.0.o_proj", "layers.0.self_attn", "layers.0.self_attn.qkv_proj"]
    assert table.calls_of_module("layers.0.self_attn.qkv_proj") == [0, 1, 2]
    assert table.calls_under("layers.0.self_attn") == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert table.calls_under("") == list(range(15))
    # Call 1 = row 1 of the qkv projection: owning module, its return slot inside the fx node's output, its arguments
    assert table.owning_module(1) == "layers.0.self_attn.qkv_proj"
    r = table.return_slot(1, 0)
    assert r.node == "linear" and r.output == 0 and r.row == 1 and r.leaves == (8, 16) and r.view.shape == (3, 8) and r.view.dtype == "bfloat16"
    x = table.arg_slot(1, 0)
    assert x.source == "input" and x.name == "x" and x.arg == 0 and x.leaves == (16, 32) and x.view.shape == (3, 16) and x.view.contiguous is True
    w = table.arg_slot(1, 1)
    assert w.source == "weight" and w.name == "layers.0.self_attn.qkv_proj.weight" and w.leaves == (0, 128) and w.spans == ()
    rec = table.record(1)
    assert rec.rule == "Linear[profile]_v1" and rec.op == "aten.linear.default" and rec.lifetime["in_place"] is False
    with pytest.raises(KeyError):
        table.return_slot(1, 7)
    with pytest.raises(KeyError):
        table.arg_slot(1, 2)
    # the JSON schema names the version and the top-level shape
    assert JSON_SCHEMA["$id"] == SCHEMA and set(JSON_SCHEMA["required"]) == {"schema", "body", "inputs", "occurrences", "calls"}


def test_load_without_the_namespace_raises():
    rep = _tp_rank_program()
    bare = Program(rep.program.fn, annotations={})
    with pytest.raises(KeyError):
        load_correspondence(bare)
    with pytest.raises(KeyError):
        load_correspondence({"annotations": {}})
