from __future__ import annotations

import numpy as np
import pytest

from verity_vllm import commit, dtypes, identity, relations, required, synthetic
from verity_vllm.capture import Value
from verity_vllm.description import Description, NodeInstance
from verity_vllm.negatives import PatchedReader


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return synthetic.build_toy_run(tmp_path_factory.mktemp("toy") / "run", seed=1)


@pytest.mark.usefixtures("pristine_registry")  # review fix (W2c): W2's registry before any plugin
def test_registry_shape():
    assert relations.get("residual_add.bf16.v1").klass == "independent"
    assert relations.get("gemm.same_runtime.v1").klass == "same_runtime"
    assert relations.get("gemm.same_runtime.v1").includes_full_op
    assert not relations.get("gemm.bf16.v1").implemented
    unimplemented = relations.ids(implemented=False)
    for rid, *_ in relations.SLOTS:
        assert rid in unimplemented
    with pytest.raises(relations.UnknownRelation):
        relations.get("nope")
    with pytest.raises(ValueError):
        relations.register(
            "residual_add.bf16.v1", klass="independent", op_type="x", includes_full_op=False
        )(lambda i, x: b"")
    entry = next(r for r in relations.registry_json() if r["relation_id"] == "gemm.bf16.v1")
    assert entry["owner"] == "W4" and entry["implemented"] is False


def make_inst(op_type="residual_add", dtype="bfloat16", shape=("H",)):
    from verity_vllm.description import Node

    node = Node(op="x", layer="0", op_type=op_type, dtype=dtype, shape=tuple(shape), relation="r")
    return NodeInstance(node, "r", 0, "prefill")


def bf16(values):
    return Value(
        "w=x", "bfloat16", (len(values),), dtypes.f32_to_bf16_bytes(np.array(values, np.float32))
    )


def test_residual_add_rounds_to_nearest_even():
    a = bf16([1.0, 1.0, 1.0, 256.0])
    b = bf16([2**-8, 2**-8 + 2**-9, 3 * 2**-8, 1.0])  # halfway cases, a carry
    out = relations.residual_add_bf16(make_inst(), {"a": a, "b": b})
    got = dtypes.bf16_bytes_to_f32(out, (4,)).tolist()
    # 1 + 2^-8 is exactly halfway between 1 and 1+2^-7 -> even (1.0);
    # 1 + 2^-8 + 2^-9 rounds up; 1 + 3*2^-8 is halfway -> even (1 + 2^-6); 257 -> 256 (even)
    assert got == [1.0, 1.0 + 2**-7, 1.0 + 2**-6, 256.0]
    with pytest.raises(relations.RelationError):
        relations.residual_add_bf16(make_inst(), {"a": a, "b": bf16([1.0])})


def test_argmax_tie_rule_lowest_index():
    logits = bf16([0.5, 2.0, 2.0, -1.0])
    out = relations.argmax_lowest_index(make_inst("sampler", "int32", ()), {"logits": logits})
    assert np.frombuffer(out, np.int32)[0] == 1
    out = relations.argmax_lowest_index(make_inst("sampler", "int64", ()), {"logits": logits})
    assert np.frombuffer(out, np.int64)[0] == 1
    with pytest.raises(relations.RelationError):
        relations.argmax_lowest_index(
            make_inst("sampler", "int32", ()),
            {"logits": Value("w=x", "bfloat16", (1,), b"\xc0\x7f")},
        )


def test_embedding_row_is_a_byte_slice():
    table = Value("w=e", "bfloat16", (4, 3), bytes(range(24)))
    token = Value(
        "p=0123456789ab/r=r/b=prompt_token/t=0", "int32", (), np.array(2, np.int32).tobytes()
    )
    assert relations.embedding_row(
        make_inst("embedding"), {"token": token, "table": table}
    ) == bytes(range(12, 18))
    with pytest.raises(relations.RelationError):
        relations.embedding_row(
            make_inst("embedding"),
            {
                "token": Value(token.canonical_id, "int32", (), np.array(9, np.int32).tobytes()),
                "table": table,
            },
        )


def test_token_equals_and_const_fill():
    token = Value(
        "p=0123456789ab/r=r/b=output_token/t=3", "int32", (), np.array(17, np.int32).tobytes()
    )
    assert (
        relations.token_equals(make_inst("sampler", "int32", ()), {"token": token})
        == np.array(17, np.int32).tobytes()
    )
    inst = make_inst("expert", "bfloat16", (4,))
    out = relations.const_fill(inst, {"value": {"const": 0}, "__symbols__": {}})
    assert out == bytes(8)
    inst = make_inst("expert", "bfloat16", ("H",))
    out = relations.const_fill(inst, {"value": 1.0, "__symbols__": {"H": 2}})
    assert out == b"\x80\x3f\x80\x3f"


@pytest.mark.usefixtures("pristine_registry")
def test_unimplemented_slot_raises_clearly():
    with pytest.raises(relations.RelationNotImplemented, match="W4"):
        relations.get("gemm.bf16.v1").checker(make_inst("gemm"), {})


def test_check_exhaustively_on_the_toy(toy):
    reader = toy.reader()
    assert reader.validate() == []
    assert commit.verify_commitments(reader, toy.commitments) == []
    assert commit.verify_source_linkage(reader, toy.commitments) == []
    cov = required.coverage(toy.required, toy.description, reader)
    assert (cov.bound, cov.required, cov.missing, cov.unexpected_ids) == (14, 14, [], [])
    sample = []
    for rid in reader.request_ids():
        sample += list(toy.description.expected_ids(reader.boundary(rid), reader.profile_id))
    challenge = relations.make_challenge(toy.commitments, toy.description, sample, toy.required)
    report = relations.check(
        toy.description,
        reader,
        toy.commitments,
        sample,
        challenge=challenge,
        required_doc=toy.required,
    )
    assert report.all_exact(), report.statuses()
    assert report.openings_verified == report.openings_attempted > len(sample)
    rc = report.relation_checks()
    assert rc["same_runtime"] == {"exact": 0, "attempted": 0}
    assert rc["independent"]["exact"] == rc["independent"]["attempted"] == len(sample)
    fams = report.families()
    assert set(fams) == {"embedding", "gemm", "lm_head", "residual_add", "sampler"}
    assert all(not f.includes_full_op for fam in fams.values() for f in fam.values())
    assert relations.verify_sample_coverage(sample, report).ok
    assert report.challenge_digest == challenge.digest().hex()
    doc = report.to_json()
    assert doc["authentication"]["valid"] == report.openings_verified
    for rid in reader.request_ids():
        assert relations.check_output_linkage(toy.description, reader, toy.commitments, rid).ok


def test_two_adjacent_vus_share_the_authenticated_value(toy):
    """15.1: one VU produces a value the next consumes; both are checked from openings."""
    reader = toy.reader()
    producer = toy.value_id("req-1", 2, "layers.0.mlp.gemm")
    consumer = toy.value_id("req-1", 2, "layers.0.residual")
    inst = toy.description.instance_of(consumer, reader.boundary("req-1"))
    args = toy.description.resolve_args(inst, reader.boundary("req-1"), reader.profile_id)
    assert args["b"] == producer
    report = relations.check(toy.description, reader, toy.commitments, [producer, consumer])
    assert [r.status for r in report.results] == ["exact", "exact"]
    # corrupting the shared value consistently breaks the consumer's check but not the producer's
    mutated = (
        bytes(b ^ 0x01 for b in reader.value(producer).data[:2]) + reader.value(producer).data[2:]
    )
    patched = PatchedReader(reader, values={producer: mutated})
    commitments = commit.build_commitments(patched, write=False)
    report = relations.check(toy.description, patched, commitments, [producer, consumer])
    assert [r.status for r in report.results] == ["mismatch", "mismatch"]
    assert report.openings_verified == report.openings_attempted


def test_check_never_trusts_unauthenticated_openings(toy):
    reader = toy.reader()
    cid = toy.value_id("req-0", 1, "layers.1.residual")
    tampered = dict(toy.commitments)
    tampered["trees"] = dict(toy.commitments["trees"])
    tree_id = reader.tree_of(cid)
    rec = dict(tampered["trees"][tree_id])
    rec["root"] = "00" * 32
    tampered["trees"][tree_id] = rec
    report = relations.check(toy.description, reader, tampered, [cid])
    assert report.results[0].status == "auth_failed"
    assert report.openings_verified == 0


def test_check_reports_structural_and_schema_problems(toy):
    reader = toy.reader()
    good = toy.value_id("req-0", 1, "layers.1.residual")
    bogus = str(identity.parse(good).with_(t=99))
    weight = "w=embed_tokens.weight"
    report = relations.check(toy.description, reader, toy.commitments, [good, bogus, weight])
    assert [r.status for r in report.results] == ["exact", "structural", "structural"]
    # the description says another dtype than the leaf carries
    doc = toy.description.to_json()
    for n in doc["nodes"]:
        if n["op"] == "layers.1.residual":
            n["dtype"] = "float16"
    report = relations.check(Description(doc), reader, toy.commitments, [good])
    assert report.results[0].status == "schema"


@pytest.mark.usefixtures("pristine_registry")
def test_check_records_unimplemented_relations(toy):
    reader = toy.reader()
    cid = toy.value_id("req-0", 1, "layers.1.mlp.gemm")
    doc = toy.description.to_json()
    for n in doc["nodes"]:
        if n["op_type"] == "gemm":
            n["relation"] = "gemm.bf16.v1"
    report = relations.check(Description(doc), reader, toy.commitments, [cid])
    r = report.results[0]
    assert r.status == "unimplemented" and "W4" in (r.detail or "")
    fam = report.families()["gemm"]["independent"]
    assert fam.unimplemented == 1 and fam.attempted == 0 and fam.exact == 0
    assert report.relation_checks()["independent"] == {"exact": 0, "attempted": 0}


def test_checker_errors_are_failures_not_passes(toy):
    reader = toy.reader()
    cid = toy.value_id("req-0", 1, "layers.1.residual")

    def broken(inst, inputs):
        raise ZeroDivisionError("bug")

    registry = dict(relations.REGISTRY)
    registry["residual_add.bf16.v1"] = relations.Relation(
        "residual_add.bf16.v1", "independent", "residual_add", broken, False
    )
    report = relations.check(toy.description, reader, toy.commitments, [cid], registry=registry)
    assert report.results[0].status == "error" and "ZeroDivisionError" in report.results[0].detail


def test_challenge_binding(toy):
    reader = toy.reader()
    sample = relations.draw_sample(toy.description, reader, 5, seed=3)
    challenge = relations.make_challenge(toy.commitments, toy.description, sample, toy.required)
    relations.check(
        toy.description,
        reader,
        toy.commitments,
        sample,
        challenge=challenge,
        required_doc=toy.required,
    )
    with pytest.raises(relations.ChallengeBindingError, match="sample"):
        relations.check(
            toy.description,
            reader,
            toy.commitments,
            sample[:-1],
            challenge=challenge,
            required_doc=toy.required,
        )
    with pytest.raises(relations.ChallengeBindingError, match="required"):
        relations.check(toy.description, reader, toy.commitments, sample, challenge=challenge)
    again = relations.Challenge.from_json(challenge.to_json())
    assert again == challenge
    with pytest.raises(relations.ChallengeBindingError):
        relations.Challenge.from_json({**challenge.to_json(), "digest": "00" * 32})


def test_draw_sample_is_stratified_and_deterministic(toy):
    reader = toy.reader()
    a = relations.draw_sample(toy.description, reader, 10, seed=7)
    b = relations.draw_sample(toy.description, reader, 10, seed=7)
    assert a == b and len(a) == 10 and len(set(a)) == 10
    families = {toy.description.node(identity.parse(c).op).op_type for c in a}
    assert families == {"embedding", "gemm", "lm_head", "residual_add", "sampler"}
    only = relations.draw_sample(toy.description, reader, 4, op_types=["sampler"])
    assert all("/op=sampler.select/" in c for c in only)


def test_check_refuses_a_live_object(toy):
    with pytest.raises(TypeError):
        relations.check(toy.description, object(), toy.commitments, [])  # type: ignore[arg-type]


def test_torch_same_runtime_checkers_if_available(toy):
    torch = pytest.importorskip("torch")
    from verity_vllm import relations_torch

    ids = relations_torch.install()
    assert "residual_add.same_runtime.v1" in ids
    reader = toy.reader()
    cid = toy.value_id("req-0", 1, "layers.1.residual")
    doc = toy.description.to_json()
    for n in doc["nodes"]:
        if n["op_type"] == "residual_add":
            n["relation"] = "residual_add.same_runtime.v1"
    report = relations.check(Description(doc), reader, toy.commitments, [cid])
    assert report.results[0].status == "exact", report.results[0].detail
    assert report.relation_checks()["same_runtime"] == {"exact": 1, "attempted": 1}
    assert report.families()["residual_add"]["same_runtime"].includes_full_op
    del torch
