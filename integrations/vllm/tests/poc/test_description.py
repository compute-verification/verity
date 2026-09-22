from __future__ import annotations

import json

import pytest

from verity_vllm import required
from verity_vllm.capture import Boundary, StepRecord
from verity_vllm.description import Description, DescriptionError
from verity_vllm.synthetic import toy_description

PID = "0123456789ab"


@pytest.fixture
def toy_desc() -> Description:
    return Description(toy_description(PID, layers=2, hidden=8, vocab=32))


@pytest.fixture
def boundary() -> Boundary:
    return Boundary.build(
        "req-0",
        [1, 2, 3, 4],
        [5, 6, 7],
        {"max_tokens": 3},
        seed=None,
        finish_reason="length",
        steps=[StepRecord(0, "prefill", (0, 1, 2, 3)), StepRecord(1, "decode", (4,)),
               StepRecord(2, "decode", (5,))],
    )  # fmt: skip


def test_round_trip_and_digest(toy_desc):
    doc = toy_desc.to_json()
    again = Description(doc)
    assert again.digest() == toy_desc.digest()
    assert again.op_families() == ["embedding", "gemm", "lm_head", "residual_add", "sampler"]
    assert toy_desc.leaf_spec(toy_desc.node("lm_head.gemm")) == ("bfloat16", (32,))
    assert toy_desc.leaf_spec(toy_desc.node("sampler.select")) == ("int32", ())
    doc["nodes"][0]["runtime_impl"] = "changed"
    assert Description(doc).digest() != toy_desc.digest()


def test_instances_follow_phases_and_positions(toy_desc, boundary):
    ids = toy_desc.expected_ids(boundary, PID)
    # 6 processed positions x (embed + 2 gemm + 2 residual) + 3 sampled x (lm_head + sampler)
    assert len(ids) == 6 * 5 + 3 * 2
    sampled = [c for c in ids if "/l=sampler/" in c]
    assert sorted(int(c.split("/t=")[1].split("/")[0]) for c in sampled) == [3, 4, 5]
    inst = toy_desc.instance_of(
        f"p={PID}/r=req-0/t=4/ph=decode/l=1/op=layers.1.residual/port=0", boundary
    )
    args = toy_desc.resolve_args(inst, boundary, PID)
    assert args == {
        "a": f"p={PID}/r=req-0/t=4/ph=decode/l=0/op=layers.0.residual/port=0",
        "b": f"p={PID}/r=req-0/t=4/ph=decode/l=1/op=layers.1.mlp.gemm/port=0",
    }
    emb = toy_desc.instance_of(
        f"p={PID}/r=req-0/t=4/ph=decode/l=embed/op=embed_tokens/port=0", boundary
    )
    args = toy_desc.resolve_args(emb, boundary, PID)
    assert args["token"] == f"p={PID}/r=req-0/b=output_token/t=3"  # sampled at t=3, fed at t=4
    assert args["table"] == "w=embed_tokens.weight"
    emb0 = toy_desc.instance_of(
        f"p={PID}/r=req-0/t=0/ph=prefill/l=embed/op=embed_tokens/port=0", boundary
    )
    assert (
        toy_desc.resolve_args(emb0, boundary, PID)["token"] == f"p={PID}/r=req-0/b=prompt_token/t=0"
    )


def test_instance_of_rejects_unpredicted_ids(toy_desc, boundary):
    with pytest.raises(DescriptionError):  # sampler has no output at a non-sampled position
        toy_desc.instance_of(
            f"p={PID}/r=req-0/t=1/ph=prefill/l=sampler/op=sampler.select/port=0", boundary
        )
    with pytest.raises(DescriptionError):  # wrong layer for the op
        toy_desc.instance_of(
            f"p={PID}/r=req-0/t=1/ph=prefill/l=3/op=layers.1.residual/port=0", boundary
        )
    with pytest.raises(DescriptionError):  # wrong phase for the position
        toy_desc.instance_of(
            f"p={PID}/r=req-0/t=1/ph=decode/l=1/op=layers.1.residual/port=0", boundary
        )
    with pytest.raises(KeyError):
        toy_desc.instance_of(f"p={PID}/r=req-0/t=1/ph=prefill/l=1/op=nope/port=0", boundary)


def test_arg_rules_all_le_and_prev(boundary):
    doc = toy_description(PID, 1, 8, 32)
    doc["nodes"].append(
        {
            "op": "layers.0.self_attn.attn",
            "layer": "0",
            "op_type": "attention",
            "dtype": "bfloat16",
            "shape": ["H"],
            "args": [
                {"name": "q", "source": "node", "op": "layers.0.mlp.gemm", "port": 0},
                {
                    "name": "kv",
                    "source": "node",
                    "op": "layers.0.residual",
                    "port": 0,
                    "t": "all_le",
                },
                {"name": "prev", "source": "node", "op": "embed_tokens", "port": 0, "t": "prev"},
                {"name": "eps", "source": "const", "value": 1e-6},
            ],
            "relation": "attention.bf16.v1",
        }
    )
    desc = Description(doc)
    inst = desc.instance_of(
        f"p={PID}/r=req-0/t=4/ph=decode/l=0/op=layers.0.self_attn.attn/port=0", boundary
    )
    args = desc.resolve_args(inst, boundary, PID)
    assert len(args["kv"]) == 5 and args["kv"][0].endswith(
        "t=0/ph=prefill/l=0/op=layers.0.residual/port=0"
    )
    assert args["kv"][4].endswith("t=4/ph=decode/l=0/op=layers.0.residual/port=0")
    assert args["prev"].endswith("t=3/ph=prefill/l=embed/op=embed_tokens/port=0")
    assert args["eps"] == {"const": 1e-6}


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["nodes"][1]["args"][0].update(op="ghost"), "refers to"),
        (lambda d: d["nodes"][0].update(shape=["Z"]), "unknown symbol"),
        (lambda d: d["nodes"].append(dict(d["nodes"][0])), "duplicate node"),
        (lambda d: d["nodes"][0].update(op_type="magic"), "op_type"),
        (lambda d: d["nodes"][0].update(layer="head"), "layer"),
        (lambda d: d["nodes"][0].update(positions="some"), "positions"),
        (lambda d: d["nodes"][0].update(origin="guessed"), "origin"),
        (
            lambda d: d["nodes"][0]["args"].append(
                {"name": "token", "source": "const", "value": 1}
            ),
            "duplicate arg",
        ),
        (lambda d: d["nodes"][0]["args"].append({"name": "x", "source": "node"}), "needs op"),
        (
            lambda d: d["nodes"][0]["args"].append(
                {"name": "x", "source": "boundary", "field": "prompt_token"}
            ),
            "field",
        ),
        (lambda d: d.update(schema="other"), "schema"),
        (lambda d: d.update(symbols={"H": -1}), "symbol"),
    ],
)
def test_validation_errors(mutate, message):
    doc = toy_description(PID, 1, 8, 32)
    mutate(doc)
    with pytest.raises(DescriptionError, match=message):
        Description(doc)


def test_known_relations_check(toy_desc):
    toy_desc.validate(
        known_relations=[
            "embedding.row.v1",
            "gemm.toy.f32acc.v1",
            "residual_add.bf16.v1",
            "argmax.lowest_index.v1",
        ]
    )
    with pytest.raises(DescriptionError, match="unknown relation"):
        toy_desc.validate(known_relations=["embedding.row.v1"])


# --- required.json -------------------------------------------------------------------------------


def test_derive_and_freeze_required(toy_desc, tmp_path):
    partition = {"partition_id": "row", "decomposition": {"default": "row"}}
    doc = required.derive_required(toy_desc, partition)
    assert doc["schema"] == "verity-vllm/required/v1"
    # 7 nodes x 2 phases
    assert len(doc["interfaces"]) == 14
    assert doc["description_digest"] == toy_desc.digest().hex()
    path = tmp_path / "required.json"
    required.freeze_required(path, toy_desc, partition)
    # a coarser candidate for one op family is a different (equal-size) set: allowed
    coarser = {
        "partition_id": "row-coord",
        "decomposition": {"default": "row", "gemm": "coordinate"},
    }
    with pytest.raises(required.RequiredShrinkError):
        required.freeze_required(path, toy_desc, coarser)
    # dropping a phase shrinks the set: refused, file untouched
    with pytest.raises(required.RequiredShrinkError):
        required.freeze_required(path, toy_desc, {**partition, "phases": ["decode"]})
    assert json.loads(path.read_text()) == doc
    # a partition asking for a decomposition the node does not offer is refused
    with pytest.raises(required.RequiredError):
        required.derive_required(
            toy_desc, {"partition_id": "x", "decomposition": {"default": "scalar"}}
        )
    # the same partition again is a no-op
    assert required.freeze_required(path, toy_desc, partition) == doc
    with pytest.raises(required.RequiredError):
        required.parse_required({**doc, "interfaces": doc["interfaces"] + doc["interfaces"][:1]})
