from __future__ import annotations

import pytest

from verity_vllm import identity
from verity_vllm.identity import IdentityError, parse, value_id

PID = "0123456789ab"


def test_value_id_round_trip_and_field_order():
    cid = value_id(PID, "req-0", 5, "decode", 3, "layers.3.self_attn.attn", 0)
    text = str(cid)
    assert text == f"p={PID}/r=req-0/t=5/ph=decode/l=3/op=layers.3.self_attn.attn/port=0"
    assert parse(text) == cid
    full = value_id(
        PID,
        "req-0",
        5,
        "decode",
        3,
        "layers.3.mlp.experts.expert.out",
        1,
        shard=1,
        expert=7,
        coordinate=(3, 9),
    )
    text = str(full)
    assert text.endswith("/port=1/s=1/e=7/c=3,9")
    assert parse(text) == full
    assert parse(text).kind == "value"


def test_optional_fields_are_omitted_not_blank():
    cid = value_id(PID, "r", 0, "prefill", "embed", "embed_tokens", 0, expert=2)
    assert "/s=" not in str(cid) and "/e=2" in str(cid)
    with pytest.raises(IdentityError):
        parse(f"p={PID}/r=r/t=0/ph=prefill/l=embed/op=embed_tokens/port=0/s=")


@pytest.mark.parametrize(
    "bad",
    [
        "",
        f"p={PID}/r=r/t=0/ph=prefill/l=embed/op=e/port=0/s=1/e=1/c=1/z=1",  # unknown field
        f"p={PID}/r=r/t=0/ph=prefill/l=embed/op=e/port=0/x=1/c=1",  # x (speculative step) after c only
        f"p={PID}/r=r/t=0/ph=prefill/l=embed/op=e/port=0/x=01",  # leading zero in x
        "w=a/x=1",  # x is a value-id field
        f"p={PID}/r=r/t=0/ph=prefill/l=embed/op=e/port=0/e=1/s=1",  # out of order
        f"p={PID}/r=r/t=0/ph=prefill/l=embed/op=e",  # missing port
        f"p={PID}/r=r/t=00/ph=prefill/l=embed/op=e/port=0",  # leading zero
        f"p={PID}/r=r/t=0/ph=chunk/l=embed/op=e/port=0",  # bad phase
        f"p={PID}/r=r/t=0/ph=prefill/l=head/op=e/port=0",  # bad layer name
        f"p={PID}/r=r/t=0/ph=prefill/l=embed/op=e/port=-1",  # negative
        f"p={PID}/r=r/t=0/ph=prefill/l=embed/op=e/port=0/c=1,,2",  # bad coordinate
        f"p={PID}/r=r/t=0/ph=prefill/l=embed/op=e/port=0/c=1,",
        "p=ABCDEF012345/r=r/t=0/ph=prefill/l=embed/op=e/port=0",  # uppercase hex
        f"p={PID}/r=r=1/t=0/ph=prefill/l=embed/op=e/port=0",  # two '=' in a field
        f"p={PID}/r=r/r=q/t=0/ph=prefill/l=embed/op=e/port=0",  # duplicate key
        f"p={PID}/r=r/t=0/ph=prefill/l=embed/op=e/port=0/",  # trailing slash
        f"p={PID}/t=0/r=r/ph=prefill/l=embed/op=e/port=0",  # r after t
        "w=",  # empty param
        "w=a/s=x",
        "w=a/e=1",
        f"p={PID}/r=r/b=seed/t=1",  # t not allowed for seed
        f"p={PID}/r=r/b=prompt_token",  # t required
        f"p={PID}/r=r/b=unknown/t=1",
        f"p={PID}/b=seed",  # missing r
        "r=r/p=" + PID + "/b=seed",  # wrong order
    ],
)
def test_parse_rejects_non_canonical(bad):
    with pytest.raises(IdentityError):
        parse(bad)
    assert not identity.is_canonical(bad)


def test_weight_ids():
    assert str(identity.weight_id("model.layers.0.mlp.down_proj.weight")) == (
        "w=model.layers.0.mlp.down_proj.weight"
    )
    assert (
        str(identity.weight_id("model.embed_tokens.weight", 1)) == "w=model.embed_tokens.weight/s=1"
    )
    cid = parse("w=model.embed_tokens.weight/s=1")
    assert cid.kind == "weight" and cid.param == "model.embed_tokens.weight" and cid.shard == 1
    with pytest.raises(IdentityError):
        identity.weight_id("bad/param")


def test_boundary_ids():
    assert str(identity.boundary_id(PID, "req-0", "prompt_token", 3)) == (
        f"p={PID}/r=req-0/b=prompt_token/t=3"
    )
    assert str(identity.boundary_id(PID, "req-0", "seed")) == f"p={PID}/r=req-0/b=seed"
    cid = parse(f"p={PID}/r=req-0/b=output_presence/t=9")
    assert cid.kind == "boundary" and cid.field == "output_presence" and cid.t == 9


def test_validation_of_constructor_arguments():
    with pytest.raises(IdentityError):
        value_id("short", "r", 0, "prefill", "embed", "e", 0)
    with pytest.raises(IdentityError):
        value_id(PID, "bad/req", 0, "prefill", "embed", "e", 0)
    with pytest.raises(IdentityError):
        value_id(PID, "r", -1, "prefill", "embed", "e", 0)
    with pytest.raises(IdentityError):
        value_id(PID, "r", 0, "prefill", "embed", "e", True)  # bool is not an int here
    with pytest.raises(IdentityError):
        value_id(PID, "r", 0, "prefill", "embed", "bad op", 0)


def test_with_revalidates():
    cid = value_id(PID, "r", 0, "prefill", "embed", "e", 0)
    assert parse(str(cid.with_(t=4))).t == 4
    with pytest.raises(IdentityError):
        cid.with_(phase="nope")


def test_tree_order_is_code_point_order():
    ids = [str(value_id(PID, "r", t, "prefill", "0", "gemm", 0)) for t in (10, 2, 1, 9)]
    ordered = identity.tree_sorted(ids)
    assert ordered == sorted(ids)
    assert [ord(c) for c in "".join(ordered)] == [ord(c) for c in "".join(sorted(ids))]
    # t=10 sorts before t=2 in tree order (string order), the structured key restores numeric order
    assert ordered[0].endswith("t=1/ph=prefill/l=0/op=gemm/port=0")
    assert ordered[1].endswith("t=10/ph=prefill/l=0/op=gemm/port=0")
    numeric = sorted(ids, key=identity.structured_key)
    assert [parse(x).t for x in numeric] == [1, 2, 9, 10]


def test_structured_key_layers():
    keys = [
        identity.structured_key(str(value_id(PID, "r", 0, "prefill", layer, "op", 0)))
        for layer in ("embed", "0", "11", "norm", "lm_head", "sampler")
    ]
    assert keys == sorted(keys)


def test_speculative_step_field():
    """Lane SD (docs/vllm-poc/speculative.md §3): ``x=<step>`` qualifies a speculative, uncommitted value;
    the unqualified id at the same position is the committed value and sorts first."""
    cid = value_id(PID, "r", t=7, phase="decode", layer="3", op="layers.3.self_attn.qkv_proj.gemm", port=0, speculative=4)
    text = str(cid)
    assert text == f"p={PID}/r=r/t=7/ph=decode/l=3/op=layers.3.self_attn.qkv_proj.gemm/port=0/x=4"
    assert parse(text) == cid and parse(text).speculative == 4
    plain = cid.with_(speculative=None)
    assert parse(str(plain)).speculative is None
    assert sorted([text, str(plain)]) == [str(plain), text]
    assert str(value_id(PID, "r", t=7, phase="decode", layer="3", op="o", port=0, coordinate="1,2", speculative=0)).endswith("/c=1,2/x=0")
