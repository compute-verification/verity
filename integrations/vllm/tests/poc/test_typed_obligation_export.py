"""The serving side's ``typed-obligation/v0`` exporter, on a synthetic authenticated GEMM coordinate.

The real Stage-2 record (``fixtures/typed-obligation-v0/stage2--*.json``) is
exported from a persisted fixture of W3c's branch; this test builds a small
fixture of the same layout in a temporary directory -- real trees under the
adapter's framing, real openings -- so the exporter is exercised here without
the 6 MB of persisted operands.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from verity_vllm import hashing, identity
from verity_vllm.commit import Opening, fold_path, path_shape
from verity_vllm.typed_obligation_export import (
    ExportError,
    export_gemm_coordinate,
    export_record,
    load_fixture,
    packed_witness,
)

veritor_interface = pytest.importorskip("veritor.interface")
from veritor.interface import VLLM_FRAMING, check_relations, gemm_coordinate, to_statement, validate  # noqa: E402
from veritor.protocol.messages import Reject  # noqa: E402
from veritor.protocol.proofs import (  # noqa: E402
    PackedLeaf,
    PackedWitness,
    PositionRef,
    TransparentBackend,
    decode_packed_witness,
    decode_statement,
    encode_packed_witness,
    packed_schema,
    parse_packed_schema,
)

PROFILE = "0123456789ab"
REQUEST = "req-1"


def _tree(leaves: list[tuple[str, str, tuple[int, ...], bytes]]) -> tuple[bytes, list[list[bytes | None]]]:
    """A Merkle tree over ``leaves`` (tree order) under the adapter's framing; returns the root and every path."""

    level = [hashing.leaf_hash(c, d, s, v) for c, d, s, v in leaves]
    paths: list[list[bytes | None]] = [[] for _ in leaves]
    positions = list(range(len(leaves)))
    while len(level) > 1:
        nxt = []
        for i in range(0, len(level) - 1, 2):
            nxt.append(hashing.node_hash(level[i], level[i + 1]))
        if len(level) % 2:
            nxt.append(hashing.lift_hash(level[-1]))
        for leaf, pos in enumerate(positions):
            if pos % 2 == 0 and pos + 1 < len(level):
                paths[leaf].append(level[pos + 1])
            elif pos % 2 == 1:
                paths[leaf].append(level[pos - 1])
            else:
                paths[leaf].append(None)
        positions = [p // 2 for p in positions]
        level = nxt
    return level[0], paths


def _write_fixture(tmp: Path, k: int = 32, n: int = 4, j: int = 2, *, corrupt: bool = False) -> Path:
    rng = np.random.default_rng(7)
    x = rng.integers(0x3C00, 0x4000, size=k, dtype=np.uint16)  # BF16 words around 1.0
    w = rng.integers(0xBC00, 0xBE00, size=(n, k), dtype=np.uint16)  # around -0.01

    from veritor.core import make_tc_ampere_bf16_gate_set

    gs = make_tc_ampere_bf16_gate_set()
    acc = gs["tc_dot16_0"].evaluate(tuple(int(v) for v in (*x[:16], *w[j, :16])))
    for s in range(16, k, 16):
        acc = gs["tc_dot16"].evaluate((acc, *(int(v) for v in x[s : s + 16]), *(int(v) for v in w[j, s : s + 16])))
    out_word = gs["f32_to_bf16"].evaluate((acc,))
    out = np.zeros(n, dtype=np.uint16)
    out[j] = out_word + (1 if corrupt else 0)

    x_id = str(identity.value_id(PROFILE, REQUEST, 3, "decode", "0", "layers.0.input_layernorm.rmsnorm", 0))
    out_id = str(identity.value_id(PROFILE, REQUEST, 3, "decode", "0", "layers.0.self_attn.qkv_proj.gemm", 0))
    w_id = f"w=layers.0.self_attn.qkv_proj.weight/c={j}"  # weights v2 row-block id
    ru_leaves = [(x_id, "bfloat16", (k,), x.tobytes()), (out_id, "bfloat16", (n,), out.tobytes())]
    ru_leaves.sort(key=lambda leaf: leaf[0])
    w_leaves = [(f"w=layers.0.self_attn.qkv_proj.weight/c={r}", "bfloat16", (1, k), w[r].tobytes()) for r in range(n)]
    w_leaves.sort(key=lambda leaf: leaf[0])
    ru_root, ru_paths = _tree(ru_leaves)
    w_root, w_paths = _tree(w_leaves)
    ru_tree = f"ru/{REQUEST}/000002"

    def opening(tree, root, leaves, paths, cid):
        idx = [c for c, *_ in leaves].index(cid)
        c, d, s, v = leaves[idx]
        return Opening(tree, root, len(leaves), idx, c, d, s, v, tuple(paths[idx])).to_json()

    fixture_dir = tmp / "qkv_proj_decode_0"
    fixture_dir.mkdir()
    (fixture_dir / "openings.json").write_text(
        json.dumps(
            {
                "x": opening(ru_tree, ru_root, ru_leaves, ru_paths, x_id),
                "w_row": opening("weights", w_root, w_leaves, w_paths, w_id),
                "out_row": opening(ru_tree, ru_root, ru_leaves, ru_paths, out_id),
            }
        )
    )
    (fixture_dir / "fixture.json").write_text(
        json.dumps(
            {
                "K": k,
                "N": n,
                "j": j,
                "canonical_id": out_id,
                "x_id": x_id,
                "w_id": w_id,
                "x_arg_cols": None,
                "family": "qkv_proj",
                "phase": "decode",
                "op": "layers.0.self_attn.qkv_proj.gemm",
                "profile_id": PROFILE,
                "committed_word": int(out[j]),
                "predicted_word": int(out_word),
                "kernel": {"kernel_path": "matmul_kernel_persistent", "runtime_impl": "test"},
            }
        )
    )
    for name, arr in (("x.u16", x), ("w_row.u16", w[j]), ("out_row.u16", out)):
        (fixture_dir / name).write_bytes(arr.tobytes())
    (tmp / "commitments.json").write_text(
        json.dumps(
            {
                "schema": "verity-vllm/commitments/v1",
                "profile_id": PROFILE,
                "trees": {
                    ru_tree: {
                        "kind": "ru",
                        "root": ru_root.hex(),
                        "leaf_count": len(ru_leaves),
                        "ids_digest": hashing.ids_digest([c for c, *_ in ru_leaves]).hex(),
                    },
                    "weights": {
                        "kind": "weights",
                        "root": w_root.hex(),
                        "leaf_count": len(w_leaves),
                        "ids_digest": hashing.ids_digest([c for c, *_ in w_leaves]).hex(),
                    },
                },
            }
        )
    )
    return fixture_dir


def test_exporter_builds_the_canonical_program_and_the_relation_holds(tmp_path: Path) -> None:
    fixture_dir = _write_fixture(tmp_path)
    fx = load_fixture(fixture_dir)
    obj, row, how = export_gemm_coordinate(fx)
    gate_set = validate(obj)
    assert obj.family.id == "veritor.tc-ampere-bf16@2"  # L8: the hardware NaN word in the store cast
    assert obj.subcircuits == (gemm_coordinate(32),)
    (obligation,) = obj.obligations
    assert obligation.target is not None and obligation.target.encoding == "call"
    assert obligation.declared_interface is not None
    assert len(obligation.boundary.incoming) == 64 and len(obligation.boundary.outgoing) == 1
    assert all(r.framing == VLLM_FRAMING for r in obj.roots)
    assert {r.owner for r in obj.roots} == {-2, 2}
    assert how["x"] == "verify_opening_in" and how["out_row"] == "verify_opening_in"
    # the v2 row-block id is outside a v1 identity grammar: authenticated by the hash chain
    assert how["w_row"].startswith("hash-chain") or how["w_row"] == "verify_opening_in"
    assert check_relations(obj, [row], gate_set)
    record = export_record(fx, source="test")
    assert record["verdict"] == "accept" and record["relations"] == "accept"
    # packed-word authentication: every position a packed/v1 word of a row leaf under
    # Track A's framing; the format-3 witness built from the openings authenticates
    assert record["authenticatable"] is True and record["authenticity_reasons"] == []
    statement = decode_statement(bytes.fromhex(record["statement"]))
    schemas = {p.schema for p in statement.obligations[0].positions}
    assert all(s.startswith("packed/v1:verity-vllm/leaf/v1:u16le:") for s in schemas)
    assert record["witness_verdict"] == "accept"
    witness = decode_packed_witness(bytes.fromhex(record["witness"]))
    # three leaves in two trees (the replay-unit tree: x row and out row; the weights tree: the row block)
    assert sorted(len(tree) for tree in witness.trees) == [1, 2]
    assert record["subcircuits_well_formed"] == [True]
    # no pin: the weight Input gates are not sampled by this subcircuit
    assert record["lowering"]["obligations"][0]["expected"] == {}


def test_exporter_refuses_words_nobody_committed(tmp_path: Path) -> None:
    fixture_dir = _write_fixture(tmp_path)
    doc = json.loads((fixture_dir / "openings.json").read_text())
    value = bytearray(base64.b64decode(doc["x"]["value_b64"]))
    value[0] ^= 1
    doc["x"]["value_b64"] = base64.b64encode(bytes(value)).decode()
    (fixture_dir / "openings.json").write_text(json.dumps(doc))
    with pytest.raises(ExportError):
        export_gemm_coordinate(load_fixture(fixture_dir))


def test_a_wrong_committed_word_is_accepted_as_an_object_and_rejected_as_a_relation(tmp_path: Path) -> None:
    fixture_dir = _write_fixture(tmp_path, corrupt=True)
    fx = load_fixture(fixture_dir)
    obj, row, _ = export_gemm_coordinate(fx)
    gate_set = validate(obj)
    assert not check_relations(obj, [row], gate_set)
    # and the guest-facing checker: the leaves authenticate (they are what was committed),
    # the relation on the extracted words is what fails
    statement = to_statement(obj)
    proof = encode_packed_witness(packed_witness(fx, statement))
    with pytest.raises(Reject, match="relation_rejected"):
        TransparentBackend(gate_set).verify(statement, proof)


def test_packed_witness_negatives(tmp_path: Path) -> None:
    """The two negatives of the packed-word smoke test, on the synthetic fixture: a tampered
    leaf fails the path (invalid opening, before any relation), a shifted word index changes
    the supplied value (relation rejected)."""

    fx = load_fixture(_write_fixture(tmp_path))
    obj, _, _ = export_gemm_coordinate(fx)
    gate_set = validate(obj)
    statement = to_statement(obj)
    witness = packed_witness(fx, statement)
    TransparentBackend(gate_set).verify(statement, encode_packed_witness(witness))

    # 1. one bit of the weight row flipped in the witness
    trees = [list(t) for t in witness.trees]
    for tree in trees:
        for k, leaf in enumerate(tree):
            if leaf.canonical_id.startswith("w="):
                value = bytearray(leaf.value)
                value[6] ^= 0x01
                tree[k] = PackedLeaf(leaf.canonical_id, leaf.dtype, leaf.shape_text, bytes(value), leaf.path)
    tampered = PackedWitness(tuple(tuple(t) for t in trees))
    with pytest.raises(Reject, match="invalid_opening"):
        TransparentBackend(gate_set).verify(statement, encode_packed_witness(tampered))

    # 2. the statement's word index shifted by one for one operand (a different statement:
    #    the honest witness authenticates, the extracted word is another committed word)
    o = statement.obligations[0]
    positions = list(o.positions)
    p = positions[0]
    schema = parse_packed_schema(p.schema)
    assert schema is not None
    shifted = PositionRef(p.commitment, p.rank, p.position, packed_schema(schema.framing, schema.width, schema.little_endian, schema.count, (schema.index + 1) % schema.count), p.expected)
    positions[0] = shifted
    from dataclasses import replace

    statement2 = replace(statement, obligations=(replace(o, positions=tuple(positions)),))
    with pytest.raises(Reject, match="relation_rejected"):
        TransparentBackend(gate_set).verify(statement2, encode_packed_witness(witness))


def test_tree_helper_matches_the_adapter_paths(tmp_path: Path) -> None:
    leaves = [(f"w=p/c={r}", "bfloat16", (1, 2), bytes([r, 0, r, 0])) for r in range(5)]
    root, paths = _tree(leaves)
    for i, (c, d, s, v) in enumerate(leaves):
        assert len(paths[i]) == len(path_shape(len(leaves), i))
        assert fold_path(hashing.leaf_hash(c, d, s, v), i, paths[i]) == root
