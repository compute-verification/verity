"""The leaf commitment scheme: trees over whole tensor leaves, openings, and what an opening binds."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity.ml.commitments.leaves import (
    LeafOpening,
    LeafTree,
    LeafValue,
    empty_root,
    fold_path,
    ids_digest,
    leaf_hash,
    parse_shape_text,
    path_shape,
    shape_text,
    tree_depth,
    verify_opening,
)
from verity.errors import InvalidArtifact

FIXTURES = next(p for p in Path(__file__).resolve().parents if (p / "uv.lock").exists()) / "fixtures" / "typed-obligation-v0"


def _values(n: int) -> list[LeafValue]:
    return [LeafValue(f"row/{i:03d}", "bfloat16", (4,), bytes([i, 0, i, 1, i, 2, i, 3])) for i in range(n)]


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_every_leaf_opens_and_authenticates(n: int) -> None:
    values = _values(n)
    tree = LeafTree.from_values(values, "t")
    assert tree.leaf_count == n and tree.depth == tree_depth(n)
    for value in values:
        opening = tree.open(value)
        assert opening.index == tree.index_of(value.canonical_id)
        assert len(opening.path) == len(path_shape(n, opening.index))
        assert verify_opening(tree.root, n, opening)
        assert fold_path(opening.leaf(), opening.index, opening.path) == tree.root


def test_tree_order_is_code_point_order_and_the_ids_digest_is_the_domain() -> None:
    tree = LeafTree.from_values([LeafValue(cid, "int32", (), b"\0\0\0\0") for cid in ("b", "a", "ab")], "t")
    assert tree.ids == ["a", "ab", "b"]
    assert tree.ids_digest == ids_digest(["b", "ab", "a"])
    assert tree.record("weights")["ids_digest"] == tree.ids_digest.hex()


def test_odd_trailing_nodes_are_lifted_not_duplicated() -> None:
    values = _values(3)
    tree = LeafTree.from_values(values, "t")
    last = tree.open(values[2])
    assert last.path[0] is None  # leaf 2 has no sibling at level 0: lifted
    assert last.path[1] is not None
    # a "sibling" where the shape says lift, or a lift where it says sibling, is refused before hashing
    assert not verify_opening(tree.root, 3, last.replace(path=(bytes(32), last.path[1])))
    first = tree.open(values[0])
    assert not verify_opening(tree.root, 3, first.replace(path=(None, first.path[1])))


def test_an_opening_binds_id_dtype_shape_and_bytes() -> None:
    values = _values(4)
    tree = LeafTree.from_values(values, "t")
    opening = tree.open(values[1])
    assert verify_opening(tree.root, 4, opening)
    for changes in (
        {"canonical_id": "row/999"},
        {"dtype": "float16"},
        {"shape": (2, 2)},
        {"value": opening.value[:-1] + bytes([opening.value[-1] ^ 1])},
        {"index": 2},
        {"root": bytes(32)},
        {"leaf_count": 5},
    ):
        assert not verify_opening(tree.root, 4, opening.replace(**changes)), changes
    with pytest.raises(InvalidArtifact):
        tree.open(LeafValue(values[1].canonical_id, values[1].dtype, values[1].shape, b"\xff" * 8))
    with pytest.raises(InvalidArtifact):
        LeafTree.from_values(values + [values[0]])


def test_opening_json_round_trip() -> None:
    tree = LeafTree.from_values(_values(5), "t")
    opening = tree.open(_values(5)[4])
    again = LeafOpening.from_json(json.loads(json.dumps(opening.to_json())))
    assert again == opening
    with pytest.raises(InvalidArtifact):
        LeafOpening.from_json({"tree": "t"})


def test_shape_text_and_empty_root() -> None:
    assert shape_text(()) == "[]" and shape_text((4, 8)) == "[4,8]"
    assert parse_shape_text("[4,8]") == (4, 8) and parse_shape_text("[]") == ()
    for bad in ("4,8", "[04]", "[a]"):
        with pytest.raises(InvalidArtifact):
            parse_shape_text(bad)
    with pytest.raises(InvalidArtifact):
        shape_text((True,))
    assert LeafTree({}, "empty").root == empty_root()
    assert leaf_hash("x", "bfloat16", (2,), b"ab") == leaf_hash("x", "bfloat16", "[2]", b"ab")


def test_recorded_stage2_leaves_authenticate_against_their_recorded_roots() -> None:
    """The serving side's real openings (the ``stage2--*`` conformance vectors, witness format 3) fold to the
    roots the objects name: the scheme here is the one those roots were committed under."""

    from verity_sp1.proofs.packed import decode_packed_witness
    from verity_sp1.proofs.statement import Statement
    from verity_sp1.proofs.wire import decode_statement

    checked = 0
    for path in sorted(FIXTURES.glob("stage2--*.json")):
        record = json.loads(path.read_text())
        if "witness" not in record or record["verdict"] != "accept":
            continue
        statement: Statement = decode_statement(bytes.fromhex(record["statement"]))
        witness = decode_packed_witness(bytes.fromhex(record["witness"]))
        roots = {bytes.fromhex(root["root"]): root for root in record["object"]["roots"]}
        from verity_sp1.proofs.packed import packed_table

        for tree, leaves in zip(packed_table(statement).trees, witness.trees, strict=True):
            root = roots[tree.commitment.root]
            for rank, leaf in zip(tree.ranks, leaves, strict=True):
                opening = LeafOpening(
                    root["id"], tree.commitment.root, root["count"], rank,
                    leaf.canonical_id, leaf.dtype, parse_shape_text(leaf.shape_text), leaf.value, tuple(leaf.path),
                )
                assert verify_opening(tree.commitment.root, root["count"], opening), (path.name, leaf.canonical_id)
                checked += 1
    assert checked >= 3
