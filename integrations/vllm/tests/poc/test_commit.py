from __future__ import annotations

import json

import numpy as np
import pytest
from _helpers import TINY_MANIFEST

from verity_vllm import commit, hashing
from verity_vllm.capture import CaptureReader, RunWriter, Value
from verity_vllm.commit import MerkleTree, Opening, path_shape, verify_opening
from verity_vllm.identity import value_id

PID = "0123456789ab"


def leaf_values(n: int) -> list[Value]:
    return [
        Value(
            str(value_id(PID, "req-0", t, "prefill", "0", "layers.0.mlp.residual", 0)),
            "bfloat16",
            (2,),
            (t * 0x0101).to_bytes(2, "little") + b"\x00\x00",
        )
        for t in range(n)
    ]


def test_tree_vectors(tree_vectors):
    for case in tree_vectors["trees"]:
        n = case["leaf_count"]
        values = leaf_values(n)
        tree = MerkleTree.from_values(values, "ru/req-0/000000")
        assert tree.ids == case["ids"]
        assert [tree.leaves[c].hex() for c in tree.ids] == case["leaves"]
        assert tree.root.hex() == case["root"]
        assert tree.ids_digest.hex() == case["ids_digest"]
        for value, expected in zip(values, case["openings"], strict=True):
            opening = tree.open(value)
            assert opening.index == expected["index"]
            assert [None if p is None else p.hex() for p in opening.path] == expected["path"]
            assert verify_opening(tree.root, n, opening)
            assert verify_opening(tree.root, n, Opening.from_json(opening.to_json()))


def test_path_shape_and_lift_semantics():
    assert path_shape(1, 0) == []
    assert path_shape(2, 0) == [True]
    assert path_shape(3, 2) == [False, True]  # lifted at level 0, then paired with node(0,1)
    assert path_shape(3, 0) == [True, True]
    assert path_shape(5, 4) == [False, False, True]
    assert path_shape(9, 8) == [False, False, False, True]
    with pytest.raises(commit.CommitError):
        path_shape(3, 3)
    # three leaves: root = node(node(l0, l1), lift(l2)); nothing is duplicated
    values = leaf_values(3)
    tree = MerkleTree.from_values(values)
    l0, l1, l2 = (v.leaf() for v in values)
    assert tree.root == hashing.node_hash(hashing.node_hash(l0, l1), hashing.lift_hash(l2))
    assert tree.root != hashing.node_hash(hashing.node_hash(l0, l1), hashing.node_hash(l2, l2))
    assert MerkleTree({}).root == hashing.empty_root()


def test_tree_order_is_independent_of_insertion_order():
    values = leaf_values(6)
    a = MerkleTree.from_values(values)
    b = MerkleTree.from_values(reversed(values))
    assert a.root == b.root and a.ids == b.ids


def test_verify_rejects_tampering():
    values = leaf_values(5)
    tree = MerkleTree.from_values(values, "ru/req-0/000000")
    root, n = tree.root, tree.leaf_count
    good = tree.open(values[3])
    assert verify_opening(root, n, good)
    # N01: altered value, root/path kept
    bad_value = good.replace(value=b"\xff" + good.value[1:])
    assert not verify_opening(root, n, bad_value)
    # N03: relabel under another id (same bytes)
    bad_id = good.replace(canonical_id=values[1].canonical_id)
    assert not verify_opening(root, n, bad_id)
    # N04: reinterpret dtype or shape
    assert not verify_opening(root, n, good.replace(dtype="float16"))
    assert not verify_opening(root, n, good.replace(shape=(1, 2)))
    # wrong index, wrong root, wrong leaf count
    assert not verify_opening(root, n, good.replace(index=2))
    assert not verify_opening(bytes(32), n, good.replace(root=bytes(32)))
    # the opening's self-description must agree with the verifier's parameters
    assert not verify_opening(root, n, good.replace(leaf_count=n + 1))
    # path shape: turning the lift into a sibling, or dropping a level
    last = tree.open(values[4])
    assert last.path[0] is None
    forged = last.replace(path=(bytes(32),) + last.path[1:])
    assert not verify_opening(root, n, forged)
    # a leaf count that implies another path shape for this index is rejected
    assert not verify_opening(root, n + 1, last.replace(leaf_count=n + 1))
    assert not verify_opening(root, n, last.replace(path=last.path[1:]))
    # non-canonical id never verifies
    assert not verify_opening(root, n, good.replace(canonical_id="x=1"))
    # opening a value whose bytes differ from the committed leaf is refused at open time
    with pytest.raises(commit.CommitError):
        tree.open(Value(values[0].canonical_id, "bfloat16", (2,), b"\x01\x02\x03\x04"))


def test_build_and_verify_commitments(tiny_run):
    doc = commit.build_commitments(tiny_run)
    reader = CaptureReader(tiny_run)
    assert (tiny_run / "commitments.json").exists()
    assert set(doc["trees"]) == {"ru/req-0/000000", "ru/req-0/000001", "boundary/req-0", "weights"}
    assert doc["trees"]["ru/req-0/000000"]["leaf_count"] == 7
    assert doc["trees"]["weights"]["leaf_count"] == 2
    assert doc["hash"]["leaf"] == "verity-vllm/leaf/v1"
    assert commit.verify_commitments(reader, doc) == []
    assert commit.load_commitments(tiny_run) == doc
    cache = commit.TreeCache(reader)
    for cid in [
        str(value_id(reader.profile_id, "req-0", 1, "prefill", "embed", "embed_tokens", 0)),
        str(value_id(reader.profile_id, "req-0", 3, "decode", "sampler", "sampler.select", 0)),
        "w=lm_head.weight",
        f"p={reader.profile_id}/r=req-0/b=output_token/t=3",
        f"p={reader.profile_id}/r=req-0/b=seed",
    ]:
        opening = cache.open(cid)
        assert commit.verify_opening_in(doc, opening), cid
        other = "weights" if opening.tree != "weights" else "boundary/req-0"
        assert not commit.verify_opening_in(doc, opening.replace(tree=other))
    assert (
        commit.verify_source_linkage(reader, doc) == []
        or commit.verify_source_linkage(reader, doc)
        == ["weights/index.json source_sha256 != manifest.model.weights_sha256"] * 0
    )
    # trusting index leaves gives the same roots as rehashing
    fast = commit.build_commitments(reader, recompute_leaves=False, write=False)
    assert fast["trees"] == doc["trees"]


def test_verify_commitments_detects_changes(tiny_run):
    doc = commit.build_commitments(tiny_run)
    reader = CaptureReader(tiny_run)
    values = tiny_run / "ru" / "req-0" / "000001" / "values.bin"
    raw = bytearray(values.read_bytes())
    raw[0] ^= 1
    values.write_bytes(bytes(raw))
    problems = commit.verify_commitments(reader, doc)
    assert any("ru/req-0/000001: root mismatch" in p for p in problems)
    tampered = json.loads(json.dumps(doc))
    del tampered["trees"]["weights"]
    assert any("weights" in p for p in commit.verify_commitments(reader, tampered))


def test_tainted_unit_is_refused(tmp_path):
    run = RunWriter(tmp_path, TINY_MANIFEST)
    pid = run.profile_id
    cid = str(value_id(pid, "req-0", 0, "prefill", "embed", "embed_tokens", 0))
    ru = run.ru("req-0", 0, phase="prefill", positions=[0], on_duplicate="flag")
    ru.add(cid, "bfloat16", [2], bytes(4))
    ru.add(cid, "bfloat16", [2], b"\x01\x00\x00\x00")
    ru.finish()
    with pytest.raises(commit.CommitError):
        commit.build_commitments(tmp_path)


def test_source_linkage(tmp_path):
    manifest = dict(TINY_MANIFEST)
    raw = np.arange(8, dtype=np.uint16).tobytes()
    manifest["model"] = {
        "repo": "toy",
        "weights_sha256": {"weights/values.bin": hashing.sha256_hex(raw)},
    }
    run = RunWriter(tmp_path, manifest)
    w = run.weights(store_values=True)
    w.add("embed_tokens.weight", "bfloat16", [4, 2], raw)
    w.finish()
    run.finish()
    doc = commit.build_commitments(tmp_path)
    reader = CaptureReader(tmp_path)
    assert commit.verify_source_linkage(reader, doc) == []
    # substitute the stored weight bytes, keep everything else
    (tmp_path / "weights" / "values.bin").write_bytes(raw[::-1])
    problems = commit.verify_source_linkage(CaptureReader(tmp_path), doc)
    assert any("values.bin sha256" in p for p in problems) and any(
        "sha256 != bytes" in p for p in problems
    )
