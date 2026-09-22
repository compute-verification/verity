"""Capture format v2: row-block weight leaves, coordinate weight ids, range openings, conversion."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from _helpers import TINY_MANIFEST, make_tiny_run

from verity_vllm import commit, hashing, identity, weights_v2
from verity_vllm.capture import CaptureError, CaptureReader, RunWriter, WeightsReader
from verity_vllm.negatives import PatchedReader


def _v2_run(run_dir: Path, rows_per_leaf: int = 1) -> tuple[RunWriter, dict[str, np.ndarray]]:
    run = RunWriter(run_dir, TINY_MANIFEST)
    arrays = {
        "embed_tokens.weight": np.arange(16 * 4, dtype=np.uint16).reshape(16, 4),
        "layers.0.mlp.down_proj.weight": (np.arange(6 * 8, dtype=np.uint16) * 3).reshape(6, 8),
        "layers.0.input_layernorm.weight": np.arange(8, dtype=np.uint16) + 100,
    }
    w = run.weights(store_values=True, format_version=2)
    w.add("embed_tokens.weight", "bfloat16", [16, 4], arrays["embed_tokens.weight"], rows_per_leaf=rows_per_leaf)
    w.add(
        "layers.0.mlp.down_proj.weight",
        "bfloat16",
        [6, 8],
        arrays["layers.0.mlp.down_proj.weight"],
        rows_per_leaf=rows_per_leaf,
    )
    w.add("layers.0.input_layernorm.weight", "bfloat16", [8], arrays["layers.0.input_layernorm.weight"])
    w.finish()
    run.finish()
    return run, arrays


# --- ids ------------------------------------------------------------------------------------------


def test_weight_id_coordinate_round_trip() -> None:
    wid = identity.weight_id("layers.0.mlp.down_proj.weight", None, 5)
    s = str(wid)
    assert s.endswith("/c=5")
    parsed = identity.parse(s)
    assert int(parsed.coordinate) == 5
    assert identity.format_id(parsed) == s
    base = identity.weight_id("layers.0.mlp.down_proj.weight")
    assert str(base) + "/c=5" == s
    with pytest.raises(identity.IdentityError):
        identity.weight_id("x.weight", None, "05")
    with pytest.raises(identity.IdentityError):
        identity.weight_id("x.weight", None, -1)
    # tree order is code-point order (frozen): rows of one tensor are contiguous but "c=10" < "c=2";
    # the report order is numeric
    ids = [str(identity.weight_id("w", None, 10)), str(identity.weight_id("w", None, 2)), str(base)]
    assert identity.tree_sorted(ids) == [str(base), ids[0], ids[1]]
    assert sorted(ids, key=identity.structured_key) == [str(base), ids[1], ids[0]]


# --- writer / reader ------------------------------------------------------------------------------


def test_v2_round_trip_and_leaves(tmp_path: Path) -> None:
    run, arrays = _v2_run(tmp_path)
    reader = CaptureReader(tmp_path)
    w = reader.weights
    assert w.index.format_version == 2
    assert (tmp_path / "weights" / "leaves.bin").exists()
    assert w.index.leaf_count() == 16 + 6 + 1
    ids = w.ids()
    assert len(ids) == 16 + 6 + 1
    whole = str(identity.weight_id("embed_tokens.weight"))
    row3 = str(identity.weight_id("embed_tokens.weight", None, 3))
    assert whole not in ids  # a rows-layout tensor has no whole-tensor leaf
    assert row3 in ids
    assert w.is_leaf(row3) and not w.is_leaf(whole)
    assert whole in w  # ...but the whole tensor is still readable
    # whole read is the concatenation of the rows; row read is the row's bytes
    raw = arrays["embed_tokens.weight"].tobytes()
    assert w.read(whole) == raw
    assert w.read(row3) == arrays["embed_tokens.weight"][3].tobytes()
    v = w.value(row3)
    assert v.dtype == "bfloat16" and tuple(v.shape) == (1, 4)
    v_whole = w.value(whole)
    assert tuple(v_whole.shape) == (16, 4)
    # stored leaves == recomputed leaves == hashing.leaf_hash of the row block under the row id
    stored = w.leaves()
    recomputed = w.leaves(recompute=True)
    assert stored == recomputed
    assert stored[row3] == hashing.leaf_hash(row3, "bfloat16", (1, 4), arrays["embed_tokens.weight"][3].tobytes())
    vec = str(identity.weight_id("layers.0.input_layernorm.weight"))
    assert stored[vec] == hashing.leaf_hash(vec, "bfloat16", (8,), arrays["layers.0.input_layernorm.weight"].tobytes())
    assert w.validate(check_bytes=True) == []
    # index.json is loadable by the v1-only leaves() path? no: it must refuse
    with pytest.raises(CaptureError):
        w.index.leaves()


def test_v2_rows_per_leaf_blocks(tmp_path: Path) -> None:
    _v2_run(tmp_path, rows_per_leaf=4)
    w = CaptureReader(tmp_path).weights
    # 16 rows / 4 = 4 blocks; 6 rows / 4 = 2 blocks (last has 2 rows); 1 vector
    assert w.index.leaf_count() == 4 + 2 + 1
    e = w.entry(str(identity.weight_id("layers.0.mlp.down_proj.weight")))
    assert e.block_of(5) == (4, 6) and e.block_shape(4, 6) == (2, 8) and e.block_shape(0, 4) == (4, 8)
    b1 = str(identity.weight_id("layers.0.mlp.down_proj.weight", None, 4))
    assert w.is_leaf(b1)
    mid = str(identity.weight_id("layers.0.mlp.down_proj.weight", None, 5))
    assert mid not in w  # only block starts are addressable ids
    with pytest.raises(KeyError):
        w.is_leaf(mid)
    # a coordinate inside a block resolves to the block id
    assert w.leaf_id_for_row(str(identity.weight_id("layers.0.mlp.down_proj.weight")), 5) == b1
    assert w.value(b1).shape == (2, 8)


def test_v1_bundles_still_load(tiny_run: Path) -> None:
    w = CaptureReader(tiny_run).weights
    assert w.index.format_version == 1
    assert w.index.leaf_count() == 2
    assert w.leaves() == w.index.leaves() == w.leaves(recompute=True)
    assert "layout" not in json.loads((tiny_run / "weights" / "index.json").read_text())["entries"][0]


# --- openings -------------------------------------------------------------------------------------


def test_row_opening_and_range_opening(tmp_path: Path) -> None:
    _v2_run(tmp_path)
    reader = CaptureReader(tmp_path)
    doc = commit.build_commitments(reader)
    assert doc["trees"]["weights"]["leaf_count"] == 23
    cache = commit.TreeCache(reader)
    row3 = str(identity.weight_id("embed_tokens.weight", None, 3))
    o = cache.open(row3)
    assert isinstance(o, commit.Opening)
    assert commit.verify_opening_in(doc, o)
    assert len(o.value) == 8  # one bf16 row of 4
    # opening the whole tensor of a rows-layout entry is a range opening over its 16 leaves
    whole = str(identity.weight_id("embed_tokens.weight"))
    r = cache.open(whole)
    assert isinstance(r, commit.RangeOpening)
    assert r.hi - r.lo == 16 and len(r.leaves) == 16
    assert commit.verify_opening_in(doc, r)
    assert r.as_value().shape == (16, 4)
    assert len(r.value) == 16 * 8
    # bytes: the range carries one sibling per level at most on each side, not 16 paths
    assert len(r.path) < 16 * len(o.path)
    # tamper: a leaf byte (value and leaf together), the aggregate alone, or the range bounds -> reject
    root = bytes.fromhex(doc["trees"]["weights"]["root"])
    flipped = r.value[:-1] + bytes([r.value[-1] ^ 1])
    leaves = list(r.leaves)
    c, d, s, b = leaves[-1]
    leaves[-1] = (c, d, s, b[:-1] + bytes([b[-1] ^ 1]))
    assert not commit.verify_range_opening(root, 23, r.replace(value=flipped, leaves=tuple(leaves)))
    assert not commit.verify_range_opening(root, 23, r.replace(value=flipped))
    assert not commit.verify_range_opening(root, 23, r.replace(lo=r.lo + 1, hi=r.hi + 1))
    assert not commit.verify_range_opening(root, 23, r.replace(root=bytes(32)))
    # JSON round trip (generic loader dispatches on kind)
    r2 = commit.opening_from_json(json.loads(json.dumps(r.to_json())))
    assert isinstance(r2, commit.RangeOpening) and commit.verify_opening_in(doc, r2)
    o2 = commit.opening_from_json(json.loads(json.dumps(o.to_json())))
    assert isinstance(o2, commit.Opening) and commit.verify_opening_in(doc, o2)
    # a vector (whole leaf) opens as a plain Opening
    vec = str(identity.weight_id("layers.0.input_layernorm.weight"))
    ov = cache.open(vec)
    assert isinstance(ov, commit.Opening) and commit.verify_opening_in(doc, ov)
    # verify_commitments / source linkage are clean
    assert commit.verify_commitments(reader, doc) == []
    assert commit.verify_source_linkage(reader, doc) == []


def test_range_opening_every_window(tmp_path: Path) -> None:
    """Every [lo, hi) window of a small tree verifies; shifted windows do not."""
    _v2_run(tmp_path)
    reader = CaptureReader(tmp_path)
    tree = commit.weights_tree(reader)
    n = tree.leaf_count
    hashes = [tree.leaves[cid] for cid in tree.ids]
    for lo in range(n):
        for hi in range(lo + 1, n + 1):
            path = tree.range_path(lo, hi)
            assert commit.fold_range(hashes[lo:hi], n, lo, path) == tree.root, (lo, hi)
            if hi < n:
                shifted_ok = True
                try:
                    shifted_ok = commit.fold_range(hashes[lo:hi], n, lo + 1, path) == tree.root
                except commit.CommitError:
                    shifted_ok = False
                assert not shifted_ok, (lo, hi)
    # a tree with an odd number of leaves at several levels (23 here) exercises the lift rule;
    # also check a power-of-two tree and a 1-leaf tree
    for count in (1, 2, 5, 8, 13):
        t = commit.MerkleTree({f"w=p{i}.weight": hashing.leaf_hash(f"w=p{i}.weight", "int32", (), (i).to_bytes(4, "little")) for i in range(count)}, "t")
        hs = [t.leaves[c] for c in t.ids]
        for lo in range(count):
            for hi in range(lo + 1, count + 1):
                assert commit.fold_range(hs[lo:hi], count, lo, t.range_path(lo, hi)) == t.root


# --- negatives: patched rows ----------------------------------------------------------------------


def test_patched_weights_v2_change_only_their_leaf(tmp_path: Path) -> None:
    _v2_run(tmp_path)
    reader = CaptureReader(tmp_path)
    base = commit.weights_tree(reader)
    row3 = str(identity.weight_id("embed_tokens.weight", None, 3))
    original = reader.weights.read(row3)
    mutated = bytes([original[0] ^ 1]) + original[1:]
    patched = PatchedReader(reader, weights={row3: mutated})
    assert patched.weights.read(row3) == mutated
    whole = str(identity.weight_id("embed_tokens.weight"))
    assert patched.weights.read(whole)[24:32] == mutated
    leaves = patched.weights.leaves()
    base_leaves = reader.weights.leaves()
    changed = {k for k in leaves if leaves[k] != base_leaves[k]}
    assert changed == {row3}
    tree = commit.weights_tree(patched, recompute=True)
    assert tree.root != base.root
    # patching through the whole-tensor id also works
    whole_mut = bytearray(reader.weights.read(whole))
    whole_mut[0] ^= 1
    patched2 = PatchedReader(reader, weights={whole: bytes(whole_mut)})
    l2 = patched2.weights.leaves()
    assert {k for k in l2 if l2[k] != base_leaves[k]} == {str(identity.weight_id("embed_tokens.weight", None, 0))}


# --- conversion -----------------------------------------------------------------------------------


def test_convert_v1_bundle_to_v2(tmp_path: Path) -> None:
    src = tmp_path / "v1"
    make_tiny_run(src)
    v1_doc = commit.build_commitments(CaptureReader(src))
    dst = tmp_path / "v2"
    info = weights_v2.convert_bundle(src, dst, rows_per_leaf=1)
    assert info["from"]["leaf_count"] == 2 and info["from"]["root"] == v1_doc["trees"]["weights"]["root"]
    assert info["to"]["leaf_count"] == 16 + 4 and info["values_sha256_unchanged"]
    r2 = CaptureReader(dst)
    assert r2.weights.index.format_version == 2
    assert (dst / "weights" / "values.bin").read_bytes() == (src / "weights" / "values.bin").read_bytes()
    doc = r2.commitments
    assert doc["trees"]["weights"]["root"] == info["to"]["root"]
    for tid in v1_doc["trees"]:
        if tid != "weights":
            assert doc["trees"][tid] == v1_doc["trees"][tid]
    assert commit.verify_commitments(r2, doc) == []
    assert commit.verify_source_linkage(r2, doc) == []
    # replay units still readable through hard links
    assert sorted(r2.ru_keys()) == sorted(CaptureReader(src).ru_keys())
    # the v1 reader's whole-tensor ids resolve to the same bytes in v2
    whole = str(identity.weight_id("embed_tokens.weight"))
    assert r2.weights.read(whole) == CaptureReader(src).weights.read(whole)
    with pytest.raises(CaptureError):
        weights_v2.convert_bundle(src, dst)
