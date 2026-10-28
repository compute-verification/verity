"""frame-v3 leaves: the reference reproduces its vectors, the leaf layouts describe exactly what the tree binds, and the
plain-Python BLAKE3 is BLAKE3."""

from __future__ import annotations

import hashlib

import pytest

from verity.ml.crypto import blake3 as b3
from verity.ml.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
from verity.ml.commitments.frame_v3 import vectors as V
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree
from verity.ml.commitments.rowleaf import SCHEMA_BLAKE3_ROW, SCHEMA_SHA256_ROW, blake3_row_digest, row_leaf, sha256_row_digest
from verity.ml.commitments.scheme import HASH_CLASS, CommitmentScheme, LeafLayout

VECTORS = V.load()


def _b(text: str) -> bytes:
    return bytes.fromhex(text)


def test_vectors_file_is_what_the_reference_generates() -> None:
    assert V.main(["--check"]) == 0, "rerun `python -m verity.ml.commitments.frame_v3.vectors`"


def test_blake3_known_answer() -> None:
    assert b3.blake3(b"").hex() == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"


def test_blake3_matches_the_package() -> None:
    blake3 = pytest.importorskip("blake3")
    for case in VECTORS["blake3"]:
        data = bytes(i % 251 for i in range(case["len"]))
        assert blake3.blake3(data).hexdigest() == case["hash"]
        assert blake3.blake3(data, key=_b(case["key"])).hexdigest() == case["keyed_hash"]


@pytest.mark.parametrize("case", VECTORS["row_leaves"], ids=lambda c: f"{c['hash']}-r{c['role']}-{c['word_bits']}b-{c['n_words']}")
def test_row_leaf_vectors(case: dict) -> None:
    value = _b(case["row_bytes"])
    layout = RowLeaf(case["hash"], case["role"], case["word_bits"], case["n_words"]).layout()
    assert layout.prefix.hex() == case["prefix"]
    assert (layout.key.hex() if layout.key else None) == case["key"]
    assert layout.digest(value).hex() == case["digest"]
    step = case["word_bits"] // 8
    words = [int.from_bytes(value[i:i + step], "little") for i in range(0, len(value), step)]
    fn = sha256_row_digest if case["hash"] == "sha256" else blake3_row_digest
    assert fn(words, case["word_bits"], case["role"]).hex() == case["digest"]


@pytest.mark.parametrize("case", VECTORS["row_trees"], ids=lambda c: f"{c['hash']}-n{len(c['rows'])}")
def test_row_tree_vectors(case: dict) -> None:
    n = len(case["rows"])
    d = CommitmentDomain(_b(case["domain"]["binding"]), case["domain"]["owner"], RangeIndexedDomain(*case["domain"]["range"]), hash="sha256")
    assert d.domain_id.hex() == case["domain"]["domain_id"]
    scheme = FrameV3({"rows": Port(d, RowLeaf(case["hash"], case["role"], case["word_bits"], case["n_words"]))})
    leaves = [_b(x) for x in case["leaves"]]
    for r in range(n):
        digest = _b(case["digests"][r])
        assert scheme.leaf_layout("rows", r).digest(_b(case["rows"][r])) == digest
        assert scheme.tree_leaf("rows", r, digest) == leaves[r] == row_leaf(d, r, digest, case["schema"])
        path = tuple(_b(p) for p in case["paths"][r])
        assert scheme.verify_path("rows", _b(case["root"]), n, r, leaves[r], path)
        assert not scheme.verify_path("rows", _b(case["root"]), n, r, scheme.tree_leaf("rows", r, bytes(32)), path)
    tree = MerkleTree(d, {r: _b(case["digests"][r]) for r in range(n)}, lambda _p: case["schema"])
    assert tree.commitment.root.hex() == case["root"]


def test_word_leaf_layout_is_the_frame_v3_leaf() -> None:
    for case in VECTORS["word_leaves"]:
        d = CommitmentDomain(_b(case["domain"]["binding"]), case["domain"]["owner"], RangeIndexedDomain(*case["domain"]["range"]), hash="sha256")
        scheme = FrameV3({"y": Port(d, WordLeaf(case["schema"], 2))})
        layout = scheme.leaf_layout("y", case["position"])
        assert layout.prefix.hex() == case["prefix"]
        assert layout.digest(_b(case["value"])).hex() == case["leaf"]
        assert scheme.tree_leaf("y", case["position"], _b(case["leaf"])).hex() == case["leaf"]


def test_sha256_prefix_is_one_block_and_binds_role_width_length() -> None:
    layouts = {(role, bits, n): RowLeaf("sha256", role, bits, n).layout() for role in (1, 2) for bits in (8, 16) for n in (4, 8)}
    assert all(len(lay.prefix) == 64 for lay in layouts.values())
    assert len({lay.prefix for lay in layouts.values()}) == len(layouts)
    assert sha256_row_digest([0] * 16, 8, 1) != sha256_row_digest([0] * 8, 16, 1) != sha256_row_digest([0] * 8, 16, 2)
    prefix = layouts[(1, 8, 8)].prefix
    assert sha256_row_digest([7] * 8, 8, 1) == hashlib.sha256(prefix + bytes([7] * 8)).digest()


def test_configuration_reports_its_hashes() -> None:
    d = CommitmentDomain(bytes(32), 0, RangeIndexedDomain(2), hash="sha256")
    assert FrameV3({"x": Port(d, RowLeaf("sha256", 1, 16, 4))}).hashes == ("sha256",)
    both = FrameV3({"x": Port(d, RowLeaf("sha256", 1, 16, 4)), "w": Port(d, RowLeaf("blake3", 2, 16, 4))})
    assert both.hashes == ("sha256", "blake3") and all(HASH_CLASS[h] == "standard" for h in both.hashes)
    assert isinstance(both, CommitmentScheme)
    assert {RowLeaf("sha256", 1, 8, 1).schema, RowLeaf("blake3", 1, 8, 1).schema} == {SCHEMA_SHA256_ROW, SCHEMA_BLAKE3_ROW}
    with pytest.raises(ValueError):
        RowLeaf("poseidon2", 1, 16, 4)
    with pytest.raises(ValueError):
        LeafLayout("sha256", b"", 1, key=bytes(32))
