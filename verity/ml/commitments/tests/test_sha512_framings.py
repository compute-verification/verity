"""``frame-v3-sha512`` and ``vllm-v1-sha512``: each reference reproduces its vectors, the variant is the SHA-256 framing's bytes
through SHA-512, openings verify only under their own hash, and a configuration never mixes hashes."""

from __future__ import annotations

import hashlib

import pytest

from verity.ml.commitments import vllm_v1 as ref
from verity.ml.commitments.frame_v3 import NAME_SHA512 as FRAME_SHA512
from verity.ml.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
from verity.ml.commitments.frame_v3 import vectors_sha512 as FV
from verity.ml.commitments.identity import identity_digest_sha512, tagged_sha512
from verity.ml.commitments.indexed import ExplicitIndexedDomain, RangeIndexedDomain, RangeU64Domain
from verity.ml.commitments.leaves import LIFT_TAG, NODE_TAG, H
from verity.ml.commitments.limits import VerificationLimits
from verity.ml.commitments.merkle import (RANGE_U64_MESSAGE_BYTES, Commitment, CommitmentDomain, MerkleTree, Opening, _hash, _uint,
                                          range_u64_domain_message, validate_commitment, verify_opening)
from verity.ml.commitments.multiproof import multiproof, verify_multiproof
from verity.ml.commitments.rowleaf import ROLE_X, SCHEMA_HM96_SHA512_ROW, SCHEMA_SHA512_ROW, row_leaf, sha512_row_digest
from verity.ml.commitments.scheme import CommitmentScheme
from verity.ml.commitments.vllm_v1 import vectors_sha512 as VV
from verity.errors import InvalidArtifact, ProtocolError

F = FV.load()
W = VV.load()


def _b(text: str) -> bytes:
    return bytes.fromhex(text)


def _path(entries: list[str | None]) -> tuple[bytes | None, ...]:
    return tuple(None if e is None else _b(e) for e in entries)


def _domain(doc: dict) -> CommitmentDomain:
    return CommitmentDomain(_b(doc["binding"]), doc["owner"], RangeIndexedDomain(*doc["range"]), hash=doc["hash"])


# --- frame-v3-sha512 ---------------------------------------------------------------------------------------------


def test_frame_vectors_file_is_what_the_reference_generates() -> None:
    assert FV.main(["--check"]) == 0, "rerun `python -m verity.ml.commitments.frame_v3.vectors_sha512`"


def test_a_sha512_domain_is_the_sha256_frame_through_sha512() -> None:
    d256 = CommitmentDomain(hashlib.sha256(b"x").digest(), 0, RangeIndexedDomain(4))
    d512 = CommitmentDomain(d256.binding, 0, RangeIndexedDomain(4), hash="sha512")
    assert d512 != d256 and len(d512.domain_id) == 64 and d512.digest_bytes == 64
    message = d512.leaf_prefix(0, 0, "u8", 3) + b"abc"
    framed = (32).to_bytes(8, "big") + d256.domain_id, (64).to_bytes(8, "big") + d512.domain_id
    assert d512.leaf_prefix(0, 0, "u8", 3) == d256.leaf_prefix(0, 0, "u8", 3).replace(*framed)
    assert d512.leaf(0, 0, "u8", b"abc") == hashlib.sha512(message).digest()
    with pytest.raises(ValueError):
        CommitmentDomain(d256.binding, 0, RangeIndexedDomain(4), hash="md5")


@pytest.mark.parametrize("case", F["row_leaves"], ids=lambda c: f"r{c['role']}-{c['word_bits']}b-{c['n_words']}")
def test_frame_row_leaf_vectors(case: dict) -> None:
    value = _b(case["row_bytes"])
    layout = RowLeaf("sha512", case["role"], case["word_bits"], case["n_words"]).layout()
    assert layout.hash == "sha512" and layout.prefix.hex() == case["prefix"] and len(layout.prefix) == 128
    assert layout.digest(value).hex() == case["digest"]
    step = case["word_bits"] // 8
    words = [int.from_bytes(value[i:i + step], "little") for i in range(0, len(value), step)]
    assert sha512_row_digest(words, case["word_bits"], case["role"]).hex() == case["digest"]


@pytest.mark.parametrize("case", F["row_trees"] + F["binding64_trees"],
                         ids=lambda c: f"binding{len(c['domain']['binding']) // 2}-n{len(c['rows'])}")
def test_frame_row_tree_vectors(case: dict) -> None:
    n = len(case["rows"])
    d = _domain(case["domain"])
    assert d.domain_id.hex() == case["domain"]["domain_id"]
    scheme = FrameV3({"rows": Port(d, RowLeaf("sha512", case["role"], case["word_bits"], case["n_words"]))})
    assert isinstance(scheme, CommitmentScheme) and scheme.name == FRAME_SHA512 == F["scheme"] and scheme.hashes == ("sha512",)
    digests = [scheme.leaf_layout("rows", r).digest(_b(v)) for r, v in enumerate(case["rows"])]
    assert [x.hex() for x in digests] == case["digests"]
    leaves = [scheme.tree_leaf("rows", r, dg) for r, dg in enumerate(digests)]
    assert [x.hex() for x in leaves] == case["leaves"]
    assert leaves == [row_leaf(d, r, dg, SCHEMA_SHA512_ROW) for r, dg in enumerate(digests)]
    root = scheme.root("rows", leaves)
    assert root.hex() == case["root"] and len(root) == 64
    for r in range(n):
        path = _path(case["paths"][r])
        assert scheme.verify_path("rows", root, n, r, leaves[r], path)
        assert not scheme.verify_path("rows", root, n, r, leaves[r][:32], path)
    tree = MerkleTree(d, {r: _b(v) for r, v in enumerate(case["rows"])}, lambda _p: "raw")
    op = tree.open(n - 1)
    assert len(tree.commitment.root) == 64 and verify_opening(d, tree.commitment, op, "raw", VerificationLimits())


@pytest.mark.parametrize("case", F["binding64_trees"], ids=lambda c: f"n{len(c['rows'])}")
def test_frame_binding64_vectors_spell_out_the_domain(case: dict) -> None:
    doc, n = case["domain"], len(case["rows"])
    binding = _b(doc["binding"])
    assert binding == identity_digest_sha512(case["binding_identity"]["tag"], case["binding_identity"]["doc"]) and len(binding) == 64
    assert _b(case["positions_sha512"]) == RangeIndexedDomain(*doc["range"]).identity_sha512()
    parts = (binding, _uint(doc["owner"] + 2), _b(case["positions_sha512"]), _uint(n))
    message = b"veritor/protocol/merkle/frame/v3\0" + (6).to_bytes(4, "big") + b"domain" + b"".join(len(p).to_bytes(8, "big") + p for p in parts)
    assert _b(case["domain_message"]) == message
    assert hashlib.sha512(message).hexdigest() == doc["domain_id"] == _domain(doc).domain_id.hex()


# --- range-u64 ---------------------------------------------------------------------------------------------------

_FRAME = b"veritor/protocol/merkle/frame/v3\0"
#: every count 0..512, and 2^k - 1, 2^k and 2^k + 1 up to the largest, 2^64 - 1
_U64_COUNTS = sorted({*range(513), *(2**k + e for k in range(64) for e in (-1, 0, 1)), 2**64 - 1})


def _range_u64(doc: dict) -> CommitmentDomain:
    return CommitmentDomain(_b(doc["binding"]), doc["owner"], RangeU64Domain(doc["start"], doc["count"]), hash=doc["hash"])


def _range_u64_message(binding: bytes, owner: int, start: int, count: int) -> bytes:
    """PROTOCOL.md §6's range-u64 layout, byte by byte."""
    words = ((owner + 2).to_bytes(8, "big"), start.to_bytes(8, "big"), count.to_bytes(8, "big"))
    return (_FRAME + (16).to_bytes(4, "big") + b"domain/range-u64" + (64).to_bytes(8, "big") + binding
            + b"".join((8).to_bytes(8, "big") + w for w in words))


def test_range_u64_vectors_pin_the_counts_and_share_their_first_block() -> None:
    cases = F["range_u64_domains"]
    assert [c["domain"]["count"] for c in cases] == [0, 1, 255, 256, 2**32, 2**64 - 1]
    assert len({_b(c["message"])[:128] for c in cases}) == 1


@pytest.mark.parametrize("case", F["range_u64_domains"], ids=lambda c: f"count{c['domain']['count']}")
def test_range_u64_vectors_spell_out_the_domain(case: dict) -> None:
    doc = case["domain"]
    binding, message, d = _b(doc["binding"]), _b(case["message"]), _range_u64(doc)
    assert binding == identity_digest_sha512(case["binding_identity"]["tag"], case["binding_identity"]["doc"]) and len(binding) == 64
    assert message == _range_u64_message(binding, doc["owner"], doc["start"], doc["count"])
    assert message == range_u64_domain_message(binding, doc["owner"], doc["start"], doc["count"])
    assert len(message) == RANGE_U64_MESSAGE_BYTES == 173 and message[165:] == doc["count"].to_bytes(8, "big")
    assert hashlib.sha512(message).hexdigest() == doc["domain_id"] == d.domain_id.hex()
    if doc["count"] == 0:
        empty = _FRAME + (5).to_bytes(4, "big") + b"empty" + (64).to_bytes(8, "big") + d.domain_id
        assert d.empty_root().hex() == case["empty_root"] == hashlib.sha512(empty).hexdigest()


def test_range_u64_message_is_173_bytes_at_every_count() -> None:
    binding = identity_digest_sha512("verity/test/range-u64", {})
    for count in _U64_COUNTS:
        for start in {0, 2**64 - 1 - count}:
            for owner in (-2, 4):
                message = range_u64_domain_message(binding, owner, start, count)
                assert len(message) == 173 and message[165:] == count.to_bytes(8, "big")
                assert message == _range_u64_message(binding, owner, start, count)


@pytest.mark.parametrize("case", F["range_u64_domains"], ids=lambda c: f"count{c['domain']['count']}")
def test_range_u64_id_differs_from_the_json_range_domains(case: dict) -> None:
    doc = case["domain"]
    start, count = doc["start"], doc["count"]
    u64 = _range_u64(doc)
    json_domain = CommitmentDomain(u64.binding, u64.owner, RangeIndexedDomain(start, start + count), hash="sha512")
    assert json_domain.count == u64.count
    for p in {start, start + count - 1} if count else ():
        assert json_domain.positions.rank(p) == u64.positions.rank(p)
    assert u64.domain_id != json_domain.domain_id


@pytest.mark.parametrize("case", F["range_u64_trees"], ids=lambda c: f"start{c['domain']['start']}-n{c['domain']['count']}")
def test_range_u64_tree_round_trips_through_verify_opening(case: dict) -> None:
    doc, positions, values = case["domain"], case["positions"], [_b(v) for v in case["values"]]
    d = _range_u64(doc)
    assert d.domain_id.hex() == doc["domain_id"] and positions == list(d.positions)
    leaves = [d.leaf(r, p, "u16", v) for r, (p, v) in enumerate(zip(positions, values, strict=True))]
    assert [x.hex() for x in leaves] == case["leaves"]
    assert leaves == [hashlib.sha512(d.leaf_prefix(r, p, "u16", 2) + v).digest() for r, (p, v) in enumerate(zip(positions, values, strict=True))]
    assert [d.pad(r).hex() for r in range(len(positions), 8)] == case["pads"]
    assert d.node(0, 0, leaves[0], leaves[1]) == _hash(b"node", d.domain_id, _uint(0), _uint(0), leaves[0], leaves[1], hash="sha512")
    tree = MerkleTree(d, dict(zip(positions, values, strict=True)), lambda _p: "u16")
    assert tree.commitment == Commitment(_b(case["root"]), doc["count"])
    json_domain = CommitmentDomain(d.binding, d.owner, RangeIndexedDomain(positions[0], positions[-1] + 1), hash="sha512")
    limits = VerificationLimits()
    for r, p in enumerate(positions):
        op = tree.open(p)
        assert [x.hex() for x in op.path] == case["paths"][r]
        assert verify_opening(d, tree.commitment, op, "u16", limits)
        assert not verify_opening(d, tree.commitment, Opening(p, bytes([op.value[0] ^ 1]) + op.value[1:], op.path), "u16", limits)
        assert not verify_opening(json_domain, tree.commitment, op, "u16", limits)
    for outside in (positions[0] - 1, positions[-1] + 1):
        assert not verify_opening(d, tree.commitment, Opening(outside, values[0], tree.open(positions[0]).path), "u16", limits)


def test_range_u64_refuses_other_hashes_short_bindings_and_counts_past_2_64() -> None:
    b64 = identity_digest_sha512("verity/test/range-u64", {})
    positions = RangeU64Domain(0, 4)
    for h in ("sha256", "blake3", "blake3-s256"):
        with pytest.raises(ProtocolError, match="SHA-512 only"):
            CommitmentDomain(b64, 0, positions, hash=h)
    with pytest.raises(ValueError):
        CommitmentDomain(b64, 0, positions, hash="sha384")
    for width in (0, 32, 63, 65, 128):
        with pytest.raises(ProtocolError, match="binding"):
            CommitmentDomain(bytes(width), 0, positions, hash="sha512")
    for start, count in ((0, 2**64), (1, 2**64 - 1), (2**64 - 1, 1), (-1, 1), (0, -1)):
        with pytest.raises(ProtocolError, match=r"2\^64"):
            range_u64_domain_message(b64, 0, start, count)
    for owner in (-3, 2**64 - 2):
        with pytest.raises(ProtocolError, match="owner"):
            range_u64_domain_message(b64, owner, 0, 1)


def test_every_frame_binds_a_64_byte_identity_apart_from_its_halves() -> None:
    b64 = identity_digest_sha512("verity/test/binding/v1", {"x": 1})
    positions = RangeIndexedDomain(4)
    d = CommitmentDomain(b64, -1, positions, hash="sha512")
    assert d.domain_id == _hash(b"domain", b64, _uint(1), positions.identity_sha512(), _uint(4), hash="sha512")
    s = CommitmentDomain(b64, -1, positions, hash="sha256")
    assert s.domain_id == _hash(b"domain", b64, _uint(1), bytes.fromhex(positions.identity_digest), _uint(4))
    for h in ("sha512", "sha256", "blake3", "blake3-s256"):
        whole = CommitmentDomain(b64, -1, positions, hash=h).domain_id
        assert whole != CommitmentDomain(b64[:32], -1, positions, hash=h).domain_id
        assert whole != CommitmentDomain(b64[32:], -1, positions, hash=h).domain_id
        for width in (0, 31, 33, 63, 65, 128):
            with pytest.raises(ProtocolError, match="binding"):
                CommitmentDomain(bytes(width), -1, positions, hash=h)
    with pytest.raises(ProtocolError, match="binding"):
        CommitmentDomain(bytearray(b64), -1, positions, hash="sha512")   # type: ignore[arg-type]


def test_frame_word_leaves_and_empty_root() -> None:
    for case in F["word_leaves"]:
        d = _domain(case["domain"])
        scheme = FrameV3({"y": Port(d, WordLeaf("u16", 2))})
        layout = scheme.leaf_layout("y", case["position"])
        assert layout.hash == "sha512" and layout.prefix.hex() == case["prefix"] and layout.digest(_b(case["value"])).hex() == case["leaf"]
    assert _domain(F["empty_root"]["domain"]).empty_root().hex() == F["empty_root"]["root"]


def test_frame_multiproofs_run_under_the_domains_hash() -> None:
    d = CommitmentDomain(hashlib.sha256(b"mp").digest(), 0, RangeIndexedDomain(9), hash="sha512")
    tree = MerkleTree(d, {p: bytes([p]) for p in range(9)}, lambda _p: "raw")
    ranks = [1, 4, 8]
    proof = multiproof(tree, ranks)
    leaves = {r: d.leaf(r, r, "raw", bytes([r])) for r in ranks}
    assert verify_multiproof(tree.commitment.root, 9, d.domain_id, leaves, proof, hash="sha512")
    assert not verify_multiproof(tree.commitment.root, 9, d.domain_id, leaves, proof)


def test_a_frame_configuration_never_mixes_hashes() -> None:
    d256 = CommitmentDomain(hashlib.sha256(b"a").digest(), 0, RangeIndexedDomain(2))
    d512 = CommitmentDomain(d256.binding, 0, RangeIndexedDomain(2), hash="sha512")
    with pytest.raises(ValueError):
        FrameV3({"a": Port(d256, WordLeaf("u8", 1)), "b": Port(d512, WordLeaf("u8", 1))})
    for row_hash in ("sha256", "blake3"):
        with pytest.raises(ValueError):
            FrameV3({"x": Port(d512, RowLeaf(row_hash, ROLE_X, 16, 4))})
    with pytest.raises(ValueError):
        FrameV3({"x": Port(d256, RowLeaf("sha512", ROLE_X, 16, 4))})
    with pytest.raises(ValueError):
        FrameV3({"x": Port(d512, RowLeaf("sha512", ROLE_X, 16, 4))}).tree_leaf("x", 0, bytes(32))


# --- vllm-v1-sha512 ----------------------------------------------------------------------------------------------


def test_vllm_vectors_file_is_what_the_reference_generates() -> None:
    assert VV.main(["--check"]) == 0, "rerun `python -m verity.ml.commitments.vllm_v1.vectors_sha512`"


def test_vllm_leaves_and_tree_hash_vectors() -> None:
    for case in W["hash"]:
        assert H(case["tag"], *[_b(p) for p in case["parts"]], hash="sha512").hex() == case["digest"]
    for case in W["pos_leaves"]:
        value = _b(case["value"])
        assert ref.pos_leaf(value, "sha512").hex() == case["leaf"]
        assert ref.pos_leaf_layout(len(value), "sha512").digest(value).hex() == case["leaf"]
        assert case["leaf"] == hashlib.sha512(ref.POS_LEAF_TAG + len(value).to_bytes(8, "big") + value).hexdigest()
    for case in W["chunk_leaves"]:
        assert ref.chunk_header(**case["fields"]).hex() == case["header"]
        assert ref.chunk_leaf(_b(case["header"]), _b(case["chunk"]), "sha512").hex() == case["leaf"]
    for case in W["thread_leaves"]:
        assert ref.thread_leaf(_b(case["header"]) + _b(case["words"]), "sha512").hex() == case["leaf"]


@pytest.mark.parametrize("case", W["trees"], ids=lambda c: f"n{c['leaf_count']}")
def test_vllm_tree_vectors(case: dict) -> None:
    leaves = [_b(x) for x in case["leaves"]]
    root = _b(case["root"])
    assert ref.TREE_SHA512.root(leaves) == root and len(root) == 64
    for i in range(case["leaf_count"]):
        path = _path(case["paths"][i])
        assert ref.TREE_SHA512.path(leaves, i) == path
        assert ref.TREE_SHA512.verify_path(root, case["leaf_count"], i, leaves[i], path)
        assert not ref.TREE.verify_path(root, case["leaf_count"], i, leaves[i], path)
    if case["leaf_count"] == 3:
        a, b, c = leaves
        assert root == H(NODE_TAG, H(NODE_TAG, a, b, hash="sha512"), H(LIFT_TAG, c, hash="sha512"), hash="sha512")


def test_vllm_root_vectors() -> None:
    fns = {"semantic": ref.semantic_root, "step": ref.step_root, "run": ref.run_root, "weights": ref.weights_root,
           "stream": ref.stream_root, "thread": ref.thread_root}
    for case in W["roots"]:
        i = case["inputs"]
        args = {
            "semantic": lambda: (_b(i["program_digest"]), _b(i["query_id"]), _b(i["template_digest"]), _b(i["ctx_digest"]), i["epoch"], i["n"],
                                 _b(i["tree_root"])),
            "step": lambda: (_b(i["program_digest"]), _b(i["ctx_digest"]), _b(i["geo_digest"]), _b(i["layout_digest"]), i["n"], _b(i["tree_root"])),
            "run": lambda: (_b(i["program_digest"]), _b(i["geo_digest"]), [_b(s) for s in i["step_roots"]]),
            "weights": lambda: (_b(i["geo_digest"]), _b(i["names_digest"]), [_b(t) for t in i["tensor_roots"]]),
            "stream": lambda: (_b(i["layout_digest"]), _b(i["ctx_digest"]), _b(i["tree_root"])),
            "thread": lambda: (_b(i["geo_digest"]), _b(i["ctx_digest"]), _b(i["tree_root"])),
        }[case["kind"]]()
        assert fns[case["kind"]](*args, hash="sha512").hex() == case["root"]
        if case["kind"] == "stream":
            assert ref.stream_ctx_digest(launch_tag=i["launch_tag"], src_mask=i["src_mask"], chunk_words=i["chunk_words"],
                                         leaf_count=i["leaf_count"], hash="sha512").hex() == i["ctx_digest"]
        if case["kind"] == "thread":
            assert ref.thread_ctx_digest(launch_tag=i["launch_tag"], src_mask=i["src_mask"], hash="sha512").hex() == i["ctx_digest"]


def test_vllm_openings_accept_exactly_the_honest_values() -> None:
    assert [(c["kind"], c["accept"]) for c in W["openings"]] == [(k, a) for k in ("semantic", "step", "stream") for a in (True, False)]


def test_vllm_configuration_names_its_hash_and_never_mixes() -> None:
    d = hashlib.sha256(b"cfg").digest()
    s512 = ref.StepDomain(d, d, d, d, 3, hash="sha512")
    scheme = ref.VllmV1({"v": s512}, value_nbytes={"v": lambda _i: 8})
    assert isinstance(scheme, CommitmentScheme)
    assert (scheme.name, scheme.hashes) == (ref.NAME_SHA512, ("sha512",)) == (W["scheme"], ("sha512",))
    leaves = [scheme.tree_leaf("v", i, scheme.leaf_layout("v", i).digest(bytes(8))) for i in range(3)]
    root = scheme.root("v", leaves)
    assert scheme.verify_path("v", root, 3, 1, leaves[1], scheme.path("v", leaves, 1))
    with pytest.raises(InvalidArtifact):
        scheme.tree_leaf("v", 0, bytes(32))
    with pytest.raises(InvalidArtifact):
        ref.VllmV1({"a": s512, "b": ref.StepDomain(d, d, d, d, 3)})
    assert ref.domain_digest(s512) != ref.domain_digest(ref.StepDomain(d, d, d, d, 3))
    assert ref.VllmV1({}).name == ref.NAME


# --- the red team's conditions on PR #93 ---------------------------------------------------------------------------


def test_the_tagged_sha512_identity_frame() -> None:
    tag, doc = "verity/test/v1", {"a": 1}
    payload = b'{"a":1}'
    expect = hashlib.sha512(b"verity/tagged-sha512/v1\0" + len(tag).to_bytes(4, "big") + tag.encode() + len(payload).to_bytes(8, "big") + payload)
    assert tagged_sha512(tag, payload) == identity_digest_sha512(tag, doc) == expect.digest()


def test_a_sha512_frame_absorbs_the_positions_sha512_identity() -> None:
    binding = hashlib.sha256(b"b").digest()
    for positions in (RangeIndexedDomain(2, 11, 3), ExplicitIndexedDomain([4, 9, 1])):
        d = CommitmentDomain(binding, 1, positions, hash="sha512")
        assert len(positions.identity_sha512()) == 64
        assert d.domain_id == _hash(b"domain", binding, _uint(3), positions.identity_sha512(), _uint(positions.count), hash="sha512")
        assert d.domain_id != _hash(b"domain", binding, _uint(3), bytes.fromhex(positions.identity_digest), _uint(positions.count), hash="sha512")

    class Opaque:
        count = 2
        identity_digest = "00" * 32

    with pytest.raises(ProtocolError):
        CommitmentDomain(binding, 0, Opaque(), hash="sha512")   # type: ignore[arg-type]


def test_validate_commitment_rejects_a_root_of_the_other_width() -> None:
    d256 = CommitmentDomain(hashlib.sha256(b"w").digest(), 0, RangeIndexedDomain(3))
    d512 = CommitmentDomain(d256.binding, 0, RangeIndexedDomain(3), hash="sha512")
    t256 = MerkleTree(d256, {p: bytes([p]) for p in range(3)}, lambda _p: "raw")
    t512 = MerkleTree(d512, {p: bytes([p]) for p in range(3)}, lambda _p: "raw")
    assert validate_commitment(d256, t256.commitment) and validate_commitment(d512, t512.commitment)
    assert not validate_commitment(d512, Commitment(t256.commitment.root, 3))
    assert not validate_commitment(d256, Commitment(t512.commitment.root, 3))
    assert not verify_opening(d512, Commitment(t512.commitment.root[:32], 3), t512.open(1), "raw", VerificationLimits())


def test_vllm_sha512_domain_digests_are_sha512() -> None:
    d = hashlib.sha256(b"dd").digest()
    s512 = ref.StepDomain(d, d, d, d, 3, hash="sha512")
    doc = {"kind": "step", "leaf_count": 3, "program_digest": d.hex(), "ctx_digest": d.hex(), "geo_digest": d.hex(), "layout_digest": d.hex(),
           "chunk": None, "hash": "sha512"}
    assert ref.domain_digest(s512) == identity_digest_sha512(ref.DOMAIN_DIGEST_TAG, doc)
    assert len(ref.domain_digest(s512)) == 64 and len(ref.domain_digest(ref.StepDomain(d, d, d, d, 3))) == 32
    assert all(len(_b(c["domain_digest"])) == 64 for c in W["openings"])


def test_hiding_row_schemas_are_named_at_their_widths() -> None:
    d256 = CommitmentDomain(hashlib.sha256(b"h").digest(), 0, RangeIndexedDomain(1))
    d512 = CommitmentDomain(d256.binding, 0, RangeIndexedDomain(1), hash="sha512")
    assert row_leaf(d512, 0, bytes(64), SCHEMA_HM96_SHA512_ROW) == d512.leaf(0, 0, "hm96-sha512/row/v1", bytes(64))
    assert row_leaf(d512, 0, bytes(64), SCHEMA_HM96_SHA512_ROW) != row_leaf(d512, 0, bytes(64), SCHEMA_SHA512_ROW)
    with pytest.raises(ValueError):
        row_leaf(d512, 0, bytes(32), SCHEMA_HM96_SHA512_ROW)
    with pytest.raises(ValueError, match="unknown row schema"):
        row_leaf(d256, 0, bytes(32), "hm96-sha256/row/v1")
