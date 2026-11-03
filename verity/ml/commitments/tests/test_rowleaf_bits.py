"""sha512/row/v2 (frame_v3/PROTOCOL.md §6, bit rows): the prefix and the bytes rebuilt by hand from the spec, the message decoding
to one bit string at every length (so the encoding is injective), element rows as the tensor's memory, the refusals, the empty
row, frame-v3-sha512 admitting the rows, and a hiding row opened from its root with ``verity.ml.commitments`` alone."""

from __future__ import annotations

import hashlib
import random

import pytest

from verity.ml.commitments import hm96
from verity.ml.commitments import rowleaf as R
from verity.ml.commitments.frame_v3 import NAME_SHA512, BitRowLeaf, FrameV3, Port, RowLeaf
from verity.ml.commitments.frame_v3 import vectors_sha512 as FV
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain

F = FV.load()
TAG = b"verity/sha512-row/v2\0"


def _b(text: str) -> bytes:
    return bytes.fromhex(text)


def _prefix(n_bits: int) -> bytes:
    return (TAG + n_bits.to_bytes(8, "big")).ljust(128, b"\0")


def _bits(values: list[int], bits: int) -> list[int]:
    return [v >> j & 1 for v in values for j in range(bits)]


def _parse(message: bytes) -> list[int]:
    """The bit string a hashed message encodes, refusing any message that is not ``P(l) || row`` with a zero fill: the left
    inverse that makes the encoding injective."""
    head, row = message[:128], message[128:]
    assert head[:len(TAG)] == TAG and head[len(TAG) + 8:] == bytes(128 - len(TAG) - 8)
    n_bits = int.from_bytes(head[len(TAG):len(TAG) + 8], "big")
    assert len(row) == (n_bits + 7) // 8
    bits = _bits(list(row), 8)
    assert not any(bits[n_bits:])
    return bits[:n_bits]


def _sha512_domain(n: int, label: str) -> CommitmentDomain:
    return CommitmentDomain(hashlib.sha256(label.encode()).digest(), 0, RangeIndexedDomain(n), hash="sha512")


def _domain(doc: dict) -> CommitmentDomain:
    d = CommitmentDomain(_b(doc["binding"]), doc["owner"], RangeIndexedDomain(*doc["range"]), hash=doc["hash"])
    assert d.domain_id.hex() == doc["domain_id"]
    return d


def test_the_prefix_is_the_tag_and_the_length_in_bits_in_one_block() -> None:
    assert R.SHA512_ROW_V2_TAG == TAG and TAG.count(0) == 1 and R.SCHEMA_SHA512_ROW_V2 == "sha512/row/v2"
    assert R.SCHEMA_HM96_SHA512_ROW_V2 == "hm96-sha512/row/v2"
    assert len(TAG) == len(R.SHA512_ROW_TAG) and TAG[:-1] != R.SHA512_ROW_TAG[:-1]
    for n in (0, 1, 7, 8, 1023, 1024, 1 << 32, (1 << 64) - 1):
        p = R.sha512_row_v2_prefix(n)
        assert p == _prefix(n) and len(p) == 128 and p[:len(TAG)] == TAG
        layout = R.sha512_row_v2_layout(n)
        assert (layout.hash, layout.prefix, layout.value_len, layout.suffix, layout.key) == ("sha512", p, -(-n // 8), b"", None)
    for bad in (-1, 1 << 64, True, 1.0, "8"):
        with pytest.raises(ValueError):
            R.sha512_row_v2_prefix(bad)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            BitRowLeaf(bad)  # type: ignore[arg-type]


def test_the_vectors_file_is_what_the_reference_generates() -> None:
    assert FV.main(["--check"]) == 0, "rerun `python -m verity.ml.commitments.frame_v3.vectors_sha512`"
    names = [c["name"] for c in F["bit_row_leaves"]]
    assert names == [f"bits/{n}" for n in FV.BIT_ROW_CASES] + [n for n, _spec in FV.FORMAT_ROW_CASES]
    lengths = {c["n_bits"] for c in F["bit_row_leaves"]}
    assert 0 in lengths and any(n % 8 for n in lengths) and any(0 < n < 1024 for n in lengths) and any(n > 2048 for n in lengths)


@pytest.mark.parametrize("case", F["bit_row_leaves"], ids=lambda c: c["name"])
def test_bit_row_vectors_rebuilt_from_the_spec(case: dict) -> None:
    n_bits, row = case["n_bits"], _b(case["row_bytes"])
    assert case["schema"] == "sha512/row/v2" and len(row) == (n_bits + 7) // 8
    assert case["prefix"] == _prefix(n_bits).hex() == R.sha512_row_v2_prefix(n_bits).hex()
    assert case["digest"] == hashlib.sha512(_prefix(n_bits) + row).hexdigest() == R.sha512_row_v2_digest(row, n_bits).hex()
    bits = _parse(_prefix(n_bits) + row)
    layout = BitRowLeaf(n_bits).layout()
    assert (layout.hash, layout.prefix, layout.value_len, layout.digest(row).hex()) == ("sha512", _b(case["prefix"]), len(row), case["digest"])
    if "parts" in case:
        parts = [(p["values"], p["bits"]) for p in case["parts"]]
        assert R.bit_row(*parts) == (row, n_bits)
        assert bits == [b for values, w in parts for b in _bits(values, w)]


def test_element_rows_are_the_tensors_memory() -> None:
    assert R.bit_row() == (b"", 0)
    assert R.bit_row(([1, 0, 1], 1)) == (b"\x05", 3)
    assert R.bit_row(([0xA, 0x3], 4)) == (b"\x3a", 8)
    assert R.bit_row(([1], 1), ([0x7F], 7), ([0x5], 3)) == (b"\xff\x05", 11)
    assert R.bit_row(([0x1234, 0xBEEF], 16)) == (b"\x34\x12\xef\xbe", 32)
    rng = random.Random(20261001)
    for k in (16, 32, 1536, 2048):
        codes, scales = [rng.randrange(16) for _ in range(k)], [rng.randrange(256) for _ in range(k // 16)]
        assert R.bit_row((codes, 4), (scales, 8)) == (R.nvfp4_row_bytes(codes, scales), 4 * k + 8 * (k // 16))
        mx = scales[:k // 32]
        assert R.bit_row((codes, 4), (mx, 8))[0] == bytes(codes[2 * i] | codes[2 * i + 1] << 4 for i in range(k // 2)) + bytes(mx)
        e4m3 = [rng.randrange(256) for _ in range(k)]
        assert R.bit_row((e4m3, 8)) == (bytes(e4m3), 8 * k)
        bf16 = [rng.randrange(1 << 16) for _ in range(k)]
        assert R.bit_row((bf16, 16)) == (b"".join(v.to_bytes(2, "little") for v in bf16), 16 * k)


def test_the_message_decodes_to_one_bit_string_at_every_length() -> None:
    """Every length to two blocks and a bit, with random bits: ``_parse`` recovers the row from ``P(l) || row``, so two rows with
    equal messages are the same bit string, and no ``P(l)`` equals a ``sha512/row/v1`` prefix (their tags differ)."""
    rng = random.Random(5)
    for n_bits in range(0, 2 * 1024 + 2):
        bits = [rng.randrange(2) for _ in range(n_bits)]
        row, n = R.bit_row((bits, 1))
        assert n == n_bits and _parse(R.sha512_row_v2_prefix(n_bits) + row) == bits
    v1 = {R.sha512_row_prefix(role, w, n)[:len(TAG)] for role in (R.ROLE_X, R.ROLE_W) for w in (8, 16, 32) for n in (1, 64, 1024)}
    assert v1 == {R.SHA512_ROW_TAG} and TAG not in v1


def test_the_empty_row_is_a_row() -> None:
    """l = 0 is allowed: the message is P(0) alone, which no other row's message equals, so the encoding stays injective and no
    length needs a special case.  The empty row, one zero bit and one zero byte hash the same row bytes or none, and differ."""
    assert R.sha512_row_v2_digest(b"", 0) == hashlib.sha512(_prefix(0)).digest() == BitRowLeaf(0).layout().digest(b"")
    assert F["bit_row_leaves"][0]["n_bits"] == 0 and F["bit_row_leaves"][0]["row_bytes"] == ""
    assert len({R.sha512_row_v2_digest(b"", 0), R.sha512_row_v2_digest(b"\0", 1), R.sha512_row_v2_digest(b"\0", 8)}) == 3


@pytest.mark.parametrize(("call", "why"), [
    (lambda: R.bit_row(([2], 1)), "does not fit 1 bits"),
    (lambda: R.bit_row(([16], 4)), "does not fit 4 bits"),
    (lambda: R.bit_row(([-1], 8)), "does not fit 8 bits"),
    (lambda: R.bit_row(([0], 0)), "positive number of bits"),
    (lambda: R.bit_row(([0], True)), "positive number of bits"),
    (lambda: R.sha512_row_v2_digest(b"\x00\x20", 13), "zero bits"),
    (lambda: R.sha512_row_v2_digest(b"\x80", 7), "zero bits"),
    (lambda: R.sha512_row_v2_digest(b"\x00", 9), "the layout says 2"),
    (lambda: R.sha512_row_v2_digest(b"\x00\x00", 8), "the layout says 1"),
])
def test_malformed_rows_are_refused(call, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        call()


def test_equal_bits_give_one_digest_and_the_tree_leaf_separates_ports() -> None:
    """The row digest binds the bits and their length only: an x row and a W column with equal bits have one digest.  The tree
    leaf binds the port's domain and the row's rank and schema, so the two rows' leaves still differ."""
    row, n_bits = R.bit_row(([3, 1, 4, 1, 5, 9, 2, 6], 4))
    digest = R.sha512_row_v2_digest(row, n_bits)
    dx, dw = _sha512_domain(2, "ports/x"), _sha512_domain(2, "ports/w")
    fv = FrameV3({"x": Port(dx, BitRowLeaf(n_bits)), "w": Port(dw, BitRowLeaf(n_bits))})
    assert fv.leaf_layout("x", 0) == fv.leaf_layout("w", 0) == R.sha512_row_v2_layout(n_bits)
    leaves = {fv.tree_leaf("x", 0, digest), fv.tree_leaf("x", 1, digest), fv.tree_leaf("w", 0, digest)}
    assert len(leaves) == 3
    assert R.row_leaf(dx, 0, digest, R.SCHEMA_SHA512_ROW_V2) != R.row_leaf(dx, 0, digest, R.SCHEMA_SHA512_ROW)


@pytest.mark.parametrize("case", F["bit_row_trees"], ids=lambda c: str(c["n_bits"]))
def test_bit_row_tree_vectors(case: dict) -> None:
    n, d = len(case["rows"]), _domain(case["domain"])
    scheme = FrameV3({"rows": Port(d, BitRowLeaf(case["n_bits"]))})
    assert scheme.name == NAME_SHA512 and scheme.hashes == ("sha512",) and scheme.ports["rows"].leaf.schema == case["schema"]
    digests = [scheme.leaf_layout("rows", r).digest(_b(v)) for r, v in enumerate(case["rows"])]
    assert [x.hex() for x in digests] == case["digests"]
    leaves = [scheme.tree_leaf("rows", r, dg) for r, dg in enumerate(digests)]
    assert [x.hex() for x in leaves] == case["leaves"] == [R.row_leaf(d, r, dg, case["schema"]).hex() for r, dg in enumerate(digests)]
    root = scheme.root("rows", leaves)
    assert root.hex() == case["root"]
    for r in range(n):
        assert scheme.verify_path("rows", root, n, r, leaves[r], tuple(_b(p) for p in case["paths"][r]))


@pytest.mark.parametrize("case", F["bit_row_hiding_leaves"], ids=lambda c: c["name"])
def test_a_hiding_bit_row_opens_from_its_root_with_core_alone(case: dict) -> None:
    """The opening is the row's bytes, its length, its salt and its path.  The checker recomputes the row digest, ``b || c``, the
    hm96 leaf and the frame leaf under ``hm96-sha512/row/v2``, then checks the path.  Each tampering fails."""
    inst, n_bits, rank, d = hm96.SHA512, case["n_bits"], case["rank"], _domain(case["domain"])
    assert hashlib.sha512(inst.default_key).hexdigest() == case["key_sha512"]
    leaf_kind = BitRowLeaf(n_bits)
    assert (leaf_kind.hiding_schema, leaf_kind.schema) == (case["schema"], case["inner"]) == ("hm96-sha512/row/v2", "sha512/row/v2")
    row, salt = _b(case["row_bytes"]), _b(case["salt"])

    def leaf_of(row: bytes, salt: bytes, n_bits: int = n_bits, schema: str = case["schema"]) -> bytes:
        bc = inst.commit_string(inst.default_key, R.sha512_row_v2_digest(row, n_bits), salt)
        return R.row_leaf(d, rank, inst.tree_leaf(inst.default_key, bc), schema)

    assert R.sha512_row_v2_digest(row, n_bits).hex() == case["digest"]
    assert inst.commit_string(inst.default_key, _b(case["digest"]), salt).hex() == case["bc"]
    assert inst.tree_leaf(inst.default_key, _b(case["bc"])).hex() == case["hm96_leaf"]
    assert leaf_of(row, salt).hex() == case["leaf"]

    scheme = FrameV3({"rows": Port(d, leaf_kind)})
    other = R.row_leaf(d, 0, inst.tree_leaf(inst.default_key, bytes(128)), case["schema"])
    leaves = [other, _b(case["leaf"])]
    root = scheme.root("rows", leaves)
    path = scheme.path("rows", leaves, rank)
    assert scheme.verify_path("rows", root, 2, rank, leaf_of(row, salt), path)
    shorter = bytes(row[:-1]) + bytes([row[-1] & (1 << (n_bits - 1) % 8) - 1]) if (n_bits - 1) % 8 else row[:-1]
    tampered = [leaf_of(bytes([row[0] ^ 1]) + row[1:], salt), leaf_of(row, bytes([salt[0] ^ 1]) + salt[1:]),
                leaf_of(shorter, salt, n_bits - 1), leaf_of(row + b"\0", salt, n_bits + 8),
                leaf_of(row, salt, schema=case["inner"]), leaf_of(row, salt, schema=R.SCHEMA_HM96_SHA512_ROW)]
    assert not any(scheme.verify_path("rows", root, 2, rank, t, path) for t in tampered)


def test_frame_v3_admits_bit_rows_under_sha512_only() -> None:
    d256 = CommitmentDomain(hashlib.sha256(b"b").digest(), 0, RangeIndexedDomain(2), hash="sha256")
    d512 = _sha512_domain(2, "b")
    for n_bits in (0, 13, 144, 9216):
        with pytest.raises(ValueError):
            FrameV3({"x": Port(d256, BitRowLeaf(n_bits))})
        fv = FrameV3({"x": Port(d512, BitRowLeaf(n_bits)), "w": Port(d512, RowLeaf("sha512", R.ROLE_W, 16, 8))})
        assert fv.name == NAME_SHA512 and fv.leaf_layout("x", 1) == R.sha512_row_v2_layout(n_bits)
        with pytest.raises(ValueError):
            fv.tree_leaf("x", 0, bytes(32))
        assert fv.tree_leaf("x", 0, bytes(64)) == R.row_leaf(d512, 0, bytes(64), R.SCHEMA_SHA512_ROW_V2)


def test_row_leaf_takes_the_v2_schemas_at_64_bytes_and_each_gives_its_own_leaf() -> None:
    d = _sha512_domain(1, "row-leaf")
    schemas = [R.SCHEMA_SHA512_ROW_V2, R.SCHEMA_HM96_SHA512_ROW_V2, R.SCHEMA_SHA512_ROW, R.SCHEMA_HM96_SHA512_ROW]
    assert len({R.row_leaf(d, 0, bytes(64), s) for s in schemas}) == len(schemas)
    for s in schemas[:2]:
        with pytest.raises(ValueError):
            R.row_leaf(d, 0, bytes(32), s)
