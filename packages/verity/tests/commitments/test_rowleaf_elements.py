"""sha512/row-nvfp4/v1, sha512/row-mxfp4/v1 and sha512/row-e4m3/v1 (frame_v3/PROTOCOL.md §6, element rows): the prefix and the
bytes rebuilt by hand from the spec, NVFP4's bytes being §4a's, the range checks, one prefix per (schema, role, k) and none equal to
a ``sha512/row/v1`` prefix, frame-v3-sha512 admitting the rows, and a hiding row opened from its root with ``verity.commitments``
alone."""

from __future__ import annotations

import hashlib
import random

import pytest

from verity.commitments import hm96
from verity.commitments import rowleaf as R
from verity.commitments.frame_v3 import NAME_SHA512, ElementRowLeaf, FrameV3, Port, RowLeaf
from verity.commitments.frame_v3 import vectors_sha512 as FV
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain

F = FV.load()
#: the spec's table: tag, code bits, codes per scale (0: no scales)
SPEC = {"nvfp4": (b"verity/sha512-row-nvfp4/v1\0", 4, 16), "mxfp4": (b"verity/sha512-row-mxfp4/v1\0", 4, 32),
        "e4m3": (b"verity/sha512-row-e4m3/v1\0", 8, 0)}
#: past every cell's K (16384), with the old frames' word counts over the same range
K_MAX = 20480


def _b(text: str) -> bytes:
    return bytes.fromhex(text)


def _prefix(fmt: str, role: int, k: int) -> bytes:
    tag, bits, group = SPEC[fmt]
    return (tag + bytes([role, bits]) + k.to_bytes(4, "big") + (k // group if group else 0).to_bytes(4, "big")).ljust(128, b"\0")


def _decode(fmt: str, k: int, row: bytes) -> tuple[list[int], list[int]]:
    _tag, bits, group = SPEC[fmt]
    n = k * bits // 8
    codes = [c for b in row[:n] for c in (b & 15, b >> 4)] if bits == 4 else list(row[:n])
    scales = list(row[n:])
    assert len(codes) == k and len(scales) == (k // group if group else 0)
    return codes, scales


def _parse(prefix: bytes) -> tuple[str, int, int, int, int]:
    """(schema, role, code bits, k, n_scales) from an element-row prefix, reading the tag up to its zero byte."""
    end = prefix.index(0)
    tag = prefix[:end + 1]
    (fmt,) = [f for f, (t, _b, _g) in SPEC.items() if t == tag]
    role, bits = prefix[end + 1], prefix[end + 2]
    k, n = int.from_bytes(prefix[end + 3:end + 7], "big"), int.from_bytes(prefix[end + 7:end + 11], "big")
    assert len(prefix) == 128 and not any(prefix[end + 11:])
    return R.ELEMENT_FORMATS[fmt].schema, role, bits, k, n


def _sha512_domain(n: int, label: str) -> CommitmentDomain:
    return CommitmentDomain(hashlib.sha256(label.encode()).digest(), 0, RangeIndexedDomain(n), hash="sha512")


def test_the_formats_and_tags_are_the_spec_table() -> None:
    assert {f: (x.tag, x.code_bits, x.scale_group) for f, x in R.ELEMENT_FORMATS.items()} == SPEC
    assert [(x.code, x.scale) for x in (R.NVFP4, R.MXFP4, R.E4M3)] == [("e2m1", "ue4m3"), ("e2m1", "ue8m0"), ("e4m3", None)]
    assert [x.schema for x in R.ELEMENT_FORMATS.values()] == ["sha512/row-nvfp4/v1", "sha512/row-mxfp4/v1", "sha512/row-e4m3/v1"]
    assert [x.hiding_schema for x in R.ELEMENT_FORMATS.values()] == [f"hm96-sha512/row-{f}/v1" for f in ("nvfp4", "mxfp4", "e4m3")]
    assert all(t.count(0) == 1 and t.endswith(b"\0") for t, _b, _g in SPEC.values())
    tags = [t for t, _b, _g in SPEC.values()] + [R.SHA512_ROW_TAG]
    assert not any(a != b and b.startswith(a) for a in tags for b in tags)


def test_the_vectors_file_is_what_the_reference_generates() -> None:
    assert FV.main(["--check"]) == 0, "rerun `python -m verity.commitments.frame_v3.vectors_sha512`"
    assert {(c["format"], c["role"], c["k"]) for c in F["element_row_leaves"]} == set(FV.ELEMENT_CASES)


@pytest.mark.parametrize("case", F["element_row_leaves"], ids=lambda c: f"{c['format']}-r{c['role']}-k{c['k']}")
def test_element_row_vectors_rebuilt_from_the_spec(case: dict) -> None:
    fmt, role, k, row = case["format"], case["role"], case["k"], _b(case["row_bytes"])
    _tag, bits, group = SPEC[fmt]
    assert len(row) == k * bits // 8 + (k // group if group else 0) == R.ELEMENT_FORMATS[fmt].row_len(k)
    assert case["prefix"] == _prefix(fmt, role, k).hex() == R.sha512_element_row_prefix(fmt, role, k).hex()
    assert case["digest"] == hashlib.sha512(_prefix(fmt, role, k) + row).hexdigest()
    assert _parse(_b(case["prefix"])) == (case["schema"], role, bits, k, case["n_scales"])
    codes, scales = _decode(fmt, k, row)
    assert R.element_row_bytes(fmt, codes, scales) == row
    assert R.sha512_element_row_digest(fmt, codes, scales, role).hex() == case["digest"]
    layout = ElementRowLeaf(fmt, role, k).layout()
    assert (layout.hash, layout.prefix, layout.value_len, layout.digest(row).hex()) == ("sha512", _b(case["prefix"]), len(row), case["digest"])


def test_nvfp4_element_rows_are_section_4a_bytes() -> None:
    rng = random.Random(20261001)
    for k in (16, 1536, 2048):
        codes, scales = [rng.randrange(16) for _ in range(k)], [rng.randrange(256) for _ in range(k // 16)]
        assert R.element_row_bytes(R.NVFP4, codes, scales) == R.element_row_bytes("nvfp4", codes, scales) == R.nvfp4_row_bytes(codes, scales)
        assert R.sha512_element_row_layout("nvfp4", R.ROLE_X, k).value_len == R.sha256_row_nvfp4_layout(R.ROLE_X, k).value_len


@pytest.mark.parametrize(("fmt", "codes", "scales", "why"), [
    ("nvfp4", [0] * 15, [0], "multiple of 16"),
    ("mxfp4", [0] * 16, [], "multiple of 32"),
    ("e4m3", [], [], "positive number of codes"),
    ("nvfp4", [0] * 16, [0, 0], "scale bytes"),
    ("mxfp4", [0] * 64, [0], "scale bytes"),
    ("e4m3", [0] * 4, [0], "scale bytes"),
    ("nvfp4", [16] + [0] * 15, [0], "E2M1 code is 4 bits"),
    ("mxfp4", [-1] + [0] * 31, [0], "E2M1 code is 4 bits"),
    ("e4m3", [256], [], "E4M3 code is 8 bits"),
    ("nvfp4", [0] * 16, [256], "UE4M3 scale is one byte"),
    ("mxfp4", [0] * 32, [-1], "UE8M0 scale is one byte"),
    ("bf16", [0] * 16, [], "element formats"),
])
def test_malformed_element_rows_are_refused(fmt: str, codes: list[int], scales: list[int], why: str) -> None:
    with pytest.raises(ValueError, match=why):
        R.element_row_bytes(fmt, codes, scales)


@pytest.mark.parametrize(("fmt", "role", "k"), [("nvfp4", 3, 16), ("nvfp4", 0, 16), ("mxfp4", 1, 48), ("e4m3", 1, 0), ("e4m3", 2, 1 << 32),
                                                ("fp8", 1, 16)])
def test_malformed_prefixes_and_leaves_are_refused(fmt: str, role: int, k: int) -> None:
    with pytest.raises(ValueError):
        R.sha512_element_row_prefix(fmt, role, k)
    with pytest.raises(ValueError):
        ElementRowLeaf(fmt, role, k)


def test_one_prefix_is_one_schema_role_and_k_and_never_a_word_row_prefix() -> None:
    """Every element-row prefix up to K_MAX (each format, role and k) against every ``sha512/row/v1`` prefix over the same lengths
    (roles 1 and 2, words of 8, 16 and 32 bits): the old frames' ports are all among the latter.  No two are equal, and each
    element-row prefix parses back to its own (schema, role, k)."""
    seen: dict[bytes, tuple] = {}
    for fmt, (_tag, bits, group) in SPEC.items():
        step = group or 1
        for role in (R.ROLE_X, R.ROLE_W):
            for k in range(step, K_MAX + 1, step):
                p = R.sha512_element_row_prefix(fmt, role, k)
                assert p not in seen, (fmt, role, k, seen.get(p))
                seen[p] = (fmt, role, k)
                if k in (step, K_MAX) or k % 2048 == 0:
                    assert _parse(p) == (R.ELEMENT_FORMATS[fmt].schema, role, bits, k, k // group if group else 0)
    new = len(seen)
    for role in (R.ROLE_X, R.ROLE_W):
        for bits in (8, 16, 32):
            for n in range(1, K_MAX + 1):
                p = R.sha512_row_prefix(role, bits, n)
                assert p not in seen, ("sha512/row/v1", role, bits, n, seen.get(p))
                seen[p] = ("sha512/row/v1", role, bits, n)
    assert new == 2 * (K_MAX // 16 + K_MAX // 32 + K_MAX) and len(seen) == new + 6 * K_MAX


def test_equal_bytes_in_another_format_role_or_length_give_another_digest() -> None:
    rng = random.Random(7)
    row = bytes(rng.randrange(256) for _ in range(R.NVFP4.row_len(2048)))
    digests = {R.sha512_element_row_layout(fmt, role, k).digest(row[:R.ELEMENT_FORMATS[fmt].row_len(k)])
               for fmt, k in (("nvfp4", 2048), ("e4m3", 1152), ("e4m3", 1024), ("mxfp4", 2048)) for role in (R.ROLE_X, R.ROLE_W)}
    words = {R.sha512_row_layout(role, bits, 1152 * 8 // bits).digest(row) for role in (R.ROLE_X, R.ROLE_W) for bits in (8, 16, 32)}
    assert len(digests) == 8 and len(words) == 6 and not digests & words


@pytest.mark.parametrize("case", F["element_row_trees"], ids=lambda c: c["format"])
def test_element_row_tree_vectors(case: dict) -> None:
    n = len(case["rows"])
    doc = case["domain"]
    d = CommitmentDomain(_b(doc["binding"]), doc["owner"], RangeIndexedDomain(*doc["range"]), hash=doc["hash"])
    assert d.domain_id.hex() == doc["domain_id"]
    scheme = FrameV3({"rows": Port(d, ElementRowLeaf(case["format"], case["role"], case["k"]))})
    assert scheme.name == NAME_SHA512 and scheme.hashes == ("sha512",) and scheme.ports["rows"].leaf.schema == case["schema"]
    digests = [scheme.leaf_layout("rows", r).digest(_b(v)) for r, v in enumerate(case["rows"])]
    assert [x.hex() for x in digests] == case["digests"]
    leaves = [scheme.tree_leaf("rows", r, dg) for r, dg in enumerate(digests)]
    assert [x.hex() for x in leaves] == case["leaves"] == [R.row_leaf(d, r, dg, case["schema"]).hex() for r, dg in enumerate(digests)]
    root = scheme.root("rows", leaves)
    assert root.hex() == case["root"]
    for r in range(n):
        assert scheme.verify_path("rows", root, n, r, leaves[r], tuple(_b(p) for p in case["paths"][r]))


@pytest.mark.parametrize("case", F["element_hiding_leaves"], ids=lambda c: c["format"])
def test_a_hiding_element_row_opens_from_its_root_with_core_alone(case: dict) -> None:
    """The opening is the row's bytes, its salt and its path. The checker decodes the codes and scales and recomputes the row
    digest, ``b || c``, the hm96 leaf and the frame leaf under the hiding schema, then checks the path. Each tampering fails."""
    inst, fmt, role, k, rank = hm96.SHA512, case["format"], case["role"], case["k"], case["rank"]
    assert hashlib.sha512(inst.default_key).hexdigest() == case["key_sha512"]
    doc = case["domain"]
    d = CommitmentDomain(_b(doc["binding"]), doc["owner"], RangeIndexedDomain(*doc["range"]), hash=doc["hash"])
    leaf_kind = ElementRowLeaf(fmt, role, k)
    assert (leaf_kind.hiding_schema, leaf_kind.schema) == (case["schema"], case["inner"])
    row, salt = _b(case["row_bytes"]), _b(case["salt"])

    def leaf_of(row: bytes, salt: bytes, role: int = role, schema: str = case["schema"]) -> bytes:
        codes, scales = _decode(fmt, k, row)
        bc = inst.commit_string(inst.default_key, R.sha512_element_row_digest(fmt, codes, scales, role), salt)
        return R.row_leaf(d, rank, inst.tree_leaf(inst.default_key, bc), schema)

    codes, scales = _decode(fmt, k, row)
    assert R.sha512_element_row_digest(fmt, codes, scales, role).hex() == case["digest"]
    assert inst.commit_string(inst.default_key, _b(case["digest"]), salt).hex() == case["bc"]
    assert inst.tree_leaf(inst.default_key, _b(case["bc"])).hex() == case["hm96_leaf"]
    assert leaf_of(row, salt).hex() == case["leaf"]

    scheme = FrameV3({"rows": Port(d, leaf_kind)})
    other = R.row_leaf(d, 0, inst.tree_leaf(inst.default_key, bytes(128)), case["schema"])
    leaves = [other, _b(case["leaf"])]
    root = scheme.root("rows", leaves)
    path = scheme.path("rows", leaves, rank)
    assert scheme.verify_path("rows", root, 2, rank, leaf_of(row, salt), path)
    flipped = bytes([row[0] ^ 1]) + row[1:]
    tampered = [leaf_of(flipped, salt), leaf_of(row, bytes([salt[0] ^ 1]) + salt[1:]), leaf_of(row, salt, role=3 - role),
                leaf_of(row, salt, schema=case["inner"]), leaf_of(row, salt, schema=R.SCHEMA_HM96_SHA512_ROW)]
    assert not any(scheme.verify_path("rows", root, 2, rank, t, path) for t in tampered)


def test_frame_v3_admits_element_rows_under_sha512_only() -> None:
    d256 = CommitmentDomain(hashlib.sha256(b"e").digest(), 0, RangeIndexedDomain(2))
    d512 = _sha512_domain(2, "e")
    for fmt, k in (("nvfp4", 16), ("mxfp4", 32), ("e4m3", 1)):
        with pytest.raises(ValueError):
            FrameV3({"x": Port(d256, ElementRowLeaf(fmt, R.ROLE_X, k))})
        fv = FrameV3({"x": Port(d512, ElementRowLeaf(fmt, R.ROLE_X, k)), "w": Port(d512, RowLeaf("sha512", R.ROLE_W, 16, 8))})
        assert fv.name == NAME_SHA512 and fv.leaf_layout("x", 1) == R.sha512_element_row_layout(fmt, R.ROLE_X, k)
        with pytest.raises(ValueError):
            fv.tree_leaf("x", 0, bytes(32))
        assert fv.tree_leaf("x", 0, bytes(64)) == R.row_leaf(d512, 0, bytes(64), R.ELEMENT_FORMATS[fmt].schema)


def test_row_leaf_takes_each_element_schema_at_64_bytes_and_each_gives_its_own_leaf() -> None:
    d = _sha512_domain(1, "row-leaf")
    schemas = [s for f in R.ELEMENT_FORMATS.values() for s in (f.schema, f.hiding_schema)] + [R.SCHEMA_SHA512_ROW, R.SCHEMA_HM96_SHA512_ROW]
    leaves = {R.row_leaf(d, 0, bytes(64), s) for s in schemas}
    assert len(leaves) == len(schemas)
    for s in schemas[:6]:
        with pytest.raises(ValueError):
            R.row_leaf(d, 0, bytes(32), s)
