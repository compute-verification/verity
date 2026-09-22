from __future__ import annotations

import hashlib
import math
import struct

import numpy as np
import pytest

from verity_vllm import dtypes, hashing


def reference_H(tag: str, *parts: bytes) -> bytes:
    """The documented byte layout, written independently of verity_vllm.hashing.H."""
    stream = struct.pack(">I", len(tag.encode())) + tag.encode()
    for part in parts:
        stream += struct.pack(">Q", len(part)) + part
    return hashlib.sha256(stream).digest()


def test_H_matches_documented_layout(hash_vectors):
    for case in hash_vectors["H"]:
        parts = [bytes.fromhex(p) for p in case["parts_hex"]]
        assert hashing.H(case["tag"], *parts).hex() == case["digest"]
        assert reference_H(case["tag"], *parts).hex() == case["digest"]


def test_tag_is_a_domain_separator():
    assert hashing.H("a", b"b") != hashing.H("ab", b"")
    assert hashing.H("a", b"b", b"") != hashing.H("a", b"b")
    assert hashing.H("a", b"bc") != hashing.H("a", b"b", b"c")


def test_leaf_vectors(hash_vectors):
    for case in hash_vectors["leaf"]:
        raw = bytes.fromhex(case["input_hex"])
        assert hashing.shape_text(case["shape"]) == case["shape_text"]
        assert (
            hashing.leaf_hash(case["canonical_id"], case["dtype"], case["shape"], raw).hex()
            == (case["leaf"])
        )
        assert (
            reference_H(
                "verity-vllm/leaf/v1",
                case["canonical_id"].encode(),
                case["dtype"].encode(),
                case["shape_text"].encode(),
                raw,
            ).hex()
            == case["leaf"]
        )


def test_node_lift_ids_empty_vectors(hash_vectors):
    node = hash_vectors["node"]
    assert (
        hashing.node_hash(bytes.fromhex(node["left"]), bytes.fromhex(node["right"])).hex()
        == (node["digest"])
    )
    lift = hash_vectors["lift"]
    assert hashing.lift_hash(bytes.fromhex(lift["child"])).hex() == lift["digest"]
    ids = hash_vectors["ids"]
    assert hashing.ids_digest(ids["ids"]).hex() == ids["digest"]
    assert hashing.empty_root().hex() == hash_vectors["empty"]
    jd = hash_vectors["json_digest"]
    assert hashing.canonical_json_bytes(jd["document"]).decode() == jd["canonical"]
    assert hashing.json_digest(jd["document"]).hex() == jd["digest"]
    assert hashing.profile_id(jd["document"]) == jd["profile_id"]
    assert (
        hashing.profile_id(jd["document"])
        == hashlib.sha256(jd["canonical"].encode()).hexdigest()[:12]
    )


def test_node_rejects_non_digests():
    with pytest.raises(ValueError):
        hashing.node_hash(b"short", bytes(32))
    with pytest.raises(ValueError):
        hashing.lift_hash(bytes(31))


def test_shape_text_round_trip():
    for shape in [(), (1,), (4, 8), (0,), (10, 0, 3)]:
        assert hashing.parse_shape_text(hashing.shape_text(shape)) == shape
    with pytest.raises(ValueError):
        hashing.shape_text((-1,))
    with pytest.raises(ValueError):
        hashing.parse_shape_text("[01]")
    with pytest.raises(ValueError):
        hashing.parse_shape_text("[1, 2]")


def test_canonical_json_is_stable():
    a = hashing.canonical_json_bytes({"b": 1, "a": [1, {"d": 2, "c": 3}]})
    b = hashing.canonical_json_bytes({"a": [1, {"c": 3, "d": 2}], "b": 1})
    assert a == b == b'{"a":[1,{"c":3,"d":2}],"b":1}'
    with pytest.raises(ValueError):
        hashing.canonical_json_bytes({"x": float("nan")})


# --- dtypes -------------------------------------------------------------------------------------


def test_dtype_table():
    assert dtypes.normalize_dtype("torch.bfloat16") == "bfloat16"
    assert dtypes.itemsize("bfloat16") == 2
    assert dtypes.itemsize("float8_e4m3fn") == 1
    assert dtypes.itemsize("json") is None
    assert dtypes.expected_nbytes("int32", [3, 4]) == 48
    with pytest.raises(dtypes.DtypeError):
        dtypes.normalize_dtype("torch.complex64")
    with pytest.raises(dtypes.DtypeError):
        dtypes.check_nbytes("bfloat16", [3], 5)


def reference_bf16(x: float) -> int:
    """Round a float32 value to BF16 by exact arithmetic, ties to even."""
    if math.isnan(x):
        return None
    bits = struct.unpack(">I", struct.pack(">f", x))[0]
    lo = bits >> 16
    if math.isinf(x):
        return lo
    hi = lo + 1
    lo_val = struct.unpack(">f", struct.pack(">I", lo << 16))[0]
    hi_val = struct.unpack(">f", struct.pack(">I", hi << 16))[0]
    if abs(x - lo_val) < abs(hi_val - x):
        return lo
    if abs(x - lo_val) > abs(hi_val - x):
        return hi
    return lo if lo % 2 == 0 else hi


def test_bf16_rne_vectors(hash_vectors):
    for case in hash_vectors["bf16_rne"]:
        f = np.array([int(case["f32_hex"], 16)], np.uint32).view(np.float32)
        assert dtypes.f32_to_bf16_bits(f)[0] == int(case["bf16_hex"], 16)


def test_bf16_rne_against_exact_reference():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2**32, size=4000, dtype=np.uint64).astype(np.uint32)
    values = bits.view(np.float32)
    ours = dtypes.f32_to_bf16_bits(values)
    for v, o in zip(values.tolist(), ours.tolist(), strict=True):
        expected = reference_bf16(v)
        if expected is None:
            assert (o & 0x7F80) == 0x7F80 and (o & 0x007F) != 0  # still a NaN
            continue
        assert o == expected, (v, hex(o), hex(expected))
    # round trip: bf16 -> f32 -> bf16 is the identity on non-NaN values
    b16 = rng.integers(0, 2**16, size=2000, dtype=np.uint32).astype(np.uint16)
    back = dtypes.f32_to_bf16_bits(dtypes.bf16_bits_to_f32(b16))
    not_nan = ~np.isnan(dtypes.bf16_bits_to_f32(b16))
    assert np.array_equal(back[not_nan], b16[not_nan])


def test_bytes_array_round_trip():
    arr = np.arange(6, dtype=np.int32).reshape(2, 3)
    raw = dtypes.array_to_bytes(arr)
    assert dtypes.bytes_to_array("int32", (2, 3), raw).tolist() == arr.tolist()
    assert dtypes.bytes_to_array("torch.bfloat16", (3,), b"\x80\x3f\x80\xbf\x00\x00").tolist() == [
        0x3F80,
        0xBF80,
        0,
    ]
    assert dtypes.to_float32("bfloat16", (2,), b"\x80\x3f\x80\xbf").tolist() == [1.0, -1.0]
