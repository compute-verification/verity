"""``hm96-sha256/v1``: the reference reproduces its vectors, the mask is the matrix product the spec defines, the family is
universal, every opening component is bound, and the hiding bound has the stated value."""

from __future__ import annotations

import hashlib
import os

import pytest

from verity.commitments import hm96, vllm_v1
from verity.commitments.hm96 import vectors as V
from verity.commitments.rowleaf import blake3_row_layout
from verity.commitments.scheme import CommitmentScheme, LeafLayout
from verity.errors import InvalidArtifact

VECTORS = V.load()
KEYS = {"default": hm96.DEFAULT_KEY, "other": bytes.fromhex(VECTORS["other_key"])}


def _b(text: str) -> bytes:
    return bytes.fromhex(text)


def _bit(data: bytes, t: int) -> int:
    return (data[t >> 3] >> (t & 7)) & 1


def naive_mask(key: bytes, salt: bytes) -> bytes:
    """``M[i][j] = key bit i + j``, ``(M y)_i = XOR_j M[i][j] y_j``, straight from the spec's text."""

    out = bytearray(32)
    for i in range(256):
        bit = 0
        for j in range(1024):
            bit ^= _bit(key, i + j) & _bit(salt, j)
        out[i >> 3] |= bit << (i & 7)
    return bytes(out)


def test_vectors_file_is_what_the_reference_generates() -> None:
    assert V.main(["--check"]) == 0, "rerun `python -m verity.commitments.hm96.vectors`"


def test_default_key_is_rederived_from_its_label() -> None:
    assert hm96.key_from_label(hm96.KEY_LABEL) == hm96.DEFAULT_KEY == _b(VECTORS["default_key"])
    assert hm96.KEY_BYTES == 160 and hm96.KEY_BITS == 1279 and hm96.SALT_BYTES == 128
    assert hm96.xor_count(hm96.DEFAULT_KEY) == VECTORS["default_key_xor_gates"]
    assert hm96.scheme_digest(hm96.DEFAULT_KEY).hex() == VECTORS["scheme_digest_default_key"]


@pytest.mark.parametrize("case", VECTORS["masks"], ids=lambda c: f"{c['key']}-{c['salt_case']}")
def test_mask_vectors(case: dict) -> None:
    key, salt = KEYS[case["key"]], _b(case["salt"])
    assert hm96.mask(key, salt).hex() == case["mask"]
    assert hm96.salt_digest(salt).hex() == case["salt_digest"]
    assert hashlib.sha256(_b(VECTORS["salt_prefix"]) + salt).hexdigest() == case["salt_digest"]


@pytest.mark.parametrize("case", [c for c in VECTORS["masks"] if c["salt_case"] in ("e0", "e1023", "expand")],
                         ids=lambda c: f"{c['key']}-{c['salt_case']}")
def test_mask_is_the_matrix_product(case: dict) -> None:
    assert naive_mask(KEYS[case["key"]], _b(case["salt"])).hex() == case["mask"]


def test_unit_salts_select_columns() -> None:
    cols = hm96.columns(hm96.DEFAULT_KEY)
    for j in (0, 1, 7, 8, 500, 1023):
        salt = bytearray(128)
        salt[j >> 3] = 1 << (j & 7)
        assert int.from_bytes(hm96.mask(hm96.DEFAULT_KEY, bytes(salt)), "little") == cols[j]


def test_family_is_universal() -> None:
    """``key -> M_key d`` is linear with rows ``d << i``; rank 256 for every nonzero d, so ``Pr_key[M d = 0] = 2^-256``."""

    def rank(vectors: list[int]) -> int:
        basis: dict[int, int] = {}
        for v in vectors:
            while v:
                top = v.bit_length() - 1
                if top not in basis:
                    basis[top] = v
                    break
                v ^= basis[top]
        return len(basis)

    for d in (1, 1 << 1023, int.from_bytes(os.urandom(128), "little") | 1):
        assert rank([d << i for i in range(256)]) == 256


@pytest.mark.parametrize("case", VECTORS["leaves"], ids=lambda c: f"{c['inner']}-{len(c['value']) // 2}")
def test_leaf_vectors(case: dict) -> None:
    value, salt = _b(case["value"]), _b(case["salt"])
    inner = LeafLayout("sha256", _b(case["inner_prefix"]), len(value), _b(case["inner_suffix"]))
    layout = hm96.HidingLayout(inner, hm96.DEFAULT_KEY)
    assert inner.digest(value).hex() == case["inner_digest"]
    proven = layout.commit(value, salt)
    assert proven.hex() == case["b"] + case["c"]
    assert layout.tree_leaf(proven).hex() == case["leaf"]
    assert hashlib.sha256(_b(VECTORS["leaf_prefix_default_key"]) + proven).hexdigest() == case["leaf"]
    b = bytes(p ^ q for p, q in zip(_b(case["inner_digest"]), naive_mask(hm96.DEFAULT_KEY, salt)))
    assert b.hex() == case["b"]


@pytest.mark.parametrize("case", VECTORS["trees"], ids=lambda c: f"n{len(c['values'])}")
def test_tree_vectors(case: dict) -> None:
    d = case["domain"]
    base = vllm_v1.VllmV1({"v": vllm_v1.StepDomain(_b(d["program"]), _b(d["ctx"]), _b(d["geo"]), _b(d["layout"]), d["leaf_count"])},
                          value_nbytes={"v": lambda _i: 40})
    scheme = hm96.Hm96Sha256(base)
    assert isinstance(scheme, CommitmentScheme)
    n = d["leaf_count"]
    leaves = [scheme.tree_leaf("v", i, scheme.leaf_layout("v", i).commit(_b(v), _b(s)))
              for i, (v, s) in enumerate(zip(case["values"], case["salts"]))]
    assert [x.hex() for x in leaves] == case["leaves"]
    root = scheme.root("v", leaves)
    assert root.hex() == case["root"]
    for i in range(n):
        path = tuple(None if p is None else _b(p) for p in case["paths"][i])
        assert scheme.verify_path("v", root, n, i, leaves[i], path)
        assert not scheme.verify_path("v", root, n, i, hashlib.sha256(leaves[i]).digest(), path)


def test_every_component_of_an_opening_is_bound() -> None:
    inner = vllm_v1.pos_leaf_layout(48)
    layout = hm96.HidingLayout(inner, hm96.DEFAULT_KEY)
    value, salt = os.urandom(48), hm96.fresh_salt()
    leaf = layout.leaf(value, salt)
    flipped_value = bytes([value[0] ^ 1]) + value[1:]
    flipped_salt = salt[:-1] + bytes([salt[-1] ^ 0x80])
    assert layout.leaf(flipped_value, salt) != leaf
    assert layout.leaf(value, flipped_salt) != leaf
    proven = layout.commit(value, salt)
    assert hm96.tree_leaf(hm96.DEFAULT_KEY, proven) == leaf
    assert hm96.tree_leaf(KEYS["other"], proven) != leaf                       # the leaf binds its key
    for k in (0, 31, 32, 63):
        tweak = bytearray(proven)
        tweak[k] ^= 1
        assert layout.tree_leaf(bytes(tweak)) != leaf


def test_malformed_inputs_are_refused() -> None:
    with pytest.raises(InvalidArtifact):
        hm96.check_key(bytes(159))
    with pytest.raises(InvalidArtifact):
        hm96.check_key(bytes(159) + b"\x80")                                  # bit 1279 set: a second encoding of a matrix
    with pytest.raises(InvalidArtifact):
        hm96.mask(hm96.DEFAULT_KEY, bytes(127))
    with pytest.raises(InvalidArtifact):
        hm96.commit_string(hm96.DEFAULT_KEY, bytes(31), bytes(128))
    with pytest.raises(InvalidArtifact):
        hm96.tree_leaf(hm96.DEFAULT_KEY, bytes(63))
    with pytest.raises(InvalidArtifact):
        hm96.HidingLayout(blake3_row_layout(1, 16, 4), hm96.DEFAULT_KEY)      # SHA-256 inner digests only
    with pytest.raises(TypeError):
        hm96.Hm96Sha256(object())                                              # type: ignore[arg-type]


def test_fresh_randomness() -> None:
    a, b = hm96.fresh_salt(), hm96.fresh_salt()
    assert len(a) == 128 and a != b
    k = hm96.fresh_key()
    assert hm96.check_key(k) == k and k != hm96.fresh_key()


def test_hiding_bound() -> None:
    assert hm96.statistical_distance_log2(1) == -256
    assert hm96.statistical_distance_log2(1 << 64) == -192
    assert hm96.statistical_distance_log2(1 << 64, key_failure_log2=64) == -128
    assert hm96.statistical_distance_log2(1 << 40, key_failure_log2=64) < -128
