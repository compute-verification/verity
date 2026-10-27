"""libvc's binding and conformance glue, without a GPU: the ctypes declarations against vc.h, the specs against the references'
blocks, the RFC 8439 salt identity against the RFC's vector, and vc_open_launch's record layout through `parse_openings` into
conformance's checks (verify_opening, verify_multiproof), fed records built from the reference tree. conformance.py runs the
library itself on a GPU node (gpu.sh)."""
from __future__ import annotations

import ctypes
import hashlib
import random
import re
import sys
from pathlib import Path

import pytest

LIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LIB))

import conformance as cf  # noqa: E402
import vclib  # noqa: E402
from verity.ml.commitments import RangeIndexedDomain, hm96  # noqa: E402
from verity.ml.commitments.merkle import CommitmentDomain, MerkleTree  # noqa: E402
from verity.ml.commitments.rowleaf import (ROLE_X, SCHEMA_HM96_SHA512_ROW, row_bytes, sha512_row_digest,  # noqa: E402
                                           sha512_row_prefix, sha512_row_v2_digest)

HEADER = (LIB / "vc.h").read_text()
SOURCE = (LIB / "vc.cu").read_text()


def _prototypes() -> dict[str, tuple[str, int]]:
    out = {}
    for ret, name, params in re.findall(r"^(int|uint64_t) (vc_\w+)\(([^)]*)\);", HEADER, re.M):
        params = " ".join(params.split())
        out[name] = (ret, 0 if params in ("", "void") else params.count(",") + 1)
    return out


def test_every_declared_function_is_bound_with_its_arity():
    protos = _prototypes()
    assert set(protos) == set(vclib._SIGS)
    for name, (ret, n) in protos.items():
        res, args = vclib._SIGS[name]
        assert (res is ctypes.c_int) == (ret == "int"), name
        assert len(args) == n, name


def test_the_spec_struct_is_vc_spec_field_for_field():
    body = re.search(r"typedef struct \{(.*?)\} vc_spec;", HEADER, re.S).group(1)
    names = []
    for line in body.splitlines():
        decl = line.split("//")[0].strip()
        if decl:
            names += [n.strip().lstrip("*") for n in re.sub(r"^(const )?\w+\*? ", "", decl).rstrip(";").split(",")]
    assert names == [f for f, _ in vclib._CSpec._fields_]


def test_constants_are_the_headers():
    enums = dict((k, int(v)) for k, v in re.findall(r"(VC_[A-Z0-9_]+) = (\d+)", HEADER))
    for k, v in enums.items():
        assert getattr(vclib, k[3:]) == v, k
    assert int(re.search(r"#define VC_ABI (\d+)", HEADER).group(1)) == vclib.ABI
    head = re.search(r"constexpr u64 OPEN_HEAD = ([\d +]+);", SOURCE).group(1)
    assert sum(int(t) for t in head.split("+")) == vclib.OPEN_HEAD


def test_specs_carry_the_references_blocks():
    did = bytes(range(64))
    s = vclib.served_rows(did, 2048, salt=("chacha-rfc", bytes(32), 5))
    assert s.row_prefix == sha512_row_prefix(ROLE_X, 16, 2048) and s.row_bits == 16 * 2048
    assert s.schema == SCHEMA_HM96_SHA512_ROW.encode() and s.salt_source == vclib.SALT_CHACHA_RFC and s.salt_nonce == 5
    assert (s.salt_prefix, s.key) == (hm96.SHA512.salt_prefix, hm96.SHA512.default_key)
    assert s.leaf_prefix == hm96.SHA512.leaf_prefix(hm96.SHA512.default_key)
    assert vclib.hm96_digests(did, salt=("prf", bytes(64))).row_scheme == vclib.ROW_NONE
    assert vclib.tiles_spec(did, 80, 128).row_prefix_b is not None and vclib.tiles_spec(did, 64, 64).row_prefix_b is None


@pytest.mark.parametrize("bad", [
    {"schema": b"x" * 19},
    {"salt_prefix": bytes(128)},
    {"salt_source": vclib.SALT_PRF, "salt_key": bytes(32)},
    {"salt_source": vclib.SALT_CHACHA64, "salt_key": bytes(32), "salt_nonce": 1 << 64},
    {"row_scheme": vclib.ROW_V2, "row_bits": 1 << 32, "row_prefix": bytes(128)},
    {"row_scheme": vclib.ROW_TILE, "tile_rows": 3, "tile_cols": 64, "row_prefix": bytes(128)},
])
def test_a_spec_the_library_does_not_cover_is_refused(bad):
    with pytest.raises(ValueError):
        vclib.Spec(domain_id=bytes(64), **bad)


def test_the_rfc_salt_identity_is_rfc_8439():
    # RFC 8439 2.3.2: key 00..1f, nonce 00:00:00:09:00:00:00:4a:00:00:00:00, block counter 1
    nonce = int.from_bytes(bytes.fromhex("000000090000004a00000000"), "little")
    assert cf.rfc_salts_identity(bytes(range(32)), nonce, 1)[64:128].hex() == (
        "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e")


def _record(position: int, offset: int, salt: bytes, bc: bytes, leaf: bytes, path) -> bytes:
    return position.to_bytes(8, "little") + offset.to_bytes(8, "little") + salt + bc + leaf + b"".join(path)


def _served(n: int, n_words: int = 64):
    rng = random.Random(n)
    dom = CommitmentDomain(rng.randbytes(64), -1, RangeIndexedDomain(0, n), hash="sha512")
    xs = [sha512_row_digest([rng.getrandbits(16) for _ in range(n_words)], 16, ROLE_X) for _ in range(n)]
    ys = cf.chunks(cf.rfc_salts_identity(rng.randbytes(32), 3, n), cf.SALT)
    bcs = [hm96.SHA512.commit_string(cf.KEY, x, y) for x, y in zip(xs, ys)]
    tree = MerkleTree(dom, {i: hm96.SHA512.tree_leaf(cf.KEY, b) for i, b in enumerate(bcs)}, lambda _: SCHEMA_HM96_SHA512_ROW)
    return dom, tree, ys, bcs


def _records(tree, ys, bcs, positions, n):
    depth = (n - 1).bit_length() if n > 1 else 0
    raw = b""
    for p in positions:
        if p < n:
            raw += _record(p, 64 * p, ys[p], bcs[p], tree._levels[0][p], tree.open(p).path)
        else:
            raw += _record(p, (1 << 64) - 1, bytes(192), bytes(128), bytes(64), [bytes(64)] * depth)
    return raw


@pytest.mark.parametrize("n", [1, 2, 37])
def test_reference_records_pass_conformances_open_checks(n):
    dom, tree, ys, bcs = _served(n)
    positions = cf.opened_positions(random.Random(7), n)
    recs = vclib.parse_openings(_records(tree, ys, bcs, positions, n), n, len(positions))
    assert [r.position for r in recs] == positions and recs[-1].offset is None
    bad = cf.check_openings(dom, tree.commitment, SCHEMA_HM96_SHA512_ROW, recs, n, tree._levels[0], lambda p: ys[p],
                            lambda p: bcs[p], lambda r: hm96.SHA512.tree_leaf(cf.KEY, r.bc))
    assert bad == {}


def test_a_changed_record_fails_them():
    n = 37
    dom, tree, ys, bcs = _served(n)
    positions = [3, 20]
    raw = bytearray(_records(tree, ys, bcs, positions, n))
    raw[16 + 192 + 5] ^= 1                                    # b of position 3
    size = vclib.OPEN_HEAD + 64 * 6
    raw[size + vclib.OPEN_HEAD + 64 * 2] ^= 1                 # position 20's sibling at level 2
    recs = vclib.parse_openings(bytes(raw), n, len(positions))
    bad = cf.check_openings(dom, tree.commitment, SCHEMA_HM96_SHA512_ROW, recs, n, tree._levels[0], lambda p: ys[p],
                            lambda p: bcs[p], lambda r: hm96.SHA512.tree_leaf(cf.KEY, r.bc))
    assert bad["bc"] == 3 and bad["verify_opening"] == 3 and bad["verify_multiproof"] is True


def test_unwritten_or_reordered_records_fail_them():
    n = 37
    dom, tree, ys, bcs = _served(n)
    positions = cf.opened_positions(random.Random(7), n)
    size = vclib.OPEN_HEAD + 64 * 6

    def judge(raw):
        return cf.check_openings(dom, tree.commitment, SCHEMA_HM96_SHA512_ROW, vclib.parse_openings(raw, n, len(positions)), n,
                                 tree._levels[0], lambda p: ys[p], lambda p: bcs[p], lambda r: hm96.SHA512.tree_leaf(cf.KEY, r.bc),
                                 asked=positions)
    assert judge(_records(tree, ys, bcs, positions, n)) == {}
    assert judge(bytes([cf.POISON]) * (size * len(positions)))
    assert judge(_records(tree, ys, bcs, positions[1:2] + positions[:1] + positions[2:], n))["positions"] is True


def test_an_empty_trees_records_are_all_outside_and_unwritten_ones_fail():
    dom, tree, ys, bcs = _served(0)
    positions = [0, 1, 7, (1 << 32) - 1]

    def judge(raw):
        return cf.check_openings(dom, tree.commitment, SCHEMA_HM96_SHA512_ROW, vclib.parse_openings(raw, 0, len(positions)), 0, (),
                                 None, None, lambda r: r.bc, asked=positions)
    raw = _records(tree, ys, bcs, positions, 0)
    assert len(raw) == vclib.OPEN_HEAD * len(positions) and judge(raw) == {}
    assert judge(bytes([cf.POISON]) * len(raw))
    inside = bytearray(raw)
    inside[8:16] = bytes(8)                                   # position 0 claimed at offset 0 of a tree with no leaves
    assert judge(bytes(inside))["outside"] == 0


def test_parse_openings_refuses_a_buffer_of_another_size():
    with pytest.raises(ValueError):
        vclib.parse_openings(bytes(vclib.OPEN_HEAD), 2, 1)


@pytest.mark.parametrize("case", cf.SV.CASES, ids=[c[0] for c in cf.SV.CASES])
def test_splitks_vectors_are_row_v2_over_each_chunk_leafs_bytes(case):
    """What case_splitk's device commit hashes, `sha512/row/v2` over a chunk leaf's raw K / n + 4 bytes at
    vclib.chunk_rows' row bits, is each pinned leaf's words_sha512 input and row digest (credit's word-level
    `registered.row_digest`), so the GPU's roots are SPLITK's at the vectors' leaves and nothing else."""
    name, _value, k, count, _nonce, _windowed = case
    v = cf.splitk_vectors()[name]
    leaves = [leaf for r in cf.SV.rows(name, k, count) for leaf in cf.SV.Z.chunk_leaves(r, k)]
    spec, _stride = vclib.chunk_rows(bytes(64), k, v["n"])
    assert len(leaves) == v["leaves"] == len(v["leaf"]) and {8 * len(x) for x in leaves} == {spec.row_bits}
    assert [hashlib.sha512(x).hexdigest() for x in leaves] == [e["words_sha512"] for e in v["leaf"]]
    assert [sha512_row_v2_digest(x, spec.row_bits).hex() for x in leaves] == [e["row_digest"] for e in v["leaf"]]


def test_chunk_rows_spec_is_row_v2_of_a_chunk_leaf():
    spec, stride = vclib.chunk_rows(bytes(64), 14336, 4, ("chacha64", bytes(32), cf.SV.A_NONCE))
    assert (spec.row_bits, stride, spec.schema) == (8 * 3588, 3600, b"hm96-sha512/row/v2")
    assert spec.salt_source == vclib.SALT_CHACHA64 and spec.salt_nonce == cf.SV.A_NONCE
    assert vclib.chunk_rows(bytes(64), 4096, 1)[1] == 4112


def test_row_bytes_are_what_the_served_kernel_hashes():
    words = list(range(1, 65))
    raw = row_bytes(words, 16)
    import hashlib
    assert hashlib.sha512(sha512_row_prefix(ROLE_X, 16, 64) + raw).digest() == sha512_row_digest(words, 16, ROLE_X)
