"""``rows-sha512`` (``verity_vllm.commit.step_rows``) is core's construction: its hiding leaves are frame-v3-sha512's conformance vectors,
its mask table gives hm96-sha512's masks, its tree and paths are core's ``tree_levels`` / ``MerkleTree`` / ``verify_opening``, its range
openings are core's multiproofs, and its roots are pinned vectors."""
from __future__ import annotations

import hashlib
import json
import os
import random

import pytest

from verity.errors import InvalidArtifact
from verity.ml.commitments import hm96, merkle
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.limits import VerificationLimits
from verity.ml.commitments.merkle import Commitment, CommitmentDomain, MerkleTree, Opening, verify_opening
from verity.ml.commitments.multiproof import multiproof, verify_multiproof
from verity.ml.commitments.rowleaf import sha512_row_v2_digest
from verity_vllm.commit import step_rows as R

_CORE = os.path.dirname(merkle.__file__)
FRAME = json.load(open(os.path.join(_CORE, "frame_v3", "vectors_sha512.json")))
HM96 = json.load(open(os.path.join(_CORE, "hm96", "vectors_sha512.json")))
OTHER_KEY = bytes.fromhex(HM96["other_key"])
PD, CTX, GEO, LAYOUT = (hashlib.sha512(t).digest() for t in (b"program", b"ctx", b"geo", b"layout"))


def _vector_domain(d: dict) -> CommitmentDomain:
    dom = CommitmentDomain(bytes.fromhex(d["binding"]), d["owner"], RangeIndexedDomain(*d["range"]), hash=d["hash"])
    assert dom.domain_id.hex() == d["domain_id"]
    return dom


@pytest.mark.parametrize("vec", FRAME["bit_row_hiding_leaves"], ids=lambda v: v["name"])
def test_hiding_leaves_are_frame_v3_sha512s_vectors(vec):
    assert vec["schema"] == R.SCHEMA and hashlib.sha512(R.KEY).hexdigest() == vec["key_sha512"]
    row, n_bits, y = bytes.fromhex(vec["row_bytes"]), vec["n_bits"], bytes.fromhex(vec["salt"])
    bc, h = R.hiding_leaf(row, y, n_bits)
    assert sha512_row_v2_digest(row, n_bits).hex() == vec["digest"]
    assert (bc.hex(), h.hex()) == (vec["bc"], vec["hm96_leaf"])
    assert R.leaf(_vector_domain(vec["domain"]), vec["rank"], h).hex() == vec["leaf"]


def _row_of(tn: bytes, v: int) -> bytes:
    """T_v from the nibble table, as the kernel reads it: chunk c is T_(v & 15)'s ^ T_(v & 0xF0)'s."""
    return b"".join(bytes(a ^ b for a, b in zip(tn[256 * c + 16 * (v & 15):][:16], tn[4096 + 256 * c + 16 * (v >> 4):][:16], strict=True))
                    for c in range(16))


@pytest.mark.parametrize("case", HM96["masks"], ids=lambda c: f"{c['key']}-{c['salt_case']}")
def test_the_nibble_table_gives_hm96_sha512s_masks(case):
    key = R.KEY if case["key"] == "default" else OTHER_KEY
    tn, y = R.t_nibbles(key), bytes.fromhex(case["salt"])
    acc = 0
    for p, v in enumerate(y):
        acc ^= int.from_bytes(_row_of(tn, v)[p:p + 64], "little")
    assert acc.to_bytes(64, "little").hex() == case["mask"] == R.INST.mask(key, y).hex()
    assert R.INST.salt_digest(y).hex() == case["salt_digest"]


def test_the_salt_is_sha512_of_the_key_position_and_counter():
    ks = bytes(range(64))
    want = b"".join(hashlib.sha512(ks + (300).to_bytes(8, "big") + r.to_bytes(8, "big")).digest() for r in range(3))
    assert R.salt(ks, 300) == want and len(want) == hm96.SHA512.salt_bytes
    with pytest.raises(ValueError):
        R.salt(ks[:32], 0)


def test_the_domain_binds_the_steps_identity_digests_and_its_length():
    doms = {R.domain(PD, CTX, GEO, 5).domain_id, R.domain(PD, GEO, CTX, 5).domain_id, R.domain(PD, CTX, GEO, 6).domain_id,
            R.domain(PD, CTX, GEO, 5, row_bits=1024).domain_id}
    assert len(doms) == 4
    dom = R.domain(PD, CTX, GEO, 5)
    assert dom.hash == "sha512" and dom.owner == R.OWNER and dom.count == 5 and len(dom.binding) == 64


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_tree_paths_and_roots_are_cores(n):
    rng = random.Random(n)
    rows = [rng.randbytes(R.ROW_BYTES) for _ in range(n)]
    ks = rng.randbytes(R.KEY_BYTES)
    dom = R.domain(PD, CTX, GEO, n)
    lv = R.levels(dom, rows, ks, 8 * R.ROW_BYTES)
    hs = {i: R.hiding_leaf(r, R.salt(ks, i), 8 * R.ROW_BYTES)[1] for i, r in enumerate(rows)}
    tree = MerkleTree(dom, hs, lambda _p: R.SCHEMA)
    root = lv[-1][0]
    assert tree.commitment.root == root and len(lv[0]) == 1 << merkle.merkle_depth(n)
    com, limits = Commitment(root, n), VerificationLimits()
    for i in range(n):
        p = R.path(lv, i)
        assert tree.open(i) == Opening(i, hs[i], p)
        assert verify_opening(dom, com, Opening(i, hs[i], p), R.SCHEMA, limits)
        refused = [Opening(i, bytes([hs[i][0] ^ 1]) + hs[i][1:], p), Opening(i, hs[i], p + (root,))]
        if n > 1:
            refused += [Opening(i ^ 1, hs[i], p), Opening(i, hs[i], p[1:]), Opening(i, hs[i], (root,) + p[1:])]
        assert not any(verify_opening(dom, com, o, R.SCHEMA, limits) for o in refused)
    assert not verify_opening(dom, com, Opening(n, hs[0], R.path(lv, 0)), R.SCHEMA, limits)


def _range_path(lv, lo: int, hi: int) -> tuple:
    out = []
    for d, (need_l, need_r) in enumerate(R.range_shape(len(lv[0]), lo, hi)):
        out.append((lv[d][lo - 1] if need_l else None, lv[d][hi] if need_r else None))
        lo, hi = lo // 2, (hi + 1) // 2
    return tuple(out)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 13])
def test_range_openings_fold_as_cores_multiproof(n):
    """A run's boundary siblings, in order, are core's multiproof of its leaves; `fold_range` folds them to the root and refuses a
    tampered leaf, a sibling missing or extra, a sibling of the wrong size and a run off the tree."""
    rng = random.Random(100 + n)
    rows = [rng.randbytes(R.ROW_BYTES) for _ in range(n)]
    ks = rng.randbytes(R.KEY_BYTES)
    dom = R.domain(PD, CTX, GEO, n)
    lv = R.levels(dom, rows, ks, 8 * R.ROW_BYTES)
    hs = [R.hiding_leaf(r, R.salt(ks, i), 8 * R.ROW_BYTES)[1] for i, r in enumerate(rows)]
    tree = MerkleTree(dom, dict(enumerate(hs)), lambda _p: R.SCHEMA)
    root = lv[-1][0]
    for lo in range(n):
        for hi in range(lo + 1, n + 1):
            sibs = _range_path(lv, lo, hi)
            proof = [s for pair in sibs for s in pair if s is not None]
            assert proof == multiproof(tree, range(lo, hi))
            assert verify_multiproof(root, n, dom.domain_id, {i: R.leaf(dom, i, hs[i]) for i in range(lo, hi)}, proof, "sha512")
            assert R.fold_range(dom, lo, hs[lo:hi], sibs) == root
            bad = [bytes([hs[lo][0] ^ 1]) + hs[lo][1:]] + hs[lo + 1:hi]
            assert R.fold_range(dom, lo, bad, sibs) != root
    sibs = _range_path(lv, 0, 1)
    assert R.fold_range(dom, 0, [], sibs) is None
    assert R.fold_range(dom, n, hs[:1], sibs) is None
    if n > 1:
        assert R.fold_range(dom, 0, hs[:1], sibs[:-1]) is None
        assert R.fold_range(dom, 0, hs[:1], ((None, None),) + sibs[1:]) is None
        assert R.fold_range(dom, 0, hs[:1], ((root, sibs[0][1]),) + sibs[1:]) is None
        assert R.fold_range(dom, 0, hs[:1], ((None, sibs[0][1][:32]),) + sibs[1:]) is None


NAMES = hashlib.sha512(b"names").digest()
STEP_ROOTS = [R.step_root(PD, CTX, GEO, LAYOUT, i + 1, hashlib.sha512(b"tree/%d" % i).digest()) for i in range(5)]
TENSOR_ROOTS = [hashlib.sha512(b"tensor/%d" % i).digest() for i in range(2)]
STEP_ROOT_0 = "a95e476deabf46609e6b0520ec53330191ad00a4c65f5a38c9f77af9b26a9d2c2cfa2aba1de5ed9d3a9382450e2ae07da562f8e9817a58c38713e77cfed14839"
RUN_ROOTS = {
    0: "e53a9c85ced3c6d3b5a704a4aaac301464a0eef1ab09837cb732f0228f4b2b6479505156048fbff6dcfcf884d555b52b108cce9fe137666c4b09df0214b344d4",
    1: "c392380d52c93c3017c5a588a3344f5ae287d87eaac5f177f43f104943772ad692a4453b5c8b4898b2df68b61eab6f58c3779fa3ffe4fa0b98173057f258c6fd",
    3: "371694ddd620cce2b8aca3777fabeec15f1ad55f8a40076f3e0a90734e0dcff15a24e966d3ccab89e6d3f43dfb9511c6a3c7c2d36014ef9f0e1518b2b4c361de",
    5: "635fbabe709dfa392d589a36b97d8ecde91c3a9cf7a29dd15d1dd5bd5e43800859d072940310009524b808c73e663385e8c89d35012b9d25e2738b0aff4c34a6",
}
WEIGHTS_ROOTS = {
    0: "921f60e1402de65b90e6367116c3c3894d8dd8819f9fbd32e85267e9b0da579aee6bde90646b9b6be6d03eb76d1c85e7e84a804b0e94c8b1103cf915da0a0c50",
    2: "da8c53c3e6b4de394e0d2958461b7d03624bf85cb377550b496c13e267f8326749fe3c02cede17e44063aa6cff3f55310a3acf8748273c993894cc351c9c4f9b",
}


def test_roots_are_the_pinned_vectors():
    """The step, run and weights roots over 64-byte identity digests: none, one, an odd and an even count of folded roots."""
    assert STEP_ROOTS[0].hex() == STEP_ROOT_0 and all(len(r) == R.ROOT_BYTES for r in STEP_ROOTS)
    assert {k: R.run_root(PD, GEO, STEP_ROOTS[:k]).hex() for k in RUN_ROOTS} == RUN_ROOTS
    assert {k: R.weights_root(GEO, NAMES, TENSOR_ROOTS[:k]).hex() for k in WEIGHTS_ROOTS} == WEIGHTS_ROOTS
    assert len(R.domain(PD, CTX, GEO, 0).empty_root()) == R.ROOT_BYTES


class _Meta:
    ordinal, name, dtype, shape, nbytes, chunk, first_leaf, n_leaves, stream_off = 0, "model.x", "bfloat16", (2, 3), 12, R.ROW_BYTES, 0, 1, 0


def test_the_identity_digests_are_whole_sha512_digests():
    """Every identity digest a root binds is SHA-512 of its text, all 64 bytes, and a Program digest binds whole: a 32-byte one (a
    SHA-256 fold, or a digest cut to 256 bits) is refused."""
    names = [["model.x", "bfloat16", [2, 3], 12]]
    got = {"program": R.run_program_digest("run-a"), "geo": R.offline_geo_digest("run-a"), "ctx": R.ctx_digest("run-a", 7),
           "model-geo": R.geo_digest({"repo": "r", "chunk": R.ROW_BYTES}), "layout": R.layout_digest([_Meta()]), "names": R.names_digest(names)}
    want = {"program": b"cmt-integ/no-program-digest/run-a", "geo": b"cmt-integ/offline-geo/run-a", "ctx": b"run-a/step=7",
            "model-geo": b'{"chunk": 4096, "repo": "r"}', "layout": b'[[0,"model.x","bfloat16",[2,3],12,4096,0,1,0]]',
            "names": b'[["model.x","bfloat16",[2,3],12]]'}
    assert got == {k: hashlib.sha512(v).digest() for k, v in want.items()} and R.ID_BYTES == 64
    assert R.program_field(PD.hex()) == PD
    for short in (PD[:32], hashlib.sha256(b"program").digest()):
        with pytest.raises(InvalidArtifact, match="64-byte"):
            R.program_field(short.hex())


@pytest.mark.parametrize("call", [
    lambda: R.step_root(PD[:31], CTX, GEO, LAYOUT, 1, STEP_ROOTS[0]),
    lambda: R.step_root(PD[:32], CTX, GEO, LAYOUT, 1, STEP_ROOTS[0]),
    lambda: R.step_root(PD, CTX[:32], GEO, LAYOUT, 1, STEP_ROOTS[0]),
    lambda: R.step_root(PD, CTX, GEO[:32], LAYOUT, 1, STEP_ROOTS[0]),
    lambda: R.step_root(PD, CTX, GEO, LAYOUT[:32], 1, STEP_ROOTS[0]),
    lambda: R.step_root(PD, CTX, GEO, LAYOUT, 1, STEP_ROOTS[0][:32]),
    lambda: R.step_root(PD, CTX, GEO, LAYOUT, -1, STEP_ROOTS[0]),
    lambda: R.step_root(PD, CTX, GEO, LAYOUT, 1 << 64, STEP_ROOTS[0]),
    lambda: R.step_root(PD, CTX, GEO, LAYOUT, True, STEP_ROOTS[0]),
    lambda: R.run_root(PD, GEO, STEP_ROOTS[:2] + [STEP_ROOTS[2][:63]]),
    lambda: R.run_root(PD, GEO, [bytearray(STEP_ROOTS[0])]),
    lambda: R.run_root(PD[:32], GEO, STEP_ROOTS[:2]),
    lambda: R.run_root(PD, GEO[:32], STEP_ROOTS[:2]),
    lambda: R.weights_root(GEO, NAMES[:16], TENSOR_ROOTS),
    lambda: R.weights_root(GEO, NAMES[:32], TENSOR_ROOTS),
    lambda: R.weights_root(GEO[:32], NAMES, TENSOR_ROOTS),
], ids=["program", "program-256", "ctx-256", "geo-256", "layout-256", "tree-root", "negative-n", "n-past-u64", "bool-n", "step-root",
        "bytearray-root", "run-program-256", "run-geo-256", "names", "names-256", "weights-geo-256"])
def test_roots_refuse_a_wrong_size_or_count(call):
    with pytest.raises(InvalidArtifact):
        call()
