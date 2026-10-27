"""``rows-sha512`` (``verity_vllm.commit.step_rows``) is core's construction: its hiding leaves are frame-v3-sha512's conformance vectors,
its mask table gives hm96-sha512's masks, its tree and paths are core's ``tree_levels`` / ``MerkleTree`` / ``verify_opening``, its range
openings are core's multiproofs, and its roots are vllm-v1's under SHA-512."""
from __future__ import annotations

import hashlib
import json
import os
import random

import pytest

from verity.ml.commitments import hm96, merkle
from verity.ml.commitments import vllm_v1 as V
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.limits import VerificationLimits
from verity.ml.commitments.merkle import Commitment, CommitmentDomain, MerkleTree, Opening, verify_opening
from verity.ml.commitments.multiproof import multiproof, multiproof_root
from verity.ml.commitments.rowleaf import sha512_row_v2_digest
from verity_vllm.commit import step_rows as R

_CORE = os.path.dirname(merkle.__file__)
FRAME = json.load(open(os.path.join(_CORE, "frame_v3", "vectors_sha512.json")))
HM96 = json.load(open(os.path.join(_CORE, "hm96", "vectors_sha512.json")))
OTHER_KEY = bytes.fromhex(HM96["other_key"])
PD, CTX, GEO, LAYOUT = (hashlib.sha256(t).digest() for t in (b"program", b"ctx", b"geo", b"layout"))


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
def test_range_openings_are_cores_multiproofs(n):
    """A run's boundary siblings, in order, are core's multiproof of its leaves, which `multiproof_root` folds to the root, and to no
    root or another with a tampered leaf, a wrong sibling, a node missing or extra, or a bound of the run moved (off the tree, too)."""
    rng = random.Random(100 + n)
    rows = [rng.randbytes(R.ROW_BYTES) for _ in range(n)]
    ks = rng.randbytes(R.KEY_BYTES)
    dom = R.domain(PD, CTX, GEO, n)
    lv = R.levels(dom, rows, ks, 8 * R.ROW_BYTES)
    hs = [R.hiding_leaf(r, R.salt(ks, i), 8 * R.ROW_BYTES)[1] for i, r in enumerate(rows)]
    tree = MerkleTree(dom, dict(enumerate(hs)), lambda _p: R.SCHEMA)
    root = lv[-1][0]

    def ok(lo, run, proof):
        return multiproof_root(n, dom.domain_id, {lo + k: R.leaf(dom, lo + k, h) for k, h in enumerate(run)}, proof, "sha512") == root

    for lo in range(n):
        for hi in range(lo + 1, n + 1):
            run, proof = hs[lo:hi], [s for pair in _range_path(lv, lo, hi) for s in pair if s is not None]
            assert proof == multiproof(tree, range(lo, hi)) and ok(lo, run, proof)
            refused = [(lo, [bytes([run[0][0] ^ 1]) + run[0][1:]] + run[1:], proof), (lo, run, proof + [root]),
                       (lo + 1, run, proof), (lo, run[:-1], proof)]
            if proof:
                refused += [(lo, run, proof[:-1]), (lo, run, [bytes(R.ROOT_BYTES)] + proof[1:])]
            assert not any(ok(*r) for r in refused)


def test_roots_are_vllm_v1s_under_sha512():
    tree_root = hashlib.sha512(b"tree").digest()
    sr = R.step_root(PD, CTX, GEO, LAYOUT, 7, tree_root)
    assert sr == V.step_root(PD, CTX, GEO, LAYOUT, 7, tree_root, hash="sha512") and len(sr) == R.ROOT_BYTES
    assert R.run_root(PD, GEO, [sr, sr]) == V.run_root(PD, GEO, [sr, sr], hash="sha512")
    assert len(R.domain(PD, CTX, GEO, 0).empty_root()) == R.ROOT_BYTES
