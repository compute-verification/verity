"""`hm96-sha256/v1` leaves on the host committer (`commit.hiding`, opt-in `leaf_scheme`): the batch path is core's reference byte for
byte, every opening carries its salt and is bound by it, the step roots bind the scheme, and the default leaves are untouched."""

from __future__ import annotations

import hashlib
import os
import random

import pytest

from verity.commitments import hm96
from verity.commitments.hm96 import vectors as V
from verity_vllm.acquire.committer_api import Opening
from verity_vllm.commit import hiding, scheme
from verity_vllm.commit.committer.native_host import NativeHostCommitter
from verity_vllm.commit.native_ranges import RangeOpening

CHUNK = 64


def stream(seed: int = 5, steps: int = 3, tensors: int = 3, nbytes: int = 700) -> list[list[tuple[str, bytes]]]:
    rng = random.Random(seed)
    return [[(f"layer{k}.op", rng.randbytes(nbytes + 11 * k)) for k in range(tensors)] for _ in range(steps)]


def committer(leaf_scheme: str = hiding.NAME, run_id: str = "run-H") -> NativeHostCommitter:
    return NativeHostCommitter(None, run_id=run_id, chunk=CHUNK, hash_threads=1, program_digest=hashlib.sha256(b"program").digest(),
                               leaf_scheme=leaf_scheme)


def test_batch_path_is_the_core_reference_on_its_vectors():
    h = hiding.HidingLeaves()
    cases = V.load()["leaves"]
    inner = [bytes.fromhex(c["inner_digest"]) for c in cases]
    salts = b"".join(bytes.fromhex(c["salt"]) for c in cases)
    assert [x.hex() for x in h.leaves_with(inner, salts)] == [c["leaf"] for c in cases]
    masks = h.masks(salts, len(cases))
    assert [bytes(m).hex() for m in masks] == [hm96.mask(hm96.DEFAULT_KEY, bytes.fromhex(c["salt"])).hex() for c in cases]


def test_batch_path_matches_core_over_many_leaves_and_fresh_salts():
    h = hiding.HidingLeaves(hm96.fresh_key())
    inner = [os.urandom(32) for _ in range(5000)]              # more than one numpy batch
    leaves, salts = h.leaves(inner)
    assert len(salts) == 128 * len(inner) and len(set(salts[i:i + 128] for i in range(0, len(salts), 128))) == len(inner)
    for i in (0, 1, 4095, 4096, 4999):
        assert leaves[i] == h.leaf(inner[i], salts[128 * i:128 * (i + 1)])
    with pytest.raises(ValueError):
        h.leaves_with(inner[:2], salts[:255])


def test_roundtrip_every_leaf_opens_with_its_salt():
    c = committer()
    run = c.commit_offline(stream())
    assert run.extra["leaf_scheme"] == hiding.NAME
    total = sum(s.n_leaves for s in run.steps)
    assert run.extra["salt_bytes"] == 128 * total
    for s in run.steps:
        ops = c.open([(s.step, i) for i in range(s.n_leaves)])
        assert all(len(o.salt) == 128 for o in ops)
        assert all(c.verify(run, o) for o in ops)
        leaf0 = hiding.HidingLeaves().leaf(scheme.pos_leaf(ops[0].value), ops[0].salt)
        assert scheme.fold_path(leaf0, 0, list(ops[0].path)) == s.merkle_root


def test_value_salt_and_path_are_bound():
    c = committer()
    run = c.commit_offline(stream())
    op = c.open([(1, 4)])[0]
    bad_value = bytes([op.value[0] ^ 1]) + op.value[1:]
    bad_salt = op.salt[:5] + bytes([op.salt[5] ^ 0x10]) + op.salt[6:]
    other = c.open([(1, 5)])[0]
    assert not c.verify(run, Opening(op.step, op.index, bad_value, op.path, op.identity, op.salt))
    assert not c.verify(run, Opening(op.step, op.index, op.value, op.path, op.identity, bad_salt))
    assert not c.verify(run, Opening(op.step, op.index, op.value, op.path, op.identity, other.salt))
    assert not c.verify(run, Opening(op.step, op.index, op.value, op.path, op.identity, op.salt[:-1]))
    assert not c.verify(run, Opening(op.step, op.index, op.value, op.path, op.identity))          # no salt


def test_two_commitments_of_the_same_stream_differ_and_do_not_cross_open():
    st = stream()
    c1, c2 = committer(), committer()
    run1, run2 = c1.commit_offline(st), c2.commit_offline(st)
    assert all(a.merkle_root != b.merkle_root for a, b in zip(run1.steps, run2.steps))
    assert not c1.verify(run1, c2.open([(0, 3)])[0])


def test_roots_bind_the_leaf_scheme_and_default_leaves_are_unchanged():
    st = stream()
    plain, hidden = committer(scheme.NAME), committer()
    run_plain, run_hidden = plain.commit_offline(st), hidden.commit_offline(st)
    assert plain.name == "native_host" and hidden.name == "native_host+hm96"
    assert plain._ctx_digest(0) == hashlib.sha256(b"run-H/step=0").digest() != hidden._ctx_digest(0)
    op = plain.open([(0, 2)])[0]
    assert op.salt == b"" and plain.verify(run_plain, op)
    assert not plain.verify(run_plain, Opening(op.step, op.index, op.value, op.path, op.identity, hm96.fresh_salt()))
    assert run_plain.extra["salt_bytes"] == 0
    assert run_hidden.persistent_bytes - run_plain.persistent_bytes == run_hidden.extra["salt_bytes"]


def test_range_openings_carry_their_salts():
    c = committer()
    run = c.commit_offline(stream())
    ro = c.open_range(2, 3, 9)
    assert len(ro.salts) == 6 * 128 and c.verify_range(run, ro)
    assert not c.verify_range(run, RangeOpening(ro.step, ro.lo, ro.hi, ro.value, ro.path, ro.salts[:-128] + bytes(128)))
    assert not c.verify_range(run, RangeOpening(ro.step, ro.lo, ro.hi, ro.value, ro.path))


def test_unsupported_configurations_are_refused():
    with pytest.raises(ValueError):
        NativeHostCommitter(None, leaf_scheme="blake3-keyed/row/v2")
    with pytest.raises(ValueError):
        NativeHostCommitter(None, leaf_scheme=hiding.NAME, gpu_tree=True)
    c = committer()
    with pytest.raises(ValueError):
        c.add_padding_steps([object()])
