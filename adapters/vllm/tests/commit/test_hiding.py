"""`hm96-sha256/v1` leaves on the host committer (`commit.hiding`, opt-in `leaf_scheme`): the batch path is core's reference byte for
byte, every opening carries its salt and is bound by it, the step roots bind the scheme, and the default leaves are untouched."""

from __future__ import annotations

import hashlib
import os
import random

import pytest

from verity.ml.commitments import hm96
from verity.ml.commitments.hm96 import vectors as V
from verity_vllm.acquire.committer_api import Opening
from verity.ml.windows import hiding
from verity_vllm.commit import scheme
from verity_vllm.commit.committer.native_host import NativeHostCommitter
from verity_vllm.commit.native_ranges import RangeOpening

CHUNK = 64
#: what the committers here read through `verity_vllm.settings`: unset, each is built with the default leaves, name and retention,
#: whatever the process's environment holds
ENV = ("VERITY_ABL", "VERITY_ARENA_GROW", "VERITY_ARENA_WAIT_S", "VERITY_FOOTPRINT", "VERITY_INLINE_D2H", "VERITY_MEMTRACE", "VERITY_PIN",
       "VERITY_RETAIN", "VERITY_RETAIN_PINNED", "VERITY_SKIP")


@pytest.fixture(autouse=True)
def _default_settings(monkeypatch):
    for k in ENV:
        monkeypatch.delenv(k, raising=False)


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


def test_batch_path_is_the_core_reference_for_hm96_sha512():
    h = hiding.HidingLeaves(instance=hm96.SHA512)
    cases = V.load(hm96.SHA512)["leaves"]
    inner = [bytes.fromhex(c["inner_digest"]) for c in cases]
    salts = b"".join(bytes.fromhex(c["salt"]) for c in cases)
    assert [x.hex() for x in h.leaves_with(inner, salts)] == [c["leaf"] for c in cases]
    leaves, fresh = h.leaves(inner)
    assert len(fresh) == 192 * len(inner) and leaves[1] == h.leaf(inner[1], fresh[192:384])


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


def test_the_leaf_rule_takes_the_instances_salt_size():
    c = committer()
    c.hiding = hiding.HidingLeaves(instance=hm96.SHA512)          # what wiring hm96-sha512 at the re-baseline will configure
    inner, salt = os.urandom(64), hm96.SHA512.fresh_salt()
    assert c._salt_bytes() == 192
    assert c._leaf_rule([inner], salt) == [c.hiding.leaf(inner, salt)]
    assert c._leaf_rule([inner], salt[:128]) is None
    c._salts[3] = salt * 2
    assert not c._unsalted(3, 2) and c._unsalted(3, 3) and c._salts_of(3, 1, 2) == salt


def test_the_context_suffix_keeps_the_context_encoding_injective():
    digest = hiding.HidingLeaves().digest
    assert len(digest) == 32 and b"/step=" not in digest and b"/leaf=" not in digest
    c = committer()
    c._salts[7] = b""
    assert c._ctx_digest(7) == hashlib.sha256(b"run-H/step=7/leaf=" + digest).digest()


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


def test_a_committer_whose_steps_take_another_leaf_path_is_refused():
    class Elsewhere(NativeHostCommitter):
        def _commit_step(self, job):
            return None

    with pytest.raises(ValueError):
        Elsewhere(None, leaf_scheme=hiding.NAME)
    assert Elsewhere(None).leaf_scheme == scheme.NAME


def test_the_native_collector_refuses_hm96_on_every_leaf_path():
    pytest.importorskip("torch")
    from verity_vllm.commit.committer.native_collect import NativeCollectCommitter

    for kw in ({"native_worker": True, "gpu_tree": False}, {"native_worker": False, "gpu_tree": False}, {}):
        with pytest.raises(ValueError):
            NativeCollectCommitter(None, leaf_scheme=hiding.NAME, **kw)


def test_no_root_binds_hm96_over_unsalted_leaves(monkeypatch):
    c = committer()
    with pytest.raises(RuntimeError):
        c._ctx_digest(0)                                        # nothing salted step 0: no hm96 context for it
    monkeypatch.setattr(c, "_salted", lambda step, leaves: leaves)
    with pytest.raises(RuntimeError):
        c.commit_offline(stream())                              # a step path that skips the salts cannot bind its root
    assert not c._done


def test_commit_block_offline_refuses_hm96(monkeypatch):
    pytest.importorskip("torch")
    from verity_vllm.commit.committer.native_host import GPU_CHUNK_BYTES, TensorMeta

    monkeypatch.setenv("VERITY_RETAIN", "host")
    c = committer()
    c.reset()
    m = TensorMeta(0, "layer0.op", "uint8", (700,), 700, GPU_CHUNK_BYTES, 0, 3)
    with pytest.raises(RuntimeError):
        c.commit_block_offline(0, os.urandom(700), [m])
    assert 0 not in c._done and 0 not in c._gpu_blocks and 0 not in c._gpu_levels


def test_a_step_without_one_salt_per_leaf_neither_opens_nor_finalizes():
    c = committer()
    run = c.commit_offline(stream())
    op, ro = c.open([(1, 3)])[0], c.open_range(1, 2, 6)
    salts = c._salts.pop(1)
    assert not c.verify(run, op) and not c.verify_range(run, ro)
    with pytest.raises(RuntimeError):
        c._leaf_record(run.steps)
    c._salts[1] = salts[:-hiding.SALT_BYTES]
    assert not c.verify(run, op) and not c.verify_range(run, ro)
    with pytest.raises(RuntimeError):
        c._leaf_record(run.steps)
    c._salts[1] = salts
    assert c.verify(run, op) and c.verify_range(run, ro)
