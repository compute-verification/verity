"""`hm96-sha256/v1` leaves on the committers (`commit.hiding`, the default `leaf_scheme`): the batch path is core's reference byte for
byte, the GPU chunk tree's salts are RFC 8439 ChaCha20 under a per-run key, every opening carries its salt and is bound by it, a value
guessed without its salt is not confirmed, the step roots bind the scheme, and vllm-v1 leaves stay selectable and unchanged."""

from __future__ import annotations

import hashlib
import inspect
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


def test_hm96_is_the_default_on_both_step_paths():
    host, gpu = NativeHostCommitter(None), NativeHostCommitter(None, gpu_tree=True)
    assert host.leaf_scheme == gpu.leaf_scheme == hiding.NAME and host.commitment_scheme == hiding.SCHEME == "vllm-v1/hm96-sha256/v1"
    assert (host.salt_rule, host.stream) == (hiding.SALT_OS, None)
    assert gpu.salt_rule == hiding.SALT_STREAM and gpu.hiding is gpu.stream.leaves
    assert NativeHostCommitter(None, leaf_scheme=scheme.NAME).commitment_scheme == scheme.NAME
    assert hiding.leaf_scheme_of({}) == hiding.leaf_scheme_of(None) == scheme.NAME


def test_unsupported_configurations_are_refused():
    with pytest.raises(ValueError):
        NativeHostCommitter(None, leaf_scheme="blake3-keyed/row/v2")
    c = committer()
    with pytest.raises(ValueError):
        c.add_padding_steps([object()])


def test_a_committer_whose_steps_take_another_leaf_path_is_refused():
    class Elsewhere(NativeHostCommitter):
        def _commit_step(self, job):
            return None

    for kw in ({}, {"leaf_scheme": hiding.NAME}, {"gpu_tree": True}):
        with pytest.raises(ValueError):
            Elsewhere(None, **kw)                               # the default fails closed: such a path opts out by name
    assert Elsewhere(None, leaf_scheme=scheme.NAME).leaf_scheme == scheme.NAME


def test_the_native_collector_refuses_hm96_off_the_gpu_chunk_tree():
    pytest.importorskip("torch")
    from verity_vllm.commit.committer.native_collect import NativeCollectCommitter

    assert NativeCollectCommitter.HM96_DEVICE_PATHS and not NativeHostCommitter.HM96_DEVICE_PATHS
    assert "salts_of" in inspect.signature(NativeCollectCommitter.reset).parameters
    for kw in ({"native_worker": True, "gpu_tree": False}, {"native_worker": False, "gpu_tree": False}):
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


def test_commit_block_offline_refuses_hm96_with_host_salts(monkeypatch):
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


# --- the GPU chunk tree: salts from a ChaCha20 stream, through its CPU reference (`commit_block_offline`) ---------------------------

RFC8439_KEY, RFC8439_NONCE = bytes(range(32)), bytes.fromhex("000000090000004a00000000")
RFC8439_BLOCK_1 = ("10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
                   "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e")        # RFC 8439 2.3.2, block counter 1
GPU_STEPS = {0: [("model.layers.0.mlp.down_proj", bytes(i % 241 for i in range(2000))), ("logits", bytes(range(256)) * 3)],
             1: [("runner.sampler/sampled_token_ids", (7).to_bytes(8, "little"))],
             2: [("model.layers.0.self_attn.qkv_proj", bytes(i % 13 for i in range(1500)))]}


def _gpu(leaf_scheme: str = hiding.NAME, run_id: str = "run-G"):
    pytest.importorskip("torch")
    from tests.check import opened

    saved = opened.NativeHostCommitter
    opened.NativeHostCommitter = lambda engine, **kw: saved(engine, leaf_scheme=leaf_scheme, **kw)
    try:
        return opened.commit_steps(GPU_STEPS, run_id=run_id)
    finally:
        opened.NativeHostCommitter = saved


def test_the_salt_stream_is_rfc8439_chacha20_at_128_bytes_per_leaf():
    assert hiding.chacha20(RFC8439_KEY, RFC8439_NONCE, 64, counter=1).hex() == RFC8439_BLOCK_1
    key = bytes(range(100, 132))
    whole = hiding.chacha20(key, RFC8439_NONCE, 64 * 9)
    assert hiding.chacha20(key, RFC8439_NONCE, 64 * 4, counter=5) == whole[320:576]
    st = hiding.StreamSalts(salt_key=key, epoch=3)
    assert st.nonce(9) == (3).to_bytes(4, "little") + (9).to_bytes(8, "little")
    assert st.salts(9, 5, 9) == hiding.chacha20(key, st.nonce(9), 128 * 9)[640:]
    assert st.salts(9, 5, 6) != hiding.StreamSalts(salt_key=key, epoch=4).salts(9, 5, 6) != st.salts(8, 5, 6)


def test_every_gpu_tree_leaf_opens_with_its_stream_salt():
    com, rc = _gpu()
    assert (rc.extra["leaf_scheme"], rc.extra["commitment_scheme"], rc.extra["salt_rule"]) == (hiding.NAME, hiding.SCHEME, hiding.SALT_STREAM)
    assert rc.extra["salt_bytes"] == 0                                  # nothing retained per leaf: the salts are the stream's
    for s in rc.steps:
        ops = com.open([(s.step, i) for i in range(s.n_leaves)])
        assert [o.salt for o in ops] == [com.stream.salts(s.step, i, i + 1) for i in range(s.n_leaves)]
        assert all(com.verify(rc, o) for o in ops)
        ro = com.open_range(s.step, 0, s.n_leaves)
        assert len(ro.salts) == 128 * s.n_leaves and com.verify_range(rc, ro)


def test_a_gpu_tree_opening_binds_its_value_and_salt():
    com, rc = _gpu()
    op, other = com.open([(0, 3), (0, 4)])
    bad_value = bytes([op.value[0] ^ 1]) + op.value[1:]
    bad_salt = op.salt[:9] + bytes([op.salt[9] ^ 4]) + op.salt[10:]
    for salt, value in ((op.salt, bad_value), (bad_salt, op.value), (other.salt, op.value), (b"", op.value), (op.salt[:-1], op.value)):
        assert not com.verify(rc, Opening(op.step, op.index, value, op.path, op.identity, salt))
    ro = com.open_range(0, 2, 6)
    assert com.verify_range(rc, ro)
    assert not com.verify_range(rc, RangeOpening(ro.step, ro.lo, ro.hi, ro.value, ro.path, ro.salts[:-128] + bytes(128)))
    assert not com.verify_range(rc, RangeOpening(ro.step, ro.lo, ro.hi, bytes([ro.value[0] ^ 1]) + ro.value[1:], ro.path, ro.salts))
    assert not com.verify_range(rc, RangeOpening(ro.step, ro.lo, ro.hi, ro.value, ro.path))


def test_a_value_guessed_without_its_salt_is_not_confirmed():
    """Under vllm-v1 a guess of a committed leaf's bytes is confirmed by recomputing its leaf, which the step tree exposes; under hm96 the
    true bytes with any salt but the committed one give another leaf, and the bare leaf of the true bytes is not in the tree."""
    plain, rc_plain = _gpu(scheme.NAME)
    hidden, rc = _gpu()
    leaf_of = lambda com, step, i, value, salt: com._range_leaves(step, i, i + 1, value, salt)[0]  # noqa: E731
    op = hidden.open([(2, 1)])[0]
    true_tree_leaf = bytes(hidden._gpu_levels[2][0][1].numpy().astype("uint32").tobytes())
    bare = leaf_of(plain, 2, 1, op.value, b"")
    assert bare == bytes(plain._gpu_levels[2][0][1].numpy().astype("uint32").tobytes())            # vllm-v1: the guess is confirmed
    assert leaf_of(hidden, 2, 1, op.value, op.salt) == true_tree_leaf
    for _ in range(4):
        assert leaf_of(hidden, 2, 1, op.value, os.urandom(128)) != true_tree_leaf
    tree = {bytes(r) for lv in hidden._gpu_levels[2] for r in lv.numpy().astype("uint32")}
    assert bare not in tree and rc.steps[2].merkle_root != rc_plain.steps[2].merkle_root


def test_every_run_takes_a_new_salt_epoch_and_a_replay_adopts_its_runs_salts():
    import copy

    com, rc = _gpu()
    blocks = {s: (bytes(com._gpu_blocks[s][0].numpy()[:com._gpu_blocks[s][1]]), [copy.copy(m) for m in com._layouts[s]]) for s in GPU_STEPS}
    frozen, epoch = copy.copy(com), com.stream.epoch

    def again():
        for s, (block, metas) in blocks.items():
            com.commit_block_offline(s, block, [copy.copy(m) for m in metas])
        return com.finalize()

    com.reset()
    assert com.stream.epoch == epoch + 1 and frozen.stream.epoch == epoch and com.stream.salt_key() == frozen.stream.salt_key()
    assert all(a.merkle_root != b.merkle_root for a, b in zip(again().steps, rc.steps))
    com.reset(salts_of=frozen)                                          # what openings_after_release's replay does
    replay = again()
    assert replay.run_root == rc.run_root and [s.root for s in replay.steps] == [s.root for s in rc.steps]
    assert frozen.verify(rc, frozen.open([(1, 0)], values_from=com)[0])


def test_a_replay_takes_the_salts_of_a_run_with_its_own_leaf_rule():
    com, _rc = _gpu()
    with pytest.raises(ValueError):
        com.adopt_salts(committer())                                    # host salts: not the GPU chunk tree's rule
    with pytest.raises(ValueError):
        com.adopt_salts(_gpu(scheme.NAME)[0])


def test_the_device_kernel_is_the_host_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("the hm96_leaves kernel runs on CUDA")
    com = NativeHostCommitter(None, gpu_tree=True)
    ext, st = com._gpu_ext_load(), com.stream
    for n, first, step in ((1, 0, 0), (777, 5, 3), (4099, 1 << 20, 1 << 40)):
        x = torch.randint(-2**31, 2**31 - 1, (n, 8), dtype=torch.int32, device="cuda")
        inner = [bytes(r) for r in x.cpu().numpy().astype("uint32")]
        got = st.hide(ext, x.clone(), step, first)
        torch.cuda.synchronize()
        assert [bytes(r) for r in got.cpu().numpy().astype("uint32")] == st.leaves_of(step, inner, first)
