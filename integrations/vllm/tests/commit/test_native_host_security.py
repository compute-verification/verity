"""The Δ_commit committer (`native_host`) under the same validity conditions, through its offline core (no engine, no CUDA):
the verifier holds the ORIGINAL `RunCommitment` (step roots, layouts, run root) and checks openings against it with the production
core (`step_rows.fold_path`), then the context binding, then the run-root membership.  Each commit draws fresh step keys; a
committer that must reproduce another's bare trees takes its keys (`replay_row_keys`).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import replace

import pytest

pytest.importorskip("torch")

from verity_vllm.acquire.committer_api import RunCommitment
from verity_vllm.commit import step_rows
from verity_vllm.commit.challenge import challenge_positions
from verity_vllm.commit.committer.native_host import NativeHostCommitter

CHUNK = step_rows.ROW_BYTES


def stream(seed: int = 3, steps: int = 3, tensors: int = 4, nbytes: int = 12000) -> list[list[tuple[str, bytes]]]:
    rng = random.Random(seed)
    return [[(f"layer{k}.op", rng.randbytes(nbytes + 7 * k)) for k in range(tensors)] for _ in range(steps)]


def committer(run_id: str = "run-A", keys_of: NativeHostCommitter | None = None) -> NativeHostCommitter:
    c = NativeHostCommitter(None, run_id=run_id, hash_threads=1, program_digest=hashlib.sha256(b"program").digest())
    if keys_of is not None:
        c.replay_row_keys = dict(keys_of.row_keys)
    return c


def test_roundtrip_all_leaves_open_and_verify():
    c = committer()
    st = stream()
    run = c.commit_offline(st)
    assert len(run.steps) == 3 and all(s.n_leaves > 0 for s in run.steps)
    for s in run.steps:
        ops = c.open([(s.step, i) for i in range(s.n_leaves)])
        assert all(c.verify(run, o) for o in ops)
        # the opened bytes ARE the committed bytes at the verifier-derived (tensor, byte range)s of the row
        for o in ops:
            assert o.identity["tensors"]
            for t in o.identity["tensors"]:
                off = o.identity["byte_lo"] - c._layouts[s.step][t["tensor"]].stream_off
                assert o.value[t["first_byte"]:t["last_byte"]] == st[s.step][t["tensor"]][1][off + t["first_byte"]:off + t["last_byte"]]


def test_altered_value_rejected():
    c = committer()
    run = c.commit_offline(stream())
    op = c.open([(1, 5)])[0]
    assert c.verify(run, op)
    bad = bytearray(op.value)
    bad[3] ^= 0x80
    assert not c.verify(run, replace(op, value=bytes(bad)))
    assert not c.verify(run, replace(op, value=op.value[:-1]))
    assert not c.verify(run, replace(op, salt=bytes(len(op.salt))))


def test_reordered_and_cross_step_positions_rejected():
    c = committer()
    run = c.commit_offline(stream())
    a, b = c.open([(1, 2), (1, 9)])
    assert c.verify(run, a) and c.verify(run, b)
    assert not c.verify(run, replace(a, index=9, identity=b.identity))                     # value_2/path_2 at index 9
    assert not c.verify(run, replace(a, index=9, path=b.path, identity=b.identity))        # value_2 with path_9 at 9
    assert not c.verify(run, replace(a, index=9, path=b.path, salt=b.salt, identity=b.identity))     # ... and with salt_9
    assert not c.verify(run, replace(a, identity=b.identity))                              # inconsistent identity claim
    assert not c.verify(run, replace(a, step=2))                                           # step 1's opening claimed for step 2
    assert not c.verify(run, replace(a, step=0))


def test_equal_multiset_other_tensor_order_is_another_commitment():
    st = stream()
    c1 = committer()
    run1 = c1.commit_offline(st)
    c2 = committer(keys_of=c1)
    st2 = [list(reversed(step)) for step in st]                         # same bytes, tensors reordered
    run2 = c2.commit_offline(st2)
    assert run1.run_root != run2.run_root and all(a.root != b.root for a, b in zip(run1.steps, run2.steps))
    assert all(a.merkle_root != b.merkle_root for a, b in zip(run1.steps, run2.steps))     # not just the layout digest: the leaf order
    # an opening from run2 never authenticates under run1 (even where the bytes are equal)
    for i in range(0, run2.steps[0].n_leaves, 5):
        assert not c1.verify(run1, c2.open([(0, i)])[0])


def test_truncated_stream_rejected():
    st = stream()
    c1 = committer()
    run1 = c1.commit_offline(st)
    c2 = committer(keys_of=c1)
    run2 = c2.commit_offline([step[:-1] for step in st])               # last tensor dropped in every step
    assert run2.run_root != run1.run_root
    for s in range(3):
        assert run2.steps[s].n_leaves < run1.steps[s].n_leaves
        for i in range(0, run2.steps[s].n_leaves, 7):
            assert not c1.verify(run1, c2.open([(s, i)])[0])
    # fewer steps: another run root
    c3 = committer(keys_of=c1)
    run3 = c3.commit_offline(st[:-1])
    assert run3.run_root != run1.run_root
    # step 0's root is legitimately equal (same bytes, same ctx, same key) — the RUN root binds the step count, so a commitment presenting the
    # truncated step list under the original run root (or vice versa) is refused by the run-root membership check
    assert run3.steps[0].root == run1.steps[0].root
    fake = RunCommitment(run1.run_root, run1.steps[:2], run1.spans, 0.0, 0.0, 0, 0)
    assert not c1.verify(fake, c1.open([(0, 0)])[0])
    fake2 = RunCommitment(run3.run_root, run1.steps, run1.spans, 0.0, 0.0, 0, 0)
    assert not c1.verify(fake2, c1.open([(0, 0)])[0])


def test_root_recomputed_after_the_fact_is_not_accepted_as_prechallenge_root():
    st = stream()
    c = committer()
    run = c.commit_offline(st)
    R = run.run_root
    chal = challenge_positions(R, [s.n_leaves for s in run.steps], 16)
    ops = c.open(chal)
    assert all(c.verify(run, o) for o in ops)
    # adversary alters one challenged byte and recommits (same context) -> R'
    s0, i0 = chal[0]
    ident = c.identity_of(s0, i0)
    t = ident["tensors"][0]
    st2 = [list(step) for step in st]
    name, raw = st2[s0][t["tensor"]]
    raw2 = bytearray(raw)
    raw2[ident["byte_lo"] + t["first_byte"] - c._layouts[s0][t["tensor"]].stream_off] ^= 0x01
    st2[s0][t["tensor"]] = (name, bytes(raw2))
    c2 = committer(keys_of=c)
    run2 = c2.commit_offline(st2)
    assert run2.run_root != R
    op2 = c2.open([(s0, i0)])[0]
    assert not c.verify(run, op2)                     # against the verifier's held (prechallenge) commitment: refused
    assert c2.verify(run2, op2)                       # a valid opening of R' — which was never the fixed root
    assert challenge_positions(run2.run_root, [s.n_leaves for s in run2.steps], 16) != chal
    # substituting only the run root (keeping the original step roots) is caught by the run-root membership check
    fake = RunCommitment(run2.run_root, run.steps, run.spans, 0.0, 0.0, 0, 0)
    assert not c.verify(fake, ops[0])


def test_context_binding_program_and_run():
    st = stream()
    a = committer("run-A")
    ra = a.commit_offline(st)
    b = committer("run-B", keys_of=a)
    rb = b.commit_offline(st)
    assert ra.run_root != rb.run_root and all(x.root != y.root for x, y in zip(ra.steps, rb.steps))
    # same bytes, same keys: the step's domain (program, run context, geometry) is in every leaf, so even the bare trees differ
    assert all(x.merkle_root != y.merkle_root for x, y in zip(ra.steps, rb.steps))
    assert not a.verify(rb, a.open([(0, 0)])[0])
    assert not b.verify(ra, b.open([(0, 0)])[0])


def test_layout_digest_binds_names_dtypes_shapes():
    st = stream()
    c1 = committer()
    r1 = c1.commit_offline(st)
    c2 = committer(keys_of=c1)
    st2 = [[(n + "x", v) for n, v in step] for step in st]             # same bytes, renamed tensors
    r2 = c2.commit_offline(st2)
    assert all(a.merkle_root == b.merkle_root for a, b in zip(r1.steps, r2.steps))
    assert all(a.root != b.root for a, b in zip(r1.steps, r2.steps))   # the layout digest is inside the step root


@pytest.mark.parametrize("nbytes", [1, CHUNK - 1, CHUNK, CHUNK + 1, 5 * CHUNK + 3])
def test_partial_last_chunk_lengths_bound(nbytes):
    c = committer()
    raw = (bytes(range(256)) * (nbytes // 256 + 1))[:nbytes]
    run = c.commit_offline([[("t", raw)]])
    n = run.steps[0].n_leaves
    assert n == (nbytes + CHUNK - 1) // CHUNK
    last = c.open([(0, n - 1)])[0]
    tail = raw[(n - 1) * CHUNK:]
    assert last.value == tail + bytes(CHUNK - len(tail)) and c.verify(run, last)       # a row opens whole, zero-padded
    assert not c.verify(run, replace(last, value=last.value + b"\0"))
    assert not c.verify(run, replace(last, value=last.value[:-1] + bytes([last.value[-1] ^ 1])))
