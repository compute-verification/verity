"""The offline core under ``rows-sha512`` (``NativeHostCommitter(None, gpu_tree=True, commit_scheme="rows-sha512")``,
``commit_block_offline``) commits ``step_rows``'s tree over each packed step block, salted from a fresh 64-byte key per step, into
64-byte step and run roots: the CPU stand-in of the windowed worker's rows-sha512 run (``test_windowed_rows_sha512`` holds that one to
the same roots).  Its leaf and range openings carry their leaves' salts and verify against those roots, through ``OpenedReader`` as the
value checks read them; a tampered row, salt, sibling or root does not verify."""
from __future__ import annotations

import random
from dataclasses import replace

import pytest

pytest.importorskip("torch")

from tests.check import opened as O
from verity_vllm.commit import step_rows as R
from verity_vllm.commit.committer.native_host import NativeHostCommitter
from verity_vllm.commit.opened import OpenedReader, OpeningNotVerified

RNG = random.Random(63)
K = R.ROW_BYTES // 256                   # sizes in 256-byte units, so every step but the first spans several rows
STEPS = {s: [(f"m{s}.{i}", RNG.randbytes(n)) for i, n in enumerate(sizes)]
         for s, sizes in enumerate([(100 * K,), (700 * K, 3, 513 * K), (2048 * K, 1000 * K)])}


def committed():
    return O.commit_steps(STEPS, commit_scheme=R.NAME, run_id="rows-offline")


def test_steps_commit_step_rows_trees_under_fresh_keys():
    com, rc = committed()
    assert com.rows and com.chunk == R.ROW_BYTES and com.name.endswith(R.NAME + "+retain_host")
    assert sorted(com.row_keys) == list(STEPS) and len(set(com.row_keys.values())) == len(STEPS)
    assert all(len(k) == R.KEY_BYTES for k in com.row_keys.values())
    for s, sc in enumerate(rc.steps):
        blk = bytes(com._gpu_blocks[s][0].numpy().tobytes())
        rows = [blk[i:i + R.ROW_BYTES].ljust(R.ROW_BYTES, b"\0") for i in range(0, len(blk), R.ROW_BYTES)]
        ctx = com._ctx_digest(s)
        tree_root = R.levels(R.domain(com.program_digest, ctx, com._geo_digest, len(rows)), rows, com.row_keys[s], 8 * R.ROW_BYTES)[-1][0]
        assert sc.n_leaves == len(rows) and sc.merkle_root == tree_root
        assert sc.root == R.step_root(com.program_digest, ctx, com._geo_digest, com.layout_digest(com._layouts[s]), len(rows), tree_root)
    assert rc.run_root == R.run_root(com.program_digest, com._geo_digest, [s.root for s in rc.steps]) and len(rc.run_root) == R.ROOT_BYTES
    assert rc.extra["commit_scheme"] == R.NAME
    again, rc2 = committed()
    assert rc2.run_root != rc.run_root and not set(again.row_keys.values()) & set(com.row_keys.values()), "each run's keys are fresh"


def test_leaf_openings_carry_their_salts_and_verify():
    com, rc = committed()
    sample = [(s, i) for s, sc in enumerate(rc.steps) for i in sorted({0, sc.n_leaves // 2, sc.n_leaves - 1})]
    ops = com.open(sample)
    assert all(com.verify(rc, o) for o in ops)
    assert all(o.salt == R.salt(com.row_keys[o.step], o.index) and len(o.path) == len(com._gpu_levels[o.step]) - 1 for o in ops)
    o = next(o for o in ops if o.path)
    assert not com.verify(rc, replace(o, value=bytes([o.value[0] ^ 1]) + o.value[1:]))
    assert not com.verify(rc, replace(o, salt=bytes([o.salt[0] ^ 1]) + o.salt[1:]))
    assert not com.verify(rc, replace(o, path=o.path[:-1]))
    assert not com.verify(rc, replace(o, path=o.path + o.path[:1]))
    assert not com.verify(rc, replace(o, path=(bytes(R.ROOT_BYTES),) + o.path[1:]))
    assert not com.verify(replace(rc, steps=[replace(sc, merkle_root=bytes(R.ROOT_BYTES)) for sc in rc.steps]), o)
    assert not com.verify(replace(rc, run_root=rc.run_root[:32]), o)


def test_range_openings_verify_through_the_opened_reader():
    com, rc = committed()
    reader = OpenedReader(com, rc)
    for s, tensors in STEPS.items():
        for name, raw in tensors:
            assert reader(s, name, 0, len(raw)) == raw
            assert reader(s, name, 1, len(raw) - 1) == raw[1:-1]
    assert reader.stats["failed"] == 0 and reader.record()["steps_bound"] == len(STEPS)
    ro = com.open_range(1, 1, 4)
    assert len(ro.salts) == 3 * R.SALT_BYTES and com.verify_range(rc, ro)
    for bad in (replace(ro, value=bytes([ro.value[0] ^ 1]) + ro.value[1:]), replace(ro, salts=bytes(len(ro.salts))),
                replace(ro, salts=ro.salts[:-1]), replace(ro, path=ro.path[:-1]), replace(ro, lo=0, hi=3)):
        assert not com.verify_range(rc, bad)
    assert not com.verify_range(replace(rc, run_root=bytes(R.ROOT_BYTES)), ro)
    assert not com.verify_range(replace(rc, run_root=rc.run_root[:32]), ro)
    blk = O.retained_block(com, 2)
    blk[0] ^= 1
    with pytest.raises(OpeningNotVerified, match="does not verify"):
        OpenedReader(com, rc)(2, "m2.0", 0, 4)


def test_what_the_offline_rows_core_refuses():
    with pytest.raises(ValueError, match="commit_scheme 'rows-sha256'"):
        NativeHostCommitter(None, gpu_tree=True, commit_scheme="rows-sha256")
    with pytest.raises(ValueError, match="GPU-tree steps"):
        NativeHostCommitter(None, commit_scheme=R.NAME)
    com, _rc = committed()
    with pytest.raises(ValueError, match=R.NAME):
        com.add_padding_steps([object()])
