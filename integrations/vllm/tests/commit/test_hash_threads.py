"""`--hash-threads` / HASH_THREADS parallelizes the Commit's hashing only: the leaves, the step roots and the run root are the same at any
thread count.  A packed step stream is split into chunk-aligned ranges, one per thread, and the leaves come back in stream order."""

from __future__ import annotations

import hashlib
import random

import pytest

from verity_vllm.commit.committer.native_host import NativeHostCommitter

CHUNK = 64
PROGRAM = hashlib.sha256(b"program").digest()


def _stream(steps: int = 3) -> list[list[tuple[str, bytes]]]:
    rng = random.Random(7)
    return [[(f"layer{k}.op", rng.randbytes(900 + 37 * k)) for k in range(4)] for _ in range(steps)]


def _committer(threads: int) -> NativeHostCommitter:
    return NativeHostCommitter(None, run_id="r", chunk=CHUNK, hash_threads=threads, program_digest=PROGRAM)


@pytest.mark.parametrize("threads", [2, 12])
def test_the_run_root_is_the_same_at_any_hash_thread_count(threads):
    one, many = _committer(1).commit_offline(_stream()), _committer(threads).commit_offline(_stream())
    assert len(one.run_root) == 32 and many.run_root == one.run_root
    assert [s.root for s in many.steps] == [s.root for s in one.steps]


@pytest.mark.parametrize("threads", [2, 12])
def test_a_packed_stream_hashes_to_the_same_leaves_in_order(threads):
    packed = memoryview(random.Random(3).randbytes(CHUNK * 517 + 29))
    assert _committer(threads)._hash_leaves(packed) == _committer(1)._hash_leaves(packed)
