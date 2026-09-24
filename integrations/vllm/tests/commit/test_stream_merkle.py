"""Tests for the streaming builder and replay-to-open (CPERF 2026-09-17 addendum).

    PYTHONPATH=. pytest -q tests/commit/test_stream_merkle.py

1. streaming root == batch ``fasttree.fold`` root == production ``MerkleTree`` root for every leaf count 0..130
   (leaves streamed in production tree order);
2. commit a deterministic value stream, retain ONLY (root, leaf_count), re-run the producer, regenerate an opening
   for a chosen leaf and verify it with the PRODUCTION ``verify_opening``; a producer that deviates in one byte
   yields a different root and a rejected opening;
3. resident state of the streaming builder stays O(log n) digests while ``MerkleTree`` holds all levels;
4. a stored RU of a synthetic run streamed in tree order reproduces ``commitments.json``'s root.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator

import pytest

from verity_vllm.commit import identity
from verity_vllm.commit.merkle import MerkleTree, Opening, verify_opening

from verity_vllm.commit import fasttree as ft
from tests.commit.stream_merkle import StreamingMerkle, replay_to_open, stream_in_tree_order, stream_root

PROFILE = "ec44fc6dd686"


def _ids(n: int) -> list[str]:
    # ids that sort in emission order under identity.tree_sorted: one op per position
    ids = [f"p={PROFILE}/r=r0/t=0/ph=decode/l=0/op=stream.leaf{i:06d}/port=0" for i in range(n)]
    assert identity.tree_sorted(ids) == ids
    return ids


def _producer(n: int, seed: int = 7, flip: int | None = None):
    ids = _ids(n)

    def gen() -> Iterator[tuple[bytes, bytes]]:
        rng = random.Random(seed)
        for i, cid in enumerate(ids):
            v = bytearray(rng.getrandbits(8) for _ in range(1152))
            if flip is not None and i == flip:
                v[0] ^= 1
            yield ft.leaf_header(cid, "bfloat16", (576,)), bytes(v)

    return ids, gen


@pytest.mark.parametrize("n", list(range(0, 40)) + [63, 64, 65, 100, 127, 128, 129, 130])
def test_streaming_root_matches_batch_and_production(n: int) -> None:
    ids, gen = _producer(n)
    leaves = list(gen())
    root_stream, count, _ = stream_root(leaves)
    root_batch = ft.fold([ft.leaf_digest(h, v) for h, v in leaves])
    assert count == n
    assert root_stream == root_batch
    if n:
        prod = MerkleTree({cid: ft.leaf_digest(h, v) for cid, (h, v) in zip(ids, leaves, strict=True)}, "t")
        assert prod.root == root_stream


@pytest.mark.parametrize("n,index", [(1, 0), (2, 1), (3, 2), (5, 4), (7, 3), (12, 11), (33, 32), (100, 57), (130, 129), (364, 200)])
def test_replay_to_open_verifies_with_production_verifier(n: int, index: int) -> None:
    ids, gen = _producer(n)
    # phase 1: commit and DISCARD everything but the root and the leaf count
    root, count, _ = stream_root(gen())
    # phase 2: challenge -> deterministic replay regenerates the opening
    root2, count2, path, value = replay_to_open(gen, index)
    assert (root2, count2) == (root, count)
    opening = Opening(tree="t", root=root, leaf_count=count, index=index, canonical_id=ids[index], dtype="bfloat16",
                      shape=(576,), value=value, path=tuple(path))
    assert verify_opening(root, count, opening)
    # a replay that is not bit-exact fails to reproduce the root and its opening is rejected
    _, bad = _producer(n, flip=(index + 1) % n if n > 1 else 0)
    root3, _, path3, value3 = replay_to_open(bad, index)
    assert root3 != root
    assert not verify_opening(root, count, opening.replace(path=tuple(path3), value=value3))


def test_resident_state_is_logarithmic() -> None:
    n = 11586  # a prefill RU
    ids, gen = _producer(n)
    b = StreamingMerkle()
    peak = 0
    for h, v in gen():
        b.add(h, v)
        peak = max(peak, b.resident_digests())
    b.finish()
    assert peak <= n.bit_length() + 1  # one pending digest per level
    # tracked: plus one sibling per level
    b2 = StreamingMerkle(track=1234)
    peak2 = 0
    for h, v in gen():
        b2.add(h, v)
        peak2 = max(peak2, b2.resident_digests())
    b2.finish()
    assert peak2 <= 2 * (n.bit_length() + 1)
    assert len(b2.path) == len(MerkleTree({c: bytes(32) for c in ids}, "t").levels) - 1


def test_stored_ru_streamed_in_tree_order_matches_commitments(tmp_path) -> None:
    from verity_vllm.observe.capture_v1 import CaptureReader
    from verity_vllm.commit.merkle import build_commitments
    from tests.commit.synthetic import build_toy_run

    run = tmp_path / "toy"
    build_toy_run(run, requests=2, prompt_len=4, gen=3)
    reader = CaptureReader(run)
    doc = build_commitments(reader, recompute_leaves=True, write=True)
    reader = CaptureReader(run)
    for rid, step in reader.ru_keys():
        root, leaves, peak = stream_in_tree_order(reader, rid, step)
        rec = doc["trees"][f"ru/{rid}/{step:06d}"]
        assert root.hex() == rec["root"]
        assert leaves == rec["leaf_count"]
        assert peak < 64 * (leaves.bit_length() + 2)


def test_noop_backend_streams_too() -> None:
    _, gen = _producer(10)
    root, count, _ = stream_root(gen(), ft._Noop)
    assert count == 10 and root == bytes(32)


def test_sha256_backend_is_hashlib() -> None:
    assert StreamingMerkle().new is hashlib.sha256
