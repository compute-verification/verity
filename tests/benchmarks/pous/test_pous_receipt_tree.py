"""The P2 session prover's receipt tree (`ReceiptTree`, built in worker processes) is `verity.commitments.merkle.levels`
over the pass's leaves and pads: its root and every level equal one `levels` call's, for leaf counts at the bounds of
a power of two and of a chunk, with and without a pool.  Seeded and exhaustive over the sizes it names: the leaf counts
are fixed, the receipts and salts come from one seeded generator."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

from verity.commitments import merkle
from verity.protocols.resource_accounting.memory.poub.prover.p2_stage import bandwidth as B
from verity.protocols.resource_accounting.memory.poub.verifier import receipts

SEED = hashlib.sha512(b"test_pous_receipt_tree/seed").digest()
DIGEST = receipts.DIGEST_BYTES
SIZES = sorted({*range(1, 70), 127, 128, 129, 255, 256, 257, *random.Random(2).sample(range(70, 400), 200)})
LARGE = (1023, 1024, 1025, 3001, 4096, 4097)
CHUNKS = (4, 16, B.ReceiptTree.CHUNK)


def passes(n_leaves: int, rng: np.random.Generator) -> tuple[receipts.Plan, np.ndarray, bytes]:
    plan = receipts.Plan.whole(receipts.PER_LEAF * (n_leaves - 1) + int(rng.integers(1, receipts.PER_LEAF + 1)))
    ys = rng.integers(0, 256, (plan.length, DIGEST), dtype=np.uint8)
    salts = rng.integers(0, 256, n_leaves * receipts.SALT_BYTES, dtype=np.uint8).tobytes()
    return plan, ys, salts


def one_levels_call(plan: receipts.Plan, ys: np.ndarray, salts: bytes) -> tuple[tuple[bytes, ...], ...]:
    d, n = receipts.domain(SEED, plan), receipts.leaves(plan.length)
    leaves = [d.leaf(c, c, receipts.SCHEMA, receipts.hidden(ys[receipts.PER_LEAF * c:receipts.PER_LEAF * (c + 1)].tobytes(),
                                                          salts[receipts.SALT_BYTES * c:receipts.SALT_BYTES * (c + 1)]))
              for c in range(n)]
    pads = [d.pad(rank) for rank in range(n, 1 << merkle.merkle_depth(n))]
    return merkle.levels(d.domain_id, [*leaves, *pads], hash=d.hash)


def split(level: bytes) -> tuple[bytes, ...]:
    return tuple(level[i:i + DIGEST] for i in range(0, len(level), DIGEST))


def agrees(plan: receipts.Plan, ys: np.ndarray, salts: bytes, pool: ProcessPoolExecutor | None = None) -> None:
    got, ref = B.ReceiptTree(SEED, plan, ys, salts, pool), one_levels_call(plan, ys, salts)
    assert tuple(split(level) for level in got.levels) == ref
    assert got.root == ref[-1][0]


@pytest.mark.parametrize("chunk", CHUNKS)
def test_receipt_tree_is_one_levels_call(chunk, monkeypatch) -> None:
    monkeypatch.setattr(B.ReceiptTree, "CHUNK", chunk)
    rng = np.random.default_rng(chunk)
    for n in SIZES:
        agrees(*passes(n, rng))


@pytest.mark.parametrize("chunk", (64, B.ReceiptTree.CHUNK))
def test_receipt_tree_over_thousands_of_leaves_is_one_levels_call(chunk, monkeypatch) -> None:
    monkeypatch.setattr(B.ReceiptTree, "CHUNK", chunk)
    rng = np.random.default_rng(chunk)
    for n in LARGE:
        agrees(*passes(n, rng))


def test_receipt_tree_in_a_pool_is_one_levels_call(monkeypatch) -> None:
    monkeypatch.setattr(B.ReceiptTree, "CHUNK", 8)
    rng = np.random.default_rng(8)
    with ProcessPoolExecutor(2, mp_context=mp.get_context("fork")) as pool:
        for n in (1, 7, 8, 9, 16, 17, 100, 129):
            agrees(*passes(n, rng), pool)


@pytest.mark.parametrize("n", [B.ReceiptTree.CHUNK - 1, B.ReceiptTree.CHUNK, B.ReceiptTree.CHUNK + 1])
def test_receipt_tree_at_its_chunk_is_one_levels_call(n) -> None:
    agrees(*passes(n, np.random.default_rng(n)))
