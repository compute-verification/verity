"""A pass's plan and its tree as functions of their arguments alone, which the store's `registry` reads and hands on
under their names: the seed chained in a window (`chain`), a pass's blocks as canonical runs (`runs_of`, `canonical`,
`distinct`), the PoUB driver's `Plan` over them and its digest (`plan_of`, `plan_digest`), the driver's receipt tree by
its leaves and nodes (`leaf_digest`, `node_of`, `aligned`), and a byte string under its length (`frame`).
"""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from verity.protocols.accounting.poub.verifier import receipts as rb
from verity.protocols.accounting.poub.verifier.receipts import PER_LEAF, Plan

from .interface import Segment


def frame(b: bytes) -> bytes:
    return struct.pack(">Q", len(b)) + b


def chain(challenge: bytes, previous: bytes, index: int) -> bytes:
    """Pass ``index``'s seed in a window: the driver's chained `rb.seed`, ``SHA512(D ‖ chal_t ‖ u32be index ‖ root
    before it)``."""
    return rb.seed(challenge, index, previous)


def runs_of(blocks: np.ndarray | Sequence[int]) -> np.ndarray:
    """``(R, 2)`` int64 ``(first, count)``: ``blocks`` (in read order) as runs of consecutive indices."""
    b = np.asarray(blocks, np.int64)
    if not len(b):
        return np.zeros((0, 2), np.int64)
    cut = np.flatnonzero(np.diff(b) != 1) + 1
    starts = np.concatenate([[0], cut])
    return np.stack([b[starts], np.diff(np.concatenate([starts, [len(b)]]))], axis=1)


def canonical(runs: np.ndarray) -> np.ndarray:
    """``runs`` with each run that starts where the one before it ends merged into it (the driver's canonical plan)."""
    if len(runs) < 2:
        return runs
    new = np.concatenate([[True], runs[1:, 0] != runs[:-1, 0] + runs[:-1, 1]])
    at = np.flatnonzero(new)
    return np.stack([runs[at, 0], np.add.reduceat(runs[:, 1], at)], axis=1)


def distinct(runs: np.ndarray) -> bool:
    """No block is in two runs."""
    r = runs[np.argsort(runs[:, 0], kind="stable")]
    return bool((r[:-1, 0] + r[:-1, 1] <= r[1:, 0]).all())


def plan_digest(runs: np.ndarray) -> bytes:
    """`Plan.digest` of canonical ``runs``, without building the `Plan`: its construction and digest loop over the runs
    in Python, about 200 ms a pass at the 245k runs of a step with scattered live pages."""
    return hashlib.sha512(rb.PLAN_DOMAIN + struct.pack(">I", len(runs)) + np.asarray(runs, ">u8").tobytes()).digest()


def plan_of(runs: np.ndarray, blocks: int) -> Plan:
    """The driver's `Plan` of canonical ``runs`` in a store of ``blocks`` blocks; it refuses a plan that reads a block
    twice."""
    return Plan(tuple(map(tuple, runs.tolist())), blocks)


def leaf_digest(dom: Any, c: int, receipts: bytes, salt: bytes) -> bytes:
    """Leaf ``c`` of the driver's receipt tree: its receipts `rb.hidden` under ``salt``."""
    return dom.leaf(c, c, rb.SCHEMA, rb.hidden(receipts, salt))


def node_of(segment: Segment) -> tuple[int, int]:
    """``(level, index)``: the least node of the pass's tree over the segment's leaves, its subtree's root when the
    segment is `aligned`."""
    a, b = segment.position // PER_LEAF, (segment.position + segment.count - 1) // PER_LEAF
    level = (a ^ b).bit_length()
    return level, a >> level


def aligned(segment: Segment) -> bool:
    """Whether the segment is exactly a subtree: ``PER_LEAF · 2^h`` blocks from a multiple of that."""
    level, index = node_of(segment)
    return segment.count == PER_LEAF << level and segment.position == (index << level) * PER_LEAF


def _node(dom: Any, leaves: int, level: int, index: int, leaf: Callable[[int], bytes], memo: dict) -> bytes:
    """Node ``(level, index)`` of the tree over ``leaves`` leaves (``leaf(c)``), pads past them; ``memo`` holds the nodes
    known and gains those computed."""
    key = (level, index)
    if key not in memo:
        if level == 0:
            memo[key] = leaf(index) if index < leaves else dom.pad(index)
        else:
            memo[key] = dom.node(level - 1, index, _node(dom, leaves, level - 1, 2 * index, leaf, memo),
                                 _node(dom, leaves, level - 1, 2 * index + 1, leaf, memo))
    return memo[key]
