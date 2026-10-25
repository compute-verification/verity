"""`Plan.from_array`, a pass's plan checked and encoded column by column from an array's buffer, against `Plan` on the
same rows: the same plan, encoding, digest and positions, or the same refusal, at every edge of the format, on the
pinned vectors' plans, and on random plans and their near misses."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.remote_bandwidth import receipts
from verity.core.protocols.remote_bandwidth.receipts import Plan

N = 100
TOP = (1 << 64) - 1


def rows(a: np.ndarray) -> object:
    """What the reference is handed for an array: its rows as tuples."""
    out = a.tolist()
    return tuple(tuple(r) if type(r) is list else r for r in out) if type(out) is list else out


def same(a: np.ndarray, blocks: object) -> Plan | str:
    """`Plan.from_array(a, blocks)` and `Plan(rows(a), blocks)` agree: the plan, or the refusal's message."""
    outcomes = []
    for build in (lambda: Plan(rows(a), blocks), lambda: Plan.from_array(a, blocks)):
        try:
            outcomes.append(build())
        except ValueError as e:
            outcomes.append(str(e))
    want, got = outcomes
    if isinstance(want, str):
        assert got == want, (a, blocks)
        return want
    assert isinstance(got, Plan) and got == want and hash(got) == hash(want)
    assert got.encode() == want.encode() and got.digest == want.digest and got.runs == want.runs
    assert got.blocks == want.blocks and got.length == want.length
    n, rng = want.length, random.Random(want.length)
    positions = sorted({0, n - 1, *(rng.randrange(n) for _ in range(512))})
    assert [got.block(p) for p in positions] == [want.block(p) for p in positions]
    return want


RUN = "a run is (first, count) with count ≥ 1, inside the store"


@pytest.mark.parametrize("runs, blocks, want", [
    ([[0, N]], N, None),                                 # the whole store
    ([[N - 1, 1]], N, None),                             # the last block
    ([[7, 3], [0, 2], [12, 1]], N, None),
    ([[5, 3], [0, 5]], N, None),                         # touching, out of order: not one run
    ([[0, 1]], 1, None),
    ([[N, 1]], N, RUN),                                  # past the store
    ([[N - 1, 2]], N, RUN),
    ([[0, N + 1]], N, RUN),
    ([[0, 0]], N, RUN),                                  # an empty run
    ([[-1, 2]], N, RUN),
    ([[5, -1]], N, RUN),
    ([[0, 5], [5, 3]], N, "a run that starts where the one before it ends is part of it"),
    ([[0, 0], [0, 5], [5, 1]], N, RUN),                  # the run rule comes first
    ([[0, 5], [5, 1], [2, 1]], N, "a run that starts where the one before it ends is part of it"),
    ([[0, 5], [3, 4]], N, "a plan's runs are disjoint: no block is read twice in a pass"),
    ([[10, 5], [0, 11]], N, "a plan's runs are disjoint: no block is read twice in a pass"),
    ([[3, 1], [3, 2]], N, "a plan's runs are disjoint: no block is read twice in a pass"),
    ([[3, 1], [3, 1]], N, "a plan's runs are disjoint: no block is read twice in a pass"),
    ([[0, 10], [2, 1]], N, "a plan's runs are disjoint: no block is read twice in a pass"),
    ([[20, 2], [0, 10], [9, 1]], N, "a plan's runs are disjoint: no block is read twice in a pass"),
])
def test_the_format_s_edges(runs, blocks, want) -> None:
    for dtype in (np.int64, np.int32, np.int16, np.int8, ">i8"):
        got = same(np.array(runs, dtype), blocks)
        assert isinstance(got, Plan) if want is None else got == want
    if all(x >= 0 for r in runs for x in r):
        for dtype in (np.uint64, np.uint32, np.uint8, ">u8"):
            same(np.array(runs, dtype), blocks)


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16, np.uint8])
def test_native_integers_are_read_from_the_buffer(monkeypatch, dtype) -> None:
    """The tests above aren't vacuous: an array of native integers in C order is checked and encoded from its buffer,
    never through `Plan(rows)`, which every other array falls back to."""
    want = Plan(((7, 3), (0, 2), (12, 1)), N)
    monkeypatch.setattr(Plan, "__init__", lambda *_: pytest.fail("fell back to Plan(rows)"))
    got = Plan.from_array(np.array([[7, 3], [0, 2], [12, 1]], dtype), N)
    assert got == want and got.digest == want.digest and list(got) == list(want)
    with pytest.raises(ValueError, match="disjoint"):
        Plan.from_array(np.array([[0, 5], [3, 4]], dtype), N)


@pytest.mark.parametrize("runs, blocks, accepted", [
    ([[TOP - 1, 1]], TOP, True),                         # the last block of the largest store
    ([[0, TOP]], TOP, True),
    ([[TOP, 1]], TOP, False),                            # first + count is 2^64: it wraps in u64
    ([[1, TOP]], TOP, False),
    ([[1 << 63, 1 << 63]], TOP, False),
    ([[TOP, TOP]], TOP, False),
    ([[(1 << 63) - 1, 1]], 1 << 63, True),               # int64's largest first
    ([[0, 1 << 32], [1 << 32, 1]], 1 << 33, False),
])
def test_u64_edges_do_not_overflow(runs, blocks, accepted) -> None:
    assert isinstance(same(np.array(runs, np.uint64), blocks), Plan) == accepted


@pytest.mark.parametrize("blocks", [0, -1, True, 1.0, 1 << 64, None])
def test_a_store_s_blocks_are_checked_first(blocks) -> None:
    assert same(np.array([[0, 1]]), blocks) == "a store is 1 to 2^64 − 1 blocks"


@pytest.mark.parametrize("a", [
    np.zeros((0, 2), np.int64),                          # no runs
    np.zeros((0,), np.int64),
    np.array([0, 5]),                                    # one run, flat
    np.array(3),
    np.array([[0, 5, 1]]),
    np.array([[0]]),
    np.zeros((2, 0), np.int64),
    np.array([[[0, 5]]]),
    np.array([[0.0, 5.0]]),
    np.array([[True, True]]),
    np.array([[0, 5], [7, 1]], object),                  # Python ints: a plan
    np.array([["0", "5"]]),
    np.array([[0, 5]], np.int64)[:, ::-1][:, ::-1],      # a strided view
    np.asfortranarray(np.array([[7, 3], [0, 2]])),
])
def test_other_shapes_and_dtypes_meet_the_same_rule(a) -> None:
    same(a, N)


def test_the_pinned_plans() -> None:
    pinned = json.loads(Path(__file__).with_name("vectors.json").read_text())
    for p in pinned["passes"]["passes"]:
        plan = same(np.array(p["plan"], np.int64), pinned["registration"]["blocks"])
        assert plan.encode().hex() == p["plan_encoding"] and plan.digest.hex() == p["plan_digest"]
        assert plan.digest == hashlib.sha512(receipts.PLAN_DOMAIN + bytes.fromhex(p["plan_encoding"])).digest()


def random_plan(rng: random.Random, blocks: int, runs: int) -> list[list[int]]:
    """Canonical runs of distinct blocks in a random order: disjoint ranges of the store, shuffled, with no run
    starting where the one before it ends."""
    cuts = sorted(rng.sample(range(1, blocks), min(blocks - 1, 2 * runs)))
    ranges = [[a, b - a] for a, b in zip([0, *cuts], [*cuts, blocks])][::2]
    rng.shuffle(ranges)
    out: list[list[int]] = []
    for f, c in ranges:
        if out and out[-1][0] + out[-1][1] == f:
            out[-1][1] += c
        else:
            out.append([f, c])
    return out


def near_miss(rng: random.Random, runs: list[list[int]], blocks: int) -> list[list[int]]:
    """`runs` broken one way: a run moved, grown, emptied, repeated or split, or a run taken whole or pushed off the
    end of the store."""
    out = [list(r) for r in runs]
    k = rng.randrange(len(out))
    match rng.randrange(6):
        case 0:
            out[k][0] += rng.choice((-2, -1, 1, 2))
        case 1:
            out[k][1] += rng.choice((-1, 1, 2))
        case 2:
            out.insert(rng.randrange(len(out) + 1), list(out[k]))
        case 3:
            f, c = out[k]
            if c > 1:
                out[k:k + 1] = [[f, c - 1], [f + c - 1, 1]]
        case 4:
            out.append([rng.randrange(blocks), 1])
        case 5:
            out[k] = [blocks - out[k][1] + rng.randrange(2), out[k][1]]
    return out


def test_random_plans_and_their_near_misses() -> None:
    rng = random.Random(20261012)
    refusals = set()
    for _ in range(600):
        blocks = rng.choice((1, 2, 3, 17, 100, 4096, (1 << 64) - 1))
        if blocks < 3:
            runs = [[0, blocks]]
        elif blocks >= 1 << 32:
            runs = [[rng.randrange(blocks >> 1), rng.randrange(1, 1 << 40)]]
        else:
            runs = random_plan(rng, blocks, rng.randrange(1, max(2, blocks // 4)))
        assert isinstance(same(np.array(runs, np.uint64), blocks), Plan)
        broken = near_miss(rng, runs, blocks)
        dtype = np.int64 if all(-(1 << 63) <= x < 1 << 63 for r in broken for x in r) else np.uint64
        if dtype is np.uint64 and any(x < 0 for r in broken for x in r):
            continue
        out = same(np.array(broken, dtype), blocks)
        if isinstance(out, str):
            refusals.add(out)
    assert len(refusals) == 3


def test_a_kv_sized_plan() -> None:
    """245k runs of a block each, every other block of 2^22, shuffled: a decode pass over scattered KV pages."""
    rng = random.Random(7)
    a = np.array([[2 * j, 1] for j in rng.sample(range(1 << 21), 245_000)], np.int64)
    assert same(a, 1 << 22).length == 245_000
    k = rng.randrange(len(a) - 1)
    a[k] = a[k + 1]
    assert same(a, 1 << 22) == "a plan's runs are disjoint: no block is read twice in a pass"
