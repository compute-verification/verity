"""served_h1's side computation (`served_h1/receipts.py`) is `h1tile.reference`'s receipt, unit by unit: on random s8 calls at
K = 64 and 128 (a partial row block left uncovered), and saturated operands at K = 64; and each receipt reads the unit's
position, salt and roots and every operand byte of its tile.  A call's receipt tree (`served_h1/commit.py`) is the same built in
pool tasks or leaf by leaf, opens a unit's leaf, and refuses a flipped bit or another call's salt label."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1] / "served_h1"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"served_h1_{name}", HERE / f"{name}.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RC, CM = _load("receipts"), _load("commit")
SALT, RA, RB = bytes(range(64)), bytes(range(64, 128)), bytes(range(128, 192))


def _call(m: int, n: int, k: int, seed: int) -> "RC.Call":
    g = torch.Generator().manual_seed(seed)
    return RC.Call(torch.randint(-127, 128, (m, k), generator=g, dtype=torch.int8),
                   torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8))


@pytest.mark.parametrize("m,n,k", [(64, 64, 64), (33, 96, 64), (32, 64, 128)])
def test_receipts_are_the_reference(m, n, k):
    call = _call(m, n, k, m + n + k)
    got = RC.call_receipts(call, SALT, 1 << 40, RA, RB)
    assert len(call.units) == (m // 32) * (n // 32) == got.shape[0]
    for u in range(len(call.units)):
        assert list(got[u]) == RC.reference(call, SALT, 1 << 40, RA, RB, u)[1]


def test_saturated_operands():
    call = RC.Call(torch.full((32, 64), 127, dtype=torch.int8), torch.full((32, 64), -127, dtype=torch.int8))
    assert list(RC.call_receipts(call, SALT, 5, RA, RB)[0]) == RC.reference(call, SALT, 5, RA, RB, 0)[1]


def test_receipt_reads_everything():
    call = _call(32, 32, 64, 9)
    base = RC.call_receipts(call, SALT, 3, RA, RB)[0]
    assert list(RC.call_receipts(call, SALT, 4, RA, RB)[0]) != list(base)
    assert list(RC.call_receipts(call, RB, 3, RA, RB)[0]) != list(base)
    assert list(RC.call_receipts(call, SALT, 3, RB, RA)[0]) != list(base)
    for side, (i, t) in (("a", (31, 63)), ("b", (0, 0))):
        x = getattr(call, side).clone()
        x[i, t] = x[i, t] + 1 if x[i, t] < 127 else x[i, t] - 1
        moved = RC.Call(x, call.b) if side == "a" else RC.Call(call.a, x)
        assert list(RC.call_receipts(moved, SALT, 3, RA, RB)[0]) != list(base)


def test_receipt_tree_opens_and_refuses():
    receipts = RC.call_receipts(_call(64, 96, 64, 4), SALT, 0, RA, RB)
    rows = CM.receipt_rows(receipts)
    dom = CM.domain({"call": 0, "units": len(rows)}, len(rows))
    key = bytes(64)
    pooled = CM.levels(dom, CM.split(CM.leaves((dom.binding, dom.count, key, "r/0", 0, rows))))
    direct = CM.levels(dom, [CM.leaf(dom, key, "r/0", u, bytes(r))[0] for u, r in enumerate(rows)])
    assert pooled == direct
    root, u = pooled[-1][0], 5
    assert CM.verify(dom, root, u, CM.leaf(dom, key, "r/0", u, bytes(rows[u]))[0], CM.path(pooled, u))
    flipped = bytearray(rows[u])
    flipped[17] ^= 0x10
    assert not CM.verify(dom, root, u, CM.leaf(dom, key, "r/0", u, bytes(flipped))[0], CM.path(pooled, u))
    assert not CM.verify(dom, root, u, CM.leaf(dom, key, "r/1", u, bytes(rows[u]))[0], CM.path(pooled, u))
