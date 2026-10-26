"""h1_lanes' known rows (`h1_lanes/run.py`'s `known_rows`, now `pouw.leaf_lanes.known_rows`'): on a few leaves every
statement's are the rows the harness built itself before core produced them, and lane l's ``pub`` is h1_prove's tile row
(`h1unit.pub_words`) with l as its 101st word."""
from __future__ import annotations

import dataclasses
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.pouw import leaf_lanes as LL
from verity.core.protocols.pouw.circuit import seeds as SD

_spec = importlib.util.spec_from_file_location("h1_lanes_run", Path(__file__).resolve().parents[1] / "h1_lanes" / "run.py")
RN = sys.modules[_spec.name] = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RN)
U = RN.U


def _old_bits_words(bits: list[int]) -> list[int]:
    b = list(bits) + [0] * (-len(bits) % 16)
    return np.packbits(np.asarray(b, np.uint8), bitorder="little").view("<u2").astype(np.uint16).tolist()


def _old_known_rows(ctx, w, kind: str, ln: int | None, index: int) -> dict:
    """run.py's `known_rows` (with its `pub_words` and `bits_words`) as it was before it called `leaf_lanes.known_rows`."""
    vals = LL.known(*RN.RC, kind, ln)
    out = {LL.OK: (vals[LL.OK], [int(x) for x in U.OK], U.ok_salt(f"{ctx.name}#{index}", ctx.position))}
    if kind == LL.LANE:
        lf = w.leaf
        bits = (SD.leaves(lf.salt, True) + SD.position_leaves(lf.position, True) + SD.leaves(w.roots[0], True)
                + SD.leaves(w.roots[1], True) + [(ln >> t) & 1 for t in range(16)])
        out[LL.PUB] = (vals[LL.PUB], _old_bits_words(bits), U.pub_salt(f"{ctx.name}#lane{ln}", lf.position))
        out[LL.ADMIT] = (vals[LL.ADMIT], _old_bits_words(lf.admit), U.pub_salt(f"{ctx.name}#admit", lf.position))
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_known_rows_are_the_harness_own(seed: int):
    """Seed 1's leaf is h1_lanes' (every row admitted, `leaf_data`), the others' admit bits drawn."""
    rng = random.Random(seed)
    position = rng.getrandbits(64)
    leaf = RN.leaf_data(seed, 32, rng.randbytes(64), position)
    if seed > 1:
        leaf = dataclasses.replace(leaf, admit=[rng.getrandbits(1) for _ in range(RN.RC[0])])
    roots = (rng.randbytes(64), rng.randbytes(64))
    w = RN.World("honest", leaf, roots, {}, "", "", None, {}, {})
    ctx = RN.Ctx(Path("."), f"h1-lanes-{seed}", 32, position)
    for index, (kind, ln) in enumerate(LL.statements(*RN.RC)):
        got = RN.known_rows(ctx, w, kind, ln, index)
        assert got == _old_known_rows(ctx, w, kind, ln, index), (kind, ln)
        if kind == LL.LANE:
            assert got[LL.PUB][1] == U.pub_words(leaf.salt, leaf.position, roots).tolist() + [ln]


def _rows_or_refusal(fn, ctx, w, kind: str, ln: int | None, index: int):
    try:
        return fn(ctx, w, kind, ln, index)
    except (ValueError, OverflowError):
        return "refused"


def test_off_the_served_leaf_the_known_rows_refuse_and_take_what_the_harness_did():
    """Admit rows of any length and positions in and out of a u64 (the inputs `PouwWindow.knownRows`' difftest draws
    through `known_rows`): the rows, or the refusal, are the old harness's."""
    rng = random.Random(0)
    refused = 0
    for _ in range(200):
        position = rng.choice((0, (1 << 32) - 1, 1 << 32, (1 << 64) - 1, rng.getrandbits(64), 1 << 64, -1))
        leaf = RN.Leaf([], [], [], [], (), [rng.getrandbits(1) for _ in range(rng.choice((32, 0, 1, 15, 17, 79)))],
                       rng.randbytes(64), position)
        w = RN.World("honest", leaf, (rng.randbytes(64), rng.randbytes(64)), {}, "", "", None, {}, {})
        ctx = RN.Ctx(Path("."), "h1-lanes", 32, position)
        ln = rng.choice((None, 0, 31, rng.randrange(32)))
        kind, index = (LL.COMBINE, 32) if ln is None else (LL.LANE, ln)
        got = _rows_or_refusal(RN.known_rows, ctx, w, kind, ln, index)
        assert got == _rows_or_refusal(_old_known_rows, ctx, w, kind, ln, index), (position, len(leaf.admit), ln)
        refused += got == "refused"
    assert 0 < refused < 200
