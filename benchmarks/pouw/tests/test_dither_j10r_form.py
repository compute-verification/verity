"""Dither's J10-R units ($DITHER_SO_UNITS=j10r: `benchmarks/hash/dither/j10r_units.py`, the host units the gate holds the
GPU's receipts to word for word) in RT-ALGO-16's verifier form (#3574): a call's first position u0 is credit's (the
window's earlier calls' units), its unit u = ri (N / 32) + cj at position u0 + u reads its call's seed_c at u0 and the
blocks core's credit derives for tile t = u at the call's slot M (`core_check.j10r_blocks`), so its public rows are
credit's `public_table` and its receipt `h1tile.j10r_reference`'s at them; and `forms.form_of` labels such an arm by
#3574 when it is on credit's layout (`forms.j10r_part`)."""
from __future__ import annotations

import functools
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import credit as Cr
from verity.ml.protocols.pouw.circuit import h1tile as HT

DITHER = Path(__file__).resolve().parents[2] / "hash" / "dither"
SALT, RA, RB = (hashlib.sha512(x).digest() for x in (b"salt", b"A", b"B"))
TILE = (32, 32)
#: Llama-3.1-8B's linears (n, K): qkv, o, gate_up, down
LLAMA = [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)]


@functools.cache
def dither(name: str):
    """`benchmarks/hash/dither/<name>.py`, imported as its scripts import it (its directory on sys.path, and j10r_units'
    own entries); sys.path and the ``receipts`` module the suite may hold are restored once it's loaded."""
    path, shadow = list(sys.path), sys.modules.pop("receipts", None)
    sys.path.insert(0, str(DITHER))
    try:
        return __import__(name)
    finally:
        sys.path[:] = path
        sys.modules.pop(name, None)
        sys.modules.pop("receipts", None)
        if shadow is not None:
            sys.modules["receipts"] = shadow


@functools.cache
def served(name: str):
    """`benchmarks/pouw/served_h1/<name>.py`, imported as dither imports it (served_h1 on sys.path); sys.path is restored
    and the served_h1 modules it loaded are dropped from sys.modules."""
    here, path, before = DITHER.parents[1] / "pouw" / "served_h1", list(sys.path), set(sys.modules)
    sys.path.insert(0, str(here))
    try:
        return __import__(name)
    finally:
        sys.path[:] = path
        for m in set(sys.modules) - before:
            if Path(getattr(sys.modules[m], "__file__", None) or "/").parent == here:
                sys.modules.pop(m)


def _units(m: int, n: int) -> list[tuple[int, int]]:
    """A call's units (ri, cj) in dither's order, unit u = ri (n / 32) + cj (`j10r_units.call_receipts`)."""
    return [divmod(u, n // TILE[1]) for u in range((m // TILE[0]) * (n // TILE[1]))]


@pytest.mark.parametrize("m", [64, 2048])
def test_each_units_blocks_are_the_ones_credit_derives_at_the_slot(m):
    """At the decode slot T = 64 (its second row block past the 32 served rows, computed in full) and prefill's 2048, for
    every Llama-3.1-8B linear: unit u's blocks (`j10r_units.blocks`) are `core_check.j10r_blocks` of credit's tile t = u,
    (t // (n/C), t % (n/C)), at slot M = T, and the slot is `h1window.slot`'s for T rows."""
    from verity.ml.protocols.pouw.circuit import h1window as HW
    JU = dither("j10r_units")
    assert HW.slot(m, TILE) == m
    for n, k in LLAMA:
        per = n // TILE[1]
        units = _units(m, n)
        units = units if m == 64 else units[:per] + units[-per:]
        assert all(JU.blocks(m, k, ri, cj) == CC.j10r_blocks(m, k, TILE, t // per, t % per)
                   for ri, cj in units for t in [ri * per + cj]), (n, k)


@pytest.mark.parametrize("t", [64, 2048])
def test_each_calls_first_position_is_credits_u0(t):
    """The arm's call w of step s starts at s U + off(w) (`serve.static_layout`, dither's `u_step` and `offs`): credit's
    u0 for that call, the sum of `credit.call_work`'s sizes before it in the window's layout at M = T (two layers of
    Llama-3.1-8B's linears, three steps)."""
    SV = served("serve")
    names = [f"{x}{layer}" for layer in range(2) for x in ("qkv", "o", "gate_up", "down")]
    nk = LLAMA * 2
    steps = 3
    layout = {"calls": [[nm, s] for s in range(steps) for nm in names], "m": t, "tile": list(TILE),
              "linears": [[nm, k, n] for nm, (n, k) in zip(names, nk)]}
    sizes = [size for size, _work in Cr.call_work(layout)]
    offs, u = SV.static_layout([n for n, _k in nk], t)
    want = [s * u + offs[w] for s in range(steps) for w in range(len(names))]
    assert [sum(sizes[:i]) for i in range(len(sizes))] == want


def test_the_arms_public_rows_are_credits():
    """A call's public rows from the arm's own call (u0) and blocks are credit's `public_table` rows for that call (the
    J10-R row: today's, then u64be(u0), u64be(b0), u64be(b1), 112 words)."""
    JU = dither("j10r_units")
    m, (n, k), u0 = 64, LLAMA[3], 3 * 2 ** 32 + 128
    per = n // TILE[1]
    credit_blocks = [CC.j10r_blocks(m, k, TILE, t // per, t % per) for t in range(len(_units(m, n)))]
    rows, _salts, _reg = Cr.public_table("ab" * 64, "pub@0", SALT.hex(), "cd" * 64, u0, len(credit_blocks),
                                         (RA.hex(), RB.hex()), credit_blocks)
    arm = [CC.public_words(SALT, u0 + u, (RA, RB), (u0, JU.blocks(m, k, ri, cj)))
           for u, (ri, cj) in enumerate(_units(m, n))]
    assert arm == rows and len(rows[0]) == 112


def _receipts_match(m: int, n: int, k: int, only: list[int], seed: int) -> None:
    JU = dither("j10r_units")
    g = np.random.default_rng(seed)
    a = g.integers(-128, 128, size=(m, k), dtype=np.int8)
    a[32:] = 0
    b = g.integers(-128, 128, size=(n, k), dtype=np.int8)
    u0, per = 5 * 2 ** 32 + 96 * seed, n // TILE[1]
    got = JU.call_receipts(torch.from_numpy(a), torch.from_numpy(b), SALT, u0, RA, RB, only=only)
    for i, t in enumerate(only):
        ri, cj = t // per, t % per
        _z, want = HT.j10r_reference(a[32 * ri:32 * ri + 32].view(np.uint8).tolist(),
                                     b[32 * cj:32 * cj + 32].view(np.uint8).tolist(), SALT, u0 + t, (RA, RB), u0,
                                     CC.j10r_blocks(m, k, TILE, ri, cj), pairs=[(0, 0)])
        assert [int(x) for x in got[i]] == want, (k, t)


def test_the_arms_receipts_are_the_definitions_at_credits_call_and_blocks():
    """`j10r_units.call_receipts` (unit t at u0 + t) is `h1tile.j10r_reference`'s receipt seeded at credit's call u0 and
    `core_check.j10r_blocks`, on every unit of a 64-row slot (its second row block zero rows, computed in full) at K = 512."""
    _receipts_match(64, 64, 512, [0, 1, 2, 3], 1)


@pytest.mark.slow
def test_the_arms_receipts_at_llamas_k():
    """As above at K = 4096 and 14336 (m_K = 2 and 3), a unit in each row block: what K = 512's one digit can't show is
    the row blocks' RB past one digit, which places every column block (M RB + C J CB)."""
    _receipts_match(64, 96, 4096, [1, 5], 2)
    _receipts_match(64, 64, 14336, [3], 3)


def test_a_j10r_arm_in_the_order_is_ask71_a_with_3574_as_its_basis():
    """J10-R is an adopted part (`forms.ADOPTED_PARTS`, #3574): a -so -zy arm under it is form A with J10-R's basis after
    the z's, and an arm with an unadopted part stays unadopted."""
    F = dither("forms")
    assert "j10r" not in F.UNADOPTED_PARTS
    lab = F.form_of("dither-a", "zy", True, (), ("j10r",))
    assert lab["form"] == F.FORM_A
    assert lab["form_basis"] == f"{F.Z_FORMS['zy'][1]}; {F.ADOPTED_PARTS['j10r']}"
    assert "credit.SHARED_J10R" in lab["form_basis"]
    assert F.form_of("dither-a", "zy", True, ("ov",), ("j10r",))["form"] == F.UNADOPTED
    assert F.form_of("dither-a", "zy", True) == {"form": F.FORM_A, "form_basis": F.Z_FORMS["zy"][1]}


def test_only_a_j10r_arm_on_credits_layout_takes_3574s_basis():
    """`forms.j10r_part`: the slot rows mode on a slot T (64, 2048), every row block computed, is "j10r"; the live rows
    mode, skipped dead tiles, or a T credit's window can't have as its M (96, 16) is "j10r-off", which leaves the arm
    unadopted."""
    from verity.ml.protocols.pouw.circuit import h1window as HW
    F = dither("forms")
    assert [F.j10r_part("slot", False, t) for t in (64, 2048)] == ["j10r"] * 2
    assert all(HW.slot(t, TILE) == t for t in (64, 2048)) and all(HW.slot(t, TILE) != t for t in (96, 16))
    off = [("live", False, 64), ("slot", True, 64), ("slot", False, 96), ("slot", False, 16)]
    assert [F.j10r_part(*x) for x in off] == ["j10r-off"] * 4
    assert F.form_of("dither-a", "zy", True, ("j10r-off",))["form"] == F.UNADOPTED
