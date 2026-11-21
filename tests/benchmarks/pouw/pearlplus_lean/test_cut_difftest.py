"""The tile rule's inert-work cut in Lean, run, against its Python forms on zero products.

`pouw-pearlplus-cut-difftest` (`PouwPearlPlus/CutDifftest.lean`) serves `Rule.cuts` (one atom step's cut flags from an
accumulator word, `Rule.cutFold` on the H100's atom, which must equal `Pouw.PearlC.Mend.cutFold`), `Rule.wordCuts` (a word's
flags along its chain, from +0 at each promotion group's first atom) and `Rule.wordMend`. A product of zero magnitude is
cut: `aligned` is 0 at zero magnitude, whatever the group's exponent.

Each case here is held to both Python forms:

* the batch forms `served_inert_work.py` and `verify_run.py --fast` replay with (`fast_tile._cut` per atom, `_mends` per word);
* the scalar reference (`pearl_c_inert_work.cut_flags`, `chain_cuts`, `word_mend`).

A zero product's nominal exponent (ea + eb − 14) may lie above its group's exponent when nothing in the group (no nonzero
product and no incoming value) lifts the group above it, so a form that shifts the zero significand by top − e meets a
negative count there: the cases at a group's start, of a zero against 448, and of an all-zero atom after a subnormal or
−0 accumulator are those.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.protocols.accounting.compute.pouw.prover.schemes import pearl_c_inert_work as IW

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "verity" / "protocols" / "accounting" / "compute" / "pouw" / "verifier" / "lean" / "pearlplus"
EXE = "pouw-pearlplus-cut-difftest"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


def _module(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[name])
    return sys.modules[name]


FT = _module("pearl_c_vllm_fast_tile", ROOT / "benchmarks" / "pouw" / "pearl_c_vllm" / "fast_tile.py")
H100 = IW.H100
ONE, MIN_NORMAL, MAX, NEG_ZERO = 0x38, 0x08, 0x7E, 0x80
F32_ONE, F32_2_24, F32_MIN_SUB = 0x3F800000, 0x4B800000, 0x00000001


def _put(base: int, at: dict[int, int]) -> list[int]:
    row = [base] * 32
    for i, c in at.items():
        row[i] = c
    return row


#: one atom step: name -> (accumulator word, a codes, b codes, Lean's pinned cut count where the brief states it)
ATOMS = {
    "all_zero_at_group_start": (0, [0] * 32, [ONE] * 32, 32),
    "zero_against_max_at_start": (0, _put(MIN_NORMAL, {0: 0}), _put(MIN_NORMAL, {0: MAX}), 1),
    "zero_against_one": (0, _put(ONE, {0: 0}), [ONE] * 32, 1),
    "zero_against_max_in_middle": (0, _put(MIN_NORMAL, {17: 0}), _put(MIN_NORMAL, {17: MAX}), None),
    "zero_in_middle_against_one": (0, _put(ONE, {17: 0}), [ONE] * 32, None),
    "both_operands_zero_one": (0, _put(MIN_NORMAL, {5: 0}), _put(MIN_NORMAL, {5: 0}), None),
    "both_operands_zero_all": (0, [0] * 32, [0] * 32, None),
    "negative_zero_against_max": (0, _put(MIN_NORMAL, {0: NEG_ZERO}), _put(MIN_NORMAL, {0: MAX}), None),
    "negative_zero_against_one": (0, _put(ONE, {0: NEG_ZERO}), [ONE] * 32, None),
    "negative_zeros_all": (0, [NEG_ZERO] * 32, [NEG_ZERO] * 32, None),
    "all_zero_after_one": (F32_ONE, [0] * 32, [ONE] * 32, None),
    "all_zero_after_subnormal": (F32_MIN_SUB, [0] * 32, [ONE] * 32, None),
    "all_zero_after_negative_zero": (0x80000000, [0] * 32, [ONE] * 32, None),
    "nonzero_shifted_out": (F32_2_24, [MIN_NORMAL] * 32, [MIN_NORMAL] * 32, None),
    "nothing_cut": (0, [ONE] * 32, [ONE] * 32, None),
}


def _word(atoms: list[tuple[list[int], list[int]]]) -> tuple[list[int], list[int]]:
    return [c for a, _ in atoms for c in a], [c for _, b in atoms for c in b]


_ONES = ([ONE] * 32, [ONE] * 32)
_ZERO = ([0] * 32, [ONE] * 32)
#: a word's rows, k = 32 · atoms, G = 4 atoms a promotion group
WORDS = {
    "zero_atom_inside_a_group": _word([_ONES, _ZERO, _ONES, _ONES]),
    "zero_atoms_from_group_start": _word([_ZERO, _ZERO, _ONES, _ONES]),
    "zero_atom_at_second_group_start": _word([_ONES] * 4 + [_ZERO, _ONES, _ONES, _ONES]),
    "zero_against_max_inside_a_group": _word([_ONES, (_put(MIN_NORMAL, {3: 0}), _put(MIN_NORMAL, {3: MAX})), _ONES, _ONES]),
    "zero_against_max_at_start": _word([(_put(MIN_NORMAL, {0: 0}), _put(MIN_NORMAL, {0: MAX})), _ONES, _ONES, _ONES]),
    "negative_zero_at_start": _word([(_put(MIN_NORMAL, {9: NEG_ZERO}), _put(MIN_NORMAL, {9: MAX})), _ONES, _ONES, _ONES]),
    "no_zero": _word([_ONES] * 4),
}

def _batch_cut(acc: int, a: list[int], b: list[int]) -> list[bool]:
    return [bool(x) for x in FT._cut(np.array([acc], np.uint32), np.array([a]), np.array([b]), H100)[0]]


def _both_cuts(v: dict) -> object:
    """`_cut`'s flags where `cut_flags` gives the same, else both, which no Lean output equals."""
    batch, ref = _batch_cut(v["acc"], v["a"], v["b"]), IW.cut_flags(v["acc"], v["a"], v["b"], H100)
    return batch if batch == ref else {"_cut": batch, "cut_flags": ref}


def _ends(a: list[int], minus: list[int] | None = None) -> dict:
    return {"plus": a, "minus": a if minus is None else minus, "dead": [False] * len(a)}


def _mend_input(name: str) -> dict:
    a, b = WORDS[name]
    return {"a": _ends(a), "b": _ends(b)}


def _py_ends(e: dict) -> tuple:
    return e["plus"], e["minus"], e["dead"]


CUTS = D.Port(EXE, "cuts", _both_cuts, st.nothing(), package=PACKAGE, need_rejects=False)
server = D.server_fixture(CUTS)


def _lean(server: D.Server, fn: str, value) -> object:
    kind, text = server.call(fn, value)
    assert kind == "output", (fn, text)
    return json.loads(text)


@pytest.mark.parametrize("name", list(ATOMS))
def test_lean_cuts_are_fast_tiles_and_the_references(server, name):
    acc, a, b, pinned = ATOMS[name]
    lean = _lean(server, "cuts", {"acc": acc, "a": a, "b": b})
    assert lean == _batch_cut(acc, a, b)
    assert lean == IW.cut_flags(acc, a, b, H100)
    assert all(lean[i] for i in range(32) if not (a[i] & 0x7F and b[i] & 0x7F)), "a zero product is cut"
    if pinned is not None:
        assert sum(lean) == pinned


@pytest.mark.parametrize("name", list(WORDS))
def test_lean_word_cuts_and_mend_are_fast_tiles_and_the_references(server, name):
    """`_mends` and `word_mend` on one word whose rows' two ends are equal and nothing salt-dead: the products cut on its
    chain, which are `chain_cuts`' flags."""
    v = _mend_input(name)
    a, b = v["a"]["plus"], v["b"]["plus"]
    lean_cuts, lean = _lean(server, "word_cuts", {"a": a, "b": b}), _lean(server, "word_mend", v)
    assert lean_cuts == IW.chain_cuts(a, b, H100)
    assert lean == sum(lean_cuts)
    assert lean == FT._mends([_py_ends(v["a"])], [_py_ends(v["b"])], H100)
    assert lean == IW.word_mend(_py_ends(v["a"]), _py_ends(v["b"]), H100)


def test_lean_word_mend_counts_cuts_at_both_ends(server):
    """A product zero at the plus end alone is no mend; one zero at both ends is."""
    a, b = WORDS["zero_atom_inside_a_group"]
    minus = list(a)
    minus[32:48] = [ONE] * 16
    v = {"a": _ends(a, minus), "b": _ends(b)}
    lean = _lean(server, "word_mend", v)
    assert lean == FT._mends([_py_ends(v["a"])], [_py_ends(v["b"])], H100) == 16
    assert lean == IW.word_mend(_py_ends(v["a"]), _py_ends(v["b"]), H100)


_CODE = st.sampled_from([0x00, NEG_ZERO, 0x01, MIN_NORMAL, 0x30, ONE, 0x40, MAX, 0xB8, 0xFE])
_ACC = st.sampled_from([0, 0x80000000, F32_MIN_SUB, 0x00800000, F32_ONE, 0xBF800000, F32_2_24, 0x3A800000])
_ATOM = st.lists(_CODE, min_size=32, max_size=32)


def test_lean_cuts_are_fast_tiles_and_the_references_on_zero_rich_atoms(server):
    """Random atoms over zeros of both signs, subnormal, normal and saturated codes, from finite accumulators of each
    class: Lean, `_cut` and `cut_flags` agree on every one (300 examples, seed 0)."""
    port = D.Port(EXE, "cuts", CUTS.reference, st.fixed_dictionaries({"acc": _ACC, "a": _ATOM, "b": _ATOM}),
                  package=PACKAGE, need_rejects=False)
    D.run(port, examples=300, seed=0, server=server)
