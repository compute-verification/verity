"""The tile rule's inert-work cut in Lean, run, against its Python forms on zero products.

`pouw-pearlplus-cut-difftest` (`PouwPearlPlus/CutDifftest.lean`) serves `Rule.cuts` (one atom step's cut flags from an
accumulator word, `Rule.cutFold` on the H100's atom, which must equal `Pouw.PearlC.Mend.cutFold`), `Rule.wordCuts` (a word's
flags along its chain, from +0 at each promotion group's first atom) and `Rule.wordMend`. A product of zero magnitude is
cut: `aligned` is 0 at zero magnitude, whatever the group's exponent.

Each case here is held to two Python forms:

* the batch forms `served_inert_work.py` and `verify_run.py --fast` replay with (`fast_tile._cut` per atom, `_mends` per word);
* the scalar reference (`pearl_c_inert_work.cut_flags`, `chain_cuts`, `word_mend`).

The reference rebuilds a zero product at its nominal exponent (ea + eb − 14) and shifts its zero significand right by
top − e, which is negative when nothing in its group (no nonzero product and no incoming value) lifts top above that
exponent: it raises `ValueError: negative shift count`. The kernels lane holds that mend (`cut_flags`' shift becoming
`not mag or (scaled >> (top - e)) == 0`), so the cases that raise today are `xfail(strict=True, raises=ValueError)`
under `REFERENCE_RAISES`: a wrong answer still fails, and the mend makes them XPASS, which strict turns red until the
mend's PR deletes the mark. Every other assertion here holds today.
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
from verity.protocols.resource_accounting.compute.pouw.prover.schemes import pearl_c_inert_work as IW

ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "verity" / "protocols" / "resource_accounting" / "compute" / "pouw" / "verifier" / "lean" / "pearlplus"
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

#: the cases on which `pearl_c_inert_work`'s reference raises `negative shift count` before kernels' mend
REFERENCE_RAISES = {
    "all_zero_at_group_start", "zero_against_max_at_start", "zero_against_max_in_middle", "both_operands_zero_all",
    "negative_zero_against_max", "negative_zeros_all", "all_zero_after_subnormal", "all_zero_after_negative_zero",
    "zero_atoms_from_group_start", "zero_atom_at_second_group_start", "negative_zero_at_start",
}
WAITS = ("kernels' mend of pearl_c_inert_work.cut_flags (a zero product is cut: `not mag or (scaled >> (top - e)) == 0`); "
         "strict, so the mend turns this red and its PR deletes the mark")


def _reference_cases(names) -> list:
    return [pytest.param(n, marks=pytest.mark.xfail(strict=True, raises=ValueError, reason=WAITS))
            if n in REFERENCE_RAISES else n for n in names]


def _batch_cut(acc: int, a: list[int], b: list[int]) -> list[bool]:
    return [bool(x) for x in FT._cut(np.array([acc], np.uint32), np.array([a]), np.array([b]), H100)[0]]


def _ends(a: list[int], minus: list[int] | None = None) -> dict:
    return {"plus": a, "minus": a if minus is None else minus, "dead": [False] * len(a)}


def _mend_input(name: str) -> dict:
    a, b = WORDS[name]
    return {"a": _ends(a), "b": _ends(b)}


def _py_ends(e: dict) -> tuple:
    return e["plus"], e["minus"], e["dead"]


CUTS = D.Port(EXE, "cuts", lambda v: _batch_cut(v["acc"], v["a"], v["b"]), st.nothing(), package=PACKAGE,
              need_rejects=False)
server = D.server_fixture(CUTS)


def _lean(server: D.Server, fn: str, value) -> object:
    kind, text = server.call(fn, value)
    assert kind == "output", (fn, text)
    return json.loads(text)


@pytest.mark.parametrize("name", list(ATOMS))
def test_lean_cuts_are_fast_tiles(server, name):
    acc, a, b, pinned = ATOMS[name]
    lean = _lean(server, "cuts", {"acc": acc, "a": a, "b": b})
    assert lean == _batch_cut(acc, a, b)
    assert all(lean[i] for i in range(32) if not (a[i] & 0x7F and b[i] & 0x7F)), "a zero product is cut"
    if pinned is not None:
        assert sum(lean) == pinned


@pytest.mark.parametrize("name", _reference_cases(ATOMS))
def test_lean_cuts_are_the_references(server, name):
    acc, a, b, _ = ATOMS[name]
    assert _lean(server, "cuts", {"acc": acc, "a": a, "b": b}) == IW.cut_flags(acc, a, b, H100)


@pytest.mark.parametrize("name", list(WORDS))
def test_lean_word_mend_is_fast_tiles(server, name):
    """`_mends` on one word whose rows' two ends are equal and nothing salt-dead: the products cut on its chain."""
    v = _mend_input(name)
    lean = _lean(server, "word_mend", v)
    assert lean == FT._mends([_py_ends(v["a"])], [_py_ends(v["b"])], H100)
    assert lean == sum(_lean(server, "word_cuts", {"a": v["a"]["plus"], "b": v["b"]["plus"]}))


def test_lean_word_mend_counts_cuts_at_both_ends(server):
    """A product zero at the plus end alone is no mend; one zero at both ends is."""
    a, b = WORDS["zero_atom_inside_a_group"]
    minus = list(a)
    minus[32:48] = [ONE] * 16
    v = {"a": _ends(a, minus), "b": _ends(b)}
    lean = _lean(server, "word_mend", v)
    assert lean == FT._mends([_py_ends(v["a"])], [_py_ends(v["b"])], H100) == 16


@pytest.mark.parametrize("name", _reference_cases(WORDS))
def test_lean_word_cuts_are_the_references(server, name):
    a, b = WORDS[name]
    assert _lean(server, "word_cuts", {"a": a, "b": b}) == IW.chain_cuts(a, b, H100)
    v = _mend_input(name)
    assert _lean(server, "word_mend", v) == IW.word_mend(_py_ends(v["a"]), _py_ends(v["b"]), H100)


def test_reference_raises_only_where_listed():
    """`REFERENCE_RAISES` is exactly where the reference raises today: the list the xfail marks read stays honest."""
    def raises(f) -> bool:
        try:
            f()
        except ValueError as e:
            assert "negative shift count" in str(e)
            return True
        return False
    got = {n for n, (acc, a, b, _) in ATOMS.items() if raises(lambda: IW.cut_flags(acc, a, b, H100))}
    got |= {n for n, (a, b) in WORDS.items() if raises(lambda: IW.chain_cuts(a, b, H100))}
    assert got == REFERENCE_RAISES or not got, sorted(got ^ REFERENCE_RAISES)


_CODE = st.sampled_from([0x00, NEG_ZERO, 0x01, MIN_NORMAL, 0x30, ONE, 0x40, MAX, 0xB8, 0xFE])
_ACC = st.sampled_from([0, 0x80000000, F32_MIN_SUB, 0x00800000, F32_ONE, 0xBF800000, F32_2_24, 0x3A800000])
_ATOM = st.lists(_CODE, min_size=32, max_size=32)


def test_lean_cuts_are_fast_tiles_on_zero_rich_atoms(server):
    """Random atoms over zeros of both signs, subnormal, normal and saturated codes, from accumulators of each class:
    `_cut` and Lean agree on every one (300 examples, seed 0)."""
    port = D.Port(EXE, "cuts", CUTS.reference, st.fixed_dictionaries({"acc": _ACC, "a": _ATOM, "b": _ATOM}),
                  package=PACKAGE, need_rejects=False)
    D.run(port, examples=300, seed=0, server=server)
