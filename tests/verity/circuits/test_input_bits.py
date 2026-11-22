"""`program.input_bits`, a declared circuit's input bits by binding: on the cases Lean's port is held to, and on the rows
`Flock.Program.inputBits`' `#guard`s pin."""

from __future__ import annotations

import pytest

import input_bits_cases as C

from verity.circuits.program import REGISTRATION, RUN, SUPPLIED, input_bits, input_bits_of


@pytest.mark.parametrize("name", list(C.CASES))
def test_each_case_gives_its_bits_or_its_refusal(name):
    make, rebind, want = C.CASES[name]
    got = C.python(C.instance(name))
    assert got == ({"ok": True, **want} if isinstance(want, dict) else {"ok": False, "error": want})
    if rebind is None:
        assert input_bits(make().root) == want


def test_the_guards_rows():
    rows = [("Input8_v1", RUN, "primitive", 9), ("RegisteredInput16_v1", REGISTRATION, "batch", 3),
            ("SuppliedInput4_v1", SUPPLIED, "primitive", 1), ("Input2_v1", RUN, "batch", 5), ("Step_v1", None, "scan", 7)]
    assert input_bits_of(rows) == {"run": 18, "registration": 48, "supplied": 4}
    assert input_bits_of([]) == {"run": 0, "registration": 0, "supplied": 0}


@pytest.mark.parametrize(("row", "code"), [
    (("Input8_v1", "later", "primitive", 1), "input-binding"),
    (("Input8_v1", REGISTRATION, "primitive", 1), "input-binding"),
    (("Input8_v1", None, "batch", 2), "input-binding"),
    (("Step_v1", RUN, "primitive", 1), "input-width"),
])
def test_the_guards_refusals(row, code):
    with pytest.raises(ValueError, match=f"^{code}: root node 0: "):
        input_bits_of([row])
