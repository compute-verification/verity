"""`word_shape_scan.py`: the shape's smallest case is found, a word Definition reaching a word primitive is not, and
`one_linear`'s circuit has no Definition of the shape."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "one_stage"))

import one_linear as OL  # noqa: E402
import word_shape_scan as WS  # noqa: E402


def test_the_smallest_case_is_found_and_a_word_definition_is_not():
    WS.selftest()


def test_one_linears_circuit_has_no_definition_of_the_shape():
    program, _query = OL.circuit(registered=False).program()
    out = WS.scan_program(program, "one_linear")
    assert out["shape"] == 0 and out["composites_reached"] >= 1, out
