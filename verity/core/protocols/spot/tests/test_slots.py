"""`one_stage.slots` (`FilePart.ofCircuit`'s port) on a small circuit file: Δ's source keys, the sources a unit owns,
and the layout's positions.
`verity/core/service/spot/tests/test_difftest.py` holds it to `Slots.layout` on a deployed statement."""
from __future__ import annotations

import json
from pathlib import Path

from verity.core.protocols.spot.one_stage import slots as SL


def _row(a: list[int], b: list[int]) -> str:
    return " ".join(map(str, [len(a), *a, len(b), *b]))


def _circuit(d: Path) -> Path:
    """A unit of two input words, two copies: leaf ports `x0` (column 0) and `x1` (column 1), and port `y` (column 128) past
    the leaf group, which copy 0's leaf cut fills from message word 1 and copy 1's wire from the stage. Its computed rows
    read columns 0, 1 and 128. Copy 0's `x1` is the zero; both copies' `x0` is word 0's low bit; copy 1's `x1` and copy 0's
    `y` are word 1's."""
    n = 2 * 128
    unit = [f"flock-ir-unit/v2 unit {n + 4} {n + 3} 2 0 1 2 1 1 1 1 1 1 1 2 1 1 1"]
    unit += [_row([i], [i]) for i in range(n)]
    unit += [_row([0], [1]), _row([128], [128]), _row([n], [n + 1]), _row([n + 3], [n + 3])]
    stage = ["flock-ir-unit/v2 stage 3 2 1 0 1 1 1 1 1 1 1 1", _row([0], [0]), _row([0], [0]), _row([2], [2])]
    meta = {"units_per_vu": 2, "ports": [["x", 2, 1]], "leaves_in": [[0, -1], [0, 1]], "leaf_cuts": [["unit", 0, 2, 1]],
            "wires": [["unit", 1, 2, "stage", 0, 0]]}
    circuit = d / "circuit.txt"
    circuit.write_text("\n".join(["flock-circuit/v1 t", f"META {json.dumps(meta)}", f"CIRCUIT unit {len(unit)}", *unit,
                                  f"CIRCUIT stage {len(stage)}", *stage]) + "\n")
    return circuit


def test_a_unit_owns_the_first_input_of_each_key_a_row_reads(tmp_path):
    meta, nets = SL.read_circuit(_circuit(tmp_path))
    keys = SL.src_keys(meta, nets)
    assert len(keys) == 2 * 256
    assert keys[0] == keys[256] == ("msg", 0, 0, 0)
    assert keys[1] == ("zero",)
    assert keys[257] == keys[128] == ("msg", 0, 2, 0)
    assert keys[384] == ("wire", "stage", 0, 128)
    assert keys[2] == ("pad", 0, 2)
    f = SL.of_circuit(_circuit(tmp_path), 3)
    assert f == SL.FilePart(3, 512, 2 * (259 - 256 + 1), (0, 128, 384))
    assert (f.in_gates, f.size) == (2 + 3 * 512, 2 + 3 * 512 + 3 * 3 * 8)
    assert f.block(1) == [(f.in_gates + 24, f.in_gates + 48), (514, 515), (642, 643), (898, 899)]


def test_the_layout_is_the_summary_unit_by_unit_in_the_strata():
    f = SL.FilePart(3, 2, 1, (1,))
    (pop,) = SL.layout(f, SL.sizes([("a", [(0, 2)]), ("b", [(2, 3)])]))
    assert pop == [[{"pus": [[8, 11], [3, 4]], "unit": 0}, {"pus": [[11, 14], [5, 6]], "unit": 1}],
                   [{"pus": [[14, 17], [7, 8]], "unit": 2}]]
    assert SL.checked([pop], [2, 0]) == [[0, 0], [2, 0]]


def test_strata_that_are_not_the_units_in_order_have_no_sizes():
    assert SL.sizes([("a", [(0, 1), (2, 3)]), ("b", [(1, 2)])]) is None
    assert SL.sizes([("a", [(1, 2)])]) is None
    assert SL.sizes([]) == []


def test_last_le_is_the_last_entry_at_most_x():
    assert [SL.last_le([0, 2, 2, 5], 2), SL.last_le([0, 2, 5], 7), SL.last_le([0, 2, 5], 0)] == [2, 2, 0]
