"""`one_stage.slots` on small circuit files: Δ's source keys, the sources an instance owns, and the layout's positions.
`verity/core/service/sampled_proofs/tests/test_difftest.py` holds it to `Slots.layout` on a deployed statement."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity.core.protocols.sampled_proofs.one_stage import slots as SL


def _row(a: list[int], b: list[int]) -> str:
    return " ".join(map(str, [len(a), *a, len(b), *b]))


def _files(d: Path, indices: list[int], name: str = "") -> tuple[Path, Path]:
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
    circuit, public = d / f"circuit{name}.txt", d / f"pub{name}.bin"
    circuit.write_text("\n".join(["flock-circuit/v1 t", f"META {json.dumps(meta)}", f"CIRCUIT unit {len(unit)}", *unit,
                                  f"CIRCUIT stage {len(stage)}", *stage]) + "\n")
    public.write_bytes(json.dumps({"instances": len(indices), "units": {"indices": indices}}).encode() + b"\nrest")
    return circuit, public


def test_an_instance_owns_the_first_input_of_each_key_a_row_reads(tmp_path):
    meta, nets = SL.read_circuit(_files(tmp_path, [0])[0])
    keys = SL.src_keys(meta, nets)
    assert len(keys) == 2 * 256
    assert keys[0] == keys[256] == ("msg", 0, 0, 0)
    assert keys[1] == ("zero",)
    assert keys[257] == keys[128] == ("msg", 0, 2, 0)
    assert keys[384] == ("wire", "stage", 0, 128)
    assert keys[2] == ("pad", 0, 2)
    f = SL.file_units(*_files(tmp_path, [4, 5, 6]))
    assert f == SL.FileUnits((4, 5, 6), 512, 2 * (259 - 256 + 1), (0, 128, 384))
    assert (f.inputs, f.size) == (2 + 3 * 512, 2 + 3 * 512 + 3 * 3 * 8)
    assert f.block(1) == [(f.inputs + 24, f.inputs + 48), (514, 515), (642, 643), (898, 899)]


def test_the_layout_lays_the_files_end_to_end_in_the_order_of_their_first_unit(tmp_path):
    lo, hi = _files(tmp_path, [0, 1], "lo"), _files(tmp_path, [2], "hi")
    (pop,) = SL.layout([hi, lo])
    f = SL.file_units(*lo)
    assert [[b["unit"] for b in s] for s in pop] == [[0, 1], [2]]
    assert pop[1][0]["pus"][0] == [f.size + 2 + 512 + 0, f.size + 2 + 512 + 24]
    assert SL.checked([pop], [2, 0]) == [[0, 0], [2, 0]]


def test_a_public_file_that_does_not_name_its_instances_units_is_refused(tmp_path):
    circuit, public = _files(tmp_path, [0, 1])
    public.write_bytes(json.dumps({"instances": 3, "units": {"indices": [0, 1]}}).encode() + b"\n")
    with pytest.raises(ValueError, match="names 2 units for its 3 instances"):
        SL.file_units(circuit, public)
