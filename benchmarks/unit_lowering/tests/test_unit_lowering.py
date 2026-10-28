"""run.py's reading of a circuit file (its SHA-512, bytes and each section's AND rows, against `ir_lower`'s own layout of a
small circuit) and the guard at the smallest shape the H1 tile takes."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run
from verity_flock import ir_lower as IL
from verity_flock.gf2 import ONE, Circuit


def small() -> IL.Layout:
    """Two input groups, ANDs, a committed bit (B the constant column alone, like an assertion's), an assertion and one
    output group."""
    C = Circuit()
    a, b = C.input("a", 8), C.input("b", 200)
    x = [C.AND(a[i], b[i] ^ b[i + 8]) for i in range(8)]
    (y,) = C.commit([x[0] ^ x[1] ^ ONE])
    z = C.AND(y, x[2] ^ a[3])
    C.assert_zero("t", z ^ x[4] ^ x[5])
    C.output("o", [*x, y, z])
    return IL.layout(C, ["a", "b"], ["o"], in_split=[1])


def test_scan_counts_the_and_rows_ir_lower_lays_out(tmp_path):
    lay = small()
    text = IL.netlist(lay, "unit")
    path = tmp_path / "small.circuit.txt"
    path.write_bytes(b"flock-circuit small\nMETA {}\n" + b"CIRCUIT unit %d\n" % text.count("\n") + text.encode()
                     + b"CIRCUIT again %d\n" % text.count("\n") + text.encode())
    got = run.scan(path)
    data = path.read_bytes()
    assert (got["sha512"], got["bytes"], got["format"]) == (hashlib.sha512(data).hexdigest(), len(data), "flock-circuit")
    want = {"rows": lay.useful, "ands": lay.counts["ands"], "asserts": lay.counts["asserts"]}
    assert want == {"rows": lay.useful, "ands": 10, "asserts": 1}
    assert got["ands"] == 10 and got["sections"] == {"unit": want, "again": want}


def test_scan_refuses_a_section_whose_line_count_is_not_its_netlist(tmp_path):
    text = IL.netlist(small(), "unit")
    path = tmp_path / "short.circuit.txt"
    path.write_bytes(b"flock-circuit short\nMETA {}\n" + b"CIRCUIT unit %d\n" % (text.count("\n") - 1) + text.encode())
    with pytest.raises(ValueError, match="holds a netlist of"):
        run.scan(path)


@pytest.mark.slow
def test_the_guard_passes_at_the_smallest_shape(tmp_path):
    """The flat file at K = 64 on the 4 x 8 tile, the smallest the H1 tile takes (32 | K, K >= 64; 4 | R, 8 | C), is
    byte for byte reference.json's (d40fe21d26's lowering). About 2 min and 5 GiB on that lowering: nothing cheaper catches a
    change to the flat file's bytes, since the unit's netlist, sha512x3's, hm96's and META all come out of the whole
    lowering."""
    g = run.guard(tmp_path, shapes=[(64, 4, 8)])
    (shape,) = g["shapes"]
    assert g["guard"] == "pass", f"sha512 {shape['sha512']}, reference {shape['reference']}"
    assert shape["ands"] == 2_384_742 and not (tmp_path / "k64-4x8-flat" / "k64.circuit.txt").exists()
