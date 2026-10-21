"""``verity_flock.tmpl``: the templated form of a flat ``flock-ir-unit/v2`` unit unrolls to that unit's text, line for line,
on a dot product's unit as ``ir_lower.layout`` lays it out; its Definitions repeat (a multiply-accumulate step, its bit
cells), so its text is a fraction of the flat text's; and the Lean fixture of ``Flock/Test/Tmpl.lean`` unrolls to the
flat text written out there."""
from __future__ import annotations

import numpy as np

from verity.ml.circuits.boolean import forms as F
from verity_flock import ir_lower as IL
from verity_flock import tmpl as T


def _dot(n: int) -> list[str]:
    C = F.Circuit()
    a, b = C.input("a", 8 * n), C.input("b", 8 * n)
    acc = [F.ZERO] * 20
    for i in range(n):
        acc = C.commit(F.add(C, acc, F.multiply(C, a[8 * i:8 * i + 8], b[8 * i:8 * i + 8]), width=20))
    C.output("acc", acc)
    lay = IL.layout(C, ["a", "b"], ["acc"])
    return IL.netlist(lay, "dot", leaves={"a": [1]}).splitlines()


def test_a_dot_products_template_unrolls_to_its_flat_unit_and_does_not_grow_with_it():
    sizes = []
    for n in (48, 192):
        lines = _dot(n)
        text = T.write(T.emit(T.parse_flat(lines))[0])
        assert T.flat_lines(T.unroll(T.parse(text))) == lines
        sizes.append((sum(map(len, lines)), len(text)))
    (flat_s, t_s), (flat_l, t_l) = sizes
    assert flat_l > 3 * flat_s and t_l < 1.1 * t_s and t_l * 10 < flat_l, sizes


def test_the_aligned_template_unrolls_to_the_padded_unit_with_every_run_at_an_aligned_power_of_two():
    flat = T.parse_flat(_dot(48))
    t, _ = T.emit(flat)
    t2, f2, st = T.align(t, flat)
    got = T.unroll(T.parse(T.write(t2)))
    assert T.flat_lines(got) == T.flat_lines(f2)
    runs = [c for d in t2.defs for c in d.calls if c.count > 1]
    assert runs and all(c.stride & (c.stride - 1) == 0 and c.base % c.stride == 0 for c in runs), st
    # the same rows, each moved by an increasing map: input rows in place, the constant last
    n, n2 = flat.useful, f2.useful
    assert n2 >= n and f2.const == n2 - 1 and np.array_equal(np.diff(f2.ptr)[np.diff(f2.ptr) > 0], np.diff(flat.ptr)[np.diff(flat.ptr) > 0])
    assert [f2.row(i) for i in range(48 * 16)] == [flat.row(i) for i in range(48 * 16)]


def test_the_lean_fixture_unrolls_to_its_flat_text():
    tmpl = "\n".join(["flock-ir-tmpl/v1 t 2 1 0 1 1 3 1 2 1 1 2", "DEF 1 2 1 0", "0 1 2 1 1", "DEF 3 2 1 1", "2 2 1 0 1 5",
                      "CALL 0 2 0 1 1 2 3 0", "DEF 400 0 6 1", "0 1 0 1 0", "1 1 1 1 1", "2 1 2 1 2", "256 1 130 1 399",
                      "257 1 133 1 399", "300 3 0 1 0 1 400", "CALL 1 2 128 3 1 2 0 1"])
    rows = {0: "1 0 1 0", 1: "1 1 1 1", 2: "1 2 1 2", 128: "1 1 1 0", 129: "1 1 1 0", 130: "2 128 129 1 399",
            131: "1 2 1 1", 132: "1 2 1 1", 133: "2 131 132 1 399", 256: "1 130 1 399", 257: "1 133 1 399",
            300: "1 1 1 399", 399: "1 399 1 399"}
    flat = ["flock-ir-unit/v2 t 400 399 1 0 1 1 3 1 2 1 1 2"] + [rows.get(i, "0 0") for i in range(400)]
    assert T.flat_lines(T.unroll(T.parse(tmpl))) == flat
