"""``verity_flock.tmpl``: the templated form of a flat ``flock-ir-unit/v2`` unit unrolls to that unit's text, line for line,
on a dot product's unit as ``ir_lower.layout`` lays it out; its Definitions repeat (a multiply-accumulate step, its bit
cells), so its text is a fraction of the flat text's; and the Lean fixture of ``Flock/Test/Tmpl.lean`` unrolls to the
flat text written out there. Both are in the row order ``Tmpl.ordered`` checks, and ``legalize`` splits a segment the
check refuses only for straddling two input groups into ones it accepts, unrolling to the same rows."""
from __future__ import annotations

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
        t, stats = T.emit(T.parse_flat(lines))
        text = T.write(t)
        assert T.flat_lines(T.unroll(T.parse(text))) == lines
        assert stats["order"]["ok"] and T.ordered(T.parse(text))["ok"], stats["order"]
        sizes.append((sum(map(len, lines)), len(text)))
    (flat_s, t_s), (flat_l, t_l) = sizes
    assert flat_l > 3 * flat_s and t_l < 1.1 * t_s and t_l * 10 < flat_l, sizes


def test_the_lean_fixture_unrolls_to_its_flat_text():
    tmpl = "\n".join(["flock-ir-tmpl/v1 t 2 1 0 1 1 3 1 2 1 1 2", "DEF 1 2 1 0", "0 1 2 1 1", "DEF 3 2 1 1", "2 2 1 0 1 5",
                      "CALL 0 2 0 1 1 2 3 0", "DEF 400 0 6 1", "0 1 0 1 0", "1 1 1 1 1", "2 1 2 1 2", "256 1 130 1 399",
                      "257 1 133 1 399", "300 3 0 1 0 1 400", "CALL 1 2 128 3 1 2 0 1"])
    rows = {0: "1 0 1 0", 1: "1 1 1 1", 2: "1 2 1 2", 128: "1 1 1 0", 129: "1 1 1 0", 130: "2 128 129 1 399",
            131: "1 2 1 1", 132: "1 2 1 1", 133: "2 131 132 1 399", 256: "1 130 1 399", 257: "1 133 1 399",
            300: "1 1 1 399", 399: "1 399 1 399"}
    flat = ["flock-ir-unit/v2 t 400 399 1 0 1 1 3 1 2 1 1 2"] + [rows.get(i, "0 0") for i in range(400)]
    assert T.flat_lines(T.unroll(T.parse(tmpl))) == flat
    assert T.ordered(T.parse(tmpl))["ok"]
    assert not T.ordered(T.parse(tmpl.replace("CALL 0 2 0 1 1 2 3 0", "CALL 0 2 0 1 1 2 2 0")))["ok"]


def test_legalize_splits_a_root_segment_across_two_groups_port_bits_and_unrolls_the_same():
    tmpl = T.parse("\n".join(["flock-ir-tmpl/v1 s 2 2 0 1 1 128 1 1 1 128 1 2 1 1 2", "DEF 1 0 1 0", "0 1 0 1 0",
                              "DEF 1 2 1 0", "0 1 2 1 1", "DEF 400 0 0 2", "CALL 0 256 0 1 0", "CALL 1 2 256 1 1 2 127 1"]))
    before = T.ordered(tmpl)
    assert not before["ok"] and before["padding"] == 1, before
    legal, st = T.legalize(tmpl)
    assert st == {"split_segs": 1, "single_calls": 2}, st
    assert T.ordered(legal)["ok"] and T.ordered(T.parse(T.write(legal)))["ok"]
    assert T.flat_lines(T.unroll(legal)) == T.flat_lines(T.unroll(tmpl))
    assert T.flat_lines(T.unroll(legal))[1 + 256:1 + 258] == ["1 128 1 127", "1 129 1 128"]
