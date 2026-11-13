"""``verity_flock.tmpl``: the templated form of a flat ``flock-ir-unit/v2`` unit unrolls to that unit's text, line for line,
on a dot product's unit as ``ir_lower.layout`` lays it out; its Definitions repeat (a multiply-accumulate step, its bit
cells), so its text is a fraction of the flat text's; and the Lean fixture of ``Flock/Test/Tmpl.lean`` unrolls to the
flat text written out there. Both are in the row order ``Tmpl.ordered`` checks, and ``legalize`` splits a segment the
check refuses only for straddling two input groups into ones it accepts, unrolling to the same rows. ``evaluate`` gives the
unrolled template's values or refuses, on a one-copy call whose stride is under its callee's columns and on random
templates."""
from __future__ import annotations

import numpy as np
import pytest

from verity.circuits.boolean import forms as F
from verity_flock import ir_lower as IL
from verity_flock import struct_lower as S
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
    tmpl = _lean_fixture()
    rows = {0: "1 0 1 0", 1: "1 1 1 1", 2: "1 2 1 2", 128: "1 1 1 0", 129: "1 1 1 0", 130: "2 128 129 1 399",
            131: "1 2 1 1", 132: "1 2 1 1", 133: "2 131 132 1 399", 256: "1 130 1 399", 257: "1 133 1 399",
            300: "1 1 1 399", 399: "1 399 1 399"}
    flat = ["flock-ir-unit/v2 t 400 399 1 0 1 1 3 1 2 1 1 2"] + [rows.get(i, "0 0") for i in range(400)]
    assert T.flat_lines(T.unroll(T.parse(tmpl))) == flat
    assert T.ordered(T.parse(tmpl))["ok"]
    assert not T.ordered(T.parse(tmpl.replace("CALL 0 2 0 1 1 2 3 0 1 0", "CALL 0 2 0 1 1 2 2 0 1 0")))["ok"]
    two = _lean_fixture("CALL 0 2 0 1 1 1 3 0 2 1", "CALL 1 2 128 3 1 1 0 1 2 1")
    assert T.flat_lines(T.unroll(T.parse(two))) == flat and T.ordered(T.parse(two))["ok"]


def _lean_fixture(pair_call: str = "CALL 0 2 0 1 1 2 3 0 1 0", root_call: str = "CALL 1 2 128 3 1 2 0 1 1 0") -> str:
    """``Flock.Test.Tmpl``'s template."""
    return "\n".join(["flock-ir-tmpl/v2 t 2 1 0 1 1 3 1 2 1 1 2", "DEF 1 2 1 0", "0 1 2 1 1", "DEF 3 2 1 1",
                      "2 2 1 0 1 5", pair_call, "DEF 400 0 6 1", "0 1 0 1 0", "1 1 1 1 1", "2 1 2 1 2", "256 1 130 1 399",
                      "257 1 133 1 399", "300 3 0 1 0 1 400", root_call])


def test_pack_writes_runs_of_segments_as_two_level_ones_where_the_order_check_takes_them():
    one = T.parse(_lean_fixture("CALL 0 2 0 1 2 1 3 0 1 0 1 4 0 1 0", "CALL 1 2 128 3 2 1 0 1 1 0 1 1 1 1 0"))
    packed = T.pack(one)
    assert T.write(packed) == T.write(T.parse(_lean_fixture("CALL 0 2 0 1 1 1 3 0 2 1", "CALL 1 2 128 3 1 1 0 1 2 1")))
    assert T.flat_lines(T.unroll(packed)) == T.flat_lines(T.unroll(one)) and T.ordered(packed)["ok"]
    # a port bit and the constant, each taken alone, are not one ordered segment
    split = T.parse(_lean_fixture(root_call="CALL 1 1 128 3 2 1 2 0 1 0 1 399 0 1 0"))
    assert T.ordered(split)["ok"]
    assert T.pack(split).defs[2].calls == split.defs[2].calls


def test_legalize_splits_a_root_segment_across_two_groups_port_bits_and_unrolls_the_same():
    tmpl = T.parse("\n".join(["flock-ir-tmpl/v2 s 2 2 0 1 1 128 1 1 1 128 1 2 1 1 2", "DEF 1 0 1 0", "0 1 0 1 0",
                              "DEF 1 2 1 0", "0 1 2 1 1", "DEF 400 0 0 2", "CALL 0 256 0 1 0",
                              "CALL 1 2 256 1 1 2 127 1 1 0"]))
    before = T.ordered(tmpl)
    assert not before["ok"] and before["padding"] == 1, before
    legal, st = T.legalize(tmpl)
    assert st == {"split_segs": 1, "split_runs": 0, "single_calls": 2}, st
    assert T.ordered(legal)["ok"] and T.ordered(T.parse(T.write(legal)))["ok"]
    assert T.flat_lines(T.unroll(legal)) == T.flat_lines(T.unroll(tmpl))
    assert T.flat_lines(T.unroll(legal))[1 + 256:1 + 258] == ["1 128 1 127", "1 129 1 128"]


def _broadcast(nb: int = 6, w: int = 80) -> list[str]:
    """Blocks of ``w`` rows from 128: a port bit's copy, a chain, and a last row reading the chain's end and block 0's
    first row, which in block 0 is a column of its own."""
    k = 128 + nb * w
    rows = {i: f"1 {i} 1 {i}" for i in range(8)}
    for b in range(nb):
        o = 128 + b * w
        rows[o] = f"1 0 1 {k}"
        rows |= {o + t: f"1 {o + t - 1} 1 {k}" for t in range(1, w - 1)}
        rows[o + w - 1] = f"1 {o + w - 2} 1 128"
    rows[k] = f"1 {k} 1 {k}"
    return [f"flock-ir-unit/v2 bcast {k + 1} {k} 1 0 1 1 8 0"] + [rows.get(i, "0 0") for i in range(k + 1)]


def test_the_emitter_makes_no_run_whose_copy_0_reads_its_own_columns_through_an_input(monkeypatch):
    lines = _broadcast()
    t, stats = T.emit(T.parse_flat(lines))
    assert stats["order"]["ok"] and stats["unaligned_runs"] > 0, stats
    assert T.flat_lines(T.unroll(T.parse(T.write(t)))) == lines
    monkeypatch.setattr(T._Emitter, "_aligned", lambda self, lv, i0, P, m, D: m)
    assert T.emit(T.parse_flat(lines))[1]["order"]["late"] > 0


def _unrolled(t: T.Tmpl, lanes: np.ndarray, out_at: int, n: int) -> list[int]:
    """The unrolled template's columns ``[out_at, out_at + n)``, evaluated in column order."""
    return S.evaluate(T.unroll(t), [int(v) for v in lanes], len(lanes))[out_at:out_at + n]


@pytest.mark.parametrize("stride", [0, 1, 2, 5])
def test_evaluate_reads_every_column_of_a_one_copy_call_whatever_its_stride(stride):
    """One copy of a two-column callee at 128, the output reading its second column: a single copy's stride places
    nothing (``Tmpl.place``, ``Call.copyOf``), so under the callee's columns it must not cut the copy short."""
    t = T.parse("\n".join(["flock-ir-tmpl/v2 one 1 1 0 1 1 1 1 2 1 1 1", "DEF 2 1 2 0", "0 1 2 1 3", "1 1 0 1 3",
                           "DEF 400 0 2 1", "0 1 0 1 0", "256 1 129 1 400", f"CALL 0 1 128 {stride} 1 1 0 0 1 0"]))
    lanes = np.array([0x0123456789ABCDEF], np.uint64)
    assert [int(v) for v in T.evaluate(t, lanes[:, None])[:, 0]] == _unrolled(t, lanes, 256, 1) == [0x0123456789ABCDEF]


_IN, _BODY, _OUT, _COLS = 4, (128, 176), 256, 400


def _random_tmpl(rng) -> T.Tmpl:
    """Three Definitions and a root, each a run of rows and calls of earlier ones (so calls nest): copies of any stride
    from 0 to past the callee's columns, overlapping only where no two write one column, and now and then a caller's row
    in a column its copies leave unwritten."""
    defs: list[T.Def] = []
    wrote: list[set[int]] = []

    def body(cols, ins, lo, hi, written, d):
        const, rows, calls, holes, x = cols + ins, [], [], [], lo
        while x < hi:
            pool = written + list(range(cols, const + 1))

            def pick():
                return int(pool[rng.integers(len(pool))])

            if holes and rng.random() < 0.08:
                h = holes.pop(int(rng.integers(len(holes))))
                rows.append((h, tuple(c for c in (pick(), pick()) if c < h or c >= cols), (const,)))
                written.append(h)
                continue
            if d and rng.random() < 0.4:
                e = int(rng.integers(d))
                Le, count = defs[e].cols, int(rng.integers(1, 4))
                stride = Le if rng.random() < 0.4 else int(rng.integers(0, Le + 3))
                occ = [j * stride + y for j in range(count) for y in sorted(wrote[e])]
                if len(set(occ)) < len(occ):
                    stride = Le
                    occ = [j * stride + y for j in range(count) for y in sorted(wrote[e])]
                end = x + (count - 1) * stride + Le
                if end <= hi:
                    segs = tuple((1, pick(), int(rng.choice([0, 0, 1, 2])), 1, 0) for _ in range(defs[e].ins))
                    calls.append(T.Call(e, count, x, stride, segs))
                    written += [x + y for y in occ]
                    holes, x = [x + y for y in range(end - x) if y not in set(occ)], end
                    continue
            holes = []
            if rng.random() < 0.7:
                a = tuple(pick() for _ in range(int(rng.integers(1, 4))))
                rows.append((x, a, (const,) if rng.random() < 0.5 else tuple(pick() for _ in range(int(rng.integers(1, 3))))))
                written.append(x)
            x += 1
        return rows, calls, written

    for d in range(3):
        cols, ins = int(rng.integers(2 + 3 * d, 5 + 6 * d)), int(rng.integers(1, 4))
        rows, calls, written = body(cols, ins, 0, cols, [], d)
        defs.append(T.Def(cols, ins, tuple(sorted(rows)), tuple(calls)))
        wrote.append(set(written))
    rows, calls, _ = body(_COLS, 0, *_BODY, list(range(_IN)), 3)
    outs = [(_OUT + j, tuple(int(rng.integers(*_BODY)) for _ in range(2)), (_COLS,)) for j in range(8)]
    root = [(i, (i,), (i,)) for i in range(_IN)] + rows + outs
    defs.append(T.Def(_COLS, 0, tuple(sorted(root)), tuple(calls)))
    return T.Tmpl("probe", 3, ["1", "0", "1", "1", str(_IN), "1", str(_OUT // T.WORD), "1", "1", "8"], defs)


def _shapes(t: T.Tmpl) -> set[str]:
    out = set()
    for d in t.defs:
        for c in d.calls:
            Le, end = t.defs[c.e].cols, c.base + (c.count - 1) * c.stride + t.defs[c.e].cols
            out |= {"nested"} if t.defs[c.e].calls else set()
            out |= {"overlapping copies"} if c.count > 1 and c.stride < Le else set()
            out |= {"one copy, stride under its columns"} if c.count == 1 and c.stride < Le else set()
            out |= {"caller row inside copies"} if any(c.base <= r[0] < end for r in d.rows) else set()
    return out


def test_evaluate_is_the_unrolled_templates_values_or_refuses_on_random_templates():
    """1,000 seeded random templates over nested calls, overlapping copies, caller rows inside copies and one-copy calls
    under their callee's columns: ``evaluate`` either refuses one or gives the unrolled template's values."""
    rng = np.random.default_rng(0)
    seen: dict[str, int] = {}
    differ = []
    for i in range(1000):
        t = _random_tmpl(rng)
        lanes = rng.integers(0, 1 << 64, _IN, dtype=np.uint64)
        try:
            want = _unrolled(t, lanes, _OUT, 8)
        except AssertionError:
            verdict = "not unrollable"
        else:
            try:
                got = [int(v) for v in T.evaluate(t, lanes[:, None])[:, 0]]
            except ValueError:
                verdict = "refused"
            else:
                verdict = "agreed"
                if got != want:
                    differ.append(i)
                    verdict = "differed"
        for s in _shapes(t) | {""}:
            seen[f"{verdict} {s}".strip()] = seen.get(f"{verdict} {s}".strip(), 0) + 1
    assert not differ, (differ[:5], seen)
    assert all(seen.get(k, 0) >= 10 for k in ("agreed nested", "agreed one copy, stride under its columns",
                                                 "refused overlapping copies", "refused caller row inside copies")), seen
