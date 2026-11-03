"""`read_slots`: a unit's reads of the pinned MUFU tables split out to Flock lookup slots. On a unit reading EX2, then RCP of
a sum over it (depth 2, a sigmoid's shape), with a depth-0 word returned by the last stage: the unit, the slots and the stages
compute what the flat circuit does on every lane, and a flipped unit row fails its lane; the wires fill every stage and slot
input once, each from a port of its width, each stage reading only earlier stages and slots; a cap on a net's ANDs cuts the
same unit into more stages, none past the cap, computing the same; `net_rows` is every net's laid-out rows, and `ands_cap` the
most ANDs a slot holds (at the SparseMlaSplitPart T=64 unit's ports: 11,534,336 in 2^24); the automatic caps cut a unit whose
AND chain overflows its slot into nets of the next slot down, and keep a split no cap shrinks; a unit reading no slot table
isn't split. Through `UnitStager`, the composed statement names the slots and the last stage's returned outputs (slotted when
the output words aren't a power of two times 16 bits), and its staged outputs are the IR reference's."""
from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

import numpy as np
import pytest

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops.fp32 import F32AddFtz
from verity.ml.circuits.ops.mufu import MufuEx2Ftz, MufuRcpFtz
from verity.ml.circuits.refs import tuple_of
from verity.ml.circuits.types import Array, Tuple, Value
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import class_statement as CS
from verity_flock import gf2
from verity_flock import ir_lower as IL
from verity_flock import read_slots as RS
from verity_flock import tail as TL

F32 = Value(32)
IN = [f"in{j}" for j in range(4)]
OUT = [f"out{o}" for o in range(3)]
EDGES = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00000, 1, 0x00800000, 0x3F800000, 0xC2FE0000, 0x42FE0000]


def _units(n_in: int, n_out: int) -> IL.Units:
    return IL.Units(target=None, in_widths=[16] * n_in, out_widths=[16] * n_out, in_src=[[("leaf", p) for p in range(n_in)]],
                    out_src=[[("ret", o) for o in range(n_out)]], signature=(), gates=[], arg_gates=[], ret_gates=[])


def _sigmoid(C: gf2.Circuit, reads: bool = True, chain: int = 0) -> None:
    """``chain``: that many more ANDs on the inputs, one after another, the last one XORed into ``out2``'s low bit."""
    w = [C.input(nm, 16) for nm in IN]
    x, y = w[0] + w[1], w[2] + w[3]
    r = IL.PIECES["MufuRcpFtz_v1"](C, IL.PIECES["F32AddFtz_v1"](C, IL.PIECES["MufuEx2Ftz_v1"](C, x), y)) if reads else x
    C.output("out0", r[:16])
    C.output("out1", r[16:])
    o = [C.AND(a, b) for a, b in zip(x[:16], y[16:])]
    fs = x + y + [a ^ b for a, b in zip(x, y)]
    c = fs[0]
    for i, j in itertools.islice(itertools.combinations(range(len(fs)), 2), chain):
        c = C.AND(c ^ fs[i], fs[j])
    if chain:
        o[0] = o[0] ^ c
    C.output("out2", o)


@pytest.fixture(scope="module")
def split():
    C = gf2.Circuit()
    _sigmoid(C)
    low = RS.split(C, IN, OUT, "unit/read-slots-toy", SimpleNamespace(id="toy"), _units(4, 3), stage_ands=None)
    return C, low


def _flat(xs, ys) -> np.ndarray:
    xs, ys = np.asarray(xs, np.uint64), np.asarray(ys, np.uint64)
    return np.stack([xs & 0xFFFF, xs >> 16, ys & 0xFFFF, ys >> 16], axis=1)


def test_the_slots_and_stages_compute_the_flat_circuit(split):
    C, low = split
    assert [(t, s) for _g, t, s in low.reads] == [("ex2", 0), ("rcp", 1)] and len(low.tail.stages) == 2
    assert low.layout.counts["reads"] == 0 and all(st.layout.counts["reads"] == 0 for st in low.tail.stages)
    rng = np.random.default_rng(7)
    xs = EDGES + [int(v) for v in rng.integers(0, 1 << 32, 54, dtype=np.uint64)]
    ys = EDGES[::-1] + [int(v) for v in rng.integers(0, 1 << 32, 54, dtype=np.uint64)]
    flat = _flat(xs, ys)
    T = len(xs)
    got, cut, ok = low.evaluate(flat)
    assert ok == (1 << T) - 1 and cut.shape == (T, 0)
    want, fail = C.evaluate({nm: IL.slice_words(flat[:, j], 16) for j, nm in enumerate(IN)}, T)
    assert fail == 0
    for o, nm in enumerate(OUT):
        assert np.array_equal(got[:, o], IL.unslice_words(want[nm], T)), nm
    row = low.layout.in_words * IL.WORD
    _, _, bad = low.evaluate(flat, flip=(row, 3))
    assert bad == ok & ~(1 << 3)


def test_the_wires_fill_every_input_once_from_a_port_of_its_width(split):
    _check_wires(split[1])


def _check_wires(low) -> None:
    st = low.tail.stages
    outs = {"unit": low.layout.out_bits, **{f"stage{i}": s.layout.out_bits for i, s in enumerate(st)}}
    ins = {f"stage{i}": s.layout.in_bits for i, s in enumerate(st)}
    order = {"unit": 0, **{f"stage{i}": i + 1 for i in range(len(st))}}
    for g, t, _s in low.reads:
        outs[f"lookup:{t}"] = [TL.VALUE_BITS]
        ins[f"lookup:{t}"] = [TL.TABLE_BITS[t]]
    seen = []
    for dn, di, dp, sn, si, sp in low.tail.wires:
        assert di == si == 0 and ins[dn][dp] == outs[sn][sp], (dn, dp, sn, sp)
        if not dn.startswith("lookup:") and not sn.startswith("lookup:"):
            assert order[sn] < order[dn]
        seen.append((dn, dp))
    assert sorted(seen) == sorted((n, p) for n, ws in ins.items() for p in range(len(ws)))
    assert [t[0] for t in st[-1].outs] == ["ret"] * 3 and low.tail.slotted_rets
    assert low.units.tail_rets == [0, 1, 2] and low.units.cut and {k for k, _ in low.units.out_src[0]} == {"cut"}


def test_a_cap_on_a_nets_ands_splits_it_into_more_stages_computing_the_same():
    C = gf2.Circuit()
    _sigmoid(C)
    whole = RS.split(C, IN, OUT, "unit/read-slots-toy", SimpleNamespace(id="toy"), _units(4, 3), stage_ands=None)
    cap = max(whole.layout.counts["ands"], *(st.layout.counts["ands"] for st in whole.tail.stages)) // 3
    low = RS.split(C, IN, OUT, "unit/read-slots-capped", SimpleNamespace(id="toy"), _units(4, 3), stage_ands=cap)
    nets = [low.layout, *(st.layout for st in low.tail.stages)]
    assert len(nets) > 3 and all(0 < lay.counts["ands"] <= cap for lay in nets[:-1]) and nets[-1].counts["ands"] <= cap
    assert sum(lay.counts["ands"] for lay in nets) == sum(lay.counts["ands"] for lay in [whole.layout, *(st.layout for st in whole.tail.stages)])
    _check_wires(low)
    rng = np.random.default_rng(11)
    xs = EDGES + [int(v) for v in rng.integers(0, 1 << 32, 22, dtype=np.uint64)]
    ys = EDGES[::-1] + [int(v) for v in rng.integers(0, 1 << 32, 22, dtype=np.uint64)]
    flat = _flat(xs, ys)
    got, _cut, ok = low.evaluate(flat)
    want, _ = whole.evaluate(flat)[:2]
    assert ok == (1 << len(xs)) - 1 and np.array_equal(got, want)
    with pytest.raises(ValueError, match="stage_ands"):
        RS.split(C, IN, OUT, "unit/read-slots-zero", SimpleNamespace(id="toy"), _units(4, 3), stage_ands=0)


def _nets(low) -> list:
    return [low.layout, *(st.layout for st in low.tail.stages)]


def _cells(low) -> int:
    return sum(1 << RS.net_log(lay.useful) for lay in _nets(low))


def test_net_rows_is_each_nets_laid_out_rows():
    C = gf2.Circuit()
    _sigmoid(C, chain=1500)
    for stage_ands in (None, 700, "auto"):
        low = RS.split(C, IN, OUT, "unit/read-slots-rows", SimpleNamespace(id="toy"), _units(4, 3), stage_ands=stage_ands)
        for lay in _nets(low):
            c = lay.counts
            assert RS.net_rows(c["ands"], c["asserts"], sum(lay.in_bits), sum(lay.out_bits)) == lay.useful, stage_ands
            L = RS.net_log(lay.useful)
            assert RS.ands_cap(L, c["asserts"], sum(lay.in_bits), sum(lay.out_bits)) >= c["ands"]
            assert RS.ands_cap(L - 1, c["asserts"], sum(lay.in_bits), sum(lay.out_bits)) < c["ands"]


def test_the_cap_at_the_t64_units_ports_fills_two_to_the_24():
    """SparseMlaSplitPart_v1{T=64}'s unit under reads in slots: 599,040 input bits (a 2^20-row group), an output group of
    2^21 rows. 12,582,912 ANDs put it one row past 2^24 (a slot of 2^25, and a block of 2^28); 11,534,336, the cap that run
    took, is the most a 2^24 slot holds."""
    for out_bits in ((1 << 20) + 1, 1 << 21):
        assert RS.net_rows(12_582_912, 0, 599_040, out_bits) == (1 << 24) + 1
        assert RS.ands_cap(24, 0, 599_040, out_bits) == 11_534_336
        assert RS.net_log(RS.net_rows(11_534_336, 0, 599_040, out_bits)) == 24
        assert RS.net_log(RS.net_rows(11_534_337, 0, 599_040, out_bits)) == 25
    assert RS.ands_cap(21, 0, 599_040, 1 << 21) < 1


def test_the_automatic_caps_cut_an_overflowing_unit_into_nets_of_the_next_slot_down():
    C = gf2.Circuit()
    _sigmoid(C, chain=1500)
    whole = RS.split(C, IN, OUT, "unit/read-slots-chain", SimpleNamespace(id="toy"), _units(4, 3), stage_ands=None)
    low = RS.split(C, IN, OUT, "unit/read-slots-chain", SimpleNamespace(id="toy"), _units(4, 3))
    top = max(RS.net_log(lay.useful) for lay in _nets(whole))
    assert RS.net_log(whole.layout.useful) == top and low.caps is not None and whole.caps is None
    assert all(RS.net_log(lay.useful) <= top - 1 for lay in _nets(low)) and _cells(low) < _cells(whole)
    assert sum(lay.counts["ands"] for lay in _nets(low)) == sum(lay.counts["ands"] for lay in _nets(whole))
    _check_wires(low)
    rng = np.random.default_rng(13)
    xs = EDGES + [int(v) for v in rng.integers(0, 1 << 32, 22, dtype=np.uint64)]
    ys = EDGES[::-1] + [int(v) for v in rng.integers(0, 1 << 32, 22, dtype=np.uint64)]
    flat = _flat(xs, ys)
    got, _cut, ok = low.evaluate(flat)
    want, _cut, ok_whole = whole.evaluate(flat)
    assert ok == ok_whole == (1 << len(xs)) - 1 and np.array_equal(got, want)


def test_the_automatic_caps_keep_a_split_no_cap_shrinks(split):
    C, whole = split
    low = RS.split(C, IN, OUT, "unit/read-slots-toy", SimpleNamespace(id="toy"), _units(4, 3))
    assert low.caps is None and [lay.useful for lay in _nets(low)] == [lay.useful for lay in _nets(whole)]
    with pytest.raises(ValueError, match="stage_ands"):
        RS.split(C, IN, OUT, "unit/read-slots-bad", SimpleNamespace(id="toy"), _units(4, 3), stage_ands="smallest")


def test_a_unit_reading_no_slot_table_is_not_split():
    C = gf2.Circuit()
    _sigmoid(C, reads=False)
    assert RS.split(C, IN, OUT, "unit/no-reads", SimpleNamespace(id="toy"), _units(4, 3)) is None


#: the stager's rows: a and b, and w (unread), a row spanning SHA-512 blocks, as a composed circuit needs one row to
ROWS = [("ab", V2, 64, ["a", "b"]), ("w", V2, 2048, ["w"])]


def _sigmoid_def(outputs: int):
    def body(B, S, a, b, w):
        e = B.call(MufuEx2Ftz, a)
        r = B.call(MufuRcpFtz, B.call(F32AddFtz, e, b))
        return tuple_of(r, e, b) if outputs == 3 else r

    ret = Tuple(F32, F32, F32) if outputs == 3 else F32
    return bind(CompositeDefinition(f"ReadSlotsToy{outputs}", 1, (), lambda S: ((("a", F32), ("b", F32), ("w", Array(64, F32))), ret),
                                    body, register=False))


@pytest.mark.parametrize("outputs", [1, 3])
def test_a_stager_puts_the_reads_in_slots_and_stages_the_reference(outputs, tmp_path):
    d = _sigmoid_def(outputs)
    st = CS.UnitStager(d, f"unit/read-slots-{outputs}", ROWS, [(V2, tuple(range(outputs)))], reads="slots")
    m = st.meta
    assert set(m["lookups"]) == {"ex2", "rcp"} and m["per_vu"]["lookup:ex2"] == m["per_vu"]["lookup:rcp"] == 1
    assert m["out_net"] == "stage1" and m["leaves_out"] == [[]]
    words = 2 * outputs
    assert m["out_net_ports"] == ([[0, j] for j in range(words)] if outputs == 3 else list(range(words)))
    unit = next(r for r in m["ranges"] if r.get("net") == "unit")
    assert unit.get("packed") is True
    rng = np.random.default_rng(outputs)
    a = np.array(EDGES[:8] + [int(v) for v in rng.integers(0, 1 << 32, 8, dtype=np.uint64)], np.uint64)
    b = rng.integers(0, 1 << 32, len(a), dtype=np.uint64)
    w = rng.integers(0, 1 << 16, (len(a), 128), dtype=np.uint16)
    got = st.stage({"ab": _flat(a, b).astype(np.uint16), "w": w}, tmp_path, b"k" * 32)
    assert sorted(got["outs"]) == [f"o{j}" for j in range(outputs)]
    assert got["ands"] == st.low.counts["ands"]
    with pytest.raises(ValueError, match="inline"):
        CS.UnitStager(d, "unit/read-slots-tmpl", ROWS, [(V2, tuple(range(outputs)))], reads="slots", layout="iv/hybrid/4")
    assert json.loads((tmp_path / "circuit.txt").read_text().split("\n", 2)[1][5:])["wires"][-1][0].startswith("lookup:")
