"""`lookup_split`: a unit split at its table reads computes the flat circuit's outputs on every instance (the unit, the stages
and the reads' values evaluated in turn), reads a depth-0 bit as a cut word and hands a later bit on as a pass port; a read
of a table without a lookup slot is refused; and `UnitStager` with ``lookups`` composes the general instance form (META's
lookups, the last stage returning the outputs, every wire's source and destination ports one width) and stages the IR
reference's outputs."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.ops import fp32, mufu
from verity.ml.circuits.types import Array
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import gf2
from verity_flock import ir_lower as IL
from verity_flock import lookup_split as LS

F32 = mufu.F32
#: y = ex2(x0), z = ex2(y + x1), out = (z + y) + x2 over 64 f32 words (a row past one SHA-512 block): two reads, y handed on
CHAIN = bind(CompositeDefinition(
    "LookupSplitChain", 1, (), lambda S: ((("x", Array(64, F32)),), F32),
    lambda B, S, x: (lambda y: B.call(fp32.F32Add, B.call(fp32.F32Add, B.call(mufu.MufuEx2Ftz, B.call(fp32.F32Add, y, x[1])), y), x[2]))(
        B.call(mufu.MufuEx2Ftz, x[0])), register=False))


def _floats(rng, n: int, k: int) -> np.ndarray:
    return rng.uniform(-6, 3, (n, k)).astype(np.float32).view(np.uint32)


def _circuit():
    C = CS._UnitCircuit()
    ins = [C.input(f"in{j}", 16) for j in range(4)]
    x0, x1 = ins[0] + ins[1], ins[2] + ins[3]
    (y,) = IL.lower(C, mufu.MufuEx2Ftz, [x0])
    (s,) = IL.lower(C, fp32.F32Add, [y, x1])
    (z,) = IL.lower(C, mufu.MufuEx2Ftz, [s])
    C.output("out0", z[:16])
    C.output("out1", z[16:])
    C.output("out2", [C.AND(a, b) for a, b in zip(x0[:16], y[:16])])
    C.output("out3", x1[16:])
    return C


def test_a_split_unit_computes_the_flat_circuits_outputs():
    rng = np.random.default_rng(1)
    words = _floats(rng, 40, 2)
    flat = np.stack([words[:, 0] & 0xFFFF, words[:, 0] >> 16, words[:, 1] & 0xFFFF, words[:, 1] >> 16], axis=1).astype(np.uint64)
    C = _circuit()
    T = flat.shape[0]
    bits = {f"in{j}": [sum(((int(flat[t, j]) >> b) & 1) << t for t in range(T)) for b in range(16)] for j in range(4)}
    want, fail = C.evaluate(bits, T)
    assert fail == 0
    low = LS.split(C, "lookup-split-test", type("D", (), {"id": "lookup-split-test"})(), 4, 4)
    got, _cut, ok = low.evaluate(flat)
    assert int(ok) == (1 << T) - 1
    for o in range(4):
        w = [sum(((want[f"out{o}"][b] >> t) & 1) << b for b in range(16)) for t in range(T)]
        assert got[:, o].tolist() == w
    tail = low.tail
    assert [t for t, _s in tail.lookups] == ["ex2", "ex2"] and [s for _t, s in tail.lookups] == [0, 1]
    assert len(tail.stages) == 3 and low.circuit is None and all(st.circuit is None for st in tail.stages)
    assert any(tag[0] == "pass" for tag in tail.stages[1].outs) and any(tag[0] == "pass" for tag in tail.stages[2].ins)
    assert any(tag[0] == "cut" for tag in tail.stages[2].ins)
    assert [tag for tag in tail.stages[2].outs] == [("ret", o) for o in range(4)]


def test_a_read_of_a_table_without_a_lookup_slot_is_refused():
    name = "lookup-split-unpinned"
    gf2.TABLES[name] = lambda: np.arange(1 << 8, dtype=np.uint32)
    C = gf2.Circuit()
    x = C.input("in0", 16)
    v = C.lookup(name, x[:8], 31)
    C.output("out0", v[:16])
    with pytest.raises(NotImplementedError, match="has no lookup slot"):
        LS.split(C, name, type("D", (), {"id": name})(), 1, 1)


def test_a_stager_with_lookup_slots_composes_the_instance_form_and_stages_the_references_outputs(tmp_path):
    st = CS.UnitStager(CHAIN, "unit/lookup-split-chain", [("x", V2, 2048, ["x"])], [(V2, (0,))], lookups=True)
    meta = FC.meta_of(st.text)
    assert meta["per_vu"]["lookup:ex2"] == 2 and meta["lookups"]["ex2"]["sha256"] == IL.TABLES["ex2"]
    assert meta["out_net"] == f"stage{len(st.low.tail.stages) - 1}" and meta["leaves_out"] == [[]]
    widths = {}
    for nm, net in [("unit", st.low.layout)] + [(f"stage{i}", s.layout) for i, s in enumerate(st.low.tail.stages)]:
        widths[nm] = (net.in_bits, net.out_bits)
    for d, _di, dp, s, _si, sp in meta["wires"]:
        if d.startswith("lookup:") or s.startswith("lookup:"):
            continue
        if d in widths and s in widths:
            assert widths[d][0][dp] == widths[s][1][sp]
    rng = np.random.default_rng(2)
    x = _floats(rng, 3, 64)
    rows = {"x": x.view(np.uint16).reshape(3, 128)}
    got = st.stage(rows, tmp_path, b"\x01" * 32)
    y = [mufu.MufuEx2Ftz.evaluate(int(v)) for v in x[:, 0]]
    z = [mufu.MufuEx2Ftz.evaluate(fp32.F32Add.evaluate(a, int(b))) for a, b in zip(y, x[:, 1])]
    want = [fp32.F32Add.evaluate(fp32.F32Add.evaluate(c, a), int(b)) for c, a, b in zip(z, y, x[:, 2])]
    assert got["outs"]["o0"].astype(np.uint32).tolist() == [[w & 0xFFFF, w >> 16] for w in want]
    assert json.loads((tmp_path / "circuit.txt").read_text().split("\n", 2)[1][5:])["lookups"]["ex2"]["in_bits"] == 23
    counts = LS.counts(st.low)
    assert counts["lookup:ex2"]["per_vu"] == 2 and counts["nnz"] < 1 << 32
