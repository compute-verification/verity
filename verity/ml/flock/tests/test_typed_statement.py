"""The typed circuit statement (`verity_flock.typed_statement`): a flat class's statement carries its unit as a circuit type
and a layout, and no rows. The rows the verifier derives from them are the old statement's byte for byte, every other part
of the statement is unchanged, and the prover's copy of the rows is those rows."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from verity_flock import circuit as FC
from verity_flock import circuit_types as CT
from verity_flock import derive as D
from verity_flock import layouts as LY
from verity_flock import typed_statement as TS

pytest.importorskip("verity_numerical")
pytest.importorskip("verity_vllm.program.registry.prims")
from verity_numerical.bench import lowerings  # noqa: E402
from verity_numerical.bench import templates as TM  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))


def _low(template, **params):
    sub = TM.subcircuit(template, **params)
    return lowerings.registry("C-Flock").module(template).lowering(sub), sub


@pytest.mark.parametrize("template,params", [("rope-head", {"D": 64}), ("silu-mul", {"I": 8192})])
@pytest.mark.slow
def test_a_flat_class_carries_types_not_rows(template, params):
    low, sub = _low(template, **params)
    old = FC.compose(low, sub)
    typed = TS.compose(low, sub)
    h0, m0, s0 = TS.sections(old)
    h1, m1, s1 = TS.sections(typed.text)
    # the statement is named anew, its unit a type and a layout, its unit's rows gone
    assert h1 == f"{TS.FORMAT} {low.name}" and m1["statement"] == TS.STATEMENT
    assert [n for n, _ in s1] == [n for n, _ in s0 if n != "unit"] and dict(s1) == {n: s for n, s in s0 if n != "unit"}
    new_keys = {"types", "layouts", "unit_layout", "unit_type", "policy", "unit_in"}
    assert set(m1) - set(m0) == new_keys
    assert {k: v for k, v in m1.items() if k not in new_keys | {"format", "statement"}} == \
        {k: v for k, v in m0.items() if k not in {"format", "statement"}}
    # what the verifier derives from META is the old unit section, byte for byte, and the prover's copy is it
    types = {d: CT.from_content(c) for d, c in m1["types"].items()}
    for t in types.values():
        CT.validate(t, types.__getitem__)
    layouts = [LY.from_json(r) for r in m1["layouts"]]
    assert LY.validate(types, layouts, m1["unit_layout"], {}.get)
    dv = D.Derivation(types, layouts, m1["unit_layout"], {})
    rows = D.netlist(dv.unit, low.name)
    assert rows == dict(s0)["unit"] == typed.unit_rows
    # unit_sha256 is unchanged: the lowering's unit text, the derived rows followed by its LEAVES (and CUT) lines
    extra = "LEAVES " + json.dumps(low.leaves, sort_keys=True, separators=(",", ":")) + "\n"
    if low.cut is not None:
        extra += "CUT " + json.dumps(low.cut, sort_keys=True, separators=(",", ":")) + "\n"
    assert m1["unit_sha256"] == m0["unit_sha256"] == hashlib.sha256((rows + extra).encode()).hexdigest()
    assert m1["unit_type"] == layouts[m1["unit_layout"]].type


def _set(sub, n, tmp_path):
    from verity_numerical.bench.generate import generate
    from verity_numerical.bench.input_sets import InputSet

    return InputSet.open(generate(sub, n, 20260926, tmp_path)["path"])


def _root_inputs(s, sub, T):
    """The set's first T instances as the instance type's input bits: each input row's words in port order, 16 bits each,
    bit-sliced over T lanes."""
    bits = []
    for p in sub.inputs:
        W = np.asarray(s.port(p.name, 0, T), dtype=np.uint64).reshape(T, -1)
        for wi in range(W.shape[1]):
            for b in range(16):
                col = (W[:, wi] >> np.uint64(b)) & np.uint64(1)
                bits.append(int(sum(int(v) << lane for lane, v in enumerate(col))))
    return bits


def _root_outputs(s, sub, T):
    want = [np.asarray(s.port(p.name, 0, T), dtype=np.uint64).reshape(T, -1).tolist() for p in sub.outputs]
    return [sum((w[lane] for w in want), []) for lane in range(T)]


def _words(out, T):
    return [[sum(((out[16 * o + b] >> lane) & 1) << b for b in range(16)) for o in range(len(out) // 16)] for lane in range(T)]


def test_gemm_is_one_instance_type(tmp_path):
    """GEMM's instance type (4 k-step calls, the epilogue's gates as its own rows) computes the set's outputs, and so do the
    rows derived from the statement's types and layouts, solved in logical order with every row holding."""
    from test_derive import solve

    sub = TM.parse_subcircuit_id("gemm-coordinate/k64/sm80-mma-bf16")
    s = _set(sub, 64, tmp_path)
    low = FC.lowering_for_set(s)
    old = FC.compose(low, sub, statics=FC.set_statics(s))
    typed = TS.compose(low, sub, statics=FC.set_statics(s))
    _, m0, s0 = TS.sections(old)
    head, m, secs = TS.sections(typed.text)
    assert head == f"{TS.FORMAT} {low.name}" and m["statement"] == TS.STATEMENT
    assert [n for n, _ in secs] == [FC.SHA_NET, FC.HM_NET] and dict(secs) == {n: t for n, t in s0 if n in (FC.SHA_NET, FC.HM_NET)}
    assert not set(TS.WIRING_KEYS) - {"per_vu", "ranges", "k_log", "vus_per_block"} & set(m)
    types = {d: CT.from_content(c) for d, c in m["types"].items()}
    layouts = [LY.from_json(r) for r in m["layouts"]]
    root = types[m["unit_type"]]
    assert layouts[m["unit_layout"]].type == root.digest() and layouts[m["unit_layout"]].range_log is None
    assert m["per_vu"][layouts[0].digest()] == low.units.n == sum(1 for it in root.items if it[0] == CT.CALL)
    placed = {r["net"]: r["count"] for r in m["ranges"]}
    assert placed["root"] == m["vus_per_block"] and placed[layouts[0].digest()] == m["vus_per_block"] * low.units.n
    # the unit layout derives to the flat statement's unit section
    assert typed.rows[layouts[0].digest()].split("\n")[1:] == dict(s0)["unit"].split("\n")[1:]
    T = 64
    ins, want = _root_inputs(s, sub, T), _root_outputs(s, sub, T)
    assert _words(CT.evaluate(types, root.digest(), ins, T, lambda sha, i: 0), T) == want
    dv = typed.derivation
    got, holds = solve(dv, ins)
    assert holds and _words(got, T) == want


@pytest.mark.slow
def test_attention_is_one_instance_type(tmp_path):
    """Attention's instance type (its tensor-core units as calls, its softmax stages' gates as its own rows, its ex2 and rcp
    reads of the measured tables) computes the set's outputs."""
    from verity.ml.circuits import library
    from verity_flock import gf2
    from verity_flock import tail as TL

    sub = TM.attention_head(16, 16, T=17)
    s = _set(sub, 16, tmp_path)
    low = FC.lowering_for_set(s)
    inst = TS.instance_type(low, TL.lower_tail(low))
    types = {inst.root.digest(): inst.root, inst.unit.digest(): inst.unit}
    for t in (inst.unit, inst.root):
        CT.validate(t, types.__getitem__)
    assert [n for n, _, _, _ in inst.tables] == ["ex2", "rcp"]
    words = {library.table(n).sha512: np.asarray(gf2.table(n), dtype=np.uint32) for n, _, _, _ in inst.tables}
    T = 16
    out = CT.evaluate(types, inst.root.digest(), _root_inputs(s, sub, T), T, lambda sha, i: int(words[sha][i]))
    assert _words(out, T) == _root_outputs(s, sub, T)


@pytest.mark.slow
def test_attention_composes_with_placed_reads(tmp_path):
    """Attention's typed statement: its reads are placed `table/v2` layouts in block ranges of their own, one per MUFU read."""
    sub = TM.attention_head(64, 16, T=9)
    s = _set(sub, 2, tmp_path)
    low = FC.lowering_for_set(s)
    typed = TS.compose(low, sub, statics=FC.set_statics(s))
    _, m, secs = TS.sections(typed.text)
    layouts = [LY.from_json(r) for r in m["layouts"]]
    gens = [l for l in layouts if l.own[0] == "gen"]
    assert {(l.own[2], l.own[3], l.own[4]) for l in gens} == {(23, 31, 14)} and len(gens) == 2
    root = layouts[m["unit_layout"]]
    assert sum(m["per_vu"][g.digest()] for g in gens) == len(root.reads) == 10
    assert all(e[1] is not None for e in root.reads) and len(root.calls) == low.units.n == m["per_vu"][layouts[0].digest()]


def test_the_statement_parses_back():
    low, sub = _low("rope-head", D=64)
    typed = TS.compose(low, sub)
    head, meta, secs = TS.sections(typed.text)
    rebuilt = head + "\nMETA " + json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n" + \
        "".join(f"CIRCUIT {n} {s.count(chr(10))}\n{s}" for n, s in secs)
    assert rebuilt == typed.text
