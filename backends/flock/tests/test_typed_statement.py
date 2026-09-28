"""The typed circuit statement (`verity_flock.typed_statement`): a flat class's statement carries its unit as a circuit type
and a layout, and no rows. The rows the verifier derives from them are the old statement's byte for byte, every other part
of the statement is unchanged, and the prover's copy of the rows is those rows."""
from __future__ import annotations

import hashlib
import json

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


def _low(template, **params):
    sub = TM.subcircuit(template, **params)
    return lowerings.registry("C-Flock").module(template).lowering(sub), sub


@pytest.mark.parametrize("template,params", [("rope-head", {"D": 64}), ("silu-mul", {"I": 8192})])
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


def test_a_template_with_a_tail_waits_for_the_next_step():
    class U:
        cut, tail_rets = [(0, 0)], []

    class Low:
        units, name = U(), "attention"

    with pytest.raises(NotImplementedError, match="tail"):
        TS.compose(Low(), None)


def test_the_statement_parses_back():
    low, sub = _low("rope-head", D=64)
    typed = TS.compose(low, sub)
    head, meta, secs = TS.sections(typed.text)
    rebuilt = head + "\nMETA " + json.dumps(meta, sort_keys=True, separators=(",", ":")) + "\n" + \
        "".join(f"CIRCUIT {n} {s.count(chr(10))}\n{s}" for n, s in secs)
    assert rebuilt == typed.text
