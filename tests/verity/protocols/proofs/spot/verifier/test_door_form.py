"""The door's one form: `one_circuit.admitted` is `Spot.OneStageZk.admitted`, case by case. Every `#guard refusedBy` of
`Spot/OneStageZkStatement.lean` is replayed here on its own descriptor, query and cuts, and the Python refuses it by the
same clause with the same text; the descriptor Lean admits is admitted with the same rows and fields; and Friday's
benchmark template is admitted."""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from verity.circuits import partition_object as PO
from verity.circuits import restrict as R
from verity.circuits.codec import descriptor_id
from verity.circuits.program import input_id
from verity.protocols.proofs.spot.verifier.one_stage import one_circuit as OC

LEAN = (Path(OC.__file__).resolve().parents[1] / "lean" / "Spot" / "OneStageZkStatement.lean")
GUARD = re.compile(r'^#guard refusedBy \((admitE|lowerE|admitIn) (.*)\) "([a-z-]+)" "(.*)"$', re.M)
STR = r'"((?:[^"\\]|\\.)*)"'


def _unescape(s: str) -> str:
    return s.replace('\\"', '"').replace("\\\\", "\\")


@cache
def lean_text() -> str:
    return LEAN.read_text()


@cache
def descriptor(name: str) -> str:
    m = re.search(r"^def " + name + r' : String := r#"(.*)"#$', lean_text(), re.M)
    assert m, name
    return m.group(1)


def _args(s: str) -> tuple[str, str, list[tuple[str, int]]]:
    """`admitE`'s arguments as the guard writes them: a descriptor, or one with `.replace a b`, then the template and
    the cuts, each optional."""
    s = s.strip()
    if m := re.match(r"\((\w+)\.replace " + STR + " " + STR + r"\)\s*", s):
        d = descriptor(m.group(1)).replace(_unescape(m.group(2)), _unescape(m.group(3)))
    else:
        m = re.match(r"(\w+)\s*", s)
        d = descriptor(m.group(1))
    s = s[m.end():]
    t = "E_v1"
    if m := re.match(STR + r"\s*", s):
        t, s = m.group(1), s[m.end():]
    cs = [(p, int(k)) for p, k in re.findall(r'\("(\w+)", (\d+)\)', s)]
    return d, t, cs


def _args_in(s: str) -> tuple[str, list[str], list[int]]:
    """`admitIn`'s arguments: a descriptor, the templates the query lists, and the block's units."""
    name, ts, us = re.fullmatch(r"(\w+) \[(.*)\] #\[(.*)\]", s.strip()).groups()
    return descriptor(name), re.findall(STR, ts), [int(u) for u in us.split(",")]


def guards() -> list[tuple[str, str, Any, list, str, str]]:
    out = []
    for m in GUARD.finditer(lean_text()):
        d, t, cs = _args_in(m.group(2)) if m.group(1) == "admitIn" else _args(m.group(2))
        out.append((m.group(1), d, t, cs, m.group(3), _unescape(m.group(4))))
    return out


def _admit(d: str, t: str, cs: list[tuple[str, int]]):
    """`admitE`: C_S of unit 0 of descriptor ``d``, under a query listing ``t``, at the template's cuts ``cs``."""
    desc, c = json.loads(d), d.encode()
    D0 = R.program(c)
    key = next(descriptor_id(nd.fn) for nd in D0.root.body.nodes if input_id(nd.fn.id) is None)
    part = PO.build(desc, PO.template_instance_query(key), [[(0, 1)]])
    D = R.program(OC.restrict(c, part, [(0, 1)]))
    return OC.admitted(D, {"query": PO.template_instance_query(t)}, {t: dict(cs)} if cs else None)


def _admit_in(d: str, ts: list[str], us: list[int]):
    """`admitIn`: C_S of the block of units ``us`` of descriptor ``d``, under a ``Q_template_instances`` query listing
    ``ts``."""
    c = d.encode()
    q = PO.template_instances_query(ts)
    part = PO.build(json.loads(d), q, [[(min(us), max(us) + 1)]])
    D = R.program(OC.restrict(c, part, [(min(us), max(us) + 1)]))
    return OC.admitted(D, {"query": q}, None)


def test_every_lean_guard_is_read():
    assert len(guards()) == lean_text().count("#guard refusedBy")
    assert {g[4] for g in guards()} >= {"one-template", "listed", "port-names", "on-bits", "cut-rows", "whole-words"}


def test_the_prefix_is_leans():
    assert f'def NOT_ADMITTED : String := "{OC.NOT_ADMITTED}"' in lean_text()


@pytest.mark.parametrize("case", guards(), ids=lambda g: f"{g[4]}:{g[5][:48]}")
def test_python_refuses_what_lean_refuses_with_its_text(case):
    how, d, t, cs, clause, detail = case
    with pytest.raises(OC.Refused) as e:
        _admit_in(d, t, cs) if how == "admitIn" else _admit(d, t, cs)
    assert str(e.value) == f"E_LOWER: {OC.NOT_ADMITTED}, {clause}: {detail}"


def test_python_admits_what_lean_admits():
    """`E16`, which Lean's guard admits with rows x, y and fields o0, o1 of one word each."""
    assert _admit(descriptor("E16"), "E_v1", []) == ("E_v1", [("x", 16), ("y", 16)], [("o0", 16), ("o1", 16)])


UNDECODED = re.compile(r'^#guard undecoded (\w+) "(.*)"$', re.M)


@pytest.mark.parametrize("name,why", UNDECODED.findall(lean_text()))
def test_what_lean_does_not_decode_python_does_not(name, why):
    """A template wide on its return alone, and one whose lift to bits would refuse, are no descriptors: the decoders
    refuse them before `admitted` reads them, Python by the same code."""
    with pytest.raises(R.Refused) as e:
        R.program(descriptor(name).encode())
    code = re.search(r"\bE_[A-Z_]+", why).group(0)
    assert code in str(e.value)
    if name == "R16":
        assert str(e.value) == f"E_DESCRIPTOR: {_unescape(why)}"


def test_a_refused_block_reaches_no_executable():
    def never():
        raise AssertionError("built an executable for a C_S the door refuses")

    d = descriptor("F16")
    c, desc = d.encode(), json.loads(d)
    part = PO.build(desc, PO.template_instance_query("E_v1"), [[(0, 1)]])
    with pytest.raises(OC.Refused, match=re.escape(f"{OC.NOT_ADMITTED}, whole-words: ")):
        OC.lower(OC.restrict(c, part, [(0, 1)]), part, never)


@pytest.mark.slow
def test_the_benchmark_template_is_admitted():
    """Friday's benchmark, `product(k=576, s=8, units=288, block=2, registered=False)`: its one template's block, x and
    w whole rows, one return field of 8 words."""
    from verity.circuits.ops.product import product
    p = product(k=576, s=8, units=288, block=2, registered=False, k_log=24, g=2)
    c, part = p.description, p.partition
    key, rows, fields = OC.admitted(R.program(OC.restrict(c, part, OC.block(c, part, 0))), part, OC.cuts(c, part))
    assert key.startswith("Gemm_v2{K=576,N=8,")
    assert (rows, fields) == ([("x", 9216), ("w", 73728)], [("o0", 128)])
