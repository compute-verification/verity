"""`network-records verdicts`' `members` (`NetworkCheck.Circuit.members`, after the record's checks) against
`interface.members`: each member's name, first unit among its template's, first VU and units, on stated records of an
embedding read, a `Gemm_v2{K=16, N=8}` and a `Gemm_v2{K=16, N=16}` whose ``x`` is cut in two, with wires, registered
reads, table reads and openings at drawn units and a drawn cap (the shapes of `test_interface.py`'s ``declared``), and on
`period_case.record_audit`'s records; refused on both sides at a cap of 0 or a wire past its reader's units."""
from __future__ import annotations

import json
import shutil
from itertools import accumulate

import pytest
from hypothesis import strategies as st

import period_case as PC
import test_network_check_protocol as V2
import test_network_records_protocol as NR
import verity_difftest as D
from verity.ml.circuits import codec
from verity.ml.circuits.codec import descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as GM
from verity.ml.windows import interface as IF

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

GEMM = bind(GM.Gemm, K=16, N=16, **GM.GEMM_AMPERE)
G8 = bind(GM.Gemm, K=16, N=8, **GM.GEMM_AMPERE)
EMB = IF.resolve("EmbeddingRead_v1{V=4,H=16}")
GID, G8ID, EID = (descriptor_id(d) for d in (GEMM, G8, EMB))
CATALOG = {GID: GEMM, G8ID: G8, EID: EMB, descriptor_id(PC.RECORD_BITS): PC.RECORD_BITS}
WEIGHTS = 512 + 4 * 16
ORDER = "verity/wire-order/v1"


def members_reference(case: dict) -> list[dict]:
    """`interface.members` of the circuit the record states, each member as `membersFn` writes it."""
    doc = case["circuit"]
    try:
        wires, cuts, registered, _ = IF.declarations(doc)
        c = IF.Circuit("members", tuple((CATALOG[t], n) for t, n in doc["templates"]), wires=wires, cuts=cuts,
                       registered=registered)
        ms = IF.members(c, doc["cap"])
    except ValueError as e:
        raise D.Reject(str(e)) from e
    first = dict(zip((t for t, _ in doc["templates"]), accumulate((n for _, n in doc["templates"]), initial=0)))
    return [{"base": m.base, "n": m.n, "name": m.name, "start": m.base - first[m.template]} for m in ms]


@st.composite
def spans(draw, n: int, most: int = 3) -> list[tuple[int, int]]:
    """Disjoint unit ranges `(at, units)` among `n` units."""
    ends = sorted(draw(st.sets(st.integers(0, n), max_size=2 * most)))
    return [(a, b - a) for a, b in zip(ends[::2], ends[1::2])]


def _case(c: IF.Circuit, cap: int, order: dict) -> dict:
    doc = IF.descriptor_record(c, max(cap, 1))
    return {"program": codec.encode_program(c.program()[0]), "circuit": doc | {"cap": cap}, "order": order}


@st.composite
def drawn(draw) -> dict:
    """A stated record over the embedding read (or not), the `K=16, N=8` and the `K=16, N=16` Gemm, its wires each from
    an earlier template to a later one (so the order ranks each template's units by its place), registered reads of
    both Gemms' ``w``, the embedding's table reads and openings, at drawn units and a drawn cap; now and then a wire past
    its reader's units or a cap of 0."""
    emb = draw(st.booleans())
    ne, n8, ng = draw(st.integers(1, 6)), draw(st.integers(1, 12)), draw(st.integers(1, 10))
    templates = ([(EMB, ne)] if emb else []) + [(G8, n8), (GEMM, ng)]
    wires = []
    if emb:
        for a, k in draw(spans(n8)):
            fan = k > ne or draw(st.booleans())
            b = draw(st.integers(0, ne - 1 if fan else ne - k))
            wires.append(IF.Wire(G8ID, "x", EID, "o0", at=a, radix=(k,), base=b, step=(0 if fan else 1,)))
    for a, k in draw(spans(ng)):
        s = draw(st.integers(0, 2))
        if (top := n8 - 2 - (k - 1) * s) >= 0:
            wires.append(IF.Wire(GID, "x", G8ID, "o0", at=a, radix=(k, 2), base=draw(st.integers(0, top)), step=(s, 1),
                                 cut=(0, 2)))
    reads = [IF.Read(G8ID, "w", "weights", a, (k,), draw(st.sampled_from([0, 128, 256])), (0,)) for a, k in draw(spans(n8))]
    reads += [IF.Read(GID, "w", "weights", a, (k,), draw(st.sampled_from([0, 256])), (0,)) for a, k in draw(spans(ng))]
    tables, roots, openings = (), (), ()
    if emb:
        tables = (IF.Table("embed", "weights", 512, 16, 4),)
        roots = tuple(IF.TableRead(EID, p, "embed", p, a, k) for p in ("root", "domain") for a, k in draw(spans(ne, 2)))
        openings = tuple(IF.TableOpening(EID, "embed", "tok", ("row", "salt", "path"), "o1", a, k)
                         for a, k in draw(spans(ne, 2)))
    reg = IF.Registered({"weights": WEIGHTS}, tables, tuple(reads), roots, openings)
    c = IF.Circuit("members", tuple(templates), wires=tuple(wires), cuts={GID: {"x": 2}}, registered=reg)
    place = {descriptor_id(t): i for i, (t, _) in enumerate(templates)}
    order = {"format": ORDER,
             "segments": [{"template": descriptor_id(t), "at": 0, "radix": [n], "c": i, "r": [0]}
                          for i, (t, n) in enumerate(templates)],
             "wires": [[place[w.writer], place[w.reader]] for w in wires]}
    case = _case(c, draw(st.integers(0, 2 * max(ne, n8, ng)) if draw(st.integers(0, 9)) else st.just(0)), order)
    if wires and not draw(st.integers(0, 9)):
        i = draw(st.integers(0, len(wires) - 1))
        doc = case["circuit"]
        doc["wires"][i] = doc["wires"][i] | {"at": dict(doc["templates"])[doc["wires"][i]["reader"]]}
    return case


@st.composite
def recorded(draw) -> dict:
    """`period_case.record_audit`'s Program, record and order (self-wires through two layers of one Gemm, or 16
    `Bf16Add` units writing one Gemm unit's cut ``x``), at a drawn cap."""
    program, record, order = (json.loads(x) for x in PC.record_audit(draw(st.booleans())))
    return {"program": program, "circuit": record | {"cap": draw(st.integers(0, 20))}, "order": order}


MEMBERS = NR.VerdictsPort("network-records", "members", members_reference, st.one_of(drawn(), recorded()),
                          package=V2.LEAN)
members_server = D.server_fixture(MEMBERS)


@pytest.mark.slow
def test_the_members_are_interfaces(members_server):
    """Lean's members are `interface.members`' on every drawn record, and both refuse the same records."""
    stats = D.run(MEMBERS, examples=400, seed=1, server=members_server)
    assert stats.outputs > 200 and stats.rejects > 10, stats


def test_a_cap_that_splits_a_segment(members_server):
    """`declared`'s shape at cap 2: each template's units cut at its wires' and reads' ends, its segments of 3 split
    2 + 1, and the chunks counted across the template."""
    reg = IF.Registered({"weights": WEIGHTS}, (IF.Table("embed", "weights", 512, 16, 4),),
                        (IF.Read(G8ID, "w", "weights", 0, (3, 2), 0, (0, 128)), IF.Read(GID, "w", "weights", 1, (2,), 256, (0,))),
                        (IF.TableRead(EID, "root", "embed", "root", 0, 3),),
                        (IF.TableOpening(EID, "embed", "tok", ("row", "salt", "path"), "o1", 0, 3),))
    c = IF.Circuit("declared", ((EMB, 3), (G8, 6), (GEMM, 3)), cuts={GID: {"x": 2}}, registered=reg, wires=(
        IF.Wire(G8ID, "x", EID, "o0", at=0, radix=(3, 2), base=0, step=(1, 0)),
        IF.Wire(GID, "x", G8ID, "o0", at=0, radix=(3, 2), base=0, step=(2, 1), cut=(0, 2))))
    order = {"format": ORDER, "segments": [{"template": t, "at": 0, "radix": [n], "c": i, "r": [0]}
                                           for i, (t, n) in enumerate(((EID, 3), (G8ID, 6), (GID, 3)))],
             "wires": [[0, 1], [1, 2]]}
    got = D.replay(MEMBERS, _case(c, 2, order), members_server)
    assert got[0] == "output", got
    assert [(m["name"], m["start"], m["base"], m["n"]) for m in json.loads(got[1])] == [
        (f"{EID}/c0", 0, 0, 2), (f"{EID}/c1", 2, 2, 1), (f"{G8ID}/c0", 0, 3, 2), (f"{G8ID}/c1", 2, 5, 2),
        (f"{G8ID}/c2", 4, 7, 2), (f"{GID}/c0", 0, 9, 1), (f"{GID}/c1", 1, 10, 2)]
