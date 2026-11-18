"""That a cut Program is a wired one's cut under its circuit record (`Spot.Cut.check`, the `cut` piece) against the cut
itself (`population.cut_wires`, else the wired builder over the record with no wires, which is what it returns): Lean
accepts a pair exactly when the cut's digest is that one's, and each forged cut and each wired root the cut refuses is
refused by name."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.circuits import codec
from verity.circuits.codec import canonical_json, descriptor_id
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.ops import gemm as G
from verity.circuits.ops.boolean import cast as BC, gemm as BG
from verity.circuits.population import WIRED_FIELDS, wired_population_program
from verity.circuits.program import Program
from verity.circuits.refs import Coll, coll_concat, concat
from verity.circuits.types import Array, Tuple, Value
from verity.commitments.periods import interface as I

PACKAGE = (Path(__file__).resolve().parents[7] / "verity" / "protocols" / "proofs" / "spot" / "verifier" / "lean")
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

GEMM = bind(G.Gemm, K=16, N=16, **G.GEMM_AMPERE)
GID = descriptor_id(GEMM)


def rounded(n: int = 6) -> I.Circuit:
    """``n`` f32 words rounded to bf16 and widened back, each widening's ``a`` wired from its rounding's output."""
    first, then = bind(BG.F2fpBf16), bind(BC.Bf16ToF32)
    wire = I.Wire(descriptor_id(then), "a", descriptor_id(first), "o0", at=0, radix=(n,), base=0, step=(1,))
    return I.Circuit("bits", ((first, n), (then, n)), wires=(wire,))


def chain(requests: int = 2, steps: int = 2, layers: int = 2) -> I.Circuit:
    """One ``Gemm_v2`` template reading itself: unit u >= ``requests`` reads unit u - ``requests``, in ranges of an order."""
    n = layers * steps * requests
    wires = (I.Wire(GID, "x", GID, "o0", at=requests, radix=(n - requests,), base=0, step=(1,)),)
    return I.Circuit("chain", ((GEMM, n),), external=(f"{GID}/x",), wires=wires,
                     order=I.ordered({GID: n}, wires, {GID: range(0, n, requests)}))


def held(values: tuple[str, ...]) -> I.Circuit:
    """Four ``Gemm_v2`` units each reading registered ``w`` whole, ``values`` the registered values."""
    reads = (I.Read(GID, "w", "w", 0, (4,), 0, (0,)),)
    return I.Circuit("reads", ((GEMM, 4),), registered=I.Registered({"w": 256}, reads=reads, values=values))


def record(circuit: I.Circuit) -> dict:
    return {**circuit.description(), "external": list(circuit.external)}


def digest(rec: dict) -> str:
    return hashlib.sha512(canonical_json({k: rec[k] for k in WIRED_FIELDS if k in rec})).hexdigest()


V16 = Value(16)
COPY = PrimitiveDefinition("CutTestCopy16", 1, (("a", V16),), V16, lambda a: a, conformance="protocol-defined")
PAIR_RET = Tuple(Array(2, V16), Array(2, V16))
PAIR = bind(CompositeDefinition("CutTestPair", 1, (), lambda S: ((("x", Array(2, V16)),), PAIR_RET),
                                lambda B, S, x: coll_concat(PAIR_RET, [B.call(COPY, x[i]) for i in (0, 1, 0, 1)]),
                                register=False))


def reader(leaves: int):
    return bind(CompositeDefinition("CutTestRead", leaves, (), lambda S: ((("x", Array(leaves, V16)),), V16),
                                    lambda B, S, x: B.call(COPY, x[0]), register=False))


def by_hand(leaves: int, offsets: list[int], cut: int = 1, writers: int = 2, units: int | None = None):
    """A ``WiredPopulations_v1`` root and its record: ``writers`` `PAIR` units (two fields of two leaves each), then a
    reader of ``leaves`` leaves a unit in ``cut`` rows, unit i reading the writers' output leaves from ``offsets[i]``;
    the record names ``units`` writers (default: ``writers``)."""
    rd = reader(leaves)
    ids, counts = (descriptor_id(PAIR), descriptor_id(rd)), (units or writers, len(offsets))
    rec = {"templates": [[t, n] for t, n in zip(ids, counts)], **({"cuts": {ids[1]: {"x": cut}}} if cut > 1 else {})}
    rets = Tuple(Array(writers, PAIR_RET), Array(len(offsets), Value(16)))

    def body(B, S, x):
        out = B.batch(PAIR, x, axes=(0,))
        arg = Coll(Array(len(offsets), Array(leaves, V16)), concat([out.refs.slice(o, o + leaves) for o in offsets]))
        return coll_concat(rets, [out, B.batch(rd, arg, axes=(0,))])

    defn = CompositeDefinition("WiredPopulations", 1, ("templates", "record", "counts"),
                               lambda S: ((("0.x", Array(writers, Array(2, V16))),), rets), body, register=False)
    wired = Program(bind(defn, templates=ids, record=digest(rec), counts=counts))
    return wired, rec, [(PAIR, writers), (rd, counts[1])]


def _desc(program) -> dict:
    return json.loads(D.canonical(codec.encode_program(program)))


def cut_of(wired, rec: dict, templates) -> tuple:
    """The cut and its query: circuits' cut where the tree has it, else the wired builder over the record with no wires."""
    try:
        from verity.circuits.population import cut_wires
    except ImportError:
        return wired_population_program(templates, {**rec, "wires": []})
    cut, query, _ = cut_wires(wired, rec)
    return cut, query


def _case(wired, rec: dict, templates, cut=None) -> dict:
    """The piece's input: the wired Program, a cut (default: the cut's), the record and the cut's query, with the
    cut's digest (``want``), which the piece doesn't read."""
    honest, query = cut_of(wired, rec, templates)
    return {"wired": _desc(wired), "cut": _desc(cut if cut is not None else honest), "record": rec, "query": query,
            "want": codec.program_digest(honest)}


def _circuit(c: I.Circuit) -> dict:
    wired, _ = c.program()
    return _case(wired, record(c), list(c.templates))


def _root(desc: dict) -> dict:
    return desc["definitions"][desc["root"]]["body"]


def _forged(j: dict, edit) -> dict:
    j = copy.deepcopy(j)
    edit(j)
    return j


def _swap(j: dict) -> None:
    """The cut's last two batches in each other's place, the return following them."""
    body = _root(j["cut"])
    nodes, last = body["nodes"], len(body["nodes"]) - 1
    nodes[-1], nodes[-2] = nodes[-2], nodes[-1]
    body["ret"] = json.loads(json.dumps(body["ret"]).replace(f"\"n\", {last}, ", "@").replace(
        f"\"n\", {last - 1}, ", f"\"n\", {last}, ").replace("@", f"\"n\", {last - 1}, "))


def _extra(j: dict) -> None:
    nodes = _root(j["cut"])["nodes"]
    nodes.append(copy.deepcopy(nodes[-1]))


def _left(j: dict) -> None:
    """The wired root under the cut's key: its wires left in place."""
    w, key = copy.deepcopy(j["wired"]), j["cut"]["root"]
    w["definitions"][key] = {**w["definitions"].pop(w["root"]), "id": key}
    w["root"] = key
    j["cut"] = w


def _other_record(j: dict) -> None:
    j["record"] = {**j["record"], "cuts": {GID: {"x": 2}}}


ROUNDED, CHAIN = _circuit(rounded()), _circuit(chain())
HONEST = [("rounded", ROUNDED), ("chain", CHAIN), ("held, no values", _circuit(held(()))),
          ("held, w a value", _circuit(held(("w",)))),
          ("by hand, uncut", _case(*by_hand(2, [6, 0]))), ("by hand, two rows", _case(*by_hand(4, [2], cut=2)))]
FORGED = [
    ("templates swapped", _forged(ROUNDED, _swap), "cut-order"),
    ("an extra root node", _forged(ROUNDED, _extra), "cut-extra"),
    ("a wire left in place", _forged(ROUNDED, _left), "cut-arg"),
    ("the wired root as its own cut", {**CHAIN, "cut": CHAIN["wired"]}, "cut-record"),
    ("another record's wires", _forged(CHAIN, _other_record), "cut-record"),
    ("a root that is no wired population", {**CHAIN, "wired": _desc(Program(GEMM))}, "cut-input"),
]
REFUSED = [("off a field's first leaf", by_hand(2, [1]), "cut-offset"),
           ("over a unit's two fields", by_hand(4, [0]), "cut-fields"),
           ("across two writer units", by_hand(4, [2]), "cut-units"),
           ("a cut row off a field's first leaf", by_hand(4, [1], cut=2), "cut-offset")]


def reference(j: dict) -> dict:
    """Accepts exactly the cut's own digest under the record the wired root names."""
    return {"accept": codec.program_digest(j["cut"]) == j["want"]}


CUT = D.Port("spot-difftest", "cut", reference,
             st.sampled_from([c for _, c in HONEST] + [c for _, c, _ in FORGED[:3]]),
             package=PACKAGE, need_rejects=False, of_record="lean")
pieces = D.server_fixture(CUT)


@pytest.mark.parametrize("name, j", HONEST, ids=[c[0] for c in HONEST])
def test_the_cut_is_accepted(pieces, name, j):
    """The cut of each wired Program under its record is accepted."""
    assert reference(j) == {"accept": True}
    assert pieces.call("cut_why", j) == ("output", D.canonical({"refusal": None}))


@pytest.mark.parametrize("name, j, code", FORGED, ids=[c[0] for c in FORGED])
def test_a_forged_cut_is_refused_by_name(pieces, name, j, code):
    """Each forged cut is refused by the rule's name."""
    assert pieces.call("cut_why", j) == ("output", D.canonical({"refusal": code}))


@pytest.mark.parametrize("name, made, code", REFUSED, ids=[c[0] for c in REFUSED])
def test_a_wired_root_the_cut_refuses_is_refused(pieces, name, made, code):
    """Each wired root circuits' cut refuses by name, Lean refuses by the same name."""
    wired, rec, templates = made
    try:
        from verity.circuits.population import cut_wires
    except ImportError:
        pass
    else:
        with pytest.raises(ValueError, match=code):
            cut_wires(wired, rec)
    cut, query = wired_population_program(templates, {**rec, "wires": []})
    j = {"wired": _desc(wired), "cut": _desc(cut), "record": rec, "query": query}
    assert pieces.call("cut_why", j) == ("output", D.canonical({"refusal": code}))


def test_lean_accepts_exactly_the_cut(pieces):
    """Lean and the digest agree on the cuts and the forgeries the port samples."""
    stats = D.run(CUT, examples=30, server=pieces)
    assert stats.outputs > 0, stats
