"""That an unwired Program is a wired one's cut (`Spot.Cut.check`, the `cut` piece) against the cut itself
(`population.unwired_population_program`, else the periods' unwired builder it returns the output of): Lean accepts a
pair exactly when the unwired Program's digest is the cut's, and each forged unwired Program is refused by its name."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
from hypothesis import strategies as st

import verity_difftest as D
from verity.circuits import codec
from verity.circuits.codec import descriptor_id
from verity.circuits.defs import bind
from verity.circuits.ops import gemm as G
from verity.circuits.ops.boolean import cast as BC, gemm as BG
from verity.commitments.periods import interface as I
from verity.commitments.periods.population import population_program

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


def held() -> I.Circuit:
    """Four ``Gemm_v2`` units each reading registered ``w`` whole, ``w`` held: a value the unwired root keeps."""
    reads = (I.Read(GID, "w", "w", 0, (4,), 0, (0,)),)
    return I.Circuit("reads", ((GEMM, 4),), registered=I.Registered({"w": 256}, reads=reads, values=("w",)))


def _desc(program) -> dict:
    return json.loads(D.canonical(codec.encode_program(program)))


def _values(circuit: I.Circuit) -> list[tuple[str, int, int]]:
    return list(I.program_values(circuit.registered))


def cut(circuit: I.Circuit, wired) -> tuple:
    """The cut's unwired Program and query: circuits' cut where the tree has it, else the builder whose output it is."""
    try:
        from verity.circuits.population import unwired_population_program
    except ImportError:
        return population_program(list(circuit.templates), values=_values(circuit))
    unwired, query, _ = unwired_population_program(wired, circuit.cuts, [v for v, _n, _w in _values(circuit)])
    return unwired, query


def _case(circuit: I.Circuit, templates=None) -> dict:
    """The piece's input: the circuit's wired Program, an unwired one over ``templates`` (default: the cut's) and the
    cut's query, with the cut's digest (``want``), which the piece doesn't read."""
    wired, _ = circuit.program()
    unwired, query = cut(circuit, wired)
    want = codec.program_digest(unwired)
    if templates is not None:
        unwired, _ = population_program(templates, several=True, values=_values(circuit))
    return {"wired": _desc(wired), "unwired": _desc(unwired), "query": query, "want": want}


def _root(desc: dict) -> dict:
    return desc["definitions"][desc["root"]]["body"]


def _forged(j: dict, edit) -> dict:
    j = copy.deepcopy(j)
    edit(j["unwired"])
    return j


def _swap(desc: dict) -> None:
    nodes = _root(desc)["nodes"]
    nodes[-1], nodes[-2] = nodes[-2], nodes[-1]


ROUNDED = rounded()
FIRST, THEN = (t for t, _n in ROUNDED.templates)
HONEST = [("rounded", _case(ROUNDED)), ("chain", _case(chain())), ("held", _case(held()))]
FORGED = [
    ("templates swapped", _forged(HONEST[0][1], _swap), "cut-order"),
    ("templates swapped by the builder", _case(ROUNDED, [(THEN, 6), (FIRST, 6)]), "cut-order"),
    ("a wrong batch count", _case(ROUNDED, [(FIRST, 6), (THEN, 5)]), "cut-count"),
    ("a template the query doesn't name", _case(ROUNDED, [(FIRST, 6), (GEMM, 6)]), "cut-template"),
    ("a wired parameter left in place", {**HONEST[0][1], "unwired": HONEST[0][1]["wired"]}, "cut-arg"),
    ("an extra root node", _case(ROUNDED, [(FIRST, 6), (THEN, 6), (GEMM, 6)]), "cut-extra"),
]


def _variants() -> list[dict]:
    """Unwired Programs the builder writes over the rounded circuit's templates, at every order and count near the cut's."""
    return [_case(ROUNDED, ts) for a in (5, 6, 7) for b in (5, 6, 7)
            for ts in ([(FIRST, a), (THEN, b)], [(THEN, b), (FIRST, a)], [(FIRST, a)], [(FIRST, a), (THEN, b), (GEMM, 1)])]


def reference(j: dict) -> dict:
    """Accepts exactly the cut's own digest."""
    return {"accept": codec.program_digest(j["unwired"]) == j["want"]}


CUT = D.Port("spot-difftest", "cut", reference,
             st.sampled_from([c for _, c in HONEST] + [c for _, c, _ in FORGED] + _variants()),
             package=PACKAGE, need_rejects=False, of_record="lean")
pieces = D.server_fixture(CUT)


@pytest.mark.parametrize("name, j", HONEST, ids=[c[0] for c in HONEST])
def test_the_cut_is_accepted(pieces, name, j):
    """The cut of each circuit's wired Program is accepted."""
    assert reference(j) == {"accept": True}
    assert pieces.call("cut", j) == ("output", D.canonical({"accept": True}))


@pytest.mark.parametrize("name, j, code", FORGED, ids=[c[0] for c in FORGED])
def test_a_forged_unwired_program_is_refused_by_name(pieces, name, j, code):
    """Each forged unwired Program is not the cut, by digest, and Lean refuses it by the rule's name."""
    assert reference(j) == {"accept": False}
    assert pieces.call("cut_why", j) == ("output", D.canonical({"refusal": code}))


def test_every_refusal_has_a_case():
    assert {c for _, _, c in FORGED} == {"cut-order", "cut-template", "cut-count", "cut-arg", "cut-extra"}


def test_lean_accepts_exactly_the_cut(pieces):
    """Lean and the digest agree on the cut, the forgeries and the builder's other populations of the templates."""
    stats = D.run(CUT, examples=60, server=pieces)
    assert stats.outputs > 0, stats
