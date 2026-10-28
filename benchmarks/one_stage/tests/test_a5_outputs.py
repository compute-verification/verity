"""a5's output class (`a5.output_closure`): a window's units whose outputs are served out are proved whole beside a work
draw, on the closure the verifier's own map gives every draw. Over several coin seeds the stream's draw of record (the Lean
`Select.select`, agreeing with the Python) draws its k units uniformly and proves the drawn units and the whole class, and
its outcome is accepted with the Lean's `Run.run` agreeing; the proved set less one output unit, and the draw with its
closure dropped, are refused (`a5.output_negatives`); a class that mixes a member, a law that is not a streamed work law,
and a class outside the population are refused before the draw (`a5.output_refusal`). A toy window of one template."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.sampled_proofs import service as S
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5 as A5  # noqa: E402
import a5_coins as AC  # noqa: E402

XOR8 = PrimitiveDefinition("A5OutputsTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("A5OutputsTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
N, K, CAP = 32, 4, 8
OUTPUTS = [[16, 32]]
QUERY = P.template_instances_query((PAIR,))
TABLE = {P.template_id(PAIR): 1}
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-a5-outputs-test"
ROOT = "cd" * 64
SESSION = "5e" * 64
VALUES = (S.Value("x", SCHEMA, "serve"), S.Value("out", SCHEMA, "serve"))
PUBLIC = tuple(S.Public(i, "structure", r) for i, r in (
    ("population", "N, the draw's range"), ("window", "provenance"), ("leaves:x", "x's leaf count"),
    ("domain:x", "x's served domain"), ("leaves:out", "out's leaf count"), ("domain:out", "out's served domain"),
    (f"population:{P.template_id(PAIR)}", "a template's stratum")))
SPEC = S.Spec(VALUES, S.Law(f"work:{K}"), S.Check((P.program_digest(Program(PAIR)),), QUERY, proof_deadline=60.0), PUBLIC,
              bytes.fromhex(ROOT))
VERDICTS = {"lean": {"of_record": True, "accepted": True, "digest": "dd" * 64, "sessions": [SESSION]}}
OUTS = list(range(16, 32))


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def args(outputs=OUTPUTS, coins=None, verifier_coins=True) -> SimpleNamespace:
    return SimpleNamespace(outputs=outputs, coins=coins, verifier_coins=verifier_coins)


def program() -> Program:
    return P.mixed_population_program([(PAIR, N)])


def _ports() -> dict:
    part = P.partition(program(), QUERY)
    return {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, SCHEMA, leaves, RUN)}
            for p, leaves in (("x", N), ("out", 2 * N))}


def _record() -> dict:
    roots = [{"port": p, "root": f"{i + 1:02x}" * 64, **e} for i, (p, e) in enumerate(_ports().items())]
    return R.record(program(), QUERY, {"law": "work", "k": K}, S.SCHEME, roots, {"kind": "served", "run": RUN, "row": 0},
                    work_table=TABLE)


def stream_of(coins) -> S.Stream:
    return S.Stream(SPEC, (PAIR,), budget=1, delta=0.5, challenger="a5 outputs test, verifier", coins=coins, clock=Clock())


def register(stream: S.Stream) -> S.WindowCommitmentRecord:
    return stream.register(_record(), _ports(), window_root=ROOT, counts={P.template_id(PAIR): N}, work_table=TABLE)


def outcome(stream: S.Stream, draw: S.Draw, evidence: dict | None) -> tuple[S.Outcome, dict | None]:
    assert stream.deliver(draw, [SESSION]) is None
    return stream.outcome(draw, VERDICTS, {"statement": "stub"}, coins=evidence), stream.records[-1]["audit"]


def seeded(seed: int):
    rng = random.Random(seed)
    return lambda n: rng.randbytes(n)


@pytest.mark.parametrize("seed", range(6))
def test_every_draw_proves_the_output_class_whole_beside_its_uniform_k_and_is_accepted(seed):
    """The stream's draw of record on the verifier's coins, with the class's closure map: k drawn (the work law over one
    template of work 1 is the uniform k-subset), the closure every output unit not drawn, so the draw proves the drawn
    units and the whole class and nothing else; the outcome is accepted, the Lean's run agreeing."""
    closure = A5.output_closure(args(), N)
    assert closure is not None and set(closure) == set(range(N)) and all(cs == OUTS for cs in closure.values())
    coins = AC.VerifierCoins(seeded(seed))
    stream = stream_of(coins)
    draw = stream.select(register(stream), closure=closure)
    units, proved = draw.draw["units"], D.proved(draw.draw)
    assert draw.draw["law"] == "work" and len(units) == K
    assert set(OUTS) <= set(proved) and set(proved) == set(units) | set(OUTS)
    assert sorted(draw.draw.get("closure", [])) == sorted(set(OUTS) - set(units))
    got, audit = outcome(stream, draw, None)
    assert got.accepted and audit["accepted"] and "run" not in audit["checks"] and audit["verifiers"]["agree"]
    neg = A5.output_negatives(args(), draw.draw, proved)
    assert sorted(neg) == ["draw-without-the-output-closure", "output-unit-left-out-of-the-proved-set"]
    assert all(n["refused"] for n in neg.values()), neg


def test_the_honest_verifier_coin_negatives_hold_with_the_closure():
    """`a5_coins.verifier_negatives` selects under the same closure map, so its replay draws the honest draw and each
    negative is refused for its own reason."""
    closure = A5.output_closure(args(), N)
    coins = AC.VerifierCoins(seeded(7))
    stream = stream_of(coins)
    draw = stream.select(register(stream), closure=closure)
    got, _ = outcome(stream, draw, None)
    assert got.accepted
    neg = AC.verifier_negatives(stream_of, register, stream, coins, draw, outcome, ROOT, closure)
    assert all(n["refused"] for n in neg.values()), neg


def mine(members: list[tuple[int, int]]) -> SimpleNamespace:
    return SimpleNamespace(population=N, members=[SimpleNamespace(base=b, n=n) for b, n in members])


def test_an_output_class_is_refused_unless_a_streamed_work_draw_over_whole_members():
    whole = mine([(b, CAP) for b in range(0, N, CAP)])
    work = {"law": "work"}
    assert A5.output_refusal(args(), whole, work) is None
    assert A5.output_refusal(args(outputs=None), whole, {"law": "subset"}) is None
    assert A5.output_refusal(args(), whole, {"law": "subset"}) == {"output-class-needs-a-streamed-work-draw": "subset"}
    assert A5.output_refusal(args(verifier_coins=False), whole, work) == {"output-class-needs-a-streamed-work-draw": "work"}
    assert A5.output_refusal(args(outputs=[[12, 32]]), whole, work) == {"output-class-mixes-a-member": [1]}
    assert A5.output_refusal(args(outputs=[[16, 33]]), whole, work) == {"output-class-outside-the-population": 32}
    assert A5.output_closure(args(outputs=None), N) is None
