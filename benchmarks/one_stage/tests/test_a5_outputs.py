"""a5's output class (`a5.output_units`): a window's units whose outputs are served out would be proved whole beside a work
sample only on a closure map, which the stream refuses (`closure`), so a5 refuses a window with an output class before the
sample (`a5.output_refusal`). On a test's own challenger (`fake_challenger`) the stream refuses the class's closure map
before reading a coin, and the same registration's work sample without one is accepted, the Lean's `Run.run` agreeing. A
toy window of one template."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.types import Array, Value as Bits
from verity.protocols.spot.verifier import service as S
from verity.protocols.spot.verifier.one_stage import draw as D, partition as P, registration as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "verity" / "protocols" / "spot" / "verifier" / "tests"))
import a5 as A5  # noqa: E402
import fake_challenger as FC  # noqa: E402

XOR8 = PrimitiveDefinition("A5OutputsTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("A5OutputsTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
N, K = 32, 4
OUTPUTS = [[16, 32]]
QUERY = P.template_instances_query((PAIR,))
TABLE = {P.template_id(PAIR): 1}
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-a5-outputs-test"
ROOT = FC.WINDOW["root"]
SESSION = "5e" * 64
VALUES = (S.Value("x", SCHEMA, "serve"), S.Value("out", SCHEMA, "serve"))
PUBLIC = tuple(S.Public(i, "structure", r) for i, r in (
    ("population", "N, the draw's range"), ("window", "provenance"), ("leaves:x", "x's leaf count"),
    ("domain:x", "x's served domain"), ("leaves:out", "out's leaf count"), ("domain:out", "out's served domain"),
    (f"population:{P.template_id(PAIR)}", "a template's stratum")))
#: K of the N proof blocks, an eighth of however many the window registers
RULE = S.Law("work-rate:1/8")
SPEC = S.Spec(VALUES, RULE, S.Check((P.program_digest(Program(PAIR)),), QUERY, proof_deadline=60.0), PUBLIC,
              bytes.fromhex(ROOT))
VERDICTS = {"lean": {"of_record": True, "accepted": True, "digest": "dd" * 64, "sessions": [SESSION]}}
OUTS = list(range(16, 32))


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def args(outputs=OUTPUTS) -> SimpleNamespace:
    return SimpleNamespace(outputs=outputs)


def program() -> Program:
    return P.mixed_population_program([(PAIR, N)])


def _ports() -> dict:
    part = P.partition(program(), QUERY)
    return {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, SCHEMA, leaves, RUN)}
            for p, leaves in (("x", N), ("out", 2 * N))}


def _record() -> dict:
    roots = [{"port": p, "root": f"{i + 1:02x}" * 64, **e} for i, (p, e) in enumerate(_ports().items())]
    return R.record(program(), QUERY, S.selection_law(RULE, N), S.SCHEME, roots, {"kind": "served", "run": RUN, "row": 0},
                    work_table=S.selection_table(RULE, program(), QUERY, TABLE))


def stream_of(coins: FC.Challenger) -> S.Stream:
    return FC.Stream(SPEC, (PAIR,), budget=1, delta=0.5, challenger="a5 outputs test, challenger", coins=coins, clock=Clock())


def register(stream: S.Stream) -> S.WindowCommitmentRecord:
    return stream.register(_record(), _ports(), window_root=ROOT, counts={P.template_id(PAIR): N}, work_table=TABLE)


def outcome(stream: S.Stream, draw: S.Draw) -> tuple[S.Outcome, dict | None]:
    assert stream.deliver(draw, [SESSION]) is None
    return stream.outcome(draw, VERDICTS, {"statement": "stub"}), stream.records[-1]["audit"]


def test_an_output_class_is_refused_before_the_sample():
    assert A5.output_refusal(args()) == {"output-class-needs-a-closure-map": {"units": len(OUTS)}}
    assert A5.output_refusal(args(outputs=None)) is None and A5.output_refusal(args(outputs=[])) is None


@pytest.mark.parametrize("seed", range(3))
def test_the_class_closure_map_is_refused_and_the_work_draw_without_it_accepted(seed):
    """The stream refuses the class's closure map (every unit's closure the whole class) before any coin is read, so its
    sample of record afterwards is the one a stream on the same challenger samples with no refusal before it: k units (the
    work sampling design over one template of work 1 is the uniform k-subset), proving exactly those, and accepted, the Lean's run
    agreeing."""
    stream = stream_of(FC.Challenger(f"a5-outputs/{seed}"))
    reg = register(stream)
    with pytest.raises(S.Refused) as e:
        stream.select(reg, closure={u: OUTS for u in range(N)})
    assert e.value.verdict.codes == ("closure",)
    draw = stream.select(reg)
    fresh = stream_of(FC.Challenger(f"a5-outputs/{seed}"))
    assert draw.draw == fresh.select(register(fresh)).draw
    units = draw.draw["units"]
    assert draw.draw["law"] == "work" and len(units) == K and "closure" not in draw.draw and D.proved(draw.draw) == units
    got, audit = outcome(stream, draw)
    assert got.accepted and audit["accepted"] and "run" not in audit["checks"] and audit["verifiers"]["agree"]
