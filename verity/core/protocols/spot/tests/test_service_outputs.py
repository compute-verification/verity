"""A stream's records chained by output values (`registration.check`'s `outputs`): a later registration's registered value
is a port an earlier record committed, so the state one window leaves is the state a later one reads, under the earlier
record's root, domain and hiding leaves. The stream hands R7 the record each output value names however long its log, and
the run of record (`Run.run`) checks the output values the registration did, in one stage and in a two-stage design's
interior registration."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

import verity.ml.circuits.ops.scalar  # noqa: F401
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.spot import service as S
from verity.core.protocols.spot.one_stage import draw as D, partition as P, registration as R, verifier as V

import fake_challenger as FC

XOR8 = PrimitiveDefinition("OutputsTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
#: A proof block: a fresh input `x` and the `state` an earlier record left.
CARRY = bind(CompositeDefinition(
    "OutputsTestCarry", 0, (), lambda S_: ((("x", Array(2, Bits(8))), ("state", Array(2, Bits(8)))), Bits(8)),
    lambda B, S_, x, s: B.call(XOR8, B.call(XOR8, x[0], s[0]), B.call(XOR8, x[1], s[1])), register=False))
#: A replay block: three proof blocks, instances of CARRY.
BLOCK = bind(CompositeDefinition(
    "OutputsTestBlock", 0, (),
    lambda S_: ((("x", Array(3, Array(2, Bits(8)))), ("state", Array(3, Array(2, Bits(8))))), Array(3, Bits(8))),
    lambda B, S_, x, s: B.batch(CARRY, x, s, axes=(0, 0)), register=False))
QUERY = P.template_instance_query(CARRY)
GRAIN = P.template_instance_query(BLOCK)
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-outputs-test"
SESSION = "5e" * 64
DELTA = 2.0 ** -40
LAW = {"law": "subset", "k": 2}
N = 6
#: `next`: the state a window leaves, which a later window reads as its registered `state`.
PORTS = ("x", "out", "next")
#: A two-stage design's boundary values (`carry`: the state its replay blocks leave) and interior values.
BOUNDARY, INTERIOR = ("bx", "bout", "carry"), ("x", "out")
N_RHO, NB, Q, K = 3, 4, 2, 1
STATE = S.Value("state", None, "serve", registered=True)


def _public(ports) -> tuple[S.Public, ...]:
    return (S.Public("population", "structure", "N, the sample's range"),
            S.Public("window", "structure", "provenance: served, on test prompts"),
            *(S.Public(f"{k}:{v}", "structure", f"{v}'s {k}") for v in ports for k in ("leaves", "domain")),
            *(S.Public(f"{k}:state", "structure", f"the state's {k}") for k in ("leaves", "words", "domain")))


def _spec() -> S.Spec:
    return S.Spec((*(S.Value(v, SCHEMA, "serve") for v in PORTS), STATE), S.Law("rate:1/3"),
                  S.Check(P.program_digest(Program(CARRY)), QUERY, proof_deadline=60.0), _public(PORTS),
                  b"outputs-test")


def _two_stage_spec() -> S.Spec:
    values = (*(S.Value(v, SCHEMA, "serve") for v in BOUNDARY), *(S.Value(v, SCHEMA, "interior") for v in INTERIOR), STATE)
    return S.Spec(values, S.TwoStage(GRAIN, "rate:1/2", K),
                  S.Check(P.program_digest(Program(CARRY)), QUERY, proof_deadline=60.0), _public(BOUNDARY + INTERIOR),
                  b"outputs-two-stage-test")


def _stream(spec: S.Spec | None = None, **kwargs) -> S.Stream:
    return FC.Stream(spec or _spec(), CARRY, **{"budget": 10, "delta": DELTA, "challenger": "challenger-test",
                                                "clock": lambda: 0.0, **kwargs})


def _record(program: Program, query, sizes: Mapping[str, int], law, row: int, previous: str | None, registered=(),
            **window) -> tuple[dict, dict]:
    part = P.partition(program, query)
    ports = {v: {"leaves": n, "domain": R.window_domain(part["program"], P.digest(part), v, SCHEMA, n, RUN)}
             for v, n in sizes.items()}
    roots = [{"port": v, "root": f"{row:02x}{i + 1:02x}" * 32, **ports[v]} for i, v in enumerate(sizes)]
    return R.record(program, query, law, S.SCHEME, roots, {"kind": "served", "run": RUN, "row": row, **window},
                    previous=previous, registered=list(registered)), ports


def _window(row: int, previous: str | None, registered=()) -> tuple[dict, dict]:
    return _record(P.population_program(CARRY, N), QUERY, dict.fromkeys(PORTS, N), LAW, row, previous, registered)


def _left(rec: Mapping, port: str = "next") -> dict:
    """The registered entry of `state` that is `rec`'s port `port`: its root and domain, its leaves the state's rows."""
    e = next(r for r in rec["roots"] if r["port"] == port)
    return {"value": "state", "root": e["root"], "leaves": e["leaves"], "words": 2, "domain": e["domain"]}


def _lean(registered: Mapping[str, str] | None = None, checked: Mapping | None = None) -> dict:
    """The verdict of record, reporting the registered roots its session's reads opened against, and naming what it
    checked of the session of second-stage statement `checked` (`fake_challenger.checked`)."""
    lean = {"of_record": True, "accepted": True, "digest": "dd" * 64, "sessions": [SESSION]}
    if registered:
        lean["registered"] = {v: {"value": v, "root": r} for v, r in registered.items()}
    if checked is not None:
        lean |= FC.checked(checked)
    return {"lean": lean}


def _refused(f) -> tuple[str, ...]:
    with pytest.raises(S.Refused) as e:
        f()
    return e.value.verdict.codes


def _close(stream: S.Stream, reg: S.WindowCommitmentRecord, registered: Mapping[str, str] | None = None) -> S.Outcome:
    draw = stream.select(reg)
    assert stream.deliver(draw, [SESSION]) is None
    return stream.outcome(draw, _lean(registered), {"statement": "stub"})


def _three_windows(stream: S.Stream) -> str:
    """Windows 0 to 2, each accepted; the last one's digest."""
    prev = None
    for row in range(3):
        reg = stream.register(*_window(row, prev))
        assert _close(stream, reg).accepted
        prev = reg.digest
    return prev


def test_a_window_reads_the_state_an_earlier_window_left():
    """Window 0 commits the state it leaves as its port `next`; windows 1 and 2 follow it; window 3 registers that port as
    its `state`. Its check gets window 0's record though it is neither the last record nor a value's first registration,
    the verdict is the whole log's, and its outcome is accepted, the run of record holding the output value as R7 did.
    Window 0's record left out of the log R7 reads is `R7-output`, as is a record the log doesn't hold; a root other than
    the port's is `R7-held`; and the value registered with no output named is refused under its own domain."""
    stream = _stream()
    prev = _three_windows(stream)
    w0 = stream.log[0]
    out = {"state": {"record": R.digest(w0), "port": "next"}}
    cand, ports = _window(3, prev, [_left(w0)])

    def check(log: list) -> V.Verdict:
        return V.check(cand, P.population_program(CARRY, N), QUERY, LAW, S.SCHEME, ports, None, (), log, None, None,
                       {"state": 2}, window_root=FC.WINDOW["root"], outputs=out)

    assert stream._earlier(cand) == [stream.log[-1]] and check(stream._earlier(cand)).codes[0] == "R7-output"
    assert stream._earlier(cand, None, out) == [w0, stream.log[-1]]
    assert check(stream._earlier(cand, None, out)) == check(stream.log) and check(stream.log).ok

    wrong, _ = _window(3, prev, [{**_left(w0), "root": "ee" * 64}])
    assert _refused(lambda: stream.register(wrong, ports, registered={"state": 2}, outputs=out)) == ("R7-held",)
    assert "R7-domain" in _refused(lambda: stream.register(cand, ports, registered={"state": 2}))
    gone = {"state": {"record": "ee" * 64, "port": "next"}}
    assert _refused(lambda: stream.register(cand, ports, registered={"state": 2}, outputs=gone))[0] == "R7-output"

    reg = stream.register(cand, ports, registered={"state": 2}, outputs=out)
    assert stream._entries[-1].copies[-1] == out
    outcome = _close(stream, reg, {"state": _left(w0)["root"]})
    assert outcome.accepted and outcome.verdict.codes == () and stream.records[-1]["audit"]["accepted"]


def test_the_run_of_record_reads_the_output_values(monkeypatch):
    """The stream's run of record gets the output values R7 took (`verifier.run`'s `value_outputs`, in the Lean's
    params): a run that drops them refuses the registration the stream accepted, and the outcome with it."""
    stream = _stream()
    prev = _three_windows(stream)
    w0 = stream.log[0]
    cand, ports = _window(3, prev, [_left(w0)])
    reg = stream.register(cand, ports, registered={"state": 2}, outputs={"state": {"record": R.digest(w0), "port": "next"}})
    run = V.run
    monkeypatch.setattr(S.V, "run", lambda *a, value_outputs=None, **kw: run(*a, **kw))
    outcome = _close(stream, reg, {"state": _left(w0)["root"]})
    assert not outcome.accepted and stream.records[-1]["audit"]["checks"]["run"]["codes"] == [V.DISAGREE]


def test_an_interior_registration_reads_a_port_of_its_boundary():
    """Under a two-stage design the interior registration's registered `state` is the boundary registration's port
    `carry`, the state its replay blocks leave: R7 takes it in the interior registration, the interior's copies carry
    it to the run of record, which agrees on the outcome binding both registrations. Its one failure is round 2's
    (`second-read`): the replay block's Definition links CARRY's `x`, `state` and output to the boundary's ports of
    them, which this boundary doesn't register, and its session, which the verdict of record names as it checked it,
    reads `state` from the registered value instead. Registered with no output named, it is refused under its own
    domain."""
    stream = _stream(_two_stage_spec(), replay=BLOCK)
    assert stream.n_rho == N_RHO
    m = Q * N_RHO
    boundary, bports = _record(P.population_program(BLOCK, NB), GRAIN, {"bx": NB, "bout": NB, "carry": m},
                               {"law": "subset", "k": Q}, 0, None)
    reg = stream.register(boundary, bports)
    first = stream.select(reg)
    out = {"state": {"record": reg.digest, "port": "carry"}}
    interior, iports = _record(P.population_program(CARRY, m), QUERY, dict.fromkeys(INTERIOR, m), {"law": "subset", "k": m},
                               0, reg.digest, [_left(boundary, "carry")], interior=True,
                               replay_sample=D.proved(first.draw))
    assert "R7-domain" in _refused(lambda: stream.register_interiors(first, interior, iports, registered={"state": 2}))
    ireg = stream.register_interiors(first, interior, iports, registered={"state": 2}, outputs=out)
    assert stream._entries[reg.order].interior_copies[-1] == out
    second = stream.select(ireg)
    assert stream.deliver(second, [SESSION]) is None
    (honest,) = stream.statements(second)
    outcome = stream.outcome(second, _lean({"state": _left(boundary, "carry")["root"]}, honest), {"statement": "stub"},
                             statements=[honest])
    checks = stream.records[-1]["audit"]["checks"]
    assert not outcome.accepted and outcome.verdict.detail["reject"]["checks"] == ["statements"]
    assert checks["statements"]["codes"] == ["second-read"] and "run" not in checks
