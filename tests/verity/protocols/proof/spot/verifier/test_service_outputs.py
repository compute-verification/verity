"""A stream's records chained by output values (`registration.check`'s `outputs`): a later registration's registered value
is a port an earlier record committed, so the state one window leaves is the state a later one reads, under the earlier
record's root, domain and hiding leaves. The stream hands R7 the record each output value names however long its log, and
the run of record (`Run.run`) checks the output values the docket did, in one stage and in a two-stage design's
interior docket. And a stream's certified entries (`registration.check`'s `certified`): a registered value that is
none of the program's parameters, which the stream's check and its run of record hold to the entry the verifier built
for it, and an interior docket takes none."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from verity.commitments import docket_digest as DD
from verity.circuits import population as PN
import verity.circuits.ops.scalar  # noqa: F401
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.types import Array, Value as Bits
from verity.protocols.proof.spot.verifier import service as S
from verity.protocols.proof.spot.verifier.one_stage import partition as P, docket as R, verifier as V
from verity.protocols.proof.spot.prover import docket as DK

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
SAMPLING_DESIGN = {"law": "subset", "k": 2}
N = 6
#: `next`: the state a window leaves, which a later window reads as its registered `state`.
PORTS = ("x", "out", "next")
#: A two-stage design's boundary values (`carry`: the state its replay blocks leave) and interior values.
BOUNDARY, INTERIOR = ("bx", "bout", "carry"), ("x", "out")
N_RHO, NB, Q, K = 3, 4, 2, 1
STATE = S.Value("state", None, "serve", registered=True)
#: A link's certified rows: a registered value that is none of the program's parameters, with the entry and root the
#: verifier built for it from the docket it read under the deployment's key.
ROWS = S.Value("link-0/in/7", None, "serve", registered=True)
CERTIFIED = {ROWS.name: {"root": "c5" * 64, "leaves": 3, "words": 64, "domain": "c4" * 64}}
ROWS_ENTRY = {"value": ROWS.name, **CERTIFIED[ROWS.name]}


def _public(ports, registered=(STATE,)) -> tuple[S.Public, ...]:
    return (S.Public("population", "structure", "N, the sample's range"),
            S.Public("window", "structure", "provenance: served, on test prompts"),
            *(S.Public(f"{k}:{v}", "structure", f"{v}'s {k}") for v in ports for k in ("leaves", "domain")),
            *(S.Public(f"{k}:{v.name}", "structure", f"the {v.name}'s {k}") for v in registered
              for k in ("leaves", "words", "domain")))


def _spec(registered=(STATE,)) -> S.Spec:
    return S.Spec((*(S.Value(v, SCHEMA, "serve") for v in PORTS), *registered), S.SamplingDesign("rate:1/3"),
                  S.Check(P.program_digest(Program(CARRY)), QUERY, proof_deadline=60.0), _public(PORTS, registered),
                  b"outputs-test")


def _stream(spec: S.Spec | None = None, **kwargs) -> S.Stream:
    return FC.Stream(spec or _spec(), CARRY, **{"budget": 10, "delta": DELTA, "challenger": "challenger-test",
                                                "clock": lambda: 0.0, **kwargs})


def _record(program: Program, query, sizes: Mapping[str, int], sampling_design, row: int, previous: str | None, registered=(),
            **window) -> tuple[dict, dict]:
    part = P.partition(program, query)
    ports = {v: {"leaves": n, "domain": R.window_domain(part["program"], P.digest(part), v, SCHEMA, n, RUN)}
             for v, n in sizes.items()}
    roots = [{"port": v, "root": f"{row:02x}{i + 1:02x}" * 32, **ports[v]} for i, v in enumerate(sizes)]
    return DK.record(program, query, sampling_design, S.SCHEME, roots, {"kind": "served", "run": RUN, "row": row, **window},
                    previous=previous, registered=list(registered)), ports


def _window(row: int, previous: str | None, registered=()) -> tuple[dict, dict]:
    return _record(PN.population_program(CARRY, N), QUERY, dict.fromkeys(PORTS, N), SAMPLING_DESIGN, row, previous, registered)


def _left(rec: Mapping, port: str = "next") -> dict:
    """The registered entry of `state` that is `rec`'s port `port`: its root and domain, its leaves the state's rows."""
    e = next(r for r in rec["roots"] if r["port"] == port)
    return {"value": "state", "root": e["root"], "leaves": e["leaves"], "words": 2, "domain": e["domain"]}


def _lean(registered: Mapping[str, str] | None = None) -> dict:
    """The verdict of record, reporting the registered roots its session's reads opened against."""
    lean = {"of_record": True, "accepted": True, "digest": "dd" * 64, "sessions": [SESSION]}
    if registered:
        lean["registered"] = {v: {"value": v, "root": r} for v, r in registered.items()}
    return {"lean": lean}


def _refused(f) -> tuple[str, ...]:
    with pytest.raises(S.Refused) as e:
        f()
    return e.value.verdict.codes


def _close(stream: S.Stream, reg: S.WindowCommitmentRecord, registered: Mapping[str, str] | None = None,
           verdicts: Mapping | None = None) -> S.Outcome:
    draw = stream.select(reg)
    assert stream.deliver(draw, [SESSION]) is None
    return stream.outcome(draw, _lean(registered) if verdicts is None else verdicts, {"statement": "stub"})


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
    its `state`. Its check gets window 0's record though it is neither the last record nor a value's first docket,
    the verdict is the whole log's, and its outcome is accepted, the run of record holding the output value as R7 did.
    Window 0's record left out of the log R7 reads is `R7-output`, as is a record the log doesn't hold; a root other than
    the port's is `R7-held`; and the value registered with no output named is refused under its own domain."""
    stream = _stream()
    prev = _three_windows(stream)
    w0 = stream.log[0]
    out = {"state": {"record": DD.digest(w0), "port": "next"}}
    cand, ports = _window(3, prev, [_left(w0)])

    def check(log: list) -> V.Verdict:
        return V.check(cand, PN.population_program(CARRY, N), QUERY, SAMPLING_DESIGN, S.SCHEME, ports, None, (), log, None, None,
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
    params): a run that drops them refuses the docket the stream accepted, and the outcome with it."""
    stream = _stream()
    prev = _three_windows(stream)
    w0 = stream.log[0]
    cand, ports = _window(3, prev, [_left(w0)])
    reg = stream.register(cand, ports, registered={"state": 2}, outputs={"state": {"record": DD.digest(w0), "port": "next"}})
    run = V.run
    monkeypatch.setattr(S.V, "run", lambda *a, value_outputs=None, **kw: run(*a, **kw))
    outcome = _close(stream, reg, {"state": _left(w0)["root"]})
    assert not outcome.accepted and stream.records[-1]["audit"]["checks"]["run"]["codes"] == [V.DISAGREE]


def test_a_stream_takes_the_verifiers_certified_entries():
    """A window's record registers a link's certified rows, a value that is none of the program's parameters, and no
    other: the stream refuses it without the verifier's certified entries (`keys`: expecting nothing registered, it
    expects a v1 record), and with an entry at another root (`R7-certified`) or of other leaves (`R7-leaves`). With the
    entries it accepts the record and keeps them, its run of record reads them again, and the outcome is accepted. A
    later window registers the state an earlier one left beside the rows: its registered values are the verifier's own
    and the certified one, and without the entries it is refused (`R7-values`)."""
    stream = _stream(_spec((STATE, ROWS)))
    rows = CERTIFIED[ROWS.name]
    cand, ports = _window(0, None, [ROWS_ENTRY])
    assert _refused(lambda: stream.register(cand, ports)) == ("keys",)
    assert _refused(lambda: stream.register(cand, ports, certified={ROWS.name: {**rows, "root": "c6" * 64}})) == (
        "R7-certified",)
    assert _refused(lambda: stream.register(cand, ports, certified={ROWS.name: {**rows, "leaves": 4}})) == ("R7-leaves",)
    reg = stream.register(cand, ports, certified=CERTIFIED)
    assert stream._entries[-1].certified == CERTIFIED
    outcome = _close(stream, reg, {ROWS.name: rows["root"]})
    assert outcome.accepted and outcome.verdict.codes == () and stream.records[-1]["audit"]["accepted"]

    w0 = stream.log[0]
    out = {"state": {"record": DD.digest(w0), "port": "next"}}
    later, ports = _window(1, reg.digest, [_left(w0), ROWS_ENTRY])
    assert _refused(lambda: stream.register(later, ports, registered={"state": 2}, outputs=out)) == ("R7-values",)
    reg = stream.register(later, ports, registered={"state": 2}, outputs=out, certified=CERTIFIED)
    outcome = _close(stream, reg, {"state": _left(w0)["root"], ROWS.name: rows["root"]})
    assert outcome.accepted and outcome.verdict.codes == () and stream.records[-1]["audit"]["accepted"]


def test_the_run_of_record_reads_the_certified_entries(monkeypatch):
    """The stream's run of record gets the certified entries R7 took (`verifier.run`'s `certified`, in the Lean's
    params): a run that drops them refuses the docket the stream accepted, and the outcome with it."""
    stream = _stream(_spec((STATE, ROWS)))
    cand, ports = _window(0, None, [ROWS_ENTRY])
    reg = stream.register(cand, ports, certified=CERTIFIED)
    run = V.run
    monkeypatch.setattr(S.V, "run", lambda *a, certified=None, **kw: run(*a, **kw))
    outcome = _close(stream, reg, {ROWS.name: CERTIFIED[ROWS.name]["root"]})
    assert not outcome.accepted and stream.records[-1]["audit"]["checks"]["run"]["codes"] == [V.DISAGREE]


def test_a_stream_holds_a_certified_value_to_a_reported_read():
    """A window's record registers a link's certified rows, and the stream took the verifier's entry for them. Its
    outcome is refused by the audit record's check `certified`, naming the rows, when the verdict of record reports no
    read of them, or reports their port under a value the record does not list and at another root; the run of record
    is the Lean's own and agrees (no `run` failure). With a read of the rows at their root the outcome is accepted
    (`test_a_stream_takes_the_verifiers_certified_entries`)."""
    for reads in (None, {ROWS.name: {"value": "unlisted", "root": "ab" * 64}}):
        stream = _stream(_spec((STATE, ROWS)))
        cand, ports = _window(0, None, [ROWS_ENTRY])
        reg = stream.register(cand, ports, certified=CERTIFIED)
        verdicts = _lean()
        if reads is not None:
            verdicts["lean"]["registered"] = reads
        outcome = _close(stream, reg, verdicts=verdicts)
        checks = stream.records[-1]["audit"]["checks"]
        assert not outcome.accepted and not stream.records[-1]["audit"]["accepted"]
        assert checks["certified"] == {"ok": False, "codes": ["certified"],
                                       "detail": {"certified": [ROWS.name], "unread": [ROWS.name]}}
        assert "run" not in checks and "registered" not in checks
