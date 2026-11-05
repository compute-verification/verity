"""a5's coins: a window's `service.Stream` samples only from the auditor's challenger's coins (`a5_coins.RankCoins`, or
`Replay` of an honest sample's bytes for the negatives), refusing any other source (`service.NO_OWN_COINS`): the verifier
samples no coins of its own. A sample on the challenger's bytes with no evidence is rejected by the Lean verifier of record's
whole run (`Run.run`) at its select (`coins`). A toy window of two templates; and the session's connect."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value as Bits
from verity.core.protocols.sampled_proofs import coins as CC, service as S
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R, verifier as V

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5_coins as AC  # noqa: E402

XOR8 = PrimitiveDefinition("A5CoinsTestXor8", 1, (("a", Bits(8)), ("b", Bits(8))), Bits(8), lambda a, b: a ^ b,
                           register=False)
PAIR = bind(CompositeDefinition("A5CoinsTestPair", 0, (), lambda S_: ((("x", Array(2, Bits(8))),), Bits(8)),
                                lambda B, S_, x: B.call(XOR8, x[0], x[1]), register=False))
TRIPLE = bind(CompositeDefinition("A5CoinsTestTriple", 0, (), lambda S_: ((("x", Array(3, Bits(8))),), Bits(8)),
                                  lambda B, S_, x: B.call(XOR8, B.call(XOR8, x[0], x[1]), x[2]), register=False))
WINDOW = (PAIR, TRIPLE)
QUERY = P.template_instances_query(WINDOW)
COUNTS = {P.template_id(PAIR): 1, P.template_id(TRIPLE): 9}
SCHEMA = "hm96-sha512/row/v2"
RUN = "r-a5-coins-test"
ROOT = "ab" * 64
SESSION = "5e" * 64
VALUES = (S.Value("x", SCHEMA, "serve"), S.Value("out", SCHEMA, "serve"))
PUBLIC = tuple(S.Public(i, "structure", r) for i, r in (
    ("population", "N, the draw's range"), ("window", "provenance"), ("leaves:x", "x's leaf count"),
    ("domain:x", "x's served domain"), ("leaves:out", "out's leaf count"), ("domain:out", "out's served domain"),
    *((f"population:{P.template_id(t)}", "a template's stratum") for t in WINDOW)))
SPEC = S.Spec(VALUES, S.Law("stratified:3"),
              S.Check(tuple(P.program_digest(Program(t)) for t in WINDOW), QUERY, proof_deadline=60.0), PUBLIC,
              bytes.fromhex(ROOT))
VERDICTS = {"lean": {"of_record": True, "accepted": True, "digest": "dd" * 64, "sessions": [SESSION]}}


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _ports() -> dict:
    part = P.partition(P.mixed_population_program([(t, COUNTS[P.template_id(t)]) for t in WINDOW]), QUERY)
    n = sum(COUNTS.values())
    return {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, SCHEMA, leaves, RUN)}
            for p, leaves in (("x", n), ("out", 2 * n))}


def _record() -> dict:
    ports = _ports()
    roots = [{"port": p, "root": f"{i + 1:02x}" * 64, **ports[p]} for i, p in enumerate(ports)]
    program = P.mixed_population_program([(t, COUNTS[P.template_id(t)]) for t in WINDOW])
    return R.record(program, QUERY, {"law": "stratified", "k": 3}, S.SCHEME, roots, {"kind": "served", "run": RUN, "row": 0})


def stream_of(coins) -> S.Stream:
    return S.Stream(SPEC, WINDOW, budget=1, delta=0.5, challenger="a5 test, challenger", coins=coins, clock=Clock())


def register(stream: S.Stream) -> S.WindowCommitmentRecord:
    return stream.register(_record(), _ports(), window_root=ROOT, counts=COUNTS)


def test_a5s_stream_draws_on_no_coins_of_its_own():
    """The verifier's own coins (the Lean's `IO.getRandomBytes`, `verifier.coins`) or any callable but a challenger's are
    refused before the stream exists; the challenger's (`RankCoins`, `Replay`) are taken."""
    for own in (V.coins, lambda n: bytes([0x5a]) * n, None):
        with pytest.raises(ValueError, match="no coins of its own"):
            stream_of(own)
    assert isinstance(AC.Replay(b""), CC.ChallengerCoins) and issubclass(AC.RankCoins, CC.ChallengerCoins)
    assert stream_of(AC.Replay(b"")).digest


def test_the_challengers_bytes_with_no_evidence_are_rejected_by_the_lean_run():
    """Bytes passed off as the challenger's with no log to show they are its: the outcome fails `run`, the Lean refusing
    at its select (`coins`), and Python's `coins_reason` refuses the missing evidence."""
    stream = stream_of(AC.Replay(bytes([0x5a]) * 4096))
    draw = stream.select(register(stream))
    assert stream.deliver(draw, [SESSION]) is None
    got = stream.outcome(draw, VERDICTS, {"statement": "stub"}, coins=None)
    audit = stream.records[-1]["audit"]
    assert not got.accepted and not audit["checks"]["run"]["ok"]
    side = AC.lean_side(audit)
    assert (side["step"], side["codes"]) == AC.SELECT_COINS
    assert CC.coins_reason(None, ROOT, bytes.fromhex(draw.registration.digest), b"\x5a", AC.lms_verify) is not None


def test_the_replay_reads_no_more_than_the_honest_draw():
    replay = AC.Replay(b"\x01\x02\x03")
    assert replay(2) == b"\x01\x02"
    with pytest.raises(AC.Ch.CoinError, match="past the honest draw"):
        replay(2)


def _clock_and_sleep():
    now, slept = [0.0], []

    def sleep(s):
        slept.append(s)
        now[0] += s

    return (lambda: now[0]), sleep, slept


def test_a_session_whose_tunnel_listens_late_is_connected_once_it_listens():
    clock, sleep, slept = _clock_and_sleep()
    tries = []

    def connect(address, secret):
        tries.append((address, secret))
        if len(tries) < 3:
            raise ConnectionRefusedError(111, "Connection refused")
        return "client"

    assert AC.connect_when_listening("127.0.0.1:7811", b"s", connect, 120, clock, sleep) == "client"
    assert tries == [("127.0.0.1:7811", b"s")] * 3 and slept == [0.5, 0.5]


def test_a_session_that_never_listens_or_fails_otherwise_raises():
    clock, sleep, slept = _clock_and_sleep()

    def refused(address, secret):
        raise ConnectionRefusedError(111, "Connection refused")

    with pytest.raises(AC.Ch.CoinError, match="nothing listened"):
        AC.connect_when_listening("127.0.0.1:7811", None, refused, 2, clock, sleep)
    assert len(slept) == 4

    def hello_refused(address, secret):
        raise AC.Ch.CoinError("the challenger refused the hello")

    clock, sleep, slept = _clock_and_sleep()
    with pytest.raises(AC.Ch.CoinError):
        AC.connect_when_listening("127.0.0.1:7811", None, hello_refused, 120, clock, sleep)
    assert slept == []
