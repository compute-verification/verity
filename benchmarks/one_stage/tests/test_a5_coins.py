"""a5's draw on the verifier's own coins (`a5.py --coins verifier`, `a5_coins.VerifierCoins`): a window's `service.Stream` on
them has the verifier as its coin source, and its outcome, with no coins' evidence, is accepted with the Lean verifier of
record's whole run (`Run.run`, `coin_source` verifier) agreeing. `a5_coins.verifier_negatives` refuses each negative for its
own reason: the bytes passed off as the challenger's, or as the verifier's with evidence, at the Lean run's select (`coins`);
a redraw (`drawn`) and a draw before the registration (`unknown`), with no coin read. A toy window of two templates."""

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


def fixed(byte: int):
    """A deterministic stand-in for the Lean's OS coins: `n` copies of `byte`, read through `VerifierCoins` as the run
    reads `verifier.coins`."""
    return lambda n: bytes([byte]) * n


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
    name = "a5 test, challenger" if isinstance(coins, CC.ChallengerCoins) else "a5 test, verifier"
    return S.Stream(SPEC, WINDOW, budget=1, delta=0.5, challenger=name, coins=coins, clock=Clock())


def register(stream: S.Stream) -> S.WindowCommitmentRecord:
    return stream.register(_record(), _ports(), window_root=ROOT, counts=COUNTS)


def outcome(stream: S.Stream, draw: S.Draw, evidence: dict | None) -> tuple[S.Outcome, dict | None]:
    assert stream.deliver(draw, [SESSION]) is None
    return stream.outcome(draw, VERDICTS, {"statement": "stub"}, coins=evidence), stream.records[-1]["audit"]


def honest(byte: int = 0x5a) -> tuple[S.Stream, AC.VerifierCoins, S.Draw, S.Outcome, dict]:
    coins = AC.VerifierCoins(fixed(byte))
    stream = stream_of(coins)
    draw = stream.select(register(stream))
    got, audit = outcome(stream, draw, None)
    return stream, coins, draw, got, audit


def test_verifier_coins_default_to_the_leans_os_coins():
    """With no source, `VerifierCoins` reads the service's default coins, the Lean's `IO.getRandomBytes`
    (`verifier.coins`), and is no `ChallengerCoins`."""
    coins = AC.VerifierCoins()
    assert coins.source is V.coins and not isinstance(coins, CC.ChallengerCoins)


def test_a_window_drawn_on_the_verifiers_coins_is_accepted_with_the_lean_run_agreeing():
    """The stream's coin source is the verifier; its select reads the coins only after the registration (a stratum each,
    PAIR's by its floor), `read` holds exactly the bytes the draw read, and the outcome with no evidence is accepted, the
    Lean's `Run.run` agreeing (no `run` check failed)."""
    stream, coins, draw, got, audit = honest()
    assert stream.coin_source == "verifier"
    assert draw.draw["law"] == "stratified" and len(draw.units) == 4 and 0 in draw.units
    assert bytes(coins.read) and bytes(coins.read) == stream._entries[draw.registration.order].read
    assert got.accepted and audit["accepted"] and "run" not in audit["checks"]
    assert audit["verifiers"]["agree"]


def test_a_verifier_draw_with_any_evidence_is_rejected_by_the_lean_run():
    """`coins` is None on the verifier's own coins: an outcome given evidence fails `run`, the Lean refusing at its select."""
    coins = AC.VerifierCoins(fixed(0x5a))
    stream = stream_of(coins)
    draw = stream.select(register(stream))
    got, audit = outcome(stream, draw, AC.FOREIGN_EVIDENCE)
    assert not got.accepted and not audit["checks"]["run"]["ok"]
    side = AC.lean_side(audit)
    assert (side["step"], side["codes"]) == AC.SELECT_COINS


def test_every_verifier_coin_negative_is_refused_for_its_own_reason():
    stream, coins, draw, got, _ = honest()
    assert got.accepted
    neg = AC.verifier_negatives(stream_of, register, stream, coins, draw, outcome, ROOT)
    assert sorted(neg) == ["verifier-coins-before-the-registration", "verifier-coins-claimed-as-challenger",
                           "verifier-coins-redraw", "verifier-coins-with-evidence"]
    assert all(n["refused"] for n in neg.values()), neg
    claimed, evidence = neg["verifier-coins-claimed-as-challenger"], neg["verifier-coins-with-evidence"]
    assert claimed["source"] == "challenger" and claimed["python"] == CC.COINS_FORM and claimed["accepted"] is False
    assert evidence["source"] == "verifier" and evidence["accepted"] is False and "python" not in evidence
    for n in (claimed, evidence):
        assert (n["lean"]["step"], n["lean"]["codes"]) == AC.SELECT_COINS
    assert neg["verifier-coins-redraw"] == {"refused": True, "stream": ["drawn"], "coins_read": 0}
    assert neg["verifier-coins-before-the-registration"] == {"refused": True, "stream": ["unknown"], "coins_read": 0}


def test_a_negative_the_stream_accepts_is_not_counted_refused(monkeypatch):
    """The refusal is the stream's, not the label's: where the replayed draw is the verifier's own with no evidence (the
    honest run again), neither outcome negative counts as refused, even with the Lean's side read as refusing at select."""
    stream, coins, draw, _, _ = honest()
    monkeypatch.setattr(AC, "FOREIGN_EVIDENCE", None)
    monkeypatch.setattr(AC, "Replay", AC.OwnReplay)
    monkeypatch.setattr(AC, "lean_side", lambda audit: {"step": AC.SELECT_COINS[0], "codes": AC.SELECT_COINS[1]})
    neg = AC.verifier_negatives(stream_of, register, stream, coins, draw, outcome, ROOT)
    for name in ("verifier-coins-claimed-as-challenger", "verifier-coins-with-evidence"):
        assert neg[name]["accepted"] is True and neg[name]["source"] == "verifier" and not neg[name]["refused"], neg[name]


def test_an_outcome_negative_counts_refused_only_where_its_own_check_refuses(monkeypatch):
    """A rejected outcome is not enough: a negative the Lean's run refuses elsewhere than its select's `coins` is not
    refused, nor one passed off as the challenger's whose evidence Python's `coins_reason` would take."""
    stream, coins, draw, _, _ = honest()
    monkeypatch.setattr(AC, "SELECT_COINS", ("outcome", ["coins"]))
    neg = AC.verifier_negatives(stream_of, register, stream, coins, draw, outcome, ROOT)
    for name in ("verifier-coins-claimed-as-challenger", "verifier-coins-with-evidence"):
        assert neg[name]["accepted"] is False and not neg[name]["refused"], neg[name]
    monkeypatch.undo()
    stream, coins, draw, _, _ = honest()
    monkeypatch.setattr(CC, "coins_reason", lambda *_: None)
    neg = AC.verifier_negatives(stream_of, register, stream, coins, draw, outcome, ROOT)
    assert neg["verifier-coins-claimed-as-challenger"]["python"] is None
    assert not neg["verifier-coins-claimed-as-challenger"]["refused"]
    assert neg["verifier-coins-with-evidence"]["refused"]


def test_a_select_counts_refused_only_for_its_own_code_with_no_coin_read():
    def refuse(code):
        raise S.Refused(S.Verdict(False, (code,), {code: None}))

    read = [0]

    def reads_then_refuses():
        read[0] += 1
        refuse("drawn")

    assert AC.stream_refusal(lambda: refuse("drawn"), "drawn", lambda: read[0])["refused"]
    assert not AC.stream_refusal(lambda: refuse("unknown"), "drawn", lambda: read[0])["refused"]
    assert AC.stream_refusal(reads_then_refuses, "drawn", lambda: read[0]) == {"refused": False, "stream": ["drawn"],
                                                                             "coins_read": 1}
    assert AC.stream_refusal(lambda: None, "drawn", lambda: read[0]) == {"refused": False, "stream": None, "coins_read": 0}


def test_the_replay_reads_no_more_than_the_honest_draw():
    replay = AC.OwnReplay(b"\x01\x02\x03")
    assert replay(2) == b"\x01\x02" and replay.at == 2
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
