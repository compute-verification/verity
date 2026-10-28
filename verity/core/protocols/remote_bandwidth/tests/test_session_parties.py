"""The bandwidth session as two parties of the deployment layer (`session_parties`), run in simulated time: the verifier
makes the case and answer `period.session` makes on the same coins and clock readings, times each root by its arrival,
and replays from its own transcript."""

from __future__ import annotations

import functools
import json

import pytest
from test_verifier_party import STORE, Entropy, dropping

from verity.core.deploy import sources as S
from verity.core.deploy.runtime import ProtocolDef, run
from verity.core.protocols.remote_bandwidth import audit
from verity.core.protocols.remote_bandwidth import period as P
from verity.core.protocols.remote_bandwidth import session_parties as SP

SEED = b"remote bandwidth session parties"
BOTH = [SP.VERIFIER, SP.PROVER]
MS = 1_000_000


def simulated(state, request=None, cheat=None, prover=SP.Prover, verifier=SP.Verifier, **kw):
    store = P.Local("n0", STORE, cheat=cheat, entropy=Entropy(b"salts"))
    proto = ProtocolDef(SP.MANIFEST, {SP.VERIFIER: verifier, SP.PROVER: prover})
    return run({"protocol": SP.ID, "params": request or SP.request_of("n0")}, BOTH, "simulated", state,
               protocols={SP.ID: proto}, seed=SEED, authenticate=False, capabilities={SP.PROVER: {"store": store}}, **kw)


def events(case) -> list[dict]:
    return json.loads(bytes.fromhex(case["evidence"]))["events"]


class Readings:
    """A clock that gives `readings` in turn, and a wait that doesn't move it."""

    def __init__(self, readings) -> None:
        self.readings = iter(readings)

    def __call__(self) -> int:
        return next(self.readings)

    def sleep_until(self, clock, t: int) -> None:
        pass


WINDOWS = [("start", 0), ("receive", 0), ("boundary", 100 * MS), ("receive", 100 * MS), ("boundary", 200 * MS),
           ("close", 200 * MS), ("verify", None)]
"""The log in simulated time, where a frame takes no time to cross: each boundary window_ns after its window's challenge,
window 0 closed once window 1's root is in, and the last window just past its deadline (its boundary plus 500 ms)."""


@pytest.mark.parametrize("cheat, verdict, log", [(None, "accept", WINDOWS + [("close", 700 * MS + 1), ("verify", None)]),
                                                 (dropping, "reject", WINDOWS)], ids=["honest", "dropping"])
def test_the_parties_session_is_the_periods(tmp_path, monkeypatch, cheat, verdict, log) -> None:
    """The verifier draws the period's secret and the registration's nonce from its coins: on those, and on the clock
    readings its log holds, `period.session` makes the same answer and case."""
    r = simulated(tmp_path, cheat=cheat)
    assert r.status == "completed", r.outcomes
    out = r.outcomes[SP.VERIFIER]
    assert out["refused"] is None and out["outcome"]["answer"]["verdict"] == verdict
    assert r.outcomes[SP.PROVER] == {"outcome": {"passes": 2}, "refused": None}
    got = out["outcome"]["case"]
    assert [(e["kind"], e.get("now")) for e in events(got)] == log

    drawn = S.seeded(SEED, SP.VERIFIER, "coins").draw(SP.SECRET_BYTES + audit.NONCE_BYTES)
    coins, nonce = drawn[:SP.SECRET_BYTES], drawn[SP.SECRET_BYTES:]
    monkeypatch.setattr(audit, "register", functools.partial(audit.register, nonce=nonce))
    c = Readings([got["params"]["not_before_ns"]] + [now or 0 for _, now in log])
    period = P.session("n0", P.Local("n0", STORE, cheat=cheat, entropy=Entropy(b"salts")), coins, clock=c,
                       sleep_until=c.sleep_until)
    assert (period.answer, period.case) == (out["outcome"]["answer"], got)


def test_a_root_is_timed_by_its_arrival_and_past_its_deadline_is_late(tmp_path) -> None:
    """Window 1's root, held back 700 ms on its way, arrives at 800 ms, past the window's deadline (its boundary at
    200 ms plus 500 ms): the verifier logs it at its arrival and the session rejects it `late`."""
    held = {(SP.PROVER, SP.VERIFIER): {"seed": 0, "after": 2, "delay": 1.0, "delay_ns": 700 * MS}}
    r = simulated(tmp_path, adversaries=held)
    out = r.outcomes[SP.VERIFIER]
    assert out["refused"] is None
    answer = out["outcome"]["answer"]
    assert answer["verdict"] == "reject" and "late" in answer["reason"], answer
    late = [e for e in events(out["outcome"]["case"]) if e["kind"] == "receive"][-1]
    assert (late["t"], late["i"], late["now"]) == (1, 0, 800 * MS)


def test_the_verifier_replays_from_its_transcript(tmp_path) -> None:
    """Each root's arrival is in the verifier's transcript, so a replay of it makes the same case (at a few samples,
    whose openings the transcript keeps whole)."""
    first = simulated(tmp_path, SP.request_of("n0", samples=4))
    assert first.outcomes[SP.VERIFIER]["outcome"]["answer"]["verdict"] == "accept", first.outcomes
    again = run({"protocol": SP.ID, "params": SP.request_of("n0", samples=4)}, [SP.VERIFIER], "replay", None,
                protocols=SP.PROTOCOLS, replay_dir=tmp_path / "runs" / "1")
    assert (again.status, again.divergence) == ("completed", None)
    assert again.outcomes[SP.VERIFIER] == first.outcomes[SP.VERIFIER]


class Twice(SP.Prover):
    """Sends each root twice: the second is one the verifier didn't ask for."""

    def on(self, r):
        out = super().on(r)
        return out + out if r.message.type == "Run" else out


class Withholding(SP.Verifier):
    """Discloses the session's coin commitment at its end, but none of the coins it released."""

    def disclosed(self) -> bytes:
        doc = json.loads(super().disclosed())
        return json.dumps({"commitment": doc["commitment"], "openings": []}).encode()


def test_the_prover_checks_each_challenge_against_the_coins_disclosed_at_the_end(tmp_path) -> None:
    r = simulated(tmp_path, verifier=Withholding)
    assert r.outcomes[SP.VERIFIER]["outcome"]["answer"]["verdict"] == "accept"
    assert r.outcomes[SP.PROVER] == {"outcome": {"passes": 2}, "refused": "coins"}


def test_a_root_it_did_not_ask_for_ends_the_session_out_of_phase(tmp_path) -> None:
    out = simulated(tmp_path, prover=Twice).outcomes[SP.VERIFIER]
    assert out["refused"] == "out-of-phase"
    assert out["outcome"]["answer"]["verdict"] == "undecided"
    assert [e["kind"] for e in events(out["outcome"]["case"])] == ["start", "receive"]
