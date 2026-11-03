"""`remote-audit-protocol serve` (`RemoteAudit.Protocol`, onsite Verity's bandwidth component) against the Python verifier
of the same case (`protocol`: the case decoded as the Lean decodes it, its events through `audit.Session` as
`ra_cases.session` replays them, and `remote_bandwidth.period.answer` of the session after them): every field of the
answer but its reason, the certified `q_bytes`, `eps` and the profile among them. Both refuse a session paced at one
window in flight, which no property of the Lean covers.

The cases are a small period on one node (`period.run` on a clock the test sets): honest, its prover reading a store
that dropped every other chunk, and stopped before its last window is verified; every recorded scenario
(`ra_cases.SCENARIOS`) as a case; drawn honest sessions of drawn shapes; and recorded ones with one or two edits
(`test_drawn.edited`) or one edit of the case's framing."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from fractions import Fraction

import pytest
from hypothesis import strategies as st

import ra_cases as M
import verity_difftest as D
from test_drawn import BASES, edited, shaped
from verity_difftest import canonical
from verity_difftest import protocol as P
from verity.core.protocols.remote_bandwidth import audit, period

pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

EXE = "remote-audit-protocol"


def _keys(j, want: list[str], ctx: str) -> None:
    if not isinstance(j, dict):
        raise D.Reject(f"{ctx}: not an object")
    if sorted(j) != want:
        raise D.Reject(f"{ctx}: the keys must be {want}, not {sorted(j)}")


def _params(j) -> dict:
    _keys(j, ["not_before_ns", "params"], "params")
    return j


def _registration(b: bytes) -> audit.Registration:
    with M.decoding():
        return M.registration_of(P.canonical_json(b))


def _evidence(b: bytes) -> dict:
    j = P.canonical_json(b)
    _keys(j, ["append_coins", "coins", "events"], "evidence")
    M._list(j, "events")
    return j


def _check(p: dict, reg: audit.Registration, e: dict) -> tuple:
    try:
        with M.decoding():
            kw, committed, appends, not_before = M.session_of(p | {"append_coins": e["append_coins"], "coins": e["coins"]})
            events = [M.event_of(x) for x in e["events"]]
            params = M.params_of(reg, kw)
        clock = M.Clock()
        s = audit.Session(params, reg, None, M.replayed(reg, 0, params.windows, committed, appends, params.groups_per_coin),
                          not_before_ns=not_before, clock=clock)
        for x in events:
            M.apply(s, clock, x)
    except M.REFUSALS as why:
        return "reject", str(why)
    a = period.answer(s)
    return (a["verdict"], a["public"], a["profile"]) if a["verdict"] == "accept" else (a["verdict"], a["reason"])


def protocol(case) -> dict:
    """`RemoteAudit.Protocol.protocol.answer` in Python."""
    return P.answer(period.PROTOCOL, case, params=_params, commitment=_registration, evidence=_evidence, check=_check,
                    digest=lambda r: r.digest)


def case_of(session: dict) -> dict:
    """The protocol's case of a `session` difftest's case: the registration, the log, and the parameters."""
    s = session if isinstance(session, dict) else {}
    return {"commitment": canonical(s.get("registration")).encode().hex(),
            "evidence": canonical({k: s[k] for k in ("append_coins", "coins", "events") if k in s}).encode().hex(),
            "params": {k: s[k] for k in ("not_before_ns", "params") if k in s}}


def _json(h: str):
    return json.loads(bytes.fromhex(h))


#: One edit of a case's framing: its params, its evidence's keys or bytes, its commitment's bytes, a field missing.
FRAMING = [
    lambda c: c | {"params": c["params"] | {"x": 1}},
    lambda c: c | {"params": {k: v for k, v in c["params"].items() if k != "params"}},
    lambda c: c | {"params": [c["params"]]},
    lambda c: c | {"evidence": json.dumps(_json(c["evidence"])).encode().hex()},
    lambda c: c | {"evidence": canonical(_json(c["evidence"]) | {"x": []}).encode().hex()},
    lambda c: c | {"evidence": canonical(_json(c["evidence"]) | {"events": {}}).encode().hex()},
    lambda c: c | {"commitment": json.dumps(_json(c["commitment"])).encode().hex()},
    lambda c: c | {"commitment": c["commitment"].upper()},
    lambda c: c | {"commitment": c["commitment"][:-2]},
    lambda c: {k: v for k, v in c.items() if k != "evidence"},
]

PORT = D.ProtocolPort(EXE, protocol, st.nothing(), package=M.PACKAGE)
server = D.server_fixture(PORT)


STORE = hashlib.shake_256(b"a tiny MLP's weights").digest(3 * 4032 - 17)
COINS = hashlib.sha512(b"the period's live coins").digest()


def periods() -> dict[str, period.Run]:
    """The small period on node `n0`, honest, under a prover that dropped every other chunk, and one verified only up
    to its first window."""
    out = {}
    for name, kw in (("honest", {}), ("dropping", {"cheat": M.dropping(lambda j: j % 2 == 1)}), ("unverified", {"verify": 1})):
        c = M.Clock(10 ** 9, 1_000)
        out[name] = period.run("n0", STORE, COINS, clock=c, sleep_until=c.sleep_until, entropy=M.Entropy(b"salts"), **kw)
    return out


def _part(answer: dict) -> dict:
    return {k: v for k, v in answer.items() if k != "reason"}


def test_a_small_periods_answers_agree(server) -> None:
    """The hook's own answer (the live session's) is the Python verifier's of its case and the Lean's: accepted with the
    reads its openings certify, rejected for the dropped chunks' paths, and undecided with a window unverified."""
    got = {}
    for name, p in periods().items():
        assert _part(protocol(copy.deepcopy(p.case))) == _part(p.answer), name
        kind, text = D.replay(PORT, p.case, server)
        got[name] = json.loads(text)
    assert {n: a["verdict"] for n, a in got.items()} == {"honest": "accept", "dropping": "reject", "unverified": "undecided"}
    honest = got["honest"]
    assert honest["public"]["q_bytes"] == honest["profile"]["receipts"] * 4096 > 0 and honest["profile"]["gaps"] == []
    draws = sum(s for _, s in honest["profile"]["windows"])
    assert Fraction(honest["public"]["eps"]) == Fraction(2 ** 24 + draws, 2 ** 64), honest


def _covered(kw: dict) -> bool:
    """A scenario's session is one the check replays (`RemoteAudit.Protocol.sessionOf`): not paced at one window in
    flight."""
    pace = kw.get("pace")
    return pace is None or pace.in_flight == 2


@pytest.mark.parametrize("name", list(M.SCENARIOS))
def test_a_recorded_scenario_agrees(server, name) -> None:
    """Each scenario's verdict is its recorded outcome's, and a reject for a session the check refuses to replay."""
    case, outcome = M.recorded(name)
    kind, text = D.replay(PORT, case_of(case), server)
    _, kw, _, recorded = M.SCENARIOS[name]
    want = {"accepted": "accept", None: "undecided"}.get(recorded, "reject") if _covered(kw) else "reject"
    assert json.loads(text)["verdict"] == want, (name, text)


def test_drawn_honest_sessions_agree(server) -> None:
    stats = D.run(D.ProtocolPort(EXE, protocol, shaped().map(case_of), package=M.PACKAGE, need_rejects=False),
                  examples=25, server=server)
    assert stats.counts.get("accept", 0) == 25, stats


@pytest.mark.slow
def test_drawn_edits_agree(server) -> None:
    base = st.sampled_from(BASES).map(lambda name: M.recorded(name)[0])
    framed = st.builds(lambda c, m: m(c), base.map(case_of), st.sampled_from(FRAMING))
    stats = D.run(D.ProtocolPort(EXE, protocol, st.one_of(edited(base).map(case_of), framed), package=M.PACKAGE),
                  examples=300, server=server)
    assert all(stats.counts.get(v, 0) > 10 for v in ("accept", "reject", "undecided")), stats
