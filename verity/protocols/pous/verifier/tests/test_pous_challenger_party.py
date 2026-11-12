"""`challenger_party` and `challenger_session`: the memory challenger's side and the prover's over a socket pair, each
in its own thread, the challenger on an injected clock. Given the period's coins, the challenger's side designates
the id `period.run` does in one process and challenges the same indices. The run's check accepts its session against
the honest prover and refuses it against one that answers zeros. The warm-ups before round 0 are on a public index,
and neither timed nor judged. The prover answers with the collector paused, as the challenger does. The deployment
layer's runtime, running the same two parties on the same coins and clock readings, signs the session the stream
does. `side_of` reads `period.json`'s challenger host, and `placement` says whether `trusted-host`'s premise holds."""

from __future__ import annotations

import functools
import gc
import hashlib
import json
import os
import re
import socket
import threading

import pytest

import pous_fakes
import pous_test_signer as ts
from verity.configuration.deploy import sources as S
from verity.configuration.deploy.party import Send
from verity.configuration.deploy.runtime import run as deploy_run
from verity.configuration.deploy.stream import Side
from verity.protocols.pous.device import challenger_party as CP
from verity.protocols.pous.device import challenger_session as CS
from verity.protocols.pous.device import memory_challenger as mc
from verity.protocols.pous.verifier import period, run
from verity.protocols.pous.verifier.schemes.p2 import P2Params
from verity.configuration.wire import Message

STORE = hashlib.shake_256(b"a tiny MLP's weights").digest(600)
COINS = hashlib.sha512(b"the period's live coins").digest()
TEST_ID = mc.TEST_PREFIX + period.PROTOCOL
Clock = functools.partial(pous_fakes.Clock, 10 ** 9, 1_000)


def _in_process() -> tuple[period.Run, run.Point, run.Setup, list[bytes], bytes]:
    """`period.run` in one process under a test signer at the small point: its period, point, published setup, `C*`'s
    labels and table."""
    point = period.point_for(len(STORE), period.W, period.C0)
    p = period.run("n0", STORE, COINS, signer=ts.TestSigner(b"in process", 2), log=ts.session_log(), clock=Clock(),
                   point=point, schemes=ts.SCHEMES, protocol=TEST_ID)
    published = run.Setup.from_json(json.loads(bytes.fromhex(p.case["commitment"])))
    _, _, encoded = period.setup_inputs(p.setup_case)
    blocks = [b for seg in encoded for b in seg]
    return p, point, published, blocks, b"".join(mc.block_digest(b) for b in blocks)


def _split(point: run.Point, root: bytes, respond, signer: mc.Signer, warmups: int = CP.WARMUPS,
           draw=os.urandom) -> tuple[bytes, run.ChallengePhaseRecord, int]:
    """One session with the challenger's side in a thread and the prover's here, over a socket pair."""
    a, b = socket.socketpair()
    got: dict = {}

    def side() -> None:
        with a:
            got["session"] = CP.serve(CS.Polled(a), CS.request_of("n0", point, root), signer=signer,
                                      log=ts.session_log(), coins=COINS, domain=period.SESSION_DOMAIN, clock=Clock(),
                                      warmups=warmups, draw=draw)

    t = threading.Thread(target=side)
    t.start()
    with b:
        rounds = CS.answer(CS.Polled(b), point.params.block_bytes, respond)
    t.join()
    return *got["session"], rounds


def test_the_split_session_is_the_periods() -> None:
    p, point, published, blocks, digests = _in_process()
    signer = ts.TestSigner(b"challenger host", 2)
    sid, session, rounds = _split(point, published.root, lambda i: blocks[i], signer)
    mine = run.ChallengePhaseRecord.from_json(json.loads(bytes.fromhex(p.case["evidence"]))["session"])
    assert sid.hex() == p.case["params"]["session_id"]
    assert [r.index for r in session.signed.rounds] == [r.index for r in mine.signed.rounds]
    assert rounds == CS.WARMUPS + len(session.signed.rounds) == CS.WARMUPS + point.k
    assert session.signed.ended == mc.COMPLETE
    inputs = period.setup_inputs(p.setup_case)
    a = period.answer(point, ts.SCHEMES, (signer.public_key,), published, digests, session, sid, TEST_ID,
                      setup_inputs=inputs)
    assert a["verdict"] == "accept", a
    zeros = _split(point, published.root, lambda i: bytes(len(blocks[i])), ts.TestSigner(b"challenger host", 2))
    refused = period.answer(point, ts.SCHEMES, (signer.public_key,), published, digests, zeros[1], zeros[0], TEST_ID,
                            setup_inputs=inputs)
    assert refused["verdict"] == "reject" and refused["reason"].endswith("answers are not their blocks of C"), refused


def test_the_warmups_come_first_and_are_neither_timed_nor_judged() -> None:
    """The prover sees `WARMUPS` frames on `WARMUP_INDEX`, then the session's indices. The session is the one served with
    no warm-up, round for round under the same clock, so no warm-up read it; a prover whose warm-up answers are zeros is
    accepted."""
    p, point, published, blocks, digests = _in_process()
    seen: list[int] = []
    sid, session, _ = _split(point, published.root, lambda i: (seen.append(i), blocks[i])[1], ts.TestSigner(b"w", 2))
    bare_sid, bare, _ = _split(point, published.root, lambda i: blocks[i], ts.TestSigner(b"w", 2), warmups=0)
    assert seen == [CS.WARMUP_INDEX] * CS.WARMUPS + [r.index for r in session.signed.rounds]
    assert (sid, session.signed.ended, session.signed.rounds) == (bare_sid, bare.signed.ended, bare.signed.rounds)
    n = [0]

    def cold(i: int) -> bytes:
        n[0] += 1
        return bytes(len(blocks[i])) if n[0] <= CS.WARMUPS else blocks[i]

    key = ts.TestSigner(b"cold", 2)
    sid, session, _ = _split(point, published.root, cold, key)
    a = period.answer(point, ts.SCHEMES, (key.public_key,), published, digests, session, sid, TEST_ID,
                      setup_inputs=period.setup_inputs(p.setup_case))
    assert a["verdict"] == "accept", a


class GivenCoins:
    """A coins source that answers its first sample with the period's coins, as `serve` is given them, and the rest
    from `draw`."""

    def __init__(self, draw) -> None:
        self.draw, self.given = draw, [COINS]

    def start(self, params, identity) -> list:
        return []

    def on(self, msg) -> list:
        n = msg.message.fields["n"]
        return [Send(S.OWNER, Message("Coins", body=self.given.pop() if self.given else self.draw(n)))]


def test_the_runtime_runs_the_same_parties_to_the_same_session() -> None:
    """`Challenger` and `Prover` in the deployment layer's runtime, on `serve`'s coins and a clock that reads what
    `Clock` does, its hiding key and salts sampled from one seeded stream, sign the session `serve` signs over the stream,
    and the prover answers the same frames."""
    _, point, published, blocks, _ = _in_process()
    sid, old, rounds = _split(point, published.root, lambda i: blocks[i], ts.TestSigner(b"same key", 2),
                              draw=S.SeededCoins(b"hiding").draw)
    caps = {CS.CHALLENGER: {"signer": ts.TestSigner(b"same key", 2), "log": ts.session_log()},
            CS.PROVER: {"store": lambda i: blocks[i]}}
    clock = S.VirtualClock(S.VirtualTime(Clock()()), 1_000)
    r = deploy_run({"protocol": CS.ID, "params": CS.request_of("n0", point, published.root)},
                   [CS.CHALLENGER, CS.PROVER], "live", None, protocols={CS.ID: CP.PROTOCOL}, capabilities=caps,
                   authenticate=False,
                   sources={CS.CHALLENGER: {"clock": clock, "coins": GivenCoins(S.SeededCoins(b"hiding").draw)}})
    assert r.status == "completed", r.outcomes
    assert r.outcomes[CS.CHALLENGER] == {"outcome": {"session": old.to_json(), "session_id": sid.hex()},
                                         "refused": None}
    assert r.outcomes[CS.PROVER] == {"outcome": rounds, "refused": None}


class Script:
    """A socket whose reads follow `steps`: bytes, or None while nothing has arrived. It records whether each read
    polled (`MSG_DONTWAIT`) or blocked."""

    def __init__(self, *steps: bytes | None) -> None:
        self.steps, self.reads = list(steps), []

    def recv_into(self, view, n: int, flags: int = 0) -> int:
        polled = bool(flags & socket.MSG_DONTWAIT)
        self.reads.append("poll" if polled else "block")
        step = self.steps.pop(0)
        if step is None:
            assert polled, "a blocking read waits for bytes"
            raise BlockingIOError
        view[:len(step)] = step
        return len(step)


def test_a_polled_read_polls_for_its_budget_then_blocks_and_gathers_the_frame() -> None:
    """Each wait polls until `spin_ns` has passed on the side's clock since it began, then blocks; any bytes start the
    next wait afresh, and the frame comes back whole."""
    sock = Script(None, None, None, b"abc", b"de", None, None, b"fgh")
    clock = iter([0, 5, 10, 20, 30]).__next__
    assert CS.Polled(sock, spin_ns=10, clock=clock).read(8) == b"abcdefgh"
    assert sock.reads == ["poll", "poll", "poll", "block", "poll", "poll", "poll", "block"]


def test_a_polled_read_the_peer_ends_early_is_short_and_refused() -> None:
    sock = Script(bytes(3), None, b"")
    clock = iter([0]).__next__
    with pytest.raises(ConnectionError, match="closed after 3 of 8 bytes"):
        CS.Frames().read(CS.Polled(sock, spin_ns=10, clock=clock))


def test_a_short_answer_is_refused_by_the_prover() -> None:
    a, b = socket.socketpair()
    with a, b, a.makefile("rwb") as fa, b.makefile("rwb") as fb:
        fa.write((3).to_bytes(8, "big"))
        fa.flush()
        with pytest.raises(ValueError, match="whole 8-byte block"):
            CS.answer(fb, 8, lambda i: bytes(7))


def test_the_prover_answers_with_no_collection_inside_the_rounds() -> None:
    seen: list[bool] = []
    a, b = socket.socketpair()
    with a, b, a.makefile("rwb") as fa, b.makefile("rwb") as fb:
        fa.write(b"".join(i.to_bytes(8, "big") for i in (5, 1, 4, CS.END)))
        fa.flush()
        assert CS.answer(fb, 8, lambda i: (seen.append(gc.isenabled()), bytes([i]) * 8)[1]) == 3
        assert fa.read(24) == bytes([5]) * 8 + bytes([1]) * 8 + bytes([4]) * 8
    assert seen == [False] * 3 and gc.isenabled()


AUDIT = hashlib.sha512(b"window 3's binding").digest()


def _salt_log(tmp_path) -> mc.SaltLog:
    mc.SaltLog.create(tmp_path / "salts")
    return mc.SaltLog(tmp_path / "salts")


def _audit_session(point: run.Point, published: run.Setup, respond, signer: mc.Signer, salts: mc.SaltLog,
                   log: mc.SessionLog) -> tuple[bytes, run.ChallengePhaseRecord]:
    """One session of the audit `AUDIT` on `published`, the challenger's side in a thread, the prover's here."""
    a, b = socket.socketpair()
    got: dict = {}
    request = CS.request_of("n0", point, published.root, audit=AUDIT, published=published)

    def side() -> None:
        with a, a.makefile("rwb") as f:
            got["session"] = CP.serve(f, request, signer=signer, log=log, salts=salts, coins=COINS,
                                      domain=period.SESSION_DOMAIN, clock=Clock())

    t = threading.Thread(target=side)
    t.start()
    with b, b.makefile("rwb") as f:
        CS.answer(f, point.params.block_bytes, respond)
    t.join()
    return got["session"]


def _refused(request, salts: mc.SaltLog, log: mc.SessionLog) -> str:
    """What `designate` refuses `request` for; `serve` refuses it the same, before it sends a frame."""
    with pytest.raises(CP.Refusal) as e:
        CP.designate(request, salts, log)
    a, b = socket.socketpair()
    with a, b, a.makefile("rwb") as f:
        with pytest.raises(CP.Refusal) as served:
            CP.serve(f, request, signer=ts.TestSigner(b"unused", 2), log=log, salts=salts, coins=COINS,
                     domain=period.SESSION_DOMAIN, clock=Clock())
    assert str(served.value) == str(e.value)
    return str(e.value)


def test_an_audits_session_runs_under_the_id_its_salt_fixes(tmp_path) -> None:
    """The challenger samples the setup's salt after the commitment and logs it under the audit's binding; the setup salted
    from that source publishes it, the session runs under `session_id(audit, commitment, salt)`, which the verifier
    computes itself, and the run is accepted. The same session under the id its report could have named (the period's,
    from the coins) is refused."""
    p, point, published, blocks, digests = _in_process()
    salts, log = _salt_log(tmp_path), ts.session_log(tmp_path)
    source = CP.give_salt(salts, {"audit": AUDIT.hex(), "commitment": published.commitment.hex()}, draw=lambda n: COINS)
    assert source == COINS and salts.get(AUDIT) == (published.commitment, published.salt)
    signer = ts.TestSigner(b"auditor", 2)
    sid, session = _audit_session(point, published, lambda i: blocks[i], signer, salts, log)
    assert sid == CS.session_id(AUDIT, published.commitment, published.salt) == session.commitment.session
    assert sid == hashlib.sha512(CS.SESSION_ID_TAG + AUDIT + published.commitment + published.salt).digest()[:32]
    assert sid in log and session.signed.ended == mc.COMPLETE
    inputs = period.setup_inputs(p.setup_case)
    ok = period.answer(point, ts.SCHEMES, (signer.public_key,), published, digests, session, sid, TEST_ID,
                       setup_inputs=inputs)
    assert ok["verdict"] == "accept", ok
    other = bytes.fromhex(p.case["params"]["session_id"])
    refused = period.answer(point, ts.SCHEMES, (signer.public_key,), published, digests, session, other, TEST_ID,
                            setup_inputs=inputs)
    assert refused["reason"] == "the session does not run under the id the auditor designated"


def test_a_second_session_for_an_audit_is_refused_never_redrawn(tmp_path) -> None:
    """Once an audit's session has run, the challenger refuses another for it, before any frame: no retry flips the
    coins again, and the session log holds the one id."""
    _, point, published, blocks, _ = _in_process()
    salts, log = _salt_log(tmp_path), ts.session_log(tmp_path)
    CP.give_salt(salts, {"audit": AUDIT.hex(), "commitment": published.commitment.hex()}, draw=lambda n: COINS)
    sid, _ = _audit_session(point, published, lambda i: blocks[i], ts.TestSigner(b"once", 2), salts, log)
    request = CS.request_of("n0", point, published.root, audit=AUDIT, published=published)
    assert _refused(request, salts, log) == (f"a session has run for audit {AUDIT.hex()}: the memory challenger runs one "
                                             "an audit, and a retry is refused, never redrawn")
    assert log.path.read_text().splitlines()[1:] == [sid.hex()]


def test_a_second_salt_for_an_audit_is_refused_and_the_first_kept(tmp_path) -> None:
    salts = _salt_log(tmp_path)
    q = {"audit": AUDIT.hex(), "commitment": (b"W" * 64).hex()}
    first = CP.give_salt(salts, q)
    with pytest.raises(CP.Refusal, match="draws one salt an audit, and a retry never redraws it"):
        CP.give_salt(salts, q | {"commitment": (b"V" * 64).hex()})
    assert salts.get(AUDIT) == (b"W" * 64, period.setup.draw_salt(first, b"W" * 64))


def test_a_session_whose_setup_salt_is_not_the_challengers_is_refused(tmp_path) -> None:
    """Setup's published outputs must carry the salt the challenger logged for the audit and the commitment it drew it
    after, and the session must audit their root; an audit with no salt logged has no session."""
    _, point, published, _, _ = _in_process()
    salts, log = _salt_log(tmp_path), ts.session_log(tmp_path)
    request = CS.request_of("n0", point, published.root, audit=AUDIT, published=published)
    assert _refused(request, salts, log).startswith(f"the memory challenger drew no salt for audit {AUDIT.hex()}")
    CP.give_salt(salts, {"audit": AUDIT.hex(), "commitment": published.commitment.hex()})
    assert _refused(request, salts, log) == ("setup's published outputs do not carry the commitment and salt the memory "
                                             f"challenger logged for audit {AUDIT.hex()}")
    logged = run.Setup(published.blocks, published.commitment, published.root, salts.get(AUDIT)[1])
    elsewhere = CS.request_of("n0", point, b"\x01" * 64, audit=AUDIT, published=logged)
    assert _refused(elsewhere, salts, log).startswith("the session does not audit the C* setup published")
    assert _refused({k: v for k, v in request.items() if k != "published"}, salts, log).startswith(
        "the request names no audit and setup's published outputs")
    assert log.path.read_text() == f"{mc.SESSION_LOG}\n"


def test_a_setup_under_the_provers_salt_fails_the_verifier_though_its_root_carries_the_challengers(tmp_path) -> None:
    """A prover that shows the challenger the salt it logged beside the root of a `C*` encoded under a salt of its own
    gets a session bound to that root, since the challenger can't see which salt the root is under. The verifier, which
    takes setup's published outputs under the prover's salt, computes another id, so the run is refused."""
    _, point, _, _, _ = _in_process()
    mine = period.setup_store(STORE, hashlib.sha512(b"the prover's own coins").digest(), point)
    salts, log = _salt_log(tmp_path), ts.session_log(tmp_path)
    CP.give_salt(salts, {"audit": AUDIT.hex(), "commitment": mine.published.commitment.hex()}, draw=lambda n: COINS)
    shown = run.Setup(mine.published.blocks, mine.published.commitment, mine.published.root, salts.get(AUDIT)[1])
    assert shown.salt != mine.published.salt
    signer = ts.TestSigner(b"fooled", 2)
    sid, session = _audit_session(point, shown, lambda i: mine.blocks[i], signer, salts, log)
    expected = CS.session_id(AUDIT, mine.published.commitment, mine.published.salt)
    assert sid != expected
    a = period.answer(point, ts.SCHEMES, (signer.public_key,), mine.published, mine.digests, session, expected, TEST_ID,
                      setup_inputs=period.setup_inputs(mine.setup_case))
    assert a["reason"] == "the session does not run under the id the auditor designated", a


def test_the_side_serves_a_salt_request_and_refuses_before_it_listens(tmp_path) -> None:
    """`main` as a process (`command`): a salt request is answered with its source, logged in the device's salt log; a
    second for the audit is refused, as is a session whose setup doesn't carry the logged salt, before the side listens
    or loads its key."""
    device = tmp_path / "device"
    side = Side("auditor-host", CS.command(device))
    commitment = hashlib.sha512(b"W").digest()
    source = CS.salt(side, AUDIT, commitment)
    assert len(source) == CP.SOURCE_BYTES
    assert mc.SaltLog(device / "salts").get(AUDIT) == (commitment, period.setup.draw_salt(source, commitment))
    with pytest.raises(RuntimeError, match="the memory challenger on auditor-host drew no salt: audit .* has its salt"):
        CS.salt(side, AUDIT, commitment)
    point = period.point_for(len(STORE), period.W, period.C0)
    wrong = run.Setup(point.params.blocks, commitment, b"\x02" * 64, bytes(24))
    with pytest.raises(RuntimeError, match="the side on auditor-host refused: setup's published outputs do not carry"):
        side.talk(CS.request_of("n0", point, wrong.root, audit=AUDIT, published=wrong), lambda s: None)
    assert not (device / "seed").exists() and mc.SessionLog(device / "sessions").path.read_text() == f"{mc.SESSION_LOG}\n"


def test_the_party_follows_period_json(tmp_path) -> None:
    local = CS.side_of({}, "cpu-node", tmp_path)
    assert (local.host, local.address) == ("cpu-node", None) and str(tmp_path) in local.command
    assert CS.side_of({"challenger_host": "cpu-node"}, "cpu-node", tmp_path) == local
    with pytest.raises(RuntimeError, match="auditor-host isn't wired for pous"):
        CS.side_of({"challenger_host": "auditor-host", "challenger": {"bandwidth": {"command": ["x"]}}},
                   "cpu-node", tmp_path)
    wired = CS.side_of({"challenger_host": "auditor-host", "challenger": {"pous": {
        "command": ["ssh", "gate", "pous-challenger"], "address": "127.0.0.1:7960"}}}, "cpu-node", tmp_path)
    assert wired == Side("auditor-host", ("ssh", "gate", "pous-challenger"), ("127.0.0.1", 7960))


PINNED, OTHER = CS.AUDITOR_KEYS["vy-auditor-2"], hashlib.sha512(b"a key that is not the auditor's").hexdigest()
REHEARSAL = "rehearsal: key not the auditor's pinned key, no bound; the names say "


def test_a_rehearsals_challenger_under_a_key_that_is_not_the_pin_bounds_nothing() -> None:
    """The rehearsal's names, a challenger host that is not the node, say the challenger ran off the node; under a key
    the tree pins for no host, the record is a rehearsal's."""
    assert CS.placement("vy-dc-h200-3", "on-node-rehearsal", OTHER) == {
        "parties": {"memory_challenger": "on-node-rehearsal", "prover": "vy-dc-h200-3", "verifier": "vy-dc-h200-3"},
        "kind": "own-host",
        CS.PREMISE: REHEARSAL + "the memory challenger ran on on-node-rehearsal, not the prover's vy-dc-h200-3"}
    assert CS.placement("n0", "n0", OTHER) == {
        "parties": {"memory_challenger": "n0", "prover": "n0", "verifier": "n0"}, "kind": "prover-node",
        CS.PREMISE: REHEARSAL + "the memory challenger ran on the prover's node n0"}


def test_the_auditors_host_under_its_pinned_key_holds() -> None:
    """vy-auditor-2 under the key the tree pins for it: `trusted-host` holds. The pins are the module's alone."""
    where = CS.placement("vy-dc-h200-3", "vy-auditor-2", PINNED, verifier="vy-auditor-2")
    assert where == {
        "parties": {"memory_challenger": "vy-auditor-2", "prover": "vy-dc-h200-3", "verifier": "vy-auditor-2"},
        "kind": "own-host",
        CS.PREMISE: "holds: the memory challenger ran on vy-auditor-2, not the prover's vy-dc-h200-3, "
                    "under the key pinned for vy-auditor-2"}
    assert re.fullmatch("[0-9a-f]{128}", PINNED)
    with pytest.raises(TypeError):
        CS.AUDITOR_KEYS["on-node-rehearsal"] = OTHER


def test_the_auditors_host_name_under_another_key_is_a_rehearsal() -> None:
    """The name vy-auditor-2 is the run's to give: under any key but its pinned one the record is a rehearsal's, and
    so it is when the prover's node takes that name too, even under the pinned key."""
    assert CS.placement("vy-dc-h200-3", "vy-auditor-2", OTHER)[CS.PREMISE] == (
        REHEARSAL + "the memory challenger ran on vy-auditor-2, not the prover's vy-dc-h200-3")
    same = CS.placement("vy-auditor-2", "vy-auditor-2", PINNED)
    assert same["kind"] == "prover-node" and same[CS.PREMISE] == (
        "rehearsal: the prover's node has the auditor host's name, no bound; "
        "the names say the memory challenger ran on the prover's node vy-auditor-2")


def test_a_request_carries_its_setup_and_the_challenger_keeps_each_completed_session_privately(tmp_path) -> None:
    """A request naming its audit carries setup's published outputs and the point's parameters, its setup refused unless
    its root is the binding, its blocks and parameters the session's and its salt PoUS's; the kept record is the request
    and the id the challenger designated, 0600 in a 0700 directory, written once."""
    _, point, published, _, _ = _in_process()
    req = CS.request_of("n0", point, published.root, audit=AUDIT, published=published)
    assert CS.setup_of(req) == req["published"] | {"params": req["params"]}
    assert run.Setup.from_json(req["published"]) == published
    assert P2Params(**req["params"]) == point.params and CS.setup_of(CS.request_of("n0", point, published.root)) is None
    q = point.params
    for bad, why in (({"root": "11" * 64}, "not the session's binding"), ({"blocks": q.blocks + 1}, "blocks"),
                     ({"salt": "00"}, "salt PoUS's"), ({"extra": 1}, "a request's published outputs are")):
        with pytest.raises(ValueError, match=why):
            CS.setup_of(req | {"published": req["published"] | bad})
    for params, why in ((req["params"] | {"segments": q.segments + 1}, "not the session's blocks"),
                        ({k: v for k, v in req["params"].items() if k != "w"}, "a setup's params are")):
        with pytest.raises(ValueError, match=why):
            CS.setup_of(req | {"params": params})
    with pytest.raises(ValueError, match="a request's published outputs are"):
        CS.setup_of({k: v for k, v in req.items() if k != "published"})
    sid = bytes(range(32))
    path = CP.keep(tmp_path, sid, req)
    assert path == tmp_path / CS.SETUPS / f"{sid.hex()}.json"
    assert json.loads(path.read_text()) == {"request": req, "session_id": sid.hex()}
    assert [p.stat().st_mode & 0o777 for p in (path.parent, path)] == [0o700, 0o600]
    with pytest.raises(FileExistsError):
        CP.keep(tmp_path, sid, req | {"node": "another"})
