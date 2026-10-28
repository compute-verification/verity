"""`challenger_party`: the memory challenger's side and the prover's over a socket pair, each in its own thread, the
challenger on an injected clock. Given the period's coins, the challenger's side designates the id `period.run` does in
one process and challenges the same indices. The run's check accepts its session against the honest prover and refuses
it against one that answers zeros. The warm-ups before round 0 are on a public index, and neither timed nor judged. The
prover answers with the collector paused, as the challenger does. `Party.of` reads
`period.json`'s challenger host, and `placement` says whether `trusted-host`'s premise holds."""

from __future__ import annotations

import gc
import hashlib
import json
import socket
import threading

import pytest

import pous_test_signer as ts
from verity.core.protocols.pous import challenger_party as CP
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period, run

STORE = hashlib.shake_256(b"a tiny MLP's weights").digest(600)
COINS = hashlib.sha512(b"the period's live coins").digest()
TEST_ID = mc.TEST_PREFIX + period.PROTOCOL


class Clock:
    def __init__(self) -> None:
        self.now = 10 ** 9

    def __call__(self) -> int:
        self.now += 1_000
        return self.now


def _in_process() -> tuple[period.Period, run.Point, run.Setup, list[bytes], bytes]:
    """`period.run` in one process under a test signer at the small point: its period, point, published setup, `C*`'s
    labels and table."""
    point = period.point_for(len(STORE), period.W, period.C0)
    p = period.run("n0", STORE, COINS, signer=ts.TestSigner(b"in process", 2), log=ts.session_log(), clock=Clock(),
                   point=point, schemes=ts.SCHEMES, protocol=TEST_ID)
    published = run.Setup.from_json(json.loads(bytes.fromhex(p.case["commitment"])))
    _, _, encoded = period.setup_inputs(p.setup_case)
    blocks = [b for seg in encoded for b in seg]
    return p, point, published, blocks, b"".join(mc.block_digest(b) for b in blocks)


def _split(point: run.Point, root: bytes, respond, signer: mc.Signer,
           warmups: int = CP.WARMUPS) -> tuple[bytes, run.ChallengerRecord, int]:
    """One session with the challenger's side in a thread and the prover's here, over a socket pair."""
    a, b = socket.socketpair()
    got: dict = {}

    def side() -> None:
        with a, a.makefile("rwb") as f:
            got["session"] = CP.serve(f, CP.request_of("n0", point, root), signer=signer, log=ts.session_log(),
                                      coins=COINS, domain=period.SESSION_DOMAIN, clock=Clock(), warmups=warmups)

    t = threading.Thread(target=side)
    t.start()
    with b, b.makefile("rwb") as f:
        rounds = CP.answer(f, point.params.block_bytes, respond)
    t.join()
    return *got["session"], rounds


def test_the_split_session_is_the_periods() -> None:
    p, point, published, blocks, digests = _in_process()
    signer = ts.TestSigner(b"challenger host", 2)
    sid, session, rounds = _split(point, published.root, lambda i: blocks[i], signer)
    mine = run.ChallengerRecord.from_json(json.loads(bytes.fromhex(p.case["evidence"]))["session"])
    assert sid.hex() == p.case["params"]["session_id"]
    assert [r.index for r in session.signed.rounds] == [r.index for r in mine.signed.rounds]
    assert rounds == CP.WARMUPS + len(session.signed.rounds) == CP.WARMUPS + point.k
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
    assert seen == [CP.WARMUP_INDEX] * CP.WARMUPS + [r.index for r in session.signed.rounds]
    assert (sid, session.signed.ended, session.signed.rounds) == (bare_sid, bare.signed.ended, bare.signed.rounds)
    n = [0]

    def cold(i: int) -> bytes:
        n[0] += 1
        return bytes(len(blocks[i])) if n[0] <= CP.WARMUPS else blocks[i]

    key = ts.TestSigner(b"cold", 2)
    sid, session, _ = _split(point, published.root, cold, key)
    a = period.answer(point, ts.SCHEMES, (key.public_key,), published, digests, session, sid, TEST_ID,
                      setup_inputs=period.setup_inputs(p.setup_case))
    assert a["verdict"] == "accept", a


def test_a_short_answer_is_refused_by_the_prover() -> None:
    a, b = socket.socketpair()
    with a, b, a.makefile("rwb") as fa, b.makefile("rwb") as fb:
        fa.write((3).to_bytes(8, "big"))
        fa.flush()
        with pytest.raises(ValueError, match="whole 8-byte block"):
            CP.answer(fb, 8, lambda i: bytes(7))


def test_the_prover_answers_with_no_collection_inside_the_rounds() -> None:
    seen: list[bool] = []
    a, b = socket.socketpair()
    with a, b, a.makefile("rwb") as fa, b.makefile("rwb") as fb:
        fa.write(b"".join(i.to_bytes(8, "big") for i in (5, 1, 4, CP.END)))
        fa.flush()
        assert CP.answer(fb, 8, lambda i: (seen.append(gc.isenabled()), bytes([i]) * 8)[1]) == 3
        assert fa.read(24) == bytes([5]) * 8 + bytes([1]) * 8 + bytes([4]) * 8
    assert seen == [False] * 3 and gc.isenabled()


def test_the_party_follows_period_json(tmp_path) -> None:
    local = CP.Party.of({}, "vy-nebius-cpu-4", tmp_path)
    assert (local.host, local.address) == ("vy-nebius-cpu-4", None) and str(tmp_path) in local.command
    assert CP.Party.of({"challenger_host": "vy-nebius-cpu-4"}, "vy-nebius-cpu-4", tmp_path) == local
    with pytest.raises(RuntimeError, match="vy-auditor-1 isn't wired for pous"):
        CP.Party.of({"challenger_host": "vy-auditor-1", "challenger": {"bandwidth": {"command": ["x"]}}},
                    "vy-nebius-cpu-4", tmp_path)
    wired = CP.Party.of({"challenger_host": "vy-auditor-1", "challenger": {"pous": {
        "command": ["ssh", "gate", "pous-challenger"], "address": "127.0.0.1:7960"}}}, "vy-nebius-cpu-4", tmp_path)
    assert wired == CP.Party("vy-auditor-1", ("ssh", "gate", "pous-challenger"), ("127.0.0.1", 7960))
    assert CP.placement("n0", "n0")[CP.PREMISE].startswith("unmet: ")
    assert CP.placement("n0", "vy-auditor-1") == {
        "parties": {"memory_challenger": "vy-auditor-1", "prover": "n0", "verifier": "n0"},
        CP.PREMISE: "holds: the memory challenger ran on vy-auditor-1, not the prover's n0"}
