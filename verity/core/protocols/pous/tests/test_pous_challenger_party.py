"""`challenger_party`: the memory challenger's side and the prover's over a socket pair, each in its own thread, the
challenger on an injected clock. Given the period's coins, the challenger's side designates the id `period.run` does in
one process and challenges the same indices. The run's check accepts its session against the honest prover and refuses
it against one that answers zeros, and so at a batched point, whose answer certifies nothing. The prover answers with
the collector paused, as the challenger does. `Party.of` reads `period.json`'s challenger host, and `placement` says
whether `trusted-nebius-host`'s premise holds."""

from __future__ import annotations

import dataclasses
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
    """`period.run` in one process under a test signer: its period, point, published setup, `C*`'s blocks and table."""
    p = period.run("n0", STORE, COINS, signer=ts.TestSigner(b"in process", 2), log=ts.session_log(), clock=Clock(),
                   schemes=ts.SCHEMES, protocol=TEST_ID)
    point = period.point_for(len(STORE))
    published = run.Setup.from_json(json.loads(bytes.fromhex(p.case["commitment"])))
    _, _, encoded = period.setup_inputs(p.setup_case)
    blocks = [b for seg in encoded for b in seg]
    return p, point, published, blocks, b"".join(mc.block_digest(b) for b in blocks)


def _split(point: run.Point, root: bytes, respond, signer: mc.Signer, respond_batch=None) -> tuple[bytes, run.Session, int]:
    """One session with the challenger's side in a thread and the prover's here, over a socket pair."""
    a, b = socket.socketpair()
    got: dict = {}

    def side() -> None:
        with a, a.makefile("rwb") as f:
            got["session"] = CP.serve(f, CP.request_of("n0", point, root), signer=signer, log=ts.session_log(),
                                      coins=COINS, domain=period.SESSION_DOMAIN, clock=Clock())

    t = threading.Thread(target=side)
    t.start()
    with b, b.makefile("rwb") as f:
        rounds = CP.answer(f, point.params.block_bytes, respond, batch=point.batch, respond_batch=respond_batch)
    t.join()
    return *got["session"], rounds


def test_the_split_session_is_the_periods() -> None:
    p, point, published, blocks, digests = _in_process()
    signer = ts.TestSigner(b"challenger host", 2)
    sid, session, rounds = _split(point, published.root, lambda i: blocks[i], signer)
    mine = run.Session.from_json(json.loads(bytes.fromhex(p.case["evidence"]))["session"])
    assert sid.hex() == p.case["params"]["session_id"]
    assert [r.index for r in session.signed.rounds] == [r.index for r in mine.signed.rounds]
    assert rounds == len(session.signed.rounds) == point.k and session.signed.ended == mc.COMPLETE
    inputs = period.setup_inputs(p.setup_case)
    a = period.answer(point, ts.SCHEMES, (signer.public_key,), published, digests, session, sid, TEST_ID,
                      setup_inputs=inputs)
    assert a["verdict"] == "accept", a
    zeros = _split(point, published.root, lambda i: bytes(len(blocks[i])), ts.TestSigner(b"challenger host", 2))
    refused = period.answer(point, ts.SCHEMES, (signer.public_key,), published, digests, zeros[1], zeros[0], TEST_ID,
                            setup_inputs=inputs)
    assert refused["verdict"] == "reject" and refused["reason"].endswith("answers are not their blocks of C"), refused


def test_a_batched_split_session_is_the_periods_and_its_point_has_a_gap() -> None:
    """At a batched point (7 indices a frame, k = 105 = 15 batches) the split session challenges what the in-process
    period does, the check accepts it honest and refuses it with a block answered as zeros, and the answer certifies
    nothing, its profile naming the batch as a gap."""
    point = dataclasses.replace(period.point_for(len(STORE)), batch=7, limit_ns=50_000_000)
    p = period.run("n0", STORE, COINS, signer=ts.TestSigner(b"in process", 2), log=ts.session_log(), clock=Clock(),
                   schemes=ts.SCHEMES, protocol=TEST_ID, point=point)
    published = run.Setup.from_json(json.loads(bytes.fromhex(p.case["commitment"])))
    inputs = period.setup_inputs(p.setup_case)
    blocks = [b for seg in inputs[2] for b in seg]
    digests = b"".join(mc.block_digest(b) for b in blocks)
    frames: list[int] = []
    signer = ts.TestSigner(b"challenger host", 2)
    sid, session, rounds = _split(point, published.root, None, signer,
                                  lambda idx: (frames.append(len(idx)), [blocks[i] for i in idx])[1])
    mine = run.Session.from_json(json.loads(bytes.fromhex(p.case["evidence"]))["session"])
    assert sid.hex() == p.case["params"]["session_id"] and frames == [7] * 15 and rounds == 105
    assert [r.index for r in session.signed.rounds] == [r.index for r in mine.signed.rounds]
    assert session.commitment.batch == 7 and p.verdict == "accept"
    a = period.answer(point, ts.SCHEMES, (signer.public_key,), published, digests, session, sid, TEST_ID,
                      setup_inputs=inputs)
    assert a["verdict"] == "accept" and a["public"] == {"certified": False, "eps": "0", "s_bits": 0}, a
    assert a["profile"]["batch"] == 7 and any("7 indices a round under one deadline" in g for g in a["profile"]["gaps"])
    assert period.point_of(period.point_json(point)) == point and "batch" not in period.point_json(period.point_for(1))
    seq = dataclasses.replace(point, batch=1)
    assert period.answer(seq, ts.SCHEMES, (signer.public_key,), published, digests, session, sid, TEST_ID,
                         setup_inputs=inputs)["reason"] == "the session reveals 7 indices a round, not the point's 1"
    gone = session.signed.rounds[0].index
    zeros = _split(point, published.root, None, ts.TestSigner(b"challenger host", 2),
                   lambda idx: [bytes(len(blocks[i])) if i == gone else blocks[i] for i in idx])
    refused = period.answer(point, ts.SCHEMES, (signer.public_key,), published, digests, zeros[1], zeros[0], TEST_ID,
                            setup_inputs=inputs)
    assert refused["verdict"] == "reject" and refused["reason"].endswith("answers are not their blocks of C"), refused


def test_the_prover_answers_a_batch_in_one_frame() -> None:
    a, b = socket.socketpair()
    with a, b, a.makefile("rwb") as fa, b.makefile("rwb") as fb:
        fa.write(b"".join(i.to_bytes(8, "big") for i in (5, 1, 4, 2, 2, 0, CP.END)))
        fa.flush()
        assert CP.answer(fb, 8, None, batch=3, respond_batch=lambda idx: [bytes([i]) * 8 for i in idx]) == 6
        assert fa.read(48) == b"".join(bytes([i]) * 8 for i in (5, 1, 4, 2, 2, 0))
    a, b = socket.socketpair()
    with a, b, a.makefile("rwb") as fa, b.makefile("rwb") as fb:
        fa.write(b"".join(i.to_bytes(8, "big") for i in (5, 1, 4)))
        fa.flush()
        with pytest.raises(ValueError, match="a batch's answer is 3 blocks, not 2"):
            CP.answer(fb, 8, None, batch=3, respond_batch=lambda idx: [bytes(8)] * 2)


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
