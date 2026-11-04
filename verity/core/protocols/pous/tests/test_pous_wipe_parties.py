"""The wipe's memory challenger and prover as parties of the deployment layer (`wipe_parties`). On the coins, clock
readings and key `wipe_party.serve` is given, the ported challenger signs the same bytes, whatever its fill frames' size;
an honest prover keeping the fill in host memory (`wipe_store.StoreProver`) is accepted by `verity/wipe/v0`'s verdict, one
keeping foreign content is rejected and one that leaves is undecided; the challenger's transcript keeps each large fill
frame as its digest, and the challenger replays from it."""

from __future__ import annotations

import hashlib
import itertools
import socket
import threading
import types

import pytest

import pous_test_signer as ts
import settings
from verity.core.deploy import sources as S
from verity.core.deploy.party import Finish
from verity.core.deploy.runtime import ProtocolDef, run
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import streamed_wipe as sw
from verity.core.protocols.pous import wipe_parties as WPS
from verity.core.protocols.pous import wipe_party as WP
from verity.core.protocols.pous import wipe_period as W
from verity.core.protocols.pous import wipe_store as WS
from verity.core.wire import canonical
from verity.ml.crypto import lms

TEST_ID = mc.TEST_PREFIX + W.PROTOCOL
C, P = WPS.CHALLENGER, WPS.PROVER
BOTH = [C, P]
PARAMS = sw.WipeParams(256, 64, 1, 10_000, (("host-dram", 256 * sw.LABEL_BYTES),))
SEED = b"wipe parties"
T0, STEP = 10 ** 9, 1_000
KEY = lms.Params(64, 1, 3)


class PartyKey:
    """A party's handshake key, apart from the challenger's signing key."""

    def __init__(self, party: str) -> None:
        self.key, self.q = lms.keygen(KEY, hashlib.sha512(b"wipe parties " + party.encode()).digest()), 0
        self.public_key = lms.public_key(self.key)

    def sign(self, message: bytes) -> bytes:
        self.q += 1
        return lms.sign_at(self.key, self.q - 1, message).encode()


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    return lms.verify(KEY, public, message, lms.Signature.decode(KEY, signature))


def serve(monkeypatch, params: sw.WipeParams, secret: bytes) -> sw.WipeMemoryChallenger:
    """`wipe_party.serve` over a socket pair, its OS draws taken in order from `SEED`'s seeded coins."""
    coins = S.SeededCoins(SEED)
    monkeypatch.setattr(WP, "secrets", types.SimpleNamespace(token_bytes=coins.draw))
    monkeypatch.setattr(sw, "os", types.SimpleNamespace(urandom=coins.draw))
    signer, log = ts.TestSigner(secret, 2), ts.session_log()
    a, b = socket.socketpair()
    got: dict = {}

    def side() -> None:
        with a, a.makefile("rwb") as f:
            got["c"] = WP.serve(f, WP.request_of("n0", params), signer=signer, log=log, egress_bits=0, reads=params.k,
                                clock=itertools.count(T0, STEP).__next__)

    t = threading.Thread(target=side)
    t.start()
    with b, b.makefile("rwb") as f:
        WP.answer(f, sw.Prover(params))
    t.join()
    return got["c"]


def challenger_caps(secret: bytes, params: sw.WipeParams) -> dict:
    return {"signer": ts.TestSigner(secret, 2), "log": ts.session_log(), "egress_bits": 0, "reads": params.k}


def wipe(params: sw.WipeParams, secret: bytes, *, mode: str = "live", chunk: int = WP.CHUNK, store=sw.Prover,
         prover=WPS.Prover, state=None, request=None):
    caps = {C: challenger_caps(secret, params), P: {"store": store}}
    protocol = ProtocolDef(WPS.MANIFEST, {C: lambda: WPS.MemoryChallengerParty(chunk), P: prover})
    given = {} if mode == "simulated" else {
        "sources": {C: {"clock": S.VirtualClock(S.VirtualTime(T0), STEP), "coins": S.SeededCoins(SEED)}}}
    return run({"protocol": WPS.ID, "params": request or WP.request_of("n0", params)}, BOTH, mode, state,
               protocols={WPS.ID: protocol}, keys={p: PartyKey(p) for p in BOTH}, verify=verify, capabilities=caps,
               seed=SEED if mode == "simulated" else None, clock_step_ns=STEP, **given)


def verdict(out: dict, secret: bytes) -> dict:
    commitment = WP.commitment_of(out["commitment"])
    signed = None if out["signed"] is None else WP.signed_of(out["signed"])
    return W.answer(b"n0", commitment.session, (ts.public_key(secret, 2),), commitment, signed, ts.SCHEMES, TEST_ID)


@pytest.mark.parametrize("chunk", [WP.CHUNK, 16])
def test_the_ported_challenger_signs_the_bytes_serve_signs(monkeypatch, chunk):
    """Sixteen frames of 16 labels take two windows, the prover taking the first before the second is sent."""
    old = serve(monkeypatch, PARAMS, b"same key")
    r = wipe(PARAMS, b"same key", chunk=chunk)
    assert r.status == "completed", r.outcomes
    out = r.outcomes[C]["outcome"]
    assert canonical(out["signed"]) == canonical(old.signed.to_json())
    new = WP.signed_of(out["signed"])
    assert new.message() == old.signed.message() and new.signature == old.signed.signature
    assert (out["ended"], out["reads"]) == (sw.COMPLETE, PARAMS.k)
    fills = [e for e in r.transcripts[C] if e.dir == "out" and e.channel == P and e.kind == "private"]
    assert len(fills) == PARAMS.labels // min(chunk, PARAMS.labels)


def test_an_honest_prover_on_host_memory_is_accepted():
    store = lambda p: WS.StoreProver(p, WS.HostMemory(), clock=itertools.count(0, 1).__next__)  # noqa: E731
    params = sw.WipeParams(1 << 16, 64, 1, 10_000, (("host-dram", (1 << 16) * sw.LABEL_BYTES),))
    r = wipe(params, b"host memory", mode="simulated", chunk=1 << 12, store=store)
    assert r.status == "completed", r.outcomes
    assert r.outcomes[P] == {"outcome": {"answered": 64}, "refused": None}
    assert verdict(r.outcomes[C]["outcome"], b"host memory")["verdict"] == "accept"


def test_a_prover_keeping_foreign_content_is_rejected():
    r = wipe(PARAMS, b"foreign", store=W.keeping(PARAMS.labels // 2))
    out = r.outcomes[C]["outcome"]
    assert out["ended"] == sw.WRONG and verdict(out, b"foreign")["verdict"] == "reject"


def test_parameters_that_name_the_reads_are_refused():
    """The challenger times the reads its machine gives it; a run whose parameters name them is refused by both
    parties before any fill, unsigned."""
    r = wipe(PARAMS, b"reads", request=WP.request_of("n0", PARAMS) | {"k": 9})
    assert r.outcomes[C] == {"outcome": {"ended": None, "signed": None}, "refused": "schema"}
    assert r.outcomes[P] == {"outcome": {"answered": 0}, "refused": "schema"}
    assert not [e for e in r.transcripts[C] if e.dir == "out" and e.channel == P and e.kind == "private"]


class Leaving(WPS.Prover):
    def on(self, r):
        if r.message.type == "Read" and self.answered == 3:
            return [Finish({"answered": 3}, "left")]
        return super().on(r)


def test_a_prover_that_leaves_leaves_the_session_unsigned():
    r = wipe(PARAMS, b"left", prover=Leaving)
    assert r.outcomes[C]["refused"] == "closed"
    out = r.outcomes[C]["outcome"]
    assert (out["reads"], out["signed"]) == (3, None)
    assert verdict(out, b"left")["verdict"] == "undecided"


def test_large_fill_frames_are_kept_as_digests_and_the_challenger_replays(tmp_path):
    params = sw.WipeParams(1 << 12, 16, 1, 10_000, (("host-dram", (1 << 12) * sw.LABEL_BYTES),))
    r = wipe(params, b"replay", mode="simulated", chunk=1 << 11, state=tmp_path)
    assert r.status == "completed", r.outcomes
    t = r.transcripts[C]
    fills = [e for e in t if e.dir == "out" and e.channel == P and e.kind == "private"]
    assert [(e.elided, len(e.data)) for e in fills] == [(True, 64), (True, 64)]
    assert all(e.kind != "private" for e in t.public())
    caps = {C: challenger_caps(b"replay", params)}
    protocol = ProtocolDef(WPS.MANIFEST, {C: lambda: WPS.MemoryChallengerParty(1 << 11), P: WPS.Prover})
    again = run({"protocol": WPS.ID, "params": WP.request_of("n0", params)}, [C], "replay", None,
                protocols={WPS.ID: protocol}, capabilities=caps, replay_dir=tmp_path / "runs/1")
    assert (again.status, again.divergence) == ("completed", None)
    assert again.outcomes[C]["outcome"]["signed"] == r.outcomes[C]["outcome"]["signed"]


@pytest.mark.slow
@pytest.mark.skipif(not settings.SLOW, reason="about two minutes for a 2 GiB fill: VERITY_POUS_SLOW=1")
def test_a_two_gib_fill_on_host_memory():
    """The deployment layer streams a whole 2 GiB fill to a prover on host memory a window at a time, its transcripts
    a digest a frame, and the session is accepted."""
    labels = (2 << 30) // sw.LABEL_BYTES
    params = sw.WipeParams(labels, 64, 1, 10_000, (("host-dram", 2 << 30),))
    store = lambda p: WS.StoreProver(p, WS.HostMemory(), clock=itertools.count(0, 1).__next__)  # noqa: E731
    r = wipe(params, b"two gib", mode="simulated", store=store)
    assert r.status == "completed", r.outcomes
    assert verdict(r.outcomes[C]["outcome"], b"two gib")["verdict"] == "accept"
    assert len(r.transcripts[C].to_bytes()) < 16 << 20
