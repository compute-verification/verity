"""PoUW's read of the network certifier's salt (`verity.core.protocols.pouw.salt`) against the certifier itself
(`certifier.active.Proxy`, driven tick by tick on an injected clock): the salt's form, the record's and the prover id's
are pinned; a context names its PoUW window; `revealed` gives PoUW window p's salt on p's registration once p is
published, once, and fails closed when it is unregistered, early, late, revealed already or missing; `pouw_window` takes
one whole PoUW window aligned at 0 and nothing else; `Epoch.certified` applies `pouw_salt` of the reveal, the window,
the stream and the prover, 64 bytes, which the seeds and the identifier take, with the challenger's draw coin; two
registrations of one window are salted apart; `core_check.pass_epoch` passes a certified window's pass only when its
salt is that of the certifier's reveal and its weights, stream and prover those the prover recorded with the certifier,
signed, before it drew that salt (`Proxy.commit_weights`, read back by `committed`); `Registrations` admits a window
once per stream and prover; a certified window credits only units that read its salt (`unit_reason`); and a live
window's open root (`window_open_root`) is pinned. The prover salts its pass from the salt it receives in-band (`certifier`'s `drawn`), so the
verifier's one reveal comes after the registration that names the pass."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import random

import pytest

import epoch_coins as E
from verity.ml.crypto import lms
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.active import Proxy
from verity.core.protocols.certifier.certificate import LmsKey
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import identifier as I
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw.audit import Epoch, Workload, commit_rows, weight_binding, weights_root
from verity.core.protocols.pouw.coins import coin_reason
from verity.core.protocols.pouw.circuit import seeds as SD
from verity.core.protocols.pouw.protocol import DomainError, Matmul, Shape
from verity.core.protocols.pouw.schemes import ncp
from verity.core.protocols.sampled_proofs.one_stage import registration as R
from verity.core.protocols.sampled_proofs.service import Registration

EGRESS = GridParams(buckets=6, allocation=3, frame_bytes=512)
INGRESS = GridParams(buckets=6, allocation=2, frame_bytes=512)
SYNC_SET = SyncSet((0, 1), (1, 2))
SYNCS = [ClockSync(0, 1)] * 4
MANIFEST = hashlib.sha512(b"the pass's manifest").hexdigest()
STREAM = hashlib.sha512(b"a stream").digest()
#: a certifier window's seconds: PoUW window p of k ends at (p + 1)·k·W, and its reveal deadline is k·W after
W = INGRESS.buckets * INGRESS.bucket_ms / 1000
#: the deployed LMS scheme (N = 64, W = 1) on 2^6 leaves, so a key takes a fraction of a second
SIGNING = lms.Params(64, 1, 6)


def prover_key(seed: bytes):
    """A prover's LMS key as the certifier registers it (`LmsKey`), and a signer at one-time keys 0, 1, ... in turn."""
    k, leaves = lms.keygen(SIGNING, hashlib.sha512(seed).digest()), itertools.count()
    return LmsKey(SIGNING, lms.public_key(k)), lambda message: lms.sign_at(k, next(leaves), message).encode()


KEY, SIGN = prover_key(b"the prover")
_, FORGE = prover_key(b"someone else")
PROVER = PS.prover_id(KEY.public)


def signed(c: Proxy, p: int, record: bytes, sign=SIGN, context: bytes | None = None) -> bytes:
    """`sign`'s signature of `record` for PoUW window p behind `c`, under `context` (by default `c`'s, the stream)."""
    ctx = c.prover_context if context is None else context
    return sign(active.weights_message(c.bucketer.certifier.link, p, record, ctx))


def registration(p: int, ctx: str | None = None, window: dict | None = None) -> Registration:
    """PoUW's registration of window p as the proof service returns it: a record whose served window's context names p,
    with its receipt and digest."""
    w = {"kind": "served", "context": PS.context("served-zk/test", p, MANIFEST) if ctx is None else ctx, "public": {}}
    record = {"window": w if window is None else window}
    return Registration(record, R.receipt(record, "test", received=str(p)), R.digest(record), p)


def certifier(windows: int, records: dict[int, bytes] | None = None, clock=lambda: 0.0,
              **kwargs) -> tuple[Proxy, dict[int, bytes]]:
    """The certifier with PoUW's reader and the prover's key, the stream its context, on clock `clock`, handed `records`
    by PoUW window before its first tick (`commit_weights`, signed by the prover), after `windows` certifier windows of
    ingress ticks, and the salts it drew by PoUW window: what the prover's server receives in-band."""
    drawn: dict[int, bytes] = {}

    def source(p: int, n: int) -> bytes:
        drawn[p] = hashlib.shake_256(b"certifier salt %d" % p).digest(n)
        return drawn[p]

    c = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, ingress_link="requests", clock=clock,
              **{"salt": source, "registered": PS.registered, "prover_key": KEY, "prover_context": STREAM, **kwargs})
    for p, record in (records or {}).items():
        c.commit_weights(p, record, signed(c, p, record))
    for _ in range(windows * INGRESS.buckets):
        c.bucketer.tick()
    return c, drawn


def test_pouw_salt_form():
    """The salt PoUW applies in window p for a registration of a stream and a prover is SHA-512(b"verity/pouw-salt/v1" ‖
    reveal ‖ u64be(p) ‖ stream ‖ prover), what PoUW records for p before it SHA-512(b"verity/pouw-record/v1" ‖ weights'
    root ‖ stream ‖ prover), and the prover's id SHA-512(b"verity/pouw-prover/v1" ‖ its public key), each pinned on one
    vector, every part of the first two fixed-width: a reveal, a stream, a prover or a weights' root that isn't 64 bytes,
    and a window that isn't a u64, are refused, as is a public key that isn't non-empty bytes."""
    reveal, stream, prover, root = (bytes(range(i, i + 64)) for i in (0, 64, 128, 192))
    assert PS.pouw_salt(reveal, 3, stream, prover).hex() == (
        "7c43e79ad05bb29ccb777671767a31fac039bf1cd89279318d8232b97c3094cd"
        "ad0f7eed8b3a29b773d37c0bc8d73da3e7bf2d35452ebe3befa0e8f4fef167ac")
    assert PS.pouw_salt(reveal, 3, stream, prover) == hashlib.sha512(
        b"verity/pouw-salt/v1" + reveal + (3).to_bytes(8, "big") + stream + prover).digest()
    assert PS.pouw_record(root, stream, prover).hex() == (
        "43e12ade4369d0f34e766035df309f57b880fb3d4281983184e52778ff847c56"
        "cad7f390714e34a523be4f1a7a3f2df59c51d37cdfd650fc5f49d24623b3f6c6")
    assert PS.pouw_record(root, stream, prover) == hashlib.sha512(b"verity/pouw-record/v1" + root + stream + prover).digest()
    assert PS.pouw_salt(reveal, 4, stream, prover) != PS.pouw_salt(reveal, 3, stream, prover)
    for bad in (reveal[:32], reveal + b"\0", bytearray(reveal), reveal.hex()):
        with pytest.raises(DomainError, match="64 bytes"):
            PS.pouw_salt(bad, 3, stream, prover)
        with pytest.raises(DomainError, match="64 bytes"):
            PS.pouw_record(bad, stream, prover)
        with pytest.raises(DomainError, match="stream is a 64-byte digest"):
            PS.pouw_salt(reveal, 3, bad, prover)
        with pytest.raises(DomainError, match="prover is a 64-byte id"):
            PS.pouw_record(root, stream, bad)
    for bad in (-1, 1 << 64, True, 3.0):
        with pytest.raises(DomainError, match="u64"):
            PS.pouw_salt(reveal, bad, stream, prover)
    assert PS.prover_id(root).hex() == (
        "4c0208343944eb3dfe641fbf4735121a83f926c667f396fe970d4948781e2527"
        "793979a2186ab7117ab2d7720e3ae13816715d3df65f7a15df1e316066922cc8")
    assert PS.prover_id(root) == hashlib.sha512(b"verity/pouw-prover/v1" + root).digest()
    for bad in (b"", bytearray(root), root.hex(), None):
        with pytest.raises(DomainError, match="public key"):
            PS.prover_id(bad)


def test_window_open_root_form():
    """A live window's open root is SHA-512(b"verity/pouw-window-open/v1" ‖ weights' root ‖ round-0 request table's root
    ‖ SHA-512(the session's ASCII id)), every part 64 bytes, pinned on one vector: another weights' root, request table
    or session is another root, a root that isn't 64 bytes is refused, and so is a session id that isn't 1 to 64 of
    [A-Za-z0-9._:-] starting alphanumeric."""
    weights, requests, session = bytes(range(64)), bytes(range(64, 128)), "s20261020T114752Z-9e52"
    root = PS.window_open_root(weights, requests, session)
    assert root.hex() == (
        "b3662ee71a379a471ac908a360224aff68114b81ead11a41c9fed0b197f363fc"
        "d71005d30b1e82370f5459c68389f33d4e873baa8b6e8eeb886cb66f2c47328e")
    assert root == hashlib.sha512(b"verity/pouw-window-open/v1" + weights + requests
                                  + hashlib.sha512(session.encode()).digest()).digest()
    assert len({root, PS.window_open_root(requests, weights, session), PS.window_open_root(weights, weights, session),
                PS.window_open_root(weights, requests, "s20261020T114752Z-9e53")}) == 4
    for bad in (weights[:32], weights + b"\0", bytearray(weights), weights.hex(), None):
        with pytest.raises(DomainError, match="weights' root is 64 bytes"):
            PS.window_open_root(bad, requests, session)
        with pytest.raises(DomainError, match="request table's root is 64 bytes"):
            PS.window_open_root(weights, bad, session)
    for bad in ("", "-s1", "s1 ", "s1\n", "s/1", "é", "s" * 65, b"s1", None, 1):
        with pytest.raises(DomainError, match="session's id"):
            PS.window_open_root(weights, requests, bad)
    assert PS.window_open_root(weights, requests, "s" * 64) != root


def test_epoch_open_root_form():
    """A Pearl-C salt epoch's open root is SHA-512(b"verity/pouw-epoch-open/v1" ‖ weights' root ‖ u64be(k)), 97 bytes
    hashed, pinned on one vector: another weights' root or epoch is another root, no epoch root is a window-open root's
    tag, a root that isn't 64 bytes is refused, and so is an epoch number that isn't an integer in [0, 2^64)."""
    weights = bytes(range(64))
    root = PS.epoch_open_root(weights, 0)
    assert root.hex() == (
        "e0dac4297e47d4ac5b82943ee85b7f76269ffc92c0da80a2c89a76813910064c"
        "97853737d76148923908e6a266f73f75abd3a8b9c512d38a726a3656e7067701")
    assert PS.EPOCH_OPEN_TAG == b"verity/pouw-epoch-open/v1" and len(PS.EPOCH_OPEN_TAG + weights + bytes(8)) == 97
    assert PS.epoch_open_root(weights, 258) == hashlib.sha512(PS.EPOCH_OPEN_TAG + weights + bytes(6) + b"\1\2").digest()
    assert PS.epoch_open_root(weights, (1 << 64) - 1) == hashlib.sha512(PS.EPOCH_OPEN_TAG + weights + b"\xff" * 8).digest()
    assert len({root, PS.epoch_open_root(weights, 1), PS.epoch_open_root(bytes(64), 0)}) == 3
    for bad in (weights[:32], weights + b"\0", bytearray(weights), weights.hex(), None):
        with pytest.raises(DomainError, match="weights' root is 64 bytes"):
            PS.epoch_open_root(bad, 0)
    for bad in (-1, 1 << 64, 1.0, True, "0", None):
        with pytest.raises(DomainError, match="salt epoch's number"):
            PS.epoch_open_root(weights, bad)


def test_anchored_epoch_root_form():
    """An H1 window's salt epoch's open root is SHA-512(b"verity/pouw-epoch-open/v2" ‖ W ‖ X ‖ u64be(k)), 161 bytes
    hashed, pinned on one vector: another W, X or epoch is another root, none is a Pearl-C epoch's root, and a root
    that isn't 64 bytes or an epoch number that isn't an integer in [0, 2^64) is refused."""
    weights, anchors = bytes(range(64)), bytes(range(64, 128))
    root = PS.anchored_epoch_root(weights, anchors, 0)
    assert root.hex() == (
        "9a94dc1e8c5ea75e23145eb1f4f384ce8f7b516f555d746ce45ceba6bfdea693"
        "09f2889235afecd91aa709280bbbff11148c45ec97b2cad9c11a364c39a619ed")
    assert PS.ANCHORED_EPOCH_TAG == b"verity/pouw-epoch-open/v2"
    assert len(PS.ANCHORED_EPOCH_TAG + weights + anchors + bytes(8)) == 161
    assert PS.anchored_epoch_root(weights, anchors, 258) == hashlib.sha512(
        PS.ANCHORED_EPOCH_TAG + weights + anchors + bytes(6) + b"\1\2").digest()
    assert len({root, PS.anchored_epoch_root(weights, anchors, 1), PS.anchored_epoch_root(bytes(64), anchors, 0),
                PS.anchored_epoch_root(weights, bytes(64), 0), PS.anchored_epoch_root(anchors, weights, 0),
                PS.epoch_open_root(weights, 0)}) == 6
    for bad in (weights[:32], weights + b"\0", bytearray(weights), weights.hex(), None):
        with pytest.raises(DomainError, match="weights' root is 64 bytes"):
            PS.anchored_epoch_root(bad, anchors, 0)
        with pytest.raises(DomainError, match="anchors' root is 64 bytes"):
            PS.anchored_epoch_root(weights, bad, 0)
    for bad in (-1, 1 << 64, 1.0, True, "0", None):
        with pytest.raises(DomainError, match="salt epoch's number"):
            PS.anchored_epoch_root(weights, anchors, bad)


def test_a_context_names_its_pouw_window():
    """`context` names the window before the manifest's digest, which stays the last segment (`pass_epoch`'s), and
    `window_of` reads it back; anything else names no window."""
    ctx = PS.context("served-zk/run", 7, MANIFEST)
    assert ctx == f"served-zk/run/pouw-window/7/{MANIFEST}" and PS.window_of(ctx) == 7
    assert PS.window_of(PS.context("", 0, MANIFEST)) == 0 and PS.window_of(f"pouw-window/{(1 << 64) - 1}/m") == (1 << 64) - 1
    for bad in (f"run/pouw-window/07/{MANIFEST}", f"run/pouw-window/-1/{MANIFEST}", f"run/pouw-window/x/{MANIFEST}",
                f"run/pouw-window/{1 << 64}/{MANIFEST}", f"run/pouw-window/ 7/{MANIFEST}", "run/pouw-window/7",
                f"run/window/7/{MANIFEST}", f"served-zk/run/{MANIFEST}", "", None, 7):
        assert PS.window_of(bad) is None, bad
    with pytest.raises(DomainError, match="u64"):
        PS.context("run", -1, MANIFEST)


def test_the_certifier_reveals_a_pouw_window_on_its_registration():
    """With PoUW windows of two certifier windows and PoUW's reader, the certifier reveals PoUW window 0's salt once
    windows 0 and 1 are published, on PoUW's registration of window 0, and `revealed` gives it whole; PoUW window 1's
    after windows 2 and 3."""
    c, drawn = certifier(4, pouw_window=2)
    assert sorted(drawn) == [0, 1] and {len(s) for s in drawn.values()} == {PS.SALT_BYTES}
    for p in (0, 1):
        assert PS.registered(registration(p)) == p
        assert PS.revealed(c, p, registration(p)) == drawn[p]


def test_revealed_fails_closed():
    """`revealed` refuses (`Unrevealed`) a window that is unregistered: no registration, another window's, the network
    certifier's own provenance, a context that names no window, a window with another key; and, through the
    certifier, a receipt or digest that isn't the record's, and PoUW's registration to a certifier without PoUW's
    reader. It refuses one that is early: before its last certifier window is published. One that is late: its reveal
    deadline, one PoUW window after its end on the certifier's clock, has passed (the same reveal at the deadline is
    given). One revealed already: a second reveal on the same registration or another naming the window, so behind one
    certifier one registration of a window gets its salt. And one whose salt is missing: never drawn (a certifier back
    from maintenance after its first window), the salt off, or a salt of another width. The certifier's refusals of
    time and the once are matched by PoUW's words only, so its own wording isn't pinned here."""
    c, _ = certifier(2, pouw_window=2)
    reg = registration(0)
    other = {"window": {"kind": "served", "context": PS.context("served-zk/test", 1, MANIFEST), "public": {}}}
    unregistered = [None, registration(1), registration(0, window={"kind": "served", "window": 0}),
                    registration(0, ctx=f"served-zk/test/{MANIFEST}"),
                    registration(0, window={**reg.record["window"], "row": 3}),
                    registration(0, window={**reg.record["window"], "kind": "stand-in"}),
                    registration(0, window={**reg.record["window"], "public": []}),
                    {"record": reg.record, "receipt": reg.receipt}]
    for r in unregistered:
        with pytest.raises(PS.Unrevealed, match="unregistered"):
            PS.revealed(c, 0, r)
    receipts = [dataclasses.replace(reg, receipt={**reg.receipt, "registration": R.digest(other)}),
                dataclasses.replace(reg, digest=R.digest(other)), dataclasses.replace(reg, receipt=None),
                dataclasses.replace(reg, receipt={**reg.receipt, "format": "verity/registration-receipt/v0"})]
    for r in receipts:
        with pytest.raises(PS.Unrevealed, match="no registration"):
            PS.revealed(c, 0, r)
    network, _ = certifier(2, pouw_window=2, registered=active.served_window)
    with pytest.raises(PS.Unrevealed, match="no registration"):
        PS.revealed(network, 0, reg)

    early, _ = certifier(1, pouw_window=2)
    with pytest.raises(PS.Unrevealed, match="published"):
        PS.revealed(early, 0, reg)
    with pytest.raises(PS.Unrevealed, match="published"):
        PS.revealed(c, 1, registration(1))

    refused, record = "the certifier doesn't reveal PoUW window 0's salt", PS.pouw_record(bytes(64), STREAM, PROVER)
    late, _ = certifier(2, {0: record}, clock=lambda: 4 * W + 1e-3, pouw_window=2)
    with pytest.raises(PS.Unrevealed, match=refused):
        PS.revealed(late, 0, reg)
    with pytest.raises(PS.Unrevealed, match="the certifier doesn't give what PoUW recorded for window 0"):
        PS.committed(late, 0, reg)
    timely, drawn = certifier(2, {0: record}, clock=lambda: 4 * W, pouw_window=2)
    assert PS.revealed(timely, 0, reg) == drawn[0]
    for again in (reg, registration(0, ctx=PS.context("served-zk/another", 0, MANIFEST))):
        with pytest.raises(PS.Unrevealed, match=refused):
            PS.revealed(timely, 0, again)
    assert PS.committed(timely, 0, reg) == record

    back, drawn = certifier(2, pouw_window=2, start=1)
    assert sorted(drawn) == [1]
    with pytest.raises(PS.Unrevealed, match="wasn't drawn"):
        PS.revealed(back, 0, reg)
    off, _ = certifier(2, pouw_window=2, salt=None)
    with pytest.raises(PS.Unrevealed, match="salt is off"):
        PS.revealed(off, 0, reg)
    narrow, _ = certifier(2, pouw_window=2, salt_bits=256)
    with pytest.raises(PS.Unrevealed, match="missing"):
        PS.revealed(narrow, 0, reg)

    @dataclasses.dataclass
    class Gives:
        salt: object

        def reveal(self, window, registration):
            return self.salt

    for salt in (None, bytearray(64), bytes(63), "00" * 64):
        with pytest.raises(PS.Unrevealed, match="missing"):
            PS.revealed(Gives(salt), 0, reg)
    assert PS.revealed(Gives(bytes(64)), 0, reg) == bytes(64)


def test_pouw_window_is_one_whole_pouw_window_aligned_at_0():
    """Certifier windows p·k … p·k + k − 1, in order, make PoUW window p; any other run of windows is refused, as is a k
    that isn't a positive integer."""
    assert PS.pouw_window(range(0, 1), 1) == 0 and PS.pouw_window([4, 5], 2) == 2 and PS.pouw_window(range(9, 12), 3) == 3
    assert PS.pouw_window([7], 1) == 7
    for ws, k in (([], 2), ([1, 2], 2), ([0], 2), ([0, 1, 2], 2), ([0, 2], 2), ([1, 0], 2), ([2, 3, 4], 3),
                  ([False, True], 2), ([-2, -1], 2), ([0.0, 1.0], 2), ([2 * (1 << 64), 2 * (1 << 64) + 1], 2)):
        with pytest.raises(DomainError, match="not one PoUW window"):
            PS.pouw_window(ws, k)
    for k in (0, -1, 2.0, True, None):
        with pytest.raises(DomainError, match="positive number"):
            PS.pouw_window([0, 1], k)


def setup():
    rng = random.Random(3)
    work = Workload((Matmul(0, Shape(32, 64, 64), 7),))
    rows = [ncp.int7_bytes([rng.randint(-64, 63) for _ in range(64)]) for _ in range(64)]
    return work, {7: commit_rows(weight_binding(7), -2, rows).commitment.root}


def test_a_certified_epoch_applies_its_pouw_windows_salt():
    """`Epoch.certified` of PoUW window 1's certifier windows (2 and 3, k = 2) under a stream and a prover has salt
    `pouw_salt(reveal, 1, stream, prover)`, 64 bytes: the seeds take it and Epoch.start's 32 bytes they refuse, the
    identifier names it whole, and the challenger's draw coin opens against its coins. Misaligned windows, a reveal,
    stream or prover of another width and weights that aren't the workload's are refused."""
    work, roots = setup()
    c, drawn = certifier(4, pouw_window=2)
    reveal = PS.revealed(c, 1, registration(1))
    ch = E.challenger()
    epoch = Epoch.certified(work, roots, [2, 3], 2, reveal, STREAM, PROVER, coins=ch.commitment)
    assert epoch.salt == PS.pouw_salt(drawn[1], 1, STREAM, PROVER) and len(epoch.salt) == 64 and epoch.coins == ch.commitment
    assert epoch.salt != Epoch.certified(work, roots, [0, 1], 2, reveal, STREAM, PROVER).salt

    root = hashlib.sha512(b"root").digest()
    assert SD.seed(epoch.salt, 0, root, root) != SD.receipt_seed(epoch.salt, 0, root, root)
    started = Epoch.start(work, roots, E.COINS, E.challenger().open("salt", epoch.weights_root))
    assert len(started.salt) == 32
    for f in (SD.seed, SD.receipt_seed):
        with pytest.raises(ValueError, match="64-byte"):
            f(started.salt, 0, root, root)
    assert I.identifier_v1("ncp-v1", epoch)["salt"] == epoch.salt.hex()

    assert coin_reason(epoch.coins, ch.open("draw", root), "draw", root) is None

    for ws in ([1, 2], [2], [2, 3, 4]):
        with pytest.raises(DomainError, match="not one PoUW window"):
            Epoch.certified(work, roots, ws, 2, reveal, STREAM, PROVER)
    with pytest.raises(DomainError, match="64 bytes"):
        Epoch.certified(work, roots, [2, 3], 2, reveal[:32], STREAM, PROVER)
    with pytest.raises(DomainError, match="stream"):
        Epoch.certified(work, roots, [2, 3], 2, reveal, STREAM[:32], PROVER)
    with pytest.raises(DomainError, match="prover"):
        Epoch.certified(work, roots, [2, 3], 2, reveal, STREAM, PROVER.hex())
    with pytest.raises(DomainError, match="every weight"):
        Epoch.certified(work, {**roots, 8: root}, [2, 3], 2, reveal, STREAM, PROVER)


def manifest(work: Workload, roots: dict[int, bytes], p: int, salt: bytes, run: str = "r") -> tuple[bytes, str]:
    """A certified pass's manifest over `work` and its weights' `roots`, salted `salt`, and its window's context, which
    names PoUW window p and the manifest."""
    man = {"schema": CC.MANIFEST_CERTIFIED, "salt": salt.hex(), "run": run,
           "matmuls": [{"index": u.index, "m": u.shape.m, "k": u.shape.k, "n": u.shape.n, "weight": u.weight}
                       for u in work.matmuls],
           "weights": {str(w): {"root": r.hex()} for w, r in roots.items()}}
    raw = json.dumps(man).encode()
    return raw, PS.context(f"served-zk/{run}", p, hashlib.sha512(raw).hexdigest())


def test_a_certified_window_passes_only_a_pass_salted_by_its_certifiers_reveal():
    """`core_check.pass_epoch` in a certified window, its context naming PoUW window 1 and the pass, with the reveal
    `revealed` and the record `committed` read from the certifier on that registration, and the window's stream and
    prover: the pass the prover salted by `Epoch.certified` of windows 2 and 3 from the salt it received passes
    (``certifier``). Salted by window 0's epoch, by the raw reveal, or checked against window 0's reveal, it doesn't, nor
    does it pass at a verifier without the reveal."""
    work, roots = setup()
    c, drawn = certifier(4, {1: PS.pouw_record(weights_root(roots), STREAM, PROVER)}, pouw_window=2)

    raw, ctx = manifest(work, roots, 1, Epoch.certified(work, roots, [2, 3], 2, drawn[1], STREAM, PROVER).salt)
    reveal = PS.revealed(c, 1, registration(1, ctx=ctx))
    assert reveal == drawn[1]
    record = PS.committed(c, 1, registration(1, ctx=ctx))
    bound = {"committed": record, "stream": STREAM, "prover": PROVER}
    assert CC.pass_epoch(None, raw, ctx, reveal=reveal, **bound) == ("certifier", None)

    mismatch = ("coin", "the pass's salt is not the certifier's for its window, stream and prover")
    for salt in (Epoch.certified(work, roots, [0, 1], 2, reveal, STREAM, PROVER).salt, reveal):
        assert CC.pass_epoch(None, *manifest(work, roots, 1, salt), reveal=reveal, **bound) == (None, mismatch)
    assert CC.pass_epoch(None, raw, ctx, reveal=PS.revealed(c, 0, registration(0)), **bound) == (None, mismatch)
    how, refused = CC.pass_epoch(None, raw, ctx)
    assert how is None and refused[0] == "coin" and refused[1].startswith(f"the pass's manifest is {CC.MANIFEST_CERTIFIED}")


def test_two_registrations_of_one_window_are_salted_apart():
    """Two registrations of PoUW window 1 behind one certifier, of the same weights and the same operands, differing only
    in their stream or only in their prover, get different salts, and so a different seed and receipt seed at every
    matmul index: to a unit that reads its seed, a tile computed once is no other registration's. Each one's pass is
    refused at the other's verifier, whichever the certifier recorded. The certifier reveals the window once, so its
    second registration gets no salt at all. Each certifier draws its own salts, so two certifiers' salts for window 1
    (two proxies' lifetimes, each counting from its origin) differ too, with no epoch bound."""
    work, roots = setup()
    root = weights_root(roots)
    other_stream, other_prover = hashlib.sha512(b"another stream").digest(), PS.prover_id(prover_key(b"another")[0].public)
    bindings = [(STREAM, PROVER), (other_stream, PROVER), (STREAM, other_prover)]
    c, _ = certifier(4, pouw_window=2)
    reveal = PS.revealed(c, 1, registration(1))
    with pytest.raises(PS.Unrevealed, match="the certifier doesn't reveal PoUW window 1's salt"):
        PS.revealed(c, 1, registration(1, ctx=PS.context("served-zk/another", 1, MANIFEST)))
    salts = [Epoch.certified(work, roots, [2, 3], 2, reveal, *b).salt for b in bindings]
    assert len(set(salts)) == 3
    operands = [hashlib.sha512(b"A %d" % j).digest() for j in range(4)]
    for f in (SD.seed, SD.receipt_seed):
        assert all(len({f(s, j, operands[j], root) for s in salts}) == 3 for j in range(4))

    for (stream, prover), salt in zip(bindings, salts):
        raw, ctx = manifest(work, roots, 1, salt)
        for (s2, v2) in bindings:
            got = CC.pass_epoch(None, raw, ctx, reveal=reveal, committed=PS.pouw_record(root, s2, v2), stream=s2, prover=v2)
            assert got == (("certifier", None) if (s2, v2) == (stream, prover) else
                           (None, ("coin", "the pass's salt is not the certifier's for its window, stream and prover")))
            if (s2, v2) != (stream, prover):
                assert CC.pass_epoch(None, raw, ctx, reveal=reveal, committed=PS.pouw_record(root, stream, prover),
                                     stream=s2, prover=v2)[1][1].startswith("the pass's weights, stream and prover are not")

    lives = [certifier(4, pouw_window=2, salt=active.fresh_salt)[0] for _ in range(2)]
    assert PS.pouw_salt(PS.revealed(lives[0], 1, registration(1)), 1, STREAM, PROVER) != \
        PS.pouw_salt(PS.revealed(lives[1], 1, registration(1)), 1, STREAM, PROVER)


def test_a_certified_window_passes_only_weights_and_a_binding_recorded_before_its_salt():
    """The certifier's salt binds no weights, so a certified window passes only weights, a stream and a prover the
    prover recorded with the certifier before it drew the window's salt (`pouw_record`, through `Proxy.commit_weights`,
    signed by the prover's registered key over `weights_message` under the stream), which `committed` reads on
    `reveal`'s conditions. End to end against the certifier, PoUW windows of two certifier windows: window 0's record
    is refused unsigned, signed by another key, signed for another window or under another stream's context; signed,
    made before its draw, it is read back and its pass passes, and a second record for it is refused. Window 1's, made
    after its draw, is refused, so `committed` refuses window 1 and its pass, salted by its own salt, is refused. A pass
    over other weights, salted the same since the salt doesn't read them, is refused, as are a stream or prover other
    than those recorded, a registration that isn't the window's and a certifier that gives anything but 64 bytes.
    `recorded_reason`, the record's rule short of the reveal, which a challenger runs before it asks for one, refuses
    what `pass_epoch` refuses of the record, in its words. The certifier's refusals are matched by type and PoUW's by
    its own words, so neither's wording is pinned to the other's."""
    work, roots = setup()
    root = weights_root(roots)
    record = PS.pouw_record(root, STREAM, PROVER)
    c, drawn = certifier(0, pouw_window=2)
    for bad in (b"", None, signed(c, 0, record, sign=FORGE), signed(c, 1, record),
                signed(c, 0, record, context=hashlib.sha512(b"another stream").digest())):
        with pytest.raises(RuntimeError):
            c.commit_weights(0, record, bad)
    c.commit_weights(0, record, signed(c, 0, record))
    other = PS.pouw_record(root, STREAM, PS.prover_id(prover_key(b"another")[0].public))
    with pytest.raises(RuntimeError):
        c.commit_weights(0, other, signed(c, 0, other))
    for _ in range(2 * INGRESS.buckets + 1):   # windows 0 and 1, then window 2's first tick: PoUW window 1's salt
        c.bucketer.tick()
    with pytest.raises(RuntimeError):
        c.commit_weights(1, record, signed(c, 1, record))
    for _ in range(2 * INGRESS.buckets - 1):
        c.bucketer.tick()

    raw, ctx = manifest(work, roots, 0, Epoch.certified(work, roots, [0, 1], 2, drawn[0], STREAM, PROVER).salt)
    reveal = PS.revealed(c, 0, registration(0, ctx=ctx))
    got = PS.committed(c, 0, registration(0, ctx=ctx))
    assert got == record
    assert CC.pass_epoch(None, raw, ctx, reveal=reveal, committed=got, stream=STREAM, prover=PROVER) == ("certifier", None)

    raw1, ctx1 = manifest(work, roots, 1, Epoch.certified(work, roots, [2, 3], 2, drawn[1], STREAM, PROVER).salt)
    late = PS.revealed(c, 1, registration(1, ctx=ctx1))
    with pytest.raises(PS.Unrevealed, match="the certifier doesn't give what PoUW recorded for window 1"):
        PS.committed(c, 1, registration(1, ctx=ctx1))
    unrecorded = (None, ("coin", "the verifier holds no record the certifier made before it drew the window's salt"))
    assert CC.pass_epoch(None, raw1, ctx1, reveal=late, committed=None, stream=STREAM, prover=PROVER) == unrecorded
    with pytest.raises(PS.Unrevealed, match="unregistered"):
        PS.committed(c, 0, registration(1))

    @dataclasses.dataclass
    class Gives:
        record: object

        def committed(self, window, registration):
            return self.record

    for bad in (record[:32], record.hex(), bytearray(record), None):
        with pytest.raises(PS.Unrevealed, match="is not 64 bytes"):
            PS.committed(Gives(bad), 0, registration(0))
    assert PS.committed(Gives(record), 0, registration(0)) == record

    later = {w: hashlib.sha512(r).digest() for w, r in roots.items()}
    assert Epoch.certified(work, later, [0, 1], 2, reveal, STREAM, PROVER).salt == \
        Epoch.certified(work, roots, [0, 1], 2, reveal, STREAM, PROVER).salt
    other = (None, ("coin", "the pass's weights, stream and prover are not those the certifier recorded before it drew "
                            "the window's salt"))
    assert CC.pass_epoch(None, *manifest(work, later, 0, Epoch.certified(work, later, [0, 1], 2, reveal, STREAM, PROVER).salt),
                         reveal=reveal, committed=got, stream=STREAM, prover=PROVER) == other
    for stream, prover in ((hashlib.sha512(STREAM).digest(), PROVER), (STREAM, hashlib.sha512(PROVER).digest())):
        assert CC.pass_epoch(None, raw, ctx, reveal=reveal, committed=got, stream=stream, prover=prover) == other
    for none in (None, record[:32]):
        assert CC.pass_epoch(None, raw, ctx, reveal=reveal, committed=none, stream=STREAM, prover=PROVER) == unrecorded
    for stream, prover in ((None, PROVER), (STREAM, None), (STREAM[:32], PROVER), (STREAM, PROVER + b"\0")):
        assert CC.pass_epoch(None, raw, ctx, reveal=reveal, committed=got, stream=stream, prover=prover) == \
            (None, ("coin", "the verifier holds no 64-byte stream digest and 64-byte prover id for the window"))

    assert CC.recorded_reason(raw, got, STREAM, PROVER) is None
    for r, bound in ((raw, (None, STREAM, PROVER)), (raw, (got, hashlib.sha512(STREAM).digest(), PROVER)),
                     (raw, (got, STREAM, PROVER[:32])), (None, (got, STREAM, PROVER)),
                     (json.dumps({**json.loads(raw), "schema": CC.MANIFEST_SCHEMA}).encode(), (got, STREAM, PROVER))):
        _, (_, why) = CC.pass_epoch(None, r, ctx, reveal=reveal, **dict(zip(("committed", "stream", "prover"), bound)))
        assert CC.recorded_reason(r, *bound) == why


def test_a_certified_window_registers_once_per_stream_and_prover():
    """`Registrations` admits PoUW window p once per stream and prover: a second registration of p under them, which has
    their salt and so their seeds, is refused (`Reregistered`), naming the first; p under another stream or prover, and
    another window under them, are admitted. A window that isn't a u64, or a stream or prover that isn't 64 bytes, is
    refused."""
    other = hashlib.sha512(b"another").digest()
    ledger = PS.Registrations()
    ledger.admit(1, STREAM, PROVER, "aa" * 64)
    with pytest.raises(PS.Reregistered, match=r"PoUW window 1 is already registered \(aaaaaaaaaaaaaaaa\)"):
        ledger.admit(1, STREAM, PROVER, "bb" * 64)
    for window, stream, prover in ((1, other, PROVER), (1, STREAM, other), (2, STREAM, PROVER)):
        ledger.admit(window, stream, prover, "cc" * 64)
    with pytest.raises(PS.Reregistered):
        ledger.admit(2, STREAM, PROVER, "dd" * 64)
    for window, stream, prover in ((-1, STREAM, PROVER), (1, STREAM[:32], PROVER), (1, STREAM, None)):
        with pytest.raises(DomainError):
            ledger.admit(window, stream, prover, "ee" * 64)
    assert len(ledger) == 4 and (1, other, PROVER) in ledger and (3, STREAM, PROVER) not in ledger


def test_a_certified_window_credits_only_units_that_read_its_salt():
    """`unit_reason` refuses the served-zk window's drawn unit, `Pc8TileSide`, which reads two committed side rows and
    no salt (it declares no port), and passes the compiled Program's H1 tile (`PouwH1Tile`), whose ``salt`` is a coin
    port over the 64-byte salt; anything without a signature is refused."""
    from verity.core.protocols.pouw.circuit import compile as C
    from verity.core.protocols.pouw.circuit import h1tile as H
    from verity.core.protocols.pouw.circuit import pc8
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.ops.boolean import gemm as BG

    side = pc8.side_tile(128)
    why = PS.unit_reason(side)
    assert why is not None and side.id in why and "reads none" in why
    grid = C.Tile(4, 8, tuple((i, j) for i in range(4) for j in range(8)))
    h1 = bind(H.PouwH1Tile, EXACT=bind(BG.GemmAccumulatorS8, K=64, **BG.GEMM_SM120_S8), R=4, C=8, PAIRS=grid.pairs)
    assert PS.unit_reason(h1) is None
    assert PS.unit_reason(object()) is not None
