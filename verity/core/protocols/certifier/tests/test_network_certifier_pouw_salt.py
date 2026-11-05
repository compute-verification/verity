"""The certifier's side of PoUW's salt (`active.Proxy` with PoUW's reader, `pouw.salt.registered`, driven tick by tick
on an injected clock), read through PoUW's `salt.revealed` and `salt.committed`: it reveals PoUW window p's salt on p's
registration once p is published, once, and refuses one unregistered, early, late, revealed already or missing; it
records a window's weights, stream and prover only signed by the registered prover before it draws that window's salt,
and gives them back on `reveal`'s conditions; and `core_check.pass_epoch` passes a certified window's pass end to end
against it. PoUW's own reading of a reveal and a record (the salt's form, `Epoch.certified`, `pass_epoch`'s refusals) is
PoUW's suite's, against a stand-in."""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import random

import pytest

from verity.ml.crypto import lms
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.active import Proxy
from verity.core.protocols.certifier.certificate import LmsKey
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw.audit import Epoch, Workload, commit_rows, weight_binding, weights_root
from verity.core.protocols.pouw.protocol import Matmul, Shape
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
    """PoUW's registration of window p as the proof system returns it: a record whose served window's context names p,
    with its receipt and digest."""
    w = {"kind": "served", "context": PS.context("served-zk/test", p, MANIFEST) if ctx is None else ctx, "public": {}}
    record = {"window": w if window is None else window}
    return Registration(record, R.receipt(record, "test", received=str(p)), R.digest(record), p)


def certifier(windows: int, records: dict[int, bytes] | None = None, clock=lambda: 0.0,
              **kwargs) -> tuple[Proxy, dict[int, bytes]]:
    """The certifier with PoUW's reader and the prover's key, the stream its context, on clock `clock`, handed `records`
    by PoUW window before its first tick (`commit_weights`, signed by the prover), after `windows` certifier windows of
    ingress ticks, and the salts it sampled by PoUW window: what the prover's server receives in-band."""
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


def setup():
    rng = random.Random(3)
    work = Workload((Matmul(0, Shape(32, 64, 64), 7),))
    rows = [ncp.int7_bytes([rng.randint(-64, 63) for _ in range(64)]) for _ in range(64)]
    return work, {7: commit_rows(weight_binding(7), -2, rows).commitment.root}


def manifest(work: Workload, roots: dict[int, bytes], p: int, salt: bytes, run: str = "r") -> tuple[bytes, str]:
    """A certified pass's manifest over `work` and its weights' `roots`, salted `salt`, and its window's context, which
    names PoUW window p and the manifest."""
    man = {"schema": CC.MANIFEST_CERTIFIED, "salt": salt.hex(), "run": run,
           "matmuls": [{"index": u.index, "m": u.shape.m, "k": u.shape.k, "n": u.shape.n, "weight": u.weight}
                       for u in work.matmuls],
           "weights": {str(w): {"root": r.hex()} for w, r in roots.items()}}
    raw = json.dumps(man).encode()
    return raw, PS.context(f"served-zk/{run}", p, hashlib.sha512(raw).hexdigest())


def test_the_certifier_reveals_a_pouw_window_on_its_registration():
    """With PoUW windows of two certifier windows and PoUW's reader, the certifier reveals PoUW window 0's salt once
    windows 0 and 1 are published, on PoUW's registration of window 0, and `revealed` gives it whole; PoUW window 1's
    after windows 2 and 3."""
    c, drawn = certifier(4, pouw_window=2)
    assert sorted(drawn) == [0, 1] and {len(s) for s in drawn.values()} == {PS.SALT_BYTES}
    for p in (0, 1):
        assert PS.registered(registration(p)) == p
        assert PS.revealed(c, p, registration(p)) == drawn[p]


def test_the_certifiers_refusals_fail_revealed_closed():
    """Through the certifier, `revealed` refuses (`Unrevealed`) a receipt or digest that isn't the record's, and PoUW's
    registration to a certifier without PoUW's reader. It refuses one that is early: before its last certifier window is
    published. One that is late: its reveal deadline, one PoUW window after its end on the certifier's clock, has passed
    (the same reveal at the deadline is given). One revealed already: a second reveal on the same registration or another
    naming the window, so behind one certifier one registration of a window gets its salt. And one whose salt is missing:
    never sampled (a certifier back from maintenance after its first window), the salt off, or a salt of another width.
    The certifier's refusals of time and the once are matched by PoUW's words only, so its own wording isn't pinned
    here."""
    c, _ = certifier(2, pouw_window=2)
    reg = registration(0)
    other = {"window": {"kind": "served", "context": PS.context("served-zk/test", 1, MANIFEST), "public": {}}}
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


def test_a_certified_window_passes_a_pass_salted_by_its_certifiers_reveal():
    """End to end against the certifier, PoUW windows of two certifier windows: the pass the prover salted by
    `Epoch.certified` of windows 2 and 3 from the salt it received in-band passes `core_check.pass_epoch` with the reveal
    `revealed` and the record `committed` read from the certifier on that registration (``certifier``)."""
    work, roots = setup()
    c, drawn = certifier(4, {1: PS.pouw_record(weights_root(roots), STREAM, PROVER)}, pouw_window=2)
    raw, ctx = manifest(work, roots, 1, Epoch.certified(work, roots, [2, 3], 2, drawn[1], STREAM, PROVER).salt)
    reveal = PS.revealed(c, 1, registration(1, ctx=ctx))
    assert reveal == drawn[1]
    record = PS.committed(c, 1, registration(1, ctx=ctx))
    assert CC.pass_epoch(None, raw, ctx, reveal=reveal, committed=record, stream=STREAM, prover=PROVER) == \
        ("certifier", None)


def test_a_certifier_reveals_a_window_once_and_draws_its_own_salts():
    """The certifier reveals PoUW window 1 once, so a second registration of it, another stream's, gets no salt at all.
    Each certifier draws its own salts, so two certifiers' salts for window 1 (two proxies' lifetimes, each counting from
    its origin) differ too, with no epoch bound."""
    c, _ = certifier(4, pouw_window=2)
    PS.revealed(c, 1, registration(1))
    with pytest.raises(PS.Unrevealed, match="the certifier doesn't reveal PoUW window 1's salt"):
        PS.revealed(c, 1, registration(1, ctx=PS.context("served-zk/another", 1, MANIFEST)))
    lives = [certifier(4, pouw_window=2, salt=active.fresh_salt)[0] for _ in range(2)]
    assert PS.pouw_salt(PS.revealed(lives[0], 1, registration(1)), 1, STREAM, PROVER) != \
        PS.pouw_salt(PS.revealed(lives[1], 1, registration(1)), 1, STREAM, PROVER)


def test_the_certifier_records_only_signed_weights_before_its_salt():
    """The certifier records a window's weights, stream and prover (`pouw_record`, through `Proxy.commit_weights`) only
    signed by the prover's registered key over `weights_message` under the stream, before it samples the window's salt,
    and `committed` reads it back on `reveal`'s conditions. PoUW windows of two certifier windows: window 0's record is
    refused unsigned, signed by another key, signed for another window or under another stream's context; signed, made
    before its sample, it is read back and its pass passes, and a second record for it is refused. Window 1's, made after
    its sample, is refused, so `committed` refuses window 1 and its pass, salted by its own salt, is refused. The
    certifier's refusals are matched by type and PoUW's by its own words, so neither's wording is pinned to the
    other's."""
    work, roots = setup()
    record = PS.pouw_record(weights_root(roots), STREAM, PROVER)
    c, drawn = certifier(0, pouw_window=2)
    for bad in (b"", None, signed(c, 0, record, sign=FORGE), signed(c, 1, record),
                signed(c, 0, record, context=hashlib.sha512(b"another stream").digest())):
        with pytest.raises(RuntimeError):
            c.commit_weights(0, record, bad)
    c.commit_weights(0, record, signed(c, 0, record))
    other = PS.pouw_record(weights_root(roots), STREAM, PS.prover_id(prover_key(b"another")[0].public))
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
