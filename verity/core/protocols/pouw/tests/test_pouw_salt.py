"""PoUW's read of the network certifier's salt (`verity.core.protocols.pouw.salt`) against the certifier itself
(`certifier.active.Proxy`, driven tick by tick): the salt's form is pinned; a context names its PoUW window; `revealed`
gives PoUW window p's salt on p's registration once p is published, and fails closed when it is unregistered, early or
missing; `pouw_window` takes one whole PoUW window aligned at 0 and nothing else; `Epoch.certified` applies
`pouw_salt` of the reveal, 64 bytes, which the seeds and the identifier take, with the challenger's draw coin; and
`core_check.pass_epoch` passes a certified window's pass only when its salt is that of the certifier's reveal."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random

import pytest

import epoch_coins as E
from verity.core.protocols.certifier import active
from verity.core.protocols.certifier.active import Proxy
from verity.core.protocols.certifier.grid import GridParams
from verity.core.protocols.certifier.schedule import ClockSync, SyncSet
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import identifier as I
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw.audit import Epoch, Workload, commit_rows, weight_binding
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


def registration(p: int, ctx: str | None = None, window: dict | None = None) -> Registration:
    """PoUW's registration of window p as the proof service returns it: a record whose served window's context names p,
    with its receipt and digest."""
    w = {"kind": "served", "context": PS.context("served-zk/test", p, MANIFEST) if ctx is None else ctx, "public": {}}
    record = {"window": w if window is None else window}
    return Registration(record, R.receipt(record, "test", received=str(p)), R.digest(record), p)


def certifier(windows: int, **kwargs) -> tuple[Proxy, dict[int, bytes]]:
    """The certifier with PoUW's reader, after `windows` certifier windows of ingress ticks, and the salts it drew by
    PoUW window."""
    drawn: dict[int, bytes] = {}

    def source(p: int, n: int) -> bytes:
        drawn[p] = hashlib.shake_256(b"certifier salt %d" % p).digest(n)
        return drawn[p]

    c = Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0,
              **{"salt": source, "registered": PS.registered, **kwargs})
    for _ in range(windows * INGRESS.buckets):
        c.bucketer.tick()
    return c, drawn


def test_pouw_salt_form():
    """The salt PoUW applies in window p is SHA-512(reveal ‖ u64be(p)), pinned on one vector; a reveal that isn't 64
    bytes and a window that isn't a u64 are refused."""
    reveal = bytes(range(64))
    assert PS.pouw_salt(reveal, 3).hex() == (
        "b761cc000dcb04d328fb60eca2836461f12782a72d1406d8073c9a4c6d06c629"
        "8fba66fb6e8c0b6dc8f06fe991409e9d79db525b094ee402a2d8d236c1faf43e")
    assert PS.pouw_salt(reveal, 3) == hashlib.sha512(reveal + (3).to_bytes(8, "big")).digest()
    assert PS.pouw_salt(reveal, 4) != PS.pouw_salt(reveal, 3)
    for bad in (reveal[:32], reveal + b"\0", bytearray(reveal), reveal.hex()):
        with pytest.raises(DomainError, match="64 bytes"):
            PS.pouw_salt(bad, 3)
    for bad in (-1, 1 << 64, True, 3.0):
        with pytest.raises(DomainError, match="u64"):
            PS.pouw_salt(reveal, bad)


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
    reader. It refuses one that is early: before its last certifier window is published. And one whose salt is
    missing: never drawn (a certifier back from maintenance after its first window), the salt off, or a salt of
    another width."""
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
    """`Epoch.certified` of PoUW window 1's certifier windows (2 and 3, k = 2) has salt `pouw_salt(reveal, 1)`, 64
    bytes: the seeds take it and Epoch.start's 32 bytes they refuse, the identifier names it whole, and the challenger's
    draw coin opens against its coins. Misaligned windows, a reveal of another width and weights that aren't the
    workload's are refused."""
    work, roots = setup()
    c, drawn = certifier(4, pouw_window=2)
    reveal = PS.revealed(c, 1, registration(1))
    ch = E.challenger()
    epoch = Epoch.certified(work, roots, [2, 3], 2, reveal, coins=ch.commitment)
    assert epoch.salt == PS.pouw_salt(drawn[1], 1) and len(epoch.salt) == 64 and epoch.coins == ch.commitment
    assert epoch.salt != Epoch.certified(work, roots, [0, 1], 2, reveal).salt

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
            Epoch.certified(work, roots, ws, 2, reveal)
    with pytest.raises(DomainError, match="64 bytes"):
        Epoch.certified(work, roots, [2, 3], 2, reveal[:32])
    with pytest.raises(DomainError, match="every weight"):
        Epoch.certified(work, {**roots, 8: root}, [2, 3], 2, reveal)


def test_a_certified_window_passes_only_a_pass_salted_by_its_certifiers_reveal():
    """`core_check.pass_epoch` in a certified window, its context naming PoUW window 1 and the pass, with the reveal
    `revealed` read from the certifier on that registration: the pass salted by `Epoch.certified` of windows 2 and 3
    passes (``certifier``). Salted by window 0's epoch, by the raw reveal, or checked against window 0's reveal, it
    doesn't, nor does it pass at a verifier without the reveal."""
    work, roots = setup()
    c, _ = certifier(4, pouw_window=2)

    def manifest(salt: bytes) -> tuple[bytes, str]:
        man = {"schema": CC.MANIFEST_CERTIFIED, "salt": salt.hex(), "run": "r",
               "matmuls": [{"index": u.index, "m": u.shape.m, "k": u.shape.k, "n": u.shape.n, "weight": u.weight}
                           for u in work.matmuls],
               "weights": {str(w): {"root": r.hex()} for w, r in roots.items()}}
        raw = json.dumps(man).encode()
        return raw, PS.context("served-zk/r", 1, hashlib.sha512(raw).hexdigest())

    reveal = PS.revealed(c, 1, registration(1))
    raw, ctx = manifest(Epoch.certified(work, roots, [2, 3], 2, reveal).salt)
    assert PS.revealed(c, 1, registration(1, ctx=ctx)) == reveal
    assert CC.pass_epoch(None, raw, ctx, reveal=reveal) == ("certifier", None)

    mismatch = ("coin", "the pass's salt is not the certifier's for its window")
    for salt in (Epoch.certified(work, roots, [0, 1], 2, reveal).salt, reveal):
        assert CC.pass_epoch(None, *manifest(salt), reveal=reveal) == (None, mismatch)
    assert CC.pass_epoch(None, raw, ctx, reveal=PS.revealed(c, 0, registration(0))) == (None, mismatch)
    how, refused = CC.pass_epoch(None, raw, ctx)
    assert how is None and refused[0] == "coin" and refused[1].startswith(f"the pass's manifest is {CC.MANIFEST_CERTIFIED}")
