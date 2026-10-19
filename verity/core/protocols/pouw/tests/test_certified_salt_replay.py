"""Witnesses on the certified salt, against the network certifier that records the weights (`certifier.active.Proxy`
with `commit_weights` and `committed`) and PoUW's reader of it (`pouw.salt`), driven tick by tick. They pass while
the weakness they show is there:

* one PoUW window's salt is every registration's that names the window: `Proxy.reveal` gives `reveal(p)` to any number
  of registrations of p, and `pouw_salt(reveal, p)` reads nothing but p, so two passes registered in p (two streams,
  two levels or two provers behind one link) are salted alike and every tile they share gets one seed. One-stage's
  window salt binds its stream and prover (#2640); the certified salt binds neither;
* the weights' order holds end to end on the certifier's own record: weights committed before p's first ingress tick
  are the only ones a certified pass of p passes over;
* `commit_weights` takes the first root anyone hands it, so a root handed first refuses the honest one."""

from __future__ import annotations

import hashlib
import json

import pytest

from verity.core.protocols.certifier.active import Proxy
from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw.audit import Epoch, weights_root
from verity.core.protocols.pouw.circuit import seeds as SD

from test_pouw_salt import EGRESS, INGRESS, SYNC_SET, SYNCS, registration, setup


def recording(salts: dict[int, bytes]) -> Proxy:
    def source(p: int, n: int) -> bytes:
        salts[p] = hashlib.shake_256(b"certifier salt %d" % p).digest(n)
        return salts[p]

    return Proxy(("127.0.0.1", 9), EGRESS, INGRESS, SYNC_SET, SYNCS, origin=0.0, salt=source, registered=PS.registered)


def certified_pass(work, roots, p: int, salt: bytes, run: str) -> tuple[bytes, str]:
    """Run `run`'s certified pass over `work`, salted `salt`, and its window's context, naming PoUW window p and the pass."""
    man = {"schema": CC.MANIFEST_CERTIFIED, "salt": salt.hex(), "run": run,
           "matmuls": [{"index": u.index, "m": u.shape.m, "k": u.shape.k, "n": u.shape.n, "weight": u.weight}
                       for u in work.matmuls],
           "weights": {str(w): {"root": r.hex()} for w, r in roots.items()}}
    raw = json.dumps(man).encode()
    return raw, PS.context(f"served-zk/{run}", p, hashlib.sha512(raw).hexdigest())


def test_every_registration_of_a_pouw_window_is_salted_alike():
    """Two runs, each its own pass and its own registration of PoUW window 0 (other contexts, other manifests, other
    digests), both pass `pass_epoch` on the certifier's reveal and recorded weights, under one salt, so each tile's seed
    and receipt seed, `SD.seed(salt, j, root A, root B)`, is the same in both: a tile computed once is two passes'
    work. The certifier counts no registrations: eight of p all get its salt."""
    work, roots = setup()
    salts: dict[int, bytes] = {}
    c = recording(salts)
    c.commit_weights(0, weights_root(roots))
    for _ in range(INGRESS.buckets):
        c.bucketer.tick()

    salt = Epoch.certified(work, roots, [0], 1, PS.revealed(c, 0, registration(0))).salt
    got = []
    for run in ("stream-a/prover-1", "stream-b/prover-2"):
        raw, ctx = certified_pass(work, roots, 0, salt, run)
        reg = registration(0, ctx=ctx)
        reveal, weights = PS.revealed(c, 0, reg), PS.committed(c, 0, reg)
        assert CC.pass_epoch(None, raw, ctx, reveal=reveal, weights=weights) == ("certifier", None)
        got.append((raw, reg.digest, PS.pouw_salt(reveal, 0)))
    (raw_a, reg_a, salt_a), (raw_b, reg_b, salt_b) = got
    assert raw_a != raw_b and reg_a != reg_b and salt_a == salt_b == salt

    ra, rb = hashlib.sha512(b"A rows").digest(), hashlib.sha512(b"B rows").digest()
    for j in (u.index for u in work.matmuls):
        assert SD.seed(salt_a, j, ra, rb) == SD.seed(salt_b, j, ra, rb)
        assert SD.receipt_seed(salt_a, j, ra, rb) == SD.receipt_seed(salt_b, j, ra, rb)

    regs = [registration(0, ctx=PS.context(f"served-zk/stream-{i}", 0, hashlib.sha512(b"%d" % i).hexdigest()))
            for i in range(8)]
    assert {PS.revealed(c, 0, r) for r in regs} == {salts[0]}


def test_the_weights_order_holds_end_to_end_on_the_certifiers_record():
    """On the certifier's own record: weights committed for PoUW window 0 before its first ingress tick are read back on
    the window's registration, and a certified pass over them passes; window 1's, handed after its salt's draw, are
    refused, so a pass of window 1 has no weights to pass over; a pass of window 0 over other weights, salted alike
    since the salt reads no weights, is refused."""
    work, roots = setup()
    root = weights_root(roots)
    c = recording({})
    c.commit_weights(0, root)
    for _ in range(INGRESS.buckets + 1):
        c.bucketer.tick()
    with pytest.raises(RuntimeError, match="drawn"):
        c.commit_weights(1, root)
    for _ in range(INGRESS.buckets - 1):
        c.bucketer.tick()

    reveal = PS.revealed(c, 0, registration(0))
    raw, ctx = certified_pass(work, roots, 0, Epoch.certified(work, roots, [0], 1, reveal).salt, "r")
    weights = PS.committed(c, 0, registration(0, ctx=ctx))
    assert weights == root and CC.pass_epoch(None, raw, ctx, reveal=reveal, weights=weights) == ("certifier", None)
    with pytest.raises(PS.Unrevealed, match="uncommitted"):
        PS.committed(c, 1, registration(1))

    later = {w: hashlib.sha512(r).digest() for w, r in roots.items()}
    raw, ctx = certified_pass(work, later, 0, Epoch.certified(work, later, [0], 1, reveal).salt, "r")
    assert CC.pass_epoch(None, raw, ctx, reveal=reveal, weights=weights) == \
        (None, ("coin", "the pass's weights are not those the certifier recorded before it drew the window's salt"))


def test_the_first_root_handed_is_the_windows():
    """`commit_weights` has no caller to check: a root anyone hands first for PoUW window 0 is the window's, the honest
    root is then refused, and the honest pass of window 0 is refused for its weights."""
    work, roots = setup()
    c = recording({})
    c.commit_weights(0, hashlib.sha512(b"someone else's root").digest())
    with pytest.raises(RuntimeError, match="already recorded"):
        c.commit_weights(0, weights_root(roots))
    for _ in range(INGRESS.buckets):
        c.bucketer.tick()
    reveal = PS.revealed(c, 0, registration(0))
    raw, ctx = certified_pass(work, roots, 0, Epoch.certified(work, roots, [0], 1, reveal).salt, "r")
    assert CC.pass_epoch(None, raw, ctx, reveal=reveal, weights=PS.committed(c, 0, registration(0, ctx=ctx))) == \
        (None, ("coin", "the pass's weights are not those the certifier recorded before it drew the window's salt"))
