"""Concurrent sessions on one memory-challenger key (`memory_challenger.LmsSigner`, `period.challenger`), as the
all-protocols runner's windows share their device's: they take turns at the key's leaf state, each signing at a leaf of
its own; a crash between a leaf's reservation and its signature never reuses the leaf; and windows that start at once on
a fresh device make its key and session log once.  No clock: a thread's wait on a lock is seen through `fcntl.flock`."""
from __future__ import annotations

import fcntl
import hashlib
import json
import queue
import subprocess
import sys
import threading

import pous_test_signer as ts
from test_pous_memory_challenger import SESSION, judge, session
from verity.ml.crypto import lms
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import period


def _waiting(monkeypatch) -> queue.SimpleQueue:
    """`fcntl.flock` patched to put "waiting" on the queue it returns when a blocking exclusive lock has to wait, so a
    test knows a thread waits on a lock, with no clock."""
    real, said = fcntl.flock, queue.SimpleQueue()

    def flock(f, op):
        if op == fcntl.LOCK_EX:
            try:
                return real(f, op | fcntl.LOCK_NB)
            except BlockingIOError:
                said.put("waiting")
        return real(f, op)

    monkeypatch.setattr(fcntl, "flock", flock)
    return said


def _at_once(said: queue.SimpleQueue, release, calls) -> tuple[list, list]:
    """Each of `calls` in a thread of its own while the test holds a lock, which `release` lets go once every thread has
    said something: "waiting" on the lock, or "done" (it refused). -> what each said first, and what each returned or
    raised."""
    out: list = [None] * len(calls)

    def run(i: int) -> None:
        try:
            out[i] = calls[i]()
        except Exception as e:  # noqa: BLE001: the test reads it
            out[i] = e
        said.put("done")

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(calls))]
    for t in threads:
        t.start()
    try:
        first = [said.get() for _ in threads]
    finally:
        release()
        for t in threads:
            t.join()
    return first, out


def test_concurrent_sessions_on_one_leaf_state_each_sign_at_a_leaf_of_their_own(monkeypatch):
    """Three sessions at once, each with a signer of its own over one key and one leaf state (as three windows' subprotocols
    share their device's), wait on the state's lock while it is held, then all sign once it is let go, each at a leaf of
    its own: none is refused and no leaf is used twice."""
    one = ts.TestSigner(b"shared key", height=3)
    signers = [one, *(mc.LmsSigner(one.key, one.state, scheme=one.scheme) for _ in range(2))]
    holder = lms.LeafState(one.state, one.public_key, 3)
    said = _waiting(monkeypatch)
    first, out = _at_once(said, holder.close, [lambda s=s: session(signer=s) for s in signers])
    assert first == ["waiting"] * 3, out
    leaves = []
    for c, signed, signer in out:
        assert judge(c, signed, signer).accepted
        leaves.append(lms.Signature.decode(lms.Params(64, 1, 3), signed.signature).q)
    assert sorted(leaves) == [0, 1, 2]
    assert one.state.with_name(one.state.name + ".journal").read_text().split() == ["0", "1", "2"]


CRASH = """
import hashlib, os, sys
from verity.ml.crypto import lms
from verity.core.protocols.pous import memory_challenger as mc
signer = mc.LmsSigner(lms.keygen(lms.Params(64, 1, 3), hashlib.sha512(b"crash key").digest()), sys.argv[1], sys.argv[2])
lms.sign_at = lambda *a: os._exit(17)
signer.sign(bytes(32), b"record")
"""
"""A signer's process that dies after its leaf is reserved and before the signature is made."""


def test_a_crash_between_a_leafs_reservation_and_its_signature_never_reuses_the_leaf(tmp_path):
    """The process dies once `sign` has reserved leaf 0 and before it signs: the state is past leaf 0 on disk, so the
    next signer of the key signs at leaf 1, and leaf 0 is never used."""
    key = lms.keygen(lms.Params(64, 1, 3), hashlib.sha512(b"crash key").digest())
    state = tmp_path / "leaf-state.json"
    lms.LeafState.create(state, lms.public_key(key), 3)
    assert subprocess.run([sys.executable, "-c", CRASH, str(state), ts.scheme(3)]).returncode == 17
    assert state.with_name(state.name + ".journal").read_text().split() == ["0"]
    signed = mc.LmsSigner(key, state, scheme=ts.scheme(3)).sign(SESSION, b"record")
    assert lms.Signature.decode(key.params, signed).q == 1 and json.loads(state.read_text())["next"] == 2


def test_windows_that_start_at_once_make_the_challengers_key_once(tmp_path, monkeypatch):
    """`period.challenger` on one fresh device directory from three windows at once (the all-protocols runner's
    `--jobs 3`): the first makes the key and the session log under the directory's lock while the others wait on it, so
    there is one keygen and one key on disk, and each window's signer signs at a leaf of its own. The deployed parameters
    at a small height."""
    small = lms.Params(64, 1, 3)
    monkeypatch.setattr(lms, "DEPLOYED", small)
    made, keygen = [], lms.keygen
    monkeypatch.setattr(lms, "keygen", lambda *a, **k: made.append(a) or keygen(*a, **k))
    device = tmp_path / "pous-challenger"
    device.mkdir()
    lock = open(device / "device.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    said = _waiting(monkeypatch)
    first, signers = _at_once(said, lock.close, [lambda: period.challenger(device, workers=1)] * 3)
    assert first == ["waiting"] * 3, signers
    assert len(made) == 1 and len({s.public_key for s in signers}) == 1
    assert period.challenger(device, workers=1).public_key == signers[0].public_key and len(made) == 1
    mc.SessionLog(device / "sessions").claim(SESSION)
    assert [lms.Signature.decode(small, s.sign(SESSION, b"record")).q for s in signers] == [0, 1, 2]
