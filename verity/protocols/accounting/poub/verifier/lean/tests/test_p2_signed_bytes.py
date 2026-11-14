"""What the memory challenger signs in a P2 session (`p2_case.close_of`) and the session's answer, held to the record of
them (`p2_signed_sessions.json`): five sessions over `test_p2_party.py`'s store and channel, on the coins `b"\\x07" * 64`
and the test clock, each closed by the test signer at leaf 0 (`pous_test_signer.AtLeaf`, which keeps no state): an
honest prover accepted, one whose receipts skip a unit's reads, one whose openings are out of form, one gone mid-window
(the verifier reads the stream's end where a root was due) and one the verifier can't write to (its first `open`
fails). Each session's close (its chain, its lines and its signature's SHA-512), its case's SHA-512, its answer and how
the prover's side ended (with the answer, or what it raised) are the record's. `python test_p2_signed_bytes.py --write` rewrites the record from the party as it is."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import threading
from pathlib import Path

import pytest

from test_p2_party import TOY, MemoryLog, Prover, SESSION_ID, Clock, exe, request, signer, store  # noqa: F401
from verity.protocols.accounting.poub.prover import p2_party_prover
from verity.protocols.accounting.poub.verifier import p2_lean, p2_parties

RECORD = Path(__file__).with_name("p2_signed_sessions.json")
KINDS = ("accept", "skip", "garble", "gone", "unwritable")


class Gone(Exception):
    """The prover leaving mid-window."""


class Leaving(Prover):
    """The honest prover, gone at window 1's second pass: its side of the channel closes where that root was due."""

    def run(self, challenge, t, i, previous) -> bytes:
        if (t, i) == (1, 1):
            raise Gone
        return super().run(challenge, t, i, previous)


class Unwritable:
    """The verifier's writer, broken from its first `open` on: that write and every later one fail."""

    def __init__(self, w) -> None:
        self.w, self.broken = w, False

    def write(self, line: bytes) -> int:
        if self.broken or line.startswith(b'{"open"'):
            self.broken = True
            raise BrokenPipeError(32, "Broken pipe")
        return self.w.write(line)

    def flush(self) -> None:
        self.w.flush()


def session(exe: Path, store, kind: str) -> dict:
    """One session of `kind` over a socket pair, the verifier's side in a thread: its record."""
    clock, kept, got = Clock(), [], {}
    a, b = socket.socketpair()
    data = json.dumps(p2_parties.registration_json(store.registration)).encode()

    def verifier() -> None:
        with p2_lean.Lean(exe) as lean, b, b.makefile("rb") as r, b.makefile("wb") as w:
            reg, held = p2_parties.hold(lean, data, auditor=TOY)
            got["answer"] = p2_parties.session(
                r, Unwritable(w) if kind == "unwritable" else w, reg, coins=b"\x07" * 64, lean=lean, plan=held,
                clock=clock, sleep_until=clock.sleep_until, host="auditor-under-test", keep=kept.append, auditor=TOY,
                closer=p2_parties.Closer(signer(), MemoryLog(), SESSION_ID))

    th = threading.Thread(target=verifier)
    th.start()
    prover = (Leaving if kind == "gone" else Prover)(store, clock, skip=1 if kind == "skip" else None,
                                                     garble=kind == "garble")
    with a, a.makefile("rb") as r, a.makefile("wb") as w:
        try:
            proved = p2_party_prover.prove(r, w, prover, request(store))
        except (Gone, ConnectionError) as e:
            proved = type(e).__name__
    th.join()
    case, = kept
    close = json.loads(bytes.fromhex(case["evidence"]))["close"]
    return {"answer": got["answer"], "prover": "answer" if proved == got["answer"] else proved,
            "case_sha512": hashlib.sha512(json.dumps(case, sort_keys=True).encode()).hexdigest(),
            "close": {"chain": close["chain"], "lines": close["lines"], "public_key": close["public_key"],
                      "signature_sha512": hashlib.sha512(bytes.fromhex(close["signature"])).hexdigest()}}


@pytest.mark.parametrize("kind", KINDS)
def test_the_signed_close_and_the_answer_are_the_records(exe, store, kind) -> None:
    assert session(exe, store, kind) == json.loads(RECORD.read_text())[kind]


def test_the_record_holds_each_kinds_verdict() -> None:
    """The record is of the sessions it names: accepted, rejected at a receipt, rejected malformed three ways."""
    record = json.loads(RECORD.read_text())
    assert sorted(record) == sorted(KINDS)
    assert [(record[k]["answer"]["verdict"], record[k]["answer"].get("failure")) for k in KINDS] == [
        ("accept", None), ("reject", "receipt"), ("reject", "malformed"), ("reject", "malformed"),
        ("reject", "malformed")]
    assert [record[k]["prover"] for k in KINDS] == ["answer", "answer", "answer", "Gone", "ConnectionError"]


if __name__ == "__main__" and sys.argv[1:] == ["--write"]:
    built = p2_lean.build()
    held = store.__wrapped__()
    RECORD.write_text(json.dumps({k: session(built, held, k) for k in KINDS}, indent=1, sort_keys=True) + "\n")
