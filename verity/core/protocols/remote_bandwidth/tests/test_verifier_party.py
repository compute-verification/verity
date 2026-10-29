"""The bandwidth session split across a connection (`verifier_party`): the verifier's side and the prover's, each in its
own thread over a socket pair, make exactly the case and answer `period.run` makes in one process, and the party follows
`period.json`'s challenger host."""

from __future__ import annotations

import copy
import functools
import hashlib
import socket
import sys
import threading

import pytest

from verity.core.protocols.remote_bandwidth import audit, codec, verifier_party
from verity.core.protocols.remote_bandwidth import period as P
from remote_bandwidth_fakes import Clock, Entropy

STORE = hashlib.shake_256(b"a tiny MLP's weights").digest(3 * 4032 - 17)
COINS = hashlib.sha512(b"the period's live coins").digest()


class _Dropped:
    def __init__(self, blocks) -> None:
        self.blocks = blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, j: int) -> bytes:
        return bytes(codec.BLOCK_BYTES) if j % 2 else self.blocks[j]


def dropping(honest: audit.Store) -> audit.Store:
    out = copy.copy(honest)
    out.blocks = _Dropped(honest.blocks)
    return out


def _split(cheat=None) -> tuple[P.Period, int]:
    request = verifier_party.request_of("n0")
    a, b = socket.socketpair()
    got: dict = {}

    def verifier() -> None:
        c = Clock(10 ** 9, 1_000)
        with a, a.makefile("rwb") as f:
            got["period"] = verifier_party.serve(f, request, coins=COINS, clock=c, sleep_until=c.sleep_until)

    t = threading.Thread(target=verifier)
    t.start()
    with b, b.makefile("rwb") as f:
        passes = verifier_party.prove(f, P.Local("n0", STORE, cheat=cheat, entropy=Entropy(b"salts")))
    t.join()
    return got["period"], passes


def _in_process(cheat=None) -> P.Period:
    c = Clock(10 ** 9, 1_000)
    return P.run("n0", STORE, COINS, clock=c, sleep_until=c.sleep_until, entropy=Entropy(b"salts"), cheat=cheat)


@pytest.mark.parametrize("cheat, verdict", [(None, "accept"), (dropping, "reject")], ids=["honest", "dropping"])
def test_the_split_session_is_the_periods(monkeypatch, cheat, verdict) -> None:
    """Registration draws a fresh nonce from the OS; pinned, the two sessions are the same session."""
    monkeypatch.setattr(audit, "register", functools.partial(audit.register, nonce=bytes(audit.NONCE_BYTES)))
    split, passes = _split(cheat)
    assert split == _in_process(cheat)
    assert split.verdict == verdict and passes == 2


def test_the_registration_round_trips() -> None:
    reg = audit.register(COINS, b"n0", 3, bytes(64))
    assert P.registration_of(P.registration_json(reg)) == reg


def test_the_party_follows_period_json() -> None:
    local = verifier_party.Party.of({}, "n0")
    assert local.host == "n0" and local.address is None
    assert local.command == (sys.executable, "-m", verifier_party.MODULE, "--listen", "127.0.0.1:0")
    assert verifier_party.Party.of({"challenger_host": "n0"}, "n0") == local
    with pytest.raises(RuntimeError, match="auditor-host isn't wired for bandwidth"):
        verifier_party.Party.of({"challenger_host": "auditor-host", "challenger": {"pous": {"command": ["x"]}}}, "n0")
    wired = verifier_party.Party.of({"challenger_host": "auditor-host", "challenger": {"bandwidth": {
        "command": ["ssh", "auditor-host", "verifier"], "address": "192.0.2.7:7961"}}}, "n0")
    assert wired == verifier_party.Party("auditor-host", ("ssh", "auditor-host", "verifier"), ("192.0.2.7", 7961))
    assert verifier_party.placement("n0", "n0")["trusted-host"].startswith("unmet: ")
    held = verifier_party.placement("n0", "auditor-host")
    assert held["parties"] == {"challenger": "auditor-host", "prover": "n0", "verifier": "auditor-host"}
    assert held["trusted-host"].startswith("holds: ")
