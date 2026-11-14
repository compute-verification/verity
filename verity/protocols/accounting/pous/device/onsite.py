"""The memory challenger's key as a deployment starts it on the auditor's host, for PoUS's rounds
(`challenger_party.start`) and for the wipe (`wipe_challenger.start`): made or loaded in the party's state directory
(`period.challenger`), and refused unless the configuration pins it for the protocol it signs for. A configuration's
`[[pin]]` names a party, a protocol and the SHA-512 of its public key (`key-sha512`), never a host.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity.channel import Refused
from verity.protocols.accounting.pous.device import memory_challenger as mc
from verity.protocols.accounting.pous.verifier import period


def pin(configuration: Mapping[str, Any], party: str, protocol: str) -> str:
    """The SHA-512 of the key `configuration` pins for `party` in `protocol`, in hex; `Refused` (``pin``) when it pins
    none."""
    for p in configuration.get("pin", ()):
        if (p.get("party"), p.get("protocol")) == (party, protocol):
            return p["key-sha512"]
    raise Refused("pin", f"the configuration pins no key for {party} in {protocol}: a memory challenger signs only "
                         "under a key the auditor pinned ([[pin]])")


def signer(state: Path | None, key_sha512: str) -> mc.LmsSigner:
    """The challenger's signer, its key made or loaded in `state` (`period.challenger`); `Refused` (``state``) with no
    state directory, and (``pin``) when the SHA-512 of its public key is not `key_sha512`."""
    if state is None:
        raise Refused("state", "the memory challenger keeps its key and logs in its state directory, and has none")
    s = period.challenger(state)
    if hashlib.sha512(s.public_key).hexdigest() != key_sha512:
        raise Refused("pin", f"the key in {state} is not the one the configuration pins: its SHA-512 is "
                             f"{hashlib.sha512(s.public_key).hexdigest()}")
    return s


__all__ = ["pin", "signer"]
