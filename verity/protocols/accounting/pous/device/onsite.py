"""The memory challenger's key as a deployment starts it on the auditor's host, for PoUS's rounds
(`challenger_party.start`) and for the wipe (`wipe_challenger.start`): loaded from the party's state directory
(`period.challenger`), and refused unless the configuration pins it for the protocol it signs for. A start makes no key:
one made there could never be the pinned one, so a state directory without a key is refused (``key``). The pin is
checked against the public key the leaf state holds before the key's tree is read, so a wrong key is refused before
anything else the start reads. A configuration's `[[pin]]` names a party, a protocol and the SHA-512 of its public key
(`key-sha512`), never a host.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity.channel import Refused
from verity.protocols.accounting.pous.device import memory_challenger as mc
from verity.protocols.accounting.pous.verifier import period

#: the leaf state `period.challenger` keeps beside the key, which holds its public key
KEY_STATE = "leaf-state.json"


def pin(configuration: Mapping[str, Any], party: str, protocol: str) -> str:
    """The SHA-512 of the key `configuration` pins for `party` in `protocol`, in hex; `Refused` (``pin``) when it pins
    none."""
    for p in configuration.get("pin", ()):
        if (p.get("party"), p.get("protocol")) == (party, protocol):
            return p["key-sha512"]
    raise Refused("pin", f"the configuration pins no key for {party} in {protocol}: a memory challenger signs only "
                         "under a key the auditor pinned ([[pin]])")


def _pinned(state: Path, public: bytes, key_sha512: str) -> None:
    if (got := hashlib.sha512(public).hexdigest()) != key_sha512:
        raise Refused("pin", f"the key in {state} is not the one the configuration pins: its SHA-512 is {got}")


def signer(state: Path | None, key_sha512: str) -> mc.LmsSigner:
    """The challenger's signer, its key loaded from `state` (`period.challenger`) once the public key its leaf state
    holds is the pinned one; `Refused` (``state``) with no state directory, (``key``) when it holds no key, and
    (``pin``) when the SHA-512 of its public key is not `key_sha512`."""
    if state is None:
        raise Refused("state", "the memory challenger keeps its key and logs in its state directory, and has none")
    try:
        public = bytes.fromhex(json.loads((state / KEY_STATE).read_text())["public"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Refused("key", f"{state} holds no memory challenger key ({type(e).__name__}): it is made once on the "
                             "auditor's host (`period.challenger`) and its SHA-512 pinned ([[pin]])") from None
    if not (state / "seed").is_file():
        raise Refused("key", f"{state} holds a leaf state but no key seed")
    _pinned(state, public, key_sha512)
    s = period.challenger(state)
    _pinned(state, s.public_key, key_sha512)
    return s


__all__ = ["KEY_STATE", "pin", "signer"]
