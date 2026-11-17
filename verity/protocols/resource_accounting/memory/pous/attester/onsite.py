"""The memory attester's key as a deployment starts it on the auditor's device, for PoUS's rounds
(`challenger_party.start`) and for the wipe (`wipe_challenger.start`): loaded from the party's state directory
(`device_key`), and refused unless the configuration pins it for the protocol it signs for. A start makes no key: one
made there could never be the pinned one, so a state directory without a key is refused (``key``). The pin is checked
against the public key the leaf state holds before the key's tree is read, so a wrong key is refused before anything
else the start reads. A configuration's `[[pin]]` names a party, a protocol and the SHA-512 of its public key
(`key-sha512`), never a host.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity.channel import Refused
from verity.protocols.resource_accounting.memory.pous.attester import memory_attester as mc

#: the leaf state `device_key` keeps beside the key, which holds its public key
KEY_STATE = "leaf-state.json"

def device_key(directory: str | os.PathLike, *, workers: int | None = None) -> mc.LmsSigner:
    """The memory attester's key at S1's deployed parameters in `directory`, made there once (its seed from the OS, its
    tree computed in `workers` processes, all of them by default) with its leaf state beside it, so each session signs at
    a fresh leaf, and its session log (`sessions`). Windows that run at once share them: the first makes them under the
    directory's lock (`device.lock`) while the others wait on it, and each window's signer takes the leaf state only to
    sign (`memory_attester.LmsSigner`)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    seed_file, tree_file, state = d / "seed", d / "tree.bin", d / KEY_STATE
    with open(d / "device.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not seed_file.exists():
            seed = secrets.token_bytes(mc.SEED_BYTES)
            key = mc.deployed_key(seed, workers=(os.cpu_count() or 1) if workers is None else workers)
            tree_file.write_bytes(mc.key_tree(key))
            fd = os.open(seed_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(seed)
            mc.create_leaf_state(state, key)
        else:
            key = mc.deployed_key(seed_file.read_bytes(), tree_file.read_bytes())
        if not (d / "sessions").exists():
            mc.SessionLog.create(d / "sessions")
    return mc.LmsSigner(key, state)


def pin(configuration: Mapping[str, Any], party: str, protocol: str) -> str:
    """The SHA-512 of the key `configuration` pins for `party` in `protocol`, in hex; `Refused` (``pin``) when it pins
    none."""
    for p in configuration.get("pin", ()):
        if (p.get("party"), p.get("protocol")) == (party, protocol):
            return p["key-sha512"]
    raise Refused("pin", f"the configuration pins no key for {party} in {protocol}: a memory attester signs only "
                         "under a key the auditor pinned ([[pin]])")


def _pinned(state: Path, public: bytes, key_sha512: str) -> None:
    if (got := hashlib.sha512(public).hexdigest()) != key_sha512:
        raise Refused("pin", f"the key in {state} is not the one the configuration pins: its SHA-512 is {got}")


def signer(state: Path | None, key_sha512: str) -> mc.LmsSigner:
    """The challenger's signer, its key loaded from `state` (`device_key`) once the public key its leaf state holds is
    the pinned one; `Refused` (``state``) with no state directory, (``key``) when it holds no key, and (``pin``) when
    the SHA-512 of its public key is not `key_sha512`."""
    if state is None:
        raise Refused("state", "the memory attester keeps its key and logs in its state directory, and has none")
    try:
        public = bytes.fromhex(json.loads((state / KEY_STATE).read_text())["public"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Refused("key", f"{state} holds no memory attester key ({type(e).__name__}): it is made once on the "
                             "auditor's device (`device_key`) and its SHA-512 pinned ([[pin]])") from None
    if not (state / "seed").is_file():
        raise Refused("key", f"{state} holds a leaf state but no key seed")
    _pinned(state, public, key_sha512)
    s = device_key(state)
    _pinned(state, s.public_key, key_sha512)
    return s


__all__ = ["KEY_STATE", "device_key", "pin", "signer"]
