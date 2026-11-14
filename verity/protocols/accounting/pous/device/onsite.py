"""The memory challenger's key as a deployment starts it on the auditor's host, for PoUS's rounds
(`challenger_party.start`) and for the wipe (`wipe_challenger.start`): made or loaded in the party's state directory
(`device_key`), and refused unless the configuration pins it for the protocol it signs for. A configuration's
`[[pin]]` names a party, a protocol and the SHA-512 of its public key (`key-sha512`), never a host.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity.channel import Refused
from verity.commitments.crypto import lms
from verity.protocols.accounting.pous.device import memory_challenger as mc


def device_key(directory: str | os.PathLike, *, workers: int | None = None) -> mc.LmsSigner:
    """The memory challenger's key at S1's deployed parameters in `directory`, made there once (its seed from the OS, its
    tree computed in `workers` processes, all of them by default) with its leaf state beside it, so each session signs at
    a fresh leaf, and its session log (`sessions`). Windows that run at once share them: the first makes them under the
    directory's lock (`device.lock`) while the others wait on it, and each window's signer takes the leaf state only to
    sign (`memory_challenger.LmsSigner`)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p, top = lms.DEPLOYED, 2 ** (lms.DEPLOYED.H + 1)
    seed_file, tree_file, state = d / "seed", d / "tree.bin", d / "leaf-state.json"
    with open(d / "device.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not seed_file.exists():
            seed = secrets.token_bytes(p.N)
            key = lms.keygen(p, seed, (os.cpu_count() or 1) if workers is None else workers)
            tree_file.write_bytes(b"".join(key.tree[1:]))
            fd = os.open(seed_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(seed)
            lms.LeafState.create(state, lms.public_key(key), p.H)
        else:
            seed, data = seed_file.read_bytes(), tree_file.read_bytes()
            tree = (b"", *(data[p.N * r:p.N * (r + 1)] for r in range(top - 1)))
            key = lms.PrivateKey(p, hashlib.sha512(b"verity/lms/v0/I" + seed).digest()[:16], seed, tree)
        if not (d / "sessions").exists():
            mc.SessionLog.create(d / "sessions")
    return mc.LmsSigner(key, state)


def pin(configuration: Mapping[str, Any], party: str, protocol: str) -> str:
    """The SHA-512 of the key `configuration` pins for `party` in `protocol`, in hex; `Refused` (``pin``) when it pins
    none."""
    for p in configuration.get("pin", ()):
        if (p.get("party"), p.get("protocol")) == (party, protocol):
            return p["key-sha512"]
    raise Refused("pin", f"the configuration pins no key for {party} in {protocol}: a memory challenger signs only "
                         "under a key the auditor pinned ([[pin]])")


def signer(state: Path | None, key_sha512: str) -> mc.LmsSigner:
    """The challenger's signer, its key made or loaded in `state` (`device_key`); `Refused` (``state``) with no
    state directory, and (``pin``) when the SHA-512 of its public key is not `key_sha512`."""
    if state is None:
        raise Refused("state", "the memory challenger keeps its key and logs in its state directory, and has none")
    s = device_key(state)
    if hashlib.sha512(s.public_key).hexdigest() != key_sha512:
        raise Refused("pin", f"the key in {state} is not the one the configuration pins: its SHA-512 is "
                             f"{hashlib.sha512(s.public_key).hexdigest()}")
    return s


__all__ = ["device_key", "pin", "signer"]
