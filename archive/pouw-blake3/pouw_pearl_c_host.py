"""The Pearl-C device path's host helpers (`pouw_pearl_c_device`): device byte buffers and their uploads, the current stream's
handle, each call's keys by native BLAKE3 or SHA-512 (`_Keys`, byte for byte `call_keys`), the pinned upload ring (`_Uploads`) and the
streams a run owns (`_Lane`), and a retained pass's files (`_write`, `write_manifest`).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import blake3
from verity.ml.commitments import RangeIndexedDomain
from verity.ml.commitments.merkle import domain_message
from verity.core.protocols.pouw.audit import HM96, activation_binding, rows_domain

#: the pinned ring's key slots (each call's dom_a ‖ unit, 136 bytes, 168 under -h3s)
UPLOAD_SLOTS = 1024
#: torch's stream priority for each of a schedule's priorities (torch maps values past its range to the nearest valid one)
PRIORITY = {"least": 0, "caller": 0, "greatest": -(1 << 10)}
#: the rows of the call the native keys are held to `call_keys` at, once (any row count of the domain does)
CHECK_ROWS = 16


def torch_uint8() -> Any:
    import torch
    return torch.uint8


def _zeros(n: int, device: Any) -> Any:
    import torch
    return torch.zeros(max(n, 16), dtype=torch.uint8, device=device)


def _put(t: Any, data: bytes) -> None:
    """`data` into device bytes `t`, on the current stream (the pinned staging copy is torch's to keep alive)."""
    import torch
    host = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    t[:len(data)].copy_(host.pin_memory() if t.is_cuda else host, non_blocking=t.is_cuda)


class _TorchStream:
    """torch's Stream object for the current stream, looked up again only when the raw handle changes: `Event.record()`
    without a stream calls `torch.cuda.current_stream`, about 15 µs."""

    def __init__(self):
        self.raw, self.stream = None, None

    def __call__(self, raw: int) -> Any:
        if raw != self.raw:
            import torch
            self.raw, self.stream = raw, torch.cuda.current_stream()
        return self.stream


def _stream(device: Any) -> int:
    """The current stream's handle: torch's raw accessor, not `torch.cuda.current_stream`, which costs about 15 µs a call."""
    import torch
    if device.type != "cuda":
        return 0
    return torch._C._cuda_getCurrentRawStream(device.index if device.index is not None else torch.cuda.current_device())


def _bytes(t: Any) -> bytes:
    return t.cpu().numpy().tobytes()


def call_keys(fixture: Any, salt: bytes, root_b: bytes, index: int, m16: int, hash: str = "blake3") -> bytes:
    """A call's dom_a ‖ unit, as `fixture.keys_of` derives them for matmul `index` of m₁₆ rows, under tree hash `hash` (`HM96`
    is -h3s's)."""
    return (rows_domain(activation_binding(salt, index), index, m16, hash).domain_id
            + fixture.unit_block(salt, root_b, index, "h3s" if hash == HM96 else "h1"))


class _Keys:
    """`call_keys` natively: dom_a is `merkle.domain_message` hashed by the `blake3` package (by hashlib's SHA-512 under
    `HM96`), and unit's label hash is a constant (none under `HM96`).  The native path is held to `call_keys` once at
    construction, so a tree hash whose domain id isn't that digest of that message fails there.  ``native`` set to None takes
    `call_keys` itself (the reference path)."""

    def __init__(self, fixture: Any, salt: bytes, hash: str):
        self.native, self.fixture, self.salt, self.hash = blake3, fixture, salt, hash
        self.sha512 = hash == HM96
        self.label = b"" if self.sha512 else fixture.unit_block(salt, bytes(32), 0)[-32:]
        self.positions: dict[int, bytes] = {}
        root = bytes(64 if self.sha512 else 32)
        if self(root, 0, CHECK_ROWS) != call_keys(fixture, salt, root, 0, CHECK_ROWS, hash):
            raise ValueError(f"native keys don't frame tree hash {hash!r} as CommitmentDomain does")

    def __call__(self, root_b: bytes, index: int, m16: int) -> bytes:
        if self.native is None:
            return call_keys(self.fixture, self.salt, root_b, index, m16, self.hash)
        pos = self.positions.get(m16)
        if pos is None:
            ids = RangeIndexedDomain(m16)
            pos = self.positions[m16] = ids.identity_sha512() if self.sha512 else bytes.fromhex(ids.identity_digest)
        msg = domain_message(activation_binding(self.salt, index), index, pos, m16, "sha512" if self.sha512 else self.hash)
        dom = hashlib.sha512(msg).digest() if self.sha512 else self.native.blake3(msg).digest()
        return dom + self.salt + root_b + index.to_bytes(8, "little") + self.label


class _Lane:
    """A stream the run owns, at a schedule's priority, and its handle; off the GPU a stand-in handle (``standin``) for the
    recording device's log.  ``priority`` is the one the device gave it."""

    def __init__(self, device: Any, priority: str, standin: int):
        import torch
        if device.type == "cuda":
            self.stream = torch.cuda.Stream(device=device, priority=PRIORITY[priority])
            self.raw, self.priority = self.stream.cuda_stream, self.stream.priority
        else:
            self.stream, self.raw, self.priority = None, standin, PRIORITY[priority]
        self.event = torch.cuda.Event() if device.type == "cuda" else None


class _Uploads:
    """Small host-to-device copies through a pinned ring allocated once: each is one copy on the current stream, and a slot is
    written again only once its last copy is done (its event)."""

    def __init__(self, device: Any, slots: int = UPLOAD_SLOTS, width: int = 176):
        import torch
        self.cuda, self.slots, self.i = device.type == "cuda", slots, 0
        self.host = torch.zeros((slots, width), dtype=torch.uint8, pin_memory=self.cuda)
        self.view = self.host.numpy()
        self.events: list[Any] = [None] * slots

    def put(self, dst: Any, data: bytes, stream: Any = None) -> None:
        """`data` into `dst` on `stream` (torch's current stream when None)."""
        import numpy as np
        import torch
        i, n = self.i, len(data)
        self.i = (i + 1) % self.slots
        if self.events[i] is not None:
            self.events[i].synchronize()
        self.view[i, :n] = np.frombuffer(data, np.uint8)
        dst[:n].copy_(self.host[i, :n], non_blocking=self.cuda)
        if self.cuda:
            if self.events[i] is None:
                self.events[i] = torch.cuda.Event()
            self.events[i].record(stream)


def _write(path: Path, data: bytes, digest: bool = True) -> dict:
    """A transcript file.  Row files carry no digest: the committed roots bind them, and the verifier checks every opened row
    against them."""
    path.write_bytes(data)
    nested = path.parent.name in ("calls", "weights")
    out = {"path": f"{path.parent.name}/{path.name}" if nested else path.name, "bytes": len(data)}
    if digest:
        out["sha256"] = hashlib.sha256(data).hexdigest()
    return out


def write_manifest(d: Path, head: dict, weights: dict[str, dict], matmuls: list[dict]) -> Path:
    """The pass's manifest, which `verify_run.py` reads: `head` (the scheme and its arguments, the beacon, the salt, the
    records root, the commitment, the leaves and the tile tree), every weight (n, k, root, excluded B rows, its rows and tree
    levels) and every matmul in order (index, m, k, n, weight, real rows, |K|, excluded A rows, A's root, rows and tree
    levels).  Trees are the device's layout: level 0 padded to a power of two, then each level above it, 32 bytes a node (64
    under -h3s, whose head also holds the prover's hm96 salt key)."""
    path = d / "manifest.json"
    path.write_text(json.dumps({**head, "weights": weights, "matmuls": matmuls}, indent=1) + "\n")
    return path
