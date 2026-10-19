"""PoUW's read of the network certifier's per-window salt (`verity/core/protocols/certifier/active.py`, "The salt").

PoUW window p is k = `pouw_window` certifier windows p·k … p·k + k − 1, aligned at 0. The certifier draws p's 512-bit
salt at p's first ingress tick, releases it into the block only, and gives it to the verifier by `reveal(p,
registration)` once the proof service registered p (the receipt the registration's) and p's last ingress record is
published, once, and only until its reveal deadline on its own clock after p's end. PoUW doesn't import the certifier:
`revealed` reads any object with that `reveal`.

* `registered(registration)` is PoUW's reader of the window a registration is for, the certifier's
  `Proxy(registered=…)`: PoUW's served window ``{"kind": "served", "context", "public"}`` (`core_check.commitment`'s),
  whose context names PoUW window p as ``<prefix>/pouw-window/<p>/<the pass manifest's SHA-512>`` (`context`).
* `revealed(certifier, p, registration)` is p's certifier salt, or `Unrevealed`: it fails closed when the registration
  isn't p's (unregistered), when the certifier refuses (a receipt that isn't the registration's: unregistered; before
  t_d: early; past the reveal deadline: late; a second reveal of p, so a second registration naming p gets none; a salt
  never drawn, or the salt off: missing), and when what it gives isn't 64 bytes (missing).
* `committed(certifier, p, registration)` is what PoUW recorded with the certifier for p before it drew p's salt
  (`Proxy.commit_weights`, which takes only a record the prover registered with the certifier signed), given on
  `reveal`'s conditions but the once (`Proxy.committed`), or `Unrevealed` (uncommitted): the `pouw_record` of the
  weights' root, the stream and the prover. The salt binds no weights, so a certified window passes only weights, a
  stream and a prover recorded before its salt (`core_check.pass_epoch`).
* `pouw_salt(reveal, p, stream, prover)` is the salt PoUW applies in window p, the one function that states its form,
  `pouw_record(weights_root, stream, prover)` what PoUW records for p before the salt, and `prover_id(public_key)` the
  64-byte id of the prover whose key the certifier verifies records under.
* `Credits` is the certified windows' credit: one credit per PoUW window, stream and prover.
* `pouw_window(windows, k)` is the PoUW window certifier windows make up, refusing any that aren't one whole, aligned
  PoUW window.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .protocol import U64, DomainError

if TYPE_CHECKING:
    from verity.core.protocols.sampled_proofs.service import Registration

#: the certifier's salt, and the salt PoUW applies: SHA-512 digests
SALT_BYTES = 64
#: `pouw_salt`'s domain tag, which keeps its input apart from `window_salt`'s
SALT_TAG = b"verity/pouw-salt/v1"
#: `pouw_record`'s domain tag
RECORD_TAG = b"verity/pouw-record/v1"
#: `prover_id`'s domain tag
PROVER_TAG = b"verity/pouw-prover/v1"
#: `audit.weights_root`, a SHA-512 `tagged`
WEIGHTS_ROOT_BYTES = 64
#: a certified window's stream, the SHA-512 digest that names it, and its prover, the 64-byte id it is registered under
STREAM_BYTES = 64
PROVER_BYTES = 64
WINDOW_SEGMENT = "pouw-window"
_NAT = re.compile(r"0|[1-9][0-9]*")


class Unrevealed(DomainError):
    """A PoUW window whose certifier salt, or weights recorded before it, the verifier doesn't have: unregistered,
    early, missing or uncommitted."""


class Credited(DomainError):
    """A PoUW window already credited to a registration of its stream and prover."""


def _window(window: Any) -> int:
    if not (type(window) is int and 0 <= window < U64):
        raise DomainError(f"a PoUW window is a u64, not {window!r}")
    return window


def _bound(stream: Any, prover: Any) -> bytes:
    if type(stream) is not bytes or len(stream) != STREAM_BYTES:
        raise DomainError(f"a certified window's stream is a {STREAM_BYTES}-byte digest")
    if type(prover) is not bytes or len(prover) != PROVER_BYTES:
        raise DomainError(f"a certified window's prover is a {PROVER_BYTES}-byte id")
    return stream + prover


def pouw_salt(reveal: bytes, window: int, stream: bytes, prover: bytes) -> bytes:
    """The salt PoUW applies in window `window`, from the certifier's salt `reveal`, for the registration of `stream`
    and `prover`: SHA-512(`SALT_TAG` ‖ reveal ‖ u64be(window) ‖ stream ‖ prover), every part fixed-width. This is the
    only place that states it (Lean's `PouwCheck.pouwSalt` mirrors it), and `test_pouw_salt_form` pins it.

    `stream` and `prover` are recorded with the certifier before it draws the window's salt (`pouw_record`), so two
    registrations of one window are salted apart, and the prover can't grind its seeds over them once it holds the salt.
    The window is global, the certifier's PoUW window on its trusted clock, whose salt the certifier draws once from
    fresh randomness, so no epoch is bound beside it."""
    if type(reveal) is not bytes or len(reveal) != SALT_BYTES:
        raise DomainError(f"the certifier's salt is {SALT_BYTES} bytes")
    return hashlib.sha512(SALT_TAG + reveal + _window(window).to_bytes(8, "big") + _bound(stream, prover)).digest()


def pouw_record(weights_root: bytes, stream: bytes, prover: bytes) -> bytes:
    """What PoUW records with the certifier for a PoUW window before its salt is drawn (`Proxy.commit_weights`), and
    `committed` reads back: SHA-512(`RECORD_TAG` ‖ weights_root ‖ stream ‖ prover), every part fixed-width (Lean's
    `PouwCheck.pouwRecord`). So the certifier's word orders the weights, the stream and the prover before the salt."""
    if type(weights_root) is not bytes or len(weights_root) != WEIGHTS_ROOT_BYTES:
        raise DomainError(f"a weights' root is {WEIGHTS_ROOT_BYTES} bytes")
    return hashlib.sha512(RECORD_TAG + weights_root + _bound(stream, prover)).digest()


def prover_id(public_key: bytes) -> bytes:
    """The 64-byte id a certified window's prover is registered under: SHA-512(`PROVER_TAG` ‖ the public key the
    certifier verifies its records under, `Proxy(prover_key=…)`'s bytes). A record naming the id is taken only signed by
    that key."""
    if type(public_key) is not bytes or not public_key:
        raise DomainError("a prover's public key is non-empty bytes")
    return hashlib.sha512(PROVER_TAG + public_key).digest()


class Credits:
    """The certified windows' credit: a PoUW window is credited once per stream and prover, to one registration. A
    window's tiles are private, so this is the finest public key there is. Two registrations of one window under one
    stream and prover have one salt and so one seed per tile, and `credit` refuses the second (`Credited`). Under
    another stream or prover the seeds differ (`pouw_salt`)."""

    def __init__(self) -> None:
        self._credited: dict[tuple[int, bytes], str] = {}

    def credit(self, window: int, stream: bytes, prover: bytes, registration: str) -> None:
        """Credit PoUW window `window` to `registration` (its digest) under `stream` and `prover`, or refuse
        (`Credited`) a window already credited under them."""
        key = (_window(window), _bound(stream, prover))
        if key in self._credited:
            raise Credited(f"PoUW window {window} is already credited to registration {self._credited[key][:16]} of its "
                           "stream and prover: a tile is credited once")
        self._credited[key] = registration


def context(prefix: str, window: int, manifest: str) -> str:
    """The served window's context for PoUW window `window`, whose pass's manifest has SHA-512 `manifest` (lowercase
    hex): `core_check.pass_epoch` reads the manifest's digest off its last segment, `window_of` the window off the one
    before."""
    return f"{prefix}/{WINDOW_SEGMENT}/{_window(window)}/{manifest}"


def window_of(ctx: Any) -> int | None:
    """The PoUW window a context names (`context`), or None."""
    parts = ctx.split("/") if isinstance(ctx, str) else []
    if len(parts) < 3 or parts[-3] != WINDOW_SEGMENT or not _NAT.fullmatch(parts[-2]):
        return None
    p = int(parts[-2])
    return p if p < U64 else None


def registered(registration: Registration) -> int | None:
    """The PoUW window `registration` is for: its record's window is PoUW's served one, ``{"kind": "served", "context",
    "public"}``, whose context names it (`window_of`); else None."""
    rec = getattr(registration, "record", None)
    w = rec.get("window") if isinstance(rec, Mapping) else None
    if not (isinstance(w, Mapping) and set(w) == {"context", "kind", "public"} and w["kind"] == "served"
            and isinstance(w["public"], Mapping)):
        return None
    return window_of(w["context"])


def revealed(certifier: Any, window: int, registration: Registration | None) -> bytes:
    """PoUW window `window`'s certifier salt, `certifier.reveal(window, registration)`, given the window's registration
    as the proof service returned it. It fails closed (`Unrevealed`): unregistered (`registered` doesn't read `window`
    off the registration), refused by the certifier (its receipt isn't the registration's; early, before the window's
    last ingress record is published; late, past its reveal deadline; revealed already; missing, a salt never drawn or
    the salt off), or missing (what the certifier gave isn't a 64-byte salt)."""
    if registered(registration) != _window(window):
        raise Unrevealed(f"unregistered: no registration of PoUW window {window}")
    try:
        salt = certifier.reveal(window, registration)
    except (RuntimeError, ValueError) as e:
        raise Unrevealed(f"the certifier doesn't reveal PoUW window {window}'s salt: {e}") from None
    if type(salt) is not bytes or len(salt) != SALT_BYTES:
        raise Unrevealed(f"missing: the certifier's reveal of PoUW window {window} is not a {SALT_BYTES}-byte salt")
    return salt


def committed(certifier: Any, window: int, registration: Registration | None) -> bytes:
    """What PoUW recorded with the certifier for PoUW window `window` before it drew the window's salt, the 64-byte
    `pouw_record` of the weights' root, the stream and the prover: `certifier.committed(window, registration)`
    (`Proxy.commit_weights` records it, signed by the registered prover), given the window's registration. It fails
    closed (`Unrevealed`): unregistered, as `revealed`; refused by the certifier (on `reveal`'s conditions but the once,
    or uncommitted: nothing recorded before the draw); or uncommitted, when what the certifier gives isn't 64 bytes."""
    if registered(registration) != _window(window):
        raise Unrevealed(f"unregistered: no registration of PoUW window {window}")
    try:
        record = certifier.committed(window, registration)
    except (RuntimeError, ValueError) as e:
        raise Unrevealed(f"the certifier doesn't give what PoUW recorded for window {window}: {e}") from None
    if type(record) is not bytes or len(record) != WEIGHTS_ROOT_BYTES:
        raise Unrevealed(f"uncommitted: what the certifier recorded for PoUW window {window} is not "
                         f"{WEIGHTS_ROOT_BYTES} bytes")
    return record


def pouw_window(windows: Sequence[int], k: int) -> int:
    """The PoUW window certifier windows `windows` make up, k certifier windows each: p when they are p·k … p·k + k − 1
    in order, one whole PoUW window aligned at 0; else `DomainError`."""
    if not (type(k) is int and k >= 1):
        raise DomainError(f"a PoUW window is a positive number of certifier windows, not {k!r}")
    ws = list(windows)
    if not ws or any(type(w) is not int for w in ws) or ws[0] < 0 or ws[0] % k or ws != list(range(ws[0], ws[0] + k)) \
            or ws[0] // k >= U64:
        raise DomainError(f"certifier windows {ws} are not one PoUW window of {k}: p·{k} … p·{k} + {k - 1}, aligned at 0")
    return ws[0] // k


__all__ = ["Credited", "Credits", "PROVER_BYTES", "PROVER_TAG", "RECORD_TAG", "SALT_BYTES", "SALT_TAG", "STREAM_BYTES",
           "Unrevealed", "WEIGHTS_ROOT_BYTES", "WINDOW_SEGMENT", "committed", "context", "pouw_record", "pouw_salt",
           "pouw_window", "prover_id", "registered", "revealed", "window_of"]
