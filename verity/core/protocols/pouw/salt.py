"""PoUW's read of the network certifier's per-window salt (`verity/core/protocols/certifier/active.py`, "The salt").

PoUW window p is k = `pouw_window` certifier windows p·k … p·k + k − 1, aligned at 0. The certifier draws p's 512-bit
salt at p's first ingress tick, releases it into the block only, and gives it to the verifier by `reveal(p,
registration)` once the proof service registered p (the receipt the registration's) and p's last ingress record is
published. PoUW doesn't import the certifier: `revealed` reads any object with that `reveal`.

* `registered(registration)` is PoUW's reader of the window a registration is for, the certifier's
  `Proxy(registered=…)`: PoUW's served window ``{"kind": "served", "context", "public"}`` (`core_check.commitment`'s),
  whose context names PoUW window p as ``<prefix>/pouw-window/<p>/<the pass manifest's SHA-512>`` (`context`).
* `revealed(certifier, p, registration)` is p's certifier salt, or `Unrevealed`: it fails closed when the registration
  isn't p's (unregistered), when the certifier refuses (a receipt that isn't the registration's: unregistered; before
  t_d: early; a salt never drawn, or the salt off: missing), and when what it gives isn't 64 bytes (missing).
* `pouw_salt(reveal, p)` is the salt PoUW applies in window p, the one function that states its form.
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
WINDOW_SEGMENT = "pouw-window"
_NAT = re.compile(r"0|[1-9][0-9]*")


class Unrevealed(DomainError):
    """A PoUW window whose certifier salt the verifier doesn't have: unregistered, early or missing."""


def _window(window: Any) -> int:
    if not (type(window) is int and 0 <= window < U64):
        raise DomainError(f"a PoUW window is a u64, not {window!r}")
    return window


def pouw_salt(reveal: bytes, window: int) -> bytes:
    """The salt PoUW applies in window `window`, from the certifier's salt `reveal`: SHA-512(reveal ‖ u64be(window)).
    The form isn't settled; this is the only place that states it, and `test_pouw_salt_form` pins it."""
    if type(reveal) is not bytes or len(reveal) != SALT_BYTES:
        raise DomainError(f"the certifier's salt is {SALT_BYTES} bytes")
    return hashlib.sha512(reveal + _window(window).to_bytes(8, "big")).digest()


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
    last ingress record is published; missing, a salt never drawn or the salt off), or missing (what the certifier gave
    isn't a 64-byte salt)."""
    if registered(registration) != _window(window):
        raise Unrevealed(f"unregistered: no registration of PoUW window {window}")
    try:
        salt = certifier.reveal(window, registration)
    except (RuntimeError, ValueError) as e:
        raise Unrevealed(f"the certifier doesn't reveal PoUW window {window}'s salt: {e}") from None
    if type(salt) is not bytes or len(salt) != SALT_BYTES:
        raise Unrevealed(f"missing: the certifier's reveal of PoUW window {window} is not a {SALT_BYTES}-byte salt")
    return salt


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


__all__ = ["SALT_BYTES", "Unrevealed", "WINDOW_SEGMENT", "context", "pouw_salt", "pouw_window", "registered", "revealed",
           "window_of"]
