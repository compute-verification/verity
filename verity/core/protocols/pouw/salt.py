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
* `Registrations` is the certified windows' once-only registration ledger: one registration per PoUW window, stream and
  prover, taken when the challenger receives the window, before its draw and any proof. It is not the window's credit,
  which is its outcome's verdict.
* `unit_reason(unit)` is why a certified window can't credit a drawn unit: it credits only units that read its salt.
* `pouw_window(windows, k)` is the PoUW window certifier windows make up, refusing any that aren't one whole, aligned
  PoUW window.
* `window_open_root(weights_root, requests_root, session)` is a live window's open root: what the auditor's gate opens
  the window's challenger session on and its salt coin opens after, which the verifier recomputes from the window's
  record and the session the gate reserved (`core_check.salt`).
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from verity.ml.circuits import ports as _ports
from verity.ml.commitments import domains
from verity.ml.commitments.scheme import DIGEST_BYTES

from .protocol import U64, DomainError

if TYPE_CHECKING:
    from verity.core.protocols.sampled_proofs.service import Registration

#: the certifier's salt, and the salt PoUW applies: SHA-512 digests
SALT_BYTES = DIGEST_BYTES["sha512"]
#: `pouw_salt`'s domain tag, which keeps its input apart from `window_salt`'s
SALT_TAG = domains.POUW_PERIOD_SALT.encode()
#: `pouw_record`'s domain tag
RECORD_TAG = domains.POUW_RECORD.encode()
#: `prover_id`'s domain tag
PROVER_TAG = domains.POUW_PROVER.encode()
#: a weights' root, a SHA-512 `tagged`: an H1 window's `weights.weights_root` (`window_open_root`), an epoch's
#: `audit.weights_root` (`pouw_record`)
WEIGHTS_ROOT_BYTES = DIGEST_BYTES["sha512"]
#: a certified window's stream, the SHA-512 digest that names it, and its prover, the 64-byte id it is registered under
STREAM_BYTES = DIGEST_BYTES["sha512"]
PROVER_BYTES = DIGEST_BYTES["sha512"]
WINDOW_SEGMENT = "pouw-window"
#: `window_open_root`'s domain tag
WINDOW_OPEN_TAG = domains.POUW_PERIOD_OPEN.encode()
#: `epoch_open_root`'s domain tag
EPOCH_OPEN_TAG = domains.POUW_EPOCH_OPEN.encode()
#: `anchored_epoch_root`'s domain tag: `EPOCH_OPEN_TAG`'s, its version bumped for the anchors' root in its preimage
ANCHORED_EPOCH_TAG = domains.POUW_ANCHORED_EPOCH_OPEN.encode()
#: an H1 window's anchors' root X (`core_check.values_root` of its committed activations)
ANCHORS_ROOT_BYTES = DIGEST_BYTES["sha512"]
#: a live window's round-0 request table's root, a frame-v3-sha512 root
REQUESTS_ROOT_BYTES = DIGEST_BYTES["sha512"]
#: a challenger session's id, as the auditor's gate assigns it when it reserves the session (``s20261020T114752Z-9e52``)
SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
_NAT = re.compile(r"0|[1-9][0-9]*")


class Unrevealed(DomainError):
    """A PoUW window whose certifier salt, or weights recorded before it, the verifier doesn't have: unregistered,
    early, missing or uncommitted."""


class Reregistered(DomainError):
    """A PoUW window already registered under its stream and prover."""


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
    The window is the certifier's PoUW window, counted from its origin on its trusted clock, whose salt it draws once
    from fresh randomness, so no epoch is bound beside it. A proxy started again counts from 0 with fresh salts, so the
    stream a window is registered under names the certifier's run."""
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


def window_open_root(weights_root: bytes, requests_root: bytes, session: str) -> bytes:
    """A live window's open root: SHA-512(`WINDOW_OPEN_TAG` ‖ weights_root ‖ requests_root ‖ SHA-512(session)), every
    part 64 bytes, the session's id hashed from its ASCII (Lean's `PouwCheck.windowOpenRoot` mirrors it, and
    `test_window_open_root_form` pins it).

    `weights_root` is `weights.weights_root`, the fold of each K's registered ``<PREFIX>b/<K>`` root (the B rows, int8
    codes and s_w, of the window's linears of that K; not `audit.weights_root`, the epoch's commitment to int8 rows
    alone), `requests_root` the root of the window's round-0 request table, committed before its salt, and `session` the
    id the auditor's gate assigned when it reserved the window's challenger session, which the prover can't relabel. The gate opens that session on this root (``open --session S --root R``), its
    challenger opens the salt coin only after it, and the verifier recomputes it from the window's record and the
    session its signed reserve names (`core_check.salt`). Each session has its own root, and a root fixes one window's
    weights and requests, so one session salts one window."""
    if type(weights_root) is not bytes or len(weights_root) != WEIGHTS_ROOT_BYTES:
        raise DomainError(f"a weights' root is {WEIGHTS_ROOT_BYTES} bytes")
    if type(requests_root) is not bytes or len(requests_root) != REQUESTS_ROOT_BYTES:
        raise DomainError(f"a round-0 request table's root is {REQUESTS_ROOT_BYTES} bytes")
    if not (isinstance(session, str) and SESSION.fullmatch(session)):
        raise DomainError(f"a challenger session's id is 1 to 64 of [A-Za-z0-9._:-], starting alphanumeric, not {session!r}")
    return hashlib.sha512(WINDOW_OPEN_TAG + weights_root + requests_root
                          + hashlib.sha512(session.encode("ascii")).digest()).digest()


def epoch_open_root(weights_root: bytes, k: int) -> bytes:
    """A Pearl-C salt epoch's open root: SHA-512(`EPOCH_OPEN_TAG` ‖ weights_root ‖ u64be(k)), 97 bytes hashed with no
    separator, every part at a fixed offset (Lean's `PouwCheck.epochOpenRoot` mirrors it, and `test_epoch_open_root_form`
    pins it).

    `weights_root` is the pass's `audit.weights_root`, W, and `k` its manifest's ``pass_index``: the epoch's number among
    the epochs over W (one install, or one measured interval of an install), from 0, the same in every pass of the
    epoch. The gate opens the epoch's salt-only session on this root, its challenger opens the salt coin only after it,
    and the verifier recomputes it from the pass's manifest (`core_check.live_salt`). The ledger opens a root once, so
    the passes of one epoch read one session and one salt."""
    if type(weights_root) is not bytes or len(weights_root) != WEIGHTS_ROOT_BYTES:
        raise DomainError(f"a weights' root is {WEIGHTS_ROOT_BYTES} bytes")
    if not (type(k) is int and 0 <= k < 1 << 64):
        raise DomainError(f"a salt epoch's number is an integer in [0, 2^64), not {k!r}")
    return hashlib.sha512(EPOCH_OPEN_TAG + weights_root + k.to_bytes(8, "big")).digest()


def anchored_epoch_root(weights_root: bytes, anchors_root: bytes, k: int) -> bytes:
    """An H1 window's salt epoch's open root: SHA-512(`ANCHORED_EPOCH_TAG` ‖ weights_root ‖ anchors_root ‖ u64be(k)),
    161 bytes hashed with no separator, every part at a fixed offset (Lean's `PouwCheck.anchoredEpochRoot` mirrors it,
    and `test_anchored_epoch_root_form` pins it).

    `weights_root` is the window's W and `anchors_root` its X, the root of the activations it committed after serving
    (`core_check.window_anchors`), and `k` its salt epoch's number among W's. The gate opens the window's salt-only
    session on this root and its challenger opens the salt coin only after it, so the operands H1's cut reads at coin 0
    (x, sx, the A rows) are fixed before the salt. Its tag is not `epoch_open_root`'s, so no Pearl-C epoch's session
    salts a window."""
    if type(weights_root) is not bytes or len(weights_root) != WEIGHTS_ROOT_BYTES:
        raise DomainError(f"a weights' root is {WEIGHTS_ROOT_BYTES} bytes")
    if type(anchors_root) is not bytes or len(anchors_root) != ANCHORS_ROOT_BYTES:
        raise DomainError(f"an anchors' root is {ANCHORS_ROOT_BYTES} bytes")
    if not (type(k) is int and 0 <= k < 1 << 64):
        raise DomainError(f"a salt epoch's number is an integer in [0, 2^64), not {k!r}")
    return hashlib.sha512(ANCHORED_EPOCH_TAG + weights_root + anchors_root + k.to_bytes(8, "big")).digest()


def prover_id(public_key: bytes) -> bytes:
    """The 64-byte id a certified window's prover is registered under: SHA-512(`PROVER_TAG` ‖ the public key the
    certifier verifies its records under, `Proxy(prover_key=…)`'s bytes). A record naming the id is taken only signed by
    that key."""
    if type(public_key) is not bytes or not public_key:
        raise DomainError("a prover's public key is non-empty bytes")
    return hashlib.sha512(PROVER_TAG + public_key).digest()


def unit_reason(unit: Any) -> str | None:
    """Why a certified window can't credit `unit`, its drawn unit's Definition, or None: a certified window credits only
    units that read its salt, a unit whose ``salt`` parameter is a coin port over the whole 64-byte salt
    (`ports.In(kind="coin")`), as the compiled Program's units take it (`circuit.compile`, `PouwCompiled_v1`, whose
    ``salt-flow`` check holds it to each unit's seed). A unit that reads only committed rows passes on rows formed under any
    salt, so one product formed once would pass in every window and under every stream. This is necessary, not enough:
    the window that runs such a unit must also hold the proof's coin input to the window's salt."""
    pt = _ports.port_types(unit)
    params = unit.signature()[0] if hasattr(unit, "signature") else ()
    runs = dict(zip((name for name, _ in params), pt.params)).get("salt") if pt is not None else None
    if runs and all(p.kind == "coin" for _, _, p, _ in runs) and (runs[0][0], runs[-1][1]) == (0, 8 * SALT_BYTES) \
            and all(a[1] == b[0] for a, b in zip(runs, runs[1:])):
        return None
    name = getattr(unit, "id", repr(unit))
    return (f"a certified window credits only units that read its salt, and {name} reads none: no parameter of it is "
            f"the {SALT_BYTES}-byte salt as a coin port, as the compiled Program's units take it (`PouwCompiled_v1`), so "
            "its proof passes on rows formed under any salt")


# TODO: a window's credit is its outcome's verdict within a period: the period's cap and circuit recorded before
# window 0's first coin, each window's first coin after the previous window's draw, closed at the first refused window.
# `Registrations` is none of that.
class Registrations:
    """The certified windows' once-only registration ledger: a PoUW window registers once per stream and prover, when
    the challenger receives it, before its draw and before any proof, so a window whose units are never proved, or are
    refused, keeps its entry. A window's tiles are private, so (window, stream, prover) is the finest public key there
    is. Two registrations of one window under one stream and prover have one salt and so one seed per tile, and `admit`
    refuses the second (`Reregistered`). Under another stream or prover the salts differ (`pouw_salt`)."""

    def __init__(self) -> None:
        self._registered: dict[tuple[int, bytes], str] = {}

    def admit(self, window: int, stream: bytes, prover: bytes, registration: str) -> None:
        """Record PoUW window `window`'s registration `registration` (its digest) under `stream` and `prover`, or refuse
        (`Reregistered`) a window already registered under them."""
        key = (_window(window), _bound(stream, prover))
        if key in self._registered:
            raise Reregistered(f"PoUW window {window} is already registered ({self._registered[key][:16]}) under its "
                               "stream and prover: a window registers once")
        self._registered[key] = registration

    def __contains__(self, key: tuple[int, bytes, bytes]) -> bool:
        window, stream, prover = key
        return (_window(window), _bound(stream, prover)) in self._registered

    def __len__(self) -> int:
        return len(self._registered)


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


__all__ = ["ANCHORED_EPOCH_TAG", "ANCHORS_ROOT_BYTES", "EPOCH_OPEN_TAG", "PROVER_BYTES", "PROVER_TAG", "RECORD_TAG",
           "REQUESTS_ROOT_BYTES", "Registrations", "Reregistered", "SALT_BYTES", "SALT_TAG", "SESSION", "STREAM_BYTES",
           "Unrevealed", "WEIGHTS_ROOT_BYTES", "WINDOW_OPEN_TAG", "WINDOW_SEGMENT", "anchored_epoch_root", "committed",
           "context", "epoch_open_root", "pouw_record", "pouw_salt", "pouw_window", "prover_id", "registered",
           "revealed", "unit_reason", "window_of", "window_open_root"]
