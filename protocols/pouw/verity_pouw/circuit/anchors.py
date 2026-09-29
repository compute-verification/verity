"""The anchored inputs' anchors: what each must equal, built by the verifier (the red team's X-SPC-72).

`ncp-v2`'s index binding is Program inputs, so each needs an anchor the verifier owns:

    field       anchor                                                                       input
    salt        the run's salt                                                               `salt`
    forward     the verifier's own enumeration of the window's forwards                      `forward`
    position    (layer, site) from the registered layout: layer << 32 | site                 `position`
    weight id   the registered id of the matrix whose rows B opens under the weights root    `weight`
    row ids     the call's global row indices, distinct                                      `rowids`
    column ids  the weight's global column indices, shards included (X-SPC-40), distinct     `colids`

The verifier builds these inputs itself (`Anchors.call_inputs`, `Anchors.weight_inputs`), or compares a prover's
(`Anchors.check_call`, `Anchors.check_weight`); it never takes them as stated.  A window's `Ledger` refuses a repeated
call index, (forward, position): the Lean statement's A11, `UniqueCallIndices`.

A window's `Receipt` lists every call of the window, each with its anchors, shape and commitment, complete and immutable
before any key exists. `Ledger.open` derives the window's key from it alone, a fresh source under the receipt's digest,
and admits only the calls it lists; `Traced.draw` draws only under that key. So the key comes after every call's
commitment (the window pin's order and root's C1), and each window's source is fresh and used once (C2).
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..protocol import DomainError
from .ncp2 import Shape

M32 = (1 << 32) - 1
M64 = (1 << 64) - 1


def _lane(v: int) -> list[int]:
    return [v & M32, (v >> 32) & M32]


@dataclass(frozen=True)
class Anchors:
    """One call's anchors, as the verifier derives them."""

    salt: bytes
    forward: int
    layer: int
    site: int
    weight: int
    rows: tuple[int, ...]
    cols: tuple[int, ...]

    @property
    def position(self) -> int:
        return (self.layer << 32) | self.site

    @property
    def call_index(self) -> tuple[int, int]:
        return self.forward, self.position

    def check(self, shape: Shape) -> None:
        """Refuse anchors that cannot be a call's: the wrong counts, repeats, or values outside their fields."""
        if len(self.salt) != 32:
            raise DomainError("the salt is 32 bytes")
        if not all(0 <= v <= M64 for v in (self.forward, self.weight)) or not all(0 <= v <= M32 for v in (self.layer, self.site)):
            raise DomainError("the forward index and the weight id are 64-bit, the layer and the site 32-bit")
        for name, ids, count in (("row", self.rows, shape.m), ("column", self.cols, shape.n)):
            if len(ids) != count or len(set(ids)) != count or not all(0 <= v <= M32 for v in ids):
                raise DomainError(f"the {name} ids are {count} distinct 32-bit global indices")

    def _salt_words(self) -> list[int]:
        return [int.from_bytes(self.salt[4 * i:4 * i + 4], "little") for i in range(8)]

    def call_inputs(self) -> dict[str, list[int]]:
        """`NcpLinear`'s anchored inputs, by parameter."""
        return {"salt": self._salt_words(), "forward": _lane(self.forward), "position": _lane(self.position),
                "weight": _lane(self.weight), "rowids": list(self.rows)}

    def weight_inputs(self) -> dict[str, list[int]]:
        """`NcpWeight`'s anchored inputs, by parameter."""
        return {"salt": self._salt_words(), "weight": _lane(self.weight), "colids": list(self.cols)}

    def check_call(self, fn, args: Sequence[Sequence[int]]) -> None:
        """Refuse a call's flat input leaves whose anchored inputs are not these anchors."""
        _check(fn, args, self.call_inputs())

    def check_weight(self, fn, args: Sequence[Sequence[int]]) -> None:
        _check(fn, args, self.weight_inputs())


def _check(fn, args, want: dict[str, list[int]]) -> None:
    params, _ = fn.signature()
    names = [name for name, _t in params]
    if len(args) != len(names):
        raise DomainError(f"{fn.id} takes {len(names)} inputs, not {len(args)}")
    for name, value in want.items():
        if list(args[names.index(name)]) != value:
            raise DomainError(f"{fn.id}: the anchored input {name!r} is not the verifier's")


RECEIPT_TAG = b"verity/pouw/window-receipt/v1\0"
SOURCE_BYTES = 32


def _index_parts(index) -> tuple[int, ...]:
    return tuple(index) if isinstance(index, tuple) else (index,)


@dataclass(frozen=True)
class Receipt:
    """A window's receipt, complete and sealed: every call of the window, each with its anchors, its shape and its
    commitment (the 32- or 64-byte digest the prover committed to), listed once, in any order. It is immutable, so a
    call is in the window exactly when the receipt lists it, and every call's commitment is fixed before the key."""

    calls: tuple[tuple[Any, Any, bytes], ...]

    def __post_init__(self) -> None:
        if not self.calls:
            raise DomainError("a window's receipt lists at least one call")
        seen = set()
        for anchors, shape, commitment in self.calls:
            anchors.check(shape)
            if not (isinstance(commitment, bytes) and len(commitment) in (32, 64)):
                raise DomainError("a call's commitment is a 32- or 64-byte digest")
            if anchors.call_index in seen:
                raise DomainError(f"call index {anchors.call_index} is repeated in the receipt")
            seen.add(anchors.call_index)

    def lists(self, anchors, shape) -> bool:
        return any(a == anchors and s == shape for a, s, _c in self.calls)

    def digest(self) -> bytes:
        """SHA-512 over `RECEIPT_TAG`, the number of calls, and for each call in call-index order: its index (a part
        count, then each part as 8 bytes, big-endian), its shape (m, k, n, 8 bytes each), and its commitment (a length
        byte, then the digest)."""
        h = hashlib.sha512(RECEIPT_TAG + len(self.calls).to_bytes(8, "big"))
        for anchors, shape, commitment in sorted(self.calls, key=lambda c: _index_parts(c[0].call_index)):
            parts = _index_parts(anchors.call_index)
            h.update(len(parts).to_bytes(1, "big") + b"".join(p.to_bytes(8, "big") for p in parts))
            h.update(b"".join(v.to_bytes(8, "big") for v in (shape.m, shape.k, shape.n)))
            h.update(len(commitment).to_bytes(1, "big") + commitment)
        return h.digest()


def _fresh_source() -> bytes:
    """A window's source: 32 fresh bytes from the operating system's CSPRNG (`secrets`), drawn once the receipt is
    complete and used for this window alone (A4's model: one fresh uniform source per derivation)."""
    return secrets.token_bytes(SOURCE_BYTES)


@dataclass
class Ledger:
    """A window's calls, as the verifier admits them: a repeated call index is refused. `Traced.draw` admits each call
    here before it draws, so a window's calls draw under distinct contexts of its one key. A window's ledger is opened
    from its `Receipt` (`Ledger.open`), which fixes its key and the calls it admits; a ledger with no receipt only
    refuses repeats (a window's input builders use that) and draws nothing."""

    seen: set = field(default_factory=set)
    receipt: Receipt | None = None
    key: Any = None

    @classmethod
    def open(cls, receipt: Receipt) -> "Ledger":
        """The window's ledger: its key derived from a fresh source (`_fresh_source`) under `plan.DRAW_DOMAIN` with
        ``{"receipt": receipt.digest()}`` as its context, and only the receipt's calls admitted. No source is taken
        from the caller, so a window's secret is fresh and used once (C2)."""
        from verity.randomness import derive

        from .plan import DRAW_DOMAIN

        if not isinstance(receipt, Receipt):
            raise TypeError("a window's ledger opens from its complete Receipt")
        return cls(receipt=receipt, key=derive(_fresh_source(), DRAW_DOMAIN, {"receipt": receipt.digest()}))

    def admit(self, anchors: Anchors, shape: Shape) -> None:
        anchors.check(shape)
        if self.receipt is not None and not self.receipt.lists(anchors, shape):
            raise DomainError(f"call {anchors.call_index} is not on the window's receipt")
        if anchors.call_index in self.seen:
            raise DomainError(f"call index (forward, position) = {anchors.call_index} is repeated in the window")
        self.seen.add(anchors.call_index)

    def check_draw(self, key) -> None:
        """Refuse a draw from a ledger not opened from a receipt, or under any key but the one it derived."""
        if self.receipt is None or self.key is None:
            raise DomainError("a window draws only from a ledger opened from its complete receipt (Ledger.open)")
        if key != self.key:
            raise DomainError("the key is not the window's: it is derived from the window's receipt (Ledger.open)")


__all__ = ["Anchors", "Ledger", "Receipt"]
