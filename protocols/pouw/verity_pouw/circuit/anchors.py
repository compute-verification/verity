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

import dataclasses
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


def _frame(v) -> bytes:
    """An injective, prefix-free encoding of a receipt's values: bytes, str, ints ≥ 0, tuples and dataclasses (by type
    name, then their fields in order), each tagged and length-delimited."""
    if isinstance(v, bytes):
        return b"b" + len(v).to_bytes(8, "big") + v
    if isinstance(v, str):
        e = v.encode()
        return b"s" + len(e).to_bytes(8, "big") + e
    if isinstance(v, bool) or not isinstance(v, (int, tuple, list)) and not dataclasses.is_dataclass(v):
        raise DomainError(f"a receipt holds bytes, str, ints, tuples and dataclasses, not {type(v).__name__}")
    if isinstance(v, int):
        if v < 0:
            raise DomainError("a receipt's ints are non-negative")
        e = v.to_bytes(max(1, (v.bit_length() + 7) // 8), "big")
        return b"i" + len(e).to_bytes(1, "big") + e
    if isinstance(v, (tuple, list)):
        return b"t" + len(v).to_bytes(8, "big") + b"".join(_frame(x) for x in v)
    return b"d" + _frame(type(v).__name__) + _frame(tuple(getattr(v, f.name) for f in dataclasses.fields(v)))


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

    def entry(self, anchors, shape=None) -> tuple[Any, Any, bytes] | None:
        """The receipt's entry for ``anchors``' call index, if it lists exactly these anchors (and ``shape``)."""
        for a, s, c in self.calls:
            if a.call_index == anchors.call_index:
                return (a, s, c) if a == anchors and (shape is None or s == shape) else None
        return None

    def digest(self) -> bytes:
        """SHA-512 over `RECEIPT_TAG`, the number of calls, and each call in call-index order as `_frame((anchors, shape,
        commitment))`: its full anchors (the call index, salt, weight id, row and column ids), shape and commitment."""
        h = hashlib.sha512(RECEIPT_TAG + len(self.calls).to_bytes(8, "big"))
        for call in sorted(self.calls, key=lambda c: _index_parts(c[0].call_index)):
            h.update(_frame(call))
        return h.digest()


def _fresh_source() -> bytes:
    """A window's source: 32 fresh bytes from the operating system's CSPRNG (`secrets`), drawn once the receipt is
    complete and used for this window alone (A4's model: one fresh uniform source per derivation). A test seam only:
    production code never replaces it."""
    return secrets.token_bytes(SOURCE_BYTES)


class Ledger:
    """A window's calls, as the verifier admits them: a repeated call index is refused.

    A window's ledger comes from `Ledger.open(receipt)`, the only place its receipt, key and source are set: callers
    set no attribute of a ledger, before or after (`__setattr__` refuses), and `Traced.draw` takes the key from the
    ledger. A ledger built as `Ledger()` has none of them; it only refuses repeats (a window's input builders use
    that) and draws nothing. Once the window's draw is sent (`sent`), the ledger draws nothing more, and `record`
    gives the audit record its source and receipt digest."""

    __slots__ = ("seen", "_receipt", "_key", "_source", "_sent")

    def __init__(self) -> None:
        for name, value in (("seen", set()), ("_receipt", None), ("_key", None), ("_source", None), ("_sent", False)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value) -> None:
        raise AttributeError(f"a Ledger's {name!r} is not set by callers: a window's ledger comes from Ledger.open")

    @property
    def receipt(self) -> Receipt | None:
        return self._receipt

    @property
    def key(self):
        return self._key

    @classmethod
    def open(cls, receipt: Receipt) -> "Ledger":
        """The window's ledger: its key derived from a fresh source (`_fresh_source`) under `plan.DRAW_DOMAIN` with
        ``{"receipt": receipt.digest()}`` as its context, and only the receipt's calls admitted. No source is taken
        from the caller, so a window's secret is fresh and used once (C2)."""
        from verity.randomness import derive

        from .plan import DRAW_DOMAIN

        if not isinstance(receipt, Receipt):
            raise TypeError("a window's ledger opens from its complete Receipt")
        ledger, source = cls(), _fresh_source()
        object.__setattr__(ledger, "_receipt", receipt)
        object.__setattr__(ledger, "_source", source)
        object.__setattr__(ledger, "_key", derive(source, DRAW_DOMAIN, {"receipt": receipt.digest()}))
        return ledger

    def admit(self, anchors: Anchors, shape: Shape) -> None:
        anchors.check(shape)
        if self._receipt is not None and self._receipt.entry(anchors, shape) is None:
            raise DomainError(f"call {anchors.call_index} is not on the window's receipt")
        if anchors.call_index in self.seen:
            raise DomainError(f"call index (forward, position) = {anchors.call_index} is repeated in the window")
        self.seen.add(anchors.call_index)

    def check_draw(self) -> None:
        """Refuse a draw from a ledger not opened from a receipt, or once the window's draw has been sent."""
        if self._receipt is None or self._key is None:
            raise DomainError("a window draws only from a ledger opened from its complete receipt (Ledger.open)")
        if self._sent:
            raise DomainError("the window's draw has been sent: it draws nothing more")

    def commitment(self, anchors) -> bytes:
        """The call's commitment as the window's receipt lists it, by call index: the only commitment its openings are
        checked against (`Traced.check_openings`), never one supplied with them."""
        if self._receipt is None:
            raise DomainError("a call's commitment comes from its window's receipt (Ledger.open)")
        entry = self._receipt.entry(anchors)
        if entry is None:
            raise DomainError(f"call {anchors.call_index} is not on the window's receipt")
        return entry[2]

    def sent(self) -> None:
        """Mark the window's draw as sent to the prover: after this the ledger draws nothing, and its record is given."""
        self.check_draw()
        object.__setattr__(self, "_sent", True)

    def record(self) -> dict[str, str]:
        """The window's entry in the audit record once its draw is sent: the receipt digest and the source, from which
        anyone re-derives the key and so the draw (`replay_key`). Revealing the source then helps no prover: the draw is
        public, and the source derives nothing else and is never reused. A third party can check that the draw came
        from the recorded source, not that the source was fresh: a verifier colluding with a prover could pick a source
        whose draw misses. That is the designated verifier's trust; a beacon or a commit-and-reveal would be needed for
        a third party to check it."""
        from .plan import DRAW_DOMAIN

        if not self._sent:
            raise DomainError("a window's source is recorded only once its draw is sent")
        return {"domain": DRAW_DOMAIN, "receipt": self._receipt.digest().hex(), "source": self._source.hex()}


def replay_key(record: dict[str, str]):
    """The window's key from its audit record: `derive(source, DRAW_DOMAIN, {"receipt": digest})`."""
    from verity.randomness import derive

    from .plan import DRAW_DOMAIN

    if record.get("domain") != DRAW_DOMAIN:
        raise DomainError(f"the record's draw domain is not {DRAW_DOMAIN!r}")
    return derive(bytes.fromhex(record["source"]), DRAW_DOMAIN, {"receipt": bytes.fromhex(record["receipt"])})


__all__ = ["Anchors", "Ledger", "Receipt", "replay_key"]
