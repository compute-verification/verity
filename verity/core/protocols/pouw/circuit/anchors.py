"""The anchored inputs' anchors: what each must equal, built by the verifier.

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
call index, (forward, position): the Lean statement's A11, `UniqueCallIndices`. `SaltWindows` refuses one across every
window of a salt, since a call's noise key reads the salt and its call index.

The tile checks' (`pc4`, `pc8`) anchored inputs are derived values, not ids: the noise lines E_A, E_B, F_A, F_B and
G = F_Bᵀ·F_A, which the verifier derives from the scheme's seeds (`TileAnchors`, from `noise(ctx)`: seed_A reads the salt,
A's root, B's registered root and the matmul index, seed_B the salt and B's root).  The call's Program and the weight's both
read F_A and F_B; E_A is the call's alone, E_B and G the weight's.  In a unit's proof each is a Flock public input region
(an ``In`` region): the unit's ports (`pc4.PORTS`, `pc8.PORTS`) name it, with its recomputation (`Derivation`: the rule, the
seed, the domain and the derivation function, `Shape.derivations`; as Flock's circuit names it, `public_ports`: the seed's
label and the rule's versioned id), and the verifier values it from its own `--public-inputs` file
(`TileAnchors.public_inputs`), whose content for one unit is `TileAnchors.regions`: each region's u16 words
(`region_words`), every value unpacked, the region whole 128-bit words.  A row unit cut along k (`rowk`) reads F_A and F_B
one segment at a time: each form's region is its segment's lines (`Derivation.per_segment`), under the anchor's slice
rule (``verity/pouw/anchor/<scheme>/<anchor>/slice/v<n>``, at the anchor's own version), which only the cut's shapes
name.

A window's `Receipt` lists every call of the window, each with its anchors, shape and commitment, complete and immutable
before any key exists. `Ledger.open` derives the window's key from it alone, a fresh source under the receipt's digest,
and admits only the calls it lists; `Traced.draw` draws only under that key. So the key comes after every call's
commitment (the window pin's order and root's C1), and each window's source is fresh and used once (C2).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import secrets
from dataclasses import dataclass
from typing import Any, Callable, Sequence

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


#: the tile checks' anchored parameters, by name in their Programs
ANCHORED = ("ea", "eb", "fa", "fb", "g")
#: a public input region's word, as Flock lays out a unit's input leaves, and the input word a region fills whole
WORD_BITS, REGION_WORD_BITS = 16, 128


#: an anchor's rule id: the scheme, the anchor (with ``/slice``, one segment of it: `rowk`) and the rule's version, never
#: a workload's values or a Python name
ANCHOR_RULE = re.compile(r"verity/pouw/anchor/[a-z0-9-]+/[a-z0-9]+(/slice)?/v(0|[1-9][0-9]*)")


@dataclass(frozen=True)
class Derivation:
    """How the verifier recomputes one anchor, which its public input region names: its rule (``rule``, `ANCHOR_RULE`'s
    form: the id Flock's circuit pins, so a new rule is a new version), the seed it reads (``seed``: the scheme's
    ``noise(ctx).seed_a`` or ``.seed_b``), the scheme's domain separation of it (``domain``), the function that derives it
    (``fn(shape, seed, index)``: one line for global row ``index`` when ``per_row``, segment ``index`` of the anchor when
    ``per_segment``, else the whole anchor, flat in its parameter's leaf order) and each value's width in bits."""

    rule: str
    seed: str
    domain: str
    fn: Callable[[Any, bytes, int | None], Sequence[int]]
    bits: int
    per_row: bool = False
    per_segment: bool = False

    def __post_init__(self) -> None:
        rule = isinstance(self.rule, str) and ANCHOR_RULE.fullmatch(self.rule)
        if not rule:
            raise ValueError("an anchor's rule is a versioned id, verity/pouw/anchor/<scheme>/<anchor>[/slice]/v<n>: "
                             f"{self.rule!r}")
        if (rule.group(1) is not None) != self.per_segment or self.per_row and self.per_segment:
            raise ValueError(f"an anchor's rule names a slice exactly when it is derived per segment: {self.rule!r}")


def region_words(values: Sequence[int], bits: int) -> list[int]:
    """Values of ``bits`` bits as a public input region's u16 words, unpacked as Flock lays out a unit's input leaves: each
    value takes ceil(bits / 16) words, low half first, the bits above it zero.  So bit t of word j is the region's bit
    16j + t, and its bytes are the words little-endian (`region_bytes`).  The region must fill whole 128-bit words."""
    if not all(0 <= v < 1 << bits for v in values):
        raise DomainError(f"a region's values are {bits}-bit")
    words = [(v >> (WORD_BITS * h)) & 0xFFFF for v in values for h in range(-(-bits // WORD_BITS))]
    if len(words) * WORD_BITS % REGION_WORD_BITS:
        raise DomainError(f"a region of {len(words)} u16 words does not fill whole {REGION_WORD_BITS}-bit words")
    return words


def region_bytes(words: Sequence[int]) -> bytes:
    """A region's u16 words as its bytes, little-endian: Flock's ``bytes(region, b)`` for one unit."""
    return b"".join(w.to_bytes(2, "little") for w in words)


#: Flock's format of the verifier's own `--public-inputs` file (its PROTOCOL.md §16.10, ``HmIn.checkOwn``)
PUBLIC_INPUTS_FORMAT = "verity/flock-public-inputs/v1"


def public_ports(shape, role: str) -> dict[str, dict[str, str]]:
    """``role``'s public input ports as Flock's circuit names them (``compose``'s ``public_inputs``, META's
    ``public_inputs``), in the unit's parameter order: each region's seed by the scheme's label for it (``shape.seeds``) and
    its derivation by its rule id (`Derivation.rule`)."""
    out = {}
    for port, anchor in shape.ports[role]["regions"].items():
        d = shape.derivations[anchor]
        out[port] = {"seed": shape.seeds[d.seed], "derivation": d.rule}
    return out


@dataclass(frozen=True)
class TileAnchors:
    """One tile check's anchored inputs, as the verifier derives them: the scheme's seeds (`noise(ctx)`) and the global
    indices of the block's rows of A (``rows``) and of B (``cols``), by which each E line is drawn.  ``shape`` is the
    check's `pc4.Shape` or `pc8.Shape`: its ``derivations`` derive the lines and G on its own record, and its ``ports`` name
    each unit's public input regions."""

    shape: Any
    seed_a: bytes
    seed_b: bytes
    rows: tuple[int, ...]
    cols: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.seed_a) != 32 or len(self.seed_b) != 32:
            raise DomainError("the seeds are 32 bytes")
        for name, ids, count in (("row", self.rows, self.shape.m), ("column", self.cols, self.shape.n)):
            if len(ids) != count or len(set(ids)) != count or not all(0 <= v <= M32 for v in ids):
                raise DomainError(f"the {name} ids are {count} distinct 32-bit global indices")

    @classmethod
    def of(cls, scheme, ctx, shape, rows: Sequence[int] | None = None, cols: Sequence[int] | None = None) -> "TileAnchors":
        """The anchors of ``ctx``'s block under ``scheme`` (rows and columns 0.. by default)."""
        noise = scheme.noise(ctx)
        rows = tuple(range(shape.m)) if rows is None else tuple(rows)
        cols = tuple(range(shape.n)) if cols is None else tuple(cols)
        return cls(shape, noise.seed_a, noise.seed_b, rows, cols)

    def _derive(self, anchor: str, ids: Sequence[int], segment: int | None = None) -> list[int]:
        d = self.shape.derivations[anchor]
        seed = {"seed_a": self.seed_a, "seed_b": self.seed_b}[d.seed]
        if d.per_row:
            return [v for i in ids for v in d.fn(self.shape, seed, i)]
        return list(d.fn(self.shape, seed, segment if d.per_segment else None))

    def _ids(self, anchor: str) -> tuple[int, ...]:
        return {"ea": self.rows, "eb": self.cols}.get(anchor, ())

    def inputs(self, fn) -> dict[str, list[int]]:
        """``fn``'s anchored inputs, by parameter: those of `ANCHORED` it takes."""
        return {name: self._derive(name, self._ids(name)) for name, _t in fn.signature()[0] if name in ANCHORED}

    def check_inputs(self, fn, args: Sequence[Sequence[int]]) -> None:
        """Refuse ``fn``'s flat input leaves whose anchored inputs are not the verifier's."""
        _check(fn, args, self.inputs(fn))

    def regions(self, role: str, index: int | None = None, segment: int | None = None) -> dict[str, list[int]]:
        """One unit's `--public-inputs` content: each public input region its ports name (``shape.ports[role]["regions"]``,
        in the unit's parameter order) as its u16 words (`region_words`), derived from the seeds.  ``role``, ``index``
        and ``segment`` are the unit's coordinates: a row unit's row in the block (A's for ``row_a``, B's for ``row_b``),
        whose global id draws its E line, and a `rowk` form's segment of k, whose slice of F_A and F_B it reads; a tile
        unit has no region."""
        return {port: self._region(role, port, index, segment) for port in self.shape.ports[role]["regions"]}

    def _region(self, role: str, port: str, index: int | None, segment: int | None = None) -> list[int]:
        anchor = self.shape.ports[role]["regions"][port]
        d, ids = self.shape.derivations[anchor], self._ids(anchor)
        if d.per_row:
            if not (type(index) is int and 0 <= index < len(ids)):
                raise DomainError(f"a {role} unit's row is one of the block's {len(ids)}, not {index!r}")
            ids = (ids[index],)
        if d.per_segment:
            count = self.shape.k // self.shape.row_k
            if not (type(segment) is int and 0 <= segment < count):
                raise DomainError(f"a {role} unit's segment is one of the row's {count}, not {segment!r}")
        return region_words(self._derive(anchor, ids, segment), d.bits)

    def public_inputs(self, role: str, indexes: Sequence[int], segment: int | None = None) -> bytes:
        """The verifier's own `--public-inputs` file for the statement whose instances are ``role``'s units at ``indexes``
        (rows of the block, in instance order) and ``segment`` (a `rowk` form's), in `PUBLIC_INPUTS_FORMAT`: a header line
        ``{format, instances, ports}``, each port with its words, seed and derivation (`public_ports`), then every unit's
        regions (`regions`), u16 LE, instance-major, the ports in order.  A region the units share (F_A, F_B, G, or a
        segment's slices of F_A and F_B) is derived once and repeated in each."""
        names = public_ports(self.shape, role)
        if not names:
            raise DomainError(f"a {role} unit has no public input region")
        if not indexes:
            raise DomainError("a --public-inputs file holds at least one instance")
        shared = {p: self._region(role, p, None, segment) for p, a in self.shape.ports[role]["regions"].items()
                  if not self.shape.derivations[a].per_row}
        units = [{p: shared[p] if p in shared else self._region(role, p, i) for p in names} for i in indexes]
        ports = [{"port": p, "words": len(units[0][p]), **names[p]} for p in names]
        head = {"format": PUBLIC_INPUTS_FORMAT, "instances": len(units), "ports": ports}
        return json.dumps(head, sort_keys=True).encode() + b"\n" + b"".join(region_bytes(u[p]) for u in units for p in names)

    def check_public_inputs(self, role: str, indexes: Sequence[int], data: bytes, segment: int | None = None) -> None:
        """Refuse a `--public-inputs` file that is not the verifier's own (`public_inputs`), as Flock's Lean verifier does
        (`HmIn.checkOwn`): its header must be the verifier's, and every word its recomputation, each region in the unit's
        parameter leaf order and the instances in ``indexes``' order.  So a region in another order (Flock's read order
        of a transposed operand) or two instances swapped is refused, at the first word that differs."""
        own_head, own_body = self.public_inputs(role, indexes, segment).split(b"\n", 1)
        head, sep, body = data.partition(b"\n")
        try:
            same = sep == b"\n" and json.loads(head) == json.loads(own_head)
        except (UnicodeDecodeError, json.JSONDecodeError):
            same = False
        if not same:
            raise DomainError("--public-inputs: its header (format, instances, ports) is not the verifier's")
        if len(body) != len(own_body):
            raise DomainError("--public-inputs: length")
        if body == own_body:
            return
        ports = json.loads(own_head)["ports"]
        w = next(j for j in range(0, len(body), 2) if body[j:j + 2] != own_body[j:j + 2]) // 2
        instance, off = divmod(w, sum(p["words"] for p in ports))
        for p in ports:
            if off < p["words"]:
                raise DomainError(f"public input {p['port']}: word {off} of instance {instance} is not the verifier's own "
                                  f"recomputation (seed {p['seed']}, derivation {p['derivation']})")
            off -= p["words"]


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
        from verity.ml.randomness import derive

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


class SaltWindows:
    """Every window of one salt, opened in turn: a call index names one call across all of them, as within one window.

    A call's noise key reads the salt and its call index, so a call index that two windows of a salt both use would
    reuse one key. `open` refuses a receipt that lists a call of another salt or a call index an earlier window of this
    salt listed, and only then opens the window's ledger (`Ledger.open`). `next_forward` is the first forward index no
    window of the salt has used, where the verifier's enumeration of the next window's forwards starts."""

    __slots__ = ("salt", "_seen", "_next")

    def __init__(self, salt: bytes) -> None:
        self.salt, self._seen, self._next = bytes(salt), set(), 0

    @property
    def next_forward(self) -> int:
        return self._next

    def open(self, receipt: Receipt) -> Ledger:
        if not isinstance(receipt, Receipt):
            raise TypeError("a window opens from its complete Receipt")
        indices = [a.call_index for a, _, _ in receipt.calls]
        if any(a.salt != self.salt for a, _, _ in receipt.calls):
            raise DomainError("the window's receipt lists a call under another salt")
        repeated = sorted((i for i in indices if i in self._seen), key=_index_parts)
        if repeated:
            raise DomainError(f"call index (forward, position) = {repeated[0]} is on an earlier window of this salt")
        ledger = Ledger.open(receipt)
        self._seen.update(indices)
        self._next = max(self._next, 1 + max(f for f, _ in indices))
        return ledger


def replay_key(record: dict[str, str]):
    """The window's key from its audit record: `derive(source, DRAW_DOMAIN, {"receipt": digest})`."""
    from verity.ml.randomness import derive

    from .plan import DRAW_DOMAIN

    if record.get("domain") != DRAW_DOMAIN:
        raise DomainError(f"the record's draw domain is not {DRAW_DOMAIN!r}")
    return derive(bytes.fromhex(record["source"]), DRAW_DOMAIN, {"receipt": bytes.fromhex(record["receipt"])})


__all__ = ["ANCHORED", "ANCHOR_RULE", "Anchors", "Derivation", "Ledger", "PUBLIC_INPUTS_FORMAT", "REGION_WORD_BITS", "Receipt",
           "SaltWindows", "TileAnchors", "WORD_BITS", "public_ports", "region_bytes", "region_words", "replay_key"]
