"""What the network certifier's ties share: the id form's cells, the verifier's registered reads of a Commit's rows, and a
Beneš network. The ingress tie (`io_request`, with `io_ids` and `served_tie`) and the egress tie (`egress`, the egress
side of `scan_service.EgressTie`) build on them.

**The cells.** A `DATA` cell is the network certifier's cut of the server's bytes, P = B − 9 of them, and a response in
the id form is its events, each `id_event` padded with spaces to P, then `TRAILER`, shorter than P. So the network
certifier's cut (`warden.Warden._egress`) is exactly n `DATA` cells (`data_cell`), one event each, and an `END` that
holds `TRAILER` (`end_cell`), whenever the bytes are there by their deadlines. A slot no connection holds is B zero
bytes (`pad_cell(B)`, a `PAD` on stream 0), and a connection's slot before its first output a `PAD` on its stream
(`pad_cell(B, q)`). `parse_cell` is a cell's one reading. A slot's occupancy marks a `PAD` with `EMPTY` and the
`END` with `END_ENTRY`.

**The registered reads.** A Commit's rows are one hiding row a step (`port_row`: the step's P ids, u32be in batch-row
order, zero past its batch) in a dense frame-v3-sha512 tree of the session's steps (`domain`, `rows_levels`), whose
root the Commit registers, `hm96-sha512/row/v2` leaves. `port_reads` gives the verifier's own `--registered` ports of
a population that reads S rows each at public steps, and `open_port` the prover's openings of them.

**The routing.**

    IoTieSwitch{W}(a: Array<W>, b: Array<W>, s) -> Array<2, Array<W>>    (a, b), swapped iff s: W ANDs
    IoTieRoute{W, D}(xs: Array<M, Array<W>>, switches: Array<S_W>) -> Array<M, Array<W>>
        a Beneš network of S_W = (2D − 1)·2^(D − 1) switches: a column of 2^(D − 1) `IoTieSwitch`es, two
        `IoTieRoute{W, D − 1}`, a column, so a binding is built once whatever its uses

`benes` sets the switches that route a permutation, and `benes_apply` is the network in the clear.
"""
from __future__ import annotations

import hashlib
from typing import Any, NamedTuple, Sequence

from verity.ml.circuits.boolean.forms import ZERO, and_reduce
from verity.ml.circuits.defs import CompositeDefinition, bind
from verity.ml.circuits.types import Array
from verity.ml.commitments import hm96
from verity.ml.commitments.gates.frame_v3 import _part, leaf_ref, node_ref
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import _PAD, CommitmentDomain, frame_hash, _uint
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier.active import DATA, EVENT_END, EVENT_HEAD, HEADER, PAD
from verity.core.protocols.spot.one_stage import produced
from verity.core.protocols.spot.one_stage import registered as G

BIT = HC.BIT
ID_DIGITS = 10
ID_EVENT_HEAD = EVENT_HEAD[:EVENT_HEAD.index(b'"delta"')] + b'"token_id":'
ID_EVENT_TAIL = b',"finish_reason":null}]}' + EVENT_END
ID_EVENT_BYTES = len(ID_EVENT_HEAD) + ID_DIGITS + len(ID_EVENT_TAIL)
#: the id's first character in a cell: the header, then the event's head
ID_AT = HEADER.size + len(ID_EVENT_HEAD)
#: the least cell: the header and one event, which a `DATA` cell holds with no padding
CELL_USED = HEADER.size + ID_EVENT_BYTES
#: the network certifier's `END` kind (`warden.END`), beside `active`'s `PAD` and `DATA`
END = 0x20
#: a response's last bytes, after its events: the network certifier's `END` holds them
TRAILER = b"data: [DONE]" + EVENT_END
#: a slot's occupancy past the outputs: `EMPTY` for a `PAD` on its stream, `END_ENTRY` for its `END`
EMPTY = 0xFFFF_FFFF
END_ENTRY = EMPTY - 1
#: the proof block's port parameter: S whole rows of the Commit's port, row s a registered read at its own key `port[s]`
PORT = "port"
#: the registered value the egress port reads, the Commit's sampled ids
SAMPLED = "sampled_token_ids"


# --- the cells -------------------------------------------------------------------------------------------------------

def id_event(token_id: int, size: int = ID_EVENT_BYTES) -> bytes:
    """The canonical event of one token id in `size` bytes: `ID_EVENT_HEAD`, the id right-aligned in 10 decimal
    characters, then `ID_EVENT_TAIL` with spaces (JSON whitespace) ahead of its blank line to fill `size`."""
    if type(token_id) is not int or not 0 <= token_id < 1 << 32:
        raise ValueError(f"a token id is a u32, not {token_id!r}")
    if type(size) is not int or size < ID_EVENT_BYTES:
        raise ValueError(f"an id event is at least {ID_EVENT_BYTES} bytes, not {size!r}")
    tail = ID_EVENT_TAIL[:-len(EVENT_END)] + b" " * (size - ID_EVENT_BYTES) + EVENT_END
    return ID_EVENT_HEAD + b"%10d" % token_id + tail


def _check_cell_bytes(cell_bytes: object) -> None:
    if type(cell_bytes) is not int or cell_bytes < CELL_USED:
        raise ValueError(f"a cell has at least {CELL_USED} bytes, room for an id event, not {cell_bytes!r}")


def _check_stream(stream: object) -> None:
    if type(stream) is not int or not 0 <= stream < 1 << 32:
        raise ValueError(f"a stream is a u32, not {stream!r}")


def data_cell(token_id: int, stream: int, cell_bytes: int) -> bytes:
    """A `DATA` cell on the request's stream: the network certifier's P = B − 9 bytes of the response, its id's
    event."""
    _check_cell_bytes(cell_bytes)
    _check_stream(stream)
    p = cell_bytes - HEADER.size
    return HEADER.pack(DATA, stream, p) + id_event(token_id, p)


def pad_cell(cell_bytes: int, stream: int = 0) -> bytes:
    """A `PAD` on the stream, no payload, zero fill: B zero bytes on stream 0, a slot no connection holds."""
    _check_cell_bytes(cell_bytes)
    _check_stream(stream)
    return HEADER.pack(PAD, stream, 0).ljust(cell_bytes, b"\0")


def end_cell(stream: int, cell_bytes: int) -> bytes:
    """A response's `END` on its stream: the rest of its bytes, `TRAILER`, and zero fill."""
    _check_cell_bytes(cell_bytes)
    _check_stream(stream)
    return (HEADER.pack(END, stream, len(TRAILER)) + TRAILER).ljust(cell_bytes, b"\0")


class Parsed(NamedTuple):
    """A cell's reading: its kind (`PAD`, `DATA` or `END`), its stream, and a `DATA` cell's id (0 otherwise)."""

    kind: int
    stream: int
    id: int

    @property
    def data(self) -> bool:
        return self.kind == DATA


def parse_cell(cell: bytes) -> Parsed | None:
    """The cell's one reading, or None for a cell that is none of the three."""
    if len(cell) < CELL_USED:
        return None
    kind, stream, _ = HEADER.unpack_from(cell)
    if kind in (PAD, END):
        return Parsed(kind, stream, 0) if cell == (pad_cell(len(cell), stream) if kind == PAD else
                                                   end_cell(stream, len(cell))) else None
    digits = cell[ID_AT:ID_AT + ID_DIGITS]
    if kind != DATA or not digits.lstrip(b" ").isdigit():
        return None
    token_id = int(digits)
    return Parsed(DATA, stream, token_id) if token_id < 1 << 32 and cell == data_cell(token_id, stream, len(cell)) else None


# --- the registered reads --------------------------------------------------------------------------------------------

def route_depth(cells: int, entries: int) -> int:
    """D of the routing over M = 2^D ≥ max(cells, entries) positions."""
    return max(1, (max(cells, entries) - 1).bit_length())


def port_row(ids: Sequence[int], P: int) -> bytes:
    """A step's port row: its batch rows' sampled ids, u32be in batch order, zero past them to P."""
    if len(ids) > P or any(type(x) is not int or not 0 <= x < 1 << 32 for x in ids):
        raise ValueError(f"a port row holds at most {P} u32 ids, not {list(ids)!r}")
    return b"".join(x.to_bytes(4, "big") for x in ids).ljust(4 * P, b"\0")


def domain(binding: bytes, count: int) -> CommitmentDomain:
    """A dense frame-v3-sha512 domain of `count` rows, as the network certifier's per-window cells and the port use."""
    return CommitmentDomain(binding, -1, RangeIndexedDomain(0, count), hash="sha512")


def rows_levels(domain_id: bytes, rows: Sequence[bytes], salts: Sequence[bytes]) -> list[list[bytes]]:
    """The dense frame-v3-sha512 tree, under `domain_id`, of the rows' `hm96-sha512/row/v2` leaves, each under its salt,
    and the domain's pads to a power of two, level by level from the leaves (`tree_levels` over `domain`'s leaves)."""
    if len(rows) != len(salts) or len(rows) < 2:
        raise ValueError(f"a rows tree is over at least 2 rows, each with its salt, not {len(rows)} rows and "
                         f"{len(salts)} salts")
    level = [leaf_ref(domain_id, SCHEMA, HC.hiding_value(r, s), i) for i, (r, s) in enumerate(zip(rows, salts))]
    width = 1 << (len(rows) - 1).bit_length()
    level += [frame_hash(_PAD, domain_id, _uint(i), hash="sha512") for i in range(len(rows), width)]
    levels = [level]
    while len(level) > 1:
        level = [node_ref(domain_id, len(levels) - 1, t, level[2 * t], level[2 * t + 1]) for t in range(len(level) // 2)]
        levels.append(level)
    return levels


def rows_path(levels: Sequence[Sequence[bytes]], pos: int) -> tuple[bytes, ...]:
    """Row `pos`'s siblings from the leaves up (`FrameV3Read`'s path)."""
    return tuple(levels[k][(pos >> k) ^ 1] for k in range(len(levels) - 1))


def fake_salt(seed: bytes, i: int) -> bytes:
    """A deterministic hiding salt for a fake upstream's tests (a prover samples fresh ones)."""
    out = b""
    while len(out) < hm96.SHA512.salt_bytes:
        out += hashlib.sha512(seed + i.to_bytes(8, "big") + len(out).to_bytes(4, "big")).digest()
    return out[:hm96.SHA512.salt_bytes]


def port_reads(steps: Sequence[Sequence[int]], root: bytes, domain_id: bytes, count: int, P: int,
               value: str = SAMPLED, reads_from: int = 0) -> dict[str, dict[str, Any]]:
    """The verifier's own `--registered` ports (`warden_rows.reads_file`'s `ports`) of a population of proof blocks that
    each read S rows of the Commit's port, its `count` rows of P ids under `root` in the domain `domain_id`: `port[s]`
    (`port` when S = 1, as `registered.reads` keys a parameter of one row), the proof block's s-th port row, has proof
    block `reads_from + u` read row steps[u][s]. The steps are the verifier's map's, so every read is at a public
    position, and the Lean verifier holds each read row to its leaf of the Commit's root by its b ‖ c
    (`registered.verify_read`), as it holds a bucket row to the network certifier's (`warden_rows.bucket_port`)."""
    widths = {len(st) for st in steps}
    if len(widths) != 1 or 0 in widths:
        raise ValueError("every unit reads the same S >= 1 port rows")
    if any(type(x) is not int or not 0 <= x < count for st in steps for x in st):
        raise ValueError(f"a read is a step of the Commit's {count} rows")
    S = widths.pop()
    e = {"leaves": count, "words": 2 * P, "word_bits": G.WORD_BITS, "domain": bytes(domain_id).hex()}
    extra = {G.READS_FROM: int(reads_from)} if reads_from else {}
    return {(PORT if S == 1 else f"{PORT}[{s}]"): produced.port(value, e, bytes(root).hex(), [st[s] for st in steps],
                                                                SCHEMA) | extra for s in range(S)}


# --- the prover's openings and routing -------------------------------------------------------------------------------

def open_port(levels: Sequence[Sequence[bytes]], rows: Sequence[bytes], salts: Sequence[bytes],
              steps: Sequence[int]) -> tuple[tuple[bytes, ...], tuple[bytes, ...], tuple[tuple[bytes, ...], ...]]:
    """A bucket's port openings from the Commit's rows, salts and tree: each step's row, salt and path."""
    return (tuple(rows[s] for s in steps), tuple(salts[s] for s in steps), tuple(rows_path(levels, s) for s in steps))


def benes(perm: Sequence[int]) -> list[int]:
    """The switch bits of a Beneš network that takes input i to output perm[i] (the looping algorithm), in
    `IoTieRoute`'s order: the input column, the upper half's network, the lower half's, the output column."""
    n = len(perm)
    if n < 2 or n & (n - 1) or sorted(perm) != list(range(n)):
        raise ValueError("a Beneš network routes a permutation of a power of two at least 2")
    if n == 2:
        return [perm[0]]
    inv = [0] * n
    for i, d in enumerate(perm):
        inv[d] = i
    color: list[int | None] = [None] * n
    for start in range(0, n, 2):
        i = start
        # i goes up and its switch partner down; the input sharing that partner's output switch goes up next
        while color[i] is None:
            color[i], color[i ^ 1] = 0, 1
            i = inv[perm[i ^ 1] ^ 1]
    upper, lower, out = [0] * (n // 2), [0] * (n // 2), [0] * (n // 2)
    for i in range(n):
        (upper if color[i] == 0 else lower)[i // 2] = perm[i] // 2
        if color[i] == 0:
            out[perm[i] // 2] = perm[i] & 1
    return [color[2 * k] for k in range(n // 2)] + benes(upper) + benes(lower) + out  # type: ignore[list-item]


def benes_apply(xs: Sequence, bits: Sequence[int]) -> list:
    """`IoTieRoute` in the clear: the network's outputs for its inputs under the switch bits."""
    it = iter(bits)

    def run(v: list) -> list:
        if len(v) == 2:
            return v[::-1] if next(it) else v
        up, down = [], []
        for k in range(len(v) // 2):
            a, b = (v[2 * k + 1], v[2 * k]) if next(it) else (v[2 * k], v[2 * k + 1])
            up.append(a)
            down.append(b)
        up, down = run(up), run(down)
        res = []
        for a, b in zip(up, down):
            res += [b, a] if next(it) else [a, b]
        return res

    out = run(list(xs))
    if next(it, None) is not None:
        raise ValueError("more switch bits than the network has")
    return out


# --- the circuit -----------------------------------------------------------------------------------------------------

def _add(C, x: Sequence, y: Sequence) -> list:
    """x + y, both as wide as x, the carry out dropped: a ripple adder, 2 ANDs a bit."""
    out, c = [], ZERO
    for a, b in zip(x, y, strict=True):
        t = a ^ b
        out.append(t ^ c)
        c = C.AND(a, b) ^ C.AND(c, t)
    return out


def _digits(C, chars: Sequence[Sequence]) -> tuple[list, list]:
    """The id field's checks (spaces then digits, a nonzero first digit unless it is the last, below 2^32) and its
    value's 32 bits, least significant first."""
    space = [and_reduce(C, [HC._same(c[b], 0x20 >> b & 1) for b in range(8)]) for c in chars]
    # a digit is 0x30 ... 0x39: high nibble 0011, low nibble at most 9 (not b3 with b2 or b1)
    digit = [C.AND(and_reduce(C, [c[7].inv(), c[6].inv(), c[5], c[4]]), C.AND(c[3], C.AND(c[2].inv(), c[1].inv()).inv()).inv())
             for c in chars]
    zero = [and_reduce(C, [c[b].inv() for b in range(4)]) for c in chars]
    oks = []
    for i in range(ID_DIGITS):
        oks.append(space[i] ^ digit[i])
        if i:
            oks.append(C.AND(digit[i - 1], space[i]).inv())
        if i < ID_DIGITS - 1:
            lead = digit[i] if i == 0 else C.AND(digit[i], space[i - 1])
            oks.append(C.AND(lead, zero[i]).inv())
    oks.append(digit[-1])
    width = 34
    v = [ZERO] * width
    for c in chars:
        # v·10 + the character's low nibble (a space's is 0)
        v = _add(C, _add(C, [ZERO] * 3 + v[:-3], [ZERO] + v[:-1]), [*c[:4], *[ZERO] * (width - 4)])
    return oks + [v[32].inv(), v[33].inv()], v[:32]


def _check_route(S) -> None:
    if type(S.D) is not int or not 1 <= S.D <= 32 or type(S.W) is not int or S.W < 1:
        raise ValueError(f"a Beneš network has 1 <= D <= 32 and W >= 1, not D={S.D!r}, W={S.W!r}")


def _sig_route(S):
    _check_route(S)
    return ((("xs", Array(1 << S.D, Array(S.W, BIT))), ("switches", Array((2 * S.D - 1) << (S.D - 1), BIT))),
            Array(1 << S.D, Array(S.W, BIT)))


def _sig_switch(S):
    if type(S.W) is not int or S.W < 1:
        raise ValueError(f"a switch has W >= 1, not W={S.W!r}")
    return (("a", Array(S.W, BIT)), ("b", Array(S.W, BIT)), ("s", BIT)), Array(2, Array(S.W, BIT))


def _build_switch(C, S, a, b, s):
    t = [C.AND(s[0], x ^ y) for x, y in zip(a, b)]
    return [[x ^ u for x, u in zip(a, t)], [y ^ u for y, u in zip(b, t)]]


IoTieSwitch = _part(
    "IoTieSwitch", ("W",), _sig_switch, _build_switch,
    "(a, b), swapped iff s: W ANDs")


def _body_route(B, S, xs, switches_):
    W, h = S.W, 1 << (S.D - 1)
    sw, half = HC._refs(switches_), Array(h, Array(W, BIT))

    def words(refs: list) -> list:
        return [refs[W * k:W * (k + 1)] for k in range(len(refs) // W)]

    def column(pairs: list, bits: list) -> list:
        """The h switches' (a, b) pairs, swapped by their bits: one batch of `IoTieSwitch`."""
        out = words(HC._refs(B.batch(bind(IoTieSwitch, W=W), HC._coll(half, [r for a, _ in pairs for r in a]),
                                     HC._coll(half, [r for _, b in pairs for r in b]), HC._coll(Array(h, BIT), bits),
                                     axes=(0, 0, 0))))
        return list(zip(out[0::2], out[1::2]))

    x = words(HC._refs(xs))
    if S.D == 1:
        last = column([(x[0], x[1])], sw)
    else:
        # `benes`' bit order: the input column, the upper half's, the lower half's, the output column
        sub = (2 * S.D - 3) << (S.D - 2)
        first = column([(x[2 * k], x[2 * k + 1]) for k in range(h)], sw[:h])
        net = bind(IoTieRoute, W=W, D=S.D - 1)
        up, down = (words(HC._refs(B.call(net, HC._coll(half, [r for p in first for r in p[i]]),
                                          HC._coll(Array(sub, BIT), sw[h + i * sub:h + (i + 1) * sub]))))
                    for i in (0, 1))
        last = column(list(zip(up, down)), sw[h + 2 * sub:])
    return HC._coll(Array(2 * h, Array(W, BIT)), [r for p in last for w in p for r in w])


IoTieRoute = CompositeDefinition(
    "IoTieRoute", 1, ("W", "D"), _sig_route, _body_route, conformance="exact-model-tested",
    doc="a Beneš network on 2^D W-bit inputs: the input column's 2^(D − 1) switches (`IoTieSwitch`: s swaps "
        "inputs 2k and 2k + 1 into the halves), the upper half's `IoTieRoute{W, D − 1}`, the lower half's, the "
        "output column's switches; W ANDs a switch")


def _eq(C, x: Sequence, y: Sequence):
    return and_reduce(C, [(a ^ b).inv() for a, b in zip(x, y, strict=True)])


def _lt(C, x: Sequence, y: Sequence):
    """x < y, unsigned, least significant bit first: 1 AND a bit."""
    lt = ZERO
    for a, b in zip(x, y, strict=True):
        lt = lt ^ C.AND(a ^ b, lt ^ b)
    return lt


__all__ = ["CELL_USED", "EMPTY", "END", "END_ENTRY", "ID_AT", "ID_EVENT_BYTES", "IoTieRoute", "IoTieSwitch", "PORT",
           "Parsed", "SAMPLED", "TRAILER", "benes", "benes_apply", "data_cell", "domain", "end_cell", "fake_salt",
           "id_event", "open_port", "pad_cell", "parse_cell", "port_reads", "port_row", "route_depth", "rows_levels",
           "rows_path"]
