"""The replay's state chain: vLLM's scheduler state handed from one step unit to the next under hiding commitments
(PROTOCOL.md, "The replay's state chain").

The compile unit replays the scheduler from advice v1 and the certified record, one step unit per engine step, each
`replay_step.ReplayStep_v1` (ENC-REPLAY's step): the committed queue state, the next A certified arrivals and the next A
decisions in, the step's rows and the next state out. A window is many steps, so the state passes between units as
rows of a produced value, each unit reading its own row and the next at public positions. This module is that chain:
the row (`Shape`, `Row`, `encode`, `decode`), its commitment and reads (`commit_window`, `reads`), the verifier's check
of how the windows join (`check_chain`), and the two Boolean parts it puts in the units, `ReplayAdvance` beside each
step and `ReplayLoad` once a window, with `advance` and `load` their specifications.

**Why the row carries the arrivals and the decisions.** The step reads them from the scan's cursors, which move by
its `taken` and `used`, and the cursors are private: a step's position in the certified arrivals or the window's
decisions is the schedule itself. A registered read is at a position the verifier derives, so no unit can read the
arrival record at the cursor. The row holds instead the certified arrivals not yet taken (`queue`, at most Q) and the
window's decisions not yet read (`decisions`, at most D), front first, and the step reads their first A. Each step
shifts them by what it took, and once a window `ReplayLoad` appends the window's certified arrivals (K slots, the
warden's) and its decoded decisions, which it reads at public positions. So every read in the chain is at a public
position, and no gate hashes a row: the Lean verifier checks each read's path (`registered.verify_read`).

**The row** (`row_fields`), `Shape.width` bits, each field's elements in turn and each element least significant bit
first, then zero bits to a whole 16-bit word:

    state    run_valid R×1 | run_id R×IW | run_P R×LW | run_O R×LW | run_out R×LW | run_comp R×LW
             | wait_valid W×1 | wait_id W×IW | wait_P W×LW | wait_O W×LW | wait_out W×LW | free 1×BW
    queue    q_valid Q×1 | q_id Q×IW | q_P Q×LW | q_O Q×LW
    window   dec D×1 | left 1×DW | steps 1×PW

- `state` is ReplayStep_v1's committed state as its `_in_fields` lay it out, and its next state as its `_out_fields`
  do, bit for bit: R running slots (id, prompt P, certified output length O, outputs so far, tokens computed), W
  waiting slots (a waiting request has computed nothing) and the free block count. A running request holds
  ⌈computed / B⌉ blocks, so the KV pool is the free count, with no count per request.
- `queue` is the certified arrivals not yet taken, in certified order; the step's `arr_*` are its first A.
- `dec` is the window's decisions not yet read, the step's `bits` its first A; `left` counts them and `steps` counts the
  window's executed steps still to come.
- Widths, from the size class and the public config (`widths`): LW = bitlen(L), BW = bitlen(NB), AC = bitlen(A),
  QW = bitlen(Q), DW = bitlen(D), PW = bitlen(P), bitlen(0) being 1; IW is a request id's.
- In every row the valid elements of each group are a prefix and the rest are zero bits, as are `dec`'s bits past
  `left`. `decode` refuses any other row. The circuit needs no such check: the first row is the empty state, and every
  later row is a proved unit's output.

**A window's rows** S_0 … S_{P+1} are one produced value (`one_stage.produced`), `VALUE`, in `hm96-sha512/row/v2` bit
rows; P, the step units a window, is a public cap. Unit 0 is the load: it reads `prev` row 0 and `next` row 1 with
the window's arrivals, decisions and step count n ≤ P. Unit u ≥ 1 is step u: ReplayStep_v1 on row u's state, queue and
decisions, with `ReplayAdvance` checking row u + 1, so a row two units read is the one row `produced.shared` holds them
to. Steps past the window's n are idle (`steps` = 0): the next row is the row, and the unit's rows are dropped
(`ReplayAdvance`'s `active` = 0).

- **The carry.** Window w's row 0 is window w − 1's row P + 1, committed under the same salt (`commit_window(carry=…)`),
  so its opening b‖c is the same, and the verifier checks they're equal (`check_chain`). The load refuses a row 0 that
  left decisions or steps unread, so each window reads exactly its own.
- **The start.** The session's first row is the empty state (`empty`: nothing held, NB − 1 free blocks, vLLM's block 0
  being its null block) under the public salt `empty_salt`, so the verifier computes its opening (`empty_opening`).

**Cost** (`size`). Per step: `ReplayAdvance`'s ANDs (the shifts of the queue and the decisions, about 2·width for the
idle select and the equality), two reads and one row. Per window: P steps, one `ReplayLoad`, P + 2 rows and 2(P + 1)
reads, each a frame-v3 path of depth ⌈log2(P + 2)⌉ for the verifier.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import ZERO, Circuit, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.types import Array, Tuple
from verity.ml.commitments import hm96
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.core.protocols.certifier.head_check import BIT, _part, ands
from verity.core.protocols.sampled_proofs.one_stage import produced
from verity.core.protocols.sampled_proofs.one_stage import registered as G

VALUE = "replay-state"
WORD_BITS = 16
STATICS = ("R", "W", "A", "L", "NB", "IW", "Q", "D", "P", "K")
_INST = hm96.SHA512


class MalformedRow(ValueError):
    """A row that isn't a state's encoding."""


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


@dataclass(frozen=True)
class Shape:
    """The public size class: ReplayStep_v1's R running slots, W waiting slots and A arrivals a step, the engine's
    max_model_len L and num_gpu_blocks NB, a request id's width IW; the chain's Q pending arrivals, D decisions and P
    steps a window, and the warden's K arrival slots a window."""

    R: int
    W: int
    A: int
    L: int
    NB: int
    IW: int
    Q: int
    D: int
    P: int
    K: int

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"{f.name} is a positive integer, not {v!r}")

    def statics(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in STATICS}

    @property
    def width(self) -> int:
        """A row's bits, padded to whole 16-bit words."""
        return width(self)

    @property
    def words(self) -> int:
        return self.width // WORD_BITS


def widths(S) -> SimpleNamespace:
    return SimpleNamespace(LW=_bitlen(S.L), BW=_bitlen(S.NB), AC=_bitlen(S.A), QW=_bitlen(S.Q), DW=_bitlen(S.D),
                           PW=_bitlen(S.P))


def state_fields(S) -> list[tuple[str, int, int]]:
    """ReplayStep_v1's state, its `_in_fields`' and `_out_fields`' order and widths."""
    Z, R, W = widths(S), S.R, S.W
    return [("run_valid", R, 1), ("run_id", R, S.IW), ("run_P", R, Z.LW), ("run_O", R, Z.LW), ("run_out", R, Z.LW),
            ("run_comp", R, Z.LW), ("wait_valid", W, 1), ("wait_id", W, S.IW), ("wait_P", W, Z.LW), ("wait_O", W, Z.LW),
            ("wait_out", W, Z.LW), ("free", 1, Z.BW)]


def queue_fields(S) -> list[tuple[str, int, int]]:
    Z = widths(S)
    return [("q_valid", S.Q, 1), ("q_id", S.Q, S.IW), ("q_P", S.Q, Z.LW), ("q_O", S.Q, Z.LW)]


def arrival_fields(S) -> list[tuple[str, int, int]]:
    """`ReplayLoad`'s `arrivals`: the window's K certified arrival slots, a valid prefix."""
    Z = widths(S)
    return [("a_valid", S.K, 1), ("a_id", S.K, S.IW), ("a_P", S.K, Z.LW), ("a_O", S.K, Z.LW)]


def row_fields(S) -> list[tuple[str, int, int]]:
    Z = widths(S)
    return state_fields(S) + queue_fields(S) + [("dec", S.D, 1), ("left", 1, Z.DW), ("steps", 1, Z.PW)]


def _bits_of(fs) -> int:
    return sum(n * w for _, n, w in fs)


def width(S) -> int:
    return WORD_BITS * -(-_bits_of(row_fields(S)) // WORD_BITS)


# --- the row ----------------------------------------------------------------------------------------------------------------

Running = tuple[int, int, int, int, int]   # (id, P, O, out, computed)
Waiting = tuple[int, int, int, int]        # (id, P, O, out)
Arrival = tuple[int, int, int]             # (id, P, O)


@dataclass(frozen=True)
class Row:
    """The scheduler and the window between two units (the module's docstring, "The row")."""

    running: tuple[Running, ...] = ()
    waiting: tuple[Waiting, ...] = ()
    free: int = 0
    queue: tuple[Arrival, ...] = ()
    decisions: tuple[int, ...] = ()
    steps: int = 0


def empty(S) -> Row:
    """The session's first row: the empty engine, its NB − 1 free blocks."""
    return Row(free=S.NB - 1)


def _put(out: list[int], v: int, w: int, name: str) -> None:
    v = int(v)
    if not 0 <= v < 1 << w:
        raise MalformedRow(f"{name} = {v} doesn't fit {w} bits")
    out += [(v >> j) & 1 for j in range(w)]


def _values(row: Row, S) -> dict[str, list[int]]:
    if len(row.running) > S.R or len(row.waiting) > S.W or len(row.queue) > S.Q or len(row.decisions) > S.D:
        raise MalformedRow("the row exceeds the size class")
    if row.steps > S.P or any(b not in (0, 1) for b in row.decisions):
        raise MalformedRow(f"steps {row.steps} of at most {S.P}; decisions are bits")
    vals: dict[str, list[int]] = {}

    def group(prefix: str, names: Sequence[str], items: Sequence[Sequence[int]], n: int) -> None:
        vals[f"{prefix}_valid"] = [1] * len(items) + [0] * (n - len(items))
        for k, name in enumerate(names):
            vals[f"{prefix}_{name}"] = [x[k] for x in items] + [0] * (n - len(items))

    group("run", ("id", "P", "O", "out", "comp"), row.running, S.R)
    group("wait", ("id", "P", "O", "out"), row.waiting, S.W)
    group("q", ("id", "P", "O"), row.queue, S.Q)
    vals["free"] = [row.free]
    vals["dec"] = list(row.decisions) + [0] * (S.D - len(row.decisions))
    vals["left"] = [len(row.decisions)]
    vals["steps"] = [row.steps]
    return vals


def _flatten(vals: dict[str, list[int]], fs) -> list[int]:
    out: list[int] = []
    for name, _n, w in fs:
        for e, v in enumerate(vals[name]):
            _put(out, v, w, f"{name}[{e}]")
    return out


def encode(row: Row, S) -> tuple[int, ...]:
    """The row's `width(S)` bits."""
    bits = _flatten(_values(row, S), row_fields(S))
    return tuple(bits + [0] * (width(S) - len(bits)))


def state_bits(row: Row, S) -> tuple[int, ...]:
    """The row's `state` part: ReplayStep_v1's state input, or its next-state output, as bits."""
    return tuple(_flatten(_values(row, S), state_fields(S)))


def _unflatten(bits: Sequence[int], fs) -> tuple[dict[str, list[int]], int]:
    at, f = 0, {}
    for name, n, w in fs:
        f[name] = [sum(int(bits[at + e * w + j]) << j for j in range(w)) for e in range(n)]
        at += n * w
    return f, at


def decode(bits: Sequence[int], S) -> Row:
    """The row `bits` encode; `MalformedRow` for any other bits (the module's docstring, "The row")."""
    if len(bits) != width(S) or any(b not in (0, 1) for b in bits):
        raise MalformedRow(f"a row is {width(S)} bits")
    f, at = _unflatten(bits, row_fields(S))
    if any(bits[at:]):
        raise MalformedRow("nonzero padding")

    def group(prefix: str, names: Sequence[str]) -> tuple[tuple[int, ...], ...]:
        valid = f[f"{prefix}_valid"]
        k = sum(valid)
        if valid != [1] * k + [0] * (len(valid) - k):
            raise MalformedRow(f"{prefix}: the valid slots aren't a prefix")
        if any(f[f"{prefix}_{name}"][e] for name in names for e in range(k, len(valid))):
            raise MalformedRow(f"{prefix}: an empty slot has a bit set")
        return tuple(tuple(f[f"{prefix}_{name}"][e] for name in names) for e in range(k))

    left = f["left"][0]
    if left > S.D or any(f["dec"][left:]) or f["steps"][0] > S.P:
        raise MalformedRow(f"left {left} of {S.D} with no decision past it, steps {f['steps'][0]} of {S.P}")
    return Row(group("run", ("id", "P", "O", "out", "comp")), group("wait", ("id", "P", "O", "out")), f["free"][0],
               group("q", ("id", "P", "O")), tuple(f["dec"][:left]), f["steps"][0])


def row(bits: Sequence[int]) -> tuple[int, ...]:
    """A row's bits as the produced row's u16 words, bit i of the row bit i % 16 of word i // 16."""
    if len(bits) % WORD_BITS:
        raise ValueError("a row is whole 16-bit words")
    return tuple(sum(int(bits[w + i]) << i for i in range(WORD_BITS)) for w in range(0, len(bits), WORD_BITS))


def unrow(words: Sequence[int]) -> tuple[int, ...]:
    return tuple((int(w) >> i) & 1 for w in words for i in range(WORD_BITS))


# --- the specification of the units -----------------------------------------------------------------------------------------

def advance(S, cur: Row, ok: bool, stepped: Row, taken: int, used: int) -> Row | None:
    """The row after step unit u: `cur` when the window's steps are done (idle), else the step's next state (`stepped`'s
    running, waiting and free), the queue and the decisions past what it took and read, one step fewer; None refuses
    (the step's ok is 0, it read more decisions than are left, or the window's last step leaves some)."""
    if not cur.steps:
        return cur
    if not ok or used > len(cur.decisions) or (cur.steps == 1 and used != len(cur.decisions)):
        return None
    return Row(stepped.running, stepped.waiting, stepped.free, cur.queue[taken:], cur.decisions[used:], cur.steps - 1)


def load(S, prev: Row, arrivals: Sequence[Arrival], decisions: Sequence[int], steps: int) -> Row | None:
    """The window's row 1: row 0 (`prev`, the last window's last) with the window's certified arrivals appended to the
    queue and its decisions and executed step count loaded; None refuses (the last window left decisions or steps, the
    queue overflows Q, more than D decisions or P steps, or decisions with no step to read them)."""
    if prev.decisions or prev.steps or len(prev.queue) + len(arrivals) > S.Q or len(arrivals) > S.K:
        return None
    if len(decisions) > S.D or steps > S.P or (not steps and decisions):
        return None
    return Row(prev.running, prev.waiting, prev.free, prev.queue + tuple(tuple(a) for a in arrivals), tuple(decisions),
               steps)


# --- the chain ------------------------------------------------------------------------------------------------------------

def empty_salt(program: str) -> bytes:
    """The public salt of the session's first row (`registered.supplied_salts`' row 0 of `VALUE`)."""
    return G.supplied_salts(program, VALUE, 1)[0]


def empty_opening(program: str, S) -> bytes:
    """b‖c of the empty state under `empty_salt`, which the verifier computes."""
    return _INST.commit_string(_INST.default_key, G.row_digest(row(encode(empty(S), S)), WORD_BITS, SCHEMA),
                               empty_salt(program))


def commit_window(program: str, rows: Sequence[Row], S, registration: str, carry: G.Registered | None = None,
                  salts: Sequence[bytes] | None = None) -> G.Registered:
    """A window's produced value: its P + 2 rows. `carry`: the previous window's value, whose last row is this window's
    first, under the same salt; for the session's first window (None), the first row is `empty` under `empty_salt`.
    `salts`: the other P + 1 rows' (default fresh, `hm96`'s `fresh_salt`)."""
    if len(rows) != S.P + 2:
        raise ValueError(f"a window is {S.P + 2} rows, not {len(rows)}")
    words = [row(encode(r, S)) for r in rows]
    if carry is None:
        if rows[0] != empty(S):
            raise ValueError("the session's first row is the empty state")
        first = empty_salt(program)
    else:
        if carry.rows[-1] != words[0]:
            raise ValueError("a window starts in the row the previous one ended in")
        first = carry.salts[-1]
    rest = list(salts) if salts is not None else [_INST.fresh_salt() for _ in range(S.P + 1)]
    if len(rest) != S.P + 1:
        raise ValueError(f"{S.P + 1} salts, one per row after the first")
    return produced.commit(program, VALUE, words, registration, [first, *rest], WORD_BITS, SCHEMA)


def reads(S) -> list[dict[str, int]]:
    """Each unit's rows of `VALUE`: the load's `prev` row 0 and `next` row 1, step u's `row` u and `next` u + 1."""
    return [{"prev": 0, "next": 1}] + [{"row": u, "next": u + 1} for u in range(1, S.P + 1)]


def openings(value: G.Registered) -> list[bytes]:
    """Each row's b‖c, as the produced record's `openings` give them."""
    return list(value.bcs)


def check_chain(windows: Sequence[Sequence[bytes]], program: str, S) -> str | None:
    """The verifier's check of a session's windows, given each window's row openings b‖c in order: None, or why not.
    `C-rows`: a window isn't P + 2 rows; `C-first`: the session doesn't start in the empty state under its public salt;
    `C-carry`: a window doesn't start with the previous window's last row."""
    for w, ops in enumerate(windows):
        if len(ops) != S.P + 2:
            return f"C-rows: window {w} has {len(ops)} rows, not {S.P + 2}"
    if windows and bytes(windows[0][0]) != empty_opening(program, S):
        return "C-first: the session's first row isn't the empty state"
    for w in range(1, len(windows)):
        if bytes(windows[w][0]) != bytes(windows[w - 1][-1]):
            return f"C-carry: window {w}'s first row isn't window {w - 1}'s last"
    return None


# --- the circuit ------------------------------------------------------------------------------------------------------------

def _split(bits, fs) -> dict[str, list[list[Any]]]:
    out, at = {}, 0
    for name, n, w in fs:
        out[name] = [list(bits[at + e * w:at + (e + 1) * w]) for e in range(n)]
        at += n * w
    return out


def _flat(d, fs) -> list[Any]:
    return [b for name, _n, _w in fs for e in d[name] for b in e]


def _pad(x, w):
    return (list(x) + [ZERO] * w)[:w]


def _eq(C, a, b):
    w = max(len(a), len(b))
    return and_reduce(C, [(x ^ y).inv() for x, y in zip(_pad(a, w), _pad(b, w))])


def _lt(C, a, b):
    w = max(len(a), len(b))
    return sub(C, _pad(a, w), _pad(b, w), w)[1]


def _zero(C, a):
    return and_reduce(C, [x.inv() for x in a])


def _or(C, a, b):
    return a ^ b ^ C.AND(a, b)


def _shl(C, xs, amt):
    """Entries `xs` moved toward index 0 by `amt` (bits, low first), zero entries shifted in."""
    cur = [list(e) for e in xs]
    for j, s in enumerate(amt):
        d = 1 << j
        cur = [mux(C, s, cur[k + d] if k + d < len(cur) else [ZERO] * len(cur[k]), cur[k]) for k in range(len(cur))]
    return cur


def _shr(C, xs, amt):
    """Entries `xs` moved away from index 0 by `amt`, zero entries shifted in (the last ones fall off)."""
    cur = [list(e) for e in xs]
    for j, s in enumerate(amt):
        d = 1 << j
        cur = [mux(C, s, cur[k - d] if k >= d else [ZERO] * len(cur[k]), cur[k]) for k in range(len(cur))]
    return cur


def _popcount(C, bits, w):
    out = [ZERO] * w
    for b in bits:
        out = add(C, out, [b], width=w)
    return out


def _build_advance(C, S, ok, cur, stepped, taken, used, nxt):
    Z, Wd = widths(S), width(S)
    r = _split(cur, row_fields(S))
    queue = {name: _shl(C, r[name], taken) for name, _n, _w in queue_fields(S)}
    dec = _shl(C, r["dec"], used)
    w = max(Z.DW, Z.AC)
    left, under = sub(C, _pad(r["left"][0], w), _pad(used, w), w)
    left = left[:Z.DW]
    steps = r["steps"][0]
    active, last = or_reduce(C, steps), _eq(C, steps, const_bits(1, Z.PW))
    step_ok = and_reduce(C, [ok[0], under.inv(), C.AND(last, _zero(C, left).inv()).inv()])
    stepped_row = _pad(list(stepped) + _flat(queue, queue_fields(S)) + [e[0] for e in dec] + left
                       + sub(C, steps, const_bits(1, Z.PW), Z.PW)[0], Wd)
    same = _eq(C, mux(C, active, stepped_row, list(cur)), nxt)
    return [[C.AND(C.AND(active, step_ok.inv()).inv(), same)], [active]]


def _build_load(C, S, prev, arrivals, decisions, count, steps, nxt):
    Z, Wd = widths(S), width(S)
    p, a = _split(prev, row_fields(S)), _split(arrivals, arrival_fields(S))
    av = [e[0] for e in a["a_valid"]]
    prefix = and_reduce(C, [C.AND(av[k + 1], av[k].inv()).inv() for k in range(S.K - 1)])
    q = _popcount(C, [e[0] for e in p["q_valid"]], Z.QW)
    queue = {}
    for (qn, _n, w), (an, _k, _w) in zip(queue_fields(S), arrival_fields(S)):
        moved = _shr(C, a[an] + [[ZERO] * w for _ in range(S.Q)], q)
        if qn == "q_valid":
            overflow = or_reduce(C, [e[0] for e in moved[S.Q:]])
        queue[qn] = [[x ^ y for x, y in zip(pe, me)] for pe, me in zip(p[qn], moved[:S.Q])]
    past = [C.AND(decisions[k], _lt(C, const_bits(k, Z.DW + 1), count).inv()) for k in range(S.D)]
    checks = [prefix, overflow.inv(), _zero(C, past), _lt(C, const_bits(S.D, Z.DW + 1), count).inv(),
              _lt(C, const_bits(S.P, Z.PW + 1), steps).inv(), C.AND(_zero(C, steps), _zero(C, count).inv()).inv(),
              _zero(C, p["left"][0]), _zero(C, p["steps"][0])]
    loaded = _pad(_flat(p, state_fields(S)) + _flat(queue, queue_fields(S)) + list(decisions) + list(count) + list(steps),
                  Wd)
    return [and_reduce(C, checks + [_eq(C, loaded, nxt)])]


def _sig_advance(S):
    Z, Wd = widths(S), width(S)
    return ((("ok", BIT), ("row", Array(Wd, BIT)), ("stepped", Array(_bits_of(state_fields(S)), BIT)),
             ("taken", Array(Z.AC, BIT)), ("used", Array(Z.AC, BIT)), ("next", Array(Wd, BIT))), Tuple(BIT, BIT))


def _sig_load(S):
    Z, Wd = widths(S), width(S)
    return ((("prev", Array(Wd, BIT)), ("arrivals", Array(_bits_of(arrival_fields(S)), BIT)), ("decisions", Array(S.D, BIT)),
             ("count", Array(Z.DW, BIT)), ("steps", Array(Z.PW, BIT)), ("next", Array(Wd, BIT))), BIT)


ReplayAdvance = _part(
    "ReplayAdvance", STATICS, _sig_advance, _build_advance,
    "the chain beside one replay step: the row it reads becomes the next row it reads, the step's next state with the "
    "queue and the decisions past what it took and read (ok, and active = 0 for an idle step past the window's last)")

ReplayLoad = _part(
    "ReplayLoad", STATICS, _sig_load, _build_load,
    "a window's first unit: the last window's last row with the window's certified arrivals appended to the queue and "
    "its decisions and executed step count loaded, the last window's all read")


def advance_unit(S):
    return bind(ReplayAdvance, **_statics(S))


def load_unit(S):
    return bind(ReplayLoad, **_statics(S))


def _statics(S) -> dict[str, int]:
    return {k: getattr(S, k) for k in STATICS}


def advance_arguments(S, cur: Row, ok: bool, stepped: Row, taken: int, used: int, nxt: Sequence[int]) -> list[list[int]]:
    """`ReplayAdvance`'s arguments, each a flat bit list: `nxt` is the next row's bits."""
    Z = widths(S)
    tb, ub = [], []
    _put(tb, taken, Z.AC, "taken")
    _put(ub, used, Z.AC, "used")
    return [[int(ok)], list(encode(cur, S)), list(state_bits(stepped, S)), tb, ub, list(nxt)]


def load_arguments(S, prev: Row, arrivals: Sequence[Arrival], decisions: Sequence[int], count: int, steps: int,
                   nxt: Sequence[int]) -> list[list[int]]:
    """`ReplayLoad`'s arguments: `decisions` the D-bit field as given (bits past `count` included), `nxt` the next row's
    bits."""
    Z = widths(S)
    if len(arrivals) > S.K or len(decisions) > S.D:
        raise ValueError("the arrivals or decisions exceed the size class")
    vals = {"a_valid": [1] * len(arrivals) + [0] * (S.K - len(arrivals))}
    for k, name in enumerate(("id", "P", "O")):
        vals[f"a_{name}"] = [x[k] for x in arrivals] + [0] * (S.K - len(arrivals))
    cb, sb = [], []
    _put(cb, count, Z.DW, "count")
    _put(sb, steps, Z.PW, "steps")
    return [list(encode(prev, S)), _flatten(vals, arrival_fields(S)), list(decisions) + [0] * (S.D - len(decisions)), cb,
            sb, list(nxt)]


def _evaluate(unit, cases: Sequence[list[list[int]]]):
    import numpy as np

    args = [np.asarray(col, dtype=np.uint8) for col in zip(*cases)]
    return evaluate_bits(unit, *args).reshape(len(cases), -1).tolist()


def evaluate_advance(S, cases: Sequence[list[list[int]]]) -> list[tuple[int, int]]:
    """`ReplayAdvance` on each case's arguments (`advance_arguments`), in one evaluation: (ok, active) each."""
    return [(int(o[0]), int(o[1])) for o in _evaluate(advance_unit(S), cases)]


def evaluate_load(S, cases: Sequence[list[list[int]]]) -> list[int]:
    """`ReplayLoad` on each case's arguments (`load_arguments`): its ok bit each."""
    return [int(o[0]) for o in _evaluate(load_unit(S), cases)]


# --- cost -----------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _counted(which: str, key: tuple) -> int:
    S = SimpleNamespace(**dict(zip(STATICS, key)))
    sig, build = {"advance": (_sig_advance, _build_advance), "load": (_sig_load, _build_load)}[which]
    C = Circuit(cse=True)
    ins = [C.input(name, sum(BL.elements(t))) for name, t in sig(S)[0]]
    out = build(C, S, *ins)
    C.output("out", [b for e in out for b in (e if isinstance(e, list) else [e])])
    return C.counts()["ands"]


def size(S) -> dict[str, int]:
    """The chain's cost at a shape (traced on a counting circuit, no gates written): per step `ReplayAdvance`'s ANDs, per
    window P of them and one `ReplayLoad`, the rows, the reads and their path depth."""
    key = tuple(getattr(S, k) for k in STATICS)
    step, window = _counted("advance", key), _counted("load", key)
    return {"width": width(S), "state_bits": _bits_of(state_fields(S)), "queue_bits": _bits_of(queue_fields(S)),
            "ands_per_step": step, "ands_load": window, "ands_per_window": S.P * step + window,
            "rows_per_window": S.P + 2, "reads_per_window": 2 * (S.P + 1), "path_depth": (S.P + 1).bit_length(),
            "record_bytes_per_window": (S.P + 2) * produced.BC_BYTES}


__all__ = ["Arrival", "MalformedRow", "ReplayAdvance", "ReplayLoad", "Row", "Running", "STATICS", "Shape", "VALUE",
           "Waiting", "advance", "advance_arguments", "advance_unit", "arrival_fields", "check_chain", "commit_window",
           "decode", "empty", "empty_opening", "empty_salt", "encode", "evaluate_advance", "evaluate_load", "load",
           "load_arguments", "load_unit", "openings", "queue_fields", "reads", "row", "row_fields", "size", "state_bits",
           "state_fields", "unrow", "widths"]
