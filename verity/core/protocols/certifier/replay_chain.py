"""The replay's state chain: vLLM's scheduler state handed from one step unit to the next under hiding commitments
(PROTOCOL.md, "The replay's state chain").

The compile unit replays the scheduler from advice v1 and the certified record, one step unit per engine step, each
`replay_step.ReplayStep_v1` (the replay's step): the committed queue state, the next A certified arrivals and the next A
decisions in, the step's rows and the next state out. A window is many steps, so the state passes between units as
rows of produced values, each unit reading its rows at public positions. This module is that chain: the rows (`Shape`,
`Row`, `Window`, `encode`, `decode`), their commitment and reads (`commit_window`, `reads`), the verifier's check of how
the windows join (`check_chain`), and the three Boolean parts it puts in the units, `ReplayAdvance` beside each step,
`ReplayLoad` once a window and `ReplayClose` once a session, with `advance`, `load` and `close` their specifications.

**Why a window has a row of its own.** The step reads its arrivals and decisions at the scan's cursors, which move by
its `taken` and `used`, and the cursors are private: a step's position in the certified arrivals or the window's
decisions is the schedule itself. A registered read is at a position the verifier derives, so no unit can read the
arrival record at the cursor. So each window has one window row, the certified arrivals not yet taken when it starts
followed by the window's own (`queue`, at most Q) and the window's decoded decisions (`dec`, at most D), and every step
unit reads it at the same public position. The state row carries the two cursors as private heads (`qh`, `dh`), and
`ReplayAdvance` checks that the step's A arrivals and A decision bits are the window row's at the heads, with a log
shifter that computes only the A entries it returns. Once a window `ReplayLoad` writes the window row: the last
window row past its heads, then the window's certified arrivals (K slots, the warden's) and its decoded decisions,
which it reads at public positions. So every read in the chain is at a public position, and no gate hashes a row: the
Lean verifier checks each read's path (`registered.verify_read`).

**The rows**, each field's elements in turn and each element least significant bit first, then zero bits to a whole
16-bit word. A state row (`row_fields`, `Shape.width` bits) and a window row (`window_fields`, `Shape.window_width`):

    state    run_valid R×1 | run_id R×IW | run_P R×LW | run_O R×LW | run_out R×LW | run_comp R×LW [| run_pfx R×EW]
             | wait_valid W×1 | wait_id W×IW | wait_P W×LW | wait_O W×LW | wait_out W×LW | free 1×BW
             [| run_fin R×1 | run_cap R×1 | wait_cap W×1]
    heads    qh 1×QW | dh 1×DW | steps 1×PW

    queue    q_valid Q×1 | q_id Q×IW | q_P Q×LW | q_O Q×LW [| q_cap Q×1]
    dec      dec D×1 | left 1×DW

- `state` is ReplayStep_v1's committed state as its `_in_fields` lay it out, and its next state as its `_out_fields`
  do, bit for bit: R running slots (id, prompt P, certified output length O, outputs so far, tokens computed), W
  waiting slots (a waiting request has computed nothing) and the free block count. A running request holds
  ⌈computed / B⌉ blocks, so the KV pool is the free count, with no count per request. With EW (the walk entries'
  width, 0 for none) each running slot's walk entry `run_pfx` follows `run_comp`. With AS (async scheduling) the
  state ends with each running slot's `fin` and `cap` and each waiting slot's `cap`, and an arrival carries its
  `cap` (O is its max_tokens): `q_cap` here, the step's `arr_cap` and the load's `a_cap`.
- A `Row`'s and a `Window`'s entries are tuples of their fields in `replay_step.Req`'s order, those the shape has:
  running (id, P, O, out, computed[, cap, fin][, pfx]), waiting (id, P, O, out[, cap]) and an arrival
  (id, P, O[, cap]) (`running_names`, `waiting_names`, `arrival_names`).
- `qh` and `dh` are how many of the window row's arrivals and decisions the window's steps so far took and read;
  `steps` counts the window's executed steps still to come.
- `queue` is the arrivals in certified order, the step's `arr_*` the A at `qh`; `dec` the decisions, the step's `bits`
  the A at `dh`, and `left` counts them.
- Widths, from the size class and the public config (`widths`): LW = bitlen(L), BW = bitlen(NB), AC = bitlen(A),
  QW = bitlen(Q), DW = bitlen(D), PW = bitlen(P), bitlen(0) being 1; IW is a request id's and EW a walk entry's.
- In every row the valid elements of each group are a prefix and the rest are zero bits, as are `dec`'s bits past
  `left`. `decode` refuses any other row. The circuit needs no such check: the first rows are the empty ones, and every
  later row is a proved unit's output.

**A window** is two produced values (`one_stage.produced`) in `hm96-sha512/row/v2` bit rows: `VALUE`, its state rows
S_0 … S_{P+1}, and `WINDOW`, the last window's row and its own; P, the step units a window, is a public cap. Unit 0 is
the load: it reads state rows 0 and 1 and window rows 0 and 1 with the window's arrivals, decisions and step count
n ≤ P. Unit u ≥ 1 is step u: ReplayStep_v1 on state row u's state with the arrivals and bits `ReplayAdvance` checks
against window row 1, which also checks state row u + 1, so a row two units read is the one row `produced.shared`
holds them to. Steps past the window's n are idle (`steps` = 0): the next row is the row, and the unit's rows are
dropped (`ReplayAdvance`'s `active` = 0).

- **The carry.** Window w's state row 0 is window w − 1's state row P + 1, and its window row 0 is window w − 1's
  window row 1, each committed under the same salt (`commit_window(carry=…)`), so its opening b‖c is the same, and the
  verifier checks they're equal (`check_chain`). The load refuses a state row 0 that left steps to run, so each window
  runs exactly its own. Decisions left unread carry: the load writes the window's after the last window row's unread
  ones, as it does the queue's arrivals, since A5 ("The windows") can charge a decision up to two windows before its
  step's (at T = 20, 5 of async-kv-pressure's 17 decisions and 5 of sync-kv-pressure's 15). A decision is then read
  after its window's row loads, so it is charged no later than A5 allows.
- **The end.** `ReplayClose` reads the session's last state row (its last window's P + 1) and window row (that
  window's 1) and refuses a session that ends with a decision unread or a step to run (`close`).
- **The start.** The session's first rows are the empty state (`empty`: nothing held, NB − 1 free blocks, vLLM's block
  0 being its null block) and the empty window, each under its public salt (`empty_salts`), so the verifier computes
  their openings (`empty_openings`).

**Cost** (`size`). Per step: `ReplayAdvance`'s ANDs (the two fronts, about (Q + A·QW)·(1 + IW + 2·LW) and D + A·DW, and
about 2·width for the idle select and the equality), three reads and one state row. Per window: P steps, one
`ReplayLoad`, P + 2 state rows and two window rows, and 3P + 4 reads, each a frame-v3 path of depth ⌈log2(P + 2)⌉ or 1.
Per session one `ReplayClose` and its two reads.
On the trace corpus's 15 sync traces a step is 16,569 to 80,959 ANDs and a window 3.5e7 to 1.9e8 (PROTOCOL.md, before
the carry). AS and EW widen the rows by the step's own fields, and the parts with them: at the served vLLM run's engine
class (R, W, A) = (8, 1, 9), Q 4, D 32, P 54, K 4, `ReplayAdvance` is 1,761 ANDs sync, 1,814 at AS = 1 and 2,102 with
17-bit walk entries, and `ReplayLoad` 1,944, 2,000 and 2,144 (protocol/served-shape.md).
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
from verity.core.protocols.certifier.head_check import BIT, _part
from verity.core.protocols.sampled_proofs.one_stage import produced
from verity.core.protocols.sampled_proofs.one_stage import registered as G

VALUE = "replay-state"
WINDOW = "replay-window"
WORD_BITS = 16
STATICS = ("R", "W", "A", "L", "NB", "IW", "Q", "D", "P", "K", "AS", "EW")
_INST = hm96.SHA512


class MalformedRow(ValueError):
    """A row that isn't a state's or a window's encoding."""


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


@dataclass(frozen=True)
class Shape:
    """The public size class: ReplayStep_v1's R running slots, W waiting slots and A arrivals a step, the engine's
    max_model_len L and num_gpu_blocks NB, a request id's width IW; the chain's Q pending arrivals, D decisions and P
    steps a window, and the warden's K arrival slots a window; ReplayStep_v1's async_scheduling AS (0 or 1) and its
    walk entries' width EW (0 for none)."""

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
    AS: int = 0
    EW: int = 0

    def __post_init__(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"{f.name} is an integer, not {v!r}")
            if f.name not in ("AS", "EW") and v < 1:
                raise ValueError(f"{f.name} is a positive integer, not {v!r}")
        if self.AS not in (0, 1) or self.EW < 0:
            raise ValueError(f"AS is 0 or 1 and EW at least 0, not {self.AS!r} and {self.EW!r}")

    def statics(self) -> dict[str, int]:
        return {k: getattr(self, k) for k in STATICS}

    @property
    def width(self) -> int:
        """A state row's bits, padded to whole 16-bit words."""
        return width(self)

    @property
    def words(self) -> int:
        return self.width // WORD_BITS

    @property
    def window_width(self) -> int:
        """A window row's bits, padded to whole 16-bit words."""
        return window_width(self)


def widths(S) -> SimpleNamespace:
    return SimpleNamespace(LW=_bitlen(S.L), BW=_bitlen(S.NB), AC=_bitlen(S.A), QW=_bitlen(S.Q), DW=_bitlen(S.D),
                           PW=_bitlen(S.P))


def state_fields(S) -> list[tuple[str, int, int]]:
    """ReplayStep_v1's state, its `_in_fields`' and `_out_fields`' order and widths: with EW `run_pfx` after
    `run_comp`, with AS the async group after `free`."""
    Z, R, W = widths(S), S.R, S.W
    return ([("run_valid", R, 1), ("run_id", R, S.IW), ("run_P", R, Z.LW), ("run_O", R, Z.LW), ("run_out", R, Z.LW),
             ("run_comp", R, Z.LW)] + ([("run_pfx", R, S.EW)] if S.EW else [])
            + [("wait_valid", W, 1), ("wait_id", W, S.IW), ("wait_P", W, Z.LW), ("wait_O", W, Z.LW), ("wait_out", W, Z.LW),
               ("free", 1, Z.BW)]
            + ([("run_fin", R, 1), ("run_cap", R, 1), ("wait_cap", W, 1)] if S.AS else []))


def running_names(S) -> tuple[str, ...]:
    """A running entry's fields, in `Row.running`'s tuples."""
    return ("id", "P", "O", "out", "comp") + (("cap", "fin") if S.AS else ()) + (("pfx",) if S.EW else ())


def waiting_names(S) -> tuple[str, ...]:
    """A waiting entry's fields, in `Row.waiting`'s tuples."""
    return ("id", "P", "O", "out") + (("cap",) if S.AS else ())


def arrival_names(S) -> tuple[str, ...]:
    """An arrival's fields, in `Window.queue`'s tuples, the step's and the load's arrivals."""
    return ("id", "P", "O") + (("cap",) if S.AS else ())


def _arrivals(prefix: str, n: int, S) -> list[tuple[str, int, int]]:
    Z = widths(S)
    return ([(f"{prefix}_valid", n, 1), (f"{prefix}_id", n, S.IW), (f"{prefix}_P", n, Z.LW), (f"{prefix}_O", n, Z.LW)]
            + ([(f"{prefix}_cap", n, 1)] if S.AS else []))


def queue_fields(S) -> list[tuple[str, int, int]]:
    return _arrivals("q", S.Q, S)


def front_fields(S) -> list[tuple[str, int, int]]:
    """The step's arrivals, ReplayStep_v1's `arr_*`: the window row's A at `qh`."""
    return _arrivals("arr", S.A, S)


def arrival_fields(S) -> list[tuple[str, int, int]]:
    """`ReplayLoad`'s `arrivals`: the window's K certified arrival slots, a valid prefix."""
    return _arrivals("a", S.K, S)


def row_fields(S) -> list[tuple[str, int, int]]:
    Z = widths(S)
    return state_fields(S) + [("qh", 1, Z.QW), ("dh", 1, Z.DW), ("steps", 1, Z.PW)]


def window_fields(S) -> list[tuple[str, int, int]]:
    return queue_fields(S) + [("dec", S.D, 1), ("left", 1, widths(S).DW)]


def _bits_of(fs) -> int:
    return sum(n * w for _, n, w in fs)


def _words(bits: int) -> int:
    return WORD_BITS * -(-bits // WORD_BITS)


def width(S) -> int:
    return _words(_bits_of(row_fields(S)))


def window_width(S) -> int:
    return _words(_bits_of(window_fields(S)))


# --- the rows ---------------------------------------------------------------------------------------------------------------

Running = tuple[int, ...]   # (id, P, O, out, computed[, cap, fin][, pfx]): `running_names`
Waiting = tuple[int, ...]   # (id, P, O, out[, cap]): `waiting_names`
Arrival = tuple[int, ...]   # (id, P, O[, cap]): `arrival_names`


@dataclass(frozen=True)
class Row:
    """A state row: the scheduler and the window's heads between two units (the module's docstring, "The rows")."""

    running: tuple[Running, ...] = ()
    waiting: tuple[Waiting, ...] = ()
    free: int = 0
    qh: int = 0
    dh: int = 0
    steps: int = 0


@dataclass(frozen=True)
class Window:
    """A window row: the arrivals the window's steps may take, in certified order, and its decisions."""

    queue: tuple[Arrival, ...] = ()
    decisions: tuple[int, ...] = ()


def empty(S) -> Row:
    """The session's first state row: the empty engine, its NB − 1 free blocks."""
    return Row(free=S.NB - 1)


def _put(out: list[int], v: int, w: int, name: str) -> None:
    v = int(v)
    if not 0 <= v < 1 << w:
        raise MalformedRow(f"{name} = {v} doesn't fit {w} bits")
    out += [(v >> j) & 1 for j in range(w)]


def _group(vals: dict[str, list[int]], prefix: str, names: Sequence[str], items: Sequence[Sequence[int]], n: int) -> None:
    if any(len(x) != len(names) for x in items):
        raise MalformedRow(f"{prefix}: an entry is ({', '.join(names)}) at this shape")
    vals[f"{prefix}_valid"] = [1] * len(items) + [0] * (n - len(items))
    for k, name in enumerate(names):
        vals[f"{prefix}_{name}"] = [x[k] for x in items] + [0] * (n - len(items))


def _values(row: Row, S) -> dict[str, list[int]]:
    if len(row.running) > S.R or len(row.waiting) > S.W:
        raise MalformedRow("the row exceeds the size class")
    if row.qh > S.Q or row.dh > S.D or row.steps > S.P:
        raise MalformedRow(f"heads {row.qh}, {row.dh} of at most {S.Q}, {S.D}; steps {row.steps} of at most {S.P}")
    vals: dict[str, list[int]] = {}
    _group(vals, "run", running_names(S), row.running, S.R)
    _group(vals, "wait", waiting_names(S), row.waiting, S.W)
    vals["free"], vals["qh"], vals["dh"], vals["steps"] = [row.free], [row.qh], [row.dh], [row.steps]
    return vals


def _window_values(win: Window, S) -> dict[str, list[int]]:
    if len(win.queue) > S.Q or len(win.decisions) > S.D or any(b not in (0, 1) for b in win.decisions):
        raise MalformedRow("the window exceeds the size class, or a decision isn't a bit")
    vals: dict[str, list[int]] = {}
    _group(vals, "q", arrival_names(S), win.queue, S.Q)
    vals["dec"] = list(win.decisions) + [0] * (S.D - len(win.decisions))
    vals["left"] = [len(win.decisions)]
    return vals


def _flatten(vals: dict[str, list[int]], fs) -> list[int]:
    out: list[int] = []
    for name, _n, w in fs:
        for e, v in enumerate(vals[name]):
            _put(out, v, w, f"{name}[{e}]")
    return out


def _padded(bits: list[int], w: int) -> tuple[int, ...]:
    return tuple(bits + [0] * (w - len(bits)))


def encode(row: Row, S) -> tuple[int, ...]:
    """The state row's `width(S)` bits."""
    return _padded(_flatten(_values(row, S), row_fields(S)), width(S))


def encode_window(win: Window, S) -> tuple[int, ...]:
    """The window row's `window_width(S)` bits."""
    return _padded(_flatten(_window_values(win, S), window_fields(S)), window_width(S))


def state_bits(row: Row, S) -> tuple[int, ...]:
    """The row's `state` part: ReplayStep_v1's state input, or its next-state output, as bits."""
    return tuple(_flatten(_values(row, S), state_fields(S)))


def front(S, win: Window, row: Row) -> tuple[tuple[Arrival, ...], tuple[int, ...]]:
    """The step's arrivals and its A decision bits: the window row's at the row's heads (zeros past `left`)."""
    bits = win.decisions[row.dh:row.dh + S.A]
    return win.queue[row.qh:row.qh + S.A], tuple(bits) + (0,) * (S.A - len(bits))


def front_bits(S, arrivals: Sequence[Arrival]) -> tuple[int, ...]:
    """The step's arrivals as ReplayStep_v1's `arr_*` fields, flat."""
    if len(arrivals) > S.A:
        raise MalformedRow(f"{len(arrivals)} arrivals of at most {S.A}")
    vals: dict[str, list[int]] = {}
    _group(vals, "arr", arrival_names(S), arrivals, S.A)
    return tuple(_flatten(vals, front_fields(S)))


def _unflatten(bits: Sequence[int], fs) -> tuple[dict[str, list[int]], int]:
    at, f = 0, {}
    for name, n, w in fs:
        f[name] = [sum(int(bits[at + e * w + j]) << j for j in range(w)) for e in range(n)]
        at += n * w
    return f, at


def _fields_of(bits: Sequence[int], fs, w: int) -> dict[str, list[int]]:
    if len(bits) != w or any(b not in (0, 1) for b in bits):
        raise MalformedRow(f"a row is {w} bits")
    f, at = _unflatten(bits, fs)
    if any(bits[at:]):
        raise MalformedRow("nonzero padding")
    return f


def _items(f: dict[str, list[int]], prefix: str, names: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    valid = f[f"{prefix}_valid"]
    k = sum(valid)
    if valid != [1] * k + [0] * (len(valid) - k):
        raise MalformedRow(f"{prefix}: the valid slots aren't a prefix")
    if any(f[f"{prefix}_{name}"][e] for name in names for e in range(k, len(valid))):
        raise MalformedRow(f"{prefix}: an empty slot has a bit set")
    return tuple(tuple(f[f"{prefix}_{name}"][e] for name in names) for e in range(k))


def decode(bits: Sequence[int], S) -> Row:
    """The state row `bits` encode; `MalformedRow` for any other bits (the module's docstring, "The rows")."""
    f = _fields_of(bits, row_fields(S), width(S))
    qh, dh, steps = f["qh"][0], f["dh"][0], f["steps"][0]
    if qh > S.Q or dh > S.D or steps > S.P:
        raise MalformedRow(f"heads {qh}, {dh} of at most {S.Q}, {S.D}; steps {steps} of at most {S.P}")
    return Row(_items(f, "run", running_names(S)), _items(f, "wait", waiting_names(S)), f["free"][0], qh, dh, steps)


def decode_window(bits: Sequence[int], S) -> Window:
    """The window row `bits` encode; `MalformedRow` for any other bits."""
    f = _fields_of(bits, window_fields(S), window_width(S))
    left = f["left"][0]
    if left > S.D or any(f["dec"][left:]):
        raise MalformedRow(f"left {left} of {S.D} with no decision past it")
    return Window(_items(f, "q", arrival_names(S)), tuple(f["dec"][:left]))


def row(bits: Sequence[int]) -> tuple[int, ...]:
    """A row's bits as the produced row's u16 words, bit i of the row bit i % 16 of word i // 16."""
    if len(bits) % WORD_BITS:
        raise ValueError("a row is whole 16-bit words")
    return tuple(sum(int(bits[w + i]) << i for i in range(WORD_BITS)) for w in range(0, len(bits), WORD_BITS))


def unrow(words: Sequence[int]) -> tuple[int, ...]:
    return tuple((int(w) >> i) & 1 for w in words for i in range(WORD_BITS))


# --- the specification of the units -----------------------------------------------------------------------------------------

def advance(S, win: Window, cur: Row, ok: bool, stepped: Row, arrivals: Sequence[Arrival], bits: Sequence[int], taken: int,
            used: int) -> Row | None:
    """The state row after step unit u: `cur` when the window's steps are done (idle), else the step's next state
    (`stepped`'s running, waiting and free) with the heads past what it took and read and one step fewer. None refuses:
    the step's ok is 0, its arrivals or bits aren't the window row's at the heads, it took more than A or than are
    left, or it read more than A or than are left. Decisions the window's last step leaves unread carry (`load`)."""
    if not cur.steps:
        return cur
    want_arr, want_bits = front(S, win, cur)
    dh = cur.dh + used
    if not ok or tuple(map(tuple, arrivals)) != want_arr or tuple(bits) + (0,) * (S.A - len(bits)) != want_bits:
        return None
    if taken > len(want_arr) or used > S.A or dh > len(win.decisions):
        return None
    return Row(stepped.running, stepped.waiting, stepped.free, cur.qh + taken, dh, cur.steps - 1)


def load(S, prev: Row, last: Window, arrivals: Sequence[Arrival], decisions: Sequence[int],
         steps: int) -> tuple[Row, Window] | None:
    """The window's state row 1 and window row: state row 0 (`prev`, the last window's last) with its heads reset and
    the window's executed step count, and the last window's arrivals and decisions past `prev`'s heads with the
    window's certified arrivals and its decisions appended. None refuses: the last window left steps, or read past its
    decisions, the queue overflows Q or the decisions D, more than K arrivals or P steps."""
    if prev.steps or prev.dh > len(last.decisions):
        return None
    rest, kept = last.queue[prev.qh:], last.decisions[prev.dh:]
    if len(arrivals) > S.K or len(rest) + len(arrivals) > S.Q:
        return None
    if len(kept) + len(decisions) > S.D or steps > S.P:
        return None
    return (Row(prev.running, prev.waiting, prev.free, 0, 0, steps),
            Window(rest + tuple(tuple(a) for a in arrivals), kept + tuple(decisions)))


def close(S, row: Row, win: Window) -> bool:
    """Whether a session may end in state row `row` and window row `win` (its last window's P + 1 and 1): the window's
    steps all run and its decisions, the carried ones among them, all read."""
    return not row.steps and row.dh == len(win.decisions)


# --- the chain ------------------------------------------------------------------------------------------------------------

def empty_salts(program: str) -> tuple[bytes, bytes]:
    """The public salts of the session's first state row and first window row (`registered.supplied_salts`' row 0 of
    `VALUE` and of `WINDOW`)."""
    return G.supplied_salts(program, VALUE, 1)[0], G.supplied_salts(program, WINDOW, 1)[0]


def _opening(words: Sequence[int], salt: bytes) -> bytes:
    return _INST.commit_string(_INST.default_key, G.row_digest(words, WORD_BITS, SCHEMA), salt)


def empty_openings(program: str, S) -> tuple[bytes, bytes]:
    """b‖c of the empty state row and the empty window row under `empty_salts`, which the verifier computes."""
    s, w = empty_salts(program)
    return _opening(row(encode(empty(S), S)), s), _opening(row(encode_window(Window(), S)), w)


@dataclass(frozen=True)
class Committed:
    """A window's two produced values: its state rows (`VALUE`) and its window rows (`WINDOW`)."""

    state: G.Registered
    window: G.Registered


def commit_window(program: str, rows: Sequence[Row], window: Window, S, registration: str, carry: Committed | None = None,
                  salts: Sequence[bytes] | None = None) -> Committed:
    """A window's produced values: its P + 2 state rows, and the last window's window row with `window`. `carry`: the
    previous window's, whose last state row and window row are this window's first, under the same salts; for the
    session's first window (None), the first rows are `empty` and the empty window under `empty_salts`. `salts`: the
    other P + 1 state rows' and then the window row's (default fresh, `hm96`'s `fresh_salt`)."""
    if len(rows) != S.P + 2:
        raise ValueError(f"a window is {S.P + 2} state rows, not {len(rows)}")
    words = [row(encode(r, S)) for r in rows]
    if carry is None:
        if rows[0] != empty(S):
            raise ValueError("the session's first row is the empty state")
        first, last_salt = empty_salts(program)
        last = row(encode_window(Window(), S))
    else:
        if carry.state.rows[-1] != words[0]:
            raise ValueError("a window starts in the state row the previous one ended in")
        first, last, last_salt = carry.state.salts[-1], carry.window.rows[-1], carry.window.salts[-1]
    rest = list(salts) if salts is not None else [_INST.fresh_salt() for _ in range(S.P + 2)]
    if len(rest) != S.P + 2:
        raise ValueError(f"{S.P + 2} salts, one per state row after the first and one for the window row")
    state = produced.commit(program, VALUE, words, registration, [first, *rest[:-1]], WORD_BITS, SCHEMA)
    win = produced.commit(program, WINDOW, [last, row(encode_window(window, S))], registration, [last_salt, rest[-1]],
                          WORD_BITS, SCHEMA)
    return Committed(state, win)


def reads(S) -> list[dict[str, dict[str, int]]]:
    """Each unit's rows: the load's state rows `prev` 0 and `next` 1 and window rows `last` 0 and `window` 1; step u's
    state rows `row` u and `next` u + 1 and window row `window` 1."""
    return ([{VALUE: {"prev": 0, "next": 1}, WINDOW: {"last": 0, "window": 1}}]
            + [{VALUE: {"row": u, "next": u + 1}, WINDOW: {"window": 1}} for u in range(1, S.P + 1)])


def close_reads(S) -> dict[str, dict[str, int]]:
    """`ReplayClose`'s rows, of the session's last window: state row `row` P + 1 and window row `window` 1."""
    return {VALUE: {"row": S.P + 1}, WINDOW: {"window": 1}}


def openings(value: Committed) -> tuple[list[bytes], list[bytes]]:
    """Each state row's and window row's b‖c, as the produced record's `openings` give them."""
    return list(value.state.bcs), list(value.window.bcs)


def check_chain(windows: Sequence[tuple[Sequence[bytes], Sequence[bytes]]], program: str, S) -> str | None:
    """The verifier's check of a session's windows, given each window's state row and window row openings b‖c in
    order: None, or why not. `C-rows`: a window isn't P + 2 state rows and 2 window rows; `C-first`: the session
    doesn't start in the empty rows under their public salts; `C-carry`: a window doesn't start with the previous
    window's last state row and its window row."""
    for w, (st, wi) in enumerate(windows):
        if len(st) != S.P + 2 or len(wi) != 2:
            return f"C-rows: window {w} has {len(st)} state rows and {len(wi)} window rows, not {S.P + 2} and 2"
    if windows:
        st, wi = windows[0]
        if (bytes(st[0]), bytes(wi[0])) != empty_openings(program, S):
            return "C-first: the session's first rows aren't the empty state and window"
    for w in range(1, len(windows)):
        (st, wi), (pst, pwi) = windows[w], windows[w - 1]
        if bytes(st[0]) != bytes(pst[-1]) or bytes(wi[0]) != bytes(pwi[-1]):
            return f"C-carry: window {w}'s first rows aren't window {w - 1}'s last"
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


def _front(C, xs, amt, n):
    """xs[amt:amt + n], zero entries past the end: `_shl` with its largest shift first, each stage computing only the
    entries the later stages read (about len(xs) + n·len(amt) muxes an element bit, in place of len(xs)·len(amt))."""
    zero = [ZERO] * len(xs[0])
    cur = [list(e) for e in xs]
    for j in reversed(range(len(amt))):
        d = 1 << j
        cur = [mux(C, amt[j], cur[k + d] if k + d < len(cur) else zero, cur[k]) for k in range(min(n + d - 1, len(cur)))]
    return (cur + [zero] * n)[:n]


def _popcount(C, bits, w):
    out = [ZERO] * w
    for b in bits:
        out = add(C, out, [b], width=w)
    return out


def _build_advance(C, S, ok, cur, win, stepped, arr, bits, taken, used, nxt):
    Z, Wd = widths(S), width(S)
    r, w = _split(cur, row_fields(S)), _split(win, window_fields(S))
    qh, dh, steps = r["qh"][0], r["dh"][0], r["steps"][0]
    fr = {an: _front(C, w[qn], qh, S.A) for (qn, _n, _w), (an, _a, _x) in zip(queue_fields(S), front_fields(S))}
    fv = [e[0] for e in fr["arr_valid"]]
    took = [C.AND(_lt(C, const_bits(k, Z.AC), taken), fv[k].inv()).inv() for k in range(S.A)]
    m = max(Z.DW, Z.AC) + 1
    dh_new = add(C, _pad(dh, m), _pad(used, m), width=m)
    left = w["left"][0]
    active = or_reduce(C, steps)
    step_ok = and_reduce(C, [ok[0], _eq(C, _flat(fr, front_fields(S)), arr),
                             _eq(C, [e[0] for e in _front(C, w["dec"], dh, S.A)], bits),
                             _lt(C, const_bits(S.A, Z.AC + 1), taken).inv(), *took,
                             _lt(C, const_bits(S.A, Z.AC + 1), used).inv(), _lt(C, left, dh_new).inv()])
    stepped_row = _pad(list(stepped) + add(C, qh, _pad(taken, Z.QW), width=Z.QW) + dh_new[:Z.DW]
                       + sub(C, steps, const_bits(1, Z.PW), Z.PW)[0], Wd)
    same = _eq(C, mux(C, active, stepped_row, list(cur)), nxt)
    return [[C.AND(C.AND(active, step_ok.inv()).inv(), same)], [active]]


def _build_load(C, S, prev, last, arrivals, decisions, count, steps, nxt, window):
    checks, loaded, opened = _load_rows(C, S, prev, last, arrivals, decisions, count, steps)
    return [and_reduce(C, checks + [_eq(C, loaded, nxt), _eq(C, opened, window)])]


def _load_rows(C, S, prev, last, arrivals, decisions, count, steps):
    """`load` as wires: its checks (each 1 where it holds), the state row 1 and the window row it computes."""
    Z, Wd, Ww = widths(S), width(S), window_width(S)
    p, lw, a = _split(prev, row_fields(S)), _split(last, window_fields(S)), _split(arrivals, arrival_fields(S))
    av = [e[0] for e in a["a_valid"]]
    prefix = and_reduce(C, [C.AND(av[k + 1], av[k].inv()).inv() for k in range(S.K - 1)])
    rest = {qn: _shl(C, lw[qn], p["qh"][0]) for qn, _n, _w in queue_fields(S)}
    q = _popcount(C, [e[0] for e in rest["q_valid"]], Z.QW)
    queue = {}
    for (qn, _n, w), (an, _k, _w) in zip(queue_fields(S), arrival_fields(S)):
        moved = _shr(C, a[an] + [[ZERO] * w for _ in range(S.Q)], q)
        if qn == "q_valid":
            overflow = or_reduce(C, [e[0] for e in moved[S.Q:]])
        queue[qn] = [[x ^ y for x, y in zip(pe, me)] for pe, me in zip(rest[qn], moved[:S.Q])]
    past = [C.AND(decisions[k], _lt(C, const_bits(k, Z.DW + 1), count).inv()) for k in range(S.D)]
    dh, left = p["dh"][0], lw["left"][0]
    kept = sub(C, left, dh, Z.DW)[0]
    total = add(C, _pad(kept, Z.DW + 1), _pad(count, Z.DW + 1), width=Z.DW + 1)
    moved = _shr(C, [[d] for d in decisions], kept)
    dec = [x[0] ^ y[0] for x, y in zip(_shl(C, lw["dec"], dh), moved)]
    checks = [prefix, overflow.inv(), _zero(C, past), _lt(C, const_bits(S.D, Z.DW + 1), total).inv(),
              _lt(C, const_bits(S.P, Z.PW + 1), steps).inv(), _zero(C, p["steps"][0]), _lt(C, left, dh).inv()]
    loaded = _pad(_flat(p, state_fields(S)) + [ZERO] * (Z.QW + Z.DW) + list(steps), Wd)
    opened = _pad(_flat(queue, queue_fields(S)) + dec + total[:Z.DW], Ww)
    return checks, loaded, opened


def _build_close(C, S, row, window):
    r, w = _split(row, row_fields(S)), _split(window, window_fields(S))
    return [C.AND(_zero(C, r["steps"][0]), _eq(C, r["dh"][0], w["left"][0]))]


def _sig_advance(S):
    Z, Wd = widths(S), width(S)
    return ((("ok", BIT), ("row", Array(Wd, BIT)), ("window", Array(window_width(S), BIT)),
             ("stepped", Array(_bits_of(state_fields(S)), BIT)), ("arr", Array(_bits_of(front_fields(S)), BIT)),
             ("bits", Array(S.A, BIT)), ("taken", Array(Z.AC, BIT)), ("used", Array(Z.AC, BIT)), ("next", Array(Wd, BIT))),
            Tuple(BIT, BIT))


def _sig_load(S):
    Z, Wd, Ww = widths(S), width(S), window_width(S)
    return ((("prev", Array(Wd, BIT)), ("last", Array(Ww, BIT)), ("arrivals", Array(_bits_of(arrival_fields(S)), BIT)),
             ("decisions", Array(S.D, BIT)), ("count", Array(Z.DW, BIT)), ("steps", Array(Z.PW, BIT)),
             ("next", Array(Wd, BIT)), ("window", Array(Ww, BIT))), BIT)


def _sig_close(S):
    return (("row", Array(width(S), BIT)), ("window", Array(window_width(S), BIT))), BIT


ReplayAdvance = _part(
    "ReplayAdvance", STATICS, _sig_advance, _build_advance,
    "the chain beside one replay step: the step's arrivals and decision bits are the window row's at the state row's "
    "heads, and the state row it reads becomes the next, the step's next state with the heads past what it took and read "
    "(ok, and active = 0 for an idle step past the window's last)")

ReplayLoad = _part(
    "ReplayLoad", STATICS, _sig_load, _build_load,
    "a window's first unit: the last window's last state row with its heads reset and the window's executed step count, "
    "and the window row, the last window's arrivals and decisions past its heads with the window's certified arrivals "
    "and its decisions appended, the last window's steps all run")

ReplayClose = _part(
    "ReplayClose", STATICS, _sig_close, _build_close,
    "a session's last unit: its last state row has no step to run and has read every decision of its window row")


def advance_unit(S):
    return bind(ReplayAdvance, **_statics(S))


def load_unit(S):
    return bind(ReplayLoad, **_statics(S))


def close_unit(S):
    return bind(ReplayClose, **_statics(S))


def _statics(S) -> dict[str, int]:
    return {k: getattr(S, k) for k in STATICS}


def advance_arguments(S, win: Window, cur: Row, ok: bool, stepped: Row, arrivals: Sequence[Arrival], bits: Sequence[int],
                      taken: int, used: int, nxt: Sequence[int]) -> list[list[int]]:
    """`ReplayAdvance`'s arguments, each a flat bit list: `arrivals` and `bits` the step's (`front`), `nxt` the next
    state row's bits."""
    Z = widths(S)
    if len(bits) > S.A or any(b not in (0, 1) for b in bits):
        raise MalformedRow(f"at most {S.A} decision bits")
    tb, ub = [], []
    _put(tb, taken, Z.AC, "taken")
    _put(ub, used, Z.AC, "used")
    return [[int(ok)], list(encode(cur, S)), list(encode_window(win, S)), list(state_bits(stepped, S)),
            list(front_bits(S, arrivals)), list(bits) + [0] * (S.A - len(bits)), tb, ub, list(nxt)]


def load_arguments(S, prev: Row, last: Window, arrivals: Sequence[Arrival], decisions: Sequence[int], count: int, steps: int,
                   nxt: Sequence[int], window: Sequence[int]) -> list[list[int]]:
    """`ReplayLoad`'s arguments: `decisions` the D-bit field as given (bits past `count` included), `nxt` and `window`
    the next state row's and the window row's bits."""
    Z = widths(S)
    if len(arrivals) > S.K or len(decisions) > S.D:
        raise ValueError("the arrivals or decisions exceed the size class")
    vals: dict[str, list[int]] = {}
    _group(vals, "a", arrival_names(S), arrivals, S.K)
    cb, sb = [], []
    _put(cb, count, Z.DW, "count")
    _put(sb, steps, Z.PW, "steps")
    return [list(encode(prev, S)), list(encode_window(last, S)), _flatten(vals, arrival_fields(S)),
            list(decisions) + [0] * (S.D - len(decisions)), cb, sb, list(nxt), list(window)]


def close_arguments(S, row: Row, win: Window) -> list[list[int]]:
    """`ReplayClose`'s arguments: the session's last state row's bits and its window row's."""
    return [list(encode(row, S)), list(encode_window(win, S))]


def _evaluate(unit, cases: Sequence[list[list[int]]]):
    args = [list(col) for col in zip(*cases)]
    return evaluate_bits(unit, *args).reshape(len(cases), -1).tolist()


def evaluate_advance(S, cases: Sequence[list[list[int]]]) -> list[tuple[int, int]]:
    """`ReplayAdvance` on each case's arguments (`advance_arguments`), in one evaluation: (ok, active) each."""
    return [(int(o[0]), int(o[1])) for o in _evaluate(advance_unit(S), cases)]


def evaluate_load(S, cases: Sequence[list[list[int]]]) -> list[int]:
    """`ReplayLoad` on each case's arguments (`load_arguments`): its ok bit each."""
    return [int(o[0]) for o in _evaluate(load_unit(S), cases)]


def evaluate_close(S, cases: Sequence[list[list[int]]]) -> list[int]:
    """`ReplayClose` on each case's arguments (`close_arguments`): its ok bit each."""
    return [int(o[0]) for o in _evaluate(close_unit(S), cases)]


# --- cost -----------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _counted(which: str, key: tuple) -> int:
    S = SimpleNamespace(**dict(zip(STATICS, key)))
    sig, build = {"advance": (_sig_advance, _build_advance), "load": (_sig_load, _build_load),
                  "close": (_sig_close, _build_close)}[which]
    C = Circuit(cse=True)
    ins = [C.input(name, sum(BL.elements(t))) for name, t in sig(S)[0]]
    out = build(C, S, *ins)
    C.output("out", [b for e in out for b in (e if isinstance(e, list) else [e])])
    return C.counts()["ands"]


def size(S) -> dict[str, int]:
    """The chain's cost at a shape (traced on a counting circuit, no gates written): per step `ReplayAdvance`'s ANDs, per
    window P of them and one `ReplayLoad`, per session one `ReplayClose`, the rows, the reads and their path depth."""
    key = tuple(getattr(S, k) for k in STATICS)
    step, window = _counted("advance", key), _counted("load", key)
    return {"width": width(S), "window_width": window_width(S), "state_bits": _bits_of(state_fields(S)),
            "queue_bits": _bits_of(queue_fields(S)), "ands_per_step": step, "ands_load": window,
            "ands_close": _counted("close", key),
            "ands_per_window": S.P * step + window, "rows_per_window": S.P + 2, "window_rows_per_window": 2,
            "reads_per_window": 3 * S.P + 4, "path_depth": (S.P + 1).bit_length(),
            "record_bytes_per_window": (S.P + 4) * produced.BC_BYTES}


__all__ = ["Arrival", "Committed", "MalformedRow", "ReplayAdvance", "ReplayClose", "ReplayLoad", "Row", "Running", "STATICS",
           "Shape", "VALUE", "WINDOW", "Waiting", "Window", "advance", "advance_arguments", "advance_unit", "arrival_fields",
           "arrival_names", "check_chain", "close", "close_arguments", "close_reads", "close_unit", "commit_window",
           "decode", "decode_window", "empty", "empty_openings", "empty_salts", "encode", "encode_window",
           "evaluate_advance", "evaluate_close", "evaluate_load", "front", "front_bits", "front_fields", "load",
           "load_arguments", "load_unit", "openings", "queue_fields", "reads", "row", "row_fields", "running_names", "size",
           "state_bits", "state_fields", "unrow", "waiting_names", "widths", "window_fields", "window_width"]
