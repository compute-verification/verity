"""The window class a window circuit is of (`window_circuit`): its public caps and its slots' timing (`WindowClass`, N1's
scan class, and `CLASS_OF_RECORD`), the windows its slots take (`takes`), and the layout of PoUW's tables of a window that
its circuit reads, a parameter of the circuit with two instances on one fixed load (`FixedLoad`): H1's (`Tables`), which
the circuit reads where the model makes them, and Pearl+'s (`PearlTables`), for a circuit cut at its linears.

Given H1's tables of the window (`Tables`, by the names the Commit record registers them under: round s's A table of one K,
``row-table/<s>/k<K>``, and its input-ids table, ``input-ids/<s>``), the circuit reads them where the model makes them, each
a read of an output row (`interface.Read`), which fills no input and holds the proof block's output row to the registered
row, so a table that isn't what the model computed fails the proof of the proof block that reads it. Each quantizer at
that K is `Int8RowQuant_v5{K, R, ROUND}` at its round, whose ``o2`` is H1's leaf of its row (its codes four to a u32, then its scale: the
node's words whole), read as row c·m + i of its round's A table, c its index among its position's int8 linears of that K
and i its position's row; each position's token row (its lane's of the slot's `ScanSlots_v2`, its `TokenFeed_v1`'s or the
previous select's) is read as row i of its round's input-ids table. A position's round and row are a fixed load's (`Tables.at`):
every request of the class's prompt length and outputs, all admitted at round 0 in slot order, so the circuit is still the
class's alone, and a window that isn't that load is refused (`takes`). The circuit's program takes the tables as its values
(`interface.Registered.values`), round by round the A table then the input-ids table, so the window's registration registers
them as its program's, the outputs of the Commit record's registration of them.

Given Pearl+'s tables (`PearlTables`), the circuit is cut at its linears: PoUW credits each linear's product, so the circuit
computes none but the LM head's. A round's forward calls Pearl+ once a cut linear, and each call commits tables of its own,
which the circuit reads as the call commits them: round s's call c, the c-th cut linear of a position's forward, has its A
rows ``pearl-a/<s>/<c>`` (x widened to FP32, the round's m rows of the fixed load, `PearlTables.rows`, then zero rows up to
the call's `PearlTables.a_rows`) and its y ``pearl-y/<s>/<c>`` (its N BF16 words a row widened to FP32, the round's m rows).
Each cut linear is a unit `LinearARow_v1{K, R, ROUND, CALL}` (`registry.linear_cut`) whose output, x widened to FP32, is
held to row i of its call's A rows, i its position's row (`FixedLoad.at`), and a unit `LinearOut_v1{N, W, ROUND, CALL}`
that reads row i of its call's y and hands its BF16 words on to its readers in rows of W (`PearlTables.words`, doubled
where a unit of N / W rows would be no proof block). The circuit's program takes them as its values, round by round and
call by call the A rows then the y. Pearl+'s window record, at its version 2, registers each weight w's B rows too,
``pearl-b/<w>`` (`PearlTables.b`), at the root of the tree PoUW reads B from; the circuit, cut at its linears, reads none
of them. Its
census (`window_cut.census`) counts each cut linear by shape, and none is tied to PoUW's credited product yet.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from verity.protocols.pouw.prover.schemes import SCHEMES

from verity_vllm.protocols.pouw.pearl_c import ROW_ALIGN
from verity_vllm.protocols.pouw.pearl_c import window as PWR

from . import scan_state as SS
from .build import Unexpressible

#: the class Program's sampling (`window_circuit.class_program`), the only one a window's request may take
GREEDY = {"kind": "greedy"}


@dataclass(frozen=True)
class WindowClass:
    """A window class's public caps (N1's scan `Class`: its slots, its prompt ids and its KO) and its slots' timing (fixed
    slots a network timestep): ``slots`` class Programs a window, each of at most ``prompt`` prompt ids and
    ``outputs`` outputs, ``lanes`` of them anchored at each boundary, one every ``every`` scan buckets from the window's
    start (lanes 0: every slot at the window's one boundary). Slots are lane-major: slot ``nb · j + b`` is boundary b's lane j
    (nb = slots / lanes boundaries), anchored at bucket ``every · b``, and the window is ``buckets`` = every · nb buckets. Its
    `scan` is the network certifier's scan class its slots are."""
    slots: int
    prompt: int
    outputs: int
    lanes: int = 0
    every: int = 16

    def __post_init__(self) -> None:
        if not self.lanes:
            object.__setattr__(self, "lanes", self.slots)
        if not all(type(v) is int and v >= 1 for v in (self.slots, self.prompt, self.outputs, self.lanes, self.every)) \
                or self.slots % self.lanes:
            raise ValueError(f"a window class's slots, prompt, outputs, lanes and every are positive integers, its lanes "
                             f"dividing its slots, not {self}")

    @property
    def positions(self) -> int:
        """A class Program's positions: the longest prompt's, then every output but the last."""
        return self.prompt + self.outputs - 1

    @property
    def boundaries(self) -> tuple[int, ...]:
        """The buckets its slots are anchored at, one a boundary."""
        return tuple(self.every * b for b in range(self.slots // self.lanes))

    @property
    def buckets(self) -> int:
        return self.every * len(self.boundaries)

    def anchor(self, slot: int) -> int:
        """Slot ``slot``'s anchor: the bucket whose scan state row before it the slot's Program reads."""
        return self.boundaries[slot % len(self.boundaries)]

    def lane(self, slot: int) -> int:
        """Slot ``slot``'s lane at its boundary: its slot of the scan's state row."""
        return slot // len(self.boundaries)

    @property
    def scan(self) -> Any:
        """The network certifier's scan class at its lanes, outputs, timestep and prompt ids (`scan_state.scan_class`)."""
        return SS.scan_class(self.lanes, self.outputs, self.every, self.prompt)


#: the class of record, the network certifier's envelope at these fixed slots:
#: 8 slots a 32-bucket window, 4 at each of buckets 0 and 16 (EVERY 16, D 2.7 s), each of at most 1,024 prompt ids and 64
#: outputs
CLASS_OF_RECORD = WindowClass(8, 1024, 64, lanes=4, every=16)


class FixedLoad:
    """The load a layout of PoUW's tables is read at (module docstring): every request of the class's prompt length and
    outputs, all admitted at round 0 in slot order, a table a round of the class's ``outputs`` rounds."""
    #: the layout's tables, as a refusal names them
    WHAT = "the tables"

    @staticmethod
    def at(cls: WindowClass, slot: int, position: int) -> tuple[int, int]:
        """Slot ``slot``'s position ``position`` as its round and row under a fixed load: every request of the class's
        prompt length and outputs, all admitted at round 0 in slot order, so a prompt position is round 0's row
        ``slot · prompt + position`` and a later one round ``position - prompt + 1``'s row ``slot``."""
        L = cls.prompt
        return (0, slot * L + position) if position < L else (position - L + 1, slot)

    @staticmethod
    def schedule(cls: WindowClass) -> list[list[tuple[int, int, int]]]:
        """The fixed load's rows by round, ``(request, start, n)`` each (`steps.RowRef`): round 0 each slot's prompt, in
        slot order, and each later round each slot's next position."""
        L, S = cls.prompt, cls.slots
        return [[(k, 0, L) for k in range(S)]] + [[(k, L + s - 1, 1) for k in range(S)] for s in range(1, cls.outputs)]


@dataclass(frozen=True)
class Tables(FixedLoad):
    """H1's tables of a window that its circuit reads where its proof blocks make them (module docstring), a table a round of
    the class's ``outputs`` rounds: the A table of contraction ``k``, ``m`` rows of each of the round's ``calls`` int8
    linears of K = ``k``, call after call, each row H1's leaf (codes four to a u32, then the f32 scale), and the input-ids
    table, ``m`` rows of one u32 id. Their names are the ones the Commit record registers them under (pouw's
    `credit.commit_rows` and `credit.commit_ids`), and a position's rows are at the fixed load's public rows (`at`)."""
    WHAT = "H1's tables"
    k: int
    m: int
    calls: int

    def __post_init__(self) -> None:
        if not all(type(v) is int and v >= 1 for v in (self.k, self.m, self.calls)):
            raise ValueError(f"H1's tables' K, rows a call and calls a round are positive integers, not {self}")

    def a(self, s: int) -> str:
        """Round ``s``'s A table of K = ``k``."""
        return f"row-table/{s}/k{self.k}"

    @staticmethod
    def ids(s: int) -> str:
        """Round ``s``'s input-ids table."""
        return f"input-ids/{s}"


@dataclass(frozen=True)
class PearlTables(FixedLoad):
    """Pearl+'s tables of a window, which its circuit cut at its linears reads (module docstring), each call's as Pearl+'s
    call of it commits them: round s's call c, the c-th cut linear of the round's forward, has its A rows (x widened to FP32,
    K words a row; `linear_cut.LinearARow`), its `rows` rows then zero rows up to `a_rows`, and its y (each BF16 word widened
    to FP32, N words a row; `linear_cut.LinearOut`), its `rows` rows. A cut linear hands its y on to its readers in rows of
    ``words``, doubled where its unit would be no proof block (`window_cut.linears`). ``scheme`` is the PoUW scheme the
    window's Pearl+ serves under, whose credited product a tie would hold y to (`window_cut.census`): the deployment
    states it, since it names the device."""
    WHAT = "Pearl+'s tables"
    scheme: str
    words: int = 64

    def __post_init__(self) -> None:
        if type(self.words) is not int or self.words < 1:
            raise ValueError(f"a cut linear hands its y on in rows of a positive number of words, not {self.words!r}")
        if self.scheme not in SCHEMES:
            raise ValueError(f"{self.scheme!r} is no PoUW scheme ({', '.join(SCHEMES)})")

    @staticmethod
    def a(s: int, c: int) -> str:
        """Round ``s``'s call ``c``'s A rows, by the name Pearl+'s window record registers them under."""
        return PWR.a_value(s, c)

    @staticmethod
    def y(s: int, c: int) -> str:
        """Round ``s``'s call ``c``'s y, by the name Pearl+'s window record registers it under."""
        return PWR.y_value(s, c)

    @staticmethod
    def b(w: int) -> str:
        """Weight ``w``'s B rows, by the name Pearl+'s window record registers them under."""
        return PWR.b_value(w)

    @staticmethod
    def rows(cls: WindowClass, s: int) -> int:
        """A round-``s`` call's rows under the fixed load: round 0 every slot's prompt, a later round a row a slot."""
        return cls.slots * cls.prompt if s == 0 else cls.slots

    @staticmethod
    def a_rows(cls: WindowClass, s: int) -> int:
        """A round-``s`` call's A rows: its `rows` and then zero rows up to a multiple of Pearl+'s `ROW_ALIGN`, as the device
        stages and commits a call's A (its m₁₆)."""
        return -(-PearlTables.rows(cls, s) // ROW_ALIGN) * ROW_ALIGN


def takes(compiled: Any, cls: WindowClass, placed: Sequence[tuple[int, int | None]] | None = None,
          tables: FixedLoad | None = None) -> None:
    """Refuse a window ``cls``'s slots don't take (`window_circuit`); with ``tables``, one whose rows by step aren't the
    fixed load's (`FixedLoad.schedule`), the public rows its tables are read at."""
    units = [u for u in compiled.requests if u is not None]
    if len(units) > cls.slots:
        raise Unexpressible(f"{len(units)} requests in a window of a class of {cls.slots} slots")
    placed = [(cls.boundaries[0], None)] * len(units) if placed is None else list(placed)
    if len(placed) != len(units):
        raise ValueError(f"{len(placed)} placements for the window's {len(units)} requests")
    anchored: dict[int, int] = {}
    for u, (anchor, end) in zip(units, placed):
        if anchor not in cls.boundaries:
            raise Unexpressible(f"request {u.request}: anchored at bucket {anchor}, none of its class's boundaries "
                                f"{list(cls.boundaries)}")
        if end is not None and end < anchor:
            raise Unexpressible(f"request {u.request}: its last output at bucket {end}, before its anchor {anchor}")
        anchored[anchor] = anchored.get(anchor, 0) + 1
    for anchor, n in sorted(anchored.items()):
        if n > cls.lanes:
            raise Unexpressible(f"{n} requests anchored at bucket {anchor}, past the {cls.lanes} slots a boundary has")
    for u in units:
        if u.carried_in or u.carried_out or u.C or u.parse is not None:
            raise Unexpressible(f"request {u.request}: carried across windows, a prefix-cache hit or byte ingress, which the "
                                "window's lowering doesn't take yet")
        if u.policy != GREEDY:
            raise Unexpressible(f"request {u.request}: a {u.policy.get('kind')} request, and the class Program is greedy (a "
                                "Program that reads a seed isn't lowered yet)")
        if cls.prompt < u.LP or cls.outputs < u.T + 1:
            raise Unexpressible(f"request {u.request}: {u.LP} prompt ids and {u.T + 1} outputs, past its class's {cls.prompt} and "
                                f"{cls.outputs}")
    if tables is not None and (got := [[(r.request, r.start, r.n) for r in step] for step in compiled.steps]) \
            != tables.schedule(cls):
        raise Unexpressible(f"the window's rows by step {got} (request, start, n), not its class's fixed load "
                            f"{tables.schedule(cls)}, the rows {tables.WHAT} are read at")


__all__ = ["CLASS_OF_RECORD", "FixedLoad", "GREEDY", "PearlTables", "Tables", "WindowClass", "takes"]
