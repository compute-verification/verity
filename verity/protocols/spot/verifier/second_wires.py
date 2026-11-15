"""Three conditions on a two-stage design's wires, decided, as the verifier's `Spot.SecondWires` defines them, which the
difftest's ``second_wires`` holds this to. Round 2's check of the wires (`second_stage`'s ``second-wires``) holds each
wire to the parts, to its rows' units and to the links; these three it does not decide, and no check calls this module.

* `rows_under`: at the rows of each port a wire names, each run's last unit reads a row under the rows' ``per``, so
  every unit the run holds reads one of its writer's units of its own replay block. One lookup of a port's rows a wire
  and one comparison a run, a run being one entry however many units it holds.
* `names_once`: no two parts have one template name.
* `wires_wide`: for each wire, every bit under the ``bits`` of each port of the wire's name in the reader's circuit
  file is one the writer's circuit file writes on the wire's output row (`covers`). Beside the wires it reads each
  template's circuit file by its name, as its `Head`: the fields of the file the Lean reads of its `Flock.Circuit`,
  which round 2's check does not hold. A wire costs one pass over one unit's returned leaves' bits
  (`writes`), each looked up among the output rows; no unit of a batch and no row of the unit net is read, and the work
  follows the lists the file holds, never a number in it.

`decided` is the three together and `failures` each one's refusal, its code and reason the Lean's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from verity.protocols.spot.verifier.second_stage import Own, Part

#: `Flock.WORD`: the bits of a column word of a net.
WORD = 128

ROWS_WHY = "a wire's rows give a unit of a run a row that is not one of its writer's units of the unit's replay block"
NAMES_WHY = "two of the interior program's parts have one template name"
WIDTHS_WHY = ("a wire's reader port has a bit its writer's circuit file does not write on the wire's output row, or the "
              "reader's or the writer's circuit file is not given")


@dataclass(frozen=True)
class Head:
    """What `wires_wide` reads of a circuit file (`Flock.Circuit`): its ports' ``(name, bits)``, the input rows and
    then the output rows (`ports`), its output ports' ``(name, words)`` (`outPorts`), how many of them each output row
    holds (`outRowPorts`), its instances a unit (`upv`), each instance's returned leaves, the output word of each
    (`leavesOut`), and of its unit net the input words (`inWords`), the constant's column (`constPos`) and the first
    output group's column (`col`)."""

    ports: tuple[tuple[str, int], ...]
    out_ports: tuple[tuple[str, int], ...]
    out_row_ports: tuple[int, ...]
    upv: int
    leaves_out: tuple[tuple[int, ...], ...]
    in_words: int
    const_pos: int
    out_col: int


def rows_under(own: Own) -> bool:
    """`SecondWires.rowsUnder`: at the rows of each port a wire names, each run's last unit reads a row under ``per``
    (``row + (count - 1)·step < per``, a run of no unit its first row). A wire whose port has no rows passes: round 2's
    check of the wires refuses it."""
    for w in own.wires or ():
        r = next((r for r in own.rows if r.template == w.template and r.port == w.port), None)
        if r is not None and not all(x.row + max(x.count - 1, 0) * x.step < r.per for x in r.runs):
            return False
    return True


def names_once(parts: Sequence[Part]) -> bool:
    """`SecondWires.namesOnce`: no two parts have one template name."""
    names = [p.template for p in parts]
    return len(set(names)) == len(names)


def out_rows(c: Head) -> list[tuple[str, int, int]]:
    """`SecondWires.outRows`: the circuit file's output rows in order, each one's name (the empty name where the file
    lists no such port), the output word it starts at and its words (`Flock.HmOut.outGroup`)."""
    n_in = max(len(c.ports) - len(c.out_row_ports), 0)
    rows, first = [], 0
    for o, k in enumerate(c.out_row_ports):
        name = c.ports[n_in + o][0] if n_in + o < len(c.ports) else ""
        rows.append((name, sum(w for _, w in c.out_ports[:first]), sum(w for _, w in c.out_ports[first:first + k])))
        first += k
    return rows


def writes(c: Head, port: str) -> list[int]:
    """`SecondWires.writes`: the bits of its output row `port` that one unit of circuit file `c` writes: bit
    ``16 (x - lo) + t`` for bit `t` of a returned leaf `j` of an instance, where the leaf's output word `x` is one of
    the row's, from the row's first word `lo`, and the bit's column ``col·WORD + 16 j + t`` is a computed row of the
    unit net, at or past its inputs (``min(inWords·WORD, constPos)``) and before its constant."""
    rows = out_rows(c)
    n_in = min(c.in_words * WORD, c.const_pos)
    base = c.out_col * WORD
    out = []
    for ls in c.leaves_out[:c.upv]:
        for j, x in enumerate(ls):
            r = next((r for r in rows if r[1] <= x < r[1] + r[2]), None)
            if r is not None and r[0] == port:
                out += [16 * (x - r[1]) + t for t in range(16) if n_in <= base + 16 * j + t < c.const_pos]
    return out


def covers(c: Head, port: str, bits: int) -> bool:
    """`SecondWires.covers`: circuit file `c` writes every bit under `bits` of its output row `port`."""
    written = writes(c, port)
    return bits <= len(written) and len({b for b in written if b < bits}) == bits


def wires_wide(circuits: Sequence[tuple[str, Head]], own: Own) -> bool:
    """`SecondWires.wiresWide`: for each wire, every bit under the ``bits`` of each port of the wire's name in the
    reader's circuit file is one the writer's circuit file writes on the wire's output row. `circuits` names each
    template's circuit file, the first of a name its file; a wire whose reader or writer it names no file for fails."""
    def of(t: str) -> Head | None:
        return next((c for name, c in circuits if name == t), None)

    for w in own.wires or ():
        reader, writer = of(w.template), of(w.writer)
        if reader is None or writer is None:
            return False
        if not all(name != w.port or covers(writer, w.wport, bits) for name, bits in reader.ports):
            return False
    return True


def decided(parts: Sequence[Part], circuits: Sequence[tuple[str, Head]], own: Own) -> bool:
    """`SecondWires.decided`: the three conditions."""
    return rows_under(own) and names_once(parts) and wires_wide(circuits, own)


def failures(parts: Sequence[Part], circuits: Sequence[tuple[str, Head]], own: Own) -> list[tuple[str, str]]:
    """`SecondWires.failures`: ``second-wire-rows`` unless `rows_under`, ``second-part-names`` unless `names_once` and
    ``second-wire-widths`` unless `wires_wide`, each with its reason; none exactly when `decided`."""
    return ([] if rows_under(own) else [("second-wire-rows", ROWS_WHY)]) \
        + ([] if names_once(parts) else [("second-part-names", NAMES_WHY)]) \
        + ([] if wires_wide(circuits, own) else [("second-wire-widths", WIDTHS_WHY)])
