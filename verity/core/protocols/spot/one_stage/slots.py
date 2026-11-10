"""The claims' slots over a circuit file: `Spot.Slots.layout` and the summary it expands,
`Spot.FilePart.ofCircuit`, ported line for line; the difftest holds this to them (`slots`).

A circuit file's key program at `n` units (`Flock.Bridge.progOf` of `Placed.keyProg`): gate 0 the constant, gate 1 the
zero, then `B = upv · W` sources a unit (`W` the unit's input columns), unit `u`'s source `b` gate `2 + u·B + b`, one
position each; then each unit's `m = upv · (constPos − W + 1)` computed rows, three positions each. Unit `u`'s proof
block is its computed rows' positions, then each source it owns: the first stacked input with the key (`src_keys`, Δ's
source of a unit input, `FilePart.srcKey`) of an input column some computed row reads (`RecResiduals.readIns`), the
zero's excepted.

The layout is one class, the population, in the registration's strata (`sizes`), slot `u` unit `u`."""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: the columns of a word (`Flock.WORD`)
WORD = 128
#: the block keyword of a circuit file's circuits (`Tags.blockKeyword`)
BLOCK = "CIRCUIT"


@dataclass(frozen=True)
class Net:
    """A circuit's header (`Flock.Net`): its constant column, its port groups, `(col, words, bits)` each, and (the unit's
    only) its rows' A and B columns."""

    const_pos: int
    in_groups: tuple[tuple[int, int, tuple[int, ...]], ...]
    out_groups: tuple[tuple[int, int, tuple[int, ...]], ...]
    rows: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = ()

    @property
    def in_words(self) -> int:
        return self.in_groups[-1][0] + self.in_groups[-1][1] if self.in_groups else 0


def _groups(h: list[str], at: int) -> tuple[tuple[tuple[int, int, tuple[int, ...]], ...], int]:
    """The port groups from header field `at` (`Net.parse`'s `groups`), and the field after them."""
    count, at, out = int(h[at]), at + 1, []
    for _ in range(count):
        col, words, n = int(h[at]), int(h[at + 1]), int(h[at + 2])
        out.append((col, words, tuple(int(x) for x in h[at + 3:at + 3 + n])))
        at += 3 + n
    return tuple(out), at


def port_cols(groups: Iterable[tuple[int, int, tuple[int, ...]]]) -> list[tuple[int, int]]:
    """`Flock.portCols`: each port's first column and width, groups in order."""
    out = []
    for col, _words, bits in groups:
        at = col * WORD
        for w in bits:
            out.append((at, w))
            at += w
    return out


def read_circuit(path: Path) -> tuple[dict, dict[str, Net]]:
    """A circuit file's META and its circuits by name, the first of a name, the `unit`'s with its rows."""
    nets: dict[str, Net] = {}
    with open(path) as f:
        f.readline()
        line = f.readline()
        if not line.startswith("META "):
            raise ValueError("circuit: no META line")
        meta = json.loads(line[5:])
        while line := f.readline():
            kw, name, count = line.split()
            if kw != BLOCK:
                raise ValueError(f"circuit: expected a {BLOCK} line")
            h = f.readline().split()
            ins, at = _groups(h, 4)
            outs, _ = _groups(h, at)
            useful, const_pos = int(h[2]), int(h[3])
            rows = []
            for i in range(int(count) - 1):
                row = f.readline()
                if name == "unit" and name not in nets and i < useful:
                    v = [int(x) for x in row.split()]
                    na = v[0]
                    rows.append((tuple(v[1:1 + na]), tuple(v[2 + na:2 + na + v[1 + na]])))
            nets.setdefault(name, Net(const_pos, ins, outs, tuple(rows)))
    if "unit" not in nets:
        raise ValueError("circuit: no unit circuit")
    return meta, nets


def _port_bits(meta: dict) -> tuple[list[int], list[int]]:
    """Each row port's first word among the message's words (`KnownRead.starts`) and its row's bits (`Port.bits`)."""
    starts, at = [], 0
    for p in meta["ports"]:
        starts.append(at)
        at += p[1]
    schemas = meta.get("row_schemas")
    return starts, [s[1] for s in schemas] if schemas else [16 * p[1] for p in meta["ports"]]


def last_le(s: Sequence[int], x: int) -> int:
    """`FilePart.lastLe`: the last index of the ascending `s` whose entry is at most `x`, or 0."""
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi) // 2
        if (s[mid] if mid < len(s) else 0) <= x:
            lo = mid + 1
        else:
            hi = mid
    return max(lo - 1, 0)


def src_keys(meta: dict, nets: dict[str, Net]) -> list[tuple]:
    """`FilePart.srcKey` of every stacked unit input `x < B`, in input order: its `Index.of` (the ports' first words,
    each `(copy, port)`'s first leaf cut and its first wire into the unit), `msgAt` and `srcAt`."""
    unit = nets["unit"]
    W = unit.in_words * WORD
    starts, bits = _port_bits(meta)
    pc = port_cols(unit.in_groups)
    leaves = meta.get("leaves_in") or []
    cuts: dict[tuple[int, int], int] = {}
    for _net, k, p, w in meta.get("leaf_cuts") or []:
        cuts.setdefault((k, p), w)
    wires: dict[tuple[int, int], tuple[str, int, int]] = {}
    for dn, di, dp, sn, si, sp in meta.get("wires") or []:
        if dn == "unit":
            wires.setdefault((di, dp), (sn, si, sp))
    leaf_ports = len(unit.in_groups[0][2]) if unit.in_groups else 0

    def msg_at(x: int, t: int, wide: bool) -> tuple:
        q = last_le(starts, x)
        s = starts[q] if q < len(starts) else 0
        if (wide or t < 16) and 16 * (x - s) + t < (bits[q] if q < len(bits) else 0):
            return ("msg", q, 2 * (x - s) + t // 8, t % 8)
        return ("zero",)

    def src_at(k: int, p: int, t: int) -> tuple:
        if p < leaf_ports:
            li = leaves[k] if k < len(leaves) else []
            leaf = li[p] if p < len(li) else 0
            return ("zero",) if leaf < 0 else msg_at(leaf, t, True)
        if (k, p) in cuts:
            return msg_at(cuts[k, p], t, False)
        if (k, p) in wires:
            net, slot, j = wires[k, p]
            if net not in nets:
                raise ValueError(f"a wire from {net}, a circuit the file doesn't hold")
            cols = port_cols(nets[net].out_groups)
            return ("wire", net, slot, (cols[j][0] if j < len(cols) else 0) + t)
        return ("lit", 0)

    def key(x: int) -> tuple:
        off = x % W
        p = next((p for p, (lo, w) in enumerate(pc) if lo <= off < lo + w), None)
        return ("pad", x // W, off) if p is None else src_at(x // W, p, off - pc[p][0])

    return [key(x) for x in range(meta["units_per_vu"] * min(W, unit.const_pos))]


def read_ins(unit: Net) -> list[bool]:
    """`RecResiduals.readIns`: the input columns the unit's computed rows `nIn … constPos − 1` read, in A and in B."""
    n_in = min(unit.in_words * WORD, unit.const_pos)
    rd = [False] * n_in
    for a, b in unit.rows[n_in:unit.const_pos]:
        for c in (*a, *b):
            if c < n_in:
                rd[c] = True
    return rd


@dataclass(frozen=True)
class FilePart:
    """`Spot.FilePart.FilePart`: the units, each unit's `B` source columns and `m` computed rows, and the
    source columns a unit owns, ascending."""

    n: int
    B: int
    m: int
    srcs: tuple[int, ...]

    @property
    def in_gates(self) -> int:
        return 2 + self.n * self.B

    @property
    def size(self) -> int:
        return self.in_gates + 3 * self.n * self.m

    def block(self, u: int) -> list[tuple[int, int]]:
        """`Slots.blockOf`: `FilePart.block` as intervals, its rows' lowered gates then each source it owns."""
        return ([(self.in_gates + 3 * self.m * u, self.in_gates + 3 * self.m * (u + 1))]
                + [(2 + u * self.B + b, 2 + u * self.B + b + 1) for b in self.srcs])


def of_circuit(circuit: Path, n: int) -> FilePart:
    """`FilePart.ofCircuit`: the circuit file's key program at `n` units."""
    meta, nets = read_circuit(circuit)
    unit, upv = nets["unit"], meta["units_per_vu"]
    n_in = min(unit.in_words * WORD, unit.const_pos)
    B = upv * n_in
    rd = read_ins(unit)
    keys = src_keys(meta, nets)
    first: dict[tuple, int] = {}
    for x in range(B):
        first.setdefault(keys[x], x)
    owned = [False] * B
    for x in range(B):
        if rd[x % n_in] and keys[x] != ("zero",):
            owned[first.get(keys[x], x)] = True
    return FilePart(n, B, upv * (unit.const_pos - n_in + 1), tuple(b for b in range(B) if owned[b]))


def sizes(strata: Iterable[tuple[str, Sequence[tuple[int, int]]]]) -> list[int] | None:
    """`FilePart.sizes`: the strata's sizes (`partition.strata`), when their units read stratum by stratum are
    `0, 1, …` in order; None when they are not."""
    at, out = 0, []
    for _template, ranges in strata:
        lo = at
        for a, b in ranges:
            if a != at or b < a:
                return None
            at = b
        out.append(at - lo)
    return out


def layout(f: FilePart, ns: Sequence[int]) -> list[list[list[dict[str, Any]]]]:
    """`Slots.layout`'s JSON: one class, the strata consecutive runs of slots of sizes `ns`, slot `u` unit `u`'s proof
    block `{"unit", "pus"}`."""
    blocks = [{"pus": [list(iv) for iv in f.block(u)], "unit": u} for u in range(f.n)]
    pop, at = [], 0
    for k in ns:
        pop.append(blocks[at:at + k])
        at += k
    return [pop]


def checked(classes: list[list[list[dict[str, Any]]]], proved: Iterable[int]) -> list[list[int]]:
    """`Slots.checked`: `[slot, 0]` of every proof block whose unit is proved, in slot order."""
    proved = set(proved)
    blocks = [b for c in classes for s in c for b in s]
    return [[v, 0] for v, b in enumerate(blocks) if b["unit"] in proved]
