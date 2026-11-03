"""The claims' slots over a circuit file: `SampledProofs.Slots.layout` and `SampledProofs.FileUnits`, which the difftest
holds this to (`slots`).

A circuit file's key program at `n` units (`Flock.Bridge.progOf` of `Placed.keyProg`): gate 0 the constant, gate 1 the
zero, then `B = upv · W` sources a unit (`W` the unit's input columns), unit `u`'s source `b` gate `2 + u·B + b`, one
position each; then each unit's `m = upv · (constPos − W + 1)` computed rows, three positions each. Unit `u`'s proof
block is its computed rows' positions, then each source it owns: the first stacked input with the key (`src_key`, Δ's
source of a unit input) of an input column some computed row reads, the zero's excepted.

The layout is one class, the population, in the registration's strata (`strata_sizes`), slot `u` unit `u`."""
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
    """Each row port's first word among the message's words (`KnownRead.starts`) and its row's bits."""
    starts, at = [], 0
    for p in meta["ports"]:
        starts.append(at)
        at += p[1]
    schemas = meta.get("row_schemas")
    return starts, [s[1] for s in schemas] if schemas else [16 * p[1] for p in meta["ports"]]


def _port_at(starts: Sequence[int], x: int) -> int:
    return max((q for q in range(len(starts)) if starts[q] <= x), default=0)


def src_keys(meta: dict, nets: dict[str, Net]) -> list[tuple]:
    """Δ's source key of every stacked unit input (`FileUnits.srcKey`, `Aliased.srcKey`), in input order."""
    unit = nets["unit"]
    W = unit.in_words * WORD
    starts, bits = _port_bits(meta)
    pc = port_cols(unit.in_groups)
    mask = {lo + i for lo, w in pc for i in range(w)}
    leaves = meta.get("leaves_in") or []
    cuts = meta.get("leaf_cuts") or []
    wires = meta.get("wires") or []
    leaf_ports = len(unit.in_groups[0][2]) if unit.in_groups else 0

    def msg(x: int, t: int, wide: bool) -> tuple:
        q = _port_at(starts, x)
        start = starts[q] if q < len(starts) else 0
        if (wide or t < 16) and 16 * (x - start) + t < (bits[q] if q < len(bits) else 0):
            return ("msg", q, 2 * (x - start) + t // 8, t % 8)
        return ("zero",)

    def src_at(k: int, p: int, t: int) -> tuple:
        if p < leaf_ports:
            li = leaves[k] if k < len(leaves) else []
            leaf = li[p] if p < len(li) else 0
            return ("zero",) if leaf < 0 else msg(leaf, t, True)
        for e in cuts:
            if e[1] == k and e[2] == p:
                return msg(e[3], t, False)
        for w in wires:
            if w[0] == "unit" and w[1] == k and w[2] == p:
                if w[3] not in nets:
                    raise ValueError(f"a wire from {w[3]}, a circuit the file doesn't hold")
                cols = port_cols(nets[w[3]].out_groups)
                return ("wire", w[3], w[4], (cols[w[5]][0] if w[5] < len(cols) else 0) + t)
        return ("lit", 0)

    def key(x: int) -> tuple:
        off = x % W
        if off in mask:
            p = next((p for p, (lo, w) in enumerate(pc) if lo <= off < lo + w), 0)
            return src_at(x // W, p, off - pc[p][0])
        return ("pad", x // W, off)

    return [key(x) for x in range(meta["units_per_vu"] * min(W, unit.const_pos))]


@dataclass(frozen=True)
class FileUnits:
    """`SampledProofs.FileUnits` (`FilePart`'s fields): the units, each unit's sources and computed rows, and the sources
    a unit owns, ascending."""

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
        own = [(self.in_gates + 3 * self.m * u, self.in_gates + 3 * self.m * (u + 1))] if self.m else []
        return own + [(2 + u * self.B + b, 2 + u * self.B + b + 1) for b in self.srcs]


def file_units(circuit: Path, n: int) -> FileUnits:
    """`FileUnits.ofCircuit`: a circuit file's key program at `n` units."""
    meta, nets = read_circuit(circuit)
    unit, upv = nets["unit"], meta["units_per_vu"]
    n_in = min(unit.in_words * WORD, unit.const_pos)
    keys = src_keys(meta, nets)
    first: dict[tuple, int] = {}
    for i, k in enumerate(keys):
        first.setdefault(k, i)
    read = {c for a, b in unit.rows[n_in:unit.const_pos] for c in (*a, *b) if c < n_in}
    owned = {first[keys[k * n_in + x]] for k in range(upv) for x in read if keys[k * n_in + x] != ("zero",)}
    return FileUnits(n, upv * n_in, upv * (unit.const_pos - n_in + 1), tuple(sorted(owned)))


def strata_sizes(strata: Iterable[tuple[str, Sequence[tuple[int, int]]]]) -> list[int]:
    """`Slots.strataSizes`: the sizes of the strata (`partition.strata`), each range starting where the last ended."""
    at, ns = 0, []
    for _template, ranges in strata:
        lo = at
        for a, b in ranges:
            if a != at or b < a:
                raise ValueError("the strata, read stratum by stratum, are not the units 0, 1, … in order")
            at = b
        ns.append(at - lo)
    return ns


def layout(f: FileUnits, ns: Sequence[int]) -> list[list[list[dict[str, Any]]]]:
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
