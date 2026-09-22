"""The exported layered circuit (the text format of ``backends/gkr/src/circuit.rs``), its GKR layers as dense
wiring matrices, and the batched wire evaluation on the device.

Values are BabyBear (the export of ``bb_export.py`` reduces every coefficient mod ``p``).  ``layers()`` is the
Rust ``layers()``: layer 0 is the assertion layer over ``V^(D)`` (every wire), layer ``k`` computes ``V^(k)``
(every wire of depth ``<= k``, deeper wires zero) from ``V^(k-1)``; ``V^(0)`` is the column vector.  A gate is
``(Σ_l A[g,l] V_in(l)) · (Σ_l B[g,l] V_in(l))`` and the two dense matrices ``A``, ``B`` ``(g_n, 2^s_in)`` are what
the prover contracts the input rows with (``mm_mod``), so the per-layer wiring is data-independent and shared by
all copies (Thaler §7 preprocessing).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import torch

from .field import Limbs, P, mm_mod

R16_ROWS = 65536


def log2ceil(n: int) -> int:
    s = 0
    while (1 << s) < n:
        s += 1
    return s


@dataclass
class Lin:
    terms: list[tuple[int, int]]   # (index, coef)
    konst: int = 0


@dataclass
class Table:
    name: str
    cols: int
    rows: list[list[int]] = field(default_factory=list)


@dataclass
class Query:
    table: str
    cols: list[Lin]                 # linear forms over *columns*


@dataclass
class Circuit:
    ncols: int
    col_names: list[str]
    one_col: int
    wires: list[tuple]              # ("in", col) | ("prod", depth, Lin, Lin)   (Lin over wires)
    asserts: list[tuple[str, Lin]]  # over wires
    tables: list[Table]
    queries: list[Query]
    max_depth: int
    hash: bytes

    @property
    def nwires(self) -> int:
        return len(self.wires)

    def one_wire(self) -> int:
        for i, w in enumerate(self.wires):
            if w[0] == "in" and w[1] == self.one_col:
                return i
        raise ValueError("no constant-one wire")

    def table(self, name: str) -> Table:
        for t in self.tables:
            if t.name == name:
                return t
        raise KeyError(name)

    def col_index(self, name: str) -> int:
        return self.col_names.index(name)

    def depth_of(self) -> list[int]:
        return [0 if w[0] == "in" else w[1] for w in self.wires]


def _parse_lin(toks: list[str], pos: int) -> tuple[Lin, int]:
    n = int(toks[pos])
    pos += 1
    terms = []
    for _ in range(n):
        terms.append((int(toks[pos]), int(toks[pos + 1]) % P))
        pos += 2
    konst = int(toks[pos]) % P
    return Lin(terms, konst), pos + 1


def parse_circuit(text: str) -> Circuit:
    ncols = 0
    col_names: list[str] = []
    one_col = 0
    prods: dict[int, tuple] = {}
    nwires = 0
    asserts = []
    tables: list[Table] = []
    queries = []
    max_depth = 0
    for line in text.splitlines():
        toks = line.split()
        if not toks:
            continue
        kind = toks[0]
        if kind == "ncols":
            ncols = int(toks[1])
            col_names = [""] * ncols
        elif kind == "nwires":
            nwires = int(toks[1])
        elif kind == "one":
            one_col = int(toks[1])
        elif kind == "col":
            col_names[int(toks[1])] = toks[2]
        elif kind == "prod":
            w, depth = int(toks[1]), int(toks[2])
            a, pos = _parse_lin(toks, 3)
            b, _ = _parse_lin(toks, pos)
            max_depth = max(max_depth, depth)
            prods[w] = ("prod", depth, a, b)
        elif kind == "assert":
            l, _ = _parse_lin(toks, 2)
            asserts.append((toks[1], l))
        elif kind == "table":
            tables.append(Table(toks[1], int(toks[2])))
        elif kind == "row":
            tables[-1].rows.append([int(v) % P for v in toks[1:]])
        elif kind == "query":
            n = int(toks[2])
            pos = 3
            cols = []
            for _ in range(n):
                l, pos = _parse_lin(toks, pos)
                cols.append(l)
            queries.append(Query(toks[1], cols))
        else:
            raise ValueError(f"unknown line kind {kind}")
    wires: list[tuple] = []
    next_col = 0
    for w in range(nwires):
        if w in prods:
            wires.append(prods.pop(w))
        else:
            wires.append(("in", next_col))
            next_col += 1
    if next_col != ncols:
        raise ValueError("column/wire mismatch")
    tables.append(Table("R16", 1, [[v] for v in range(R16_ROWS)]))
    return Circuit(ncols, col_names, one_col, wires, asserts, tables, queries, max_depth, hashlib.sha256(text.encode()).digest())


def load_circuit(path) -> Circuit:
    with open(path) as f:
        return parse_circuit(f.read())


# ---------------------------------------------------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Layer:
    name: str
    s_out: int
    s_in: int
    gates: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]]   # (a terms, b terms) over the input index space
    _mats: dict = field(default_factory=dict, repr=False)
    _mats_t: dict = field(default_factory=dict, repr=False)

    @property
    def g_n(self) -> int:
        return len(self.gates)

    def matrices(self, device) -> tuple[torch.Tensor, torch.Tensor]:
        """Dense ``A``, ``B`` ``(g_n, 2^s_in)`` int64 on ``device`` (cached)."""
        m = self._mats.get(device)
        if m is None:
            n_in = 1 << self.s_in
            A = torch.zeros((self.g_n, n_in), dtype=torch.int64)
            B = torch.zeros((self.g_n, n_in), dtype=torch.int64)
            for g, (ta, tb) in enumerate(self.gates):
                for idx, c in ta:
                    A[g, idx] = (A[g, idx] + c) % P
                for idx, c in tb:
                    B[g, idx] = (B[g, idx] + c) % P
            m = (A.to(device), B.to(device))
            self._mats[device] = m
        return m

    def eval_gates(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``a[c, g] = Σ_l A[g,l] rows[c,l]`` and the same for ``B`` (``rows`` ``(copies, 2^s_in)`` base)."""
        lim = self._mats_t.get(rows.device)
        if lim is None:
            A, B = self.matrices(rows.device)
            lim = (Limbs(A.T.contiguous()), Limbs(B.T.contiguous()))
            self._mats_t[rows.device] = lim
        r = Limbs(rows)
        return mm_mod(r, lim[0]), mm_mod(r, lim[1])

    def eval_a_at(self, el: torch.Tensor) -> torch.Tensor:
        """``Ã_g(l*) = Σ_l A[g,l] eq(l*, l)`` for every gate, ``el`` ``(2^s_in, 6)`` -> ``(g_n, 6)``."""
        A, _ = self.matrices(el.device)
        return mm_mod(A, el)

    def eval_b_at(self, er: torch.Tensor) -> torch.Tensor:
        _, B = self.matrices(er.device)
        return mm_mod(B, er)


def _lin_terms(l: Lin, one: int, remap) -> list[tuple[int, int]]:
    v = [(remap(i), c) for i, c in l.terms]
    if l.konst:
        v.append((one, l.konst))
    return v


def layers(c: Circuit) -> list[Layer]:
    nw = c.nwires
    s_w = log2ceil(nw)
    s_c = log2ceil(c.ncols)
    one_w = c.one_wire()
    out = []
    s_a = log2ceil(max(1, len(c.asserts)))
    gates = [(_lin_terms(l, one_w, lambda i: i), [(one_w, 1)]) for _, l in c.asserts]
    out.append(Layer("assertions", s_a, s_w, gates))
    for k in range(c.max_depth, 0, -1):
        from_cols = k == 1
        s_in = s_c if from_cols else s_w

        def remap(i, from_cols=from_cols):
            if from_cols:
                w = c.wires[i]
                if w[0] != "in":
                    raise ValueError("depth-1 product reads a product")
                return w[1]
            return i

        one_in = c.one_col if from_cols else one_w
        gates = []
        for w, wire in enumerate(c.wires):
            if wire[0] == "in":
                gates.append(([(wire[1] if from_cols else w, 1)], [(one_in, 1)]))
            else:
                _, depth, a, b = wire
                if depth == k:
                    gates.append((_lin_terms(a, one_in, remap), _lin_terms(b, one_in, remap)))
                elif depth < k:
                    gates.append(([(w, 1)], [(one_in, 1)]))
                else:
                    gates.append(([], []))
        out.append(Layer(f"depth{k}", s_w, s_in, gates))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Wires on the device
# ---------------------------------------------------------------------------------------------------------------------

def eval_lin_rows(l: Lin, vals: torch.Tensor) -> torch.Tensor:
    """A linear form on a ``(units, width)`` base matrix -> ``(units,)``."""
    acc = torch.full((vals.shape[0],), l.konst % P, dtype=torch.int64, device=vals.device)
    for idx, c in l.terms:
        acc = (acc + vals[:, idx] * c) % P
    return acc


def eval_wires(c: Circuit, cols: torch.Tensor) -> torch.Tensor:
    """Every wire of every unit: ``cols`` ``(units, ncols)`` int64 -> ``(units, nwires)`` int32."""
    units = cols.shape[0]
    wires = torch.empty((units, c.nwires), dtype=torch.int64, device=cols.device)
    for w, wire in enumerate(c.wires):
        if wire[0] == "in":
            wires[:, w] = cols[:, wire[1]]
        else:
            _, _, a, b = wire
            wires[:, w] = (eval_lin_rows(a, wires) * eval_lin_rows(b, wires)) % P
    return wires.to(torch.int32)


def masked_wires(wires: torch.Tensor, depth_of: torch.Tensor, k_in: int) -> torch.Tensor:
    """The layer-``k_in`` view: every wire of depth ``> k_in`` zeroed."""
    mask = (depth_of <= k_in).to(wires.dtype)
    return wires * mask.unsqueeze(0)


def pad_rows(rows: torch.Tensor, copies: int, width: int) -> torch.Tensor:
    """Zero-pad ``(units, w)`` to ``(copies, width)`` int64."""
    out = torch.zeros((copies, width), dtype=torch.int64, device=rows.device)
    out[: rows.shape[0], : rows.shape[1]] = rows
    return out
