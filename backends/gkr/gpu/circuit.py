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

from . import field as _field
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
    """A public lookup table: row-listed (``rows``), an implicit range table ``R<w>`` (``range_width``), or a v2
    implicit table materialised from its ``itable`` definition (``implicit = (kind, params)``, PROTOCOL.md 14.2).
    ``nrows`` is the row count in every case; ``device_rows`` materialises the ``(nrows, cols)`` int64 matrix on a
    device (cached) so the 3.5M-row ``SHIFT`` table never exists as a Python list."""
    name: str
    cols: int
    rows: list[list[int]] = field(default_factory=list)
    range_width: int | None = None
    implicit: tuple[str, list[int]] | None = None
    _nrows: int = 0
    _dev_rows: dict = field(default_factory=dict, repr=False)

    @property
    def nrows(self) -> int:
        if self.range_width is not None:
            return 1 << self.range_width
        if self.implicit is not None:
            return self._nrows
        return len(self.rows)

    @property
    def dense(self) -> bool:
        """Key = row index (range tables and the v2 implicit tables)."""
        return self.range_width is not None or self.implicit is not None

    def device_rows(self, device) -> torch.Tensor:
        key = str(device)
        t = self._dev_rows.get(key)
        if t is None:
            if self.range_width is not None:
                t = torch.arange(self.nrows, dtype=torch.int64, device=device).unsqueeze(1)
            elif self.implicit is not None:
                t = materialise(self.implicit[0], self.implicit[1], self.nrows, device)
            else:
                t = torch.tensor(self.rows, dtype=torch.int64, device=device).reshape(len(self.rows), self.cols)
            self._dev_rows[key] = t
        return t


def range_table_width(name: str) -> int | None:
    """``R<w>``, ``1 <= w <= 16``: the implicit exact-width range table (``circuit.rs::range_width``)."""
    if not name.startswith("R"):
        return None
    try:
        w = int(name[1:])
    except ValueError:
        return None
    return w if 1 <= w <= 16 else None


def materialise(kind: str, params: list[int], rows: int, device) -> torch.Tensor:
    """``circuit.rs::materialise`` on the device (PROTOCOL.md 14.2), ``(rows, 2)`` int64 canonical mod ``p``:
    ``shift <bits> <scale> <R>``: key ``x R + 2 d + s`` -> ``(1 - 2 s) floor(x 2^scale / 2^d)``, ``x < 2^bits``;
    ``tnorm <vb> <k_rows> <k_lo>``: key ``v + 2^vb koff`` -> ``floor(v 2^(koff + k_lo))``.  Every intermediate fits
    int64 (``x 2^scale < 2^29`` for the v2 parameters; the tnorm shift is at most ``2^(15 + 24)``)."""
    key = torch.arange(rows, dtype=torch.int64, device=device)
    if kind == "shift":
        bits, scale, R = params
        if rows != (1 << bits) * R:
            raise ValueError("shift table row count")
        x, rem = key // R, key % R
        d, s = rem // 2, rem % 2
        v = (x << scale) >> d
        val = torch.where(s == 1, (-v) % P, v % P)
    elif kind == "tnorm":
        vb, k_rows, k_lo = params
        if rows != (1 << vb) * k_rows:
            raise ValueError("tnorm table row count")
        koff, v = key >> vb, key & ((1 << vb) - 1)
        k = koff + k_lo
        val = torch.where(k >= 0, v << k.clamp(min=0), v >> (-k).clamp(min=0)) % P
    else:
        raise ValueError(f"unknown implicit table kind {kind}")
    return torch.stack([key, val], dim=1)


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
        elif kind == "itable":
            # itable NAME cols rows kind params...  (v2, PROTOCOL.md 14.2): materialised on the device on first use
            tables.append(Table(toks[1], int(toks[2]), implicit=(toks[4], [int(v) for v in toks[5:]]), _nrows=int(toks[3])))
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
    # the range tables R<w> = [0, 2^w) are implicit: every one the queries reference, narrowest first
    # (``circuit.rs``; the v1 circuits reference R16 only, so their table list is unchanged)
    widths = sorted({w for q in queries if (w := range_table_width(q.table)) is not None})
    for w in widths:
        tables.append(Table(f"R{w}", 1, range_width=w))
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
    _csr: dict = field(default_factory=dict, repr=False)

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
        dev = rows.device
        if _field._fast_gate_eval is not None and rows.is_cuda:
            csr = self._csr.get(dev)
            if csr is None:
                csr = tuple(self._csr_side(side, dev) for side in (0, 1))
                self._csr[dev] = csr
            return tuple(_field._fast_gate_eval(rows, *s, self.g_n) for s in csr)
        lim = self._mats_t.get(dev)
        if lim is None:
            A, B = self.matrices(dev)
            lim = (Limbs(A.T.contiguous()), Limbs(B.T.contiguous()))
            self._mats_t[dev] = lim
        r = Limbs(rows)
        return mm_mod(r, lim[0]), mm_mod(r, lim[1])

    def _csr_side(self, side: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CSR (by gate) of one side's linear forms: ``ptr`` (g_n + 1) int32, ``idx`` int32, ``coef R mod p`` int64."""
        ptr, idx, coef = [0], [], []
        for ta, tb in self.gates:
            acc: dict[int, int] = {}
            for i, c in (ta, tb)[side]:
                acc[i] = (acc.get(i, 0) + c) % P
            for i, c in sorted(acc.items()):
                idx.append(i)
                coef.append(c * (1 << 32) % P)
            ptr.append(len(idx))
        return (torch.tensor(ptr, dtype=torch.int32, device=device), torch.tensor(idx, dtype=torch.int32, device=device),
                torch.tensor(coef, dtype=torch.int64, device=device))

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
