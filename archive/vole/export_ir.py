"""The WP2 checker as a SIEVE Circuit-IR relation for Diet Mac'n'Cheese, **v2: the compressed realisation**.

    python archive/vole/export_ir.py census [--field F] [--chunk-bits W]      gate / committed-element counts vs v1
    python archive/vole/export_ir.py unit  OUT [--n N] [--field F]           N random finite units, one relation
    python archive/vole/export_ir.py tu    OUT --root R --n N                the same on the tu-k16 fixture tier
    python archive/vole/export_ir.py vu    OUT --root R --B B                one relation with B chained K=1536 VUs
    python archive/vole/export_ir.py negatives [--root R]                    every WP2 negative and every vu-k1536-neg
                                                                              through the circuit's own evaluator

The relation is the frozen WP2 checker (``verity_numerical.checker``): the same gadgets, the same predicates, called
through a symbolic context exactly as v1 (``export_ir_v1.py``) and ``gkr_export`` do.  What changes is the *realisation*
of the predicates over F_p, p > 2^60, using the ``verity_v0`` plugin added to dietmc (``patches/``):

    eq(a, b)          a - b as a linear form; @assert_zero.  When the form contains a product wire used nowhere else,
                      the product is never committed: ``assert_mul(x, y, c)`` (a QuickSilver degree-2 check with no
                      output wire, c = the rest of the form).  Booleans are eq(b*b, b) -> assert_mul(b, b, b): 0 VOLE.
    rng(x, lo, hi)    y = x - lo in [0, 2^w): y is split into CHUNK_BITS-bit chunks, the top chunk is *linear* in y and
                      the committed lower chunks (no identity to assert); every chunk is a query to the logUp range
                      table [0, 2^CHUNK_BITS) (one committed inverse per query, the table costs 2 x 2^CHUNK_BITS per
                      call, one call per VU).  Non-power-of-two ranges are two-sided as in v1.  Obligations that are
                      implied by other realised ranges through a linear identity (the operand word = s 2^15 + t 2^7 + f;
                      t = (t - nz) + nz) are discharged by interval arithmetic and cost nothing; the evaluator checks
                      them explicitly anyway.
    lookup(T, key)    the outputs become committed hints; (key - lo, outs...) is a logUp query against the table's rows
                      combined with a verifier challenge s (2 x rows committed per call per VU + one inverse per query)
    lt_pow(x, ipow)   one committed product x * ipow, then rng(., 0, 2^B)

The exporter evaluates its own circuit in Python (``evaluate``: linear asserts, output-free multiplication checks,
range queries, table queries) and refuses to write a witness that the circuit rejects for an honest instance, or accepts
for a negative.  The witness stream order is the order of the @private gates (the ``priv`` wires of each unit in wire
order, then the epilogue's, then the claimed word).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
for rel in ("benchmarks/numerical/python", "."):
    p = str(ROOT / rel)
    if p not in sys.path:
        sys.path.insert(0, p)

from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, tc_dot
from verity_numerical.checker import (
    REAL,
    Params,
    Reject,
    check_step,
    gen_step,
)
from verity_numerical.checker import step as step_mod
from verity_numerical.checker.ctx import Table
from verity_numerical.checker.gadgets import Epilogue
from verity_numerical.checker.step import check_step_expr
from verity_numerical.gkr_export.circuit import Lin

FIELDS = {
    "f61p": 2**61 - 1,
    "f127p": 2**127 - 1,
    "f128p": 2**128 - 159,
    "f384p": int("fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffeffffffff0000000000000000ffffffff", 16),
}
UNITS_PER_VU = 96
K_VU = 1536
CHUNK_BITS = 10          # the logUp range table is [0, 2^CHUNK_BITS): 2 x 1024 committed elements per call (per VU)
PLUGIN = "verity_v0"

# v1 (bits + run-one-hot selectors, every boolean an @mul with an output wire), from ASSUMPTIONS.md section 4
V1_CENSUS = {"unit": {"private_wires": 5121, "mul_gates": 5222, "vole_correlations": 10343},
             "vu": {"vole_correlations": 993_095, "mul_gates": 501_393, "bytes_p2v_measured_f61p": 8_500_000}}


# ---------------------------------------------------------------------------------------------------------------------
# The symbolic context
# ---------------------------------------------------------------------------------------------------------------------

class VoleCtx:
    """Collects the checker's predicates as private wires, product wires, zero-assertions, output-free multiplication
    checks and logUp queries.

    ``wires[i]`` is ``("priv", derivation)`` or ``("prod", Lin, Lin)``; ``derivation`` says how the honest prover fills
    the wire: ``("input", name)``, ``("io", name)``, ``("hint", path)``, ``("chunk", Lin, j)`` (bits [cw*j, cw*(j+1))
    of the form), ``("tout", table_name, key_Lin, column)``.  ``finalize`` realises the range obligations and turns
    single-use products into ``assert_mul`` checks; after it, ``virtual[i]`` marks product wires that are never
    committed.
    """

    def __init__(self, p: int, chunk_bits: int = CHUNK_BITS) -> None:
        self.p = p
        self.cw = chunk_bits
        self.wires: list[tuple] = []
        self.asserts: list[tuple[str, Lin]] = []
        self.mulchecks: list[tuple[str, Lin, Lin, Lin]] = []
        self.rqueries: list[tuple[str, Lin]] = []
        self.tqueries: dict[str, list[tuple[str, list[Lin]]]] = {}
        self.tables: dict[str, Table] = {}
        self.pending: list[tuple[str, Lin, int]] = []      # (name, y, w): y in [0, 2^w) still to realise
        self.tight: dict[int, tuple[int, int]] = {}         # pending index -> (partner index, n): two-sided pair
        self.known: list[tuple[Lin, int, int]] = []        # proven ranges (Lin, lo, hi)
        self.discharged: list[tuple[str, int, str]] = []
        self.virtual: set[int] = set()
        self.finalized = False
        self.n_io = self.n_hint = self.n_chunk = self.n_tout = self.n_ltpow = 0
        self.lookups: dict[str, int] = {}

    # -- wires --------------------------------------------------------------------------------------------------------
    def priv(self, derivation: tuple) -> Lin:
        self.wires.append(("priv", derivation))
        kind = derivation[0]
        if kind == "chunk":
            self.n_chunk += 1
        elif kind == "tout":
            self.n_tout += 1
        elif kind == "hint":
            self.n_hint += 1
        else:
            self.n_io += 1
        return Lin(self, {len(self.wires) - 1: 1})

    def product(self, a: Lin, b: Lin) -> Lin:
        self.wires.append(("prod", a, b))
        return Lin(self, {len(self.wires) - 1: 1})

    @property
    def n_prod(self) -> int:
        return sum(1 for w in self.wires if w[0] == "prod")

    @property
    def n_prod_committed(self) -> int:
        return sum(1 for i, w in enumerate(self.wires) if w[0] == "prod" and i not in self.virtual)

    @property
    def n_priv(self) -> int:
        return len(self.wires) - self.n_prod

    # -- predicates ---------------------------------------------------------------------------------------------------
    def _const(self, x) -> int | None:
        if isinstance(x, int):
            return x
        return x.const if x.is_const else None

    def rng(self, name: str, x, lo: int, hi: int) -> None:
        v = self._const(x)
        if v is not None:
            if not lo <= v < hi:
                raise Reject(f"{name}: constant {v} not in [{lo}, {hi})")
            return
        n = hi - lo
        w = max(1, (n - 1).bit_length())
        self.pending.append((name, x - lo, w))
        if n != 1 << w:
            self.pending.append((name + ".hi", (hi - 1) - x, w))
            i = len(self.pending) - 2
            self.tight[i] = (i + 1, n)      # both sides realised  =>  x - lo in [0, n) exactly

    def eq(self, name: str, a, b) -> None:
        d = a - b
        v = self._const(d)
        if v is not None:
            if v != 0:
                raise Reject(f"{name}: constant identity fails")
            return
        self.asserts.append((name, d))

    def boolean(self, name: str, b) -> None:
        v = self._const(b)
        if v is not None:
            if v not in (0, 1):
                raise Reject(f"{name}: constant not a bit")
            return
        self.asserts.append((name, b * b - b))
        self.known.append((b, 0, 2))

    def lookup(self, name: str, table: Table, key) -> tuple:
        v = self._const(key)
        if v is not None:
            return table.row(v)
        self.tables.setdefault(table.name, table)
        self.lookups[table.name] = self.lookups.get(table.name, 0) + 1
        outs = [self.priv(("tout", table.name, key, j)) for j in range(len(table.outs))]
        self.tqueries.setdefault(table.name, []).append((name, [key - table.lo, *outs]))
        return tuple(outs)

    def lt_pow(self, name: str, x, ipow, bits: int) -> None:
        self.n_ltpow += 1
        self.rng(name, x * ipow, 0, 2**bits)

    # -- finalisation ---------------------------------------------------------------------------------------------------
    def table_rows(self, tname: str) -> list[list[int]]:
        t = self.tables[tname]
        return [[k - t.lo, *t.row(k)] for k in range(t.lo, t.hi)]

    def finalize(self) -> None:
        assert not self.finalized
        self.finalized = True
        self._realise_ranges()
        self._fold_products()

    # ranges: discharge by interval arithmetic where a linear identity implies the obligation, else chunk + query
    def _realise_ranges(self) -> None:
        order = sorted(range(len(self.pending)), key=lambda i: -self.pending[i][2])
        realised = [True] * len(self.pending)

        def bound(j: int, cur: int = -1) -> tuple[Lin, int, int]:
            y, w = self.pending[j][1], self.pending[j][2]
            if j in self.tight and self.tight[j][0] != cur and realised[self.tight[j][0]]:
                return (y, 0, self.tight[j][1])
            return (y, 0, 1 << w)

        for i in order:
            name, y, w = self.pending[i]
            partner = self.tight.get(i, (None,))[0]
            others = [bound(j, i) for j in range(len(self.pending)) if j != i and j != partner and realised[j]]
            why = self._implied(y, 0, 1 << w, self.known + others)
            if why is not None:
                realised[i] = False
                self.discharged.append((name, w, why))
        for i, (name, y, w) in enumerate(self.pending):
            if realised[i]:
                self._chunks(name, y, w)
                self.known.append(bound(i))

    @staticmethod
    def _key(x: Lin) -> frozenset:
        return frozenset(x.terms.items())

    def _bounds_of(self, x: Lin, index: dict) -> tuple[int, int] | None:
        """[lo, hi) of ``x`` if it is a known form up to a constant."""
        hit = index.get(self._key(x))
        if hit is None:
            return None
        const, lo, hi = hit
        d = x.const - const
        return lo + d, hi + d

    def _implied(self, y: Lin, lo: int, hi: int, known) -> str | None:
        index = {self._key(L): (L.const, klo, khi) for L, klo, khi in known}
        # (1) the same form, shifted
        b = self._bounds_of(y, index)
        if b is not None and lo <= b[0] and b[1] <= hi:
            return "same form"
        # (2) y = L1 + L2 with both known  (t = (t - nz) + nz)
        wires = set(y.terms)
        for L1, lo1, hi1 in known:
            if not (set(L1.terms) & wires) or self._key(L1) == self._key(y):
                continue
            rest = y - L1
            b2 = self._bounds_of(rest, index)
            if b2 is not None and lo <= lo1 + b2[0] and (hi1 - 1) + (b2[1] - 1) < hi:
                return "sum of two known forms"
        # (3) y is one wire v tied by a linear identity to known forms (the word identity)
        if len(y.terms) == 1 and y.const == 0:
            (v, cv), = y.terms.items()
            if cv == 1:
                for _, d in self.asserts:
                    if v not in d.terms or d.terms[v] not in (1, -1):
                        continue
                    k = d.terms[v]
                    rest = Lin(self, {w: -c * k for w, c in d.terms.items() if w != v}, -d.const * k)   # v = rest
                    lo_s, hi_s = rest.const, rest.const + 1
                    ok = True
                    for w, c in rest.terms.items():
                        bw = self._bounds_of(Lin(self, {w: 1}), index)
                        if bw is None:
                            ok = False
                            break
                        wl, wh = bw[0], bw[1] - 1
                        lo_s += min(c * wl, c * wh)
                        hi_s += max(c * wl, c * wh)
                    if ok and lo <= lo_s and hi_s <= hi:
                        return "linear identity over known forms"
        return None

    def _chunks(self, name: str, y: Lin, w: int) -> None:
        cw = self.cw
        m = max(1, math.ceil(w / cw))
        if m == 1:
            self.rqueries.append((name, y * (1 << (cw - w))))
            return
        acc = Lin(self)
        for j in range(m - 1):
            c = self.priv(("chunk", y, j))
            self.rqueries.append((f"{name}.c{j}", c))
            acc = acc + c * (1 << (cw * j))
        w_top = w - cw * (m - 1)
        inv = pow(1 << (cw * (m - 1)), -1, self.p)
        top = (y - acc) * inv                       # the top chunk, linear in y and the committed chunks
        self.rqueries.append((f"{name}.top", top * (1 << (cw - w_top))))

    # products used only inside one zero-assertion become output-free multiplication checks
    def _fold_products(self) -> None:
        uses: dict[int, int] = {}

        def count(x: Lin) -> None:
            for w in x.terms:
                if self.wires[w][0] == "prod":
                    uses[w] = uses.get(w, 0) + 1

        for wire in self.wires:
            if wire[0] == "prod":
                count(wire[1])
                count(wire[2])
            elif wire[1][0] in ("chunk",):
                count(wire[1][1])
            elif wire[1][0] == "tout":
                count(wire[1][2])
        for _, d in self.asserts:
            count(d)
        for _, q in self.rqueries:
            count(q)
        for qs in self.tqueries.values():
            for _, tup in qs:
                for x in tup:
                    count(x)
        if self.out is not None:
            count(self.out)
        keep: list[tuple[str, Lin]] = []
        for name, d in self.asserts:
            cands = [w for w, c in d.terms.items() if self.wires[w][0] == "prod" and uses.get(w, 0) == 1]
            if not cands:
                keep.append((name, d))
                continue
            P = cands[0]
            k = d.terms[P]
            rest = Lin(self, {w: c for w, c in d.terms.items() if w != P}, d.const)
            inv = pow(k % self.p, -1, self.p)
            c = rest * (-inv)                       # k P + rest = 0  <=>  P = -rest / k
            a, b = self.wires[P][1], self.wires[P][2]
            self.mulchecks.append((name, a, b, c))
            self.virtual.add(P)
        self.asserts = keep

    out: Lin | None = None


# ---------------------------------------------------------------------------------------------------------------------
# Hint paths (as in gkr_export)
# ---------------------------------------------------------------------------------------------------------------------

def _leaves(h, path: str, out: list[str]) -> None:
    if isinstance(h, dict):
        for k, v in h.items():
            _leaves(v, f"{path}.{k}", out)
    elif isinstance(h, list):
        for i, v in enumerate(h):
            _leaves(v, f"{path}[{i}]", out)
    else:
        out.append(path)


def _get(h, path: str):
    cur = h
    for part in path.split(".")[1:]:
        name, *idx = part.split("[")
        cur = cur[name]
        for i in idx:
            cur = cur[int(i[:-1])]
    return cur


def _set(h, path: str, value) -> None:
    parts = path.split(".")[1:]
    cur = h
    for part in parts[:-1]:
        name, *idx = part.split("[")
        cur = cur[name]
        for i in idx:
            cur = cur[int(i[:-1])]
    name, *idx = parts[-1].split("[")
    if not idx:
        cur[name] = value
        return
    cur = cur[name]
    for i in idx[:-1]:
        cur = cur[int(i[:-1])]
    cur[int(idx[-1][:-1])] = value


# ---------------------------------------------------------------------------------------------------------------------
# The unit and the epilogue as circuits
# ---------------------------------------------------------------------------------------------------------------------

class Unit:
    """The transition unit: input wire ``c`` (the incoming FP32 word), private operands and hints, output the outgoing
    FP32 word as a linear form plus the unit's logUp queries (range queries, then one block per table)."""

    def __init__(self, p_field: int, p: Params = REAL, chunk_bits: int = CHUNK_BITS) -> None:
        self.ctx = VoleCtx(p_field, chunk_bits)
        ctx = self.ctx
        k = sum(p.groups)
        self.k = k
        self.c = ctx.priv(("input", "c"))
        a = [ctx.priv(("io", f"a[{i}]")) for i in range(k)]
        b = [ctx.priv(("io", f"b[{i}]")) for i in range(k)]
        sample = gen_step(p, 0, [0] * k, [0] * k)
        self.paths: list[str] = []
        _leaves(sample, "h", self.paths)
        h_sym = copy.deepcopy(sample)
        for path in self.paths:
            _set(h_sym, path, ctx.priv(("hint", path)))
        self.out: Lin = check_step_expr(ctx, p, self.c, a, b, h_sym)
        ctx.out = self.out
        ctx.finalize()
        self.n_private = ctx.n_priv - 1   # minus the input wire c

    def witness(self, c: int, a: Sequence[int], b: Sequence[int], h: dict) -> tuple[list[int], int, str | None]:
        """Values of the private wires in @private order, the output word (mod p) and the first failing check."""
        return _evaluate(self.ctx, {"c": c, "a": a, "b": b, "h": h}, self.out)


class Epi:
    """The epilogue: input wire y32, private hints, output y16 as a linear form plus its range queries."""

    def __init__(self, p_field: int, p: Params = REAL, chunk_bits: int = CHUNK_BITS) -> None:
        self.ctx = VoleCtx(p_field, chunk_bits)
        ctx = self.ctx
        self.y32 = ctx.priv(("input", "y32"))
        sample = Epilogue.gen(p, 0)
        self.paths: list[str] = []
        _leaves(sample, "h", self.paths)
        h_sym = copy.deepcopy(sample)
        for path in self.paths:
            _set(h_sym, path, ctx.priv(("hint", path)))
        self.out: Lin = Epilogue.check(ctx, p, self.y32, h_sym)
        ctx.out = self.out
        ctx.finalize()
        assert not ctx.tqueries
        self.n_private = ctx.n_priv - 1

    def witness(self, y32: int, h: dict) -> tuple[list[int], int, str | None]:
        return _evaluate(self.ctx, {"y32": y32, "h": h}, self.out)


def _evaluate(ctx: VoleCtx, env: dict, out: Lin) -> tuple[list[int], int, str | None]:
    p = ctx.p
    vals: list[int] = [0] * len(ctx.wires)
    mask = (1 << ctx.cw) - 1

    def ev(x: Lin) -> int:
        s = x.const
        for w, c in x.terms.items():
            s += c * vals[w]
        return s % p

    def signed(v: int) -> int:
        return v if v < p // 2 else v - p

    priv: list[int] = []
    for i, wire in enumerate(ctx.wires):
        if wire[0] == "prod":
            vals[i] = ev(wire[1]) * ev(wire[2]) % p
            continue
        d = wire[1]
        kind = d[0]
        if kind == "input":
            v = env[d[1]]
        elif kind == "io":
            v = _get(env, "x." + d[1])
        elif kind == "hint":
            v = _get(env["h"], d[1])
        elif kind == "chunk":
            v = (ev(d[1]) >> (ctx.cw * d[2])) & mask
        else:  # tout
            _, tname, key, col = d
            kv = signed(ev(key))
            t = ctx.tables[tname]
            v = t.row(kv)[col] if t.lo <= kv < t.hi else 0
        v = int(v) % p
        vals[i] = v
        if kind != "input":
            priv.append(v)
    bad = None
    for name, x in ctx.asserts:
        if ev(x) != 0:
            bad = name
            break
    if bad is None:
        for name, a, b, c in ctx.mulchecks:
            if ev(a) * ev(b) % p != ev(c):
                bad = name
                break
    if bad is None:
        for name, q in ctx.rqueries:
            if not 0 <= ev(q) <= mask:
                bad = name
                break
    if bad is None:
        for tname, qs in ctx.tqueries.items():
            rows = {tuple(r) for r in ctx.table_rows(tname)}
            for name, tup in qs:
                if tuple(ev(x) for x in tup) not in rows:
                    bad = name
                    break
            if bad is not None:
                break
    return priv, ev(out), bad


# ---------------------------------------------------------------------------------------------------------------------
# IR text emission
# ---------------------------------------------------------------------------------------------------------------------

class Emitter:
    def __init__(self, p: int, lines: list[str]) -> None:
        self.p = p
        self.lines = lines
        self.next = 0
        self.n_gates = 0

    def fresh(self) -> int:
        w = self.next
        self.next += 1
        return w

    def fe(self, v: int) -> str:
        return hex(v % self.p)

    def lin(self, x: Lin, wire_of: list[int], dst: int | None = None) -> int:
        """Materialise a linear form; returns its wire id (the single wire when the form is one wire with coefficient
        one, else a fresh wire, or ``dst`` when given)."""
        terms = sorted(x.terms.items())
        if len(terms) == 1 and terms[0][1] == 1 and x.const == 0 and dst is None:
            return wire_of[terms[0][0]]
        acc = None
        n_ops = sum(1 for _, c in terms if c != 1) + max(0, len(terms) - 1) + (1 if x.const != 0 or not terms else 0)
        if n_ops == 0:                          # a single wire copied into dst
            t = dst if dst is not None else self.fresh()
            self.lines.append(f"${t} <- @addc(0: ${wire_of[terms[0][0]]}, <0x0>);")
            self.n_gates += 1
            return t
        k = 0
        for w, c in terms:
            src = wire_of[w]
            if c != 1:
                k += 1
                t = dst if (dst is not None and k == n_ops) else self.fresh()
                self.lines.append(f"${t} <- @mulc(0: ${src}, <{self.fe(c)}>);")
                self.n_gates += 1
                src = t
            if acc is None:
                acc = src
            else:
                k += 1
                t = dst if (dst is not None and k == n_ops) else self.fresh()
                self.lines.append(f"${t} <- @add(0: ${acc}, ${src});")
                self.n_gates += 1
                acc = t
        if x.const != 0 or acc is None:
            k += 1
            t = dst if (dst is not None and k == n_ops) else self.fresh()
            if acc is None:
                self.lines.append(f"${t} <- <{self.fe(x.const)}>;")
            else:
                self.lines.append(f"${t} <- @addc(0: ${acc}, <{self.fe(x.const)}>);")
            self.n_gates += 1
            acc = t
        return acc


def query_layout(ctx: VoleCtx) -> list[tuple[str, int]]:
    """The output blocks of a unit/epilogue function after the word: ("range", n), then ("POW", n), ... (sorted)."""
    blocks = [("range", len(ctx.rqueries))]
    for tname in sorted(ctx.tqueries):
        blocks.append((tname, sum(len(tup) for _, tup in ctx.tqueries[tname])))
    return [b for b in blocks if b[1] > 0]


def emit_function(name: str, ctx: VoleCtx, out: Lin, p: int) -> tuple[list[str], int]:
    """``@function(name, @out: 0:1, 0:NR, 0:NT..., @in: 0:1)``: wire 0 is the word, then the query blocks, then the
    input; body in wire order."""
    layout = query_layout(ctx)
    n_out = 1 + sum(n for _, n in layout)
    outs = ", ".join(f"0:{n}" for _, n in layout)
    lines = [f"@function({name}, @out: 0:1{', ' + outs if outs else ''}, @in: 0:1)"]
    em = Emitter(p, lines)
    em.next = n_out + 1
    wire_of: list[int] = [0] * len(ctx.wires)
    for i, wire in enumerate(ctx.wires):
        if wire[0] == "priv":
            if wire[1][0] == "input":
                wire_of[i] = n_out
                continue
            w = em.fresh()
            lines.append(f"${w} <- @private(0);")
            wire_of[i] = w
        else:
            if i in ctx.virtual:
                continue
            a = em.lin(wire[1], wire_of)
            b = em.lin(wire[2], wire_of)
            w = em.fresh()
            lines.append(f"${w} <- @mul(0: ${a}, ${b});")
            em.n_gates += 1
            wire_of[i] = w
    for _, x in ctx.asserts:
        w = em.lin(x, wire_of)
        lines.append(f"@assert_zero(0: ${w});")
    for _, a, b, c in ctx.mulchecks:
        wa = em.lin(a, wire_of)
        wb = em.lin(b, wire_of)
        wc = em.lin(c, wire_of)
        lines.append(f"@call(vy_amul, ${wa} ... ${wa}, ${wb} ... ${wb}, ${wc} ... ${wc});")
    # the query blocks, in layout order, into their output wires
    dst = 1
    for _, q in ctx.rqueries:
        em.lin(q, wire_of, dst=dst)
        dst += 1
    for tname in sorted(ctx.tqueries):
        for _, tup in ctx.tqueries[tname]:
            for x in tup:
                em.lin(x, wire_of, dst=dst)
                dst += 1
    assert dst == n_out
    em.lin(out, wire_of, dst=0)
    lines.append("@end")
    return lines, em.n_gates


def header(kind: str, p: int, plugin: bool = False) -> list[str]:
    h = ["version 2.0.0;", f"{kind};", f"@type field {hex(p)};"]
    if plugin:
        h.append(f"@plugin {PLUGIN};")
    h.append("@begin")
    return h


def plugin_decls(ctx: VoleCtx, range_n: int, table_n: dict[str, int], cw: int) -> list[str]:
    lines = ["@function(vy_amul, @in: 0:1, 0:1, 0:1)", f"  @plugin({PLUGIN}, assert_mul);"]
    if range_n:
        lines += [f"@function(vy_range, @in: 0:{range_n})", f"  @plugin({PLUGIN}, range, {cw});"]
    for tname, n in table_n.items():
        rows = ctx.table_rows(tname)
        flat = ", ".join(str(v) for r in rows for v in r)
        lines += [f"@function(vy_tab_{tname}, @in: 0:{n})",
                  f"  @plugin({PLUGIN}, lookup, {len(rows[0])}, {len(rows)}, {flat});"]
    return lines


def write_inputs(path: Path, kind: str, p: int, values: Sequence[int]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(header(kind, p)) + "\n")
        f.writelines(f"<{hex(v % p)}>;\n" for v in values)
        f.write("@end\n")


# ---------------------------------------------------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------------------------------------------------

class Batch:
    """Emits a sequence of unit/epilogue calls whose query outputs land in contiguous blocks, then one plugin call per
    block.  ``base`` is the first free wire; the word wires are handed back to the caller."""

    def __init__(self, lines: list[str], base: int, layouts: dict[str, list[tuple[str, int]]], counts: dict[str, int]) -> None:
        self.lines = lines
        self.blocks: dict[str, tuple[int, int]] = {}      # block name -> (first wire, size)
        sizes: dict[str, int] = {}
        for fn, n_calls in counts.items():
            for bname, n in layouts[fn]:
                sizes[bname] = sizes.get(bname, 0) + n * n_calls
        w = base
        for bname, size in sizes.items():
            self.blocks[bname] = (w, size)
            lines.append(f"@new(0: ${w} ... ${w + size - 1});")
            w += size
        self.next = w
        self.cursor = {b: first for b, (first, _) in self.blocks.items()}
        self.layouts = layouts

    def call(self, fn: str, in_wire: int, out_wire: int) -> None:
        outs = [f"${out_wire} ... ${out_wire}"]
        for bname, n in self.layouts[fn]:
            c = self.cursor[bname]
            outs.append(f"${c} ... ${c + n - 1}")
            self.cursor[bname] = c + n
        self.lines.append(f"{', '.join(outs)} <- @call({fn}, ${in_wire} ... ${in_wire});")

    def check(self) -> None:
        for bname, (first, size) in self.blocks.items():
            assert self.cursor[bname] == first + size, bname
            fn = "vy_range" if bname == "range" else f"vy_tab_{bname}"
            self.lines.append(f"@call({fn}, ${first} ... ${first + size - 1});")
        for bname, (first, size) in self.blocks.items():
            self.lines.append(f"@delete(0: ${first} ... ${first + size - 1});")

    @staticmethod
    def block_sizes(layouts: dict[str, list[tuple[str, int]]], counts: dict[str, int]) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for fn, n_calls in counts.items():
            for bname, n in layouts[fn]:
                sizes[bname] = sizes.get(bname, 0) + n * n_calls
        return sizes


def unit_relation(unit: Unit, n: int, io: str) -> list[str]:
    """n independent units: c and y as @public (io=public) or @private (io=private); operands are always private."""
    p = unit.ctx.p
    lines = header("circuit", p, plugin=True)
    layouts = {"unit": query_layout(unit.ctx)}
    sizes = Batch.block_sizes(layouts, {"unit": n})
    lines += plugin_decls(unit.ctx, sizes.get("range", 0), {b: s for b, s in sizes.items() if b != "range"}, unit.ctx.cw)
    body, _ = emit_function("unit", unit.ctx, unit.out, p)
    lines += body
    src = "@public(0)" if io == "public" else "@private(0)"
    # wires 0 .. 5n-1: per unit (c, y, y', -y', diff); query blocks after
    batch = Batch(lines, 5 * n, layouts, {"unit": n})
    w = 0
    for _ in range(n):
        lines.append(f"${w} <- {src};")
        batch.call("unit", w, w + 1)
        lines.append(f"${w + 2} <- {src};")
        lines.append(f"${w + 3} <- @mulc(0: ${w + 2}, <{hex(p - 1)}>);")
        lines.append(f"${w + 4} <- @add(0: ${w + 1}, ${w + 3});")
        lines.append(f"@assert_zero(0: ${w + 4});")
        w += 5
    batch.check()
    lines.append("@end")
    return lines


def vu_relation(unit: Unit, epi: Epi, B: int, io: str) -> list[str]:
    """B VUs: c_0 = 0, 96 chained unit calls, the epilogue, one logUp call per table per VU, y16 compared with the
    claimed word."""
    p = unit.ctx.p
    lines = header("circuit", p, plugin=True)
    layouts = {"unit": query_layout(unit.ctx), "epilogue": query_layout(epi.ctx)}
    counts = {"unit": UNITS_PER_VU, "epilogue": 1}
    sizes = Batch.block_sizes(layouts, counts)
    lines += plugin_decls(unit.ctx, sizes.get("range", 0), {b: s for b, s in sizes.items() if b != "range"}, unit.ctx.cw)
    body, _ = emit_function("unit", unit.ctx, unit.out, p)
    lines += body
    body, _ = emit_function("epilogue", epi.ctx, epi.out, p)
    lines += body
    src = "@public(0)" if io == "public" else "@private(0)"
    lines.append("@function(vu, @out: 0:1)")
    # $0 = y16, $1 = c_0, $2..$97 = the unit words; query blocks from $98
    lines.append("  $1 <- <0x0>;")
    batch = Batch(lines, UNITS_PER_VU + 2, layouts, counts)
    for s in range(UNITS_PER_VU):
        batch.call("unit", s + 1, s + 2)
    batch.call("epilogue", UNITS_PER_VU + 1, 0)
    batch.check()
    lines.append("@end")
    w = 0
    for _ in range(B):
        lines.append(f"${w} <- @call(vu);")
        lines.append(f"${w + 1} <- {src};")
        lines.append(f"${w + 2} <- @mulc(0: ${w + 1}, <{hex(p - 1)}>);")
        lines.append(f"${w + 3} <- @add(0: ${w}, ${w + 2});")
        lines.append(f"@assert_zero(0: ${w + 3});")
        lines.append(f"@delete(0: ${w} ... ${w + 3});")
        w += 4
    lines.append("@end")
    return lines


# ---------------------------------------------------------------------------------------------------------------------
# Witnesses
# ---------------------------------------------------------------------------------------------------------------------

class LenientCtx:
    """An IntCtx that never rejects: lets ``gen_step`` run on out-of-domain inputs so a cheating witness exists to
    feed the circuit (which must then reject it)."""

    def rng(self, name, x, lo, hi):
        pass

    def eq(self, name, a, b):
        pass

    def boolean(self, name, b):
        pass

    def lookup(self, name, table: Table, key):
        try:
            return table.row(key)
        except KeyError:
            return tuple(0 for _ in table.outs)

    def lt_pow(self, name, x, ipow, bits):
        pass


def gen_step_lenient(p: Params, c: int, a: Sequence[int], b: Sequence[int]) -> dict:
    saved = step_mod.IntCtx
    step_mod.IntCtx = LenientCtx  # type: ignore[assignment]
    try:
        return gen_step(p, c, a, b)
    finally:
        step_mod.IntCtx = saved  # type: ignore[assignment]


def vu_witness(unit: Unit, epi: Epi, a: Sequence[int], b: Sequence[int], y16: int, io: str, p: Params = REAL,
               lenient: bool = False) -> tuple[list[int], str | None]:
    """The private stream of one VU (chained through the circuit's own output words) and the first failing check."""
    k = unit.k
    priv: list[int] = []
    c = 0
    bad = None
    gen = gen_step_lenient if lenient else gen_step
    for start in range(0, len(a), k):
        aa, bb = list(a[start:start + k]), list(b[start:start + k])
        try:
            h = gen(p, c, aa, bb)
        except Reject as e:
            return priv, f"no witness: {e}"
        w, c_next, fail = unit.witness(c, aa, bb, h)
        priv += w
        bad = bad or fail
        c = c_next
    h = Epilogue.gen(p, c)
    w, got, fail = epi.witness(c, h)
    priv += w
    bad = bad or fail
    if io == "private":
        priv.append(y16 % unit.ctx.p)
    if bad is None and got != y16 % unit.ctx.p:
        bad = "y16"
    return priv, bad


def random_finite_unit(rng: random.Random, p: Params = REAL) -> tuple[int, list[int], list[int], int, dict]:
    k = sum(p.groups)

    def bf16() -> int:
        return (rng.randrange(2) << 15) | (rng.randrange(p.t_finite) << 7) | rng.randrange(2**7)

    def f32() -> int:
        return (rng.randrange(2) << 31) | (rng.randrange(p.t_finite) << 23) | rng.randrange(2**23)

    c = f32()
    a = [bf16() for _ in range(k)]
    b = [bf16() for _ in range(k)]
    y = tc_dot(AMPERE_BF16_M16N8K16, c, a, b)
    h = gen_step(p, c, a, b)
    check_step(p, c, a, b, y, h)
    return c, a, b, y, h


# ---------------------------------------------------------------------------------------------------------------------
# Census
# ---------------------------------------------------------------------------------------------------------------------

def census(field: str = "f61p", chunk_bits: int = CHUNK_BITS) -> dict:
    p = FIELDS[field]
    unit = Unit(p, chunk_bits=chunk_bits)
    epi = Epi(p, chunk_bits=chunk_bits)
    body, gates = emit_function("unit", unit.ctx, unit.out, p)
    ebody, _ = emit_function("epilogue", epi.ctx, epi.out, p)
    u, e = unit.ctx, epi.ctx
    n_rq = len(u.rqueries)
    n_tq = {t: len(qs) for t, qs in u.tqueries.items()}
    table_rows = {t: len(u.table_rows(t)) for t in u.tqueries}
    per_unit = {
        "private_wires": unit.n_private, "of_which_io": u.n_io - 1, "of_which_hints": u.n_hint,
        "of_which_range_chunks": u.n_chunk, "of_which_table_outputs": u.n_tout,
        "mul_gates_committed": u.n_prod_committed, "assert_mul_output_free": len(u.mulchecks),
        "assert_zero": len(u.asserts), "range_queries": n_rq, "table_queries": n_tq,
        "range_obligations": len(u.pending), "range_obligations_discharged": len(u.discharged),
        "discharged": [f"{n} (w={w}): {why}" for n, w, why in u.discharged],
        "committed_in_unit": unit.n_private + u.n_prod_committed,
        "committed_incl_query_inverses": unit.n_private + u.n_prod_committed + n_rq + sum(n_tq.values()),
        "mult_checks": u.n_prod_committed + len(u.mulchecks) + n_rq + sum(n_tq.values()),
        "lt_pow": u.n_ltpow, "lookups": dict(u.lookups), "linear_gates_in_ir": gates - u.n_prod_committed,
        "ir_lines": len(body),
    }
    e_rq = len(e.rqueries)
    per_epi = {"private_wires": epi.n_private, "mul_gates_committed": e.n_prod_committed,
               "assert_mul_output_free": len(e.mulchecks), "assert_zero": len(e.asserts), "range_queries": e_rq,
               "committed_incl_query_inverses": epi.n_private + e.n_prod_committed + e_rq, "ir_lines": len(ebody)}
    vu_rq = UNITS_PER_VU * n_rq + e_rq
    vu_tq = {t: UNITS_PER_VU * n for t, n in n_tq.items()}
    table_cost = 2 * (1 << chunk_bits) + sum(2 * table_rows[t] for t in vu_tq)
    vu_committed = (UNITS_PER_VU * per_unit["committed_in_unit"] + epi.n_private + e.n_prod_committed + 1
                    + vu_rq + sum(vu_tq.values()) + table_cost)
    vu_mult = UNITS_PER_VU * (u.n_prod_committed + len(u.mulchecks)) + e.n_prod_committed + len(e.mulchecks) + vu_rq + sum(vu_tq.values())
    nbytes = (p.bit_length() + 7) // 8
    return {
        "field": field, "p_bits": p.bit_length(), "chunk_bits": chunk_bits, "unit": per_unit, "epilogue": per_epi,
        "tables": {"range": {"rows": 1 << chunk_bits, "cols": 1},
                   **{t: {"rows": table_rows[t], "cols": 1 + len(u.tables[t].outs)} for t in vu_tq}},
        "vu": {"units": UNITS_PER_VU, "committed_elements": vu_committed,
               "of_which_logup_tables": table_cost, "of_which_query_inverses": vu_rq + sum(vu_tq.values()),
               "range_queries": vu_rq, "table_queries": vu_tq, "mult_checks": vu_mult,
               "bytes_prover_to_verifier_field": nbytes * vu_committed,
               "vs_v1_committed": V1_CENSUS["vu"]["vole_correlations"] / vu_committed,
               "logup_soundness_terms_per_vu": {
                   "range_call_(N+R)/p": (vu_rq + (1 << chunk_bits)),
                   **{f"{t}_call_(N+R+N*R*(cols-1))/p": n + table_rows[t] + n * table_rows[t] * len(u.tables[t].outs)
                      for t, n in vu_tq.items()}}},
        "v1": V1_CENSUS,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------

def cmd_census(args) -> int:
    print(json.dumps(census(args.field, args.chunk_bits), indent=1))
    return 0


def _write_meta(out: Path, meta: dict) -> None:
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def cmd_unit(args) -> int:
    p = FIELDS[args.field]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    unit = Unit(p, chunk_bits=args.chunk_bits)
    rng = random.Random(args.seed)
    inst: list[int] = []
    wit: list[int] = []
    for _ in range(args.n):
        c, a, b, y, h = random_finite_unit(rng)
        w, got, bad = unit.witness(c, a, b, h)
        if bad is not None or got != y:
            print(f"circuit rejects an honest unit: {bad} got={got} y={y}", file=sys.stderr)
            return 1
        if args.io == "public":
            inst += [c, y]
            wit += w
        else:
            wit += [c, *w, y]
    (out / "relation.txt").write_text("\n".join(unit_relation(unit, args.n, args.io)) + "\n")
    write_inputs(out / "instance.txt", "public_input", p, inst)
    write_inputs(out / "witness.txt", "private_input", p, wit)
    meta = {"kind": "unit", "version": 2, "n": args.n, "field": args.field, "io": args.io, "seed": args.seed,
            "chunk_bits": args.chunk_bits, "census": census(args.field, args.chunk_bits)["unit"]}
    _write_meta(out, meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "census"}))
    return 0


def _load_tier(root: str, tier: str):
    from verity_numerical.bench import instances
    return instances.load(tier, root)


def cmd_tu(args) -> int:
    p = FIELDS[args.field]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    unit = Unit(p, chunk_bits=args.chunk_bits)
    tier = _load_tier(args.root, "tu-k16")
    n = min(args.n, tier.n)
    inst: list[int] = []
    wit: list[int] = []
    rejected = 0
    for i in range(n):
        c, a, b, y = tier.instance(i)
        try:
            h = gen_step(REAL, c, a, b)
        except Reject:
            rejected += 1          # saturating / out-of-domain transition: no honest witness (D8), skipped
            continue
        w, got, bad = unit.witness(c, a, b, h)
        if bad is not None or got != y:
            print(f"circuit rejects tu-k16 instance {i}: {bad} got={got} y={y}", file=sys.stderr)
            return 1
        if args.io == "public":
            inst += [c, y]
            wit += w
        else:
            wit += [c, *w, y]
    used = n - rejected
    (out / "relation.txt").write_text("\n".join(unit_relation(unit, used, args.io)) + "\n")
    write_inputs(out / "instance.txt", "public_input", p, inst)
    write_inputs(out / "witness.txt", "private_input", p, wit)
    meta = {"kind": "tu-k16", "version": 2, "n": used, "skipped_out_of_domain": rejected, "field": args.field,
            "io": args.io, "chunk_bits": args.chunk_bits}
    _write_meta(out, meta)
    print(json.dumps(meta))
    return 0


def _vu_task(task):
    i, a, b, y, field, io, cw, lenient = task
    p = FIELDS[field]
    global _UNIT, _EPI
    if _UNIT is None or _UNIT.ctx.p != p or _UNIT.ctx.cw != cw:
        _UNIT, _EPI = Unit(p, chunk_bits=cw), Epi(p, chunk_bits=cw)
    priv, bad = vu_witness(_UNIT, _EPI, a, b, y, io, lenient=lenient)
    text = "".join(f"<{hex(v)}>;\n" for v in priv)
    return i, len(priv), text, bad


_UNIT: Unit | None = None
_EPI: Epi | None = None


def cmd_vu(args) -> int:
    p = FIELDS[args.field]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    unit, epi = Unit(p, chunk_bits=args.chunk_bits), Epi(p, chunk_bits=args.chunk_bits)
    neg = bool(getattr(args, "neg", False))     # vu-k1536-neg: cheating witnesses, dietmc's verifier must reject
    tier = _load_tier(args.root, "vu-k1536-neg" if neg else "vu-k1536")
    B = args.B
    tasks = []
    for i in range(B):
        a, b, y = tier.instance(i)
        tasks.append((i, a, b, y, args.field, args.io, args.chunk_bits, neg))
    inst: list[int] = []
    if args.io == "public":
        inst = [tasks[i][3] for i in range(B)]
    (out / "relation.txt").write_text("\n".join(vu_relation(unit, epi, B, args.io)) + "\n")
    write_inputs(out / "instance.txt", "public_input", p, inst)
    n_wit = 0
    with open(out / "witness.txt", "w") as f:
        f.write("\n".join(header("private_input", p)) + "\n")
        if args.procs > 1:
            import multiprocessing as mp
            with mp.Pool(args.procs) as pool:
                for i, n_priv, text, bad in pool.imap(_vu_task, tasks, chunksize=1):   # in order: the stream is positional
                    if bad is not None and not neg:
                        print(f"circuit rejects honest VU {i}: {bad}", file=sys.stderr)
                        return 1
                    f.write(text)
                    n_wit += n_priv
        else:
            for task in tasks:
                i, n_priv, text, bad = _vu_task(task)
                if bad is not None and not neg:
                    print(f"circuit rejects honest VU {i}: {bad}", file=sys.stderr)
                    return 1
                f.write(text)
                n_wit += n_priv
        f.write("@end\n")
    meta = {"kind": "vu-k1536-neg" if neg else "vu-k1536", "version": 2, "B": B, "instances": f"{tier.tier}[0:{B}]", "field": args.field,
            "io": args.io, "chunk_bits": args.chunk_bits, "private_values": n_wit, "private_values_per_vu": n_wit / B,
            "export_seconds": time.time() - t0, "census_vu": census(args.field, args.chunk_bits)["vu"]}
    _write_meta(out, meta)
    print(json.dumps(meta))
    return 0


def cmd_negatives(args) -> int:
    p = FIELDS[args.field]
    unit, epi = Unit(p, chunk_bits=args.chunk_bits), Epi(p, chunk_bits=args.chunk_bits)
    from verity_numerical.gkr_export.negatives import negative_cases
    n_ok = 0
    rows = []
    for name, c, a, b, y, h, predicate in negative_cases():
        _w, got, bad = unit.witness(c, a, b, h)
        verdict = bad if bad is not None else ("y" if got != y % p else None)
        if verdict is None:
            print(f"circuit ACCEPTS WP2 negative {name} (WP2 rejects at {predicate})", file=sys.stderr)
            return 1
        rows.append((name, predicate, verdict))
        n_ok += 1
    print(f"WP2 negatives: {n_ok}/{n_ok} rejected by the circuit evaluator")
    for r in rows:
        print("  ", *r, sep="\t")
    if args.root:
        tier = _load_tier(args.root, "vu-k1536-neg")
        cols = tier.columns
        n_rej = 0
        for i in range(tier.n):
            a, b, y = tier.instance(i)
            row = dict(zip(cols, tier.index[i]))
            _priv, bad = vu_witness(unit, epi, a, b, y, args.io, lenient=True)
            if bad is None:
                print(f"circuit ACCEPTS vu-k1536-neg {row.get('id')} ({row.get('mistake')})", file=sys.stderr)
                return 1
            n_rej += 1
            print(f"  neg {i:2d} {row.get('mistake')!s:40s} {row.get('verdict')!s:7s} -> circuit rejects at {bad}")
        print(f"vu-k1536-neg: {n_rej}/{tier.n} rejected")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp, root: bool = False, io: bool = True) -> None:
        sp.add_argument("--field", default="f61p", choices=sorted(FIELDS))
        sp.add_argument("--chunk-bits", type=int, default=CHUNK_BITS)
        if io:
            sp.add_argument("--io", default="private", choices=("private", "public"))
        if root:
            sp.add_argument("--root", required=True)

    c = sub.add_parser("census")
    common(c, io=False)
    c.set_defaults(fn=cmd_census)
    u = sub.add_parser("unit")
    u.add_argument("out")
    u.add_argument("--n", type=int, default=1)
    u.add_argument("--seed", type=int, default=20260922)
    common(u)
    u.set_defaults(fn=cmd_unit)
    t = sub.add_parser("tu")
    t.add_argument("out")
    t.add_argument("--n", type=int, default=4096)
    common(t, root=True)
    t.set_defaults(fn=cmd_tu)
    v = sub.add_parser("vu")
    v.add_argument("out")
    v.add_argument("--B", type=int, default=1)
    v.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    v.add_argument("--neg", action="store_true", help="export vu-k1536-neg instances (cheating witnesses; verifier must reject)")
    common(v, root=True)
    v.set_defaults(fn=cmd_vu)
    n = sub.add_parser("negatives")
    n.add_argument("--root", default=None)
    common(n)
    n.set_defaults(fn=cmd_negatives)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
