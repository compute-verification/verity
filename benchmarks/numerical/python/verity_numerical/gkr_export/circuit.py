"""The symbolic context that turns the WP2 gadgets into a circuit, and the witness generator for it."""
from __future__ import annotations

import random
import re
import struct
from collections.abc import Sequence
from pathlib import Path

from verity.ml.circuits.silicon.silicon import AMPERE_BF16_M16N8K16, tc_dot

from verity_numerical.checker import REAL, Params, Reject, check_step, gen_step
from verity_numerical.checker.ctx import Table
from verity_numerical.checker.step import check_step_expr

__all__ = [
    "CircuitCtx",
    "Lin",
    "build_unit_circuit",
    "export_circuit",
    "random_finite_unit",
    "unit_witness",
    "verify_unit",
    "write_witness",
]

GOLDILOCKS = 2**64 - 2**32 + 1
R16 = "R16"
CHUNK = 16


def range_table(w: int) -> str:
    """The implicit range table ``R<w>`` = [0, 2^w) (``w <= 16``; the prover materialises it from the name)."""
    assert 1 <= w <= CHUNK
    return f"R{w}"


def range_table_width(name: str) -> int | None:
    """Width of an implicit range table name, ``None`` for a declared table."""
    m = re.fullmatch(r"R(\d+)", name)
    return int(m.group(1)) if m and 1 <= int(m.group(1)) <= CHUNK else None


class Lin:
    """A linear form over circuit wires with an integer constant.  ``Lin * Lin`` creates a product wire."""

    __slots__ = ("b", "const", "terms")

    def __init__(self, b: CircuitCtx, terms: dict[int, int] | None = None, const: int = 0) -> None:
        self.b = b
        self.terms = {w: c for w, c in (terms or {}).items() if c != 0}
        self.const = const

    @property
    def is_const(self) -> bool:
        return not self.terms

    def __add__(self, o):
        if isinstance(o, int):
            return Lin(self.b, self.terms, self.const + o)
        t = dict(self.terms)
        for w, c in o.terms.items():
            t[w] = t.get(w, 0) + c
        return Lin(self.b, t, self.const + o.const)

    __radd__ = __add__

    def __neg__(self):
        return Lin(self.b, {w: -c for w, c in self.terms.items()}, -self.const)

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, int):
            return Lin(self.b, {w: c * o for w, c in self.terms.items()}, self.const * o)
        if self.is_const:
            return o * self.const
        if o.is_const:
            return self * o.const
        return self.b.product(self, o)

    __rmul__ = __mul__


class CircuitCtx:
    """Collects the five predicates as gates, assertions and lookup queries (see the package docstring)."""

    def __init__(self) -> None:
        self.cols: list[str] = []
        self.derived: list[tuple | None] = []
        self.wires: list[tuple] = []          # ("in", col) | ("prod", A, B)
        self.depth: list[int] = []
        self.asserts: list[tuple[str, Lin]] = []
        self.queries: list[tuple[str, list[Lin]]] = []
        self.tables: dict[str, Table] = {}
        self.one = self.column("one", ("const", 1))

    # -- wires --------------------------------------------------------------------------------------------------------
    def column(self, name: str, derived: tuple | None = None) -> Lin:
        self.cols.append(name)
        self.derived.append(derived)
        self.wires.append(("in", len(self.cols) - 1))
        self.depth.append(0)
        return Lin(self, {len(self.wires) - 1: 1})

    def product(self, a: Lin, b: Lin) -> Lin:
        d = 1 + max(self._depth(a), self._depth(b))
        self.wires.append(("prod", a, b))
        self.depth.append(d)
        return Lin(self, {len(self.wires) - 1: 1})

    def _depth(self, x: Lin) -> int:
        return max((self.depth[w] for w in x.terms), default=0)

    def _as_input_lin(self, x: Lin, name: str) -> Lin:
        if all(self.wires[w][0] == "in" for w in x.terms):
            return x
        col = self.column(f"q.{name}", ("lin", x))
        self.asserts.append((f"{name}.bind", col - x))
        return col

    # -- predicates ---------------------------------------------------------------------------------------------------
    def rng(self, name: str, x, lo: int, hi: int) -> None:
        if isinstance(x, int) or x.is_const:
            v = x if isinstance(x, int) else x.const
            if not lo <= v < hi:
                raise Reject(f"{name}: constant {v} not in [{lo}, {hi})")
            return
        n = hi - lo
        w = max(1, (n - 1).bit_length())
        self._range_pow2(x - lo, w, name)
        if n != 1 << w:
            self._range_pow2((hi - 1) - x, w, name + ".hi")

    def _range_pow2(self, y: Lin, w: int, name: str) -> None:
        y = self._as_input_lin(y, name)
        if w <= CHUNK:
            # an exact-width table: the shifted single-chunk form (y * 2^(16-w) in R16) is forgeable, since
            # y = (r + k p) / 2^(16-w) for any r < 2^16 and the k making it integral lands in R16 too
            self.queries.append((range_table(w), [y]))
            return
        if w > 2 * CHUNK:
            raise Reject(f"{name}: {w}-bit range needs more than two chunks")
        lo = self.column(f"{name}.lo16", ("lo16", y))
        hi = self.column(f"{name}.hi16", ("hi16", y))
        self.asserts.append((f"{name}.split", y - lo - 2**CHUNK * hi))
        self.queries.append((R16, [lo]))
        self.queries.append((range_table(w - CHUNK), [hi]))

    def eq(self, name: str, a, b) -> None:
        d = a - b
        if isinstance(d, int) or d.is_const:
            if (d if isinstance(d, int) else d.const) != 0:
                raise Reject(f"{name}: constant identity fails")
            return
        self.asserts.append((name, d))

    def boolean(self, name: str, b) -> None:
        if isinstance(b, int) or b.is_const:
            if (b if isinstance(b, int) else b.const) not in (0, 1):
                raise Reject(f"{name}: constant not a bit")
            return
        self.asserts.append((name, b * b - b))

    def lookup(self, name: str, table: Table, key) -> tuple:
        self.tables[table.name] = table
        if isinstance(key, int) or key.is_const:
            return table.row(key if isinstance(key, int) else key.const)
        key = self._as_input_lin(key, name)
        outs = [self.column(f"{name}.{o}", ("row", table, key, i)) for i, o in enumerate(table.outs)]
        self.queries.append((table.name, [key, *outs]))
        return tuple(outs)

    def lt_pow(self, name: str, x, ipow, bits: int) -> None:
        self.rng(name, x * ipow, 0, 2**bits)

    # -- evaluation ---------------------------------------------------------------------------------------------------
    def eval_lin(self, x: Lin, vals: list[int | None], cache: dict[int, int]) -> int:
        return x.const + sum(c * self.wire_val(w, vals, cache) for w, c in x.terms.items())

    def wire_val(self, w: int, vals: list[int | None], cache: dict[int, int]) -> int:
        kind = self.wires[w]
        if kind[0] == "in":
            v = vals[kind[1]]
            if v is None:
                raise Reject(f"column {self.cols[kind[1]]} evaluated before it is set")
            return v
        if w in cache:
            return cache[w]
        v = self.eval_lin(kind[1], vals, cache) * self.eval_lin(kind[2], vals, cache)
        cache[w] = v
        return v


# ---------------------------------------------------------------------------------------------------------------------
# The unit circuit
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


def build_unit_circuit(p: Params = REAL) -> tuple[CircuitCtx, list[str]]:
    """The transition unit as a circuit; returns the context and the base-column hint paths (``h.acc.s``, ...)."""
    ctx = CircuitCtx()
    k = sum(p.groups)
    c = ctx.column("x.c")
    a = [ctx.column(f"x.a[{i}]") for i in range(k)]
    b = [ctx.column(f"x.b[{i}]") for i in range(k)]
    y = ctx.column("y")
    sample = gen_step(p, 0, [0] * k, [0] * k)
    paths: list[str] = []
    _leaves(sample, "h", paths)
    import copy
    h_sym = copy.deepcopy(sample)
    for path in paths:
        _set(h_sym, path, ctx.column(path))
    got = check_step_expr(ctx, p, c, a, b, h_sym)
    ctx.eq("y", got, y)
    return ctx, paths


def unit_witness(ctx: CircuitCtx, paths: list[str], c: int, a: Sequence[int], b: Sequence[int], y: int, h: dict) -> list[int]:
    """Every column of one unit as integers (base columns from ``h``, derived columns evaluated in order)."""
    k = len(a)
    vals: list[int | None] = [None] * len(ctx.cols)
    vals[0] = 1
    vals[1] = c
    for i in range(k):
        vals[2 + i] = a[i]
        vals[2 + k + i] = b[i]
    vals[2 + 2 * k] = y
    base = 3 + 2 * k
    for i, path in enumerate(paths):
        vals[base + i] = int(_get(h, path))
    cache: dict[int, int] = {}
    for col, d in enumerate(ctx.derived):
        if d is None:
            continue
        kind = d[0]
        if kind == "const":
            vals[col] = d[1]
        elif kind == "lin":
            vals[col] = ctx.eval_lin(d[1], vals, cache)
        elif kind == "lo16":
            vals[col] = ctx.eval_lin(d[1], vals, cache) & (2**CHUNK - 1)
        elif kind == "hi16":
            vals[col] = ctx.eval_lin(d[1], vals, cache) >> CHUNK
        elif kind == "row":
            _, table, key, idx = d
            kv = ctx.eval_lin(key, vals, cache)
            vals[col] = table.row(kv)[idx] if table.lo <= kv < table.hi else 0
    return [int(v) for v in vals]  # type: ignore[arg-type]


def verify_unit(ctx: CircuitCtx, vals: list[int]) -> str | None:
    """The circuit's own check on one unit's columns (mod Goldilocks): the first failing assertion / query, or None."""
    cache: dict[int, int] = {}
    for name, x in ctx.asserts:
        if ctx.eval_lin(x, vals, cache) % GOLDILOCKS:
            return f"assert {name}"
    for i, (tname, lins) in enumerate(ctx.queries):
        row = tuple(ctx.eval_lin(x, vals, cache) % GOLDILOCKS for x in lins)
        if (w := range_table_width(tname)) is not None:
            if row[0] >= 2**w:
                return f"query {i} {tname} {row[0]}"
        else:
            t = ctx.tables[tname]
            key = row[0] if row[0] < 2**40 else row[0] - GOLDILOCKS
            if not t.lo <= key < t.hi or tuple(v % GOLDILOCKS for v in t.row(key)) != row[1:]:
                return f"query {i} {tname} {row}"
    return None


def random_finite_unit(rng: random.Random, p: Params = REAL) -> tuple[int, list[int], list[int], int, dict]:
    """Random finite words (exponent fields below the all-ones field), the target's output and the WP2 hints."""
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
# Export
# ---------------------------------------------------------------------------------------------------------------------

def _fe(v: int, modulus: int = GOLDILOCKS) -> int:
    return v % modulus


def _lin_wires(x: Lin, m: int = GOLDILOCKS) -> str:
    return f"{len(x.terms)} " + " ".join(f"{w} {_fe(c, m)}" for w, c in sorted(x.terms.items())) + f" {_fe(x.const, m)}"


def _lin_cols(ctx: CircuitCtx, x: Lin, m: int = GOLDILOCKS) -> str:
    return f"{len(x.terms)} " + " ".join(f"{ctx.wires[w][1]} {_fe(c, m)}" for w, c in sorted(x.terms.items())) + f" {_fe(x.const, m)}"


def export_circuit(ctx: CircuitCtx, path: Path, modulus: int = GOLDILOCKS) -> None:
    """The text format ``archive/gkr`` parses (see ``archive/gkr/src/circuit.rs``); values reduced mod ``modulus``."""
    m = modulus
    lines = [f"ncols {len(ctx.cols)}", f"nwires {len(ctx.wires)}", "one 0"]
    lines += [f"col {i} {n}" for i, n in enumerate(ctx.cols)]
    for w, kind in enumerate(ctx.wires):
        if kind[0] == "prod":
            lines.append(f"prod {w} {ctx.depth[w]} {_lin_wires(kind[1], m)} {_lin_wires(kind[2], m)}")
    for name, x in ctx.asserts:
        lines.append(f"assert {name} {_lin_wires(x, m)}")
    for tname, t in ctx.tables.items():
        lines.append(f"table {tname} {len(t.outs) + 1} {t.rows}")
        for key in range(t.lo, t.hi):
            lines.append("row " + " ".join(str(_fe(v, m)) for v in (key, *t.row(key))))
    for tname, lins in ctx.queries:
        lines.append(f"query {tname} {len(lins)} " + " ".join(_lin_cols(ctx, x, m) for x in lins))
    path.write_text("\n".join(lines) + "\n")


def write_witness(units: list[list[int]], path: Path, modulus: int = GOLDILOCKS) -> None:
    """Flat little-endian u64: ``n_units n_cols`` then the columns of every unit, reduced mod ``modulus``."""
    ncols = len(units[0]) if units else 0
    with open(path, "wb") as f:
        f.write(struct.pack("<QQ", len(units), ncols))
        f.writelines(struct.pack(f"<{ncols}Q", *(_fe(v, modulus) for v in u)) for u in units)
