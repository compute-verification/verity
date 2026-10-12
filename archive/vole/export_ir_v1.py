"""The WP2 checker as a SIEVE Circuit-IR (IR0+, text) relation for Diet Mac'n'Cheese (QuickSilver over a prime field).

    python archive/vole/export_ir.py census                         gate counts of one unit / one VU, vs explore/vole.py
    python archive/vole/export_ir.py unit  OUT [--n N] [--field F]  relation/instance/witness for N random finite units
    python archive/vole/export_ir.py tu    OUT --root R --n N      the same on the tu-k16 fixture tier
    python archive/vole/export_ir.py vu    OUT --root R --B B      one relation with B chained K=1536 VUs (vu-k1536[0:B])
    python archive/vole/export_ir.py negatives [--root R]           every WP2 negative and every vu-k1536-neg through the
                                                                     circuit's own evaluator (all must fail)

The relation is the frozen WP2 checker (``verity_numerical.checker``): the same gadgets, the same predicates, called
through a symbolic context exactly as ``gkr_export`` does.  Realisation of the five predicates over F_p, p > 2^60
(checker requirement F1, so every ``lt_pow`` product ``x * ipow`` stays below p):

    eq(a, b)            a - b as a linear form over committed wires; one @assert_zero (free: linear on IT-MACs)
    boolean(b)          one multiplication b*b and @assert_zero(b*b - b)
    rng(x, lo, hi)      x - lo written as w = ceil(log2(hi - lo)) private bits, each boolean, one linear identity;
                        if hi - lo is not a power of two, (hi - 1) - x is decomposed as well  ("bits" mode of the cost model)
    lookup(T, key)      the table's rows grouped into maximal runs of equal rows (POW 26 runs, ALIGN 27, LEAD 31, NORM 32
                        at REAL); one private one-hot selector per run (boolean each, sum = 1); the outputs are linear in
                        the selectors; key = sum sel_i lo_i + off with 0 <= off <= sum sel_i (len_i - 1) proved by two
                        bit decompositions (LEAD/NORM: every run has length 1, so off = 0 is one linear identity)
    lt_pow(x, ipow, B)  one multiplication x * ipow, then rng(., 0, 2^B)

The exporter evaluates its own circuit in Python (``evaluate``) and refuses to write a witness that the circuit
rejects for an honest instance, or accepts for a negative.  Wires inside the unit function are local, so memory in
dietmc stays per unit; the witness stream order is the order of the @private gates (documented in ``Unit.private``).
"""
from __future__ import annotations

import argparse
import copy
import json
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


# ---------------------------------------------------------------------------------------------------------------------
# The symbolic context
# ---------------------------------------------------------------------------------------------------------------------

class VoleCtx:
    """Collects the checker's predicates as private wires, product wires and zero-assertions.

    ``wires[i]`` is ``("priv", derivation)`` or ``("prod", Lin, Lin)``; ``derivation`` says how the honest prover fills
    the wire: ``("io", name)``, ``("hint", path)``, ``("bit", Lin, i)``, ``("sel", table_name, key_Lin, run_index)``.
    """

    def __init__(self, p: int) -> None:
        self.p = p
        self.wires: list[tuple] = []
        self.asserts: list[tuple[str, Lin]] = []
        self.tables: dict[str, tuple[Table, list[tuple[int, int, tuple[int, ...]]]]] = {}
        self.n_bits = 0
        self.n_sel = 0
        self.n_hint = 0
        self.n_io = 0
        self.n_ltpow = 0
        self.lookups: dict[str, int] = {}

    # -- wires --------------------------------------------------------------------------------------------------------
    def priv(self, derivation: tuple) -> Lin:
        self.wires.append(("priv", derivation))
        kind = derivation[0]
        if kind == "bit":
            self.n_bits += 1
        elif kind == "sel":
            self.n_sel += 1
        elif kind == "hint":
            self.n_hint += 1
        else:
            self.n_io += 1
        return Lin(self, {len(self.wires) - 1: 1})

    def product(self, a: Lin, b: Lin) -> Lin:
        self.wires.append(("prod", a, b))
        return Lin(self, {len(self.wires) - 1: 1})

    @property
    def n_mul(self) -> int:
        return sum(1 for w in self.wires if w[0] == "prod")

    @property
    def n_priv(self) -> int:
        return len(self.wires) - self.n_mul

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
        self._bits(x - lo, w, name)
        if n != 1 << w:
            self._bits((hi - 1) - x, w, name + ".hi")

    def _bits(self, y: Lin, w: int, name: str) -> None:
        if y.is_const:
            if not 0 <= y.const < 2**w:
                raise Reject(f"{name}: constant {y.const} not in [0, 2^{w})")
            return
        acc = Lin(self)
        for i in range(w):
            b = self.priv(("bit", y, i))
            self.asserts.append((f"{name}.bit[{i}]", b * b - b))
            acc = acc + b * 2**i
        self.asserts.append((f"{name}.bits", y - acc))

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

    def _runs(self, table: Table) -> list[tuple[int, int, tuple[int, ...]]]:
        if table.name in self.tables:
            return self.tables[table.name][1]
        runs: list[tuple[int, int, tuple[int, ...]]] = []
        for key in range(table.lo, table.hi):
            row = tuple(table.row(key))
            if runs and runs[-1][2] == row:
                lo, n, _ = runs[-1]
                runs[-1] = (lo, n + 1, row)
            else:
                runs.append((key, 1, row))
        self.tables[table.name] = (table, runs)
        return runs

    def lookup(self, name: str, table: Table, key) -> tuple:
        v = self._const(key)
        if v is not None:
            return table.row(v)
        runs = self._runs(table)
        self.lookups[table.name] = self.lookups.get(table.name, 0) + 1
        sels = [self.priv(("sel", table.name, key, i)) for i in range(len(runs))]
        total = Lin(self)
        base = Lin(self)
        span = Lin(self)
        outs = [Lin(self) for _ in table.outs]
        for s, (lo, n, row) in zip(sels, runs):
            self.asserts.append((f"{name}.sel", s * s - s))
            total = total + s
            base = base + s * lo
            span = span + s * (n - 1)
            for j, val in enumerate(row):
                outs[j] = outs[j] + s * val
        self.asserts.append((f"{name}.onehot", total - 1))
        off = key - base
        if all(n == 1 for _, n, _ in runs):
            self.asserts.append((f"{name}.key", off))
        else:
            w = max(1, max(n - 1 for _, n, _ in runs).bit_length())
            self._bits(off, w, f"{name}.off")
            self._bits(span - off, w, f"{name}.off.hi")
        return tuple(outs)

    def lt_pow(self, name: str, x, ipow, bits: int) -> None:
        self.n_ltpow += 1
        self.rng(name, x * ipow, 0, 2**bits)


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
    """The transition unit: input wire ``c`` (the incoming FP32 word, a wire of the caller), private operands and hints,
    output the outgoing FP32 word as a linear form.  ``self.out`` is the Pack expression."""

    def __init__(self, p_field: int, p: Params = REAL) -> None:
        self.ctx = VoleCtx(p_field)
        ctx = self.ctx
        k = sum(p.groups)
        self.k = k
        # wire 0 is the function's input c
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
        self.n_private = ctx.n_priv - 1   # minus the input wire c

    def witness(self, c: int, a: Sequence[int], b: Sequence[int], h: dict) -> tuple[list[int], int, str | None]:
        """Values of the private wires in @private order, the output word (mod p) and the first failing assertion."""
        return _evaluate(self.ctx, {"c": c, "a": a, "b": b, "h": h}, self.out)


class Epi:
    """The epilogue: input wire y32, private hints, output y16 as a linear form."""

    def __init__(self, p_field: int, p: Params = REAL) -> None:
        self.ctx = VoleCtx(p_field)
        ctx = self.ctx
        self.y32 = ctx.priv(("input", "y32"))
        sample = Epilogue.gen(p, 0)
        self.paths: list[str] = []
        _leaves(sample, "h", self.paths)
        h_sym = copy.deepcopy(sample)
        for path in self.paths:
            _set(h_sym, path, ctx.priv(("hint", path)))
        self.out: Lin = Epilogue.check(ctx, p, self.y32, h_sym)
        self.n_private = ctx.n_priv - 1

    def witness(self, y32: int, h: dict) -> tuple[list[int], int, str | None]:
        return _evaluate(self.ctx, {"y32": y32, "h": h}, self.out)


def _evaluate(ctx: VoleCtx, env: dict, out: Lin) -> tuple[list[int], int, str | None]:
    p = ctx.p
    vals: list[int] = [0] * len(ctx.wires)

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
        elif kind == "bit":
            v = (ev(d[1]) >> d[2]) & 1
        else:  # sel
            _, tname, key, run = d
            kv = signed(ev(key))
            lo, n, _ = ctx.tables[tname][1][run]
            v = int(lo <= kv < lo + n)
        v = int(v) % p
        vals[i] = v
        if kind != "input":
            priv.append(v)
    bad = None
    for name, x in ctx.asserts:
        if ev(x) != 0:
            bad = name
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

    def lin(self, x: Lin, wire_of: list[int]) -> int:
        """Materialise a linear form; returns its wire id (a fresh wire, or the single wire when the form is one wire)."""
        terms = sorted(x.terms.items())
        if len(terms) == 1 and terms[0][1] == 1 and x.const == 0:
            return wire_of[terms[0][0]]
        acc = None
        for w, c in terms:
            src = wire_of[w]
            if c != 1:
                t = self.fresh()
                self.lines.append(f"${t} <- @mulc(0: ${src}, <{self.fe(c)}>);")
                self.n_gates += 1
                src = t
            if acc is None:
                acc = src
            else:
                t = self.fresh()
                self.lines.append(f"${t} <- @add(0: ${acc}, ${src});")
                self.n_gates += 1
                acc = t
        if x.const != 0 or acc is None:
            t = self.fresh()
            if acc is None:
                self.lines.append(f"${t} <- <{self.fe(x.const)}>;")
            else:
                self.lines.append(f"${t} <- @addc(0: ${acc}, <{self.fe(x.const)}>);")
            self.n_gates += 1
            acc = t
        return acc


def emit_function(name: str, ctx: VoleCtx, out: Lin, p: int) -> tuple[list[str], int]:
    """``@function(name, @out: 0:1, @in: 0:1)``: wire 0 is the output, wire 1 the input; body in wire order."""
    lines = [f"@function({name}, @out: 0:1, @in: 0:1)"]
    em = Emitter(p, lines)
    em.next = 2
    wire_of: list[int] = [0] * len(ctx.wires)
    for i, wire in enumerate(ctx.wires):
        if wire[0] == "priv":
            if wire[1][0] == "input":
                wire_of[i] = 1
                continue
            w = em.fresh()
            lines.append(f"${w} <- @private(0);")
            wire_of[i] = w
        else:
            a = em.lin(wire[1], wire_of)
            b = em.lin(wire[2], wire_of)
            w = em.fresh()
            lines.append(f"${w} <- @mul(0: ${a}, ${b});")
            em.n_gates += 1
            wire_of[i] = w
    for _, x in ctx.asserts:
        w = em.lin(x, wire_of)
        lines.append(f"@assert_zero(0: ${w});")
    # the output: wire 0 must be assigned exactly once
    terms = sorted(out.terms.items())
    if len(terms) == 1 and terms[0][1] == 1 and out.const == 0:
        lines.append("$0 <- $" + str(wire_of[terms[0][0]]) + ";")
    else:
        w = em.lin(out, wire_of)
        lines.append(f"$0 <- @addc(0: ${w}, <0x0>);")
    lines.append("@end")
    return lines, em.n_gates


def header(kind: str, p: int) -> list[str]:
    return ["version 2.0.0;", f"{kind};", f"@type field {hex(p)};", "@begin"]


def write_inputs(path: Path, kind: str, p: int, values: Sequence[int]) -> None:
    with open(path, "w") as f:
        f.write("\n".join(header(kind, p)) + "\n")
        f.writelines(f"<{hex(v % p)}>;\n" for v in values)
        f.write("@end\n")


def write_inputs_stream(path: Path, kind: str, p: int, chunks) -> int:
    n = 0
    with open(path, "w") as f:
        f.write("\n".join(header(kind, p)) + "\n")
        for values in chunks:
            f.write("".join(f"<{hex(v)}>;\n" for v in values))
            n += len(values)
        f.write("@end\n")
    return n


# ---------------------------------------------------------------------------------------------------------------------
# Relations
# ---------------------------------------------------------------------------------------------------------------------

def unit_relation(unit: Unit, n: int, io: str) -> list[str]:
    """n independent units: c and y as @public (io=public) or @private (io=private); operands are always private."""
    p = unit.ctx.p
    lines = header("circuit", p)
    body, _ = emit_function("unit", unit.ctx, unit.out, p)
    lines += body
    src = "@public(0)" if io == "public" else "@private(0)"
    w = 0
    for _ in range(n):
        lines.append(f"${w} <- {src};")
        lines.append(f"${w + 1} <- @call(unit, ${w});")
        lines.append(f"${w + 2} <- {src};")
        lines.append(f"${w + 3} <- @mulc(0: ${w + 2}, <{hex(p - 1)}>);")
        lines.append(f"${w + 4} <- @add(0: ${w + 1}, ${w + 3});")
        lines.append(f"@assert_zero(0: ${w + 4});")
        lines.append(f"@delete(0: ${w} ... ${w + 4});")
        w += 5
    lines.append("@end")
    return lines


def vu_relation(unit: Unit, epi: Epi, B: int, io: str) -> list[str]:
    """B VUs: c_0 = 0, 96 chained unit calls, the epilogue, y16 compared with the claimed word."""
    p = unit.ctx.p
    lines = header("circuit", p)
    body, _ = emit_function("unit", unit.ctx, unit.out, p)
    lines += body
    body, _ = emit_function("epilogue", epi.ctx, epi.out, p)
    lines += body
    src = "@public(0)" if io == "public" else "@private(0)"
    lines.append("@function(vu, @out: 0:1)")
    lines.append("  $1 <- <0x0>;")
    for s in range(UNITS_PER_VU):
        lines.append(f"  ${s + 2} <- @call(unit, ${s + 1});")
    lines.append(f"  $0 <- @call(epilogue, ${UNITS_PER_VU + 1});")
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
    """The private stream of one VU (chained through the circuit's own output words) and the first failing assertion."""
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

def census(field: str = "f61p") -> dict:
    p = FIELDS[field]
    unit = Unit(p)
    epi = Epi(p)
    body, gates = emit_function("unit", unit.ctx, unit.out, p)
    ebody, _egates = emit_function("epilogue", epi.ctx, epi.out, p)
    u = unit.ctx
    per_unit = {
        "private_wires": unit.n_private, "of_which_io": u.n_io - 1, "of_which_hints": u.n_hint, "of_which_bits": u.n_bits,
        "of_which_selectors": u.n_sel, "mul_gates": u.n_mul, "assert_zero": len(u.asserts),
        "vole_correlations": unit.n_private + u.n_mul, "lt_pow": u.n_ltpow, "lookups": dict(u.lookups),
        "table_runs": {t: len(runs) for t, (_, runs) in u.tables.items()}, "linear_gates_in_ir": gates - u.n_mul,
        "ir_lines": len(body), "bytes_prover_to_verifier_r1": 8 * (unit.n_private + u.n_mul),
    }
    per_epi = {"private_wires": epi.n_private, "mul_gates": epi.ctx.n_mul, "assert_zero": len(epi.ctx.asserts),
               "vole_correlations": epi.n_private + epi.ctx.n_mul, "ir_lines": len(ebody)}
    vu_vole = UNITS_PER_VU * per_unit["vole_correlations"] + per_epi["vole_correlations"] + 1
    vu_mul = UNITS_PER_VU * u.n_mul + epi.ctx.n_mul
    return {
        "field": field, "p_bits": p.bit_length(), "unit": per_unit, "epilogue": per_epi,
        "vu": {"units": UNITS_PER_VU, "vole_correlations": vu_vole, "mul_gates": vu_mul,
               "bytes_prover_to_verifier_r1": 8 * vu_vole,
               "bytes_prover_to_verifier_field": (p.bit_length() + 7) // 8 * vu_vole},
        "explore_vole_py": {"unit_logup": {"vole": 871, "mult_checks": 664}, "unit_bits": {"vole": 3971, "mult_checks": 3764},
                            "vu_logup": {"vole": 83434, "mult_checks": 63749, "bytes": 667_000},
                            "vu_bits": {"vole": 381_078, "mult_checks": 361_393}},
    }


# ---------------------------------------------------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------------------------------------------------

def cmd_census(args) -> int:
    print(json.dumps(census(args.field), indent=1))
    return 0


def cmd_unit(args) -> int:
    p = FIELDS[args.field]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    unit = Unit(p)
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
    meta = {"kind": "unit", "n": args.n, "field": args.field, "io": args.io, "seed": args.seed, "census": census(args.field)["unit"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "census"}))
    return 0


def _load_tier(root: str, tier: str):
    from verity_numerical.bench import instances
    return instances.load(tier, root)


def cmd_tu(args) -> int:
    p = FIELDS[args.field]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    unit = Unit(p)
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
    meta = {"kind": "tu-k16", "n": used, "skipped_out_of_domain": rejected, "field": args.field, "io": args.io}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0


def _vu_task(task):
    i, a, b, y, field, io = task
    p = FIELDS[field]
    global _UNIT, _EPI
    if _UNIT is None or _UNIT.ctx.p != p:
        _UNIT, _EPI = Unit(p), Epi(p)
    priv, bad = vu_witness(_UNIT, _EPI, a, b, y, io)
    text = "".join(f"<{hex(v)}>;\n" for v in priv)
    return i, len(priv), text, bad


_UNIT: Unit | None = None
_EPI: Epi | None = None


def cmd_vu(args) -> int:
    p = FIELDS[args.field]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    unit, epi = Unit(p), Epi(p)
    tier = _load_tier(args.root, "vu-k1536")
    B = args.B
    tasks = []
    for i in range(B):
        a, b, y = tier.instance(i)
        tasks.append((i, a, b, y, args.field, args.io))
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
                    if bad is not None:
                        print(f"circuit rejects honest VU {i}: {bad}", file=sys.stderr)
                        return 1
                    f.write(text)
                    n_wit += n_priv
        else:
            for task in tasks:
                i, n_priv, text, bad = _vu_task(task)
                if bad is not None:
                    print(f"circuit rejects honest VU {i}: {bad}", file=sys.stderr)
                    return 1
                f.write(text)
                n_wit += n_priv
        f.write("@end\n")
    meta = {"kind": "vu-k1536", "B": B, "instances": f"vu-k1536[0:{B}]", "field": args.field, "io": args.io,
            "private_values": n_wit, "private_values_per_vu": n_wit / B, "export_seconds": time.time() - t0,
            "census_vu": census(args.field)["vu"]}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps(meta))
    return 0


def cmd_negatives(args) -> int:
    p = FIELDS[args.field]
    unit, epi = Unit(p), Epi(p)
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
    c = sub.add_parser("census")
    c.add_argument("--field", default="f61p", choices=sorted(FIELDS))
    c.set_defaults(fn=cmd_census)
    u = sub.add_parser("unit")
    u.add_argument("out")
    u.add_argument("--n", type=int, default=1)
    u.add_argument("--seed", type=int, default=20260922)
    u.add_argument("--field", default="f61p", choices=sorted(FIELDS))
    u.add_argument("--io", default="private", choices=("private", "public"))
    u.set_defaults(fn=cmd_unit)
    t = sub.add_parser("tu")
    t.add_argument("out")
    t.add_argument("--root", required=True)
    t.add_argument("--n", type=int, default=4096)
    t.add_argument("--field", default="f61p", choices=sorted(FIELDS))
    t.add_argument("--io", default="private", choices=("private", "public"))
    t.set_defaults(fn=cmd_tu)
    v = sub.add_parser("vu")
    v.add_argument("out")
    v.add_argument("--root", required=True)
    v.add_argument("--B", type=int, default=1)
    v.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    v.add_argument("--field", default="f61p", choices=sorted(FIELDS))
    v.add_argument("--io", default="private", choices=("private", "public"))
    v.set_defaults(fn=cmd_vu)
    n = sub.add_parser("negatives")
    n.add_argument("--root", default=None)
    n.add_argument("--field", default="f61p", choices=sorted(FIELDS))
    n.add_argument("--io", default="private", choices=("private", "public"))
    n.set_defaults(fn=cmd_negatives)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
