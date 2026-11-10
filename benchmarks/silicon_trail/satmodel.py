"""Cancelling trails as a satisfiability problem.

`build(...)` writes, for a width, a placement, a number of steps k and a bound W, a CNF that is satisfiable exactly when
a k-step cancelling trail (diff.Trail) of weight at most W exists; `decode` turns a model into the Trail and checks it
with code that shares nothing with the encoding.  Three other modes use the same encoding of the round and are how the
encoding is checked against published numbers and by hand:

  cancel   the chain: a_1 in the absorbed lanes, a_{i+1} = c_i xor (anything in the absorbed lanes), c_{k-1} in the
           absorbed lanes; k - 1 rounds
  free     a plain trail of Keccak-p over `rounds` rounds: nothing absorbed, both ends free (the published trail bounds)
  rows     one round from the absorbed lanes, counting active rows of b_1 instead of weight

The encoding.  Per round: the difference a (25w variables), the column parities, theta's effect, b = L(a) (wiring), the
difference c after chi, and per row of b three literals g2, g3, g4 that the row's weight forces (weight >= 2, 3, 4).
A row's (b, c) pairs are chi's difference table as a minimal clause cover (prime implicants of the table's complement).
The weight literals feed a unary adder tree (a totalizer, upward direction only), capped at W + 1, whose top literal
is denied.  The g literals are only forced upward, so a model's true weight can be below the bound and never above.
Translation along z is a symmetry of everything here; it is broken by requiring an active bit of a_1 in slice 0.
Each a_i is nonzero, so a trail of k steps is not a shorter one padded.
"""

from __future__ import annotations

import itertools
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import diff
import keccak as K


def prime_cover(nvars: int, off: set[int]) -> list[tuple[int, int]]:
    """Cubes (mask, value) inside `off` that together cover it: each denies v when v & mask == value."""
    full = (1 << nvars) - 1
    inside: set[tuple[int, int]] = set()
    for mask in range(full + 1):
        free = [i for i in range(nvars) if not mask >> i & 1]
        spans = [sum(1 << f for f, on in zip(free, bits) if on) for bits in itertools.product((0, 1), repeat=len(free))]
        sub = mask
        while True:
            if all((sub | s) in off for s in spans):
                inside.add((mask, sub))
            if sub == 0:
                break
            sub = (sub - 1) & mask
    primes = [(m, v) for (m, v) in inside
              if not any(((m & ~(1 << i)), v & ~(1 << i)) in inside for i in range(nvars) if m >> i & 1)]
    points = {p: [c for c in primes if p & c[0] == c[1]] for p in off}
    left, cover = set(off), []
    while left:
        best = max(primes, key=lambda c: sum(1 for p in left if p & c[0] == c[1]))
        cover.append(best)
        left = {p for p in left if p & best[0] != best[1]}
    assert all(any(p & m == v for m, v in cover) for p in points)
    return cover


def _row_clauses() -> list[tuple[int, int]]:
    allowed = {b | (c << 5) for b in range(32) for c in diff.ROW_OUT[b]}
    return prime_cover(10, set(range(1024)) - allowed)


def _weight_cubes(t: int) -> list[tuple[int, int]]:
    """Cubes over the 5 bits of a row of b covering exactly the rows of weight >= t."""
    on = {b for b in range(32) if diff.ROW_WEIGHT[b] >= t}
    # cubes inside `on` covering it = prime_cover with the roles swapped
    return prime_cover(5, on)


ROW_CLAUSES = _row_clauses()
WEIGHT_CUBES = {t: _weight_cubes(t) for t in (2, 3, 4)}


@dataclass
class Model:
    w: int
    placement: str
    mode: str
    rounds: int
    bound: int
    nvars: int = 0
    clauses: list[tuple[int, ...]] = field(default_factory=list)
    a: list[list[int]] = field(default_factory=list)      # per round, 25w variables
    c: list[list[int]] = field(default_factory=list)
    round_units: list[list[int]] = field(default_factory=list)   # per round, unary weight literals

    def new(self) -> int:
        self.nvars += 1
        return self.nvars

    def add(self, *lits: int) -> None:
        self.clauses.append(lits)

    def xor2(self, p: int, q: int) -> int:
        r = self.new()
        self.add(-p, -q, -r), self.add(p, q, -r), self.add(p, -q, r), self.add(-p, q, r)
        return r

    def xor(self, lits: list[int]) -> int:
        acc = lits[0]
        for v in lits[1:]:
            acc = self.xor2(acc, v)
        return acc

    def merge(self, left: list[int], right: list[int], cap: int) -> list[int]:
        n = min(len(left) + len(right), cap)
        out = [self.new() for _ in range(n)]
        for i in range(len(left) + 1):
            for j in range(len(right) + 1):
                if 0 < i + j <= n:
                    self.add(*([-left[i - 1]] if i else []), *([-right[j - 1]] if j else []), out[i + j - 1])
        return out

    def total(self, leaves: list[list[int]], cap: int) -> list[int]:
        level = [leaf[:cap] for leaf in leaves]
        while len(level) > 1:
            nxt = [self.merge(level[i], level[i + 1], cap) for i in range(0, len(level) - 1, 2)]
            if len(level) % 2:
                nxt.append(level[-1])
            level = nxt
        return level[0]

    def dimacs(self) -> str:
        head = f"p cnf {self.nvars} {len(self.clauses)}\n"
        return head + "".join(" ".join(map(str, cl)) + " 0\n" for cl in self.clauses)


def _bit(x: int, y: int, z: int, w: int) -> int:
    return w * (5 * y + x) + z


def _wiring(w: int) -> list[int]:
    """src[j] = the bit of theta's output that rho and pi move to bit j."""
    src = [0] * (25 * w)
    for x in range(5):
        for y in range(5):
            for z in range(w):
                src[_bit(y, (2 * x + 3 * y) % 5, (z + K.RHO[x][y]) % w, w)] = _bit(x, y, z, w)
    return src


def build(w: int, placement: str, bound: int, *, k: int = 0, rounds: int = 0, mode: str = "cancel",
          round_max: list[int] | None = None) -> Model:
    """The CNF.  cancel: k steps (k - 1 rounds).  free: `rounds` rounds.  rows: one round, `bound` active rows."""
    assert mode in ("cancel", "free", "rows")
    nr = {"cancel": k - 1, "free": rounds, "rows": 1}[mode]
    assert nr >= 1
    m = Model(w, placement, mode, nr, bound)
    nbits = 25 * w
    outer = {_bit(x, y, z, w) for (x, y) in K.PLACEMENTS[placement] for z in range(w)}
    src = _wiring(w)
    prev_c: list[int] | None = None
    row_leaves: list[list[list[int]]] = []
    for r in range(nr):
        if prev_c is None:
            a = [m.new() for _ in range(nbits)]
            if mode != "free":
                for j in range(nbits):
                    if j not in outer:
                        m.add(-a[j])
        elif mode == "free":
            a = prev_c
        else:
            a = [m.new() if j in outer else prev_c[j] for j in range(nbits)]
        m.add(*a)
        par = [[m.xor([a[_bit(x, y, z, w)] for y in range(5)]) for z in range(w)] for x in range(5)]
        eff = [[m.xor2(par[(x - 1) % 5][z], par[(x + 1) % 5][(z - 1) % w]) for z in range(w)] for x in range(5)]
        t = {_bit(x, y, z, w): m.xor2(a[_bit(x, y, z, w)], eff[x][z])
             for y in range(5) for x in range(5) for z in range(w)}
        b = [t[src[j]] for j in range(nbits)]
        c = [m.new() for _ in range(nbits)]
        leaves = []
        for y in range(5):
            for z in range(w):
                bv = [b[_bit(x, y, z, w)] for x in range(5)]
                cv = [c[_bit(x, y, z, w)] for x in range(5)]
                both = bv + cv
                for mask, val in ROW_CLAUSES:
                    m.add(*[(-both[i] if val >> i & 1 else both[i]) for i in range(10) if mask >> i & 1])
                g = {tt: m.new() for tt in (2, 3, 4)}
                for tt in (2, 3, 4):
                    for mask, val in WEIGHT_CUBES[tt]:
                        m.add(*[(-bv[i] if val >> i & 1 else bv[i]) for i in range(5) if mask >> i & 1], g[tt])
                leaves.append([g[2]] if mode == "rows" else [g[2], g[2], g[3], g[4]])
        row_leaves.append(leaves)
        m.a.append(a)
        m.c.append(c)
        prev_c = c
    if mode == "cancel":
        for j in range(nbits):
            if j not in outer:
                m.add(-m.c[-1][j])
    # one active bit of a_1 in slice 0
    first = [m.a[0][_bit(x, y, 0, w)] for x in range(5) for y in range(5)]
    m.add(*first)
    cap = bound + 1
    for leaves in row_leaves:
        m.round_units.append(m.total(leaves, cap))
    top = m.round_units[0]
    for u in m.round_units[1:]:
        top = m.merge(top, u, cap)
    if len(top) > bound:
        m.add(-top[bound])
    if round_max:
        for u, mx in zip(m.round_units, round_max):
            if mx is not None and len(u) > mx:
                m.add(-u[mx])
    return m


def _state(model_bits: list[bool], vs: list[int], w: int) -> list[int]:
    lanes = [0] * 25
    for j, v in enumerate(vs):
        if model_bits[v]:
            lanes[j // w] |= 1 << (j % w)
    return lanes


def decode(m: Model, assignment: list[bool]) -> diff.Trail | dict:
    """The model as a checked Trail (cancel), or the bare differences (free, rows) checked round by round."""
    a = [_state(assignment, vs, m.w) for vs in m.a]
    c = [_state(assignment, vs, m.w) for vs in m.c]
    if m.mode == "cancel":
        t = diff.trail_from_states(m.w, m.placement, a, c)
        assert t.weight <= m.bound, (t.weight, m.bound)
        return t
    b = [K.linear(s, m.w) for s in a]
    for i in range(m.rounds):
        assert diff.compatible(b[i], c[i], m.w)
        if i + 1 < m.rounds:
            assert a[i + 1] == c[i]
    weights = [diff.weight(s, m.w) for s in b]
    rows = [diff.active_rows(s, m.w) for s in b]
    if m.mode == "rows":
        assert diff.is_outer(a[0], m.placement) and any(a[0]) and rows[0] <= m.bound
    else:
        assert all(any(s) for s in a) and sum(weights) <= m.bound
    return {"w": m.w, "mode": m.mode, "a": [f"{K.to_int(s, m.w):x}" for s in a], "weights": weights, "rows": rows}


def solve(m: Model, solver: str, workdir: Path, tag: str, timeout: float | None = None,
          extra: tuple[str, ...] = ()) -> tuple[str, list[bool] | None, float]:
    """Run a DIMACS solver (kissat's or CaDiCaL's command line).  ('SAT' | 'UNSAT' | 'UNKNOWN', assignment, seconds)."""
    path = workdir / f"{tag}.cnf"
    path.write_text(m.dimacs())
    t0 = time.monotonic()
    try:
        p = subprocess.run([solver, "-q", *extra, str(path)], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "UNKNOWN", None, time.monotonic() - t0
    dt = time.monotonic() - t0
    if p.returncode == 20:
        return "UNSAT", None, dt
    if p.returncode != 10:
        return "UNKNOWN", None, dt
    return "SAT", parse_model(p.stdout, m.nvars), dt


def parse_model(text: str, nvars: int) -> list[bool]:
    val = [False] * (nvars + 1)
    for line in text.splitlines():
        if line.startswith("v "):
            for tok in line.split()[1:]:
                v = int(tok)
                if v > 0:
                    val[v] = True
    return val
