"""Adversaries that keep part of the encoding, replayed against the real labels.

A strategy is a `Holding` per segment (Lean's `ColState`): whole lower labels, lower columns `(w, u)` (component `u` of
lower label `w`) and whole top labels.  The preprocessor sees everything and keeps exactly those values; online, with
`W` and the salt free, a challenge is answered by `answer_prog`, a real oracle program whose rounds `oracle.run`
counts.

`schedule` computes, for a holding, the fewest oracle rounds after which each value is known, over every way to get it:

* a key `H(t, layer, v, parents)`: at the round its derivation's last query can be answered, each query waiting for
  the parent it reads (`primitives.key_rounds`: `j + 1` rounds for the overwrite chain's `j` newest-first absorptions
  and its pad call, 1 for a monolithic oracle);
* a lower label: held; forward, `P_(t,0,w)(W_w ⊕ key)`, one round after its key; or promoted, locally, once all `n` of
  its columns are held or revealed;
* a column `(w, u)`: held; from lower `w`; or revealed by the reverse of top `u`, `P_(t,1,u)⁻¹(c²_u) ⊕ key`, one round
  after `c²_u` once its key is known too (the key gate: the key reads `u`'s parents' whole top labels);
* a top label: held; or forward, `P_(t,1,u)(x²_u ⊕ key)`, one round after its key and all `n` of its columns.

These are the moves of Lean's two-directional column game (`colRound`), priced in oracle rounds rather than game
rounds.  `answer_prog` issues exactly the queries of the cheapest derivation, each at the round `schedule` gives it, so
a measured depth equal to `schedule`'s is a check of both.  `col_round`, `col_reach` and `col_win` transcribe the Lean
game itself for comparison.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .labels import component, from_bytes, join, salt_value, tag, w_labels
from .oracle import PinvQuery, PQuery, Prog, Tweak, run
from .params import Params
from .primitives import KeyDerivation, Primitives, key_rounds
from .scheme import encode_prog


@dataclass(frozen=True, slots=True)
class Holding:
    """Lean's `ColState n`."""

    lower_whole: frozenset[int] = frozenset()
    lower_col: frozenset[tuple[int, int]] = frozenset()
    top_whole: frozenset[int] = frozenset()

    def size(self, n: int) -> int:
        """Lean `ColState.size`, in columns: `n` per whole label, 1 per column."""
        return n * len(self.lower_whole) + len(self.lower_col) + n * len(self.top_whole)

    def saving(self, n: int) -> float:
        """Labels saved against holding all `n` top labels."""
        return n - self.size(n) / n


def col_round(params: Params, c: Holding) -> Holding:
    """Lean `colRound`: forward-lower or promotion, key-gated reverse, forward-top, all from the previous holding."""
    n = params.n
    par = [set(params.parents(v)) for v in range(n)]
    lower_whole = c.lower_whole | {w for w in range(n)
                                   if par[w] <= c.lower_whole or all((w, u) in c.lower_col for u in range(n))}
    lower_col = c.lower_col | {(w, u) for w in range(n) for u in c.top_whole if par[u] <= c.top_whole}
    top_whole = c.top_whole | {u for u in range(n) if all(w in c.lower_whole or (w, u) in c.lower_col for w in range(n))
                               and par[u] <= c.top_whole}
    return Holding(frozenset(lower_whole), frozenset(lower_col), frozenset(top_whole))


def col_reach(params: Params, c0: Holding, rounds: int) -> Holding:
    c = c0
    for _ in range(rounds):
        c = col_round(params, c)
    return c


def col_win(params: Params, c0: Holding, rounds: int) -> bool:
    """Lean `ColWin`: every top label within `rounds` game rounds."""
    return len(col_reach(params, c0, rounds).top_whole) == params.n


def game_rounds(params: Params, c0: Holding, limit: int | None = None) -> int | None:
    """The fewest game rounds to `ColWin`, or `None` within `limit` (default `4n`)."""
    c = c0
    for r in range(4 * params.n + 1 if limit is None else limit + 1):
        if len(c.top_whole) == params.n:
            return r
        c = col_round(params, c)
    return None


@dataclass(frozen=True, slots=True)
class Schedule:
    """The oracle round at which each value of one segment is first known from `holding` (0 when held), and how.

    `lower_how` is `held`, `fwd` or `prom`; `top_how` is `held` or `fwd`; `col_how[w][u]` is `held`, `lower` or `rev`.
    `None` means never (a top whose parents and columns can't all be had).
    """

    params: Params
    holding: Holding
    h: KeyDerivation
    lower: tuple[int | None, ...]
    lower_how: tuple[str | None, ...]
    lower_key: tuple[int | None, ...]
    top: tuple[int | None, ...]
    top_how: tuple[str | None, ...]
    top_key: tuple[int | None, ...]
    reverse: tuple[int | None, ...]
    col_how: tuple[tuple[str | None, ...], ...]

    def depth(self, targets: Iterable[int]) -> int:
        """Rounds to know every top in `targets`."""
        out = 0
        for u in targets:
            if self.top[u] is None:
                raise ValueError(f"top label {u} is unreachable from the holding")
            out = max(out, self.top[u])
        return out


def schedule(params: Params, holding: Holding, h: KeyDerivation) -> Schedule:
    """Knuth's generalization of Dijkstra: each value settles at its fewest rounds, in increasing order."""
    n = params.n
    par = [list(params.parents(v)) for v in range(n)]
    ch = [list(params.children(v)) for v in range(n)]
    held_cols = [set() for _ in range(n)]
    for w, u in holding.lower_col:
        held_cols[w].add(u)
    lower: list[int | None] = [None] * n
    lower_how: list[str | None] = [None] * n
    lower_key: list[int | None] = [None] * n
    top: list[int | None] = [None] * n
    top_how: list[str | None] = [None] * n
    top_key: list[int | None] = [None] * n
    reverse: list[int | None] = [None] * n
    col: list[list[int | None]] = [[None] * n for _ in range(n)]
    col_how: list[list[str | None]] = [[None] * n for _ in range(n)]
    k1_wait = [len(par[w]) for w in range(n)]
    k2_wait = [len(par[u]) for u in range(n)]
    top_missing, top_max = [n] * n, [0] * n
    prom_missing, prom_max = [n - len(held_cols[w]) for w in range(n)], [0] * n
    heap: list = []
    seq = itertools.count()

    def push(d: int, kind: str, x: int, how: str | None = None) -> None:
        heapq.heappush(heap, (d, next(seq), kind, x, how))

    def settle_col(w: int, u: int, d: int, how: str) -> None:
        col[w][u], col_how[w][u] = d, how
        top_missing[u] -= 1
        top_max[u] = max(top_max[u], d)
        if top_missing[u] == 0 and top_key[u] is not None and top[u] is None:
            push(1 + max(top_key[u], top_max[u]), "top", u, "fwd")

    for w in range(n):
        for u in held_cols[w]:
            settle_col(w, u, 0, "held")
    for w in holding.lower_whole:
        push(0, "lower", w, "held")
    for u in holding.top_whole:
        push(0, "top", u, "held")
    for v in range(n):
        if not par[v]:
            push(key_rounds(h, [], {}), "k1", v)
            push(key_rounds(h, [], {}), "k2", v)
        if prom_missing[v] == 0:
            push(0, "lower", v, "prom")

    while heap:
        d, _, kind, x, how = heapq.heappop(heap)
        if kind == "lower":
            if lower[x] is not None:
                continue
            lower[x], lower_how[x] = d, how
            for c in ch[x]:
                k1_wait[c] -= 1
                if k1_wait[c] == 0:
                    push(key_rounds(h, par[c], {p: lower[p] for p in par[c]}), "k1", c)
            for u in range(n):
                if col[x][u] is None:
                    settle_col(x, u, d, "lower")
        elif kind == "k1":
            if lower_key[x] is not None:
                continue
            lower_key[x] = d
            push(d + 1, "lower", x, "fwd")
        elif kind == "top":
            if top[x] is not None:
                continue
            top[x], top_how[x] = d, how
            for c in ch[x]:
                k2_wait[c] -= 1
                if k2_wait[c] == 0:
                    push(key_rounds(h, par[c], {p: top[p] for p in par[c]}), "k2", c)
            if top_key[x] is not None:
                push(max(d + 1, top_key[x]), "rev", x)
        elif kind == "k2":
            if top_key[x] is not None:
                continue
            top_key[x] = d
            if top[x] is not None:
                push(max(top[x] + 1, d), "rev", x)
            elif top_missing[x] == 0:
                push(1 + max(d, top_max[x]), "top", x, "fwd")
        else:
            if reverse[x] is not None:
                continue
            reverse[x] = d
            for w in range(n):
                if col[w][x] is None:
                    settle_col(w, x, d, "rev")
                if x not in held_cols[w]:
                    prom_missing[w] -= 1
                    prom_max[w] = max(prom_max[w], d)
                    if prom_missing[w] == 0:
                        push(prom_max[w], "lower", w, "prom")

    return Schedule(params, holding, h, tuple(lower), tuple(lower_how), tuple(lower_key), tuple(top), tuple(top_how),
                    tuple(top_key), tuple(reverse), tuple(tuple(row) for row in col_how))


@dataclass(frozen=True, slots=True)
class SegmentValues:
    """The held values of one segment."""

    lower_whole: dict[int, int]
    lower_col: dict[tuple[int, int], int]
    top_whole: dict[int, int]


def extract(params: Params, holding: Holding, lowers: Sequence[int], tops: Sequence[int]) -> SegmentValues:
    return SegmentValues({w: lowers[w] for w in holding.lower_whole},
                         {(w, u): component(lowers[w], u, params.wc) for w, u in holding.lower_col},
                         {u: tops[u] for u in holding.top_whole})


def _needed(s: Schedule, targets: Iterable[int]) -> tuple[set[int], set[int], set[int]]:
    """The tops, lowers and reverses on the cheapest derivation of `targets`."""
    params, n = s.params, s.params.n
    held_cols = {(w, u) for w, u in s.holding.lower_col}
    tops, lowers, reverses = set(), set(), set()
    stack = [("top", u) for u in targets]
    while stack:
        kind, x = stack.pop()
        if kind == "top":
            if x in tops:
                continue
            if s.top[x] is None:
                raise ValueError(f"top label {x} is unreachable from the holding")
            tops.add(x)
            if s.top_how[x] == "fwd":
                stack += [("top", p) for p in params.parents(x)]
                for w in range(n):
                    if s.col_how[w][x] == "lower":
                        stack.append(("lower", w))
                    elif s.col_how[w][x] != "held":
                        raise AssertionError("a top's own reverse cannot supply its columns")
        elif kind == "lower":
            if x in lowers:
                continue
            lowers.add(x)
            if s.lower_how[x] == "fwd":
                stack += [("lower", p) for p in params.parents(x)]
            elif s.lower_how[x] == "prom":
                stack += [("rev", u) for u in range(n) if (x, u) not in held_cols]
        else:
            if x in reverses:
                continue
            reverses.add(x)
            stack += [("top", x)] + [("top", p) for p in params.parents(x)]
    return tops, lowers, reverses


class _Labels(Mapping[int, int]):
    """Parent labels fetched on demand, so a key step reads only the parent it absorbs."""

    def __init__(self, get: Callable[[int], int]) -> None:
        self.get_label = get

    def __getitem__(self, u: int) -> int:
        return self.get_label(u)

    def __iter__(self):
        raise TypeError("parent labels are read one at a time")

    def __len__(self) -> int:
        raise TypeError("parent labels are read one at a time")


def answer_prog(s: Schedule, t: int, w: Sequence[int], values: SegmentValues,
                targets: Collection[int]) -> Prog[dict[int, int]]:
    """The top labels `targets` of one segment (tag `t`, lower inputs `w`) from its held values, in exactly
    `s.depth(targets)` rounds: every query is issued at the round `schedule` gives it."""
    params, n, wc, h = s.params, s.params.n, s.params.wc, s.h
    tops, lowers, reverses = _needed(s, targets)
    ops: dict[int, list[tuple]] = {}

    def at(r: int, op: tuple) -> None:
        ops.setdefault(r, []).append(op)

    steppers = {}

    def plan_key(layer: int, x: int) -> None:
        par = list(params.parents(x))
        ready = s.lower if layer == 0 else s.top
        steppers[(layer, x)] = h.stepper(t, layer, x, par)
        r = 0
        for needs in h.steps(par):
            r = max([r] + [ready[u] for u in needs]) + 1
            at(r, ("key", layer, x))

    for x in sorted(lowers):
        if s.lower_how[x] == "fwd":
            plan_key(0, x)
            at(s.lower[x], ("p1", x))
    for u in sorted({u for u in tops if s.top_how[u] == "fwd"} | reverses):
        plan_key(1, u)
    for u in sorted(tops):
        if s.top_how[u] == "fwd":
            at(s.top[u], ("p2", u))
    for u in sorted(reverses):
        at(s.top[u] + 1, ("pinv", u))

    low = dict(values.lower_whole)
    tp = dict(values.top_whole)
    keys: dict[tuple[int, int], int] = {}
    inv: dict[int, int] = {}
    revealed: dict[int, int] = {}

    def lower_label(x: int) -> int:
        if x not in low:
            if s.lower_how[x] != "prom":
                raise AssertionError(f"lower label {x} used before its round")
            low[x] = join([values.lower_col[(x, u)] if (x, u) in values.lower_col else component(revealed[u], x, wc)
                           for u in range(n)], wc)
        return low[x]

    def column(x: int, u: int) -> int:
        how = s.col_how[x][u]
        if how == "held":
            return values.lower_col[(x, u)]
        if how == "lower":
            return component(lower_label(x), u, wc)
        return component(revealed[u], x, wc)

    lower_labels, top_labels = _Labels(lower_label), _Labels(lambda u: tp[u])
    for r in range(1, max(ops, default=0) + 1):
        batch = ops.get(r, [])
        queries = []
        for op in batch:
            if op[0] == "key":
                queries.append(steppers[op[1:]].query(lower_labels if op[1] == 0 else top_labels))
            elif op[0] == "p1":
                queries.append(PQuery(Tweak(t, 0, op[1]), w[op[1]] ^ keys[(0, op[1])]))
            elif op[0] == "p2":
                x2 = join([column(v, op[1]) for v in range(n)], wc)
                queries.append(PQuery(Tweak(t, 1, op[1]), x2 ^ keys[(1, op[1])]))
            else:
                queries.append(PinvQuery(Tweak(t, 1, op[1]), tp[op[1]]))
        answers = yield queries
        for op, a in zip(batch, answers, strict=True):
            if op[0] == "key":
                st = steppers[op[1:]]
                st.feed(a)
                if st.done:
                    keys[op[1:]] = st.key
            elif op[0] == "p1":
                low[op[1]] = a
            elif op[0] == "p2":
                tp[op[1]] = a
            else:
                inv[op[1]] = a
        for u in reverses:
            if u not in revealed and u in inv and (1, u) in keys:
                revealed[u] = inv[u] ^ keys[(1, u)]
    return {u: tp[u] for u in targets}


@dataclass(frozen=True, slots=True)
class View:
    """What the preprocessor sees: everything, including each segment's lower labels."""

    params: Params
    prims: Primitives
    w: int
    salt: bytes
    lowers: tuple[tuple[int, ...], ...]
    tops: tuple[tuple[int, ...], ...]


def view(params: Params, prims: Primitives, w: bytes, salt: bytes) -> View:
    layers, _ = run(encode_prog(params, prims.h, from_bytes(w, params.w_bits), salt_value(salt)), prims)
    return View(params, prims, from_bytes(w, params.w_bits), salt, tuple(tuple(c1) for c1, _ in layers),
                tuple(tuple(c2) for _, c2 in layers))


@dataclass(slots=True)
class HoldingState:
    """A responder that keeps one holding's values per segment and answers from them (`audit.Responder`)."""

    params: Params
    holding: Holding
    h: KeyDerivation
    salt: int
    w: list[list[int]]
    values: list[SegmentValues]
    sched: Schedule = field(init=False)

    def __post_init__(self) -> None:
        self.sched = schedule(self.params, self.holding, self.h)

    @property
    def bits(self) -> int:
        """The state's size: `Holding.size` columns of `wc` bits per segment (`W` and the salt are free)."""
        return self.params.segments * self.holding.size(self.params.n) * self.params.wc

    def rounds(self, i: int) -> int:
        """The rounds `answer` will take for index `i`."""
        return self.sched.depth([i % self.params.n])

    def answer(self, j: int, i: int) -> Prog[int]:
        s, v = divmod(i, self.params.n)
        out = yield from answer_prog(self.sched, tag(self.salt, s), self.w[s], self.values[s], [v])
        return out[v]


def preprocess(v: View, holding: Holding) -> HoldingState:
    values = [extract(v.params, holding, v.lowers[s], v.tops[s]) for s in range(v.params.segments)]
    return HoldingState(v.params, holding, v.prims.h, salt_value(v.salt), w_labels(v.params, v.w), values)


def honest(params: Params) -> Holding:
    """Every top label: the honest prover's `C`."""
    return Holding(top_whole=frozenset(range(params.n)))


def recompute(params: Params) -> Holding:
    """Nothing but `W`: every answer re-encodes."""
    return Holding()


def dues(params: Params, h: KeyDerivation, dropped: Collection[int], deadline: int) -> dict[int, int]:
    """The round by which each dropped top must be known so that every dropped top meets `deadline`: a dropped parent
    of `c` must be known early enough for the rest of `c`'s key and `c`'s own `P` call."""
    due: dict[int, int] = {}
    for u in sorted(dropped, reverse=True):
        d = deadline
        for c in params.children(u):
            if c in due:
                steps = h.steps(list(params.parents(c)))
                first = next(i for i, needs in enumerate(steps) if u in needs)
                d = min(d, due[c] - 1 - (len(steps) - first))
        due[u] = d
    return due


def column_storage(params: Params, h: KeyDerivation, dropped: Collection[int], deadline: int,
                   lower_whole: Collection[int] = (), promoted: Collection[int] = ()) -> Holding:
    """Drop the top labels `dropped` and keep, for each, the columns it cannot rebuild by its due round (`dues`).

    The column-hole family of `docs/p3-cryptanalysis.md`: attack 1 (`lower_whole = promoted = ∅`), whole lower labels as
    separators (`lower_whole`), and meet in the middle (`promoted`: each promoted lower keeps its columns under the
    tops that cannot be reversed from held values, and is promoted as soon as the others are revealed).  Raises
    `ValueError` when some dropped top cannot meet `deadline`.
    """
    n = params.n
    dropped = set(dropped)
    tops = frozenset(range(n)) - dropped
    reversible = {c for c in tops if all(p in tops for p in params.parents(c))}
    cols = {(w, c) for w in promoted for c in range(n) if c not in reversible}
    base = schedule(params, Holding(frozenset(lower_whole), frozenset(cols), tops), h)
    for u, due in dues(params, h, dropped, deadline).items():
        cols |= {(w, u) for w in range(n) if w not in lower_whole and (base.lower[w] is None or base.lower[w] > due - 1)}
    holding = Holding(frozenset(lower_whole), frozenset(cols), tops)
    final = schedule(params, holding, h)
    late = [u for u in dropped if final.top[u] is None or final.top[u] > deadline]
    if late:
        raise ValueError(f"tops {sorted(late)} cannot meet {deadline} rounds")
    return holding


def staggered_runs(params: Params, h: KeyDerivation, deadline: int, run_length: int) -> Holding:
    """Attack 1 at its best on the band (`band_crosscheck.py`'s block seed): drop runs of `run_length` tops, `k` held
    tops apart, each run a chain recovered as late as its deadline allows."""
    k = max(params.k, 1)
    return column_storage(params, h, [v for v in range(params.n) if v % (run_length + k) < run_length], deadline)


@dataclass(frozen=True, slots=True)
class Measurement:
    """One strategy against one encoding: its state, and every answer's verdict and rounds."""

    name: str
    bits: int
    fraction: float
    saving: float
    rounds: tuple[int, ...]
    correct: bool
    work: int
    inverse: int
    game_rounds: int | None


def measure(name: str, v: View, holding: Holding, indices: Iterable[int] | None = None,
            game_limit: int | None = None) -> Measurement:
    """Preprocess `holding` from `v`, answer every index (default: all `B`), and compare with the true blocks.
    `game_rounds` is Lean's game on the same holding, searched up to `game_limit` (default twice the most rounds an
    answer took, plus 2)."""
    state = preprocess(v, holding)
    params = v.params
    rounds, correct, work, inverse = [], True, 0, 0
    for i in range(params.blocks) if indices is None else indices:
        value, cost = run(state.answer(0, i), v.prims)
        if cost.depth != state.rounds(i):
            raise AssertionError(f"index {i}: measured {cost.depth} rounds, scheduled {state.rounds(i)}")
        correct &= value == v.tops[i // params.n][i % params.n]
        rounds.append(cost.depth)
        work = max(work, cost.work)
        inverse = max(inverse, cost.inverse)
    limit = 2 * max(rounds, default=0) + 2 if game_limit is None else game_limit
    return Measurement(name, state.bits, state.bits / (params.blocks * params.m), holding.saving(params.n),
                       tuple(rounds), correct, work, inverse, game_rounds(params, holding, limit))
