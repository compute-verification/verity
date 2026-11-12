"""`verity.definitions.circuits.boolean.memcheck`: MemCheck_v1 accepts exactly the consistent logs (`consistent`, the multisets
themselves) on random small logs and on each with one thing wrong, the circuits agree with their references, and on the RAM
pool's own logs (`fixtures/memcheck_pool.json`, three windows of `pool_ram`'s log over the encodings' traces with every
operation kind and evictions, written by `benchmarks/mempool/vectors.py`) the parts' fingerprints are the ones pool_ram's own
field arithmetic gives, while each tampered window fails: a read's value or time changed, a read at its now, an access
dropped, an unused position not zero, and a list entry dropped, swapped, repeated, its initial value changed or used after
an unused one."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from verity.definitions.circuits.boolean import memcheck as MC
from verity.circuits.boolean.gf128 import mul
from verity.circuits.defs import bind
from verity.circuits.evaluation.bits import evaluate_bits

Access, Entry = MC.Access, MC.Entry
POOL = json.loads((Path(__file__).with_name("fixtures") / "memcheck_pool.json").read_text())
PL = MC.Widths(**POOL["layout"])
#: (Definition, statics) -> its traced circuit's (And, Xor, Not) gates
PINS = {
    ("MemAccess", (2, 128, 128, 28)): (11367, 57426, 35),
    ("MemEntry", (2, 128, 128, 28)): (10703, 54681, 137),
    ("MemCheck", (3, 3, 2, 4, 4, 3)): (21227, 111652, 44),
}
#: MemCheck's bindings the random logs run at: one chunk a record, and two
SMALL = MC.Widths(1, 2, 3, 4)
WIDE = MC.Widths(2, 64, 64, 4)


def _bits(x: int, n: int) -> list[int]:
    return [(x >> i) & 1 for i in range(n)]


def _int(bs) -> int:
    return sum(int(b) << i for i, b in enumerate(bs))


# --- random logs -------------------------------------------------------------------------------------------------------

def _initial(S, tag: int, a: int) -> int:
    return (5 * tag + 3 * a + 1) % (1 << S.V)


def _log(rng: random.Random, S, N: int, M: int) -> tuple[list[Access], list[Entry]]:
    """N accesses at times 1..N (about a fifth unused) to at most M addresses, and the touched list padded to M."""
    keys = sorted({(rng.randrange(1 << S.G), rng.randrange(1 << S.A)) for _ in range(M)})
    cells, acc = {}, []
    for j in range(N):
        if rng.random() < 0.2:
            acc.append(Access(0, 0, 0, 0, 0, 0, j + 1))
            continue
        tag, a = rng.choice(keys)
        v, t = cells.get((tag, a), (_initial(S, tag, a), 0))
        w = rng.randrange(1 << S.V)
        cells[(tag, a)] = (w, j + 1)
        acc.append(Access(1, tag, a, v, t, w, j + 1))
    ents = [Entry(1, tag, a, _initial(S, tag, a), *cells[(tag, a)]) for tag, a in sorted(cells)]
    return acc, ents + [Entry(0, 0, 0, 0, 0, 0)] * (M - len(ents))


def _tampered(rng: random.Random, S, acc: list[Access], ents: list[Entry]) -> list[tuple[str, list, list]]:
    """The log with one thing wrong, each way that applies."""
    out = []
    used = [j for j, x in enumerate(acc) if x.en]
    unused = [j for j, x in enumerate(acc) if not x.en]
    full = [i for i, e in enumerate(ents) if e.en]

    def at(xs, i, y):
        return xs[:i] + [y] + xs[i + 1:]

    if used:
        j = rng.choice(used)
        x = acc[j]
        out.append(("read value", at(acc, j, x._replace(v=x.v ^ 1 << rng.randrange(S.V))), ents))
        if later := [k for k in used if acc[k].now > 1]:
            k = rng.choice(later)
            out.append(("read time", at(acc, k, acc[k]._replace(t=(acc[k].t + 1) % acc[k].now)), ents))
        out.append(("read at its now", at(acc, j, x._replace(t=x.now)), ents))
        out.append(("access dropped", at(acc, j, Access(0, 0, 0, 0, 0, 0, x.now)), ents))
    if unused:
        j = rng.choice(unused)
        out.append(("unused not zero", at(acc, j, acc[j]._replace(a=1)), ents))
    if full:
        i = rng.choice(full)
        e = ents[i]
        out.append(("entry dropped", acc, ents[:i] + ents[i + 1:] + [Entry(0, 0, 0, 0, 0, 0)]))
        out.append(("initial value", acc, at(ents, i, e._replace(v0=e.v0 ^ 1))))
        if len(ents) > 1:
            out.append(("entry repeated", acc, at(ents, i + 1 if i + 1 < len(ents) else i - 1, e)))
        if len(full) > 1:
            out.append(("entries swapped", acc, [ents[1], ents[0]] + ents[2:]))
        if len(full) < len(ents):
            out.append(("used after unused", acc, [Entry(0, 0, 0, 0, 0, 0)] + ents[:-1]))
    return [(name, a, e[:len(ents)]) for name, a, e in out]


def _check_args(S, cases: list) -> list:
    """MemCheck's argument columns for cases (alpha, gamma, accesses, entries)."""
    cols = [[] for _ in range(14)]
    for al, ga, acc, ents in cases:
        row = [_bits(al, 128), _bits(ga, 128), [x.en for x in acc], [_bits(x.tag, S.G) for x in acc],
               [_bits(x.a, S.A) for x in acc], [_bits(x.v, S.V) for x in acc], [_bits(x.t, S.T) for x in acc],
               [_bits(x.w, S.V) for x in acc], [e.en for e in ents], [_bits(e.tag, S.G) for e in ents],
               [_bits(e.a, S.A) for e in ents], [_bits(e.v0, S.V) for e in ents], [_bits(e.v, S.V) for e in ents],
               [_bits(e.t, S.T) for e in ents]]
        for c, x in zip(cols, row):
            c.append(x)
    return cols


@pytest.mark.parametrize("S, N, M, seed", [(SMALL, 12, 6, 1), (WIDE, 5, 3, 2)])
def test_memcheck_accepts_exactly_the_consistent_logs(S, N, M, seed) -> None:
    rng = random.Random(seed)
    cases, want = [], []
    for _ in range(24 if S is SMALL else 8):
        acc, ents = _log(rng, S, N, M)
        for name, a, e in [("honest", acc, ents), *_tampered(rng, S, acc, ents)]:
            al, ga = rng.getrandbits(128), rng.getrandbits(128)
            ok = MC.consistent(a, e)
            assert ok == (name == "honest"), name
            assert MC.memcheck(S, al, ga, a, e) == int(ok), name
            cases.append((al, ga, a, e))
            want.append(int(ok))
    got = evaluate_bits(bind(MC.MemCheck, N=N, M=M, **S._asdict()), *_check_args(S, cases)).reshape(-1).tolist()
    assert got == want


def test_the_parts_agree_with_their_references() -> None:
    """Random instances at the pool's layout, a third of them unused (half of those zero), with t below now about half the
    time, and list entries after a random predecessor, first or not."""
    rng = random.Random(3)
    acc_rows, acc_want, ent_rows, ent_want = [], [], [], []
    S = PL

    def r(w):
        return rng.getrandbits(w)

    for _ in range(96):
        al, ga, R, W, ok = r(128), r(128), r(128), r(128), rng.randrange(2)
        en = int(rng.random() < 0.67)
        x = Access(en, r(S.G), r(S.A), r(S.V), r(S.T), r(S.V), r(S.T))
        if not en and rng.random() < 0.5:
            x = Access(0, 0, 0, 0, 0, 0, x.now)
        acc_rows.append([_bits(al, 128), _bits(ga, 128), _bits(R, 128), _bits(W, 128), ok, x.en, _bits(x.tag, S.G),
                         _bits(x.a, S.A), _bits(x.v, S.V), _bits(x.t, S.T), _bits(x.w, S.V), _bits(x.now, S.T)])
        acc_want.append(MC.access(S, al, ga, R, W, ok, x))
        first = int(rng.random() < 0.2)
        p = Entry(rng.randrange(2), r(S.G), r(S.A), 0, 0, 0)
        e = Entry(en, r(S.G), r(S.A), r(S.V), r(S.V), r(S.T))
        if rng.random() < 0.3:
            e = e._replace(tag=p.tag, a=p.a + rng.choice((-1, 0, 1)) & (1 << S.A) - 1)
        if not en and rng.random() < 0.5:
            e = Entry(0, 0, 0, 0, 0, 0)
        ent_rows.append([_bits(al, 128), _bits(ga, 128), _bits(R, 128), _bits(W, 128), ok, first, p.en, _bits(p.tag, S.G),
                         _bits(p.a, S.A), e.en, _bits(e.tag, S.G), _bits(e.a, S.A), _bits(e.v0, S.V), _bits(e.v, S.V),
                         _bits(e.t, S.T)])
        ent_want.append(MC.entry(S, al, ga, R, W, ok, first, p, e))
    for d, rows, want in ((MC.MemAccess, acc_rows, acc_want), (MC.MemEntry, ent_rows, ent_want)):
        out = evaluate_bits(bind(d, **S._asdict()), *[[row[k] for row in rows] for k in range(len(rows[0]))])
        got = [(_int(o[:128]), _int(o[128:256]), int(o[256])) for o in out.reshape(len(rows), -1).tolist()]
        assert got == want, d.name
        assert 0 < sum(w[2] for w in want) < len(want)


# --- the pool's logs ---------------------------------------------------------------------------------------------------

def _window(w: dict) -> tuple[list[Access], list[Entry]]:
    return [Access(*x) for x in w["accesses"]], [Entry(1, *e) for e in w["entries"]]


class _Parts:
    """Each access's and entry's (factor read, factor written, ok) from the circuits, with R = W = ok = 1 in: a chain's
    R, W and ok are the products and conjunction of these. Memoized by row, so a tampered log evaluates only its new rows."""

    def __init__(self, S, alpha: int, gamma: int) -> None:
        self.S, self.alpha, self.gamma, self.memo = S, alpha, gamma, {}

    def _rows(self, d, keys: list) -> None:
        S, one = self.S, _bits(1, 128)
        todo = sorted({k for k in keys if k not in self.memo})
        if not todo:
            return
        head = [_bits(self.alpha, 128), _bits(self.gamma, 128), one, one, 1]
        if d is MC.MemAccess:
            rows = [head + [x.en, _bits(x.tag, S.G), _bits(x.a, S.A), _bits(x.v, S.V), _bits(x.t, S.T), _bits(x.w, S.V),
                            _bits(x.now, S.T)] for _, x in todo]
        else:
            rows = [head + [f, p.en, _bits(p.tag, S.G), _bits(p.a, S.A), e.en, _bits(e.tag, S.G), _bits(e.a, S.A),
                            _bits(e.v0, S.V), _bits(e.v, S.V), _bits(e.t, S.T)] for _, (f, p, e) in todo]
        out = evaluate_bits(bind(d, **S._asdict()), *[[row[k] for row in rows] for k in range(len(rows[0]))])
        for k, o in zip(todo, out.reshape(len(rows), -1).tolist()):
            self.memo[k] = (_int(o[:128]), _int(o[128:256]), int(o[256]))

    def verdicts(self, logs: list[tuple[list[Access], list[Entry]]]) -> list[tuple[int, int, int]]:
        """Each log's (R, W, ok)."""
        keyed = []
        for acc, ents in logs:
            prev = [Entry(0, 0, 0, 0, 0, 0)] + ents[:-1]
            keyed.append([("a", x) for x in acc] + [("e", (int(i == 0), p, e)) for i, (p, e) in enumerate(zip(prev, ents))])
        self._rows(MC.MemAccess, [k for ks in keyed for k in ks if k[0] == "a"])
        self._rows(MC.MemEntry, [k for ks in keyed for k in ks if k[0] == "e"])
        out = []
        for ks in keyed:
            R, W, ok = 1, 1, 1
            for k in ks:
                fr, fw, o = self.memo[k]
                R, W, ok = mul(R, fr), mul(W, fw), ok & o
            out.append((R, W, ok))
        return out


@pytest.mark.parametrize("w", POOL["windows"], ids=[w["trace"] for w in POOL["windows"]])
def test_a_pool_window_checks_and_each_tampered_one_fails(w) -> None:
    acc, ents = _window(w)
    assert MC.consistent(acc, ents)
    parts = _Parts(PL, w["alpha"], w["gamma"])
    rng = random.Random(w["trace"])
    tampered = [t for _ in range(2) for t in _tampered(rng, PL, acc, ents)]
    assert {name for name, _, _ in tampered} >= {"read value", "read time", "read at its now", "access dropped",
                                                  "unused not zero", "entry dropped", "initial value", "entry repeated",
                                                  "entries swapped"}
    got = parts.verdicts([(acc, ents)] + [(a, e) for _, a, e in tampered])
    assert got[0] == (w["R"], w["R"], 1)
    for (name, a, e), (R, W, ok) in zip(tampered, got[1:]):
        assert not MC.consistent(a, e), name
        assert not (ok and R == W), name


def test_the_parts_cost_the_pinned_gates() -> None:
    for (name, statics), (ands, xors, nots) in PINS.items():
        assert getattr(MC, name).traced(*statics).counts == {"and": ands, "xor": xors, "not": nots}, name


def test_bad_statics_are_refused() -> None:
    with pytest.raises(ValueError, match="width A is 0"):
        MC.MemAccess.signature(MC.Widths(1, 0, 1, 1))
    with pytest.raises(ValueError, match="times 1..8 need more than T = 3 bits"):
        MC.MemCheck.signature(type("S", (), {"N": 8, "M": 1, **SMALL._asdict(), "T": 3}))
    with pytest.raises(ValueError, match="out of its 2 bits"):
        MC.pack(SMALL, 0, 4, 0, 0)
