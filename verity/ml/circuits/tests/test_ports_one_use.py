"""The one-use and seed rules (E_COIN_REUSE, `ports._Uses`) at the size of the Program's description: the check
simulates a Definition once per pattern of tracked parameter leaves and applies that summary to every batch member and
scan step, so a batch of N identical units costs the same simulations at any N (counted, never timed). It refuses what the
flattened walk it replaced refused, which this file keeps as the reference (`OldUses`, the walk it replaced,
verbatim but for its names), on every Program of `test_ports.py` and on random small Programs of batches and scans
nested, seeds, positions and coins shared across members and steps; and a refusal still names its first two readers by
Definition, node, member and step."""

from __future__ import annotations

import bisect
import random
from typing import Any, ClassVar

import pytest

from verity.ml.circuits import ports as P
from verity.ml.circuits.defs import CompositeDefinition, primitive
from verity.ml.circuits.ports import In, Out, PortError, Ports, PortTypes
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Refs, array_of, tuple_of
from verity.ml.circuits.refs import runs as ref_runs
from verity.ml.circuits.types import Array, Type, Value
from verity.ml.circuits.types import Tuple as TupleT

_coin_role, _spaces, _ROLES = P._coin_role, P._spaces, P._ROLES
_NOISE, _FOLD, _SEED, _OTHER = P._NOISE, P._FOLD, P._SEED, P._OTHER

V8, V32 = Value(8), Value(32)


# --- the reference: the flattened walk ----------------------------------------------------------------------------------------

class _OldSummary:
    """One activation of a composite whose parameters' tracked leaves are `pattern`: `reads[(i, leaf)]`, how often parameter
    i's leaf is read by a noise primitive, a receipt fold, a seed and any other primitive, with the first readers; `ret[j]`,
    what return leaf j is: ("p", i, leaf), that parameter leaf; or ("c", inst, leaf, count, readers), noise made inside and
    read there `count` times."""

    __slots__ = ("reads", "ret")

    def __init__(self, reads: dict, ret: dict) -> None:
        self.reads, self.ret = reads, ret


class OldUses:
    """The one-use and seed rules over the flattened Program: each tracked leaf (a root coin, ("coin", node, leaf); a root
    position, ("pos", node, leaf); or a noise primitive instance's output, ("c", inst, leaf)) is counted at every primitive or
    seed operand that reads it, once per member, step and instance, through every call. A composite's activation is
    summarized once per pattern of tracked parameter leaves, so a batch or scan of one body costs one simulation and a count
    per tracked leaf per member."""

    def __init__(self, root: Any, rpt: PortTypes) -> None:
        self.root, self.rpt = root, rpt
        self.memo: dict[tuple, _OldSummary] = {}
        self.coins: dict[tuple, list] = {}
        self.made: dict[tuple, list] = {}
        self.labels: list[str] = []
        self.keys: dict[int, tuple[dict, list]] = {}
        self.roles: dict[int, tuple[Any, int | None]] = {}

    def _role(self, fn: Any) -> int | None:
        hit = self.roles.get(id(fn))
        if hit is None or hit[0] is not fn:
            hit = self.roles[id(fn)] = (fn, _coin_role(fn))
        return hit[1]

    def check(self) -> None:
        self._simulate(self.root, ())

    def _new(self, label: str) -> int:
        self.labels.append(label)
        return len(self.labels) - 1

    def _sorted(self, d: dict) -> list:
        hit = self.keys.get(id(d))
        if hit is None or hit[0] is not d:
            hit = self.keys[id(d)] = (d, sorted(d))
        return hit[1]

    def _gather(self, refs: Refs, env: dict) -> dict:
        spaces: set = set()
        _spaces(refs, spaces)
        if not any(env.get(s) for s in spaces):
            return {}
        out, pos = {}, 0
        for s, i, lo, hi in ref_runs(refs):
            d = env.get((s, i))
            if d:
                keys = self._sorted(d)
                for leaf in keys[bisect.bisect_left(keys, lo):bisect.bisect_left(keys, hi)]:
                    out[pos + leaf - lo] = d[leaf]
            pos += hi - lo
        return out

    def _read(self, s: tuple, role: int, n: int, at: list, reads: dict) -> None:
        if s[0] == "pos" and role != _SEED:
            return
        if s[0] == "p":
            e = reads.setdefault((s[1], s[2]), [0, 0, 0, 0, []])
        elif s[0] in ("coin", "pos"):
            e = self.coins.setdefault(s, [0, 0, 0, 0, []])
        else:
            e = self.made[(s[1], s[2])]
        e[role] += n
        e[4].extend(at[:2 - len(e[4])])
        total = sum(e[:4])
        if s[0] == "coin" and (e[_NOISE] or e[_OTHER]) and total > 1:
            raise PortError("E_COIN_REUSE", f"{self.root.id} input node {s[1]} leaf {s[2]}: a coin that a noise primitive "
                            f"reads is read once, by that instance alone, but it is read {total} times (first by "
                            f"{e[4][0]}; then by {at[0]}): a second noise or a receipt key from a noise bit's coin puts the "
                            f"bit in the cheater's view. Draw a coin of its own for each noise and for the receipt keys")
        if s[0] == "coin" and e[_SEED] and e[_FOLD]:
            raise PortError("E_COIN_REUSE", f"{self.root.id} input node {s[1]} leaf {s[2]}: a coin that a seed reads is read "
                            f"by seeds alone, but a receipt fold reads it too (first by {e[4][0]}; then by {at[0]}): only a "
                            f"seed binds the coin to a unit's position, so a fold keyed by it computes a value no position "
                            f"separates. Key the folds from the unit's seed, or from a coin of their own")
        if s[0] == "pos" and e[_SEED] > 1:
            raise PortError("E_COIN_REUSE", f"{self.root.id} input node {s[1]} leaf {s[2]}: a position seeds one unit, but "
                            f"{e[_SEED]} seeds read it (first by {'; then by '.join(e[4])}): two seeds of one coin at one "
                            f"position are one seed, so the noise they derive is common to both. Give each seed a position "
                            f"of its own")
        if s[0] == "c" and total > 1:
            raise PortError("E_COIN_REUSE", f"{self.labels[s[1]]} return leaf {s[2]}: noise is the noise of one operand of one "
                            f"instance and is read at most once, but it is read {total} times (first by "
                            f"{'; then by '.join(e[4])}): two copies of one noise cancel, and noise shared across rows, "
                            f"matmuls, batch members or scan steps is fresh for none of them. Draw fresh coins for each use")

    def _apply(self, fn: Any, args: list, at: str, reads: dict) -> dict:
        """One instance of `fn` (a primitive or a composite) on arguments whose tracked leaves are `args` ({leaf: source}
        per parameter): count its reads, and give its return's tracked leaves."""
        from verity.ml.circuits.defs import PrimitiveDefinition

        role = self._role(fn)
        if role == _SEED or isinstance(fn, PrimitiveDefinition):
            for d in args:
                for s in d.values():
                    self._read(s, _OTHER if role is None else role, 1, [at], reads)
            if role != _NOISE:
                return {}
            leaves = fn.signature()[1].leaves
            inst = self._new(at)
            for j in range(leaves):
                self.made[(inst, j)] = [0, 0, 0, 0, []]
            return {j: ("c", inst, j) for j in range(leaves)}
        pattern = tuple(tuple(sorted(d)) for d in args)
        S = self.memo.get((id(fn), pattern))
        if S is None:
            S = self.memo[(id(fn), pattern)] = self._simulate(fn, pattern)
        for (i, leaf), e in S.reads.items():
            for role in _ROLES:
                if e[role]:
                    self._read(args[i][leaf], role, e[role], [f"{at}: {w}" for w in e[4]], reads)
        out, fresh = {}, {}
        for j, r in S.ret.items():
            if r[0] == "p":
                out[j] = args[r[1]][r[2]]
                continue
            inst = fresh.get(r[1])
            if inst is None:
                inst = fresh[r[1]] = self._new(f"{at}: {self.labels[r[1]]}")
            self.made[(inst, r[2])] = [r[3], 0, 0, 0, [f"{at}: {w}" for w in r[4]]]
            out[j] = ("c", inst, r[2])
        return out

    def _members(self, d: dict, t: Type, ax: int, n: int) -> list[dict]:
        """Tracked leaves {leaf: source} of an argument of type `t`, split into its `n` members along axis `ax`."""
        out: list[dict] = [{} for _ in range(n)]
        if not d:
            return out
        if ax == 0:
            L = t.leaves // n
            for p, s in d.items():
                out[p // L][p % L] = s
            return out
        keys, pos = self._sorted(d), Coll(t, Affine("a", 0, 0, t.leaves))
        for m in range(n):
            q = 0
            for _s, _i, lo, hi in ref_runs(pos.take(ax, m).refs):
                for p in keys[bisect.bisect_left(keys, lo):bisect.bisect_left(keys, hi)]:
                    out[m][q + p - lo] = d[p]
                q += hi - lo
        return out

    def _simulate(self, sp: Any, pattern: tuple) -> _OldSummary:
        from verity.ml.circuits.program import is_input

        root = sp is self.root
        env: dict[tuple[str, int], dict] = {("p", i): {leaf: ("p", i, leaf) for leaf in leaves} for i, leaves in enumerate(pattern)}
        if root:
            for k, lo, hi, p, _w in self.rpt.inputs:
                if p.kind in ("coin", "position"):
                    tag = "coin" if p.kind == "coin" else "pos"
                    env.setdefault(("n", k), {}).update({leaf: (tag, k, leaf) for leaf in range(lo, hi)})
        reads: dict = {}
        for k, node in enumerate(sp.body.nodes):
            fn = node.fn
            if is_input(fn):
                continue
            tracked = [self._gather(a.refs, env) for a in node.args]
            if not any(tracked):
                continue
            at = f"{sp.id} node {k} ({node.form} {fn.id})"
            _, fret = fn.signature()
            out: dict = {}
            if node.form == "batch":
                R = fret.leaves
                split = [[d] * node.n if ax is None else self._members(d, a.type, ax, node.n)
                         for d, a, ax in zip(tracked, node.args, node.axes)]
                for m in range(node.n):
                    args = [ms[m] for ms in split]
                    if any(args):
                        out.update({m * R + j: s for j, s in self._apply(fn, args, f"{at} member {m}", reads).items()})
            elif node.form == "scan":
                c = fret.fields[0].leaves
                ys = fret.leaves - c
                carry, shared = tracked[0], tracked[1 + node.n_xs:]
                xs = [self._members(d, a.type, 0, node.n) for d, a in zip(tracked[1:1 + node.n_xs], node.args[1:1 + node.n_xs])]
                last = max((i for ms in xs for i, d in enumerate(ms) if d), default=-1)
                for i in range(node.n):
                    if not carry and not any(shared) and i > last:
                        break
                    args = [carry] + [ms[i] for ms in xs] + list(shared)
                    got = self._apply(fn, args, f"{at} step {i}", reads) if any(args) else {}
                    carry = {q: s for q, s in got.items() if q < c}
                    out.update({c + i * ys + q - c: s for q, s in got.items() if q >= c})
                out.update(carry)
            else:
                out = self._apply(fn, tracked, at, reads)
            env[("n", k)] = out
        R = self._gather(sp.body.ret, env)
        ret: dict = {}
        for j, s in R.items():
            if root:
                self._read(s, _OTHER, 1, [f"{sp.id}'s outputs"], reads)
            elif s[0] == "p":
                ret[j] = s
            else:
                e = self.made[(s[1], s[2])]
                ret[j] = ("c", s[1], s[2], sum(e[:4]), tuple(e[4]))
        return _OldSummary(reads, ret)




# --- building blocks ----------------------------------------------------------------------------------------------------------

@primitive("PortsOneUseDither", 1, (("x", V8), ("key", V32)), V8, ports=Ports(params={"key": In(kind="coin")}))
def Dither(x, key):
    """A noise injection: the operand plus its coin's low bit."""
    return (x + (key & 1)) & 0xFF


@primitive("PortsOneUseKeyFold", 1, (("x", V32), ("key", V32)), V32,
           ports=Ports(params={"key": In(kind="coin")}, ret=Out(sink=True)))
def KeyFold(x, key):
    """A receipt fold keyed by a coin."""
    return ((x ^ key) * 0x9E3779B1) & 0xFFFFFFFF


@primitive("PortsOneUseFold", 1, (("x", V32),), V32, ports=Ports(ret=Out(sink=True)))
def Fold(x):
    """A receipt fold of a value: its output is a sink."""
    return x ^ 0x5A5A5A5A


@primitive("PortsOneUseAdd8", 1, (("a", V8), ("b", V8)), V8)
def Add8(a, b):
    return (a + b) & 0xFF


@primitive("PortsOneUseAdd32", 1, (("a", V32), ("b", V32)), V32)
def Add32(a, b):
    return (a + b) & 0xFFFFFFFF


@primitive("PortsOneUseWiden", 1, (("x", V8),), V32)
def Widen(x):
    return x


@primitive("PortsOneUseByte", 1, (("w", V32),), V8)
def Byte(w):
    return w & 0xFF


def comp(name: str, params, ret, body, ports=None) -> CompositeDefinition:
    return CompositeDefinition(name, 1, (), lambda S: (tuple(params), ret), body, register=False, ports=ports)


ROOT_PORTS = Ports(params={"keys": In(kind="coin"), "pos": In(kind="position")})
COINS = Ports(params={"keys": In(kind="coin")})
#: a seed: (salt, position) to a word no port rule follows
Seed = comp("PortsOneUseSeed", [("salt", V32), ("position", V32)], V32, lambda B, S, salt, position: B.call(Add32, salt, position),
            ports=Ports(params={"salt": In(kind="coin"), "position": In(kind="position")}))
def _unit(B, S, x, key):
    z = B.call(Add8, x, B.call(Dither, x, key))
    return tuple_of(z, B.call(Fold, B.call(Widen, z)))


#: a unit: x plus noise from its own coin, z = x + that, and a receipt folding z (a sink)
Unit = comp("PortsOneUseUnit", [("x", V8), ("key", V32)], TupleT(V8, V32), _unit, ports=Ports(ret=(None, Out(sink=True))))


class Counting(P._Uses):
    """`ports._Uses`, keeping each check and counting the instances it applies a summary to one at a time."""

    made: ClassVar[list] = []

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.members = 0
        Counting.made.append(self)

    def _member(self, *args: Any) -> None:
        self.members += 1
        super()._member(*args)


def _counted(monkeypatch, build) -> Counting:
    Counting.made = []
    monkeypatch.setattr(P, "_Uses", Counting)
    build()
    (u,) = Counting.made
    return u


def refused(build, *needles: str) -> str:
    with pytest.raises(PortError) as e:
        build()
    assert e.value.code == "E_COIN_REUSE", e.value
    for n in needles:
        assert n in str(e.value), (n, str(e.value))
    return str(e.value)


# --- the cost is the description's --------------------------------------------------------------------------------------------

def _units(n: int) -> Program:
    return Program(comp(f"PortsOneUseUnits{n}", [("x", Array(n, V8)), ("keys", Array(n, V32))], Array(n, TupleT(V8, V32)),
                        lambda B, S, x, keys: B.batch(Unit, x, keys, axes=(0, 0)), ports=COINS))


def receipts(n: int) -> Ports:
    """The ports of a Definition returning n units' (z, receipt): every receipt a sink."""
    return Ports(ret=P.Leaves(tuple((2 * i + 1, 2 * i + 2, Out(sink=True)) for i in range(n))))


def _nested(n: int, m: int) -> Program:
    inner = comp(f"PortsOneUseRow{m}", [("x", Array(m, V8)), ("keys", Array(m, V32))], Array(m, TupleT(V8, V32)),
                 lambda B, S, x, keys: B.batch(Unit, x, keys, axes=(0, 0)), ports=receipts(m))
    return Program(comp(f"PortsOneUseGrid{n}x{m}", [("x", Array(n, Array(m, V8))), ("keys", Array(n, Array(m, V32)))],
                        Array(n, Array(m, TupleT(V8, V32))), lambda B, S, x, keys: B.batch(inner, x, keys, axes=(0, 0)),
                        ports=COINS))


def _steps(n: int) -> Program:
    step = comp("PortsOneUseStep", [("c", V8), ("x", V8), ("key", V32)], TupleT(V8, TupleT(V8, V32)),
                lambda B, S, c, x, key: tuple_of(c, B.call(Unit, x, key)), ports=Ports(ret=(None, (None, Out(sink=True)))))
    return Program(comp(f"PortsOneUseSteps{n}", [("c", V8), ("x", Array(n, V8)), ("keys", Array(n, V32))],
                        TupleT(V8, Array(n, TupleT(V8, V32))), lambda B, S, c, x, keys: B.scan(step, c, xs=(x, keys)),
                        ports=COINS))


def test_a_batch_of_units_costs_the_same_at_any_size(monkeypatch):
    """N units of one Definition, each with a noise primitive of its own coin and a receipt sink: the check simulates the
    unit once and adds its counts for all N members at once, so neither the simulations nor the instances it applies a
    summary to one at a time (the unit's own nodes) grow with N; nor do they for a batch of batches, or a scan of N steps
    with no tracked carry."""
    for make in (_units, lambda n: _nested(n, n), _steps):
        got = {n: _counted(monkeypatch, lambda make=make, n=n: make(n)) for n in (2, 16, 256, 2048)}
        assert len({(u.simulations, u.members) for u in got.values()}) == 1, {n: (u.simulations, u.members) for n, u in got.items()}
        assert got[2].simulations <= 4 and got[2].members <= 4, (got[2].simulations, got[2].members)


def test_a_flat_body_of_calls_simulates_each_definition_once(monkeypatch):
    """A body of N calls of one unit, each on its own coin (the H1 window's shape): one simulation of the unit, whatever N;
    each call is a node of the description, applied once."""
    def flat(n):
        return Program(comp(f"PortsOneUseFlat{n}", [("x", Array(n, V8)), ("keys", Array(n, V32))], Array(n, TupleT(V8, V32)),
                            lambda B, S, x, keys: array_of([B.call(Unit, x[i], keys[i]) for i in range(n)]), ports=COINS))
    got = {n: _counted(monkeypatch, lambda n=n: flat(n)) for n in (2, 64, 512)}
    assert {u.simulations for u in got.values()} == {2}
    assert {n: u.members for n, u in got.items()} == {n: n + 2 for n in got}


# --- reuse across members and steps, by name ----------------------------------------------------------------------------------

def _at(root: str, *frames: str) -> str:
    return ": ".join((f"Root[{root}_v1]_v1 " + frames[0],) + frames[1:])


DITHER = "PortsOneUseUnit_v1 node 0 (primitive PortsOneUseDither_v1)"


def test_a_coin_shared_by_two_members_is_refused_naming_both():
    shared = comp("PortsOneUseShared", [("x", Array(4, V8)), ("keys", V32)], Array(4, TupleT(V8, V32)),
                  lambda B, S, x, keys: B.batch(Unit, x, keys, axes=(0, None)), ports=COINS)
    batch = "node 2 (batch PortsOneUseUnit_v1)"
    refused(lambda: Program(shared), "input node 1 leaf 0: a coin that a noise primitive reads is read once", "read 4 times",
            f"first by {_at('PortsOneUseShared', f'{batch} member 0', DITHER)}; "
            f"then by {_at('PortsOneUseShared', f'{batch} member 1', DITHER)})")
    again = comp("PortsOneUseAgain", [("x", Array(4, V8)), ("keys", Array(3, V32))], Array(4, TupleT(V8, V32)),
                 lambda B, S, x, keys: B.batch(Unit, x, array_of([keys[0], keys[1], keys[2], keys[1]]), axes=(0, 0)), ports=COINS)
    refused(lambda: Program(again), "input node 1 leaf 1:", "read 2 times",
            f"first by {_at('PortsOneUseAgain', f'{batch} member 1', DITHER)}; "
            f"then by {_at('PortsOneUseAgain', f'{batch} member 3', DITHER)})")


def test_a_coin_shared_by_two_steps_is_refused_naming_both():
    step = comp("PortsOneUseCoinStep", [("c", V8), ("x", V8), ("key", V32)], TupleT(V8, V8),
                lambda B, S, c, x, key: tuple_of(B.call(Add8, c, x), B.call(Dither, x, key)))
    scan, dither = "node 2 (scan PortsOneUseCoinStep_v1)", "PortsOneUseCoinStep_v1 node 1 (primitive PortsOneUseDither_v1)"
    refused(lambda: Program(comp("PortsOneUseCoinScan", [("x", Array(3, V8)), ("keys", V32)], TupleT(V8, Array(3, V8)),
                                 lambda B, S, x, keys: B.scan(step, x[0], xs=(x,), shared=(keys,)), ports=COINS)),
            "input node 1 leaf 0", "read 3 times", f"first by {_at('PortsOneUseCoinScan', f'{scan} step 0', dither)}; "
            f"then by {_at('PortsOneUseCoinScan', f'{scan} step 1', dither)})")
    refused(lambda: Program(comp("PortsOneUseCoinScanXs", [("x", Array(3, V8)), ("keys", Array(2, V32))], TupleT(V8, Array(3, V8)),
                                 lambda B, S, x, keys: B.scan(step, x[0], xs=(x, array_of([keys[0], keys[1], keys[0]]))),
                                 ports=COINS)),
            "input node 1 leaf 0", "read 2 times", f"first by {_at('PortsOneUseCoinScanXs', f'{scan} step 0', dither)}; "
            f"then by {_at('PortsOneUseCoinScanXs', f'{scan} step 2', dither)})")


def test_noise_read_by_two_members_or_two_steps_is_refused_naming_both():
    made = comp("PortsOneUseMade", [("x", V8), ("key", V32)], V8, lambda B, S, x, key: B.call(Dither, x, key))
    rows = comp("PortsOneUseRows", [("x", Array(3, V8)), ("keys", V32)], Array(3, V8),
                lambda B, S, x, keys: B.batch(Add8, x, B.call(made, x[0], keys), axes=(0, None)), ports=COINS)
    refused(lambda: Program(rows), _at("PortsOneUseRows", "node 2 (call PortsOneUseMade_v1)", "PortsOneUseMade_v1 node 0 "
                                       "(primitive PortsOneUseDither_v1) return leaf 0: noise is the noise of one operand"),
            "read 3 times", f"first by {_at('PortsOneUseRows', 'node 3 (batch PortsOneUseAdd8_v1) member 0')}; "
            f"then by {_at('PortsOneUseRows', 'node 3 (batch PortsOneUseAdd8_v1) member 1')})")
    carried = comp("PortsOneUseCarried", [("c", V8), ("x", V8)], TupleT(V8, V8), lambda B, S, c, x: tuple_of(c, B.call(Add8, c, x)))
    scan, add = "node 3 (scan PortsOneUseCarried_v1)", "PortsOneUseCarried_v1 node 0 (primitive PortsOneUseAdd8_v1)"
    refused(lambda: Program(comp("PortsOneUseCarry", [("x", Array(3, V8)), ("keys", V32)], TupleT(V8, Array(3, V8)),
                                 lambda B, S, x, keys: B.scan(carried, B.call(Dither, x[0], keys), xs=(x,)), ports=COINS)),
            "node 2 (primitive PortsOneUseDither_v1) return leaf 0: noise is the noise of one operand", "read 4 times",
            f"first by {_at('PortsOneUseCarry', f'{scan} step 0', add)}; then by {_at('PortsOneUseCarry', f'{scan} step 1', add)})")


def test_a_position_shared_by_two_members_is_refused_naming_both():
    seeded = comp("PortsOneUseSeeded", [("x", V8), ("salt", V32), ("position", V32)], V8,
                  lambda B, S, x, salt, position: B.call(Add8, x, B.call(Byte, B.call(Seed, salt, position))))
    own = comp("PortsOneUseOwnPositions", [("x", Array(3, V8)), ("keys", V32), ("pos", Array(3, V32))], Array(3, V8),
               lambda B, S, x, keys, pos: B.batch(seeded, x, keys, pos, axes=(0, None, 0)), ports=ROOT_PORTS)
    Program(own)
    one = comp("PortsOneUseOnePosition", [("x", Array(3, V8)), ("keys", V32), ("pos", V32)], Array(3, V8),
               lambda B, S, x, keys, pos: B.batch(seeded, x, keys, pos, axes=(0, None, None)), ports=ROOT_PORTS)
    batch, seed = "node 3 (batch PortsOneUseSeeded_v1)", "PortsOneUseSeeded_v1 node 0 (call PortsOneUseSeed_v1)"
    refused(lambda: Program(one), "input node 2 leaf 0: a position seeds one unit, but 3 seeds read it",
            f"first by {_at('PortsOneUseOnePosition', f'{batch} member 0', seed)}; "
            f"then by {_at('PortsOneUseOnePosition', f'{batch} member 1', seed)})")


def test_reuse_across_nested_members_names_the_outer_and_inner_member():
    inner = comp("PortsOneUseInner", [("x", Array(2, V8)), ("key", V32)], Array(2, TupleT(V8, V32)),
                 lambda B, S, x, key: B.batch(Unit, x, key, axes=(0, None)), ports=receipts(2))
    outer, unit = "node 2 (batch PortsOneUseInner_v1) member 0", "PortsOneUseInner_v1 node 0 (batch PortsOneUseUnit_v1)"
    refused(lambda: Program(comp("PortsOneUseOuter", [("x", Array(2, Array(2, V8))), ("keys", Array(2, V32))],
                                 Array(2, Array(2, TupleT(V8, V32))), lambda B, S, x, keys: B.batch(inner, x, keys, axes=(0, 0)),
                                 ports=COINS)),
            "input node 1 leaf 0", "read 2 times", f"first by {_at('PortsOneUseOuter', outer, f'{unit} member 0', DITHER)}; "
            f"then by {_at('PortsOneUseOuter', outer, f'{unit} member 1', DITHER)})")


# --- the same verdicts as the flattened walk ----------------------------------------------------------------------------------

class Both(P._Uses):
    """`ports._Uses` that also runs the reference walk on the same Program and keeps both verdicts."""

    seen: ClassVar[list] = []

    def check(self) -> None:
        if self.explain:
            return super().check()
        old = new = None
        try:
            OldUses(self.root, self.rpt).check()
        except PortError as e:
            old = e.code
        try:
            super().check()
        except PortError as e:
            new = e
        Both.seen.append((self.root.id, old, None if new is None else new.code))
        if new is not None:
            raise new


@pytest.fixture
def both(monkeypatch):
    Both.seen = []
    monkeypatch.setattr(P, "_Uses", Both)
    return Both.seen


def test_every_program_of_test_ports_has_the_walks_verdict(both):
    """Every Program `test_ports.py` builds or decodes, checked by both walks; but the tests that register a primitive of
    their own (interval rules, no coin), which a second run in one process would register twice."""
    import inspect

    import test_ports as T
    for name in sorted(n for n in dir(T) if n.startswith("test_")):
        fn = getattr(T, name)
        if not inspect.signature(fn).parameters and "primitive(" not in inspect.getsource(fn):
            fn()
    assert len(both) > 40
    assert [s for s in both if s[1] != s[2]] == []
    assert {s[2] for s in both} == {None, "E_COIN_REUSE"}


class _Gen:
    """A random small Program: a root of bytes, coins and positions whose body and callees dither, add and seed, and call,
    batch and scan callees of their own, each argument drawn from what is in scope (shared, split, repeated), returning
    some bytes and some receipt folds keyed by a coin."""

    def __init__(self, seed: int) -> None:
        self.r, self.seed, self.defs = random.Random(seed), seed, 0

    def block(self, B: Any, X: list, K: list, Q: list, depth: int) -> None:
        r = self.r
        for _ in range(r.randint(1, 4)):
            op = r.choice(("dither", "dither", "add", "seed", "call", "batch", "batch", "scan"))
            if op == "dither" and K:
                X.append(B.call(Dither, r.choice(X), r.choice(K)))
            elif op == "add":
                X.append(B.call(Add8, r.choice(X), r.choice(X)))
            elif op == "seed" and K and Q:
                X.append(B.call(Byte, B.call(Seed, r.choice(K), r.choice(Q))))
            elif op in ("call", "batch", "scan") and depth:
                getattr(self, op)(B, X, K, Q, depth - 1)

    def callee(self, kinds: list[str], depth: int, scan: bool = False) -> CompositeDefinition:
        """A Definition of parameters ``kinds`` ("x" a byte, "k" a coin, "q" a position) returning a byte, or for a scan's
        step a (carry, y) of bytes, its first parameter the carry."""
        self.defs += 1

        def body(B, S, *args):
            X = [a for a, k in zip(args, kinds) if k == "x"]
            K = [a for a, k in zip(args, kinds) if k == "k"]
            Q = [a for a, k in zip(args, kinds) if k == "q"]
            self.block(B, X, K, Q, depth)
            out = X[-1] if self.r.random() < 0.7 else B.call(Add8, X[-1], X[0])
            return tuple_of(self.r.choice(X), out) if scan else out
        params = [(f"{k}{i}", V8 if k == "x" else V32) for i, k in enumerate(kinds)]
        return comp(f"PortsOneUseGen{self.seed}x{self.defs}", params, TupleT(V8, V8) if scan else V8, body)

    def kinds(self, K: list, Q: list, lead: int = 1) -> list[str]:
        r = self.r
        rest = [r.choice("xkq") for _ in range(r.randint(0, 3))]
        return ["x"] * lead + [k for k in rest if (k != "k" or K) and (k != "q" or Q)]

    def pick(self, k: str, X: list, K: list, Q: list) -> Coll:
        return self.r.choice({"x": X, "k": K, "q": Q}[k])

    def call(self, B, X, K, Q, depth):
        kinds = self.kinds(K, Q)
        fn = self.callee(kinds, depth)
        X.append(B.call(fn, *(self.pick(k, X, K, Q) for k in kinds)))

    def batch(self, B, X, K, Q, depth):
        r, n = self.r, self.r.randint(1, 3)
        kinds = self.kinds(K, Q)
        args, axes = [], []
        for k in kinds:
            if r.random() < 0.4:
                args.append(self.pick(k, X, K, Q))
                axes.append(None)
            else:
                args.append(array_of([self.pick(k, X, K, Q) for _ in range(n)]))
                axes.append(0)
        out = B.batch(self.callee(kinds, depth), *args, axes=axes, n=n)
        X.extend(out[i] for i in r.sample(range(n), r.randint(1, n)))

    def scan(self, B, X, K, Q, depth):
        r, n = self.r, self.r.randint(1, 3)
        kinds = self.kinds(K, Q, lead=2)
        cut = r.randint(1, len(kinds) - 1)
        xs, shared = kinds[1:cut + 1], kinds[cut + 1:]
        out = B.scan(self.callee(kinds, depth, scan=True), r.choice(X),
                     xs=[array_of([self.pick(k, X, K, Q) for _ in range(n)]) for k in xs],
                     shared=[self.pick(k, X, K, Q) for k in shared])
        X.append(out[0])
        X.extend(out[1][i] for i in r.sample(range(n), r.randint(0, n)))

    def program(self) -> Program:
        r = self.r
        nx, nk, nq, no, ns = r.randint(1, 3), r.randint(1, 4), r.randint(0, 3), r.randint(1, 3), r.randint(0, 2)
        params = [("x", Array(nx, V8)), ("keys", Array(nk, V32))] + ([("pos", Array(nq, V32))] if nq else [])

        def body(B, S, x, keys, pos=None):
            X, K = [x[i] for i in range(nx)], [keys[i] for i in range(nk)]
            Q = [pos[i] for i in range(nq)] if nq else []
            self.block(B, X, K, Q, 2)
            outs = [r.choice(X[nx:] or X) for _ in range(no)]
            folds = [B.call(KeyFold, B.call(Widen, r.choice(X)), r.choice(K)) for _ in range(ns)]
            return tuple_of(*outs, *folds)
        return Program(comp(f"PortsOneUseGenRoot{self.seed}", params, TupleT(*[V8] * no, *[V32] * ns), body,
                            ports=ROOT_PORTS if nq else COINS))


#: how many random Programs the equality runs: about a quarter reach the one-use check refused, a quarter admitted
RANDOM = 3000


def test_random_programs_have_the_walks_verdict(both):
    """Random small Programs (`_Gen`): wherever the port rules before it admit one, the summarized check and the flattened
    walk give one verdict, and both verdicts occur often."""
    other: dict[str, int] = {}
    for seed in range(RANDOM):
        try:
            _Gen(seed).program()
        except PortError as e:
            other[e.code] = other.get(e.code, 0) + 1
    assert [s for s in both if s[1] != s[2]] == []
    verdicts = [s[2] for s in both]
    assert verdicts.count(None) > RANDOM // 10 and verdicts.count("E_COIN_REUSE") > RANDOM // 10, (len(both), other)
