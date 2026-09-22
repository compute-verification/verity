"""Reference checker of ``docs/PROTOCOL.md`` (A8, 2026-09-10): the five steps exactly as written, nothing else.

Pure Python, no cryptography.  "Commit" is *fixing a value in a table the verifier can read*
(:class:`Table`): a committed value cannot be changed, and after step 3 the table is sealed so nothing
can be added.  Circuits are explicit gate lists with typed domains; input gates are sources with no
equation; computed gates carry ``f``.  ``S_V`` partitions the computed gates into verification units
(VUs), ``S_R`` coarsens them into replay units (RUs); the VU quotient must be acyclic.  ``in(A)`` /
``out(A)`` follow the contract: designated circuit outputs appear in the output tuple of every
containing subcircuit.  Randomness is injected (:class:`Coins`): ``J`` is Bernoulli(q) per RU and
``Q`` is Bernoulli(s) per VU of ``J \\ F``, drawn *after* ``F`` and the step-3 commitments.

This is the reference the real verifiers are compared with, not a product: it is meant to be read
in one sitting beside the contract.  Section numbers in comments are the contract's steps.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

# ---------------------------------------------------------------- domains ---------------------------

@dataclass(frozen=True)
class Domain:
    """A gate domain: membership and the equality *the circuit specifies* (nothing is added)."""

    name: str
    contains: Callable[[Any], bool]
    equal: Callable[[Any, Any], bool] = lambda a, b: a == b


BIT = Domain("bit", lambda v: v in (0, 1))


def u(bits: int) -> Domain:
    return Domain(f"u{bits}", lambda v, m=(1 << bits) - 1: type(v) is int and 0 <= v <= m)


# IEEE-754 binary32 as a 32-bit *pattern* with bitwise equality: NaN payloads and the sign of zero are
# distinct values (a NaN payload is a legal value of this domain; +0 != -0 here).
F32_BITS = Domain("f32bits", lambda v: type(v) is int and 0 <= v < 2**32)

# ---------------------------------------------------------------- circuits --------------------------

@dataclass(frozen=True)
class Gate:
    name: str
    domain: Domain
    args: tuple[str, ...] = ()
    f: Callable[..., Any] | None = None  # None <=> input gate (a source with no equation)

    @property
    def is_input(self) -> bool:
        return self.f is None


@dataclass(frozen=True)
class Circuit:
    """Gates in a fixed order (every argument precedes its reader), the designated outputs, and the
    partitions.  ``S_V``: tuple of VUs, each a tuple of computed gate names; ``S_R``: tuple of RUs,
    each a tuple of VU indices.  ``in(C)`` is the tuple of input gates in gate order."""

    gates: tuple[Gate, ...]
    outputs: tuple[str, ...]
    S_V: tuple[tuple[str, ...], ...]
    S_R: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        by_name = {g.name: g for g in self.gates}
        if len(by_name) != len(self.gates):
            raise ValueError("duplicate gate name")
        seen: set[str] = set()
        for g in self.gates:
            for a in g.args:
                if a not in seen:
                    raise ValueError(f"gate {g.name} reads {a} before it is defined")
            seen.add(g.name)
        computed = [g.name for g in self.gates if not g.is_input]
        flat = [v for V in self.S_V for v in V]
        if sorted(flat) != sorted(computed) or len(flat) != len(set(flat)):
            raise ValueError("S_V must partition exactly the computed gates")
        flat_r = [i for R in self.S_R for i in R]
        if sorted(flat_r) != list(range(len(self.S_V))):
            raise ValueError("S_R must partition exactly the VUs")
        for o in self.outputs:
            if o not in by_name:
                raise ValueError(f"unknown output {o}")
        self._check_acyclic_vu_quotient()

    # -- structure -----------------------------------------------------------------------------------
    def gate(self, name: str) -> Gate:
        return next(g for g in self.gates if g.name == name)

    def order(self, names: Iterable[str]) -> tuple[str, ...]:
        rank = {g.name: k for k, g in enumerate(self.gates)}
        return tuple(sorted(names, key=rank.__getitem__))

    @property
    def inputs(self) -> tuple[str, ...]:  # in(C)
        return tuple(g.name for g in self.gates if g.is_input)

    def vu_of(self, gate: str) -> int:
        return next(k for k, V in enumerate(self.S_V) if gate in V)

    def ru_of_vu(self, vu: int) -> int:
        return next(k for k, R in enumerate(self.S_R) if vu in R)

    def ru_gates(self, ru: int) -> tuple[str, ...]:
        return tuple(v for k in self.S_R[ru] for v in self.S_V[k])

    def _check_acyclic_vu_quotient(self) -> None:
        edges = {k: set() for k in range(len(self.S_V))}
        for k, V in enumerate(self.S_V):
            for name in V:
                for a in self.gate(name).args:
                    if not self.gate(a).is_input and self.vu_of(a) != k:
                        edges[self.vu_of(a)].add(k)
        state: dict[int, int] = {}

        def visit(n: int) -> None:
            if state.get(n) == 1:
                raise ValueError("the VU quotient has a cycle")
            if n in state:
                return
            state[n] = 1
            for m in edges[n]:
                visit(m)
            state[n] = 2

        for n in edges:
            visit(n)

    # -- in(A), out(A) for a subcircuit A (a set of gates) --------------------------------------------
    def in_(self, A: Iterable[str]) -> tuple[str, ...]:
        A = set(A)
        return self.order({a for name in A for a in self.gate(name).args if a not in A})

    def out(self, A: Iterable[str]) -> tuple[str, ...]:
        A = set(A)
        read_outside = {a for g in self.gates if g.name not in A for a in g.args if a in A}
        designated = {o for o in self.outputs if o in A}  # designated outputs appear in every containing out(.)
        return self.order(read_outside | designated)

    # -- evaluation of a VU from committed inputs ------------------------------------------------------
    def evaluate(self, V: Iterable[str], inputs: dict[str, Any]) -> dict[str, Any]:
        """``f_V``: the values of every gate of ``V`` from ``inputs`` (the values of in(V)), in gate order."""
        local = dict(inputs)
        for name in self.order(V):
            g = self.gate(name)
            local[name] = g.f(*(local[a] for a in g.args))  # type: ignore[misc]
        return local

# ---------------------------------------------------------------- the commitment table -------------

class CommitError(Exception):
    """A prover action the abstract commitment makes impossible (changing a fixed value, adding after seal)."""


class Table:
    """"Committing values fixes them and lets the verifier read them where a check requires it." """

    def __init__(self) -> None:
        self._v: dict[str, Any] = {}
        self._sealed = False
        self.log: list[tuple[str, Any]] = []

    def commit(self, gate: str, value: Any) -> None:
        if self._sealed:
            raise CommitError(f"table sealed: cannot commit {gate}")
        if gate in self._v:
            if self._v[gate] is value or self._v[gate] == value:
                return  # restating the fixed value changes nothing
            raise CommitError(f"{gate} is already committed to {self._v[gate]!r}; cannot change it to {value!r}")
        self._v[gate] = value
        self.log.append((gate, value))

    def seal(self) -> None:
        self._sealed = True

    def __contains__(self, gate: str) -> bool:
        return gate in self._v

    def __getitem__(self, gate: str) -> Any:
        return self._v[gate]

    def snapshot(self) -> dict[str, Any]:
        return dict(self._v)

# ---------------------------------------------------------------- randomness -----------------------

class Coins:
    """Fresh randomness: ``coin(p)`` is a Bernoulli(p) draw, exact for a rational ``p``.  The default
    source is ``random.Random(seed)`` -- *forced-test randomness*, labelled so by every caller."""

    def __init__(self, seed: int | None = None, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random(seed)

    def coin(self, p: Fraction) -> bool:
        p = Fraction(p)
        return self.rng.randrange(p.denominator) < p.numerator


class FixedCoins(Coins):
    """A predetermined sequence of outcomes (tests only)."""

    def __init__(self, outcomes: Sequence[bool]) -> None:
        self._it = iter(outcomes)

    def coin(self, p: Fraction) -> bool:  # noqa: ARG002 -- the law is the caller's; the outcome is forced
        return next(self._it)

# ---------------------------------------------------------------- the protocol ---------------------

@dataclass(frozen=True)
class Verdict:
    accepted: bool
    step: int  # the step at which the verdict was reached (2, 4 or 5)
    reason: str
    J: tuple[int, ...]
    F: tuple[int, ...]
    Q: tuple[int, ...]
    table: dict[str, Any] = field(repr=False, default_factory=dict)


class Prover:
    """The prover's two messages.  ``step1(table) -> y`` commits v_in(C) and v_out(R) for every R and
    declares ``y``; ``step3(J, table) -> F`` sends the exemption request and commits the remaining
    VU outputs of J \\ F.  ``after_Q(Q, table, response)`` exists only so an *adversarial* test can
    attempt a late action; the verifier never reads anything from it."""

    def step1(self, C: Circuit, table: Table) -> tuple[Any, ...]:
        raise NotImplementedError

    def step3(self, C: Circuit, J: tuple[int, ...], table: Table) -> Iterable[int]:
        raise NotImplementedError

    def after_Q(self, C: Circuit, Q: tuple[int, ...], table: Table, F: tuple[int, ...]) -> None:
        return None


class HonestProver(Prover):
    """Evaluates the circuit on ``x`` and commits exactly what each step asks for."""

    def __init__(self, x: Sequence[Any], exempt: Iterable[int] = (), values: dict[str, Any] | None = None) -> None:
        self.x = tuple(x)
        self.exempt = set(exempt)
        self.values = values  # a full assignment to commit instead of the honest one (dishonest prover)

    def assignment(self, C: Circuit) -> dict[str, Any]:
        if self.values is not None:
            return dict(self.values)
        vals = dict(zip(C.inputs, self.x, strict=True))
        for g in C.gates:
            if not g.is_input:
                vals[g.name] = g.f(*(vals[a] for a in g.args))  # type: ignore[misc]
        return vals

    def step1(self, C: Circuit, table: Table) -> tuple[Any, ...]:
        vals = self.assignment(C)
        for g in C.inputs:
            table.commit(g, vals[g])
        for r in range(len(C.S_R)):
            for g in C.out(C.ru_gates(r)):
                table.commit(g, vals[g])
        return tuple(vals[o] for o in C.outputs)

    def step3(self, C: Circuit, J: tuple[int, ...], table: Table) -> Iterable[int]:
        vals = self.assignment(C)
        F = sorted(self.exempt & set(J))
        for r in J:
            if r in F:
                continue
            for k in C.S_R[r]:
                for g in C.out(C.S_V[k]):
                    if g not in table:
                        table.commit(g, vals[g])
        return F


def well_typed(C: Circuit, names: Sequence[str], values: Sequence[Any]) -> str | None:
    if len(names) != len(values):
        return f"arity: {len(values)} values for {len(names)} gates"
    for n, v in zip(names, values, strict=True):
        if not C.gate(n).domain.contains(v):
            return f"{n}={v!r} is not in domain {C.gate(n).domain.name}"
    return None


def run(C: Circuit, x: Sequence[Any], q: Fraction, s: Fraction, B_ex: int, prover: Prover, coins: Coins) -> Verdict:
    """Steps 1-5 of the contract, in order, with the stated barriers."""

    table = Table()
    no = ()

    # 1. Commit.  The prover declares y and commits v_in(C) and v_out(R) for every R in S_R.
    y = tuple(prover.step1(C, table))

    # 2. Check and sample.
    for g in C.inputs:
        if g not in table:
            return Verdict(False, 2, f"no committed value for input gate {g}", no, no, no, table.snapshot())
    for r in range(len(C.S_R)):
        for g in C.out(C.ru_gates(r)):
            if g not in table:
                return Verdict(False, 2, f"no committed value for output gate {g} of RU {r}", no, no, no, table.snapshot())
    if (bad := well_typed(C, C.inputs, x)) is not None:
        return Verdict(False, 2, f"x ill-typed: {bad}", no, no, no, table.snapshot())
    if (bad := well_typed(C, C.outputs, y)) is not None:
        return Verdict(False, 2, f"y ill-typed: {bad}", no, no, no, table.snapshot())
    for g, xv in zip(C.inputs, x, strict=True):
        if not C.gate(g).domain.equal(table[g], xv):
            return Verdict(False, 2, f"v_{g}={table[g]!r} != x={xv!r}", no, no, no, table.snapshot())
    for g, yv in zip(C.outputs, y, strict=True):
        if not C.gate(g).domain.equal(table[g], yv):
            return Verdict(False, 2, f"v_{g}={table[g]!r} != y={yv!r}", no, no, no, table.snapshot())
    J = tuple(r for r in range(len(C.S_R)) if coins.coin(q))  # fresh randomness, each R independently with q

    # 3. Respond.
    F = tuple(sorted(set(prover.step3(C, J, table))))
    table.seal()  # everything the verifier will read is now fixed

    # 4. Check and sample.
    if not set(F) <= set(J):
        return Verdict(False, 4, f"F={F} is not a subset of J={J}", J, F, no, table.snapshot())
    if len(F) > B_ex:
        return Verdict(False, 4, f"|F|={len(F)} exceeds B_ex={B_ex}", J, F, no, table.snapshot())
    retained = [k for r in J if r not in F for k in C.S_R[r]]
    for k in retained:
        for g in C.out(C.S_V[k]):
            if g not in table:
                return Verdict(False, 4, f"no committed value for output gate {g} of VU {k} (RU {C.ru_of_vu(k)})", J, F, no, table.snapshot())
    Q = tuple(k for k in retained if coins.coin(s))  # fresh randomness, each such V independently with s
    prover.after_Q(C, Q, table, F)  # an adversary may try something here; the table is sealed and F is fixed

    # 5. Verify.
    for k in Q:
        V = C.S_V[k]
        ins, outs = C.in_(V), C.out(V)
        vin = [table[g] if g in table else None for g in ins]
        if any(g not in table for g in ins):
            return Verdict(False, 5, f"VU {k}: input {[g for g in ins if g not in table]} has no committed value", J, F, Q, table.snapshot())
        if (bad := well_typed(C, ins, vin)) is not None:
            return Verdict(False, 5, f"VU {k}: v_in ill-typed: {bad}", J, F, Q, table.snapshot())
        vout = [table[g] for g in outs]
        if (bad := well_typed(C, outs, vout)) is not None:
            return Verdict(False, 5, f"VU {k}: v_out ill-typed: {bad}", J, F, Q, table.snapshot())
        computed = C.evaluate(V, dict(zip(ins, vin, strict=True)))
        for g in outs:
            if not C.gate(g).domain.equal(table[g], computed[g]):
                return Verdict(False, 5, f"VU {k}: v_{g}={table[g]!r} != f_V(...)={computed[g]!r}", J, F, Q, table.snapshot())
    return Verdict(True, 5, "accepted", J, F, Q, table.snapshot())

# ---------------------------------------------------------------- the required example circuits ----

def _add1(a):
    return (a + 1) & 0xFF


def _dbl(a):
    return (a * 2) & 0xFF


def _mul3(a):
    return (a * 3) & 0xFF


def _ident(a):
    return a


U8 = u(8)


def shared_input() -> Circuit:
    """An input gate read by VUs in two RUs: a = x+1 (RU 0), b = 2x (RU 1); outputs (a, b)."""
    return Circuit((Gate("x", U8), Gate("a", U8, ("x",), _add1), Gate("b", U8, ("x",), _dbl)),
                   outputs=("a", "b"), S_V=(("a",), ("b",)), S_R=((0,), (1,)))


def crossing_value() -> Circuit:
    """A computed value crossing RUs: u = x+1 (RU 0) is read by y = 3u (RU 1); output y."""
    return Circuit((Gate("x", U8), Gate("u", U8, ("x",), _add1), Gate("y", U8, ("u",), _mul3)),
                   outputs=("y",), S_V=(("u",), ("y",)), S_R=((0,), (1,)))


def output_consumed() -> Circuit:
    """A circuit output consumed later as an input by another VU: y1 = x+1 (RU 0, designated output),
    y2 = 2*y1 (RU 1, designated output); outputs (y1, y2)."""
    return Circuit((Gate("x", U8), Gate("y1", U8, ("x",), _add1), Gate("y2", U8, ("y1",), _dbl)),
                   outputs=("y1", "y2"), S_V=(("y1",), ("y2",)), S_R=((0,), (1,)))


def input_as_output() -> Circuit:
    """An input used directly as an output: outputs (x, z) with z = x+1 (one RU)."""
    return Circuit((Gate("x", U8), Gate("z", U8, ("x",), _add1)),
                   outputs=("x", "z"), S_V=(("z",),), S_R=((0,),))


def chain_u_x_y() -> Circuit:
    """The contract's ordering example: binary gates u = x (RU 0), y = u (RU 1); output y."""
    return Circuit((Gate("x", BIT), Gate("u", BIT, ("x",), _ident), Gate("y", BIT, ("u",), _ident)),
                   outputs=("y",), S_V=(("u",), ("y",)), S_R=((0,), (1,)))


def f32_passthrough() -> Circuit:
    """Bit-pattern f32 identity in two VUs of one RU: a = x (bits), y = a (bits); output y.
    The circuit's equality is bitwise, so a NaN payload or -0 is a legal, checkable value."""
    return Circuit((Gate("x", F32_BITS), Gate("a", F32_BITS, ("x",), _ident), Gate("y", F32_BITS, ("a",), _ident)),
                   outputs=("y",), S_V=(("a",), ("y",)), S_R=((0, 1),))


REQUIRED_CIRCUITS = {"shared_input": shared_input, "crossing_value": crossing_value,
                     "output_consumed": output_consumed, "input_as_output": input_as_output}
