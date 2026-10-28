"""The continuous audit's memory challenger as the Lean model states it (`Security/Definitions/Pous/Model/Continuous.lean`): the oracle
of `test_pous_continuous_difftest.py`. It imports nothing from `verity.core.protocols.pous`.

Two restatements of one set of rules:

* **In Lean's units** (`cont_records`, `cont_accepts` and the definitions under them). Each function transcribes one Lean
  definition, which is its docstring after the first line, and `test_the_oracle_transcribes_the_lean_model` finds that
  text verbatim in the Lean file the first line names. A responder is a `ContMachine` (`rest`, `step`) whose `W`, `pp`,
  `C` and `ω` are fixed for a run and so closed over, and `f` is the primitive that answers its queries.
* **On the memory challenger's clock** (`ClockOracle`): the same rules on what `audit.UnsignedRounds` observes,
  nanoseconds on its own clock. `(D, Q)` becomes `limit = Δ + RTT` and `(D_late, Q_late)` becomes
  `late_limit = Δ_late + RTT`, or `None` for a memory challenger without a hard deadline, which waits for every answer.

  1. Challenge `j` is sent no earlier than round `j − 1` closed (`contSend`).
  2. An answer's delay runs from the memory challenger's own send of its challenge (`contProg`).
  3. A round closes at its answer, or rejected once `late_limit` has passed with none; an answer past `late_limit` is
     rejected too, its label never checked (`contVerdict`'s `.reject`).
  4. Any other answer past `limit` is late, whatever its label; one within `limit` is a hit if it is the challenged
     label and wrong otherwise.
  5. With tolerance `f₀` the run accepts iff at most `f₀` rounds are not hits (`contAccepts`): those are the misses.
     The late count is the late and rejected rounds, every round whose answer was not in within `limit`.

  An action Lean's memory challenger has no step for is refused and changes nothing: a challenge while a round is open
  within its hard deadline, an answer naming a challenge other than the open one, an answer with no challenge open,
  closing a round before its hard deadline. A round with no answer closes at the first action past its hard deadline
  (`expire`, or the next `challenge`, which then sends the next index or refuses because all `n` are sent). Lean closes
  it at `D_late` units; the code's later close only sends the next challenge later (the model's docstring).

The bridge between the two: an answer that takes `d` rounds and `w` queries arrives `max(d·call_ns, w·query_ns)` after
the send, with `D = limit // call_ns`, `Q = limit // query_ns` and likewise `(D_late, Q_late)` from `late_limit`. Then
`d ≤ D ∧ w ≤ Q` iff the answer is in within `limit`, and the two restatements give the same verdicts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

HIT, WRONG, LATE, REJECT = "hit", "wrong", "late", "reject"
"""`ContVerdict`'s constructors."""


@dataclass(frozen=True)
class Ret:
    """`OracleProg.ret a`."""

    value: object


@dataclass(frozen=True)
class Round:
    """`OracleProg.round qs k`: the queries `qs` at once, then `k` on their answers."""

    qs: tuple
    k: Callable[[tuple], Ret | Round]


def run(f: Callable, p: Ret | Round) -> object:
    """Lean `OracleProg.run` (`Security/Definitions/Pous/Game/Oracle.lean`):

    def run (f : M.Qry → M.Ans) : OracleProg M α → α
      | ret a => a
      | round qs k => run f (k (qs.map f))
    """
    while isinstance(p, Round):
        p = p.k(tuple(map(f, p.qs)))
    return p.value


def depth(f: Callable, p: Ret | Round) -> int:
    """Lean `OracleProg.depth` (`Security/Definitions/Pous/Game/Oracle.lean`):

    def depth (f : M.Qry → M.Ans) : OracleProg M α → ℕ
      | ret _ => 0
      | round qs k => depth f (k (qs.map f)) + 1
    """
    n = 0
    while isinstance(p, Round):
        p, n = p.k(tuple(map(f, p.qs))), n + 1
    return n


def work(f: Callable, p: Ret | Round) -> int:
    """Lean `OracleProg.work` (`Security/Definitions/Pous/Game/Oracle.lean`):

    def work (f : M.Qry → M.Ans) : OracleProg M α → ℕ
      | ret _ => 0
      | round qs k => qs.length + work f (k (qs.map f))
    """
    n = 0
    while isinstance(p, Round):
        p, n = p.k(tuple(map(f, p.qs))), n + len(p.qs)
    return n


def machine_round(mach, h: tuple, T: int, m: object, msg: int | None) -> Ret | Round:
    """Lean `ContMachine.round` (`Security/Definitions/Pous/Model/Continuous.lean`):

    def ContMachine.round {sch : Scheme M} (mach : ContMachine sch) (W : Bits sch.n) (pp : Bits sch.t)
        (h : List (Fin sch.B)) : ℕ → List Bool → Option (Fin sch.B) → OracleProg M (Option (Bits sch.ℓ) × List Bool)
      | 0, m, msg =>
        match mach.step W pp h m msg with
        | .inl a => .ret (some a, m)
        | .inr _ => .ret (none, m)
      | T + 1, m, msg =>
        match mach.step W pp h m msg with
        | .inl a => .ret (some a, m)
        | .inr (qs, k) => .round qs fun as => mach.round W pp h T (k as) none
    """
    tag, out = mach.step(h, m, msg)
    if tag == "inl":
        return Ret((out, m))
    if T == 0:
        return Ret((None, m))
    qs, k = out
    return Round(tuple(qs), lambda answers: machine_round(mach, h, T - 1, k(answers), None))


def cont_verdict(D: int, Q: int, Q_late: int, f: Callable, p: Ret | Round, c: bytes) -> str:
    """Lean `contVerdict` (`Security/Definitions/Pous/Model/Continuous.lean`):

    def contVerdict {ℓ : ℕ} {σ : Type} (D Q Q_late : ℕ) (f : M.Qry → M.Ans) (p : OracleProg M (Option (Bits ℓ) × σ))
        (c : Bits ℓ) : ContVerdict :=
      match (p.run f).1 with
      | none => .reject
      | some a =>
        if Q_late < p.work f then .reject
        else if D < p.depth f ∨ Q < p.work f then .late
        else if a = c then .hit else .wrong
    """
    a = run(f, p)[0]
    if a is None:
        return REJECT
    if Q_late < work(f, p):
        return REJECT
    if D < depth(f, p) or Q < work(f, p):
        return LATE
    return HIT if a == c else WRONG


@dataclass(frozen=True)
class ContState:
    """Lean `ContState` (`Security/Definitions/Pous/Model/Continuous.lean`):

    structure ContState (B : ℕ) : Type where
      sent : List (Fin B)
      closed : ℕ
      mem : List Bool
    """

    sent: tuple
    closed: int
    mem: tuple


def cont_start() -> ContState:
    """Lean `ContState.start` (`Security/Definitions/Pous/Model/Continuous.lean`):

    def ContState.start {B : ℕ} : ContState B := ⟨[], 0, []⟩
    """
    return ContState((), 0, ())


def cont_send(arr: Callable[[int], int], s: ContState) -> int:
    """Lean `contSend` (`Security/Definitions/Pous/Model/Continuous.lean`):

    def contSend {B : ℕ} (arr : ℕ → ℕ) (s : ContState B) : ℕ := max (arr s.sent.length) s.closed
    """
    return max(arr(len(s.sent)), s.closed)


def cont_mem(mach, arr: Callable[[int], int], s: ContState) -> tuple:
    """Lean `contMem` (`Security/Definitions/Pous/Model/Continuous.lean`):

    def contMem {sch : Scheme M} (mach : ContMachine sch) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code) (ω : M.Ω)
        (arr : ℕ → ℕ) (s : ContState sch.B) : List Bool :=
      mach.rest W pp C ω s.sent (contSend arr s) s.mem
    """
    return mach.rest(s.sent, cont_send(arr, s), s.mem)


def cont_prog(mach, D_late: int, arr: Callable[[int], int], s: ContState, i: int) -> Ret | Round:
    """Lean `contProg` (`Security/Definitions/Pous/Model/Continuous.lean`):

    def contProg {sch : Scheme M} (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code)
        (ω : M.Ω) (arr : ℕ → ℕ) (s : ContState sch.B) (i : Fin sch.B) : OracleProg M (Option (Bits sch.ℓ) × List Bool) :=
      mach.round W pp s.sent D_late (contMem mach W pp C ω arr s) (some i)
    """
    return machine_round(mach, s.sent, D_late, cont_mem(mach, arr, s), i)


def cont_next(mach, D_late: int, f: Callable, arr: Callable[[int], int], s: ContState, i: int) -> ContState:
    """Lean `contNext` (`Security/Definitions/Pous/Model/Continuous.lean`):

    def contNext {sch : Scheme M} (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code)
        (ω : M.Ω) (arr : ℕ → ℕ) (s : ContState sch.B) (i : Fin sch.B) : ContState sch.B :=
      ⟨s.sent ++ [i], contSend arr s + (contProg mach D_late W pp C ω arr s i).depth (M.answer ω),
        ((contProg mach D_late W pp C ω arr s i).run (M.answer ω)).2⟩
    """
    p = cont_prog(mach, D_late, arr, s, i)
    return ContState(s.sent + (i,), cont_send(arr, s) + depth(f, p), run(f, p)[1])


def cont_after(mach, D_late: int, f: Callable, arr: Callable[[int], int], h: Sequence[int]) -> ContState:
    """Lean `contAfter` (`Security/Definitions/Pous/Model/Continuous.lean`):

    def contAfter {sch : Scheme M} (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
        (C : sch.Code) (ω : M.Ω) (arr : ℕ → ℕ) (h : List (Fin sch.B)) : ContState sch.B :=
      h.foldl (contNext mach D_late W pp C ω arr) .start
    """
    s = cont_start()
    for i in h:
        s = cont_next(mach, D_late, f, arr, s, i)
    return s


@dataclass(frozen=True)
class ContRecord:
    """Lean `ContRecord` (`Security/Definitions/Pous/Model/Continuous.lean`):

    structure ContRecord : Type where
      held : ℕ
      verdict : ContVerdict
    """

    held: int
    verdict: str


def cont_record(mach, D: int, Q: int, D_late: int, Q_late: int, f: Callable, C: Sequence[bytes],
                arr: Callable[[int], int], s: ContState, i: int) -> ContRecord:
    """Lean `contRecord` (`Security/Definitions/Pous/Model/Continuous.lean`):

    def contRecord {sch : Scheme M} (mach : ContMachine sch) (D Q D_late Q_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
        (C : sch.Code) (ω : M.Ω) (arr : ℕ → ℕ) (s : ContState sch.B) (i : Fin sch.B) : ContRecord :=
      ⟨(contMem mach W pp C ω arr s).length,
        contVerdict D Q Q_late (M.answer ω) (contProg mach D_late W pp C ω arr s i) (C i)⟩
    """
    return ContRecord(len(cont_mem(mach, arr, s)), cont_verdict(D, Q, Q_late, f, cont_prog(mach, D_late, arr, s, i), C[i]))


def cont_records(mach, D: int, Q: int, D_late: int, Q_late: int, f: Callable, C: Sequence[bytes],
                 arr: Callable[[int], int], s: ContState, indices: Sequence[int]) -> list[ContRecord]:
    """Lean `contRecords` (`Security/Definitions/Pous/Model/Continuous.lean`):

    def contRecords {sch : Scheme M} (mach : ContMachine sch) (D Q D_late Q_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
        (C : sch.Code) (ω : M.Ω) (arr : ℕ → ℕ) : ContState sch.B → List (Fin sch.B) → List ContRecord
      | _, [] => []
      | s, i :: is => contRecord mach D Q D_late Q_late W pp C ω arr s i ::
          contRecords mach D Q D_late Q_late W pp C ω arr (contNext mach D_late W pp C ω arr s i) is
    """
    out = []
    for i in indices:
        out.append(cont_record(mach, D, Q, D_late, Q_late, f, C, arr, s, i))
        s = cont_next(mach, D_late, f, arr, s, i)
    return out


def cont_accepts(f0: float, rs: Sequence[ContRecord]) -> bool:
    """Lean `contAccepts` (`Security/Definitions/Pous/Model/Continuous.lean`):

    def contAccepts (f₀ : ℝ) (rs : List ContRecord) : Prop :=
      ((rs.filter fun r => r.verdict ≠ .hit).length : ℝ) ≤ f₀
    """
    return len([r for r in rs if r.verdict != HIT]) <= f0


UNANSWERED, ALL_SENT, NONE_OPEN, CLOSED, NOT_REVEALED, WITHIN, NO_HARD_DEADLINE = (
    "a round is open within its hard deadline", "all n challenges are sent", "no challenge is open",
    "the named challenge is closed", "the named challenge is not revealed",
    "the open round is within its hard deadline", "no hard deadline")
"""Why `ClockOracle` refuses an action."""


@dataclass(frozen=True)
class Refused:
    reason: str


@dataclass(frozen=True)
class Record:
    """A closed round: its challenge, index, verdict, and the time from its send to its close."""

    j: int
    index: int
    verdict: str
    elapsed: int


class ClockOracle:
    """The memory challenger on its own clock (the module docstring, "On the memory challenger's clock"): `n`
    challenges, the `j`th asking for label `index(j)` of `C`. `named` is a memory challenger whose answers name their
    challenge (`ContinuousMemoryChallenger.respond(j, …)`); an unnamed one credits every answer to the open challenge.
    Each action takes the clock reading `t` and returns what the memory challenger must do: the `(j, index)` it reveals,
    the `Record` it closes, or a `Refused`."""

    def __init__(self, n: int, index: Callable[[int], int], C: Sequence[bytes], limit: int, late_limit: int | None,
                 named: bool) -> None:
        if late_limit is not None and late_limit < limit:
            raise ValueError("the hard deadline is at least the deadline")
        self.n, self.index, self.C, self.limit, self.late_limit, self.named = n, index, C, limit, late_limit, named
        self.records: list[Record] = []
        self.sends: list[int] = []
        self.open: tuple[int, int, int] | None = None
        """The open challenge's `(j, index, send time)`."""

    @property
    def done(self) -> bool:
        return len(self.records) == self.n

    @property
    def pending(self) -> int | None:
        return None if self.open is None else self.open[0]

    @property
    def misses(self) -> int:
        return sum(r.verdict != HIT for r in self.records)

    @property
    def late(self) -> int:
        return sum(r.verdict in (LATE, REJECT) for r in self.records)

    def accepts(self, f0: int) -> bool:
        return self.done and self.misses <= f0

    def _past_hard_deadline(self, t: int) -> bool:
        return self.late_limit is not None and t - self.open[2] > self.late_limit

    def _close(self, t: int, verdict: str) -> Record:
        j, i, send = self.open
        self.open = None
        self.records.append(Record(j, i, verdict, t - send))
        return self.records[-1]

    def challenge(self, t: int) -> tuple[int, int] | Refused:
        if self.open is not None:
            if not self._past_hard_deadline(t):
                return Refused(UNANSWERED)
            self._close(t, REJECT)
        if self.done:
            return Refused(ALL_SENT)
        j = len(self.records)
        self.open = (j, self.index(j), t)
        self.sends.append(t)
        return j, self.open[1]

    def respond(self, t: int, j: int, block: bytes) -> Record | Refused:
        if self.named and j != self.pending:
            return Refused(CLOSED if j < len(self.records) else NOT_REVEALED)
        if self.open is None:
            return Refused(NONE_OPEN)
        elapsed = t - self.open[2]
        if self.late_limit is not None and elapsed > self.late_limit:
            return self._close(t, REJECT)
        if elapsed > self.limit:
            return self._close(t, LATE)
        return self._close(t, HIT if block == self.C[self.open[1]] else WRONG)

    def expire(self, t: int) -> Record | Refused:
        if self.open is None:
            return Refused(NONE_OPEN)
        if self.late_limit is None:
            return Refused(NO_HARD_DEADLINE)
        if not self._past_hard_deadline(t):
            return Refused(WITHIN)
        return self._close(t, REJECT)
