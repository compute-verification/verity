import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Order.ConditionallyCompleteLattice.Indexed
import Definitions.Pous.Model.Audit

/-!
# The continuous audit: a memory challenger, and a responder that runs on its clock

Its guarantees are `Pous.Guarantees.PerAnswer` and `Pous.Guarantees.ContinuousAudit`, and, over the arrivals and the
memory at every time (the last section), `Pous.Guarantees.ContinuousTimeAverage`. Red team: needed. A round-structured
model would have to assume three requirements on the protocol (no pipelining, a hard deadline, every answer past `Δ`
counted late). Here the memory challenger is explicit and enforces each rule. The responder may
try to break any of them, and no statement takes a hypothesis about its timing.

**Time.** One unit is one round of parallel queries to the primitive (`Prog.depth`), as in the sequential audit. `D`
units and `Q` queries fit in the deadline `Δ` plus the round trip, and `D_late` units and `Q_late` queries in the hard
deadline `Δ_late` plus the round trip. The memory challenger's arrival times `arr j` are in the same units.

**The responder** (`ContMachine`) is a machine on the memory challenger's clock, with a memory of any length.

* When a challenge is sent at time `t`, its memory is `rest W pp C ω h t m`. That is any function of `W`, `pp`, the
  codeword, the whole primitive, the indices sent before (`h`), `t`, and its memory `m` when the last round closed.
  Between rounds it may recompute anything, for as long as it likes.
* During a round it runs one `step` per unit on `W`, `pp`, `h`, its memory and the memory challenger's message: the
  index at the round's first unit, `none` after. A step either answers, or sends one round of queries and replaces its
  memory by any function of their answers. During a round the responder sees neither `C` nor `ω` whole and reaches the
  primitive only through queries.

**The memory challenger.** For `j = 0, 1, …` (with the names of
`verity.protocols.accounting.space.pous.continuous.ContinuousMemoryChallenger`, which enforces the same rules on
`audit.UnsignedRounds`):

1. It sends index `i_j` at `max(arr j, c_{j−1})` (`contSend`), where `arr j` is challenge `j`'s arrival
   (`run_continuous`'s `wait(j)`), `c_{j−1}` is the time round `j − 1` closed and `c_{−1} = 0`. A challenge is never
   outstanding when the next is sent: `UnsignedRounds.challenge()` refuses while one is open within its hard deadline.
2. It times the answer from its own send time. The answer's delay is the number of units the machine took, the depth
   of the round's program (`contProg`), as `UnsignedRounds.respond()` measures `stop − start` from `challenge()`. A
   round's answer is the answer to its own index, as `ContinuousMemoryChallenger.respond(j, …)` refuses an answer to any
   challenge but the open one and `run_continuous` drops a reply naming any other.
3. It waits at most `D_late` units. The round closes at the answer, or after `D_late` units with no answer
   (`contNext`). A round with no answer by then, or whose answer took more than `Q_late` queries, is **rejected**
   (`contVerdict`): past the hard deadline, its block never checked (`Answer.rejected`, past `Δ_late + RTT`, closed by
   `respond()`, `expire()` or the next `challenge()`).
4. Any other answer that took more than `D` units or `Q` queries is **late**, whatever its block: past `Δ`
   (`Answer.on_time` is false past `Δ + RTT`). An answer within `(D, Q)` is a **hit** if it is the block `C i_j`
   (`Answer.correct`), and **wrong** otherwise.
5. With tolerance `f₀` it accepts iff at most `f₀` rounds are not hits: rejected, late or wrong (`contAccepts`), as
   `ContinuousMemoryChallenger.accepted` compares `AuditResult.misses` with `max_misses`.

`contRecords` runs rules 1–4 over the indices. Each round's record holds its verdict and the length of the responder's
memory at the send time. The memory length is the game's measurement, which the memory challenger never sees.

A memory challenger that closes a round later than rule 3 (`run_continuous` closes a silent round with `expire()` once
its transport reports nothing by the hard deadline, which may be a little after it) only sends the next index later.
That gives the responder nothing `rest` lacks: `rest` may compute anything between rounds, so a machine whose `rest`
returns the memory the responder would hold at the later send time has the same records.

**The three requirements are theorems about this memory challenger.**
`Security/Proofs/Pous/Continuous/Verifier.lean` proves each one for every machine:

1. **No pipelining** (rule 1; `PousContinuous.no_pipelining`, `PousContinuous.close_le_send`). Round `j`'s record
   depends only on `i_0, …, i_j`, and round `j + 1` is sent no earlier than round `j` closed.
2. **A hard deadline** (rule 3; `PousContinuous.closed_le`, `PousContinuous.not_reject`). Every round closes within
   `D_late` units of its own send time. A round that is not rejected answered within `D_late` units and `Q_late`
   queries.
3. **Every answer past `Δ` is counted** (rule 4; `PousContinuous.hit_on_time`). A hit took at most `D` units and `Q`
   queries, so an answer past `Δ` is late or rejected, never a hit.

The continuous bound holds at every `(D_late, Q_late)`, so it does not need the hard deadline, which the memory
challenger enforces anyway. A late answer is already a miss, and the memory is measured at each send time whatever the
machine did before. No cap like the graded audit's `D_late ≤ 12` (`audit.LATE_ROUNDS`) is a hypothesis here.

**What is measured.** `ε_j = 1 − |m_j|/|C|` (`contFreed`), where `m_j` is the memory at challenge `j`'s send time
(`contMem`). `ContinuousAudit` bounds `Σ_j max(ε_j − ε_proof, 0)`, with `ε_proof = 1 − (β − B)/|C|` (`epsProof`).

**What a reader takes from the implementation.** These are not timing rules but the scope of the bound:

* **The arrival process is the memory challenger's own randomness.** It must be independent of the setup, the indices
  and the answers (a Poisson process from its own `verity.primitives.randomness` domain, say). The number of challenges
  `N` must not adapt to the misses, though rejecting early is fine. `PerAnswer` and `ContinuousAudit` hold for every
  fixed schedule `arr`, so they hold conditional on the arrivals. Per challenge, normalize by the realized `N`, never by
  `λT`. The responder may know every arrival time in advance, and a queued challenge is sent when the last round
  closes, which the responder controls. Both only help it, and those statements hold for every machine.
  `ContinuousTimeAverage` (the last section) draws the arrivals in the game instead: the responder does not know them
  in advance, and the bound normalizes by `λT`.
* **Storage at send times, not a time average,** in `PerAnswer` and `ContinuousAudit`. `ε_j` is read at the send
  times. Between them a machine may hold less, down to nothing, and rebuild before the next send (`rest` sees `C` and
  `ω`), so those bounds say nothing about the time average. `ContinuousTimeAverage` reads the memory at every idle
  slot, and the memory at a slot cannot depend on whether a challenge arrives there. It reads nothing at a busy slot,
  and the responder chooses which slots are busy (the last section), so it bounds an average over the idle slots, not
  over the whole period.
* **No messages in flight.** A round's answer is computed from the memory at the send time and the round's queries.
  Data the responder keeps outside its memory, on the wire say, and reads back during a round is outside help, which
  the deployment must exclude from each send to its close.
* **Indices secret until sent** (`audit.challenge_key`), uniform and independent of the setup.
* **Work is modeled.** The memory challenger sees only time. `(D, Q)` and `(D_late, Q_late)` are what the adversary
  completes within `Δ` and `Δ_late`, the sequential audit's convention.
* **Wrong and rejected answers count as misses** rather than rejecting the run, as `AuditResult.misses` counts them. A
  memory challenger that rejects the run on either, or past a cap on the audit's duration (`audit_cap`), accepts fewer
  runs, so the bound covers it too.
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- A responder in the continuous audit (the module docstring, "The responder"). `rest W pp C ω h t m` is its memory
at the send time `t` of the challenge after the indices `h`, from its memory `m` when the last round closed. During a
round, `step W pp h m msg` either answers (`inl`) or sends one round of queries and continues on their answers
(`inr`). `msg` is the round's index at its first unit and `none` after. -/
structure ContMachine (sch : Scheme M) : Type where
  rest : Bits sch.n → Bits sch.t → sch.Code → M.Ω → List (Fin sch.B) → ℕ → List Bool → List Bool
  step : Bits sch.n → Bits sch.t → List (Fin sch.B) → List Bool → Option (Fin sch.B) →
    Bits sch.ℓ ⊕ (List M.Qry × (List M.Ans → List Bool))

/-- One round of the machine after the indices `h`, from memory `m`, with the message `msg` at its first unit and at
most `T` units of queries. It returns the answer, if one came within `T` units, and the memory when the round closed. -/
def ContMachine.round {sch : Scheme M} (mach : ContMachine sch) (W : Bits sch.n) (pp : Bits sch.t)
    (h : List (Fin sch.B)) : ℕ → List Bool → Option (Fin sch.B) → Prog M (Option (Bits sch.ℓ) × List Bool)
  | 0, m, msg =>
    match mach.step W pp h m msg with
    | .inl a => .ret (some a, m)
    | .inr _ => .ret (none, m)
  | T + 1, m, msg =>
    match mach.step W pp h m msg with
    | .inl a => .ret (some a, m)
    | .inr (qs, k) => .round qs fun as => mach.round W pp h T (k as) none

/-- The memory challenger's verdict on one round. -/
inductive ContVerdict : Type
  /-- Within `D` units and `Q` queries, with the right block. -/
  | hit
  /-- Within `D` units and `Q` queries, with a wrong block. -/
  | wrong
  /-- Within the hard deadline but past `(D, Q)`, whatever the block. -/
  | late
  /-- No answer within `D_late` units, or more than `Q_late` queries: past the hard deadline, the block unchecked. -/
  | reject
  deriving DecidableEq

/-- The verdict on a round whose program is `p`, when the primitive answers by `f` and the right block is `c` (rules 3
and 4). The program stops after `D_late` units on its own (`ContMachine.round`), so here a missing answer is one that
did not come within `D_late` units. -/
def contVerdict {ℓ : ℕ} {σ : Type} (D Q Q_late : ℕ) (f : M.Qry → M.Ans) (p : Prog M (Option (Bits ℓ) × σ))
    (c : Bits ℓ) : ContVerdict :=
  match (p.run f).1 with
  | none => .reject
  | some a =>
    if Q_late < p.work f then .reject
    else if D < p.depth f ∨ Q < p.work f then .late
    else if a = c then .hit else .wrong

/-- The memory challenger and the responder between rounds: the indices sent so far (oldest first), the time the last
round closed, and the responder's memory then. -/
structure ContState (B : ℕ) : Type where
  sent : List (Fin B)
  closed : ℕ
  mem : List Bool

/-- Before the first challenge: nothing sent, time `0`, empty memory. -/
def ContState.start {B : ℕ} : ContState B := ⟨[], 0, []⟩

/-- Rule 1: the next challenge is sent at its arrival time, or when the last round closed if that is later. -/
def contSend {B : ℕ} (arr : ℕ → ℕ) (s : ContState B) : ℕ := max (arr s.sent.length) s.closed

/-- The responder's memory when the next challenge is sent. -/
def contMem {sch : Scheme M} (mach : ContMachine sch) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code) (ω : M.Ω)
    (arr : ℕ → ℕ) (s : ContState sch.B) : List Bool :=
  mach.rest W pp C ω s.sent (contSend arr s) s.mem

/-- The program of the round that answers index `i` after the state `s`: the machine from its memory at the send
time, with `i` at the first unit and at most `D_late` units (rules 2 and 3). -/
def contProg {sch : Scheme M} (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code)
    (ω : M.Ω) (arr : ℕ → ℕ) (s : ContState sch.B) (i : Fin sch.B) : Prog M (Option (Bits sch.ℓ) × List Bool) :=
  mach.round W pp s.sent D_late (contMem mach W pp C ω arr s) (some i)

/-- The state after that round: `i` sent, the round closed as many units after its send time as its program took on
the primitive's path (rule 2), and the memory then. -/
def contNext {sch : Scheme M} (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code)
    (ω : M.Ω) (arr : ℕ → ℕ) (s : ContState sch.B) (i : Fin sch.B) : ContState sch.B :=
  ⟨s.sent ++ [i], contSend arr s + (contProg mach D_late W pp C ω arr s i).depth (M.answer ω),
    ((contProg mach D_late W pp C ω arr s i).run (M.answer ω)).2⟩

/-- The state after the indices `h`, from the start. -/
def contAfter {sch : Scheme M} (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
    (C : sch.Code) (ω : M.Ω) (arr : ℕ → ℕ) (h : List (Fin sch.B)) : ContState sch.B :=
  h.foldl (contNext mach D_late W pp C ω arr) .start

/-- What the audit records of one round: the length of the responder's memory at the send time (the game's
measurement, which the memory challenger never sees) and the verdict. -/
structure ContRecord : Type where
  held : ℕ
  verdict : ContVerdict

/-- The record of the round that answers index `i` after the state `s`. -/
def contRecord {sch : Scheme M} (mach : ContMachine sch) (D Q D_late Q_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
    (C : sch.Code) (ω : M.Ω) (arr : ℕ → ℕ) (s : ContState sch.B) (i : Fin sch.B) : ContRecord :=
  ⟨(contMem mach W pp C ω arr s).length,
    contVerdict D Q Q_late (M.answer ω) (contProg mach D_late W pp C ω arr s i) (C i)⟩

/-- The records of the rounds for the indices `is`, from the state `s`: rules 1–4, one round after another. -/
def contRecords {sch : Scheme M} (mach : ContMachine sch) (D Q D_late Q_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
    (C : sch.Code) (ω : M.Ω) (arr : ℕ → ℕ) : ContState sch.B → List (Fin sch.B) → List ContRecord
  | _, [] => []
  | s, i :: is => contRecord mach D Q D_late Q_late W pp C ω arr s i ::
      contRecords mach D Q D_late Q_late W pp C ω arr (contNext mach D_late W pp C ω arr s i) is

/-- Rule 5: with tolerance `f₀` the memory challenger accepts iff at most `f₀` rounds are not hits (rejected, late or
wrong). -/
def contAccepts (f₀ : ℝ) (rs : List ContRecord) : Prop :=
  ((rs.filter fun r => r.verdict ≠ .hit).length : ℝ) ≤ f₀

/-- The freed fraction of a memory of `held` bits: `ε = 1 − held/|C|`. -/
noncomputable def contFreed (sch : Scheme M) (held : ℕ) : ℝ :=
  1 - (held : ℝ) / (sch.B * sch.ℓ)

/-- The per-answer slack of a compression bound to `β` bits: `ε_proof = 1 − (β − B)/|C|`. With
`TimedINC sch D Q β π`, an answer from a memory of `(1 − ε)·|C|` bits is a hit with probability at most
`1 − ε + ε_proof`, off a setup event of probability at most `(β + 1)·π` (`Pous.Guarantees.PerAnswer`). -/
noncomputable def epsProof (sch : Scheme M) (β : ℕ) : ℝ :=
  1 - ((β : ℝ) - sch.B) / (sch.B * sch.ℓ)

/-! ## Arrivals drawn by the memory challenger, and the memory at every time

Its statement is `Pous.Guarantees.ContinuousTimeAverage`. Red team: needed.

The memory challenger draws the arrivals itself. Time is cut into `T` slots of one unit each, the unit of `Prog.depth`
and of `arr` above. At slot `t` the memory challenger draws `x t = (c, i) ∈ Fin b × Fin B`, uniform and independent of
everything else. A challenge arrives at `t` iff `c < a` (probability `a/b`): a Bernoulli process, at most one arrival
per slot, standing in for a Poisson process at `λ = (a/b)/call_ns`. A deployment is inside the statements only if it
draws its arrivals this way, one coin per slot from its own randomness. The `j`-th arrival's index is the `i` drawn at
its slot (`histAt`), and its arrival time is
`arrAt a x j`. Rules 1–5 are unchanged: `contRecords` with `arr := arrAt a x` runs the memory challenger on these
arrivals.

**Arrivals queue.** Rule 1 sends challenge `j` at `max(arrAt a x j, c_{j−1})`, so a challenge that arrives while a
round is open waits for it to close. None is dropped. This is load-bearing: at an idle slot (below) the indices sent
so far are `histAt a x t`, a function of the memory challenger's draws alone. A memory challenger that dropped arrivals
during an open round would send a history that depends on the rounds' depths, and so on the primitive.

**Idle slots.** Slot `t` is idle (`contIdle`) when every challenge that arrived before `t` has closed by `t`, so a
challenge arriving at `t` is sent at `t`. At a busy slot a round is open or a challenge is queued, and
`ContinuousTimeAverage` does not read the memory there.

**The responder chooses the busy slots.** A round lasts as many units as its program takes: at most `D` for a hit or
a wrong answer (`contVerdict`, `PousContinuous.hit_on_time`), at most `D_late` otherwise
(`PousContinuous.closed_le`). A busy slot lies inside some round, so over `n` challenges of which `F` miss, at most
`D·n + (D_late − D)·F` slots are busy when `D ≤ D_late` (not proved here). The responder may make them busy at no
cost but those misses: one that pads every hit to `D` units, answering at the deadline, misses nothing, and at
`a/b = 0.02415` (100 challenges/s on 241.5 µs slots) it makes about 2.5% of the slots busy at `D = 2` and 9.9% at
`D = 5`. At `D_late ≤ 1` no slot is busy: a round closes by the next slot. During a round the memory is not
measured: the responder may keep only what the round's answer needs, and `rest`, which sees `C` and `ω`, restores
the rest by the next idle slot.

**The memory at time `t`** (`contHeldAt`) is `rest W pp C ω h t m` after the challenges that arrived before `t`: what
the responder would present if a challenge were sent at `t`. At an idle slot it is the memory a challenge arriving at
`t` meets. `rest` takes no argument about future arrivals, so the memory at `t` cannot depend on whether a challenge
arrives at `t` (the model's PASTA property, by construction). The memory challenger never sees it; it is the game's
measurement, as `ContRecord.held` is.
-/

/-- The slots whose coin is below `a`: the challenges' arrivals, in order. -/
def arrivals (a : ℕ) {T b B : ℕ} (x : Fin T → Fin b × Fin B) : List (Fin T) :=
  (List.finRange T).filter fun t => ((x t).1 : ℕ) < a

/-- Challenge `j`'s arrival time, or `T` once the arrivals run out. -/
def arrAt (a : ℕ) {T b B : ℕ} (x : Fin T → Fin b × Fin B) (j : ℕ) : ℕ :=
  ((arrivals a x).map Fin.val).getD j T

/-- The indices of the challenges that arrived before time `t`, oldest first. -/
def histAt (a : ℕ) {T b B : ℕ} (x : Fin T → Fin b × Fin B) (t : ℕ) : List (Fin B) :=
  ((arrivals a x).filter fun s : Fin T => (s : ℕ) < t).map fun s => (x s).2

/-- Time `t` is idle: every challenge that arrived before `t` has closed by `t`, so one arriving at `t` is sent at
`t` (rule 1). -/
def contIdle {sch : Scheme M} (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
    (C : sch.Code) (ω : M.Ω) (a : ℕ) {T b : ℕ} (x : Fin T → Fin b × Fin sch.B) (t : ℕ) : Prop :=
  (contAfter mach D_late W pp C ω (arrAt a x) (histAt a x t)).closed ≤ t

instance instDecidableContIdle {sch : Scheme M} (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code)
    (ω : M.Ω) (a : ℕ) {T b : ℕ} (x : Fin T → Fin b × Fin sch.B) (t : ℕ) :
    Decidable (contIdle mach D_late W pp C ω a x t) :=
  inferInstanceAs (Decidable (_ ≤ t))

/-- The length of the responder's memory at time `t`: what it would hold if a challenge were sent at `t`. -/
def contHeldAt {sch : Scheme M} (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
    (C : sch.Code) (ω : M.Ω) (a : ℕ) {T b : ℕ} (x : Fin T → Fin b × Fin sch.B) (t : ℕ) : ℕ :=
  let s := contAfter mach D_late W pp C ω (arrAt a x) (histAt a x t)
  (mach.rest W pp C ω s.sent t s.mem).length

end Pous
