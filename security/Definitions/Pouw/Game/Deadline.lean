module

public import Definitions.Pouw.Game.Defs
public import Definitions.Core.Challenger
public import Definitions.Core.Site

/-!
# The deadline: a timed run and the verifier's rule (trusted)

A proof of compute utilization says that an accepted trace means at least `W` of work between the salt and a
deadline: a lower bound on the work (`Pouw.Guarantees.EndToEnd`), freshness (preprocessing never reads the salt, by
`Adversary`'s construction) and a deadline. This file models the deadline, minimally, on Core's one clock
(`Definitions.Core.Challenger`, `Definitions.Assumptions.Clock`):

* **A timed run** (`TimedRun`): the prover's cost split by when it is spent: it has spent `spent H s t` by real time
  `t`, so `spentIn H s a b` is what it spends in `(a, b]`.
* **The clock's two entries** (`Stamps`): the auditor's device registered under key `k` signs the salt's release, the
  attester's entry beside the salt coin's `.released` entry, which signs the coin's digest with its reading (`salt s`,
  entry `a`), and the docket, the entry that signs the trace's digest with its reading (`docket H s`, entry `b`);
  `real` is the real time at which the device signed each, which the verifier never reads.
* **The verifier's rule** (`AcceptsBy`): the Site passes the clock's check (`clockDrift`, which refuses by name a Site
  with no auditor's device of key `k`, no drift on its host, or a drift of 1 or more), the docket is bracketed by the
  salt's entry and the docket's entry with `0 ≤ ra.elapsed rb ≤ Δ` on the device's clock (`InTime`), and the audit
  accepts. It reads readings of one device only, never a real time.
* **The real window** (`realWindow`): `Δ / (1 - q)`, the real time `Δ` nanoseconds on a clock within the drift `q` the
  check returned can span, under `Assumptions.ClockWithin`.
* **The budget** (`Adversary.BoundedBy`): `Adversary.Bounded` with the online phase's cost replaced by what the run
  spends in the real window from the salt's entry.
* **What the deadline reads of an adversary** (`Adversaries`): its preprocessing, the trace it commits, and its
  online phase's cost and oracle calls. `CostModel.adversaries` reads them in this game; Pearl-C's game reads its own.

Time meets the cost model only through the named assumption `Pouw.Assumptions.HonestClock`, stated once over
`Adversaries` for both games, and the device's clock meets real time only through `Assumptions.ClockWithin`;
`Pouw.Guarantees.WorkBetweenSaltAndDeadline` is the guarantee.
-/

@[expose] public section

open scoped NNRat

namespace Pouw

variable {Q R S : Type}

/-- A timed run of the online phase. `ts` and `tr` are the salt's release and the trace's receipt as a run's other
readers take them; the deadline's statements read neither, but the device's entries (`Stamps`). On oracle `H` and
salt `s` the prover has spent `spent H s t` of cost by real time `t`. -/
structure TimedRun (Q R S : Type) where
  ts : ℝ
  tr : (Q → R) → S → ℝ
  spent : (Q → R) → S → ℝ → ℝ

/-- The cost the run spends in the time interval `(a, b]`, on oracle `H` and salt `s`. -/
def TimedRun.spentIn (run : TimedRun Q R S) (H : Q → R) (s : S) (a b : ℝ) : ℝ := run.spent H s b - run.spent H s a

/-- **The clock's two entries of a run**, both signed by the auditor's device registered under key `k`: on salt `s`,
the salt's release `salt s` (entry `a`, the attester's entry beside the salt coin's `.released` entry, which signs the
coin's digest with its reading), and on oracle `H` too, the docket's entry `docket H s` (entry `b`, which signs the
trace's digest with its reading); `real e` is the real time in nanoseconds at which the device signed `e`. The verifier
reads the entries' readings, never `real`. -/
structure Stamps (Q R S : Type) (k : String) where
  salt : S → Verity.Challenger.Attested k
  docket : (Q → R) → S → Verity.Challenger.Attested k
  real : Verity.Challenger.Attested k → ℝ

/-- The real time of the salt's entry on salt `s`. -/
def Stamps.ts {k : String} (st : Stamps Q R S k) (s : S) : ℝ := st.real (st.salt s)

/-- The real time of the docket's entry on oracle `H` and salt `s`. -/
def Stamps.tr {k : String} (st : Stamps Q R S k) (H : Q → R) (s : S) : ℝ := st.real (st.docket H s)

/-- **The protocol's encoding of a docket**: the salt coin's digest (`coin`), what a trace's docket reads of an
attester's entry (`reads`: it names the salt's entry), and the trace's digest (`digest`). -/
structure Docket (S Out : Type) (k : String) where
  coin : S → Hash.Msg
  reads : Out → Verity.Challenger.Attested k → Prop
  digest : Out → Hash.Msg

/-- Why the verifier refuses to time on a Site's device: the Site has no auditor's device of the key (`noDevice`), its
host registers no drift (`noDrift`), or the drift is 1 or more (`driftNotBelowOne`). Each makes the clock's premise
hold emptily or say nothing. -/
inductive ClockRefusal where
  | noDevice
  | noDrift
  | driftNotBelowOne
  deriving DecidableEq

/-- **The clock's check**: the first auditor's device registered under key `k` in the Site, the drift `q` its host
registers, and `q < 1`; else the refusal, by name. -/
def clockDrift (site : Verity.Site) (k : String) : Except ClockRefusal ℚ≥0 :=
  match site.locs.find? (fun x => x.kind.auditor && x.key == k) with
  | none => .error .noDevice
  | some x =>
    match (x.parent.bind site.loc?).bind (·.drift) with
    | none => .error .noDrift
    | some q => if q < 1 then .ok q else .error .driftNotBelowOne

/-- **The real window** of a deadline of `Δ` nanoseconds on the device registered under key `k`: `Δ / (1 - q)` at
the drift `q` the clock's check returns, the most real time `Δ` counts on a clock that is not slow; `0` where the
check refuses, which the verifier then refuses too. -/
noncomputable def realWindow (site : Verity.Site) (k : String) (Δ : ℕ) : ℝ :=
  match clockDrift site k with
  | .ok q => Δ / (1 - (q : ℝ))
  | .error _ => 0

/-- **In time**: the salt's entry signs the salt coin's digest; the trace's docket `e` is bracketed by the salt's entry
`a` and the docket's entry `b` (`Challenger.Bracketed`: `e` reads `a`, `b` signs `e`'s digest, both carry readings);
and the device's clock counts `0 ≤ ra.elapsed rb ≤ Δ` between them. -/
def InTime {Out : Type} {k : String} (dk : Docket S Out k) (st : Stamps Q R S k) (Δ : ℕ) (H : Q → R) (s : S)
    (e : Out) : Prop :=
  dk.coin s ∈ (st.salt s).shown ∧ ∃ ra rb : Verity.Challenger.Reading k,
    Verity.Challenger.Bracketed dk.reads dk.digest e (st.salt s) (st.docket H s) ra rb ∧
      0 ≤ ra.elapsed rb ∧ ra.elapsed rb ≤ Δ

/-- **The verifier's rule at the deadline of `Δ` nanoseconds on the device registered under key `k`**: the Site passes
the clock's check, the trace's docket `e` is in time, and the audit with the samples `c` accepts (every sampled proof
block is in `K`). -/
def AcceptsBy {Out : Type} {k : String} (site : Verity.Site) (dk : Docket S Out k) (st : Stamps Q R S k) (Δ : ℕ)
    (H : Q → R) (s : S) (e : Out) {w : ℕ → ℕ} {N t : ℕ} (K : Finset ℕ) (c : Fin t → Ticket w N) : Prop :=
  (∃ q, clockDrift site k = .ok q) ∧ InTime dk st Δ H s e ∧ AuditAccepts K c

/-- The run spends at most `T` in the real window of length `d` from the salt's entry, and preprocessing and the
online phase together make at most `q` oracle calls, for every oracle and salt (surely). -/
def Adversary.BoundedBy {CM : CostModel Q R S} {k : String} (A : Adversary CM) (U : Workload) (st : Stamps Q R S k)
    (run : TimedRun Q R S) (d T : ℝ) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), run.spentIn H s (st.ts s) (st.ts s + d) ≤ T ∧
    A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

/-- What the deadline reads of a game's adversaries, of type `Adv`: each one's preprocessing, the trace in `Out` it
commits on oracle `H` and salt `s`, its online phase's cost and oracle calls there, and the inputs it hashes with
SHA-512 there (none in a game whose break is not a collision). -/
structure Adversaries (Q R S Adv Out : Type) where
  pre : Adv → QTree Q R State
  trace : Adv → (Q → R) → S → Out
  cost : Adv → (Q → R) → S → ℝ
  calls : Adv → (Q → R) → S → ℕ
  hashed : Adv → (Q → R) → S → List Hash.Msg := fun _ _ _ => []

/-- The adversaries of the cost model `CM` on the workload `U`, as the deadline reads them. -/
def CostModel.adversaries (CM : CostModel Q R S) (U : Workload) : Adversaries Q R S (Adversary CM) Trace where
  pre A := A.pre
  trace A := A.trace U
  cost A H s := CM.cost A.online H U A.act (A.pre.eval H) s
  calls A H s := CM.calls A.online H U A.act (A.pre.eval H) s
  hashed A := A.hashed U

end Pouw
