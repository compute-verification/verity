import Definitions.Accounting.Joint
import Definitions.Pouw.Game.Defs
import Definitions.Pouw.PearlC.Game

/-!
# Per-run audits, and what makes PoUW's cost model truncate (trusted)

An audit's guarantee in PoUW's `EndToEnd` shape (`Accounting.Run.ComputeSound` and the two beside it) bounds the
probability that it accepts with a sure bound below its certificate: the bound holds on every one of the audit's own
coins. A device that does little charged work on a few coins and honest work on the rest has a large sure bound, so
those guarantees say nothing about it, and nothing about the covert work its cheap runs leave room for. A per-run
guarantee bounds the probability that the audit accepts *and the account's realized charge on that run* is below the
certificate (`Run.ComputeSoundPerRun`, `Run.BandwidthSoundPerRun`, `Run.CapacitySoundPerRun`), at every value of the
other audits' coins. It implies the sure-bound form.

PoUW's guarantee becomes per-run by truncation: the adversary that runs `A` but stops before its cost would exceed a
budget `b` costs at most `b` on every coin, and commits `A`'s transcript wherever `A` costs at most `b`. PoUW's cost
model is abstract, so whether that adversary exists is a property of the model (`BudgetAbort`), not a theorem: a model
whose cost reads what its programs cannot see (the oracle at points they never query, say) has no aborting program.

**Pricing.** The machine prices a PoUW run in two parts (Daniel's decision 9, 11 Oct): its arithmetic at W1's
instruction prices, with no fetch charge and no refund, and every first fetch of a pre-salt or supplied byte,
deduplicated within the run, at the device's price of a bit read. In the device model those fetches are reads by the
compute account (`Run.pouwRead`). The compute audit's certificate is on the priced sum, the compute account's work plus
`pRead` per bit it read from device memory (`Run.ComputeSoundPerRunPriced`, `pRead` the price of a bit read in the unit
the work is counted in), and tables cite the two terms separately. PoUW's reads and the bandwidth audit's window reads
are charged to different accounts over one window (`Run.ReadsAttributed`), so no bit read is credited to both
certificates. Pearl-C's game is a game of its own, with its own cost model; `BudgetAbortPearlC` is truncation for it,
and `RealizesPearlCPriced` is a run whose compute audit is Pearl-C's audit, its cost covered by the priced sum.
-/

namespace Accounting

open Pouw

section Pouw

variable {Q R S : Type}

/-- **PoUW's cost model truncates.** Its costs are nonnegative, and it is closed under budget-abort: for every program,
workload, activation input and budget `b ≥ 0` there is one program, fixed before the oracle, the preprocessed state and
the salt, that runs the program and stops before its cost would exceed `b`. On every oracle, preprocessed state and salt
it costs at most `b` and makes at most the program's oracle calls, and wherever the program costs at most `b` it
commits the program's transcript. -/
def BudgetAbort (CM : CostModel Q R S) : Prop :=
  (∀ (prog : CM.Prog) (H : Q → R) (U : Workload) (act : ℕ → Mat) (σ : State) (s : S),
      0 ≤ CM.cost prog H U act σ s) ∧
    ∀ (prog : CM.Prog) (U : Workload) (act : ℕ → Mat) (b : ℝ), 0 ≤ b → ∃ prog' : CM.Prog,
      ∀ (H : Q → R) (σ : State) (s : S),
        CM.cost prog' H U act σ s ≤ b ∧ CM.calls prog' H U act σ s ≤ CM.calls prog H U act σ s ∧
          (CM.cost prog H U act σ s ≤ b → CM.run prog' H U act σ s = CM.run prog H U act σ s)

/-- The adversary's realized online cost on workload `U`, oracle `H` and salt `s`. -/
def onlineCost {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (H : Q → R) (s : S) : ℝ :=
  CM.cost A.online H U A.act (A.pre.eval H) s

/-- Preprocessing and the online phase together make at most `q` oracle calls, for every oracle and salt: the oracle
half of `Pouw.Adversary.Bounded`, with no bound on the cost. -/
def CallsWithin {CM : CostModel Q R S} (A : Adversary CM) (U : Workload) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

end Pouw

section PearlC

variable {Q R S : Type}

/-- **Pearl-C's cost model truncates**: `BudgetAbort`'s statement over Pearl-C's `CostModel`, whose programs take
activation codes. Its costs are nonnegative, and for every program, workload, activation input and budget `b ≥ 0` there
is one program, fixed before the oracle, the preprocessed state and the salt, that costs at most `b`, makes at most the
program's oracle calls, and commits the program's transcript wherever the program costs at most `b`. -/
def BudgetAbortPearlC (CM : PearlC.CostModel Q R S) : Prop :=
  (∀ (prog : CM.Prog) (H : Q → R) (U : PearlC.Workload) (act : ℕ → PearlC.Codes) (σ : PearlC.State) (s : S),
      0 ≤ CM.cost prog H U act σ s) ∧
    ∀ (prog : CM.Prog) (U : PearlC.Workload) (act : ℕ → PearlC.Codes) (b : ℝ), 0 ≤ b → ∃ prog' : CM.Prog,
      ∀ (H : Q → R) (σ : PearlC.State) (s : S),
        CM.cost prog' H U act σ s ≤ b ∧ CM.calls prog' H U act σ s ≤ CM.calls prog H U act σ s ∧
          (CM.cost prog H U act σ s ≤ b → CM.run prog' H U act σ s = CM.run prog H U act σ s)

/-- A Pearl-C adversary's realized online cost on workload `U`, oracle `H` and salt `s`. -/
def onlineCostPearlC {CM : PearlC.CostModel Q R S} (A : PearlC.Adversary CM) (U : PearlC.Workload) (H : Q → R)
    (s : S) : ℝ :=
  CM.cost A.online H U A.act (A.pre.eval H) s

/-- A Pearl-C adversary's preprocessing and online phase together make at most `q` oracle calls, for every oracle and
salt. -/
def CallsWithinPearlC {CM : PearlC.CostModel Q R S} (A : PearlC.Adversary CM) (U : PearlC.Workload) (q : ℕ) : Prop :=
  ∀ (H : Q → R) (s : S), A.pre.calls H + CM.calls A.online H U A.act (A.pre.eval H) s ≤ q

/-- The compute audit's coins in Pearl-C's game at workload `U`, with `t` tickets weighted by `w`: the oracle, the salt
and the tickets. -/
abbrev PearlCCoins (Q R S : Type) (U : PearlC.Workload) (w : ℕ → ℕ) (t : ℕ) : Type :=
  (Q → R) × S × (Fin t → Ticket w U.layout.N)

end PearlC

namespace Run

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] (r : Run d α ΩW ΩQ ΩS)

/-- PoUW's reads (`read_pouw`): the bits the compute account read from device memory over the compute window. -/
def pouwRead (ω : ΩW × ΩQ × ΩS) : ℕ := (r.schedule ω).read (r.win .compute) (r.acct .compute)

/-- The bits every account read from device memory over the bandwidth window (`read_total`). -/
def totalRead (ω : ΩW × ΩQ × ΩS) : ℕ := ∑ a, (r.schedule ω).read (r.win .bandwidth) a

/-- **The reads are attributed**: PoUW's reads and the bandwidth audit's window reads are charged to different accounts
over one window. The device model charges each bit read to one account, so no bit is in both `pouwRead` and `read`,
and `read_total ≥ read + read_pouw` on every run. -/
def ReadsAttributed : Prop := r.acct .compute ≠ r.acct .bandwidth ∧ r.win .compute = r.win .bandwidth

/-- The bits read from device memory over the bandwidth window by the accounts other than the bandwidth and compute
accounts: the reads that neither the bandwidth certificate nor the priced compute certificate credits. -/
def covertRead [DecidableEq α] (ω : ΩW × ΩQ × ΩS) : ℕ :=
  ∑ a ∈ (Finset.univ.erase (r.acct .bandwidth)).erase (r.acct .compute), (r.schedule ω).read (r.win .bandwidth) a

variable [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS]

/-- **The compute audit's per-run guarantee, at price `p`, certificate `W` and error `ε`**: at any value of the other
audits' coins, the probability over the compute coins that the compute audit accepts and the compute account's work on
that run is below `W` is at most `ε`. -/
def ComputeSoundPerRun (p : Price) (W ε : ℝ) : Prop :=
  ∀ o : ΩQ × ΩS, Pouw.pr (fun x : ΩW => r.accepts .compute (x, o) ∧ (r.work p (x, o) : ℝ) < W) ≤ ε

/-- **The bandwidth audit's per-run guarantee, at certificate `Q` and error `ε`**: at any value of the other audits'
coins, the probability over the bandwidth coins that the bandwidth audit accepts and the bandwidth account read fewer
than `Q` bits from device memory on that run is at most `ε`. -/
def BandwidthSoundPerRun (Q ε : ℝ) : Prop :=
  ∀ (x : ΩW) (s : ΩS), Pouw.pr (fun q : ΩQ => r.accepts .bandwidth (x, q, s) ∧ (r.read (x, q, s) : ℝ) < Q) ≤ ε

/-- **The capacity audit's per-run guarantee, at certificate `S` bits on average and error `ε`**: at any value of the
other audits' coins, the probability over the capacity coins that the capacity audit accepts and the capacity account
held fewer than `S * len` bit-ticks of device memory over its window on that run is at most `ε`. -/
def CapacitySoundPerRun (S ε : ℝ) : Prop :=
  ∀ (x : ΩW) (q : ΩQ),
    Pouw.pr (fun s : ΩS => r.accepts .capacity (x, q, s) ∧ (r.held (x, q, s) : ℝ) < S * (r.win .capacity).len) ≤ ε

/-- **The compute audit's per-run guarantee on the priced sum, at price `p`, read price `pRead`, certificate `W` and
error `ε`**: at any value of the other audits' coins, the probability over the compute coins that the compute audit
accepts and, on that run, the compute account's work plus `pRead` per bit it read from device memory is below `W` is at
most `ε`. `pRead` is the device's price of one bit read, in the unit `p` counts work in. Tables cite the work and the
bits separately; the certificate is on their priced sum. -/
def ComputeSoundPerRunPriced (p : Price) (pRead W ε : ℝ) : Prop :=
  ∀ o : ΩQ × ΩS,
    Pouw.pr (fun x : ΩW => r.accepts .compute (x, o) ∧ (r.work p (x, o) : ℝ) + pRead * r.pouwRead (x, o) < W) ≤ ε

end Run

/-- Run `r`'s compute audit is Pearl-C's audit, with `t` tickets weighted by `w`, of an adversary of Pearl-C's game at
`CM`, `P` and `U`, one adversary for each value of the other audits' coins, whose cost the priced sum covers: the
compute account's work at price `p` plus `pRead` per bit it read from device memory. A cost model that charges its
arithmetic and a price per word it first fetches (Pearl-C's W1 with fetches priced, `w1Priced`) has its cost covered
when its arithmetic is within the compute account's work and each fetch's price within `pRead` per bit the compute
account read for it. -/
structure RealizesPearlCPriced {Q R S : Type} (CM : PearlC.CostModel Q R S) (P : PearlC.Protocol Q R S)
    (U : PearlC.Workload) (w : ℕ → ℕ) {t : ℕ} {ι : Type} {d : Device ι} {α ΩQ ΩS : Type} [Fintype α]
    (r : Run d α (PearlCCoins Q R S U w t) ΩQ ΩS) (p : Price) (pRead : ℝ) where
  /-- the adversary the device is in Pearl-C's game at the other audits' coins -/
  adv : ΩQ × ΩS → PearlC.Adversary CM
  /-- the compute audit is Pearl-C's audit of the adversary's transcript: every ticket's unit is correct -/
  accepts_iff : ∀ x o, r.accepts .compute (x, o) ↔
    AuditAccepts (PearlC.correctSet P U x.1 x.2.1 ((adv o).transcript U x.1 x.2.1)) x.2.2
  /-- the priced sum covers the adversary's online cost -/
  cost_le : ∀ x o, CM.cost (adv o).online x.1 U (adv o).act ((adv o).pre.eval x.1) x.2.1 ≤
    (r.work p (x, o) : ℝ) + pRead * r.pouwRead (x, o)
  /-- the compute account's oracle queries cover the adversary's oracle calls -/
  calls_le : ∀ x o, (adv o).pre.calls x.1 + CM.calls (adv o).online x.1 U (adv o).act ((adv o).pre.eval x.1) x.2.1 ≤
    (r.schedule (x, o)).ops (r.win .compute) (r.acct .compute) .oracle

end Accounting
