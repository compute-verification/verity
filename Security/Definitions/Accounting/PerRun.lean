import Definitions.Accounting.Joint
import Definitions.Pouw.Game.Defs

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

namespace Run

variable {ι : Type} {d : Device ι} {α ΩW ΩQ ΩS : Type} [Fintype α] (r : Run d α ΩW ΩQ ΩS)

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

end Run

end Accounting
