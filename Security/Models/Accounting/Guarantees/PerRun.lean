import Definitions.Accounting.PerRun
import Definitions.Accounting.Pouw
import Definitions.Pouw.Game.Params
import Models.Pouw.Assumptions.Game

/-!
# The per-run guarantees (trusted)

`Accounting.Guarantees.Joint` and `CovertResidual` take each audit's guarantee in the shape of PoUW's `EndToEnd`, a
sure bound on the account's charge over the audit's own coins, so they bound "accepted, and the worst run below the
certificate". A device that is cheap on a few coins and honest on the rest escapes them. These state the per-run
forms: the probability that every audit accepts and the realized charge *on that run* is below a certificate, or the
other accounts' realized use beyond a residual, is at most the summed error (`JointPerRun`, `CovertResidualPerRun`),
from each audit's per-run guarantee (`Accounting.Run.ComputeSoundPerRun` and the two beside it).

The compute audit's per-run guarantee comes from PoUW by truncation (`PouwEndToEndPerRun`, `ComputeOfPouwPerRun`),
under one more hypothesis on PoUW's cost model, that it truncates (`Accounting.BudgetAbort`); without it the per-run
form does not follow from `Pouw.Assumptions.TT`. The bandwidth audit's guarantee, `Bandwidth.Guarantees.AuditRO`, is
already per-run: its event reads the ingress of the run itself. PoUS's `EndToEndRing` bounds a server's state by its
type, not a charge that varies with the coins, so its per-run form is a matter of how a run's device memory maps to a
server's state, in the capacity embedding.

Each `def G : Prop` is proved by `Accounting.SecurityProofs.G : Accounting.Guarantees.G` (`Proofs.Accounting.PerRunPouw`
and `Proofs.Accounting.PerRunJoint`).
-/

namespace Accounting.Guarantees

open Accounting Pouw

/-- **PoUW's end-to-end guarantee, per run.** Under EndToEnd's hypotheses (TT among them), and when the cost model
truncates (`BudgetAbort`), for every in-domain workload, oracle bound `q ≤ 2^64` and adversary that makes at most `q`
oracle calls (with no bound on its cost), the audit with `t` work-weighted samples accepts while the adversary's online
cost on that run is below `(1 − γ)(1 − ε_s)·W` with probability at most `(1 − ε_s)^t + ε(q, N)`, with
`γ = 1 − (1 − γ₀)/Ω_max`. -/
def PouwEndToEndPerRun : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε → BudgetAbort CM →
    ∀ U : Workload, U.InDomain D → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary CM, CallsWithin A U q → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 →
      pr (fun x : (Q → R) × S × (Fin t → Ticket (P.Wref U.layout) U.layout.N) =>
          AuditAccepts (correctSet P U x.1 x.2.1 (A.transcript U x.1 x.2.1)) x.2.2 ∧
          onlineCost A U x.1 x.2.1 < (1 - reductionGamma γ₀ Ωmax) * (1 - εs) * totalWref P U.layout)
        ≤ (1 - εs) ^ t + ε q U.layout.N

/-- **PoUW's per-run guarantee is the compute audit's per-run guarantee.** Under `PouwEndToEndPerRun`'s hypotheses, for
any run on any device whose compute audit realizes PoUW's game at price `p`, with at least one draw of the game's coins
and the device's oracle queries over the compute window at most `2^64`, the compute audit has its per-run guarantee at
certificate `(1 − γ)(1 − εs)·W` and error `(1 − εs)^t + ε(q, N)`, where `q` is the device's oracle rate times the compute
window's length. -/
def ComputeOfPouwPerRun : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε → BudgetAbort CM →
    ∀ U : Workload, U.InDomain D → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 → Nonempty (PouwCoins P U t) →
    ∀ (ι α ΩQ ΩS : Type) [Fintype α] [Fintype ΩQ] [Fintype ΩS] (d : Device ι)
      (r : Run d α (PouwCoins P U t) ΩQ ΩS) (p : Price),
      d.rate .oracle * (r.win .compute).len ≤ 2 ^ 64 → RealizesPouw CM P U r p →
      r.ComputeSoundPerRun p ((1 - reductionGamma γ₀ Ωmax) * (1 - εs) * totalWref P U.layout)
        ((1 - εs) ^ t + ε (d.rate .oracle * (r.win .compute).len) U.layout.N)

/-- **The joint guarantee, per run.** On any device and any run of a joint audit on it whose compute, bandwidth and
capacity audits have their per-run guarantees at certificates `W`, `Q` and `S` with errors `εW`, `εQ` and `εS`, the
probability over the three audits' coins that every audit accepts and, on that run, the compute account's work is below
`W`, the bandwidth account read fewer than `Q` bits, or the capacity account held fewer than `S` bits on average, is at
most `εW + εQ + εS`. -/
def JointPerRun : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW] [Nonempty ΩQ]
    [Nonempty ΩS] (d : Device ι) (r : Run d α ΩW ΩQ ΩS) (p : Price) (W Q S εW εQ εS : ℝ),
    r.ComputeSoundPerRun p W εW → r.BandwidthSoundPerRun Q εQ → r.CapacitySoundPerRun S εS →
    pr (fun ω => r.Accepted ω ∧
        ((r.work p ω : ℝ) < W ∨ (r.read ω : ℝ) < Q ∨ (r.held ω : ℝ) < S * (r.win .capacity).len))
      ≤ εW + εQ + εS

/-- **The covert residual, per run.** Under `JointPerRun`'s hypotheses, the probability that every audit accepts and,
on that run, the other accounts did more than `F·T − W` of work over the compute window, read more than `BW·T − Q` bits
over the bandwidth window, or held more than `M − S` bits of device memory on average over the capacity window, is at
most `εW + εQ + εS`. -/
def CovertResidualPerRun : Prop :=
  ∀ (ι α ΩW ΩQ ΩS : Type) [Fintype α] [DecidableEq α] [Fintype ΩW] [Fintype ΩQ] [Fintype ΩS] [Nonempty ΩW]
    [Nonempty ΩQ] [Nonempty ΩS] (d : Device ι) (r : Run d α ΩW ΩQ ΩS) (p : Price) (W Q S εW εQ εS : ℝ),
    r.ComputeSoundPerRun p W εW → r.BandwidthSoundPerRun Q εQ → r.CapacitySoundPerRun S εS →
    pr (fun ω => r.Accepted ω ∧
        (((d.priced p * (r.win .compute).len : ℕ) : ℝ) - W < r.othersWork p ω ∨
          ((d.bw * (r.win .bandwidth).len : ℕ) : ℝ) - Q < r.othersRead ω ∨
          ((d.mem : ℝ) - S) * (r.win .capacity).len < r.othersHeld ω))
      ≤ εW + εQ + εS

end Accounting.Guarantees
