import Definitions.Accounting.Pouw
import Definitions.Pouw.Game.Params
import Definitions.Pouw.Assumptions.Game

/-!
# PoUW's guarantee as a joint run's compute guarantee (trusted)

PoUW's end-to-end guarantee (`Pouw.Guarantees.EndToEnd`) embeds into the device model unweakened: in a run that
realizes PoUW's game (`Accounting.RealizesPouw`), the compute audit has its guarantee (`Accounting.Run.ComputeSound`) at
EndToEnd's certificate and error, with the oracle-call budget `q` the device's oracle queries over the compute window.

Proved by `Accounting.SecurityProofs.ComputeOfPouw : Accounting.Guarantees.ComputeOfPouw` (`Proofs.Accounting.Pouw`),
from `Pouw.SecurityProofs.EndToEnd`.
-/

namespace Accounting.Guarantees

open Accounting Pouw

/-- **PoUW's guarantee is the compute guarantee.** Under EndToEnd's hypotheses (TT among them), for any run on any
device whose compute audit realizes PoUW's game at price `p`, with at least one draw of the game's coins and the device's
oracle queries over the compute window at most `2^64`, the compute audit has its guarantee at certificate
`(1 − γ)(1 − εs)·W` (`γ = reductionGamma γ₀ Ωmax`) and error `(1 − εs)^t + ε(q, N)`, where `q` is the device's oracle
rate times the compute window's length. -/
def ComputeOfPouw : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : UnitLayout → Prop) (γ₀ Ωmax : ℝ) (ε : ℕ → ℕ → ℝ),
    γ₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D γ₀ ε →
    ∀ U : Workload, U.InDomain D → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 → Nonempty (PouwCoins P U t) →
    ∀ (ι α ΩQ ΩS : Type) [Fintype α] [Fintype ΩQ] [Fintype ΩS] (d : Device ι)
      (r : Run d α (PouwCoins P U t) ΩQ ΩS) (p : Price),
      d.rate .oracle * (r.win .compute).len ≤ 2 ^ 64 → RealizesPouw CM P U r p →
      r.ComputeSound p ((1 - reductionGamma γ₀ Ωmax) * (1 - εs) * totalWref P U.layout)
        ((1 - εs) ^ t + ε (d.rate .oracle * (r.win .compute).len) U.layout.N)

end Accounting.Guarantees
