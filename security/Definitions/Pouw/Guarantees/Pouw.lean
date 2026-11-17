module

public import Definitions.Pouw.Pouw
public import Definitions.Pouw.Game.Params
public import Definitions.Pouw.Assumptions.Game
public import Mathlib.Data.Fintype.Sigma

/-! PoUW's end-to-end guarantee, embedded unweakened as a joint run's compute guarantee. -/

@[expose] public section

namespace ResourceAccounting.Guarantees

open ResourceAccounting Pouw

/-- **PoUW's guarantee is the compute guarantee.** Under EndToEnd's hypotheses (TT among them), for any run on any
device whose compute audit realizes PoUW's game at price `p`, with at least one sample of the game's coins and the device's
oracle queries over the compute window at most `2^64`, the compute audit has its guarantee at certificate
`(1 − g)(1 − εs)·W` (`g = reductionGap g₀ Ωmax`) and error `(1 − εs)^t + δ(q, N)` unless the device's run breaks
SHA-512 (`ResourceAccounting.Run.ComputeSoundUnless`, the break `Pouw.Adversary.Breaks` of the game's adversary), where `q` is
the device's oracle rate times the compute window's length. -/
def ComputeOfPouw : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (g₀ Ωmax : ℝ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < Ωmax →
    (∀ U : Workload, U.InDomain D → ∀ u < U.layout.N,
      (P.Wref U.layout u : ℝ) ≤ Ωmax * ((U.layout.shape u).Wmm : ℝ)) →
    Assumptions.TT CM P D g₀ δ →
    ∀ U : Workload, U.InDomain D → ∀ (t : ℕ) (εs : ℝ), εs ≤ 1 → Nonempty (PouwCoins P U t) →
    ∀ (ι α ΩQ ΩS : Type) [Fintype α] [Fintype ΩQ] [Fintype ΩS] (d : Device ι)
      (r : Run d α (PouwCoins P U t) ΩQ ΩS) (p : Price),
      d.rate .oracle * (r.win .compute).len ≤ 2 ^ 64 → ∀ real : RealizesPouw CM P U r p,
      r.ComputeSoundUnless p ((1 - reductionGap g₀ Ωmax) * (1 - εs) * totalWref P U.layout)
        ((1 - εs) ^ t + δ (d.rate .oracle * (r.win .compute).len) U.layout.N)
        (fun x o => (real.adv o).Breaks U x.1 x.2.1)

end ResourceAccounting.Guarantees
