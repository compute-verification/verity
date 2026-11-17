module

public import Definitions.Pouw.Advantage
public import Definitions.Pouw.Guarantees.Game
public import Definitions.Pouw.Guarantees.NCP

/-! PoUW's gap guarantees restated at the adversary's advantage `ε = g/(1 − g)`, with the hypotheses of their gap forms. -/

@[expose] public section

namespace Pouw.Guarantees

open Finset

/-- **(d) End to end, in ε form.** Under TT(g₀) and `Ω_max`'s bound, the protocol's advantage is
`ε = Ω_max/(1 − g₀) − 1` (`ε = g/(1 − g)` at the reduction's gap `g = 1 − (1 − g₀)/Ω_max`), and the audit with `t`
work-weighted samples satisfies `Pr[accept ∧ (1 + ε)/(1 − ε_s)·T < W ∧ the run doesn't break SHA-512] ≤
(1 − ε_s)^t + δ(q, N)` for every in-domain
workload, budget, oracle bound, adversary and `ε_s < 1` (`AuditGε`). -/
def EndToEndEps : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (g₀ Ωmax : ℝ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < Ωmax → P.WrefWithin D Ωmax → Assumptions.TT CM P D g₀ δ →
    AuditGε CM P D (Ωmax / (1 - g₀) - 1) δ

end Pouw.Guarantees

namespace Pouw.Guarantees.NCP

open Pouw.NCP Pouw.NCP.RouteU Finset

/-- **`G_ε` from TT_NCP_U at ncp-v1**: `GapFromTTNCP_U_v1` at the advantage `ε = (1 + 8/n₀)/(1 − g₀) − 1`, so
`1 + ε = Ω_max/(1 − g₀)` with `Ω_max = 1 + 8/n₀`, on an instance whose conditions are `Params.InstanceU` at `f = 24`. -/
def EpsFromTTNCP_U_v1 : Prop :=
  ∀ (Q R S : Type) [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (P : Params Q R S) (D : Layout → Prop) (g₀ : ℝ) (n₀ : ℕ) (δ : ℕ → ℕ → ℝ),
    g₀ < 1 → 0 < n₀ → P.InstanceU 24 D g₀ n₀ → Assumptions.TTNCP_U 128 CM P D g₀ δ →
    Gε CM (P.toProtocolU 128) D ((1 + 8 / n₀) / (1 - g₀) - 1) δ

end Pouw.Guarantees.NCP
