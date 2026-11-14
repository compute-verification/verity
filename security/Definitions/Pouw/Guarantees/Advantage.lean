module

public import Definitions.Pouw.Advantage
public import Definitions.Pouw.Guarantees.Game
public import Definitions.Pouw.Guarantees.NCP
public import Definitions.Pouw.Assumptions.PearlC.TTOutW1
public import Definitions.Pouw.PearlC.Device
public import Definitions.Pouw.PearlC.DeviceKernelWref
public import Definitions.Pouw.PearlC.DevicePricesLoop
public import Definitions.Pouw.PearlC.Game
public import Definitions.Pouw.PearlC.Instance

/-!
# PoUW's guarantees in ε form (trusted)

Each is a gap guarantee restated at the adversary's advantage `ε = g/(1 − g)` (`Pouw.Gε`, `Pouw.PearlC.Gε`,
`Pouw.AuditGε`), with the hypotheses of its gap form: the game's end to end (`EndToEnd`), route U's game instance at
ncp-v1 (`GapFromTTNCP_U_v1`) and Pearl-C's design point at 8192³ on the RTX PRO 6000
(`GapSm120v1LoopCast8p72Rev1Cap1000_8192`; not the served point, which is Llama 3.2 1B's linears under the H100 record).
Its proof is the theorem
`Pouw.SecurityProofs.….G : Pouw.Guarantees.….G`.
-/

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

namespace Pouw.Guarantees.PearlC

open Pouw.PearlC Pouw.PearlC.Assumptions

/-- **v1 under rev1 at the cap 1/1,000, 8192³, in ε form**: `GapSm120v1LoopCast8p72Rev1Cap1000_8192` at the
advantage `ε = g/(1 − g) = 1648180439/444424569561` (0.37086%). -/
def EpsSm120v1LoopCast8p72Rev1Cap1000_8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S),
    TTOutPearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000) →
    Pouw.PearlC.Gε (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192)
      ((1648180439 / 444424569561 : ℚ) : ℝ) δPearlC

end Pouw.Guarantees.PearlC
