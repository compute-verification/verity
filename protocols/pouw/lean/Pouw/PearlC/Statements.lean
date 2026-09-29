import Pouw.PearlC.Assumptions

/-!
# Pearl-C statements (trusted; proved in `PouwProofs.PearlC`, none pinned yet)
-/

namespace Pouw.PearlC

open Finset Assumptions

/-- **TT_OUT is satisfiable** together with an honest reference: some cost model and some protocol with Pearl-C's
headline accounting run the honest reference at `W_ref`, have an in-domain workload with positive credited work, and
satisfy `TT_OUT(1/400)` with error `0`. -/
def TTOutWitness : Prop :=
  ∃ (CM : CostModel Unit Unit Unit) (P : Protocol Unit Unit Unit),
    PearlCAccounting P Params.headline ∧ AdmitsRef CM P (pearlCDomain Params.headline) (fun _ => 0) ∧
    TTOut CM P (pearlCDomain Params.headline) (1 / 400) (fun _ _ => 0) ∧
    ∃ U : Workload, U.InDomain P (pearlCDomain Params.headline) ∧
      0 < ∑ u ∈ range U.layout.N, P.credit U.layout u

/-- **The reduction:** if every in-domain unit's reference cost is at most `Ω` times its credited work, then
`TT_OUT(γ₀)` gives the γ-gap property at `γ = 1 − (1 − γ₀)/Ω` with the same error. -/
def GammaFromTTOut : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (D : Layout → Prop) (γ₀ Ω : ℝ) (ε : ℕ → ℕ → ℝ),
    0 ≤ γ₀ → γ₀ < 1 → 0 < Ω →
    (∀ U : Workload, U.InDomain P D → ∀ u < U.layout.N, P.Wref U.layout u ≤ Ω * P.credit U.layout u) →
    TTOut CM P D γ₀ ε → Gγ CM P D (1 - (1 - γ₀) / Ω) ε

/-- The headline shape `8192³`. -/
def sh8192 : Shape := ⟨8192, 8192, 8192⟩

/-- The headline shape `16384³`. -/
def sh16384 : Shape := ⟨16384, 16384, 16384⟩

/-- **Pearl-C's γ at 8192³:** with the headline accounting and no promotion, `TT_OUT(1/400)` on the 8192³ domain gives
`γ = 929/212000` (0.438%). -/
def PearlCGamma8192 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (ε : ℕ → ℕ → ℝ),
    PearlCAccounting P Params.headline → TTOut CM P (pearlCDomainAt Params.headline sh8192) (1 / 400) ε →
      Gγ CM P (pearlCDomainAt Params.headline sh8192) ((929 / 212000 : ℚ) : ℝ) ε

/-- **Pearl-C's γ at 16384³:** the same at `16384³` gives `γ = 1441/416800` (0.346%). -/
def PearlCGamma16384 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (ε : ℕ → ℕ → ℝ),
    PearlCAccounting P Params.headline → TTOut CM P (pearlCDomainAt Params.headline sh16384) (1 / 400) ε →
      Gγ CM P (pearlCDomainAt Params.headline sh16384) ((1441 / 416800 : ℚ) : ℝ) ε

/-- **Why forming must be credited at 8192³:** crediting only the chain and the clean-up leaves `γ = 1861/106000`
(1.76%), above 1%. -/
def ChainOnlyGamma8192 : Prop :=
  gammaOf (1 / 400) (wrefOf Params.headline sh8192 / creditChainOnly Params.headline sh8192) = 1861 / 106000

end Pouw.PearlC
