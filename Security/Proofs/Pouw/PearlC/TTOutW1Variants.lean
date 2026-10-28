import Definitions.Pouw.Assumptions.PearlC.DeviceChainCap
import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnly
import Definitions.Pouw.Assumptions.PearlC.TTOutUOnly
import Definitions.Pouw.Assumptions.PearlC.TTOutChainOnlyW1
import Proofs.Pouw.PearlC.TTOutW1

/-!
# TT_OUT's chain-only, chain-cap and U-only forms at W1 are those forms at the cost model `W1.w1Priced`

As `TTOutW1`'s device forms: each W1 form is its abstract form at `CM := W1.w1Priced dv salt` with `CM.Queries`
(`w1Priced_queries`) dropped, so each reader at W1 is its reduction at that cost model.
-/

namespace Pouw.PearlC.TTOutW1

open Pouw.PearlC.Assumptions

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

theorem ttOutPearlCDevChainOnly_of_w1 {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice}
    {sem : PearlCSem Q R S} {ρ : ℚ} (h : TTOutPearlCDevChainOnlyW1 dv salt d sem ρ) :
    TTOutPearlCDevChainOnly (W1.w1Priced dv salt) d sem ρ :=
  ⟨w1Priced_queries dv salt, h⟩

theorem ttOutPearlCDevRev1ChainOnly_of_w1 {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice}
    {sem : PearlCSem Q R S} {ρ : ℚ} (h : TTOutPearlCDevRev1ChainOnlyW1 dv salt d sem ρ) :
    TTOutPearlCDevRev1ChainOnly (W1.w1Priced dv salt) d sem ρ :=
  ⟨w1Priced_queries dv salt, h⟩

theorem ttOutPearlCDevChainCap_of_w1 {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice}
    {sem : PearlCSem Q R S} {ρ : ℚ} (h : TTOutPearlCDevChainCapW1 dv salt d sem ρ) :
    TTOutPearlCDevChainCap (W1.w1Priced dv salt) d sem ρ :=
  ⟨w1Priced_queries dv salt, h⟩

theorem ttOutTilePearlCDevChainCap_of_w1 {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice}
    {sem : PearlCSem Q R S} {ρ : ℚ} (h : TTOutTilePearlCDevChainCapW1 dv salt d sem ρ) :
    TTOutTilePearlCDevChainCap (W1.w1Priced dv salt) d sem ρ :=
  ⟨w1Priced_queries dv salt, h⟩

theorem ttOutUW1_iff (dv : W1.W1Device) (salt : S → List ℕ) (P : Protocol Q R S) (D : UnitLayout → Prop) (γ₀ : ℝ)
    (ε : ℕ → ℕ → ℝ) : TTOutUW1 dv salt P D γ₀ ε ↔ TTOutU (W1.w1Priced dv salt) P D γ₀ ε :=
  ⟨fun h => ⟨w1Priced_queries dv salt, h⟩, fun h => And.right h⟩

theorem ttOutPearlCDevUOnly_of_w1 {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice}
    {sem : PearlCSem Q R S} {ρ : ℚ} (h : TTOutPearlCDevUOnlyW1 dv salt d sem ρ) :
    TTOutPearlCDevUOnly (W1.w1Priced dv salt) d sem ρ :=
  ⟨w1Priced_queries dv salt, h⟩

end Pouw.PearlC.TTOutW1
