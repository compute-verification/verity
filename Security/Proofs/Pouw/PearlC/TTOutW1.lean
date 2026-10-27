import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
import Definitions.Pouw.Assumptions.PearlC.TTOutW1

/-!
# TT_OUT at W1 is TT_OUT at the cost model `W1.w1Priced`

`W1.w1Priced` makes no oracle call, so it meets `CostModel.Queries` (`w1Priced_queries`), and each TT_OUT form at W1
(`Definitions.Pouw.Assumptions.PearlC.TTOutW1`) is the abstract form at `CM := W1.w1Priced dv salt` (`ttOutW1_iff`,
`ttOutTileW1_iff`). The `_of_w1` lemmas give the six named forms the γ readers' reductions take, so each reader at W1
is its reduction at that cost model.
-/

namespace Pouw.PearlC.TTOutW1

open Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- W1 priced makes no oracle call, so its runs are oracle trees with no query. -/
theorem w1Priced_queries (dv : W1.W1Device) (salt : S → List ℕ) :
    (W1.w1Priced dv salt : CostModel Q R S).Queries := by
  intro pr U act σ s
  exact ⟨.done (W1.readout U act (W1.final dv (salt s) pr U act σ).mem), fun _ => ⟨rfl, rfl⟩⟩

variable [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

theorem ttOutW1_iff (dv : W1.W1Device) (salt : S → List ℕ) (P : Protocol Q R S) (D : Layout → Prop) (γ₀ : ℝ)
    (ε : ℕ → ℕ → ℝ) : TTOutW1 dv salt P D γ₀ ε ↔ TTOut (W1.w1Priced dv salt) P D γ₀ ε :=
  ⟨fun h => ⟨w1Priced_queries dv salt, h⟩, fun h => And.right h⟩

theorem ttOutTileW1_iff (dv : W1.W1Device) (salt : S → List ℕ) (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) :
    TTOutTileW1 dv salt P TR D γ₀ ε ↔ TTOutTile (W1.w1Priced dv salt) P TR D γ₀ ε :=
  ⟨fun h => ⟨w1Priced_queries dv salt, h⟩, fun h => And.right h⟩

theorem ttOutPearlCDev_of_w1 {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice} {sem : PearlCSem Q R S}
    {ρ : ℚ} (h : TTOutPearlCDevW1 dv salt d sem ρ) : TTOutPearlCDev (W1.w1Priced dv salt) d sem ρ :=
  ⟨w1Priced_queries dv salt, h⟩

theorem ttOutTilePearlCDev_of_w1 {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice} {sem : PearlCSem Q R S}
    {ρ : ℚ} (h : TTOutTilePearlCDevW1 dv salt d sem ρ) : TTOutTilePearlCDev (W1.w1Priced dv salt) d sem ρ :=
  ⟨w1Priced_queries dv salt, h⟩

theorem ttOutPearlCDevRev1_of_w1 {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice} {sem : PearlCSem Q R S}
    {ρ : ℚ} (h : TTOutPearlCDevRev1W1 dv salt d sem ρ) : TTOutPearlCDevRev1 (W1.w1Priced dv salt) d sem ρ :=
  ⟨w1Priced_queries dv salt, h⟩

theorem ttOutTilePearlCDevRev1_of_w1 {dv : W1.W1Device} {salt : S → List ℕ} {d : PearlCDevice}
    {sem : PearlCSem Q R S} {ρ : ℚ} (h : TTOutTilePearlCDevRev1W1 dv salt d sem ρ) :
    TTOutTilePearlCDevRev1 (W1.w1Priced dv salt) d sem ρ :=
  ⟨w1Priced_queries dv salt, h⟩

theorem ttOutPearlCH100Rev1_of_w1 {sha fetch : ℚ} {salt : S → List ℕ} {sem : PearlCSem Q R S} {ρ : ℚ}
    (h : TTOutPearlCH100Rev1W1 sha fetch salt sem ρ) :
    TTOutPearlCH100Rev1 (W1.w1Priced (W1.W1Device.h100 sha fetch) salt) sem ρ :=
  ⟨w1Priced_queries _ salt, h⟩

theorem ttOutTilePearlCH100Rev1_of_w1 {sha fetch : ℚ} {salt : S → List ℕ} {sem : PearlCSem Q R S} {ρ : ℚ}
    (h : TTOutTilePearlCH100Rev1W1 sha fetch salt sem ρ) :
    TTOutTilePearlCH100Rev1 (W1.w1Priced (W1.W1Device.h100 sha fetch) salt) sem ρ :=
  ⟨w1Priced_queries _ salt, h⟩

end Pouw.PearlC.TTOutW1
