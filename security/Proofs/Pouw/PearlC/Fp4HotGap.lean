module

public import Proofs.Pouw.PearlC.TileGap
public import Definitions.Pouw.PearlC.DeviceFp4Hot
public import Definitions.Pouw.Assumptions.PearlC.TTOut
public import Definitions.Pouw.PearlC.DeviceFp4Issue
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.TileGap

/-! Pearl-C4 v2's price twins, at any FP32 price and scale path, held as the rest of the FP4 delta is. -/

@[expose] public section

namespace Pouw.PearlC

open Pouw.PearlC.Assumptions

theorem pearlCGapFp4HotAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (D : Layout → Prop) (d : Fp4Device) (a K : ℚ) (s : Shape) (hK : 0 < K)
    (hW0 : 0 < wrefFp4Hot d a s) (hacc : Fp4HotUnitAccounting P D d a K s) (hTT : TTOut CM P D (1 / 400) δPearlC) :
    GGap CM P D ((1 - (1 - 1 / 400) / (wrefFp4Hot d a s / K) : ℚ) : ℝ) δPearlC := by
  set ω := wrefFp4Hot d a s / K with hωdef
  have hω0 : 0 < ω := div_pos hW0 hK
  have hid : wrefFp4Hot d a s = ω * K := by rw [hωdef, div_mul_cancel₀ _ hK.ne']
  have hW : ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N, ∀ act : Codes,
      P.actOK (U.layout.shape u) act → P.inertBoundOK H s' U u act → P.Wref U.layout u ≤ (ω : ℝ) * P.credit H s' U u act := by
    intro U hU H s' u hu act ha hcap
    obtain ⟨hw, hcr⟩ := hacc U hU H s' u hu act ha hcap
    rw [hw, hid]
    push_cast
    exact mul_le_mul_of_nonneg_left hcr (by exact_mod_cast hω0.le)
  have key := gapFromTTOut CM P D (1 / 400) (ω : ℝ) δPearlC (by exact_mod_cast hω0) hW hTT
  have e : ((1 - (1 - 1 / 400) / ω : ℚ) : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by push_cast; ring
  rw [e]
  exact key

theorem pearlCSampledFp4HotAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (d : Fp4Device) (a K : ℚ) (s : Shape) (hK : 0 < K)
    (hW0 : 0 < wrefFp4Hot d a s) (hacc : Fp4HotTileAccounting P TR D d a K s)
    (hTT : TTOutTile CM P TR D (1 / 400) δPearlC) :
    GGapSampled CM P TR D ((1 - (1 - 1 / 400) / (wrefFp4Hot d a s / K) : ℚ) : ℝ) δPearlC := by
  set ω := wrefFp4Hot d a s / K with hωdef
  have hω0 : 0 < ω := div_pos hW0 hK
  have hid : wrefFp4Hot d a s = ω * K := by rw [hωdef, div_mul_cancel₀ _ hK.ne']
  have hW : ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s' : S), ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes,
      TR.inertBoundOK H s' U g act → TR.Wcred U.layout g act ≤ (ω : ℝ) * TR.credit H s' U g act := by
    intro U hU H s' g hg act hcap
    obtain ⟨t, _, hw, _, hcr⟩ := hacc U hU g hg act
    rw [hw, hid]
    have e2 : ((ω * K * t : ℚ) : ℝ) = (ω : ℝ) * ((K * t : ℚ) : ℝ) := by push_cast; ring
    rw [e2]
    exact mul_le_mul_of_nonneg_left (hcr H s' hcap) (by exact_mod_cast hω0.le)
  have hWc : ∀ U : Workload, U.InDomain P D → ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes,
      TR.Wcred U.layout g act ≤ TR.Wref U.layout g := by
    intro U hU g hg act
    obtain ⟨t, _, _, hle, _⟩ := hacc U hU g hg act
    exact hle
  have key := gapSampled CM P TR D (1 / 400) (ω : ℝ) δPearlC (by norm_num) (by exact_mod_cast hω0) hW hWc hTT
  have e : ((1 - (1 - 1 / 400) / ω : ℚ) : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by push_cast; ring
  rw [e]
  exact key

section Twins

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)

theorem pearlCGapFp4Sm120HotAt_8192
    (P : Protocol Q R S) (D : Layout → Prop) (d : Fp4Device) (fadd : ℚ) (p : Fp4ScalePath)
    (hf : 0 ≤ fadd) (hp : d.prices = Fp4Prices.sm120At fadd p)
    (hacc : Fp4HotUnitAccounting P D d (2 * fadd) ((1 - 1 / 400) * creditFp4Hot d (2 * fadd) sh8192) sh8192)
    (hTT : TTOut CM P D (1 / 400) δPearlC) :
    GGap CM P D (gapFp4Hot d (2 * fadd) (1 / 400) sh8192 : ℝ) δPearlC := by
  have key := pearlCGapFp4HotAt CM P D d _ _ _ (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh8192]
      push_cast
      nlinarith) (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [wrefFp4Hot, creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh8192]
      push_cast
      nlinarith) hacc hTT
  simpa only [gapFp4Hot] using key

theorem pearlCGapFp4Sm120HotAt_16384
    (P : Protocol Q R S) (D : Layout → Prop) (d : Fp4Device) (fadd : ℚ) (p : Fp4ScalePath)
    (hf : 0 ≤ fadd) (hp : d.prices = Fp4Prices.sm120At fadd p)
    (hacc : Fp4HotUnitAccounting P D d (2 * fadd) ((1 - 1 / 400) * creditFp4Hot d (2 * fadd) sh16384) sh16384)
    (hTT : TTOut CM P D (1 / 400) δPearlC) :
    GGap CM P D (gapFp4Hot d (2 * fadd) (1 / 400) sh16384 : ℝ) δPearlC := by
  have key := pearlCGapFp4HotAt CM P D d _ _ _ (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh16384]
      push_cast
      nlinarith) (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [wrefFp4Hot, creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh16384]
      push_cast
      nlinarith) hacc hTT
  simpa only [gapFp4Hot] using key

theorem pearlCSampledFp4Sm120HotAt_8192
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (d : Fp4Device) (fadd : ℚ) (p : Fp4ScalePath)
    (hf : 0 ≤ fadd) (hp : d.prices = Fp4Prices.sm120At fadd p)
    (hacc : Fp4HotTileAccounting P TR D d (2 * fadd) ((1 - 1 / 400) * creditFp4Hot d (2 * fadd) sh8192) sh8192)
    (hTT : TTOutTile CM P TR D (1 / 400) δPearlC) :
    GGapSampled CM P TR D (gapFp4Hot d (2 * fadd) (1 / 400) sh8192 : ℝ) δPearlC := by
  have key := pearlCSampledFp4HotAt CM P TR D d _ _ _ (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh8192]
      push_cast
      nlinarith) (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [wrefFp4Hot, creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh8192]
      push_cast
      nlinarith) hacc hTT
  simpa only [gapFp4Hot] using key

theorem pearlCSampledFp4Sm120HotAt_16384
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (d : Fp4Device) (fadd : ℚ) (p : Fp4ScalePath)
    (hf : 0 ≤ fadd) (hp : d.prices = Fp4Prices.sm120At fadd p)
    (hacc : Fp4HotTileAccounting P TR D d (2 * fadd) ((1 - 1 / 400) * creditFp4Hot d (2 * fadd) sh16384) sh16384)
    (hTT : TTOutTile CM P TR D (1 / 400) δPearlC) :
    GGapSampled CM P TR D (gapFp4Hot d (2 * fadd) (1 / 400) sh16384 : ℝ) δPearlC := by
  have key := pearlCSampledFp4HotAt CM P TR D d _ _ _ (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh16384]
      push_cast
      nlinarith) (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [wrefFp4Hot, creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh16384]
      push_cast
      nlinarith) hacc hTT
  simpa only [gapFp4Hot] using key

theorem pearlCGapFp4Sm120HotChainOnlyAt_8192
    (P : Protocol Q R S) (D : Layout → Prop) (d : Fp4Device) (fadd : ℚ) (p : Fp4ScalePath)
    (hf : 0 ≤ fadd) (hp : d.prices = Fp4Prices.sm120At fadd p)
    (hK : 0 < creditFp4HotChainOnly d (2 * fadd) sh8192 - 1 / 400 * creditFp4Hot d (2 * fadd) sh8192)
    (hacc : Fp4HotUnitAccounting P D d (2 * fadd)
      (creditFp4HotChainOnly d (2 * fadd) sh8192 - 1 / 400 * creditFp4Hot d (2 * fadd) sh8192) sh8192)
    (hTT : TTOut CM P D (1 / 400) δPearlC) :
    GGap CM P D (gapFp4HotChainOnly d (2 * fadd) (1 / 400) sh8192 : ℝ) δPearlC := by
  have key := pearlCGapFp4HotAt CM P D d _ _ _ hK (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [wrefFp4Hot, creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh8192]
      push_cast
      nlinarith) hacc hTT
  simpa only [gapFp4HotChainOnly] using key

theorem pearlCGapFp4Sm120HotChainOnlyAt_16384
    (P : Protocol Q R S) (D : Layout → Prop) (d : Fp4Device) (fadd : ℚ) (p : Fp4ScalePath)
    (hf : 0 ≤ fadd) (hp : d.prices = Fp4Prices.sm120At fadd p)
    (hK : 0 < creditFp4HotChainOnly d (2 * fadd) sh16384 - 1 / 400 * creditFp4Hot d (2 * fadd) sh16384)
    (hacc : Fp4HotUnitAccounting P D d (2 * fadd)
      (creditFp4HotChainOnly d (2 * fadd) sh16384 - 1 / 400 * creditFp4Hot d (2 * fadd) sh16384) sh16384)
    (hTT : TTOut CM P D (1 / 400) δPearlC) :
    GGap CM P D (gapFp4HotChainOnly d (2 * fadd) (1 / 400) sh16384 : ℝ) δPearlC := by
  have key := pearlCGapFp4HotAt CM P D d _ _ _ hK (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [wrefFp4Hot, creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh16384]
      push_cast
      nlinarith) hacc hTT
  simpa only [gapFp4HotChainOnly] using key

theorem pearlCSampledFp4Sm120HotChainOnlyAt_8192
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (d : Fp4Device) (fadd : ℚ) (p : Fp4ScalePath)
    (hf : 0 ≤ fadd) (hp : d.prices = Fp4Prices.sm120At fadd p)
    (hK : 0 < creditFp4HotChainOnly d (2 * fadd) sh8192 - 1 / 400 * creditFp4Hot d (2 * fadd) sh8192)
    (hacc : Fp4HotTileAccounting P TR D d (2 * fadd)
      (creditFp4HotChainOnly d (2 * fadd) sh8192 - 1 / 400 * creditFp4Hot d (2 * fadd) sh8192) sh8192)
    (hTT : TTOutTile CM P TR D (1 / 400) δPearlC) :
    GGapSampled CM P TR D (gapFp4HotChainOnly d (2 * fadd) (1 / 400) sh8192 : ℝ) δPearlC := by
  have key := pearlCSampledFp4HotAt CM P TR D d _ _ _ hK (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [wrefFp4Hot, creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh8192]
      push_cast
      nlinarith) hacc hTT
  simpa only [gapFp4HotChainOnly] using key

theorem pearlCSampledFp4Sm120HotChainOnlyAt_16384
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (d : Fp4Device) (fadd : ℚ) (p : Fp4ScalePath)
    (hf : 0 ≤ fadd) (hp : d.prices = Fp4Prices.sm120At fadd p)
    (hK : 0 < creditFp4HotChainOnly d (2 * fadd) sh16384 - 1 / 400 * creditFp4Hot d (2 * fadd) sh16384)
    (hacc : Fp4HotTileAccounting P TR D d (2 * fadd)
      (creditFp4HotChainOnly d (2 * fadd) sh16384 - 1 / 400 * creditFp4Hot d (2 * fadd) sh16384) sh16384)
    (hTT : TTOutTile CM P TR D (1 / 400) δPearlC) :
    GGapSampled CM P TR D (gapFp4HotChainOnly d (2 * fadd) (1 / 400) sh16384 : ℝ) δPearlC := by
  have key := pearlCSampledFp4HotAt CM P TR D d _ _ _ hK (by
      have h1 := p.fixed_nonneg
      have h2 := mul_nonneg p.fp32Ops_nonneg hf
      rw [wrefFp4Hot, creditFp4Hot, creditFp4, hp]
      simp only [Fp4Prices.sm120At, Params.pi, sh16384]
      push_cast
      nlinarith) hacc hTT
  simpa only [gapFp4HotChainOnly] using key

end Twins

theorem gapFp4Hot_sm120Issue_rcpApprox_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.rcpApprox) :
    gapFp4Hot d (2 * 8) (1 / 400) sh8192 = 25671209 / 3630560000 := by
  rw [gapFp4Hot, wrefFp4Hot, creditFp4Hot, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]

theorem gapFp4Hot_sm120Issue_rcpApprox_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.rcpApprox) :
    gapFp4Hot d (2 * 8) (1 / 400) sh16384 = 271674457 / 44838880000 := by
  rw [gapFp4Hot, wrefFp4Hot, creditFp4Hot, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]

theorem gapFp4Hot_sm120Issue_lut256_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256) :
    gapFp4Hot d (2 * 8) (1 / 400) sh8192 = 19503599 / 2757920000 := by
  rw [gapFp4Hot, wrefFp4Hot, creditFp4Hot, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]

theorem gapFp4Hot_sm120Issue_lut256_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256) :
    gapFp4Hot d (2 * 8) (1 / 400) sh16384 = 310423 / 51232000 := by
  rw [gapFp4Hot, wrefFp4Hot, creditFp4Hot, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]

theorem gapFp4Hot_sm120Loop_rcpApprox_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.rcpApprox) :
    gapFp4Hot d (2 * (1047 / 125)) (1 / 400) sh8192 = 549538679 / 76666880000 := by
  rw [gapFp4Hot, wrefFp4Hot, creditFp4Hot, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]

theorem gapFp4Hot_sm120Loop_rcpApprox_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.rcpApprox) :
    gapFp4Hot d (2 * (1047 / 125)) (1 / 400) sh16384 = 1174078873 / 192194560000 := by
  rw [gapFp4Hot, wrefFp4Hot, creditFp4Hot, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]

theorem gapFp4Hot_sm120Loop_lut256_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256) :
    gapFp4Hot d (2 * (1047 / 125)) (1 / 400) sh8192 = 9888398749 / 1379343520000 := by
  rw [gapFp4Hot, wrefFp4Hot, creditFp4Hot, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]

theorem gapFp4Hot_sm120Loop_lut256_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256) :
    gapFp4Hot d (2 * (1047 / 125)) (1 / 400) sh16384 = 5477935583 / 896687840000 := by
  rw [gapFp4Hot, wrefFp4Hot, creditFp4Hot, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]

theorem gapFp4HotChainOnly_sm120Issue_rcpApprox_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.rcpApprox) :
    0 < creditFp4HotChainOnly d (2 * 8) sh8192 - 1 / 400 * creditFp4Hot d (2 * 8) sh8192 ∧
      gapFp4HotChainOnly d (2 * 8) (1 / 400) sh8192 = 71274809 / 3630560000 := by
  constructor
  · rw [creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]
  · rw [gapFp4HotChainOnly, wrefFp4Hot, creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]

theorem gapFp4HotChainOnly_sm120Issue_rcpApprox_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.rcpApprox) :
    0 < creditFp4HotChainOnly d (2 * 8) sh16384 - 1 / 400 * creditFp4Hot d (2 * 8) sh16384 ∧
      gapFp4HotChainOnly d (2 * 8) (1 / 400) sh16384 = 560497257 / 44838880000 := by
  constructor
  · rw [creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]
  · rw [gapFp4HotChainOnly, wrefFp4Hot, creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]

theorem gapFp4HotChainOnly_sm120Issue_lut256_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256) :
    0 < creditFp4HotChainOnly d (2 * 8) sh8192 - 1 / 400 * creditFp4Hot d (2 * 8) sh8192 ∧
      gapFp4HotChainOnly d (2 * 8) (1 / 400) sh8192 = 52859999 / 2757920000 := by
  constructor
  · rw [creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]
  · rw [gapFp4HotChainOnly, wrefFp4Hot, creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]

theorem gapFp4HotChainOnly_sm120Issue_lut256_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256) :
    0 < creditFp4HotChainOnly d (2 * 8) sh16384 - 1 / 400 * creditFp4Hot d (2 * 8) sh16384 ∧
      gapFp4HotChainOnly d (2 * 8) (1 / 400) sh16384 = 628103 / 51232000 := by
  constructor
  · rw [creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]
  · rw [gapFp4HotChainOnly, wrefFp4Hot, creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]

theorem gapFp4HotChainOnly_sm120Loop_rcpApprox_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.rcpApprox) :
    0 < creditFp4HotChainOnly d (2 * (1047 / 125)) sh8192 - 1 / 400 * creditFp4Hot d (2 * (1047 / 125)) sh8192 ∧
      gapFp4HotChainOnly d (2 * (1047 / 125)) (1 / 400) sh8192 = 4559347637 / 230000640000 := by
  constructor
  · rw [creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]
  · rw [gapFp4HotChainOnly, wrefFp4Hot, creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]

theorem gapFp4HotChainOnly_sm120Loop_rcpApprox_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.rcpApprox) :
    0 < creditFp4HotChainOnly d (2 * (1047 / 125)) sh16384 - 1 / 400 * creditFp4Hot d (2 * (1047 / 125)) sh16384 ∧
      gapFp4HotChainOnly d (2 * (1047 / 125)) (1 / 400) sh16384 = 2421535273 / 192194560000 := by
  constructor
  · rw [creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]
  · rw [gapFp4HotChainOnly, wrefFp4Hot, creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]

theorem gapFp4HotChainOnly_sm120Loop_lut256_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256) :
    0 < creditFp4HotChainOnly d (2 * (1047 / 125)) sh8192 - 1 / 400 * creditFp4Hot d (2 * (1047 / 125)) sh8192 ∧
      gapFp4HotChainOnly d (2 * (1047 / 125)) (1 / 400) sh8192 = 26694119149 / 1379343520000 := by
  constructor
  · rw [creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]
  · rw [gapFp4HotChainOnly, wrefFp4Hot, creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]

theorem gapFp4HotChainOnly_sm120Loop_lut256_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256) :
    0 < creditFp4HotChainOnly d (2 * (1047 / 125)) sh16384 - 1 / 400 * creditFp4Hot d (2 * (1047 / 125)) sh16384 ∧
      gapFp4HotChainOnly d (2 * (1047 / 125)) (1 / 400) sh16384 = 11079842383 / 896687840000 := by
  constructor
  · rw [creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]
  · rw [gapFp4HotChainOnly, wrefFp4Hot, creditFp4HotChainOnly, creditFp4Hot, creditFp4, hp]
    norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]

end Pouw.PearlC
