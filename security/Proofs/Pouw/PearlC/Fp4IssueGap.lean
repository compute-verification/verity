module

public import Proofs.Pouw.PearlC.DeviceFp4Gap
public import Definitions.Pouw.PearlC.DeviceFp4Issue
public import Proofs.Pouw.Lemmas.PearlCRecords
public import Proofs.Pouw.Lemmas.PearlC
public import Proofs.Pouw.PearlC.Records

/-! Pearl-C4 at the issue-bound FP32 price, staged; each theorem holds at any FP4 record with these prices, for any
`sem`. -/

@[expose] public section

namespace Pouw.PearlC

open Pouw.PearlC.Assumptions

theorem pearlCGapFp4Sm120Issue_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : Fp4Device) (sem : Fp4Sem Q R S) (hp : d.prices = Fp4Prices.sm120Issue)
    (hTT : TTOutFp4 CM d sem (1 / 400)) :
    GGap CM (pearlCProtocolFp4 d sem (1 / 400)) (pearlCDomainFp4At sem sh8192)
      ((162371257 / 22950880000 : ℚ) : ℝ) δPearlC :=
  pearlCGapFp4At CM d sem _ _ (57377200 / 57114057) _ (by norm_num)
    (by rw [wrefFp4, creditFp4, hp]
        norm_num [Fp4Prices.sm120Issue, Params.pi, sh8192]) (by norm_num) hTT

theorem pearlCGapFp4Sm120Issue_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : Fp4Device) (sem : Fp4Sem Q R S) (hp : d.prices = Fp4Prices.sm120Issue)
    (hTT : TTOutFp4 CM d sem (1 / 400)) :
    GGap CM (pearlCProtocolFp4 d sem (1 / 400)) (pearlCDomainFp4At sem sh16384)
      ((814384171 / 134388640000 : ℚ) : ℝ) δPearlC :=
  pearlCGapFp4At CM d sem _ _ (335971600 / 334772571) _ (by norm_num)
    (by rw [wrefFp4, creditFp4, hp]
        norm_num [Fp4Prices.sm120Issue, Params.pi, sh16384]) (by norm_num) hTT

theorem pearlCSampledFp4Sm120Issue_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : Fp4Device) (sem : Fp4Sem Q R S) (hp : d.prices = Fp4Prices.sm120Issue)
    (hTT : TTOutTileFp4 CM d sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolFp4 d sem (1 / 400)) (pearlCTilesFp4 d sem (1 / 400)) (pearlCDomainFp4At sem sh8192)
      ((162371257 / 22950880000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledFp4At CM d sem _ _ (57377200 / 57114057) _ (by norm_num)
    (by rw [wrefFp4, creditFp4, hp]
        norm_num [Fp4Prices.sm120Issue, Params.pi, sh8192])
    (by rw [wrefFp4, creditFp4, hp]
        norm_num [Fp4Prices.sm120Issue, Params.pi, sh8192]) (by norm_num) hTT

theorem pearlCSampledFp4Sm120Issue_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : Fp4Device) (sem : Fp4Sem Q R S) (hp : d.prices = Fp4Prices.sm120Issue)
    (hTT : TTOutTileFp4 CM d sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolFp4 d sem (1 / 400)) (pearlCTilesFp4 d sem (1 / 400)) (pearlCDomainFp4At sem sh16384)
      ((814384171 / 134388640000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledFp4At CM d sem _ _ (335971600 / 334772571) _ (by norm_num)
    (by rw [wrefFp4, creditFp4, hp]
        norm_num [Fp4Prices.sm120Issue, Params.pi, sh16384])
    (by rw [wrefFp4, creditFp4, hp]
        norm_num [Fp4Prices.sm120Issue, Params.pi, sh16384]) (by norm_num) hTT

/-! The twins below hold at any record priced `Fp4Prices.sm120At fadd p`, so repricing the scale step or the FP32 price
instantiates `p` or `fadd` and re-proves nothing. -/

theorem Fp4Prices.sm120Issue_eq : Fp4Prices.sm120Issue = Fp4Prices.sm120At 8 Fp4ScalePath.rcpApprox := by
  simp only [Fp4Prices.sm120Issue, Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Fp4Prices.mk.injEq]
  norm_num

theorem pearlCGapFp4Sm120At_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : Fp4Device) (sem : Fp4Sem Q R S) (fadd : ℚ) (p : Fp4ScalePath) (hf : 0 ≤ fadd)
    (hp : d.prices = Fp4Prices.sm120At fadd p) (hTT : TTOutFp4 CM d sem (1 / 400)) :
    GGap CM (pearlCProtocolFp4 d sem (1 / 400)) (pearlCDomainFp4At sem sh8192) (gapFp4 d (1 / 400) sh8192 : ℝ)
      δPearlC := by
  have hc : 0 < creditFp4 d sh8192 := by
    have h1 := p.fixed_nonneg
    have h2 := mul_nonneg p.fp32Ops_nonneg hf
    rw [creditFp4, hp]
    simp only [Fp4Prices.sm120At, Params.pi, sh8192]
    push_cast
    nlinarith
  have hw : 0 < wrefFp4 d sh8192 := by
    have hq : 0 ≤ d.prices.qa * (sh8192.m : ℚ) * sh8192.k := by
      rw [hp]; simp only [Fp4Prices.sm120At, sh8192]; push_cast; nlinarith
    rw [wrefFp4]; linarith
  have hden : 0 < (1 - 1 / 400 : ℚ) * creditFp4 d sh8192 := by positivity
  exact pearlCGapFp4At CM d sem _ _ (wrefFp4 d sh8192 / ((1 - 1 / 400) * creditFp4 d sh8192)) _
    (div_pos hw hden) (div_mul_cancel₀ _ hden.ne').symm rfl hTT

theorem pearlCGapFp4Sm120At_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : Fp4Device) (sem : Fp4Sem Q R S) (fadd : ℚ) (p : Fp4ScalePath) (hf : 0 ≤ fadd)
    (hp : d.prices = Fp4Prices.sm120At fadd p) (hTT : TTOutFp4 CM d sem (1 / 400)) :
    GGap CM (pearlCProtocolFp4 d sem (1 / 400)) (pearlCDomainFp4At sem sh16384) (gapFp4 d (1 / 400) sh16384 : ℝ)
      δPearlC := by
  have hc : 0 < creditFp4 d sh16384 := by
    have h1 := p.fixed_nonneg
    have h2 := mul_nonneg p.fp32Ops_nonneg hf
    rw [creditFp4, hp]
    simp only [Fp4Prices.sm120At, Params.pi, sh16384]
    push_cast
    nlinarith
  have hw : 0 < wrefFp4 d sh16384 := by
    have hq : 0 ≤ d.prices.qa * (sh16384.m : ℚ) * sh16384.k := by
      rw [hp]; simp only [Fp4Prices.sm120At, sh16384]; push_cast; nlinarith
    rw [wrefFp4]; linarith
  have hden : 0 < (1 - 1 / 400 : ℚ) * creditFp4 d sh16384 := by positivity
  exact pearlCGapFp4At CM d sem _ _ (wrefFp4 d sh16384 / ((1 - 1 / 400) * creditFp4 d sh16384)) _
    (div_pos hw hden) (div_mul_cancel₀ _ hden.ne').symm rfl hTT

theorem pearlCSampledFp4Sm120At_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : Fp4Device) (sem : Fp4Sem Q R S) (fadd : ℚ) (p : Fp4ScalePath) (hf : 0 ≤ fadd)
    (hp : d.prices = Fp4Prices.sm120At fadd p) (hTT : TTOutTileFp4 CM d sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolFp4 d sem (1 / 400)) (pearlCTilesFp4 d sem (1 / 400)) (pearlCDomainFp4At sem sh8192)
      (gapFp4 d (1 / 400) sh8192 : ℝ) δPearlC := by
  have hc : 0 < creditFp4 d sh8192 := by
    have h1 := p.fixed_nonneg
    have h2 := mul_nonneg p.fp32Ops_nonneg hf
    rw [creditFp4, hp]
    simp only [Fp4Prices.sm120At, Params.pi, sh8192]
    push_cast
    nlinarith
  have hw : 0 < wrefFp4 d sh8192 := by
    have hq : 0 ≤ d.prices.qa * (sh8192.m : ℚ) * sh8192.k := by
      rw [hp]; simp only [Fp4Prices.sm120At, sh8192]; push_cast; nlinarith
    rw [wrefFp4]; linarith
  have hden : 0 < (1 - 1 / 400 : ℚ) * creditFp4 d sh8192 := by positivity
  exact pearlCSampledFp4At CM d sem _ _ (wrefFp4 d sh8192 / ((1 - 1 / 400) * creditFp4 d sh8192)) _
    (div_pos hw hden) hw.le (div_mul_cancel₀ _ hden.ne').symm rfl hTT

theorem pearlCSampledFp4Sm120At_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : Fp4Device) (sem : Fp4Sem Q R S) (fadd : ℚ) (p : Fp4ScalePath) (hf : 0 ≤ fadd)
    (hp : d.prices = Fp4Prices.sm120At fadd p) (hTT : TTOutTileFp4 CM d sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolFp4 d sem (1 / 400)) (pearlCTilesFp4 d sem (1 / 400)) (pearlCDomainFp4At sem sh16384)
      (gapFp4 d (1 / 400) sh16384 : ℝ) δPearlC := by
  have hc : 0 < creditFp4 d sh16384 := by
    have h1 := p.fixed_nonneg
    have h2 := mul_nonneg p.fp32Ops_nonneg hf
    rw [creditFp4, hp]
    simp only [Fp4Prices.sm120At, Params.pi, sh16384]
    push_cast
    nlinarith
  have hw : 0 < wrefFp4 d sh16384 := by
    have hq : 0 ≤ d.prices.qa * (sh16384.m : ℚ) * sh16384.k := by
      rw [hp]; simp only [Fp4Prices.sm120At, sh16384]; push_cast; nlinarith
    rw [wrefFp4]; linarith
  have hden : 0 < (1 - 1 / 400 : ℚ) * creditFp4 d sh16384 := by positivity
  exact pearlCSampledFp4At CM d sem _ _ (wrefFp4 d sh16384 / ((1 - 1 / 400) * creditFp4 d sh16384)) _
    (div_pos hw hden) hw.le (div_mul_cancel₀ _ hden.ne').symm rfl hTT

theorem gapFp4_sm120Issue_rcpApprox_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.rcpApprox) :
    gapFp4 d (1 / 400) sh8192 = 162371257 / 22950880000 := by
  rw [gapFp4, wrefFp4, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]

theorem gapFp4_sm120Issue_rcpApprox_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.rcpApprox) :
    gapFp4 d (1 / 400) sh16384 = 814384171 / 134388640000 := by
  rw [gapFp4, wrefFp4, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]

theorem gapFp4_sm120Loop_rcpApprox_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.rcpApprox) :
    gapFp4 d (1 / 400) sh8192 = 1646385229 / 229553920000 := by
  rw [gapFp4, wrefFp4, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh8192]

theorem gapFp4_sm120Loop_rcpApprox_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.rcpApprox) :
    gapFp4 d (1 / 400) sh16384 = 8211859687 / 1344021760000 := by
  rw [gapFp4, wrefFp4, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.rcpApprox, Params.pi, sh16384]

theorem gapFp4_sm120Issue_lut256_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256) :
    gapFp4 d (1 / 400) sh8192 = 6492677 / 917600000 := by
  rw [gapFp4, wrefFp4, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]

theorem gapFp4_sm120Issue_lut256_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At 8 Fp4ScalePath.lut256) :
    gapFp4 d (1 / 400) sh16384 = 32568847 / 5374240000 := by
  rw [gapFp4, wrefFp4, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]

theorem gapFp4_sm120Loop_lut256_8192
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256) :
    gapFp4 d (1 / 400) sh8192 = 9875013901 / 1376663200000 := by
  rw [gapFp4, wrefFp4, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh8192]

theorem gapFp4_sm120Loop_lut256_16384
    (d : Fp4Device) (hp : d.prices = Fp4Prices.sm120At (1047 / 125) Fp4ScalePath.lut256) :
    gapFp4 d (1 / 400) sh16384 = 5473473967 / 895794400000 := by
  rw [gapFp4, wrefFp4, creditFp4, hp]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Params.pi, sh16384]

example : ({ devFp4Sm120 with prices := Fp4Prices.sm120Issue } : Fp4Device).prices = Fp4Prices.sm120Issue := rfl

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GapFp4Sm120IssueLut256_16384 : Pouw.Guarantees.PearlC.GapFp4Sm120IssueLut256_16384 := @Pouw.PearlC.gapFp4_sm120Issue_lut256_16384

theorem Pouw.SecurityProofs.PearlC.GapFp4Sm120IssueLut256_8192 : Pouw.Guarantees.PearlC.GapFp4Sm120IssueLut256_8192 := @Pouw.PearlC.gapFp4_sm120Issue_lut256_8192
