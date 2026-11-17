module

public import Proofs.Pouw.Lemmas.PearlCRecords
public import Proofs.Pouw.PearlC.DeviceInertBoundGap
public import Proofs.Pouw.PearlC.DeviceChainInertBoundGap
public import Proofs.Pouw.PearlC.UOnlyGap
public import Proofs.Pouw.PearlC.FreezeP
public import Proofs.Pouw.PearlC.TTOutW1
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.DeviceInertBoundGap
import Proofs.Pouw.PearlC.TTOutW1Variants
public import Proofs.Pouw.PearlC.Records

/-! g at the RTX PRO 6000 (sm_120) records (staging, nothing pinned). -/

@[expose] public section

namespace Pouw.PearlC

open Pouw.Fp8Atom Pouw.PearlC.Assumptions

theorem devSm120v1_G : (devSm120v1 Prices.sm120).G = 4 := rfl

theorem devSm120v1_prices : (devSm120v1 Prices.sm120).prices = Prices.sm120 := rfl

theorem devSm120v2_G : (devSm120v2 Prices.sm120).G = 0 := rfl

theorem devSm120v2_prices : (devSm120v2 Prices.sm120).prices = Prices.sm120 := rfl

theorem devSm120v1_flags (pr : Prices) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code) :
    (devSm120v1 pr).flags A B = SkipP.unitFlagsP sm120E4m3 4 A B := rfl

theorem devSm120v2_flags (pr : Prices) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code)
    (i : Fin m) (j : Fin n) : (devSm120v2 pr).flags A B i j = SkipP.pureFlagsP sm120E4m3 k (A i) (B j) := rfl

theorem devAt_step {p p' : Pipeline} (h : p.step = p'.step) (G : ℕ) (pr : Prices) :
    { devAt p G pr with pipe := p' } = devAt p' G pr := by
  unfold devAt Accum.chainAt Accum.ctChainP Accum.chainP Accum.totalP PeelExactP.peelExactValP
    PeelExactP.peelExactP PeelExactP.chainWordP? PeelExactP.windowWordP? SkipP.unitFlagsAt SkipP.flagsAt
    SkipP.pureFlagsP SkipP.pureSkipP Accum.pureChain SkipP.wordFlagsP SkipP.jointFlagsP SkipP.jointAtP SkipP.keepsP
    SkipP.wordInP SkipP.stepP SkipP.inTotalP SkipP.promoteGroupP SkipP.laterWordsP Accum.groupWordP
    FormingP.saltDeadP FormingP.atomMixP
  rw [h]

theorem devAt_sm120_floor {z : ℤ} (hz : z ≤ -126) (G : ℕ) (pr : Prices) :
    { devAt (sm120E4m3K32 z) G pr with pipe := sm120E4m3 } = devAt sm120E4m3 G pr :=
  devAt_step (FreezeP.sm120_floor hz (by norm_num)) G pr

theorem devSm120v1_floor {z : ℤ} (hz : z ≤ -126) (pr : Prices) :
    { devAt (sm120E4m3K32 z) 4 pr with pipe := sm120E4m3 } = devSm120v1 pr :=
  devAt_sm120_floor hz 4 pr

theorem devSm120v2_floor {z : ℤ} (hz : z ≤ -126) (pr : Prices) :
    { devAt (sm120E4m3K32 z) 0 pr with pipe := sm120E4m3 } = devSm120v2 pr :=
  devAt_sm120_floor hz 0 pr

theorem devSm120v1_domain (s : Shape) (hs : s = sh8192 ∨ s = sh16384) :
    pearlCDomainDevAt (devSm120v1 Prices.sm120) s ⟨1, fun _ => s, fun _ => 0⟩ := by
  refine ⟨fun u _ => ?_, fun _ _ => rfl⟩
  rcases hs with rfl | rfl <;> dsimp only <;>
    exact ⟨by decide, by decide, by decide, by decide, Or.inr (by decide), by decide⟩

theorem devSm120v2_domain (s : Shape) (hs : s = sh8192 ∨ s = sh16384) :
    pearlCDomainDevAt (devSm120v2 Prices.sm120) s ⟨1, fun _ => s, fun _ => 0⟩ := by
  refine ⟨fun u _ => ?_, fun _ _ => rfl⟩
  rcases hs with rfl | rfl <;> dsimp only <;> exact ⟨by decide, by decide, by decide, by decide, Or.inl rfl, by decide⟩

section Gap

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

theorem pearlCGapSm120v1Rev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) δPearlC :=
  pearlCGapSm120Rev1_8192 CM (devSm120v1 Prices.sm120) sem rfl rfl hTT

theorem pearlCGapSm120v1Rev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2963 / 586700 : ℚ) : ℝ) δPearlC :=
  pearlCGapSm120Rev1_16384 CM (devSm120v1 Prices.sm120) sem rfl rfl hTT

theorem pearlCSampledSm120v1Rev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) δPearlC :=
  pearlCSampledSm120Rev1_8192 CM (devSm120v1 Prices.sm120) sem rfl rfl hTT

theorem pearlCSampledSm120v1Rev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2963 / 586700 : ℚ) : ℝ) δPearlC :=
  pearlCSampledSm120Rev1_16384 CM (devSm120v1 Prices.sm120) sem rfl rfl hTT

theorem pearlCGapUOnlySm120v1Rev1_8192
    (hTT : TTOutPearlCDevUOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GGapU CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) δPearlC :=
  pearlCGapUOnlySm120Rev1_8192 CM (devSm120v1 Prices.sm120) sem rfl rfl hTT

theorem pearlCGapSm120v2InertShare1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedInertShare1000_8192 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

theorem pearlCGapSm120v2InertShare1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedInertShare1000_16384 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

theorem pearlCSampledSm120v2InertShare1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCTilesDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedInertShare1000_8192 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

theorem pearlCSampledSm120v2InertShare1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCTilesDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedInertShare1000_16384 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

theorem pearlCGapUOnlySm120v2InertShare1000_8192
    (hTT : TTOutPearlCDevUOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapU CM (pearlCProtocolDevRev1 (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC :=
  pearlCGapUOnlyUnpromotedInertShare1000_8192 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

theorem pearlCGapSm120v2ChainInertShare1000_8192
    (hTT : TTOutPearlCDevChainInertBound CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((26921 / 7493750 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedChainInertShare1000_8192 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

theorem pearlCGapSm120v2ChainInertShare1000_16384
    (hTT : TTOutPearlCDevChainInertBound CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((367519 / 103656250 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedChainInertShare1000_16384 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

theorem pearlCSampledSm120v2ChainInertShare1000_8192
    (hTT : TTOutTilePearlCDevChainInertBound CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCTilesDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedChainInertShare1000_8192 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

theorem pearlCSampledSm120v2ChainInertShare1000_16384
    (hTT : TTOutTilePearlCDevChainInertBound CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCTilesDevChainInertBound (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedChainInertShare1000_16384 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

end Gap

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Rev1_16384 : Pouw.Guarantees.PearlC.GapSm120v1Rev1_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Rev1_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v1Rev1_8192 : Pouw.Guarantees.PearlC.GapSm120v1Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v1Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v2InertShare1000_16384 : Pouw.Guarantees.PearlC.GapSm120v2InertShare1000_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v2InertShare1000_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDev_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v2InertShare1000_8192 : Pouw.Guarantees.PearlC.GapSm120v2InertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v2InertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDev_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v2ChainInertShare1000_16384 : Pouw.Guarantees.PearlC.GapSm120v2ChainInertShare1000_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v2ChainInertShare1000_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevChainInertBound_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GapSm120v2ChainInertShare1000_8192 : Pouw.Guarantees.PearlC.GapSm120v2ChainInertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGapSm120v2ChainInertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevChainInertBound_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Rev1_16384 : Pouw.Guarantees.PearlC.SampledSm120v1Rev1_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1Rev1_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Rev1_8192 : Pouw.Guarantees.PearlC.SampledSm120v1Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v1Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v2InertShare1000_16384 : Pouw.Guarantees.PearlC.SampledSm120v2InertShare1000_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v2InertShare1000_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDev_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v2InertShare1000_8192 : Pouw.Guarantees.PearlC.SampledSm120v2InertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v2InertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDev_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v2ChainInertShare1000_16384 : Pouw.Guarantees.PearlC.SampledSm120v2ChainInertShare1000_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v2ChainInertShare1000_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevChainInertBound_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledSm120v2ChainInertShare1000_8192 : Pouw.Guarantees.PearlC.SampledSm120v2ChainInertShare1000_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledSm120v2ChainInertShare1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCDevChainInertBound_of_w1 hTT)
