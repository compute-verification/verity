import Definitions.Pouw.PearlC.DevicePrices
import Proofs.Pouw.PearlC.DeviceCapGamma
import Proofs.Pouw.PearlC.DeviceChainCapGamma
import Proofs.Pouw.PearlC.UOnlyGamma
import Proofs.Pouw.PearlC.TileUOnlyGamma
import Proofs.Pouw.PearlC.FreezeP
import Models.Pouw.Guarantees.PearlC

/-!
# γ at the RTX PRO 6000 (sm_120) records (staging, nothing pinned)

The records are `Device`'s `devSm120v1 Prices.sm120` (`pearl-c-sm120` v1, promotion every 4 atoms) and
`devSm120v2 Prices.sm120` (v2, no promotion): `devAt sm120E4m3 G Prices.sm120`, the FP8 lane's twins at sm_120's step
and sm_120's measured prices. This file adds no definitions.
* **The fields** (`devSm120v1_G`, `devSm120v1_prices`, `devSm120v2_G`, `devSm120v2_prices`): 4 and 0, both at
  `Prices.sm120`, by `rfl`. These are the hypotheses `hG` and `hp` of the γ instances.
* **The flags** (`devSm120v1_flags`, `devSm120v2_flags`): v1's are `SkipP.unitFlagsP sm120E4m3 4`, debit (A)'s joint
  flags per group of 4, and v2's are the pure chain's lone skips `SkipP.pureFlagsP sm120E4m3`, both by `rfl` from
  `devAt`'s `SkipP.unitFlagsAt`.
* **The floor is unobservable** (`devAt_sm120_floor`, `devSm120v1_floor`, `devSm120v2_floor`): at every floor
  `z ≤ −126`, `devAt (sm120E4m3K32 z) G pr = devAt sm120E4m3 G pr`, so the pinned −133 is no choice. Each map reads the
  step only through `p.step` (`devAt_step`), and `FreezeP.sm120_floor` gives one step for every such floor.
* **The domains are not empty** (`devSm120v1_domain`, `devSm120v2_domain`): a one-unit layout of either headline shape
  is in `pearlCDomainDevAt` at both records.
* **γ at the records**, from the instances at any record with `hG` and `hp`, each hypothesis by `rfl`:
  - v1 under rev1, cap 1/400: `16223/3177500` (0.51056%) at 8192³ and `2963/586700` (0.50503%) at 16384³, per unit
    and per audit tile, and with U-only binding (per unit and per audit tile) at 8192³;
  - v2 at the cap 1/1,000: `216793/59950000` (0.36162%) at 8192³ and `2950127/829250000` (0.35576%) at 16384³, per unit
    and per audit tile, and with U-only binding (per unit and per audit tile) at 8192³;
  - v2 at the chain cap 1/1,000: per unit `26921/7493750` (0.35925%) and `367519/103656250` (0.35456%), per audit tile
    the 1/1,000 row's.
-/

namespace Pouw.PearlC

open Pouw.Fp8Atom Pouw.PearlC.Assumptions

theorem devSm120v1_G : (devSm120v1 Prices.sm120).G = 4 := rfl

theorem devSm120v1_prices : (devSm120v1 Prices.sm120).prices = Prices.sm120 := rfl

theorem devSm120v2_G : (devSm120v2 Prices.sm120).G = 0 := rfl

theorem devSm120v2_prices : (devSm120v2 Prices.sm120).prices = Prices.sm120 := rfl

/-- **v1's flags** are debit (A)'s joint flags per promotion group of 4 on sm_120's step. -/
theorem devSm120v1_flags (pr : Prices) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code) :
    (devSm120v1 pr).flags A B = SkipP.unitFlagsP sm120E4m3 4 A B := rfl

/-- **v2's flags** are the pure chain's lone skips on sm_120's step: atom `t` is flagged when skipping it alone leaves
the unpromoted chain's output unchanged (`FreezePPure.pureFlags_iff`). -/
theorem devSm120v2_flags (pr : Prices) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code)
    (i : Fin m) (j : Fin n) : (devSm120v2 pr).flags A B i j = SkipP.pureFlagsP sm120E4m3 k (A i) (B j) := rfl

/-- The record of a captured step reads the step only through `p.step`. -/
theorem devAt_step {p p' : Pipeline} (h : p.step = p'.step) (G : ℕ) (pr : Prices) : devAt p G pr = devAt p' G pr := by
  unfold devAt Accum.chainAt Accum.ctChainP Accum.chainP Accum.totalP PeelExactP.peelExactValP
    PeelExactP.peelExactP PeelExactP.chainWordP? PeelExactP.windowWordP? SkipP.unitFlagsAt SkipP.flagsAt
    SkipP.pureFlagsP SkipP.pureSkipP Accum.pureChain SkipP.wordFlagsP SkipP.jointFlagsP SkipP.jointAtP SkipP.keepsP
    SkipP.wordInP SkipP.stepP SkipP.inTotalP SkipP.promoteGroupP SkipP.laterWordsP Accum.groupWordP
    FormingP.saltDeadP FormingP.atomMixP
  rw [h]

/-- **The floor is unobservable in the record**: at every floor at most −126 the sm_120 record is the one at −133. -/
theorem devAt_sm120_floor {z : ℤ} (hz : z ≤ -126) (G : ℕ) (pr : Prices) :
    devAt (sm120E4m3K32 z) G pr = devAt sm120E4m3 G pr :=
  devAt_step (FreezeP.sm120_floor hz (by norm_num)) G pr

theorem devSm120v1_floor {z : ℤ} (hz : z ≤ -126) (pr : Prices) : devAt (sm120E4m3K32 z) 4 pr = devSm120v1 pr :=
  devAt_sm120_floor hz 4 pr

theorem devSm120v2_floor {z : ℤ} (hz : z ≤ -126) (pr : Prices) : devAt (sm120E4m3K32 z) 0 pr = devSm120v2 pr :=
  devAt_sm120_floor hz 0 pr

/-- v1's domain at one headline shape admits a one-unit layout of that shape. -/
theorem devSm120v1_domain (s : Shape) (hs : s = sh8192 ∨ s = sh16384) :
    pearlCDomainDevAt (devSm120v1 Prices.sm120) s ⟨1, fun _ => s, fun _ => 0⟩ := by
  refine ⟨fun u _ => ?_, fun _ _ => rfl⟩
  rcases hs with rfl | rfl <;> dsimp only <;>
    exact ⟨by decide, by decide, by decide, by decide, Or.inr (by decide), by decide⟩

/-- v2's domain at one headline shape admits a one-unit layout of that shape. -/
theorem devSm120v2_domain (s : Shape) (hs : s = sh8192 ∨ s = sh16384) :
    pearlCDomainDevAt (devSm120v2 Prices.sm120) s ⟨1, fun _ => s, fun _ => 0⟩ := by
  refine ⟨fun u _ => ?_, fun _ _ => rfl⟩
  rcases hs with rfl | rfl <;> dsimp only <;> exact ⟨by decide, by decide, by decide, by decide, Or.inl rfl, by decide⟩

section Gamma

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

/-! ## v1 (`devSm120v1 Prices.sm120`, G = 4) under rev1, cap 1/400 -/

/-- **v1 under rev1 at 8192³**: `γ = 16223/3177500`. -/
theorem pearlCGammaSm120v1Rev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) εPearlC :=
  pearlCGammaSm120Rev1_8192 CM (devSm120v1 Prices.sm120) sem rfl rfl hTT

/-- **v1 under rev1 at 16384³**: `γ = 2963/586700`. -/
theorem pearlCGammaSm120v1Rev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2963 / 586700 : ℚ) : ℝ) εPearlC :=
  pearlCGammaSm120Rev1_16384 CM (devSm120v1 Prices.sm120) sem rfl rfl hTT

/-- **v1 under rev1 per audit tile at 8192³**: `γ = 16223/3177500`. -/
theorem pearlCSampledSm120v1Rev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) εPearlC :=
  pearlCSampledSm120Rev1_8192 CM (devSm120v1 Prices.sm120) sem rfl rfl hTT

/-- **v1 under rev1 per audit tile at 16384³**: `γ = 2963/586700`. -/
theorem pearlCSampledSm120v1Rev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384) ((2963 / 586700 : ℚ) : ℝ) εPearlC :=
  pearlCSampledSm120Rev1_16384 CM (devSm120v1 Prices.sm120) sem rfl rfl hTT

/-- **v1 under rev1 with U-only binding at 8192³**: `γ = 16223/3177500`. -/
theorem pearlCGammaUOnlySm120v1Rev1_8192
    (hTT : TTOutPearlCDevUOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GγU CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) εPearlC :=
  pearlCGammaUOnlySm120Rev1_8192 CM (devSm120v1 Prices.sm120) sem rfl rfl hTT

/-- **v1 under rev1 per audit tile with U-only binding at 8192³**: `γ = 16223/3177500`. -/
theorem pearlCSampledUOnlySm120v1Rev1_8192
    (hTT : TTOutTilePearlCDevUOnly CM (devSm120v1 Prices.sm120) sem (1 / 400)) :
    GγSampledU CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192) ((16223 / 3177500 : ℚ) : ℝ) εPearlC :=
  pearlCSampledUOnlySm120Rev1_8192 CM (devSm120v1 Prices.sm120) sem rfl rfl hTT

/-! ## v2 (`devSm120v2 Prices.sm120`, G = 0) at the cap 1/1,000 -/

/-- **v2 at the cap 1/1,000 at 8192³**: `γ = 216793/59950000`. -/
theorem pearlCGammaSm120v2Cap1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaUnpromotedCap1000_8192 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

/-- **v2 at the cap 1/1,000 at 16384³**: `γ = 2950127/829250000`. -/
theorem pearlCGammaSm120v2Cap1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaUnpromotedCap1000_16384 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

/-- **v2 at the cap 1/1,000 per audit tile at 8192³**: `γ = 216793/59950000`. -/
theorem pearlCSampledSm120v2Cap1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCTilesDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledUnpromotedCap1000_8192 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

/-- **v2 at the cap 1/1,000 per audit tile at 16384³**: `γ = 2950127/829250000`. -/
theorem pearlCSampledSm120v2Cap1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCTilesDev (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledUnpromotedCap1000_16384 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

/-- **v2 at the cap 1/1,000 with U-only binding at 8192³**: `γ = 216793/59950000`. -/
theorem pearlCGammaUOnlySm120v2Cap1000_8192
    (hTT : TTOutPearlCDevUOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GγU CM (pearlCProtocolDevRev1 (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaUOnlyUnpromotedCap1000_8192 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

/-- **v2 at the cap 1/1,000 per audit tile with U-only binding at 8192³**: `γ = 216793/59950000`. -/
theorem pearlCSampledUOnlySm120v2Cap1000_8192
    (hTT : TTOutTilePearlCDevUOnly CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GγSampledU CM (pearlCProtocolDevRev1 (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCTilesDevRev1 (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledUOnlyUnpromotedCap1000_8192 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

/-! ## v2 at the chain cap 1/1,000 -/

/-- **v2 at the chain cap 1/1,000 at 8192³**, per unit: `γ = 26921/7493750`. -/
theorem pearlCGammaSm120v2ChainCap1000_8192
    (hTT : TTOutPearlCDevChainCap CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((26921 / 7493750 : ℚ) : ℝ) εPearlC :=
  pearlCGammaUnpromotedChainCap1000_8192 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

/-- **v2 at the chain cap 1/1,000 at 16384³**, per unit: `γ = 367519/103656250`. -/
theorem pearlCGammaSm120v2ChainCap1000_16384
    (hTT : TTOutPearlCDevChainCap CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((367519 / 103656250 : ℚ) : ℝ) εPearlC :=
  pearlCGammaUnpromotedChainCap1000_16384 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

/-- **v2 at the chain cap 1/1,000 per audit tile at 8192³**: `γ = 216793/59950000`. -/
theorem pearlCSampledSm120v2ChainCap1000_8192
    (hTT : TTOutTilePearlCDevChainCap CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCTilesDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledUnpromotedChainCap1000_8192 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

/-- **v2 at the chain cap 1/1,000 per audit tile at 16384³**: `γ = 2950127/829250000`. -/
theorem pearlCSampledSm120v2ChainCap1000_16384
    (hTT : TTOutTilePearlCDevChainCap CM (devSm120v2 Prices.sm120) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCTilesDevChainCap (devSm120v2 Prices.sm120) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384) ((2950127 / 829250000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledUnpromotedChainCap1000_16384 CM (devSm120v2 Prices.sm120) sem rfl rfl hTT

end Gamma

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Rev1_16384 : Pouw.Guarantees.PearlC.GammaSm120v1Rev1_16384 := @Pouw.PearlC.pearlCGammaSm120v1Rev1_16384

theorem Pouw.SecurityProofs.PearlC.GammaSm120v1Rev1_8192 : Pouw.Guarantees.PearlC.GammaSm120v1Rev1_8192 := @Pouw.PearlC.pearlCGammaSm120v1Rev1_8192

theorem Pouw.SecurityProofs.PearlC.GammaSm120v2Cap1000_16384 : Pouw.Guarantees.PearlC.GammaSm120v2Cap1000_16384 := @Pouw.PearlC.pearlCGammaSm120v2Cap1000_16384

theorem Pouw.SecurityProofs.PearlC.GammaSm120v2Cap1000_8192 : Pouw.Guarantees.PearlC.GammaSm120v2Cap1000_8192 := @Pouw.PearlC.pearlCGammaSm120v2Cap1000_8192

theorem Pouw.SecurityProofs.PearlC.GammaSm120v2ChainCap1000_16384 : Pouw.Guarantees.PearlC.GammaSm120v2ChainCap1000_16384 := @Pouw.PearlC.pearlCGammaSm120v2ChainCap1000_16384

theorem Pouw.SecurityProofs.PearlC.GammaSm120v2ChainCap1000_8192 : Pouw.Guarantees.PearlC.GammaSm120v2ChainCap1000_8192 := @Pouw.PearlC.pearlCGammaSm120v2ChainCap1000_8192

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Rev1_16384 : Pouw.Guarantees.PearlC.SampledSm120v1Rev1_16384 := @Pouw.PearlC.pearlCSampledSm120v1Rev1_16384

theorem Pouw.SecurityProofs.PearlC.SampledSm120v1Rev1_8192 : Pouw.Guarantees.PearlC.SampledSm120v1Rev1_8192 := @Pouw.PearlC.pearlCSampledSm120v1Rev1_8192

theorem Pouw.SecurityProofs.PearlC.SampledSm120v2Cap1000_16384 : Pouw.Guarantees.PearlC.SampledSm120v2Cap1000_16384 := @Pouw.PearlC.pearlCSampledSm120v2Cap1000_16384

theorem Pouw.SecurityProofs.PearlC.SampledSm120v2Cap1000_8192 : Pouw.Guarantees.PearlC.SampledSm120v2Cap1000_8192 := @Pouw.PearlC.pearlCSampledSm120v2Cap1000_8192

theorem Pouw.SecurityProofs.PearlC.SampledSm120v2ChainCap1000_16384 : Pouw.Guarantees.PearlC.SampledSm120v2ChainCap1000_16384 := @Pouw.PearlC.pearlCSampledSm120v2ChainCap1000_16384

theorem Pouw.SecurityProofs.PearlC.SampledSm120v2ChainCap1000_8192 : Pouw.Guarantees.PearlC.SampledSm120v2ChainCap1000_8192 := @Pouw.PearlC.pearlCSampledSm120v2ChainCap1000_8192
