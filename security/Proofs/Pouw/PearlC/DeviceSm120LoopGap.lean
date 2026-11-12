module

public import Definitions.Pouw.Assumptions.PearlC.DeviceChainCap
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.Assumptions.PearlC.TTOutUOnly
public import Definitions.Pouw.PearlC.DevicePricesLoop
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.CapLoopGap
import Proofs.Pouw.PearlC.ChainCapLoopGap
import Proofs.Pouw.PearlC.UOnlyLoopGap
public import Proofs.Pouw.PearlC.DeviceSm120Gap
public import Proofs.Pouw.PearlC.CapLoopGap
public import Proofs.Pouw.PearlC.ChainCapLoopGap
public import Proofs.Pouw.PearlC.UOnlyLoopGap
public import Proofs.Pouw.PearlC.Records

/-!
# g at the RTX PRO 6000 (sm_120) records at the in-loop prices (staged)

The twins of `DeviceSm120Gap`'s gap corollaries at `devSm120v1 Prices.sm120Loop` and `devSm120v2 Prices.sm120Loop` in
place of `Prices.sm120`: each is `<instance at Prices.sm120Loop> CM (devSm120vX Prices.sm120Loop) sem rfl rfl hTT`, with
the instances from `CapLoopGap`, `ChainCapLoopGap` and `UOnlyLoopGap`. This file adds no definitions.
* **The domains are not empty** at the in-loop records (`devSm120v1Loop_domain`, `devSm120v2Loop_domain`).
* **v1 under rev1**, cap 1/400: `310284613/59477920000` (0.52168%) at 8192³ and `1802569231/352995040000` (0.51065%)
  at 16384³, per unit and per audit tile, and with U-only binding per unit at 8192³.
* **v2 at the cap 1/1,000**: `522517/139900000` (0.37349%) at 8192³ and `3000127/829300000` (0.36177%) at 16384³, per
  unit and per audit tile, and with U-only binding per unit at 8192³.
* **v2 at the chain cap 1/1,000**: per unit `64899/17487500` (0.37112%) and `373769/103662500` (0.36056%), per audit
  tile the 1/1,000 row's.

Each is above its twin at `Prices.sm120` (0.51056%, 0.50503%; 0.36162%, 0.35576%; 0.35925%, 0.35456%), so the gap's price
rule publishes these. `Prices.sm120Loop` rounds the A-only forming up to a whole unit, which raises g.

Every g here is an upper bound on the in-loop g: `Prices.sm120Loop` rounds the A-only forming (1.047) up to 2. The
exact in-loop values for v1 under rev1 and v2 at the cap are `DeviceSm120KernelGap`'s `…LoopCast8…` pins.
-/

@[expose] public section

namespace Pouw.PearlC

open Pouw.Fp8Atom Pouw.PearlC.Assumptions

/-- v1's domain at the in-loop prices admits a one-unit layout of either headline shape. -/
theorem devSm120v1Loop_domain (s : Shape) (hs : s = sh8192 ∨ s = sh16384) :
    pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) s ⟨1, fun _ => s, fun _ => 0⟩ := by
  refine ⟨fun u _ => ?_, fun _ _ => rfl⟩
  rcases hs with rfl | rfl <;> dsimp only <;>
    exact ⟨by decide, by decide, by decide, by decide, Or.inr (by decide), by decide⟩

/-- v2's domain at the in-loop prices admits a one-unit layout of either headline shape. -/
theorem devSm120v2Loop_domain (s : Shape) (hs : s = sh8192 ∨ s = sh16384) :
    pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) s ⟨1, fun _ => s, fun _ => 0⟩ := by
  refine ⟨fun u _ => ?_, fun _ _ => rfl⟩
  rcases hs with rfl | rfl <;> dsimp only <;> exact ⟨by decide, by decide, by decide, by decide, Or.inl rfl, by decide⟩

section Gap

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

/-! ## v1 (`devSm120v1 Prices.sm120Loop`, G = 4) under rev1, cap 1/400 -/

/-- **v1 under rev1 at 8192³**, in the loop (an upper bound): `g = 310284613/59477920000`. Twin of
`pearlCGapSm120v1Rev1_8192`. -/
theorem pearlCGapSm120v1LoopRev1_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((310284613 / 59477920000 : ℚ) : ℝ) δPearlC :=
  pearlCGapSm120LoopRev1_8192 CM (devSm120v1 Prices.sm120Loop) sem rfl rfl hTT

/-- **v1 under rev1 at 16384³**, in the loop (an upper bound): `g = 1802569231/352995040000`. Twin of
`pearlCGapSm120v1Rev1_16384`. -/
theorem pearlCGapSm120v1LoopRev1_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((1802569231 / 352995040000 : ℚ) : ℝ) δPearlC :=
  pearlCGapSm120LoopRev1_16384 CM (devSm120v1 Prices.sm120Loop) sem rfl rfl hTT

/-- **v1 under rev1 per audit tile at 8192³**, in the loop (an upper bound): `g = 310284613/59477920000`. Twin of
`pearlCSampledSm120v1Rev1_8192`. -/
theorem pearlCSampledSm120v1LoopRev1_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((310284613 / 59477920000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledSm120LoopRev1_8192 CM (devSm120v1 Prices.sm120Loop) sem rfl rfl hTT

/-- **v1 under rev1 per audit tile at 16384³**, in the loop (an upper bound): `g = 1802569231/352995040000`. Twin of
`pearlCSampledSm120v1Rev1_16384`. -/
theorem pearlCSampledSm120v1LoopRev1_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCTilesDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384) ((1802569231 / 352995040000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledSm120LoopRev1_16384 CM (devSm120v1 Prices.sm120Loop) sem rfl rfl hTT

/-- **v1 under rev1 with U-only binding at 8192³**, in the loop (an upper bound): `g = 310284613/59477920000`. Twin of
`pearlCGapUOnlySm120v1Rev1_8192`. -/
theorem pearlCGapUOnlySm120v1LoopRev1_8192
    (hTT : TTOutPearlCDevUOnly CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GGapU CM (pearlCProtocolDevRev1 (devSm120v1 Prices.sm120Loop) sem (1 / 400))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) ((310284613 / 59477920000 : ℚ) : ℝ) δPearlC :=
  pearlCGapUOnlySm120LoopRev1_8192 CM (devSm120v1 Prices.sm120Loop) sem rfl rfl hTT

/-! ## v2 (`devSm120v2 Prices.sm120Loop`, G = 0) at the cap 1/1,000 -/

/-- **v2 at the cap 1/1,000 at 8192³**, in the loop (an upper bound): `g = 522517/139900000`. Twin of
`pearlCGapSm120v2Cap1000_8192`. -/
theorem pearlCGapSm120v2LoopCap1000_8192
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((522517 / 139900000 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedCap1000Loop_8192 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

/-- **v2 at the cap 1/1,000 at 16384³**, in the loop (an upper bound): `g = 3000127/829300000`. Twin of
`pearlCGapSm120v2Cap1000_16384`. -/
theorem pearlCGapSm120v2LoopCap1000_16384
    (hTT : TTOutPearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((3000127 / 829300000 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedCap1000Loop_16384 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

/-- **v2 at the cap 1/1,000 per audit tile at 8192³**, in the loop (an upper bound): `g = 522517/139900000`. Twin of
`pearlCSampledSm120v2Cap1000_8192`. -/
theorem pearlCSampledSm120v2LoopCap1000_8192
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCTilesDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((522517 / 139900000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedCap1000Loop_8192 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

/-- **v2 at the cap 1/1,000 per audit tile at 16384³**, in the loop (an upper bound): `g = 3000127/829300000`. Twin of
`pearlCSampledSm120v2Cap1000_16384`. -/
theorem pearlCSampledSm120v2LoopCap1000_16384
    (hTT : TTOutTilePearlCDev CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCTilesDev (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((3000127 / 829300000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedCap1000Loop_16384 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

/-- **v2 at the cap 1/1,000 with U-only binding at 8192³**, in the loop (an upper bound): `g = 522517/139900000`. Twin
of `pearlCGapUOnlySm120v2Cap1000_8192`. -/
theorem pearlCGapUOnlySm120v2LoopCap1000_8192
    (hTT : TTOutPearlCDevUOnly CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapU CM (pearlCProtocolDevRev1 (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((522517 / 139900000 : ℚ) : ℝ) δPearlC :=
  pearlCGapUOnlyUnpromotedCap1000Loop_8192 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

/-! ## v2 at the chain cap 1/1,000 -/

/-- **v2 at the chain cap 1/1,000 at 8192³**, per unit, in the loop (an upper bound): `g = 64899/17487500`. Twin of
`pearlCGapSm120v2ChainCap1000_8192`. -/
theorem pearlCGapSm120v2LoopChainCap1000_8192
    (hTT : TTOutPearlCDevChainCap CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((64899 / 17487500 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedChainCap1000Loop_8192 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

/-- **v2 at the chain cap 1/1,000 at 16384³**, per unit, in the loop (an upper bound): `g = 373769/103662500`. Twin of
`pearlCGapSm120v2ChainCap1000_16384`. -/
theorem pearlCGapSm120v2LoopChainCap1000_16384
    (hTT : TTOutPearlCDevChainCap CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((373769 / 103662500 : ℚ) : ℝ) δPearlC :=
  pearlCGapUnpromotedChainCap1000Loop_16384 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

/-- **v2 at the chain cap 1/1,000 per audit tile at 8192³**, in the loop (an upper bound): `g = 522517/139900000`.
Twin of `pearlCSampledSm120v2ChainCap1000_8192`. -/
theorem pearlCSampledSm120v2LoopChainCap1000_8192
    (hTT : TTOutTilePearlCDevChainCap CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCTilesDevChainCap (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((522517 / 139900000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedChainCap1000Loop_8192 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

/-- **v2 at the chain cap 1/1,000 per audit tile at 16384³**, in the loop (an upper bound): `g = 3000127/829300000`.
Twin of `pearlCSampledSm120v2ChainCap1000_16384`. -/
theorem pearlCSampledSm120v2LoopChainCap1000_16384
    (hTT : TTOutTilePearlCDevChainCap CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainCap (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCTilesDevChainCap (devSm120v2 Prices.sm120Loop) sem (1 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((3000127 / 829300000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledUnpromotedChainCap1000Loop_16384 CM (devSm120v2 Prices.sm120Loop) sem rfl rfl hTT

end Gap

end Pouw.PearlC
