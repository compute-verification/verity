module

public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeedK
public import Definitions.Pouw.Assumptions.PearlC.HiddenTile
public import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
public import Definitions.Pouw.PearlC.DevicePricesLoop
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.DeviceCapRev1Gap
import Proofs.Pouw.PearlC.HiddenGap
import Proofs.Pouw.PearlC.RowSeedCode
import Proofs.Pouw.PearlC.RowSeedGap
public import Proofs.Pouw.PearlC.HiddenGap

/-!
# Per-row seeds on the served line: rev1 against the honest kernel's `W_ref` (compute accounting's file)

The served line's gap (`pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_8192`, `pearlCHiddenSm120v1LoopCast8p72Rev1Cap1000_8192`)
is at `pearlCProtocolDevRev1K`: rev1 with `W_ref` at `wrefDevRev1K d c`, at v1's loop record (`devSm120v1
Prices.sm120Loop`, FP32 at 8.376) with the honest cast at 8.72 (`c = 8.72 − 8.953`). Its row-seeded twin is the same g
with the cap checked per row and the error at the row count:

* **At any record** (`pearlCGapDevRev1KRowSeedAt`, `pearlCSampledDevRev1KRowSeedAt`): `RowSeedGap`'s W-generic
  lemmas at `W = wrefDevRev1K d c`. The reduction to rev1's TT_OUT on the split workload (`ttOutRowSeed_of_ttOut`,
  `ttOutTileRowSeed_of_ttOutTile`) doesn't read `W_ref`, so the kernel's `W_ref` enters only the per-unit bound
  `W_ref ≤ ω·credit` on the original layout, as in `KernelWrefGap`.
* **At the served record, cap 1/1,000, 8192³**, from rev1's TT_OUT (per unit or per audit tile) at that record, C1
  (`RowSeedNoise`) and C3 (`CM.SplitClosed`) alone; C0 is `devAt_rowLocal`, by `rfl`, and C2 is the protocol:
  `g = 1648180439/446072750000` (0.36949%), the served line's, at error `δPearlC q (8192·N)` on the domain of at most
  `2^64` rows: `pearlCGapSm120v1LoopCast8p72Rev1RowSeedCap1000_8192` and
  `pearlCSampledSm120v1LoopCast8p72Rev1RowSeedCap1000_8192`.
* **The code-form seed (`-h3`)**, adding C1-h3 (`CodeSeedNoise`) and C5 (`CM.RelabelClosed`):
  `pearlCSampledSm120v1LoopCast8p72Rev1RowSeedCodeCap1000_8192`.
* **Under the hidden audit, conditional** as the served line's (`gapHidden_of_sampled_all`):
  `pearlCHiddenSm120v1LoopCast8p72Rev1RowSeedCap1000_8192` and its code form
  `pearlCHiddenSm120v1LoopCast8p72Rev1RowSeedCodeCap1000_8192`.
-/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section Gap

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- The row-seeded TT_OUT gives `G_g` for rev1 with the per-row cap against the honest kernel's `W_ref` on the one-shape
domain at `g = 1 − (399/400)/ω`, where `wrefDevRev1K d c = ω·(1 − ρ)·creditDevRev1`, with the error at the row count. -/
theorem pearlCGapDevRev1KRowSeedAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ)
    (s : Shape) (ω g : ℚ) (hω0 : 0 < ω) (hω : wrefDevRev1K d c s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hg : g = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρ) (pearlCDomainDev d) (1 / 400)) :
    GGap CM (pearlCProtocolDevRev1KRowCap d sem ρ c) (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (g : ℝ)
      (fun q N => δPearlC q (N * s.m)) :=
  pearlCGapDevRev1RowSeedWAt CM d sem ρ (wrefDevRev1K d c) s ω g hω0 hω hg hTT

/-- The row-seeded TT_OUT per audit tile gives `GGapSampled` for rev1's tiles with the per-row cap against the honest
kernel's `W_ref` on the one-shape domain at `g = 1 − (399/400)/ω`, where `wrefDevRev1K d c = ω·(1 − ρ)·creditDevRev1 ≥ 0`,
with the error at the row count. -/
theorem pearlCSampledDevRev1KRowSeedAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ)
    (s : Shape) (ω g : ℚ) (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDevRev1K d c s)
    (hω : wrefDevRev1K d c s = ω * ((1 - ρ) * creditDevRev1 d s)) (hg : g = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρ) (pearlCTilesDevRev1RowCap d sem ρ)
      (pearlCDomainDev d) (1 / 400)) :
    GGapSampled CM (pearlCProtocolDevRev1KRowCap d sem ρ c) (pearlCTilesDevRev1KRowCap d sem ρ c)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (g : ℝ) (fun q N => δPearlC q (N * s.m)) :=
  pearlCSampledDevRev1RowSeedWAt CM d sem ρ (wrefDevRev1K d c) s ω g hω0 hw0 hω hg hTT

/-! ## The served record -/

theorem devSm120v1Loop_add : 0 ≤ (devSm120v1 Prices.sm120Loop).prices.add := by
  show (0 : ℚ) ≤ Prices.sm120Loop.add; norm_num [Prices.sm120Loop]

theorem devSm120v1Loop_bf16 : 0 ≤ (devSm120v1 Prices.sm120Loop).prices.bf16 := by
  show (0 : ℚ) ≤ Prices.sm120Loop.bf16; norm_num [Prices.sm120Loop]

theorem wrefDevRev1K_sm120v1LoopCast8p72_8192_cap1000 :
    wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) sh8192 =
      8921455 / 8919688 / (1 - 1 / 1000) *
        ((1 - 1 / 1000) * creditDevRev1 (devSm120v1 Prices.sm120Loop) sh8192) := by
  rw [wrefDevRev1K_sm120v1LoopCast8p72_8192]
  ring

variable (CM : CostModel Q R S) {sem sem' semC : PearlCSem Q R S}

/-- **v1 under rev1 with per-row seeds at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at 8.72, from rev1's
TT_OUT at the served record: `g = 1648180439/446072750000` (0.36949%), the served line's, with the error at the `8192·N`
rows. -/
theorem pearlCGapSm120v1LoopCast8p72Rev1RowSeedCap1000_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    GGap CM (pearlCProtocolDevRev1KRowCap (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64)
      ((1648180439 / 446072750000 : ℚ) : ℝ) (fun q N => δPearlC q (N * 8192)) :=
  pearlCGapDevRev1KRowSeedAt CM (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000) sh8192
    (8921455 / 8919688 / (1 - 1 / 1000)) _ (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_8192_cap1000
    (by norm_num)
    (ttOutRowSeed_of_ttOut hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) devSm120v1Loop_add devSm120v1Loop_bf16)

/-- **v1 under rev1 with per-row seeds per audit tile at the cap 1/1,000, 8192³**, FP32 at 8.376, the honest cast at
8.72, from rev1's TT_OUT per tile at the served record: `g = 1648180439/446072750000` (0.36949%), the served line's,
with the error at the `8192·N` rows. -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1RowSeedCap1000_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    GGapSampled CM
      (pearlCProtocolDevRev1KRowCap (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowCap (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64)
      ((1648180439 / 446072750000 : ℚ) : ℝ) (fun q N => δPearlC q (N * 8192)) :=
  pearlCSampledDevRev1KRowSeedAt CM (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000) sh8192
    (8921455 / 8919688 / (1 - 1 / 1000)) _ (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_8192_nonneg
    wrefDevRev1K_sm120v1LoopCast8p72_8192_cap1000 (by norm_num)
    (ttOutTileRowSeed_of_ttOutTile hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) devSm120v1Loop_add
      devSm120v1Loop_bf16)

/-- **The code-form seed (`-h3`) per audit tile at the cap 1/1,000, 8192³**, from rev1's TT_OUT per tile at the served
record, C1, C1-h3, C3 and C5: `g = 1648180439/446072750000` (0.36949%), with the error at the `8192·N` rows. -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1RowSeedCodeCap1000_8192 {π : Layout → Q ≃ Q}
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hC : CodeSeedNoise sem' semC π) (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed) :
    GGapSampled CM
      (pearlCProtocolDevRev1KRowCap (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowCap (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64)
      ((1648180439 / 446072750000 : ℚ) : ℝ) (fun q N => δPearlC q (N * 8192)) :=
  pearlCSampledDevRev1KRowSeedAt CM (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000) sh8192
    (8921455 / 8919688 / (1 - 1 / 1000)) _ (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_8192_nonneg
    wrefDevRev1K_sm120v1LoopCast8p72_8192_cap1000 (by norm_num)
    (ttOutTileRowSeedCode_of_ttOutTile hTT hN hC hCM hCM' (devAt_rowLocal _ _ _) (by norm_num) devSm120v1Loop_add
      devSm120v1Loop_bf16)

/-- **v1 under rev1 with per-row seeds per audit tile at the cap 1/1,000, 8192³, under the hidden audit,
conditional**, FP32 at 8.376, the honest cast at 8.72: `g = 1648180439/446072750000` (0.36949%), with the error at the
`8192·N` rows, for any hidden audit sound per tile up to `δzk(q)`.

Conditional as `pearlCHiddenSm120v1LoopCast8p72Rev1Cap1000_8192`: `HA` and `δzk` are free, and the open obligation
`TileProofSoundAll` holds at a degenerate hidden audit (an empty class) for any `δzk ≥ 0`, so this says nothing about
Flock until Flock's `HiddenAudit` instance and the obligation's discharge at it land. Cite it only as conditional. -/
theorem pearlCHiddenSm120v1LoopCast8p72Rev1RowSeedCap1000_8192 (HA : HiddenAudit Q R S) (δzk : ℕ → ℝ)
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed)
    (hzk : TileProofSoundAll
      (pearlCProtocolDevRev1KRowCap (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowCap (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64) HA δzk) :
    GGapHidden CM
      (pearlCProtocolDevRev1KRowCap (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowCap (devSm120v1 Prices.sm120Loop) sem' (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64) HA
      ((1648180439 / 446072750000 : ℚ) : ℝ) (fun q N => δPearlC q (N * 8192)) δzk :=
  gapHidden_of_sampled_all CM _ _ _ HA _ _ δzk
    (pearlCSampledSm120v1LoopCast8p72Rev1RowSeedCap1000_8192 CM hTT hN hCM) hzk

/-- **The code-form seed (`-h3`) per audit tile at the cap 1/1,000, 8192³, under the hidden audit, conditional**, as
`pearlCHiddenSm120v1LoopCast8p72Rev1RowSeedCap1000_8192` with C1-h3 and C5 added:
`g = 1648180439/446072750000` (0.36949%), with the error at the `8192·N` rows. Cite it only as conditional. -/
theorem pearlCHiddenSm120v1LoopCast8p72Rev1RowSeedCodeCap1000_8192 {π : Layout → Q ≃ Q} (HA : HiddenAudit Q R S)
    (δzk : ℕ → ℝ) (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000))
    (hN : RowSeedNoise sem sem') (hC : CodeSeedNoise sem' semC π) (hCM : CM.SplitClosed) (hCM' : CM.RelabelClosed)
    (hzk : TileProofSoundAll
      (pearlCProtocolDevRev1KRowCap (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowCap (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64) HA δzk) :
    GGapHidden CM
      (pearlCProtocolDevRev1KRowCap (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1KRowCap (devSm120v1 Prices.sm120Loop) semC (1 / 1000) (218 / 25 - 8953 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192 L ∧ rowsOf L ≤ 2 ^ 64) HA
      ((1648180439 / 446072750000 : ℚ) : ℝ) (fun q N => δPearlC q (N * 8192)) δzk :=
  gapHidden_of_sampled_all CM _ _ _ HA _ _ δzk
    (pearlCSampledSm120v1LoopCast8p72Rev1RowSeedCodeCap1000_8192 CM hTT hN hC hCM hCM') hzk

end Gap

end Pouw.PearlC
