module

public import Proofs.Pouw.PearlC.DeviceChainCapGap
public import Definitions.Pouw.Assumptions.PearlC.DeviceChainCapKernel
public import Proofs.Pouw.PearlC.KernelWrefGap
public import Definitions.Pouw.PearlC.DevicePricesLoop
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.KernelWrefGap
import Proofs.Pouw.PearlC.TileGap

/-!
# v2 at the chain cap 1/1,000 against the honest kernel's `W_ref`: the exact in-loop values (staged)

* **The gap at any record against the kernel's `W_ref`, with the chain cap** (`pearlCGapDevChainCapKAt`,
  `pearlCSampledDevChainCapKAt`), from the record's own chain-cap TT_OUT, which doesn't read `W_ref` (`ttOut_wref`). Per
  unit `wrefDevK d c = ω·(creditDev − ρ·chainCreditDev)`; per audit tile `wrefDevK d c = ω·(1 − ρ)·creditDev`, since
  a good tile's credit is the unit cap's.
* **The exact in-loop instances** at `devSm120v2 Prices.sm120Loop` with `c = 8 − 8953/1000`: the statement's cast at
  the credited 8.0 and the A-only forming at its measured 1.047. Per unit `754963/209826175` (0.35980%) at 8192³ and
  `490417/138208725` (0.35484%) at 16384³; per audit tile the unit cap's exact values, `1519901/419652350` (0.36218%)
  and `109351/30713050` (0.35604%). `ChainCapLoopGap`'s and `DeviceSm120LoopGap`'s chain-cap twins bound them from
  above (`qa` rounded up to 2). At FADD 8.00 the kernel's `W_ref` at the statement's cast is the statement's own, so
  `DeviceSm120Gap`'s chain-cap instances are the issue-bound values (0.35925%, 0.35456%).
-/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- TT_OUT with the chain cap gives `G_g` against the kernel's `W_ref` on the one-shape domain at
`g = 1 − (399/400)/ω`, where `wrefDevK d c = ω·(creditDev − ρ·chainCreditDev)`: an admitted unit's debit is at most
`ρ·chainCreditDev`. -/
theorem pearlCGapDevChainCapKAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (s : Shape) (ω g : ℚ) (hω0 : 0 < ω)
    (hω : wrefDevK d c s = ω * (creditDev d s - ρ * chainCreditDev d s)) (hg : g = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutPearlCDevChainCap CM d sem ρ) :
    GGap CM (pearlCProtocolDevChainCapK d sem ρ c) (pearlCDomainDevAt d s) (g : ℝ) δPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevChainCapK d sem ρ c) (pearlCDomainDevAt d s) →
      ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N, ∀ act : Codes,
        (pearlCProtocolDevChainCapK d sem ρ c).actOK (U.layout.shape u) act →
        (pearlCProtocolDevChainCapK d sem ρ c).capOK H s' U u act →
          (pearlCProtocolDevChainCapK d sem ρ c).Wref U.layout u ≤
            (ω : ℝ) * (pearlCProtocolDevChainCapK d sem ρ c).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hsh : U.layout.shape u = s := hU.2.2 u hu
    have hc : debitDev d (sem.unitDebitDev d H s' U u act) ≤ ρ * chainCreditDev d (U.layout.shape u) := hcap
    show (wrefDevK d c (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditDev d (U.layout.shape u) - debitDev d (sem.unitDebitDev d H s' U u act) : ℚ) : ℝ)
    rw [hsh] at hc ⊢
    rw [hω]
    have : ω * (creditDev d s - ρ * chainCreditDev d s) ≤
        ω * (creditDev d s - debitDev d (sem.unitDebitDev d H s' U u act)) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast this
  have key := gapFromTTOut CM (pearlCProtocolDevChainCapK d sem ρ c) (pearlCDomainDevAt d s) (1 / 400) (ω : ℝ)
    δPearlC (by exact_mod_cast hω0) hW (ttOut_mono_domain (fun _ h => h.1) (ttOut_wref _ hTT))
  have e : (g : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hg]; push_cast; ring
  rw [e]
  exact key

/-- TT_OUT per tile with the chain cap gives `GGapSampled` against the kernel's `W_ref` on the one-shape domain at
`γ = 1 − (399/400)/ω`, where `wrefDevK d c = ω·(1 − ρ)·creditDev ≥ 0`: the tiles' credit is `pearlCTilesDev`'s. -/
theorem pearlCSampledDevChainCapKAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hw0 : 0 ≤ wrefDevK d c s) (hω : wrefDevK d c s = ω * ((1 - ρ) * creditDev d s))
    (hγ : γ = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevChainCap CM d sem ρ) :
    GGapSampled CM (pearlCProtocolDevChainCapK d sem ρ c) (pearlCTilesDevChainCapK d sem ρ c) (pearlCDomainDevAt d s)
      (γ : ℝ) δPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevChainCapK d sem ρ c) (pearlCDomainDevAt d s) →
      ∀ (H : Q → R) (s' : S), ∀ g < ((pearlCTilesDevChainCapK d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevChainCapK d sem ρ c).capOK H s' U g act →
        (pearlCTilesDevChainCapK d sem ρ c).Wcred U.layout g act ≤
          (ω : ℝ) * (pearlCTilesDevChainCapK d sem ρ c).credit H s' U g act := by
    intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevChainCapK, pearlCTilesDevChainCap, pearlCTilesDev]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  have hWc : ∀ U : Workload, U.InDomain (pearlCProtocolDevChainCapK d sem ρ c) (pearlCDomainDevAt d s) →
      ∀ g < ((pearlCTilesDevChainCapK d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevChainCapK d sem ρ c).Wcred U.layout g act ≤
          (pearlCTilesDevChainCapK d sem ρ c).Wref U.layout g := by
    intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevChainCapK, pearlCTilesDevChainCap, pearlCTilesDev]
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) hw0
  have key := gapSampled CM (pearlCProtocolDevChainCapK d sem ρ c) (pearlCTilesDevChainCapK d sem ρ c)
    (pearlCDomainDevAt d s) (1 / 400) (ω : ℝ) δPearlC (by norm_num) (by exact_mod_cast hω0) hW hWc
    (ttOutTile_mono_domain (fun _ h => h.1) (ttOutTile_wref _ _ _ hTT))
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

section Instances

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

/-- **v2 at the chain cap 1/1,000, per unit, 8192³, the exact in-loop value**: FP32 at 8.376, the statement's cast at
8.0 and the A-only forming at its measured 1.047, `g = 754963/209826175` (0.35980%).
`pearlCGapSm120v2LoopChainCap1000_8192` bounds it from above (`qa` rounded up to 2). -/
theorem pearlCGapSm120v2LoopCast8ChainCap1000_8192
    (hTT : TTOutPearlCDevChainCap CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainCapK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((754963 / 209826175 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevChainCapKAt CM _ sem _ _ _ (8393047 / 8383808) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev, chainCreditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

/-- **v2 at the chain cap 1/1,000, per unit, 16384³, the exact in-loop value**: FP32 at 8.376, the statement's cast at
8.0 and the A-only forming at its measured 1.047, `g = 490417/138208725` (0.35484%).
`pearlCGapSm120v2LoopChainCap1000_16384` bounds it from above (`qa` rounded up to 2). -/
theorem pearlCGapSm120v2LoopCast8ChainCap1000_16384
    (hTT : TTOutPearlCDevChainCap CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainCapK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((490417 / 138208725 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevChainCapKAt CM _ sem _ _ _ (16585047 / 16567616) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev, chainCreditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

/-- **v2 at the chain cap 1/1,000, per audit tile, 8192³, the exact in-loop value**: FP32 at 8.376, the statement's
cast at 8.0 and the A-only forming at its measured 1.047, `g = 1519901/419652350` (0.36218%), the unit cap's.
`pearlCSampledSm120v2LoopChainCap1000_8192` bounds it from above (`qa` rounded up to 2). -/
theorem pearlCSampledSm120v2LoopCast8ChainCap1000_8192
    (hTT : TTOutTilePearlCDevChainCap CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainCapK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCTilesDevChainCapK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1519901 / 419652350 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevChainCapKAt CM _ sem _ _ _ (8393047 / 8383608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

/-- **v2 at the chain cap 1/1,000, per audit tile, 16384³, the exact in-loop value**: FP32 at 8.376, the statement's
cast at 8.0 and the A-only forming at its measured 1.047, `g = 109351/30713050` (0.35604%), the unit cap's.
`pearlCSampledSm120v2LoopChainCap1000_16384` bounds it from above (`qa` rounded up to 2). -/
theorem pearlCSampledSm120v2LoopCast8ChainCap1000_16384
    (hTT : TTOutTilePearlCDevChainCap CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainCapK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCTilesDevChainCapK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((109351 / 30713050 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevChainCapKAt CM _ sem _ _ _ (614261 / 613608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

end Instances

end Pouw.PearlC
