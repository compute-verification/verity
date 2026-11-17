module

public import Proofs.Pouw.PearlC.DeviceChainInertBoundGap
public import Definitions.Pouw.Assumptions.PearlC.DeviceChainInertBoundKernel
public import Proofs.Pouw.PearlC.KernelWrefGap
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.PearlC.KernelWrefGap
import Proofs.Pouw.PearlC.TileGap
public import Proofs.Pouw.PearlC.Records

/-! v2 at the chain cap 1/1,000 against the honest kernel's `W_ref`: the exact in-loop values (staged). -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

theorem pearlCGapDevChainInertBoundKAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (s : Shape) (ω g : ℚ) (hω0 : 0 < ω)
    (hω : wrefDevK d c s = ω * (creditDev d s - ρ * chainCreditDev d s)) (hg : g = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutPearlCDevChainInertBound CM d sem ρ) :
    GGap CM (pearlCProtocolDevChainInertBoundK d sem ρ c) (pearlCDomainDevAt d s) (g : ℝ) δPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevChainInertBoundK d sem ρ c) (pearlCDomainDevAt d s) →
      ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N, ∀ act : Codes,
        (pearlCProtocolDevChainInertBoundK d sem ρ c).actOK (U.layout.shape u) act →
        (pearlCProtocolDevChainInertBoundK d sem ρ c).inertBoundOK H s' U u act →
          (pearlCProtocolDevChainInertBoundK d sem ρ c).Wref U.layout u ≤
            (ω : ℝ) * (pearlCProtocolDevChainInertBoundK d sem ρ c).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hsh : U.layout.shape u = s := hU.2.2 u hu
    have hc : inertWorkDev d (sem.unitInertWorkDev d H s' U u act) ≤ ρ * chainCreditDev d (U.layout.shape u) := hcap
    show (wrefDevK d c (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditDev d (U.layout.shape u) - inertWorkDev d (sem.unitInertWorkDev d H s' U u act) : ℚ) : ℝ)
    rw [hsh] at hc ⊢
    rw [hω]
    have : ω * (creditDev d s - ρ * chainCreditDev d s) ≤
        ω * (creditDev d s - inertWorkDev d (sem.unitInertWorkDev d H s' U u act)) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast this
  have key := gapFromTTOut CM (pearlCProtocolDevChainInertBoundK d sem ρ c) (pearlCDomainDevAt d s) (1 / 400) (ω : ℝ)
    δPearlC (by exact_mod_cast hω0) hW (ttOut_mono_domain (fun _ h => h.1) (ttOut_wref _ hTT))
  have e : (g : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hg]; push_cast; ring
  rw [e]
  exact key

theorem pearlCSampledDevChainInertBoundKAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hw0 : 0 ≤ wrefDevK d c s) (hω : wrefDevK d c s = ω * ((1 - ρ) * creditDev d s))
    (hγ : γ = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevChainInertBound CM d sem ρ) :
    GGapSampled CM (pearlCProtocolDevChainInertBoundK d sem ρ c) (pearlCTilesDevChainInertBoundK d sem ρ c) (pearlCDomainDevAt d s)
      (γ : ℝ) δPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevChainInertBoundK d sem ρ c) (pearlCDomainDevAt d s) →
      ∀ (H : Q → R) (s' : S), ∀ g < ((pearlCTilesDevChainInertBoundK d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevChainInertBoundK d sem ρ c).inertBoundOK H s' U g act →
        (pearlCTilesDevChainInertBoundK d sem ρ c).Wcred U.layout g act ≤
          (ω : ℝ) * (pearlCTilesDevChainInertBoundK d sem ρ c).credit H s' U g act := by
    intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevChainInertBoundK, pearlCTilesDevChainInertBound, pearlCTilesDev]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  have hWc : ∀ U : Workload, U.InDomain (pearlCProtocolDevChainInertBoundK d sem ρ c) (pearlCDomainDevAt d s) →
      ∀ g < ((pearlCTilesDevChainInertBoundK d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevChainInertBoundK d sem ρ c).Wcred U.layout g act ≤
          (pearlCTilesDevChainInertBoundK d sem ρ c).Wref U.layout g := by
    intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevChainInertBoundK, pearlCTilesDevChainInertBound, pearlCTilesDev]
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) hw0
  have key := gapSampled CM (pearlCProtocolDevChainInertBoundK d sem ρ c) (pearlCTilesDevChainInertBoundK d sem ρ c)
    (pearlCDomainDevAt d s) (1 / 400) (ω : ℝ) δPearlC (by norm_num) (by exact_mod_cast hω0) hW hWc
    (ttOutTile_mono_domain (fun _ h => h.1) (ttOutTile_wref _ _ _ hTT))
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

section Instances

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

theorem pearlCGapSm120v2LoopCast8ChainInertShare1000_8192
    (hTT : TTOutPearlCDevChainInertBound CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainInertBoundK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((754963 / 209826175 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevChainInertBoundKAt CM _ sem _ _ _ (8393047 / 8383808) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev, chainCreditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCGapSm120v2LoopCast8ChainInertShare1000_16384
    (hTT : TTOutPearlCDevChainInertBound CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevChainInertBoundK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((490417 / 138208725 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevChainInertBoundKAt CM _ sem _ _ _ (16585047 / 16567616) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev, chainCreditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh16384])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast8ChainInertShare1000_8192
    (hTT : TTOutTilePearlCDevChainInertBound CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainInertBoundK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCTilesDevChainInertBoundK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh8192) ((1519901 / 419652350 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevChainInertBoundKAt CM _ sem _ _ _ (8393047 / 8383608) _ (by norm_num)
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by rw [wrefDevK, wrefDev, creditDev,
          show (devSm120v2 Prices.sm120Loop).G = 0 from rfl,
          show (devSm120v2 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
        norm_num [Prices.sm120Loop, Params.pi, sh8192])
    (by norm_num) hTT

theorem pearlCSampledSm120v2LoopCast8ChainInertShare1000_16384
    (hTT : TTOutTilePearlCDevChainInertBound CM (devSm120v2 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevChainInertBoundK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCTilesDevChainInertBoundK (devSm120v2 Prices.sm120Loop) sem (1 / 1000) (8 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v2 Prices.sm120Loop) sh16384) ((109351 / 30713050 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevChainInertBoundKAt CM _ sem _ _ _ (614261 / 613608) _ (by norm_num)
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
