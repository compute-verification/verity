module

public import Definitions.Pouw.PearlC.DeviceFp4

/-! Pearl-C4 v2's accounting at an FP4 record, staged: only what its price twins read. -/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

/-- `a` per word, in FP4 units. -/
def creditFp4Hot (d : Fp4Device) (a : ℚ) (s : Shape) : ℚ := creditFp4 d s + a * s.m * s.n

def wrefFp4Hot (d : Fp4Device) (a : ℚ) (s : Shape) : ℚ := creditFp4Hot d a s + d.prices.qa * s.m * s.k

def creditFp4HotChainOnly (d : Fp4Device) (a : ℚ) (s : Shape) : ℚ := creditFp4Hot d a s - d.prices.fs * s.m * s.k

def gapFp4Hot (d : Fp4Device) (a ρ : ℚ) (s : Shape) : ℚ :=
  1 - (1 - 1 / 400) / (wrefFp4Hot d a s / ((1 - ρ) * creditFp4Hot d a s))

def gapFp4HotChainOnly (d : Fp4Device) (a ρ : ℚ) (s : Shape) : ℚ :=
  1 - (1 - 1 / 400) / (wrefFp4Hot d a s / (creditFp4HotChainOnly d a s - ρ * creditFp4Hot d a s))

def Fp4HotUnitAccounting (P : Protocol Q R S) (D : Layout → Prop) (d : Fp4Device) (a K : ℚ) (s : Shape) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N, ∀ act : Codes,
    P.actOK (U.layout.shape u) act → P.inertBoundOK H s' U u act →
      P.Wref U.layout u = (wrefFp4Hot d a s : ℝ) ∧ (K : ℝ) ≤ P.credit H s' U u act

def Fp4HotTileAccounting (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (d : Fp4Device) (a K : ℚ)
    (s : Shape) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ g < (TR.tiling U.layout).NT, ∀ act : Codes, ∃ t : ℚ, 0 ≤ t ∧
    TR.Wcred U.layout g act = ((wrefFp4Hot d a s * t : ℚ) : ℝ) ∧ TR.Wcred U.layout g act ≤ TR.Wref U.layout g ∧
    ∀ (H : Q → R) (s' : S), TR.inertBoundOK H s' U g act → ((K * t : ℚ) : ℝ) ≤ TR.credit H s' U g act

end Pouw.PearlC
