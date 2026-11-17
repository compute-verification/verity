module

public import Definitions.Pouw.PearlC.DeviceRev1
public import Definitions.Pouw.PearlC.W1

/-! TT_OUT with its cost model the term `W1.w1Priced dv salt`, as named assumptions that speak about the deployment only at its noise `NoiseSha512.sha512Sem base salt rootB` with the cost model's `salt`. -/

@[expose] public section

namespace Pouw.PearlC.Assumptions

open Finset

variable {Q R S : Type}

/-- **TT_OUT(g₀) at W1**: `TTOut`'s game at `CM := W1.w1Priced dv salt`. With `K` the units whose ticket and useful
words are correct and whose debit is within the cap, no W1 program bounded by `T` makes
`∑_{u ∈ K} credit(u) > T / (1 − g₀)` with probability over `(H, s)` more than `δ(q, N)`. -/
def TTOutW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (P : Protocol Q R S) (D : Layout → Prop) (g₀ : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary (W1.w1Priced dv salt : CostModel Q R S), A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - g₀) < ∑ u ∈ correctSet P U x.1 x.2 (A.trace U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.trace U x.1 x.2) u))
      ≤ δ q U.layout.N

/-- **TT_OUT(g₀) per tile at W1**: `TTOutTile`'s game at `CM := W1.w1Priced dv salt`, the good tiles of `TR` in place of
the correct units. -/
def TTOutTileW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (g₀ : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary (W1.w1Priced dv salt : CostModel Q R S), A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - g₀) < ∑ g ∈ goodTiles P TR U x.1 x.2 (A.trace U x.1 x.2),
        TR.credit x.1 x.2 U g (actOf (A.trace U x.1 x.2) ((TR.tiling U.layout).unit g)))
      ≤ δ q U.layout.N

def TTOutPearlCDevW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutW1 dv salt (pearlCProtocolDev d sem ρ) (pearlCDomainDev d) (1 / 400) δPearlC

def TTOutTilePearlCDevW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileW1 dv salt (pearlCProtocolDev d sem ρ) (pearlCTilesDev d sem ρ) (pearlCDomainDev d) (1 / 400) δPearlC

/-- **TT_OUT(1/400) rev1 at a device record `d`, at W1 on `dv`**: `TTOutPearlCDevRev1` (every promotion into a +0 total
out of the credit) at `W1.w1Priced dv salt`. -/
def TTOutPearlCDevRev1W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutW1 dv salt (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400) δPearlC

/-- **TT_OUT(1/400) rev1 per audit tile at a device record `d`, at W1 on `dv`**: `TTOutTilePearlCDevRev1` at
`W1.w1Priced dv salt`. -/
def TTOutTilePearlCDevRev1W1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileW1 dv salt (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400)
    δPearlC

end Pouw.PearlC.Assumptions
