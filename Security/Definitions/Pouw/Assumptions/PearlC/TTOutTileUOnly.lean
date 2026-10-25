import Definitions.Pouw.Assumptions.PearlC.TTOutUOnly

/-!
# TT_OUT over U alone, per audit tile (trusted; the named `Prop`s)

The per-tile twin of `TTOutUOnly`: a tile is good under U-only binding when its debit is within the cap and the useful
words of its credited rows are right (`TileGoodU`, `TileGood` without its ticket conjunct). `TTOutTileU` is `TTOutTile`
over `goodTilesU`, and `TTOutTilePearlCDevUOnly` is it at rev1's protocol and tiles.

It asserts **more** than the (C̃, U) tile row: every tile good under both bindings is good under U alone
(`TileUOnlyGamma.ttOutTilePearlCDevRev1_of_uOnly`). It is a new conjecture, not a weakening.
-/

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **TT_OUT(γ₀) per audit tile over U alone**: `TTOutTile` with the good tiles under U-only binding. -/
def TTOutTileU [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (γ₀ : ℝ) (ε : ℕ → ℕ → ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - γ₀) < ∑ g ∈ goodTilesU P TR U x.1 x.2 (A.transcript U x.1 x.2),
        TR.credit x.1 x.2 U g (actOf (A.transcript U x.1 x.2) ((TR.tiling U.layout).unit g)))
      ≤ ε q U.layout.N

/-- **TT_OUT(1/400) per audit tile over U alone at a device record `d`**: `TTOutTileU` at `pearlCProtocolDevRev1 d sem ρ`,
`pearlCTilesDevRev1 d sem ρ`, `pearlCDomainDev d`, `γ₀ = 1/400` and `εPearlC`. It is `TTOutTilePearlCDevRev1` with the
ticket words unchecked, and `TTOutPearlCDevUOnly` per tile: what it adds to the (C̃, U) tile row, its falsifiers and the
evidence are `TTOutPearlCDevUOnly`'s, read on the tile's credited rows.

- **The cost model** is `TTOutPearlCDevRev1`'s: a tensor-core instruction is charged its whole fragment shape, never per
  output word. -/
def TTOutTilePearlCDevUOnly [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTileU CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400) εPearlC

end Pouw.PearlC.Assumptions
