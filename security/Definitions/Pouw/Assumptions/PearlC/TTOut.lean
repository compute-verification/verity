module

public import Definitions.Pouw.PearlC.Device

/-! The named assumption TT_OUT, output tickets force the noisy chain, as `Prop`s the statements take as hypotheses. -/

@[expose] public section

namespace Pouw.PearlC.Assumptions

open Finset

variable {Q R S : Type}

/-- **TT_OUT(g₀), output tickets force the noisy chain**, in the game of `Pouw.PearlC.GGap`; `credit(u)` is the
reference's value on the unit's committed activations and the realized noise. -/
def TTOut [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (g₀ : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - g₀) < ∑ u ∈ correctSet P U x.1 x.2 (A.trace U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.trace U x.1 x.2) u))
      ≤ δ q U.layout.N

/-- **The named assumption TT_OUT(1/400) for Pearl-C. It is falsified** (rating D): the first promotion of every word
adds into a +0 total, an exact identity any program may skip, yet `creditOf` and `wrefOf` price it; rev1 is its repair.
It carries cross-group joint skips, which no replay count bounds; `CM` charges a tensor-core instruction its whole fragment. -/
def TTOutPearlC [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (sem : PearlCSem Q R S)
    (c : Costs) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocol sem c ρ) (pearlCDomain Params.pi) (1 / 400) δPearlC

/-- **TT_OUT(g₀) per tile**: as `TTOut`, with the tiles of `TR` in place of the units. -/
def TTOutTile [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (g₀ : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - g₀) < ∑ g ∈ goodTiles P TR U x.1 x.2 (A.trace U x.1 x.2),
        TR.credit x.1 x.2 U g (actOf (A.trace U x.1 x.2) ((TR.tiling U.layout).unit g)))
      ≤ δ q U.layout.N

/-- **TT_OUT(1/400) for Pearl-C per audit tile**, falsified as `TTOutPearlC` is (rating D), with rev1 per tile its
repair. It credits good tiles of partial units at a fixed `(1 − ρ)·creditOf` share, so per tile the claim is weaker. -/
def TTOutTilePearlC [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (c : Costs) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocol sem c ρ) (pearlCTiles sem c ρ) (pearlCDomain Params.pi) (1 / 400) δPearlC

/-- **TT_OUT(1/400) for Pearl-C at a device record** `d`, a separate named assumption at each record. At `devH100`
`TTOutPearlC` gives it (`ttOutPearlCDev_h100`); `CM` charges a tensor-core instruction its whole fragment shape, never
per output word. -/
def TTOutPearlCDev [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : PearlCDevice)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOut CM (pearlCProtocolDev d sem ρ) (pearlCDomainDev d) (1 / 400) δPearlC

/-- At `devH100` `TTOutTilePearlC` gives it (`ttOutTilePearlCDev_h100`). -/
def TTOutTilePearlCDev [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : PearlCDevice)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutTile CM (pearlCProtocolDev d sem ρ) (pearlCTilesDev d sem ρ) (pearlCDomainDev d) (1 / 400) δPearlC

end Pouw.PearlC.Assumptions
