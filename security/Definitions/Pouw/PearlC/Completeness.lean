module

public import Definitions.Pouw.PearlC.Tile

/-! The sampled audit's completeness on a population `Served` of activations; none holds for every activation input, as activations aligned against the cap exceed it. -/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

def CompleteSampled [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) (Served : Workload → (ℕ → Codes) → Prop) (ε : ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → U.FixedOK P → ∀ act : ℕ → Codes, Served U act →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
    ∀ t : ℕ,
      pr (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
          ¬ AuditAccepts (goodTiles P TR U x.1 x.2.1 (honestTrace P x.1 x.2.1 U act) ∪
            blankTiles TR U (honestTrace P x.1 x.2.1 U act)) x.2.2)
        ≤ t * ε

end Pouw.PearlC
