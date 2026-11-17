module

public import Definitions.Pouw.PearlC.Tile

/-! `HonestTileInertBoundRate`, completeness's one assumption, empirical and not a hardness assumption: honest sampled tiles exceed the cap at a measured rate at most `ε`. -/

@[expose] public section

namespace Pouw.PearlC.Assumptions

open Pouw.PearlC

variable {Q R S : Type}

/-- **An honest sampled tile exceeds the cap at most `ε` of the time**: empirical, an upper confidence limit on a measured
rate per input `(U, act)`, not a hardness assumption. Every member of `Served` is assumed within `ε`, not only the
measured ones, with its activations fixed independently of the oracle and the salt. -/
def HonestTileInertBoundRate [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) (Served : Workload → (ℕ → Codes) → Prop) (ε : ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → U.FixedOK P → ∀ act : ℕ → Codes, Served U act →
    ∀ (w : ℕ → ℕ) (κ : ℝ), 0 < κ → (∀ g < (TR.tiling U.layout).NT, (w g : ℝ) = κ * TR.Wref U.layout g) →
      pr (fun x : (Q → R) × S × Ticket w (TR.tiling U.layout).NT =>
          (TR.tiling U.layout).unit x.2.2.1 < U.layout.N ∧
            ¬ TR.inertBoundOK x.1 x.2.1 U x.2.2.1 (act ((TR.tiling U.layout).unit x.2.2.1)))
        ≤ ε

end Pouw.PearlC.Assumptions
