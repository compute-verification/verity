module

public import Definitions.Pouw.PearlC.Hidden

/-! `TileProofSoundAll`, the hidden audit's open obligation that a sampled tile's proof is sound, taken as a hypothesis and not a hardness assumption. -/

@[expose] public section

open Claims renaming Ex → ex

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

open Classical in
/-- **One prover: a sampled tile's proof is accepted only on a good or blank tile, up to `δzk`.** -/
def TileProofSound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (HA : HiddenAudit Q R S) (U : Workload) (τ : (Q → R) → S → Trace) (op : (Q → R) → S → HA.Opening)
    {w : ℕ → ℕ} {t : ℕ} (σ : HA.Prover TR U τ op w t) (δzk : ℝ) : Prop :=
  ∀ i : Fin t, ex (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
      if ((x.2.2 i).1 : ℕ) ∈ goodTiles P TR U x.1 x.2.1 (τ x.1 x.2.1) ∪ blankTiles TR U (τ x.1 x.2.1) then 0 else
        (Verity.Game.prob (fun v => v.getD i false = true)
          (HA.batch x.1 x.2.1 U (HA.root U (τ x.1 x.2.1) (op x.1 x.2.1)) x.2.2) (σ x)).toReal) ≤ δzk

/-- **Open obligation, to be discharged from Flock's soundness; not a hardness assumption.** It holds vacuously at a
hidden audit whose class is empty, so it says something about Flock only at Flock's `HiddenAudit`. -/
def TileProofSoundAll [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) (HA : HiddenAudit Q R S) (δzk : ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (τ : (Q → R) → S → Trace) (op : (Q → R) → S → HA.Opening) (w : ℕ → ℕ)
    (t q : ℕ) (σ : HA.Prover TR U τ op w t), HA.ProverBounded σ q → TileProofSound P TR HA U τ op σ (δzk q)

end Pouw.PearlC.Assumptions
