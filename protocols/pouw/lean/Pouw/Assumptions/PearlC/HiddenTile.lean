import Pouw.Protocol.PearlC.Hidden

/-!
# A8 in PoUW's terms: a drawn tile's proof is sound (trusted; `Prop`s taken as hypotheses, never `axiom`s)

The hidden audit (`Hidden.lean`) accepts a drawn tile on its proof's verdict. `TileProofSoundAll` says that a prover in
the proof system's class at budget `q` gets a verdict of accept on a tile that is neither good nor blank with
probability at most `ηzk(q)` per draw. `HiddenGamma` turns it and `GγSampled` into `GγHidden`.

It is still to be discharged: from C-Flock's single-game theorem `flock_e2e_drawn_hm96_reads` through a batch lemma, the
binding of hm96-SHA-512 commitments to `τ`'s rows, the tile-check Program computing the good-tile predicate (L3) and the
pinned anchors (L8).
-/

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

open Classical in
/-- **A8 for one prover: a drawn tile's proof is accepted only on a good or blank tile, up to `ηzk`.** For the
transcripts `τ`, the prover `σ` of the proof phase and each draw `i`: averaged over the oracle, the salt and the draw,
the probability over the proof games' coins that the verdict on draw `i` is accept while tile `c i` is neither good nor
blank in `τ` is at most `ηzk`. -/
def TileProofSound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (HA : HiddenAudit Q R S) (U : Workload) (τ : (Q → R) → S → Transcript) {w : ℕ → ℕ} {t : ℕ}
    (σ : HA.Prover TR U τ w t) (ηzk : ℝ) : Prop :=
  ∀ i : Fin t, ex (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
      if ((x.2.2 i).1 : ℕ) ∈ goodTiles P TR U x.1 x.2.1 (τ x.1 x.2.1) ∪ blankTiles TR U (τ x.1 x.2.1) then 0 else
        (Verity.Game.prob (fun v => v.getD i false = true)
          (HA.batch x.1 x.2.1 U (HA.root U (τ x.1 x.2.1)) x.2.2) (σ x)).toReal) ≤ ηzk

/-- **A8 in PoUW's terms: every prover in the proof system's class is sound per tile, up to `ηzk(q)`.** For every
in-domain workload, every transcripts `τ`, ticket count `w` and `t`, every budget `q` and every prover of the proof phase
in the class at `q` (`HiddenAudit.Bounded`): `TileProofSound` at `ηzk q`. It carries C-Flock's knowledge soundness (an
accepted proof's committed rows satisfy the tile-check Program), hm96-SHA-512's binding (those rows are `τ`'s), the
tile-check Program computing the good-tile predicate (L3) and the pinned anchors (L8). -/
def TileProofSoundAll [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) (HA : HiddenAudit Q R S) (ηzk : ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (τ : (Q → R) → S → Transcript) (w : ℕ → ℕ) (t q : ℕ)
    (σ : HA.Prover TR U τ w t), HA.ProverBounded σ q → TileProofSound P TR HA U τ σ (ηzk q)

end Pouw.PearlC.Assumptions
