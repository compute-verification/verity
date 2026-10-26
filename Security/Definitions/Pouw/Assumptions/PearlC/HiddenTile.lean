import Definitions.Pouw.PearlC.Hidden

/-!
# The hidden audit's open obligation, a drawn tile's proof is sound (trusted; a `Prop` taken as a hypothesis)

The hidden audit (`Hidden.lean`) accepts a drawn tile on its proof's verdict. `TileProofSoundAll` says that a prover in
the proof system's class at budget `q` gets a verdict of accept on a tile that is neither good nor blank with
probability at most `ηzk(q)` per draw. `HiddenGamma` turns it and `GγSampled` into `GγHidden`.

It is an open obligation, not a hardness assumption, and lives here only because a `Prop` taken as a hypothesis goes in
the assumptions module. It is to be discharged from C-Flock's soundness: its single-game theorem
`flock_e2e_drawn_hm96_reads` through core's batch lemma (`Verity.Guarantees.ProbBatchInduced`), the binding of
hm96-SHA-512 commitments to `τ`'s rows under the opening, the tile-check Program computing the good-tile predicate (L3)
and the pinned anchors (L8). `tileProofSoundAll_of_tileGame` (`HiddenTileGame`) takes the batch lemma's part: it proves
`TileProofSoundAll` from a `TileGame` per tile, so the rest is C-Flock's `TileGame`. What remains assumed under it is only
C-Flock's per-finder `cr/sha-512` and `ecr/sha-512`, and the live coins.
-/

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

open Classical in
/-- **One prover: a drawn tile's proof is accepted only on a good or blank tile, up to `ηzk`.** For the transcripts `τ`
committed under the openings `op`, the prover `σ` of the proof phase and each draw `i`: averaged over the oracle, the
salt and the draw, the probability over the proof games' coins that the verdict on draw `i` is accept while tile `c i`
is neither good nor blank in `τ` is at most `ηzk`. -/
def TileProofSound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (HA : HiddenAudit Q R S) (U : Workload) (τ : (Q → R) → S → Transcript) (op : (Q → R) → S → HA.Opening)
    {w : ℕ → ℕ} {t : ℕ} (σ : HA.Prover TR U τ op w t) (ηzk : ℝ) : Prop :=
  ∀ i : Fin t, ex (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
      if ((x.2.2 i).1 : ℕ) ∈ goodTiles P TR U x.1 x.2.1 (τ x.1 x.2.1) ∪ blankTiles TR U (τ x.1 x.2.1) then 0 else
        (Verity.Game.prob (fun v => v.getD i false = true)
          (HA.batch x.1 x.2.1 U (HA.root U (τ x.1 x.2.1) (op x.1 x.2.1)) x.2.2) (σ x)).toReal) ≤ ηzk

/-- **Open obligation, to be discharged from C-Flock's soundness; not a hardness assumption: every prover in the proof
system's class is sound per tile, up to `ηzk(q)`.** For every in-domain workload, every transcripts `τ` and openings
`op` of their commitment, ticket count `w` and `t`, every budget `q` and every prover of the proof phase in the class at
`q` on `τ` under `op` (`HiddenAudit.Bounded`): `TileProofSound` at `ηzk q`.

Until its discharge lands it is cited as an open obligation, never as an assumption. It bundles C-Flock's knowledge
soundness (an accepted proof's committed rows satisfy the tile-check Program), hm96-SHA-512's binding (those rows are
`τ`'s, under `op`), the tile-check Program computing the good-tile predicate (L3) and the pinned anchors (L8). Of these
only C-Flock's per-finder `cr/sha-512` and `ecr/sha-512` are assumptions; the rest is to be proved. It holds at a
degenerate hidden audit whose class is empty (`Bounded := fun _ _ _ _ _ => False`) for any `ηzk ≥ 0`, so it says
something about C-Flock only at C-Flock's `HiddenAudit`. -/
def TileProofSoundAll [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) (HA : HiddenAudit Q R S) (ηzk : ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (τ : (Q → R) → S → Transcript) (op : (Q → R) → S → HA.Opening) (w : ℕ → ℕ)
    (t q : ℕ) (σ : HA.Prover TR U τ op w t), HA.ProverBounded σ q → TileProofSound P TR HA U τ op σ (ηzk q)

end Pouw.PearlC.Assumptions
