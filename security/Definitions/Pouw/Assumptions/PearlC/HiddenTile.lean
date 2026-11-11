module

public import Definitions.Pouw.PearlC.Hidden

/-!
# The hidden audit's open obligation, a sampled tile's proof is sound (trusted; a `Prop` taken as a hypothesis)

The hidden audit (`Hidden.lean`) accepts a sampled tile on its proof's verdict. `TileProofSoundAll` says that a prover in
the proof system's class at budget `q` gets a verdict of accept on a tile that is neither good nor blank with
probability at most `δzk(q)` per sample. `HiddenGap` turns it and `GGapSampled` into `GGapHidden`.

It is an open obligation, not a hardness assumption, and lives here only because a `Prop` taken as a hypothesis goes in
the assumptions module. It is to be discharged from Flock's soundness: its single-game theorem
`flock_e2e_drawn_hm96_reads` through core's batch lemma (`Verity.Guarantees.ProbBatchInduced`), the binding of
hm96-SHA-512 commitments to `τ`'s rows under the opening, the tile-check Program computing the good-tile predicate (L3)
and the pinned anchors (L8). `tileProofSoundAll_of_tileGame` (`HiddenTileGame`) takes the batch lemma's part: it proves
`TileProofSoundAll` from a `TileGame` per tile, so the rest is Flock's `TileGame`. What remains assumed under it is only
Flock's per-finder `cr/sha-512` and `ecr/sha-512`, and the live coins.
-/

@[expose] public section

open Claims renaming Ex → ex

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

open Classical in
/-- **One prover: a sampled tile's proof is accepted only on a good or blank tile, up to `δzk`.** For the transcripts `τ`
committed under the openings `op`, the prover `σ` of the proof phase and each sample `i`: averaged over the oracle, the
salt and the sample, the probability over the proof games' coins that the verdict on sample `i` is accept while tile `c i`
is neither good nor blank in `τ` is at most `δzk`. -/
def TileProofSound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (HA : HiddenAudit Q R S) (U : Workload) (τ : (Q → R) → S → Transcript) (op : (Q → R) → S → HA.Opening)
    {w : ℕ → ℕ} {t : ℕ} (σ : HA.Prover TR U τ op w t) (δzk : ℝ) : Prop :=
  ∀ i : Fin t, ex (fun x : (Q → R) × S × (Fin t → Ticket w (TR.tiling U.layout).NT) =>
      if ((x.2.2 i).1 : ℕ) ∈ goodTiles P TR U x.1 x.2.1 (τ x.1 x.2.1) ∪ blankTiles TR U (τ x.1 x.2.1) then 0 else
        (Verity.Game.prob (fun v => v.getD i false = true)
          (HA.batch x.1 x.2.1 U (HA.root U (τ x.1 x.2.1) (op x.1 x.2.1)) x.2.2) (σ x)).toReal) ≤ δzk

/-- **Open obligation, to be discharged from Flock's soundness; not a hardness assumption: every prover in the proof
system's class is sound per tile, up to `δzk(q)`.** For every in-domain workload, every transcripts `τ` and openings
`op` of their commitment, ticket count `w` and `t`, every budget `q` and every prover of the proof phase in the class at
`q` on `τ` under `op` (`HiddenAudit.Bounded`): `TileProofSound` at `δzk q`.

Until its discharge lands it is cited as an open obligation, never as an assumption. It bundles Flock's knowledge
soundness (an accepted proof's committed rows satisfy the tile-check Program), hm96-SHA-512's binding (those rows are
`τ`'s, under `op`), the tile-check Program computing the good-tile predicate (L3) and the pinned anchors (L8). Of these
only Flock's per-finder `cr/sha-512` and `ecr/sha-512` are assumptions; the rest is to be proved. It holds at a
degenerate hidden audit whose class is empty (`Bounded := fun _ _ _ _ _ => False`) for any `δzk ≥ 0`, so it says
something about Flock only at Flock's `HiddenAudit`. -/
def TileProofSoundAll [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) (HA : HiddenAudit Q R S) (δzk : ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (τ : (Q → R) → S → Transcript) (op : (Q → R) → S → HA.Opening) (w : ℕ → ℕ)
    (t q : ℕ) (σ : HA.Prover TR U τ op w t), HA.ProverBounded σ q → TileProofSound P TR HA U τ op σ (δzk q)

end Pouw.PearlC.Assumptions
