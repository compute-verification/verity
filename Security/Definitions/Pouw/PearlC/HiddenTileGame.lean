import Definitions.Pouw.PearlC.Hidden

/-!
# What a proof system gives per tile, toward the hidden audit's open obligation (trusted)

`TileProofSoundAll` (`HiddenTile`) is the hidden audit's open obligation. A proof system meets it with two things,
stated here for any hidden audit:

* **Its class, per tile** (`BoundedByInduced`): the hidden audit's class `HiddenAudit.Bounded` is the batch provers
  whose every induced strategy on one tile's proof game (core's `Verity.Game.InducedAt`) is in the proof system's
  single-game class `Cls`, on the transcript and opening it proves against. For C-Flock, `Cls` is the hypotheses of
  `flock_e2e_drawn_hm96_reads` at `q`.
* **Its bound, per tile** (`TileGame`): one tile's proof game is a game of the proof system's, relabelled to its
  verdict, with an event whose probability is at most `ηzk(q)` for every strategy in `Cls`, outside which an accepted
  run happens only on a good or blank tile. For C-Flock that bound is `flock_e2e_drawn_hm96_reads` and the last
  part is the hm96-binding step.

`tileProofSoundAll_of_tileGame` proves `TileProofSoundAll` from the two, through core's batch lemma
(`Verity.Guarantees.ProbBatchInduced`).
-/

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **The hidden audit's class from a single-game class.** The batch prover `σ` of `G`, on the transcript `τ` committed
under the opening `o`, is in the class at `q` when, wherever `G` is a batch `pre ++ g :: post`, every strategy that `σ`
induces on `g` (`Verity.Game.InducedAt`: `σ`'s moves in `g`, with the other games' coins fixed at some values) is in
`Cls τ o` at `q`. For C-Flock, `Cls τ o g s q` is the hypotheses of `flock_e2e_drawn_hm96_reads` at `q` for the strategy
`s` of `g`, its audit relabelled to the verdict. -/
def BoundedByInduced {O : Type} (Cls : Transcript → O → (g : Verity.Game Bool) → Verity.Game.Strategy g → ℕ → Prop)
    (τ : Transcript) (o : O) (G : Verity.Game (List Bool)) (σ : Verity.Game.Strategy G) (q : ℕ) : Prop :=
  ∀ (pre : List (Verity.Game Bool)) (g : Verity.Game Bool) (post : List (Verity.Game Bool))
    (e : Verity.Game.batch (pre ++ g :: post) = G) (s : Verity.Game.Strategy g),
    Verity.Game.InducedAt g post pre (e ▸ σ) s → Cls τ o g s q

end Pouw.PearlC
