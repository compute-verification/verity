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

/-- **What the proof system gives for one tile.** For the workload `U`, the transcript `τ` committed under the opening
`o`, the oracle `H`, the salt `s` and the tile `g`:

* `game_eq`: the hidden audit's proof game is the proof system's game `G` with its output relabelled to the verdict;
* `bad_le`: for every strategy of `G` whose relabelling is in `Cls τ o` at `q`, the verdict is accept on a run in the
  event `Bad` with probability at most `ηzk q` (for C-Flock, `flock_e2e_drawn_hm96_reads`: `Bad` is a drawn unit wrong
  under the extracted witness, or a finder's success);
* `good_of_accept`: a run that ends in accept outside `Bad` happens only when tile `g` is good or blank in `τ` (for
  C-Flock, the hm96-binding step: the rows the proof speaks of are `τ`'s under `o`, with L3 and L8).

An empty `Cls` makes a `TileGame` trivial to give (`Bad` everything), as an empty class makes `TileProofSoundAll`
vacuous (`HiddenAudit`'s docstring). So a `TileGame` says something about a proof system only when `Cls` is the proof
system's own class of single-game strategies, stated without reading which of `τ`'s tiles are good (C-Flock's: every
collision finder built from the strategy meets `cr/sha-512` or `ecr/sha-512` at `q`), and only beside the theorem that
the honest prover's induced strategies are in it. Holding the honest prover is not enough: the strategies of games that
are no bad tile's proof game hold it, and give a `TileGame` for every tile at `ηzk = 0` with `Bad` the tile being bad,
and so `GγHidden` from `GγSampled` with no proof system. -/
structure TileGame [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (HA : HiddenAudit Q R S)
    (Cls : Transcript → HA.Opening → (g : Verity.Game Bool) → Verity.Game.Strategy g → ℕ → Prop) (ηzk : ℕ → ℝ)
    (U : Workload) (τ : Transcript) (o : HA.Opening) (H : Q → R) (s : S) (g : ℕ) where
  Out : Type
  G : Verity.Game Out
  verdict : Out → Bool
  game_eq : HA.game H s U (HA.root U τ o) g = G.map verdict
  Bad : Verity.Game.Strategy G → Out → Prop
  bad_le : ∀ (q : ℕ) (σ : Verity.Game.Strategy G),
    Cls τ o _ (cast (congrArg Verity.Game.Strategy game_eq.symm) (Verity.Game.Strategy.toMap verdict G σ)) q →
    (Verity.Game.prob (fun x => verdict x = true ∧ Bad σ x) G σ).toReal ≤ ηzk q
  good_of_accept : ∀ (σ : Verity.Game.Strategy G) (x : Out), verdict x = true → ¬ Bad σ x →
    g ∈ goodTiles P TR U H s τ ∪ blankTiles TR U τ

end Pouw.PearlC
