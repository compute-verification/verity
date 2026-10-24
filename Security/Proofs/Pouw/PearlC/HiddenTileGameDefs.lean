import Definitions.Pouw.PearlC.HiddenTileGame

/-! The proof-only part of `Definitions.Pouw.PearlC.HiddenTileGame`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.PearlC

variable {Q R S : Type}

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
and so `GγHidden` from `GγSampled` with no proof system (the red team's F2 on #919). -/
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
