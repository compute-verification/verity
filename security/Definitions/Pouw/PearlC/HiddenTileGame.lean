module

public import Definitions.Pouw.PearlC.Hidden
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! What a proof system gives per tile, its class and its bound, toward the hidden audit's open obligation `TileProofSoundAll`. -/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **The hidden audit's class from a single-game class.** The batch prover `σ` of `G`, on the transcript `τ` committed
under the opening `o`, is in the class at `q` when, wherever `G` is a batch `pre ++ g :: post`, every strategy that `σ`
induces on `g` (`Verity.Game.InducedAt`: `σ`'s moves in `g`, with the other games' coins fixed at some values) is in
`Cls τ o` at `q`. For Flock, `Cls τ o g s q` is the hypotheses of `flock_e2e_sampled_hm96_reads` at `q` for the strategy
`s` of `g`, its audit relabelled to the verdict. -/
def BoundedByInduced {O : Type} (Cls : Trace → O → (g : Verity.Game Bool) → Verity.Game.Strategy g → ℕ → Prop)
    (τ : Trace) (o : O) (G : Verity.Game (List Bool)) (σ : Verity.Game.Strategy G) (q : ℕ) : Prop :=
  ∀ (pre : List (Verity.Game Bool)) (g : Verity.Game Bool) (post : List (Verity.Game Bool))
    (e : Verity.Game.batch (pre ++ g :: post) = G) (s : Verity.Game.Strategy g),
    Verity.Game.InducedAt g post pre (e ▸ σ) s → Cls τ o g s q

/-- An empty `Cls` makes it trivial, so a `TileGame` says something about a proof system only when `Cls` is that system's
own class of single-game strategies, stated without reading which tiles are good, beside the theorem that the honest
prover's strategies are in it. -/
structure TileGame [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (HA : HiddenAudit Q R S)
    (Cls : Trace → HA.Opening → (g : Verity.Game Bool) → Verity.Game.Strategy g → ℕ → Prop) (δzk : ℕ → ℝ)
    (U : Workload) (τ : Trace) (o : HA.Opening) (H : Q → R) (s : S) (g : ℕ) where
  Out : Type
  G : Verity.Game Out
  verdict : Out → Bool
  game_eq : HA.game H s U (HA.root U τ o) g = G.map verdict
  Bad : Verity.Game.Strategy G → Out → Prop
  bad_le : ∀ (q : ℕ) (σ : Verity.Game.Strategy G),
    Cls τ o _ (cast (congrArg Verity.Game.Strategy game_eq.symm) (Verity.Game.Strategy.toMap verdict G σ)) q →
    (Verity.Game.prob (fun x => verdict x = true ∧ Bad σ x) G σ).toReal ≤ δzk q
  good_of_accept : ∀ (σ : Verity.Game.Strategy G) (x : Out), verdict x = true → ¬ Bad σ x →
    g ∈ goodTiles P TR U H s τ ∪ blankTiles TR U τ

end Pouw.PearlC
