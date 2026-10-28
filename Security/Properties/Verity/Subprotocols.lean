import Properties.Verity.Game

/-!
# Each subprotocol's bound at its compiled component

`Vocab.ComponentBound a` is what subprotocol `a`'s owner proves of its compiled component in the causal game: the
probability that `a`'s component accepts its own case in what the prover sent while `a`'s claim fails is at most `a`'s
error at the strategy (`Vocab.error`: its published `δ` plus its collision term, where its finder meets `FinderCR`).
It is `Vocab.Target a` without the commit phase's file check and NCI's acceptance of the registration, so each target
follows from it (`Vocab.pouwSound_of`, `pousSound_of`, `poubSound_of`, `networkSound_of`, in
`Proofs.Properties.Verity.Subprotocols`, with the period guarantees that give it).

`Vocab.TargetFor acc a` is `a`'s target with the verdict `acc` in place of `Vocab.Verdict a` (`Target a` is
`TargetFor (Verdict a) a`), so the nontrivial checks can state that the target fails for a verifier that accepts
everything.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **Subprotocol `a`'s bound at its compiled component**: at all `θ`, `F` and `c`, and for every strategy, the
probability that `a`'s component accepts its own case in what the prover sent while `a`'s claim fails is at most `a`'s
error at that strategy. -/
def ComponentBound (a : Audit) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => ComponentAccepts a o.sent ∧ ¬ V.Holds θ F c σ a o) (V.game θ F c) σ ≤ V.error a θ F c σ

/-- **Subprotocol `a`'s target at the verdict `acc`**: at all `θ`, `F` and `c`, and for every strategy, the
probability that `acc` holds while `a`'s claim fails is at most `a`'s error at that strategy. -/
def TargetFor (acc : Outcome → Prop) (a : Audit) : Prop :=
  ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => acc o ∧ ¬ V.Holds θ F c σ a o) (V.game θ F c) σ ≤ V.error a θ F c σ

end Vocab

end Verity
