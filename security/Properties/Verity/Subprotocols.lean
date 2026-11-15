module

public import Properties.Verity.Game

/-!
# Each subprotocol's bound at its compiled component, and soundness from those bounds

`Vocab.ComponentBound k a` is what subprotocol `a`'s owner proves of its compiled component at composite `k` in the
causal game: the probability that `a`'s component at `k` accepts its own case in what the prover sent while `a`'s
claim fails is at most `a`'s error at the strategy (`Vocab.error`: its published `δ`, the published `ε_forge(t, Q)` and
its collision term, where its finder meets `FinderCR` and its forger `Unforgeable`). It is `Vocab.Target k a` without
the commit phase's file check and `k`'s acceptance as a `same` composition, so each target follows from it, one family
at every composite and subprotocol (`Vocab.sound_of`, in `Proofs.Properties.Verity.Subprotocols`; at remote's three,
`Vocab.spotSound_of`, `pouwSound_of` and `inferenceOnlySound_of`).

`SoundnessOfComponents` is `SoundnessOf` from those bounds: at every composite, each of its subprotocols' bound at its
component gives `Vocab.SoundnessAt`, with the errors summed over the composite's subprotocols (`subprotocols k`). A
composite with one more subprotocol takes one more bound in that sum, under the same statement.

`Vocab.TargetFor acc k a` is `a`'s target at `k` with the verdict `acc` in place of `Vocab.Verdict k a` (`Target k a`
is `TargetFor (Verdict k a) k a`), so the nontrivial checks can state that the target fails for a verifier that accepts
everything.
-/

@[expose] public section

namespace Verity

open Accounting Core
open scoped ENNReal

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **Subprotocol `a`'s bound at its compiled component at composite `k`**: at all `θ`, `F` and `c`, and for every
strategy, the probability that `a`'s component at `k` accepts its own case in what the prover sent while `a`'s claim
fails is at most `a`'s error at that strategy. -/
def ComponentBound (k : Verity.Registration.Composite) (a : Subprotocol) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => ComponentAccepts k a o.sent ∧ ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ ≤ V.error a θ F c σ

/-- **Subprotocol `a`'s target at composite `k` at the verdict `acc`**: at all `θ`, `F` and `c`, and for every
strategy, the probability that `acc` holds while `a`'s claim fails is at most `a`'s error at that strategy. -/
def TargetFor (acc : Exchange → Prop) (k : Verity.Registration.Composite) (a : Subprotocol) : Prop :=
  ∀ (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
    Game.prob (fun o => acc o ∧ ¬ V.Holds k θ F c σ a o) (V.game θ F c) σ ≤ V.error a θ F c σ

end Vocab

/-- **Soundness at every composite's verifier, from its subprotocols' bounds at their components.** On any
device, for any names `V` the subprotocols define and any composite `k`, if each of `k`'s subprotocols meets its bound
at its compiled component at `k` in the causal game, then at all parameters, committed files and commitments and for
every strategy, the probability that `k`'s compiled verifier accepts files that passed the commit phase's check while
the claim of one of `k`'s subprotocols fails is at most their errors at that strategy summed. -/
def SoundnessOfComponents : Prop :=
  ∀ (ι α : Type) [Fintype α] (d : Device ι) (V : Vocab d α) (k : Verity.Registration.Composite),
    (∀ a ∈ subprotocols k, V.ComponentBound k a) → V.SoundnessAt k

end Verity
