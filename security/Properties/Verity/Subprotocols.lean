module

public import Properties.Verity.Game

/-! Each subprotocol's bound at its compiled component (`Vocab.ComponentBound`), and soundness from those bounds
(`SoundnessOfComponents`). -/

@[expose] public section

namespace Verity

open ResourceAccounting Core
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
