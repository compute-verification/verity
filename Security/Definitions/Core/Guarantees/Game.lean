import Definitions.Core.Game

/-!
# How sound games compose

A protocol's guarantee is one `Sound` statement about its game, and these say how such statements combine, so a protocol
built from others states its guarantee from theirs:

* `SoundAnd`: two claims about one game, each sound, are sound together with the errors added;
* `SoundMono`: a stricter acceptance, a claim it implies, a larger error;
* `SoundInterleave`: two sound games run side by side, the developer scheduling them, are sound for both claims with the
  errors added;
* `SoundBatch`: a sound game stays sound for its own output when other games run beside it;
* `ProbBatchInduced`: the same for a class of strategies, such as the query-bounded ones a computational soundness bound
  speaks of: a bound for every strategy of the class holds for a batch strategy whose induced strategies on the game are
  in it;
* `SoundBind`: a game that continues, after anything, with a game sound with error `δ` is sound with error `δ`.
-/

namespace Verity.Guarantees

open scoped ENNReal

/-- Two claims about one game, sound with errors `δ` and `ε`, are sound together with error `δ + ε`. -/
def SoundAnd : Prop :=
  ∀ {α : Type} (g : Game α) (accept E F : α → Prop) (δ ε : ℝ≥0∞),
    Sound g accept E δ → Sound g accept F ε → Sound g accept (fun a => E a ∧ F a) (δ + ε)

/-- A game sound for `E` under `accept` is sound for any claim `F` that `E` implies under a stricter acceptance
`accept'`, with any larger error. -/
def SoundMono : Prop :=
  ∀ {α : Type} (g : Game α) (accept accept' E F : α → Prop) (δ δ' : ℝ≥0∞),
    Sound g accept E δ → (∀ a, accept' a → accept a) → (∀ a, accept' a → E a → F a) → δ ≤ δ' →
      Sound g accept' F δ'

/-- Two sound games run concurrently, the developer choosing at every step which one advances, are sound for the
conjunction of their claims when the verifier accepts both, with the errors added. -/
def SoundInterleave : Prop :=
  ∀ {α β : Type} (g₁ : Game α) (g₂ : Game β) (accept₁ E₁ : α → Prop) (accept₂ E₂ : β → Prop) (δ₁ δ₂ : ℝ≥0∞),
    Sound g₁ accept₁ E₁ δ₁ → Sound g₂ accept₂ E₂ δ₂ →
      Sound (g₁.interleave g₂) (fun x => accept₁ x.1 ∧ accept₂ x.2) (fun x => E₁ x.1 ∧ E₂ x.2) (δ₁ + δ₂)

/-- A game sound alone is sound, with the same error, for its own output in a batch of games run concurrently (`d` is
the default `getD` reads past the end, which `batch` never reaches). -/
def SoundBatch : Prop :=
  ∀ {α : Type} (pre post : List (Game α)) (g : Game α) (d : α) (accept E : α → Prop) (δ : ℝ≥0∞),
    Sound g accept E δ →
      Sound (Game.batch (pre ++ g :: post)) (fun outs => accept (outs.getD pre.length d))
        (fun outs => E (outs.getD pre.length d)) δ

/-- **A bound for a class of strategies holds in a batch.** If every strategy in the class `Cls` for `g` ends with an
output satisfying `Q` with probability at most `ε`, then a strategy `σ` for a batch with `g` at position `pre.length`
gives that position's output `Q` with probability at most `ε`, provided every strategy `σ` induces on `g` is in `Cls`
(`Game.InducedAt`: `σ`'s moves in `g`, with the other games' coins fixed at some values). -/
def ProbBatchInduced : Prop :=
  ∀ {α : Type} (pre post : List (Game α)) (g : Game α) (d : α) (Q : α → Prop) (Cls : Game.Strategy g → Prop)
    (ε : ℝ≥0∞), (∀ s, Cls s → Game.prob Q g s ≤ ε) →
      ∀ σ : Game.Strategy (Game.batch (pre ++ g :: post)), (∀ s, Game.InducedAt g post pre σ s → Cls s) →
        Game.prob (fun outs => Q (outs.getD pre.length d)) (Game.batch (pre ++ g :: post)) σ ≤ ε

/-- A game that, whatever its first part outputs, continues with a game sound with error `δ` is sound with error `δ`. -/
def SoundBind : Prop :=
  ∀ {α β : Type} (g : Game α) (f : α → Game β) (accept E : β → Prop) (δ : ℝ≥0∞),
    (∀ a, Sound (f a) accept E δ) → Sound (g.bind f) accept E δ

end Verity.Guarantees
