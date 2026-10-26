import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.ENNReal.BigOperators
import Mathlib.Basic.ENNReal.Inv
import Mathlib.Order.CompleteLattice.Basic

/-!
# Public-coin interactions with live coins, and what it means for one to be sound

A `Game α` is the verifier's side of a public-coin interaction, written as a program:

* `send M k`: the developer (the prover) sends a message `m : M` of its choice, and the interaction continues as `k m`;
* `coin C k`: the verifier draws `c : C` uniformly at random, independently of everything before, and the interaction
  continues as `k c`. This is a live coin: it is drawn after every earlier message is fixed (in Verity, the coin server
  draws it from the operating system after recording the round's digest);
* `ret a`: the interaction ends with the verifier's output `a`.

Messages may be anything: a declaration, a commitment, whole tables of an interactive oracle proof. A deterministic
prover strategy (`Strategy g`) picks a message at every `send` node as a function of the history, which the subtree it
sits in encodes. `prob p g s` is the exact probability, over the verifier's uniform coins, that the output satisfies
`p`, and `value p g` is the same probability for an optimal, computationally unbounded prover. A randomized prover is a
mixture of deterministic ones, so every bound on `value` bounds it too. `avg` and `prCoin` are the uniform average and
probability over one coin, which every area's draw reads (`Definitions.Core.Draw`).

`interleave g₁ g₂` runs two interactions side by side, the prover choosing at every step which one advances, and
`batch` runs a list of them so. A strategy for a batch plays each of its games, and `InducedAt` names the strategies it
induces on one of them: the batch strategy's moves in that game, with the other games' coins fixed at some values. A
bound for a class of strategies of the game holds at its position in the batch for every batch strategy whose induced
strategies are in the class (`Guarantees.ProbBatchInduced`).

Every protocol's guarantee has one form (`Sound`): whatever the developer does, the probability that the verifier
accepts while the claim `E` about the committed world is false is at most `δ`. Protocols differ only in their game,
their acceptance and their claim. A policy is a check on the declaration inside `accept`, a game with no rounds of its
own.
-/

namespace Verity

open scoped ENNReal

/-- The verifier's side of a public-coin interaction with live coins. -/
inductive Game (α : Type) : Type 1 where
  | ret : α → Game α
  | send (M : Type) : (M → Game α) → Game α
  | coin (C : Type) [Fintype C] [Nonempty C] : (C → Game α) → Game α

namespace Game

variable {α β : Type}

/-- Sequencing: run `g`, then continue with `f` applied to its output. -/
def bind : Game α → (α → Game β) → Game β
  | .ret a, f => f a
  | .send M k, f => .send M fun m => bind (k m) f
  | @Game.coin _ C iF iN k, f => @Game.coin _ C iF iN fun c => bind (k c) f

instance instMonad : Monad Game where
  pure := .ret
  bind := bind

/-- The prover sends a message of type `M`. -/
def recv (M : Type) : Game M := .send M .ret

/-- The verifier draws a uniform coin from `C`. -/
def draw (C : Type) [Fintype C] [Nonempty C] : Game C := .coin C .ret

/-- Relabel the output. -/
def map (f : α → β) (g : Game α) : Game β := g.bind fun a => .ret (f a)

/-- A deterministic prover strategy: a message at every `send` node, for every history (the history is encoded by the
position in the tree, so a strategy is simply a choice at each node, and after a coin it may depend on that coin). -/
def Strategy : Game α → Type
  | .ret _ => PUnit
  | .send _ k => (m : _) × Strategy (k m)
  | @Game.coin _ C _ _ k => (c : C) → Strategy (k c)

open Classical in
/-- The exact probability, over the verifier's independent uniform coins, that the interaction between the verifier
`g` and the prover strategy `s` ends with an output satisfying `p`. -/
noncomputable def prob (p : α → Prop) : (g : Game α) → Strategy g → ℝ≥0∞
  | .ret a, _ => if p a then 1 else 0
  | .send _ k, s => prob p (k s.1) s.2
  | @Game.coin _ C iF _ k, s => ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * prob p (k c) (s c)

open Classical in
/-- The same probability for an optimal, computationally unbounded prover. -/
noncomputable def value (p : α → Prop) : Game α → ℝ≥0∞
  | .ret a => if p a then 1 else 0
  | .send _ k => ⨆ m, value p (k m)
  | @Game.coin _ C iF _ k => ∑ c : C, (@Fintype.card C iF : ℝ≥0∞)⁻¹ * value p (k c)

/-- The uniform average over a finite type. -/
noncomputable def avg {C : Type} [Fintype C] (f : C → ℝ≥0∞) : ℝ≥0∞ :=
  ∑ c : C, (Fintype.card C : ℝ≥0∞)⁻¹ * f c

open Classical in
/-- The probability that a uniform coin satisfies `p`, as an average of indicators. -/
noncomputable def prCoin {C : Type} [Fintype C] (p : C → Prop) : ℝ≥0∞ :=
  avg fun c => if p c then 1 else 0

/-- The concurrent run once the first interaction's next step is fixed: `adv₁ g₂` advances the first interaction by one
node while the second is at `g₂`, and `fin₁ b` finishes the first interaction alone once the second has ended with `b`.
Recursion is on the second interaction. -/
def interleaveWith (adv₁ : Game β → Game (α × β)) (fin₁ : β → Game (α × β)) :
    Game β → Game (α × β)
  | .ret b => fin₁ b
  | .send M k => .send Bool fun first =>
      if first then adv₁ (.send M k) else .send M fun m => interleaveWith adv₁ fin₁ (k m)
  | @Game.coin _ C iF iN k => .send Bool fun first =>
      if first then adv₁ (@Game.coin _ C iF iN k)
      else @Game.coin _ C iF iN fun c => interleaveWith adv₁ fin₁ (k c)

/-- Run two interactions concurrently; at every step the prover chooses which one advances. -/
def interleave : Game α → Game β → Game (α × β)
  | .ret a => fun g₂ => g₂.map fun b => (a, b)
  | .send M k => interleaveWith (fun g₂ => .send M fun m => interleave (k m) g₂)
      (fun b => (Game.send M k).map fun a => (a, b))
  | @Game.coin _ C iF iN k => interleaveWith
      (fun g₂ => @Game.coin _ C iF iN fun c => interleave (k c) g₂)
      (fun b => (@Game.coin _ C iF iN k).map fun a => (a, b))

/-- Interactions run concurrently, the prover choosing the schedule; the outputs in order. -/
def batch : List (Game α) → Game (List α)
  | [] => .ret []
  | g :: gs => (g.interleave (batch gs)).map fun x => x.1 :: x.2

/-- A strategy for the relabelled game is a strategy for the game: the same message at every node. -/
def Strategy.ofMap (f : α → β) : (g : Game α) → Strategy (g.map f) → Strategy g
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, Strategy.ofMap f (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => Strategy.ofMap f (k c) (s c)

/-- A strategy for the game is a strategy for the relabelled game, the inverse of `ofMap`. -/
def Strategy.toMap (f : α → β) : (g : Game α) → Strategy g → Strategy (g.map f)
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, Strategy.toMap f (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => Strategy.toMap f (k c) (s c)

/-- The strategies for the first interaction that a strategy `σ` for `interleaveWith adv₁ fin₁ h` induces, given those
that `σ` induces where it advances the first interaction (`advR`) and once the second has ended with `b` (`finR b`).
Where `σ` advances the second interaction, the induced strategy follows `σ`'s message there, and the second's coin is
fixed at some value. -/
def InducedWithL (adv₁ : Game β → Game (α × β)) (fin₁ : β → Game (α × β))
    (advR : (h : Game β) → Strategy (adv₁ h) → Prop) (finR : (b : β) → Strategy (fin₁ b) → Prop) :
    (h : Game β) → Strategy (interleaveWith adv₁ fin₁ h) → Prop
  | .ret b, σ => finR b σ
  | .send M k, σ => match σ with
    | ⟨true, σ'⟩ => advR (.send M k) σ'
    | ⟨false, σ'⟩ => InducedWithL adv₁ fin₁ advR finR (k σ'.1) σ'.2
  | @Game.coin _ C iF iN k, σ => match σ with
    | ⟨true, σ'⟩ => advR (@Game.coin _ C iF iN k) σ'
    | ⟨false, σ'⟩ => ∃ c, InducedWithL adv₁ fin₁ advR finR (k c) (σ' c)

/-- **A strategy that `σ` induces on the first interaction.** `InducedL g h σ s`: `s` plays `g` as the strategy `σ` for
`interleave g h` does, with `h`'s coins fixed at some values and `h`'s messages `σ`'s. At `g`'s message `s` sends `σ`'s,
and at `g`'s coin `s` follows `σ` for every value. -/
def InducedL : (g : Game α) → (h : Game β) → Strategy (interleave g h) → Strategy g → Prop
  | .ret _, _, _, _ => True
  | .send M k, h, σ, s => InducedWithL
      (fun g₂ => .send M fun m => interleave (k m) g₂) (fun b => (Game.send M k).map fun a => (a, b))
      (fun h' σ' => ∃ s', s = ⟨σ'.1, s'⟩ ∧ InducedL (k σ'.1) h' σ'.2 s')
      (fun b σ' => s = Strategy.ofMap (fun a => (a, b)) (.send M k) σ') h σ
  | @Game.coin _ C iF iN k, h, σ, s => InducedWithL
      (fun g₂ => @Game.coin _ C iF iN fun c => interleave (k c) g₂)
      (fun b => (@Game.coin _ C iF iN k).map fun a => (a, b))
      (fun h' σ' => ∀ c, InducedL (k c) h' (σ' c) (s c))
      (fun b σ' => s = Strategy.ofMap (fun a => (a, b)) (@Game.coin _ C iF iN k) σ') h σ

/-- The strategies for the second interaction that a strategy `σ` for `interleaveWith adv₁ fin₁ h` induces, given those
that `σ` induces where it advances the first interaction (`advR`). Where `σ` advances the second, the induced strategy
sends `σ`'s message and follows `σ` for every value of the second's coin. -/
def InducedWithR (adv₁ : Game β → Game (α × β)) (fin₁ : β → Game (α × β))
    (advR : (h : Game β) → Strategy (adv₁ h) → Strategy h → Prop) :
    (h : Game β) → Strategy (interleaveWith adv₁ fin₁ h) → Strategy h → Prop
  | .ret _, _, _ => True
  | .send M k, σ, s => match σ with
    | ⟨true, σ'⟩ => advR (.send M k) σ' s
    | ⟨false, σ'⟩ => ∃ s', s = ⟨σ'.1, s'⟩ ∧ InducedWithR adv₁ fin₁ advR (k σ'.1) σ'.2 s'
  | @Game.coin _ C iF iN k, σ, s => match σ with
    | ⟨true, σ'⟩ => advR (@Game.coin _ C iF iN k) σ' s
    | ⟨false, σ'⟩ => ∀ c, InducedWithR adv₁ fin₁ advR (k c) (σ' c) (s c)

/-- **A strategy that `σ` induces on the second interaction.** `InducedR g h σ s`: `s` plays `h` as the strategy `σ` for
`interleave g h` does, with `g`'s coins fixed at some values and `g`'s messages `σ`'s. -/
def InducedR : (g : Game α) → (h : Game β) → Strategy (interleave g h) → Strategy h → Prop
  | .ret a, h, σ, s => s = Strategy.ofMap (fun b => (a, b)) h σ
  | .send M k, h, σ, s => InducedWithR
      (fun g₂ => .send M fun m => interleave (k m) g₂) (fun b => (Game.send M k).map fun a => (a, b))
      (fun h' σ' s' => InducedR (k σ'.1) h' σ'.2 s') h σ s
  | @Game.coin _ C iF iN k, h, σ, s => InducedWithR
      (fun g₂ => @Game.coin _ C iF iN fun c => interleave (k c) g₂)
      (fun b => (@Game.coin _ C iF iN k).map fun a => (a, b))
      (fun h' σ' s' => ∃ c, InducedR (k c) h' (σ' c) s') h σ s

/-- **A strategy that `σ` induces on one game of a batch.** `InducedAt g post pre σ s`: `s` plays `g`, at position
`pre.length` of `batch (pre ++ g :: post)`, as the strategy `σ` for the batch does, with every other game's coins fixed
at some values and its messages `σ`'s. -/
def InducedAt (g : Game α) (post : List (Game α)) :
    (pre : List (Game α)) → Strategy (batch (pre ++ g :: post)) → Strategy g → Prop
  | [], σ, s => InducedL g (batch post) (Strategy.ofMap (fun x : α × List α => x.1 :: x.2) _ σ) s
  | h :: pre, σ, s => ∃ σ', InducedR h (batch (pre ++ g :: post))
      (Strategy.ofMap (fun x : α × List α => x.1 :: x.2) _ σ) σ' ∧ InducedAt g post pre σ' s

end Game

/-- **Soundness**: `g` is sound for the claim `E` with error `δ` when, whatever the developer does, the verifier
accepts while `E` fails with probability at most `δ`. -/
def Sound {α : Type} (g : Game α) (accept E : α → Prop) (δ : ℝ≥0∞) : Prop :=
  Game.value (fun a => accept a ∧ ¬ E a) g ≤ δ

end Verity
