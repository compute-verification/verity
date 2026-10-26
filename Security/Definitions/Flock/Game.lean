import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Basic.ENNReal.BigOperators
import Mathlib.Basic.ENNReal.Inv
import Mathlib.Order.CompleteLattice.Basic
import Definitions.Core.Game

/-!
# Public-coin interactions with live coins

A `Game α` is the verifier's side of a public-coin interaction, written as a program:

* `send M k`: the prover sends a message `m : M` of its choice, and the interaction continues as
  `k m`;
* `coin C k`: the verifier draws `c : C` uniformly at random, independently of everything before,
  and the interaction continues as `k c`. This is a live coin: it is drawn after every earlier
  message is fixed (in verity, the coin server draws it from the operating system after recording
  the round's digest, PROTOCOL.md §7.1);
* `ret a`: the interaction ends with the verifier's output `a`.

Messages may be anything, including whole oracles (the committed tables of the interactive oracle
proof). A deterministic prover strategy (`Strategy g`) picks a message at every `send` node as a
function of the history, which the subtree it sits in encodes. `Strategy.prob` is the exact
probability, over the verifier's uniform coins, that the output satisfies a predicate, and
`value` is the same probability for an optimal, computationally unbounded prover. A randomized
prover is a mixture of deterministic ones, so every bound on `value` bounds it too
(`prob_le_value`, in `Proofs.Flock.Soundness.Game.Basic`).
-/

namespace FlockSoundness

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

/-! ## Strategies and probabilities -/

/-- A deterministic prover strategy: a message at every `send` node, for every history (the
history is encoded by the position in the tree, so a strategy is simply a choice at each node,
and after a coin it may depend on that coin). -/
def Strategy : Game α → Type
  | .ret _ => PUnit
  | .send _ k => (m : _) × Strategy (k m)
  | @Game.coin _ C _ _ k => (c : C) → Strategy (k c)

open Classical in
/-- The exact probability, over the verifier's independent uniform coins, that the interaction
between the verifier `g` and the prover strategy `s` ends with an output satisfying `p`. -/
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

export Verity.Game (avg prCoin)

end Game

end FlockSoundness
