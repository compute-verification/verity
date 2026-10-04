import FlockSoundness.Game.Basic
import Verity.Protocol.Game

/-!
# C-Flock's games as core's

C-Flock's `FlockSoundness.Game` is core's `Verity.Game` clause for clause (`ret`, `send`, `coin`, the strategies and
`prob`), in a namespace of its own. `Game.toCore` reads a C-Flock game as the same tree in core. A strategy of either
is a strategy of the other, node by node (`Strategy.toCore`, `Strategy.ofCore`, inverse to each other), and it meets
every event with the same probability (`prob_toCore`, `prob_ofCore`). So every bound C-Flock proves about a game holds of
the core game, strategy by strategy, and `toCore` commutes with `bind` and `map`.
-/

namespace FlockBridge

namespace Game

variable {α β : Type}

/-- A C-Flock game as a core game: the same tree. -/
def toCore : FlockSoundness.Game α → Verity.Game α
  | .ret a => .ret a
  | .send M k => .send M fun m => toCore (k m)
  | @FlockSoundness.Game.coin _ C iF iN k => @Verity.Game.coin _ C iF iN fun c => toCore (k c)

/-- A C-Flock strategy as a strategy of the core game: the same message at every `send` node. -/
def Strategy.toCore : (g : FlockSoundness.Game α) → FlockSoundness.Game.Strategy g →
    Verity.Game.Strategy (toCore g)
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, Strategy.toCore (k s.1) s.2⟩
  | @FlockSoundness.Game.coin _ _ _ _ k, s => fun c => Strategy.toCore (k c) (s c)

/-- A strategy of the core game as a C-Flock strategy. -/
def Strategy.ofCore : (g : FlockSoundness.Game α) → Verity.Game.Strategy (Game.toCore g) →
    FlockSoundness.Game.Strategy g
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, Strategy.ofCore (k s.1) s.2⟩
  | @FlockSoundness.Game.coin _ _ _ _ k, s => fun c => Strategy.ofCore (k c) (s c)

theorem Strategy.ofCore_toCore : ∀ (g : FlockSoundness.Game α) (s : FlockSoundness.Game.Strategy g),
    Strategy.ofCore g (Strategy.toCore g s) = s
  | .ret _, _ => rfl
  | .send _ k, s => by
      obtain ⟨m, s⟩ := s
      exact congrArg (Sigma.mk m) (Strategy.ofCore_toCore (k m) s)
  | @FlockSoundness.Game.coin _ _ _ _ k, s => funext fun c => Strategy.ofCore_toCore (k c) (s c)

theorem Strategy.toCore_ofCore : ∀ (g : FlockSoundness.Game α) (s : Verity.Game.Strategy (Game.toCore g)),
    Strategy.toCore g (Strategy.ofCore g s) = s
  | .ret _, _ => rfl
  | .send _ k, s => by
      obtain ⟨m, s⟩ := s
      exact congrArg (Sigma.mk m) (Strategy.toCore_ofCore (k m) s)
  | @FlockSoundness.Game.coin _ _ _ _ k, s => funext fun c => Strategy.toCore_ofCore (k c) (s c)

/-- **A strategy meets every event with the same probability in the core game.** -/
theorem prob_toCore (p : α → Prop) : ∀ (g : FlockSoundness.Game α) (s : FlockSoundness.Game.Strategy g),
    Verity.Game.prob p (toCore g) (Strategy.toCore g s) = FlockSoundness.Game.prob p g s
  | .ret _, _ => rfl
  | .send _ k, s => prob_toCore p (k s.1) s.2
  | @FlockSoundness.Game.coin _ C _ _ k, s => by
      show ∑ c : C, _ * Verity.Game.prob p (toCore (k c)) (Strategy.toCore (k c) (s c)) = ∑ c : C, _ * _
      exact Finset.sum_congr rfl fun c _ => by rw [prob_toCore p (k c) (s c)]

/-- **A core strategy meets every event with the same probability as its C-Flock strategy.** -/
theorem prob_ofCore (p : α → Prop) (g : FlockSoundness.Game α) (s : Verity.Game.Strategy (toCore g)) :
    FlockSoundness.Game.prob p g (Strategy.ofCore g s) = Verity.Game.prob p (toCore g) s := by
  rw [← prob_toCore, Strategy.toCore_ofCore]

theorem toCore_bind (f : α → FlockSoundness.Game β) : ∀ g : FlockSoundness.Game α,
    toCore (g.bind f) = (toCore g).bind fun a => toCore (f a)
  | .ret _ => rfl
  | .send M k => congrArg (Verity.Game.send M) (funext fun m => toCore_bind f (k m))
  | @FlockSoundness.Game.coin _ C iF iN k =>
      congrArg (@Verity.Game.coin _ C iF iN) (funext fun c => toCore_bind f (k c))

theorem toCore_map (f : α → β) (g : FlockSoundness.Game α) : toCore (g.map f) = (toCore g).map f :=
  toCore_bind _ g

end Game

end FlockBridge
