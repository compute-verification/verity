import Definitions.Flock.Game.Interleave

/-!
# Strategies of a relabelled or composed game

A strategy for `g.map f` is one for `g` (`Strategy.ofMap`), and a strategy for `g.bind f` restricts
to `g` (`Strategy.ofBind`). The strategy-level probability tools that use them (`prob_map`,
`prob_bind_le`, the union bound) are in `Proofs.Flock.Soundness.Game.Prob`.
-/

namespace FlockSoundness.Game

open scoped ENNReal
variable {α β : Type}

/-- A strategy for the relabelled game is a strategy for the game. -/
def Strategy.ofMap (f : α → β) : (g : Game α) → Strategy (g.map f) → Strategy g
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, Strategy.ofMap f (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => Strategy.ofMap f (k c) (s c)

/-- A strategy for `g.bind f`, restricted to `g`. -/
def Strategy.ofBind (f : α → Game β) : (g : Game α) → Strategy (g.bind f) → Strategy g
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, Strategy.ofBind f (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => Strategy.ofBind f (k c) (s c)

end FlockSoundness.Game
