import FlockSoundness.Protocol.Game.Union

/-!
# Probabilities for a fixed prover strategy

`value` bounds the optimal prover. The table theorems (`table_sound_exec`) bound `prob` for every
strategy instead, and the two differ when some branch has no strategy at all (a `send` of an empty
type), so the audit theorems are stated for every strategy too. These are the strategy-level tools
they use:

* `prob_mono`, `prob_le_one`, `prob_or_le`: monotonicity and the union bound;
* `prob_and_const`: an event conjoined with a fact fixed by the strategy;
* `Strategy.ofMap`, `prob_map`: relabelling the output (`map`) keeps the game's strategies.
-/

namespace FlockSoundness.Game

open scoped ENNReal

variable {α β : Type}

/-- A strategy for the relabelled game is a strategy for the game. -/
def Strategy.ofMap (f : α → β) : (g : Game α) → Strategy (g.map f) → Strategy g
  | .ret _, _ => PUnit.unit
  | .send _ k, s => ⟨s.1, Strategy.ofMap f (k s.1) s.2⟩
  | @Game.coin _ _ _ _ k, s => fun c => Strategy.ofMap f (k c) (s c)

end FlockSoundness.Game
