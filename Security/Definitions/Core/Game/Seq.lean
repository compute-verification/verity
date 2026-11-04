import Definitions.Core.Game.Expect

/-!
# Sequencing strategies, and runs as a game

A procedure that plays several games in a row, such as a collision finder that reruns the prover, is
itself a game with a fixed strategy: `Strategy.bind` plays `s` in `g` and then `t a` in `f a` after
`g` ends with `a`. `runsG g n` plays `g` `n` times in a row, and `runsStrat g s n` plays `s` in
each run. Their expectations (`expect_bind`, `expect_runsG`) are in `Proofs.Core.Game.Seq`.
-/

namespace Verity

namespace Game

variable {α β : Type}

/-- **Play `s` in `g`, then `t a` in `f a`.** -/
def Strategy.bind : (g : Game α) → (f : α → Game β) → Strategy g → ((a : α) → Strategy (f a)) →
    Strategy (g.bind f)
  | .ret a, _, _, t => t a
  | .send _ k, f, s, t => ⟨s.1, Strategy.bind (k s.1) f s.2 t⟩
  | @Game.coin _ _ _ _ k, f, s, t => fun c => Strategy.bind (k c) f (s c) t

/-- The strategy of `g.map f`: `s` in `g`. -/
def Strategy.map (f : α → β) (g : Game α) (s : Strategy g) : Strategy (g.map f) :=
  Strategy.bind g _ s fun _ => PUnit.unit

/-- **`n` runs of `g` in a row**, their outputs as a list, first run first. -/
def runsG (g : Game α) : ℕ → Game (List α)
  | 0 => .ret []
  | n + 1 => g.bind fun a => (runsG g n).map (a :: ·)

/-- `s` in every run. -/
def runsStrat (g : Game α) (s : Strategy g) : (n : ℕ) → Strategy (runsG g n)
  | 0 => PUnit.unit
  | n + 1 => Strategy.bind g _ s fun a => Strategy.map (a :: ·) _ (runsStrat g s n)

/-- The strategy of a coin: nothing to play. -/
def drawStrat (C : Type) [Fintype C] [Nonempty C] : Strategy (Game.draw C) := fun _ => PUnit.unit

end Game

end Verity
