module

public import Proofs.Core.Game.Runs
import Mathlib.Tactic.Positivity.Finset
public import Definitions.Core.Game.Seq

/-! Sequencing strategies, and runs as a game. -/

@[expose] public section

namespace Verity

namespace Game

variable {α β : Type}

theorem expect_bind (F : β → ℝ) : ∀ (g : Game α) (f : α → Game β) (s : Strategy g)
    (t : (a : α) → Strategy (f a)),
    expect F (g.bind f) (Strategy.bind g f s t) = expect (fun a => expect F (f a) (t a)) g s
  | .ret _, _, _, _ => rfl
  | .send _ k, f, s, t => expect_bind F (k s.1) f s.2 t
  | @Game.coin _ C _ _ k, f, s, t => by
      show (∑ c : C, expect F ((k c).bind f) (Strategy.bind (k c) f (s c) t)) / _ = _
      simp only [expect_bind F (k _) f (s _) t]
      rfl

theorem expect_map (F : β → ℝ) (f : α → β) (g : Game α) (s : Strategy g) :
    expect F (g.map f) (Strategy.map f g s) = expect (fun a => F (f a)) g s :=
  expect_bind F g _ s _

theorem expect_runsG (g : Game α) (s : Strategy g) :
    ∀ (n : ℕ) (F : List α → ℝ), expect F (runsG g n) (runsStrat g s n) = expectN g s n F
  | 0, _ => rfl
  | n + 1, F => by
      show expect F (g.bind _) (Strategy.bind g _ s _) = _
      rw [expect_bind]
      simp only [expectN]
      refine congrArg (fun f => expect f g s) (funext fun a => ?_)
      rw [expect_map]
      exact expect_runsG g s n _

theorem expect_sample (C : Type) [Fintype C] [Nonempty C] (F : C → ℝ) :
    expect F (Game.draw C) (sampleStrat C) = (∑ c : C, F c) / Fintype.card C := rfl

end Game

end Verity
