import Proofs.Core.Game.Runs
import Definitions.Core.Game.Seq

/-!
# Sequencing strategies, and runs as a game

A procedure that plays several games in a row, such as a collision finder that reruns the prover, is
itself a game with a fixed strategy: `Strategy.bind` plays `s` in `g` and then `t a` in `f a` after
`g` ends with `a`, and against it `g.bind f` has the expectation of `f a` against `t a`, averaged
over `g` against `s` (`expect_bind`). `runsG g n` plays `g` `n` times in a row; against
`runsStrat g s n` it has the expectation of `n` independent runs (`expect_runsG`).

The definitions are in `Definitions.Core.Game.Seq`.
-/

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

theorem expect_draw (C : Type) [Fintype C] [Nonempty C] (F : C → ℝ) :
    expect F (Game.draw C) (drawStrat C) = (∑ c : C, F c) / Fintype.card C := rfl

end Game

end Verity
