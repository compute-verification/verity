import FlockSoundness.Protocol.Game.Basic
import FlockSoundness.Protocol.Rewinding

/-!
# Expectations against a strategy, and the rewinding lemma for any run distribution

`expect f g s` is the expectation of `f` over the verifier's coins when the prover plays `s`: the
real-valued version of `prob` (`prob_toReal`). Two independent runs of the same strategy give
`expect₂`.

`expect_bad_le` is the rewinding lemma of `Rewinding.prBad_le` for any strategy's run
distribution: if each run opens at most `Q` of `N` positions, the probability that a run opens a
position off the plurality table is at most `√(N · Q · Pr₂[two runs open a position differently])`.
The proof is `Rewinding`'s counting argument with weights: at each position, a run opening it to
`v` conflicts with every run opening it to anything else, which by plurality is at least the
off-plurality mass `b_p`, so the conflict mass at `p` is at least `b_p²`; and Cauchy–Schwarz over
positions.
-/

namespace FlockSoundness

open scoped ENNReal

namespace Game

variable {α : Type}

/-- The expectation of `f` over the verifier's uniform coins, against the strategy `s`. -/
noncomputable def expect (f : α → ℝ) : (g : Game α) → Strategy g → ℝ
  | .ret a, _ => f a
  | .send _ k, s => expect f (k s.1) s.2
  | @Game.coin _ C iF _ k, s => (∑ c : C, expect f (k c) (s c)) / (@Fintype.card C iF)

open Classical in
/-- The indicator of an event, as a real function. -/
noncomputable def ind (p : α → Prop) (a : α) : ℝ := if p a then 1 else 0

end Game

/-! ## The rewinding lemma for any run distribution -/

namespace Rewinding

open Game Finset

variable {α : Type} {P V : Type} [Fintype P] [Fintype V] [DecidableEq V]

variable (op : α → P → Option V)

/-- `vstar` is a plurality table for the strategy's run distribution: at each position no value is
opened with larger probability. -/
def IsPluralityE (g : Game α) (s : Strategy g) (vstar : P → V) : Prop :=
  ∀ p v, expect (ind fun a => op a p = some v) g s ≤ expect (ind fun a => op a p = some (vstar p)) g s

end Rewinding

end FlockSoundness
