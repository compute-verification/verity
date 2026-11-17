module

public import Definitions.Core.Game
public import Definitions.Core.Rewinding

/-! Expectations against a strategy (`expect`, `expect₂`), the real-valued version of `prob`. -/

@[expose] public section

namespace Verity

open scoped ENNReal

namespace Game

variable {α : Type}

/-- The expectation of `f` over the verifier's uniform coins, against the strategy `s`. -/
noncomputable def expect (f : α → ℝ) : (g : Game α) → Strategy g → ℝ
  | .ret a, _ => f a
  | .send _ k, s => expect f (k s.1) s.2
  | @Game.coin _ C iF _ k, s => (∑ c : C, expect f (k c) (s c)) / (@Fintype.card C iF)

noncomputable def expect₂ (F : α → α → ℝ) (g : Game α) (s : Strategy g) : ℝ :=
  expect (fun a => expect (F a) g s) g s

open Classical in
/-- The indicator of an event, as a real function. -/
noncomputable def ind (p : α → Prop) (a : α) : ℝ := if p a then 1 else 0

end Game

namespace Rewinding

open Game Finset
variable {α : Type} {P V : Type} [Fintype P] [Fintype V] [DecidableEq V]
variable (op : α → P → Option V)

/-- `vstar` is a plurality table for the strategy's run distribution: at each position no value is
opened with larger probability. -/
def IsPluralityE (g : Game α) (s : Strategy g) (vstar : P → V) : Prop :=
  ∀ p v, expect (ind fun a => op a p = some v) g s ≤ expect (ind fun a => op a p = some (vstar p)) g s

end Rewinding

end Verity
