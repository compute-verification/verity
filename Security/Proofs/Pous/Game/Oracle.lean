import Definitions.Pous.Game.Oracle

/-! The proof-only part of `Definitions.Pous.Game.Oracle`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pous

/-- The plain model: no primitive and no queries. Every `Prog` is then `(0, 0)`-bounded while local computation
stays unbounded, so games in this model are information-theoretic. -/
def IdealModel.plain : IdealModel where
  Ω := Unit
  Qry := Empty
  Ans := Empty
  answer := fun _ q => q.elim

namespace Prog

variable {M : IdealModel} {α : Type}

/-- At most `D` rounds of at most `q` queries each, against every primitive. -/
def RoundBounded (D q : ℕ) (p : Prog M α) : Prop :=
  ∀ ω : M.Ω, p.depth (M.answer ω) ≤ D ∧ p.width (M.answer ω) ≤ q

end Prog

end Pous
