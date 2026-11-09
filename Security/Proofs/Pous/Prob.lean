module

public import Definitions.Pous.Game.Prob
public import Proofs.Core.Game.Basic

/-!
# `pr` as a fraction of the type

`Pous.pr` is Core's uniform probability `Verity.Game.prCoin`; `pr_def` and `pr_eq_card` give it as the fraction of the
type on which the event holds (`Verity.Game.prCoin_eq_card`), the form the bounds on `pr` are proved in.
-/

@[expose] public section

namespace Pous

open scoped ENNReal

variable {α : Type} [Fintype α]

theorem pr_def (P : α → Prop) [DecidablePred P] :
    pr P = ((Finset.univ.filter P).card : ℝ≥0∞) / Fintype.card α :=
  Verity.Game.prCoin_eq_card P

open Classical in
/-- `pr_def` at the classical decision of `P`, the one `pr`'s card form had when it was a definition of its own. -/
theorem pr_eq_card (P : α → Prop) :
    pr P = ((Finset.univ.filter P).card : ℝ≥0∞) / Fintype.card α :=
  pr_def P

end Pous
