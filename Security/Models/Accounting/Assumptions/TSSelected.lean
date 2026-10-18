import Models.Accounting.Assumptions.TS

/-!
# A padded window's named assumption: E1(c) per selected unit (trusted)

`UnitFixed`'s counterpart for a window that credits only its selected units (`Accounting.TS.SelectLayer`,
note:20261018T2200Z-draft-padded-credit-rule). Statement draft: its claim id `accounting-ts-selected-fixed` is
proposed, not yet in `verity.core.claims`, pending the red team and the decision.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Pouw.RelativeSecurity

variable {K T L U : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U]

open Classical in
/-- `accounting-ts-selected-fixed` (Lemma E1(c) per unit with its fold clause, at the selected units; H-μ at every
selected operand). **The selected units good with their selections have their words fixed, outside grinding, and a
selected unit not skipped whole has its words folded**: for a window whose positions' writers read them, whose
selections are units' outputs and whose exposed units' candidates are committed before their coins, under the run's
modes `mode` (a unit is paid only if the run computed it, from the operands it committed, under its noise), there is
an event of probability at most `μ·Σ_{u exposed} k_u` (`μ` H-μ's point mass at an operand, `k_u` unit `u`'s
candidates) outside which, on every accepted path:
* the run's owned right fixed words per round are at least the words `wt w j` of the selected units that are good with
  their selections (`SelectLayer.SelGood`: the run paid for every unit feeding it along late edges and gates) over `S`.
  A paid selection unit's selection is the honest function of its committed inputs, so the forming it certifies holds,
  and `wt` is the unit's floor at every input its selections admit, with its noise downstream of them; values
  committed before its coin are fixed when its noise is drawn (E1(c) per unit, as `UnitFixed`). An exposed unit's
  selection is a function of its noise fixed before the coin (its selection unit and the units feeding it paid, its
  candidates committed before the coin), so the selected operand is one of `k_u` values each at point mass at most
  `μ`, at most `k_u·μ` together: the event is that some exposed unit's selected operand lands on its point mass;
* the fold clause: the run's folded words per round are at least the words of the selected units not skipped whole,
  over `S`.

Unselected units, the padding, are in neither sum. It reads neither the draw nor the proofs. It stays an assumption
until the play's fixed and folded words are defined from the committed values (the adaptive model) and H-μ's point
mass at the operands is the game's (`ReceiptGame.OperandPointMass`). -/
def SelectedFixed {C B : Type} {S m s : ℕ} {J Q Pos V : Type} [Fintype C] [Fintype B] [Nonempty V] [Fintype J]
    (P : Play (Paths C m × B) S K T L U) (sl : SelectLayer C B S m s J Q Pos V)
    (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ)
    (μ : ℝ) : Prop :=
  sl.toDrawnLayer.WriterReads → sl.SelectionsComputed → sl.CandidatesEarly →
    ∃ G : Paths (Paths C m × B) S → Prop,
      prob G ≤ μ * ∑ w, ∑ j, (if sl.Exposed w j then (sl.cands w j : ℝ) else 0) ∧
      ∀ x, P.accepts x → ¬ G x →
        (∑ w, ∑ j, if sl.Selected x w j ∧ sl.SelGood mode x w j then wt w j else 0) / S ≤ (P.run x).fixed ∧
          (∑ w, ∑ j, if sl.Selected x w j ∧ mode x (w, j) ≠ .whole then wt w j else 0) / S ≤ folded x

end Accounting.Assumptions.TS
