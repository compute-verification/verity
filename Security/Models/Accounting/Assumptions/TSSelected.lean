import Models.Accounting.Assumptions.TS

/-!
# A padded window's named assumption: E1(c) per selected unit (trusted)

`UnitFixed`'s counterpart for a window that credits only its selected units (`Accounting.TS.SelectLayer`,
note:20261018T2200Z-draft-padded-credit-rule, round 3.1). Statement draft: its claim id `accounting-ts-selected-fixed`
is proposed, not yet in `verity.core.claims`, pending the red team and the decision.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Pouw.RelativeSecurity

variable {K T L U : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U]

open Classical in
/-- `accounting-ts-selected-fixed` (Lemma E1(c) per unit with its fold clause, at the selected units; the unit's floor
at every input its certificates admit). **The selected units good with their selections have their words fixed, and a
selected unit not skipped whole has its words folded**: for a window whose positions' writers read them, whose
selections have sources, whose floor operands are written by their certificates' writers, and whose selections are
committed before their units' coins, and whose noise steps that fan out are always proved (`NoiseProved`), under the
run's modes `mode` (a unit is paid only if the run computed it, from the operands it committed, under its noise), on
every accepted path where the registration's certificates and every
always-proved unit are right:
* the run's owned right fixed words per round are at least the words `wt w j` of the selected units that are good with
  their selections (`SelectLayer.SelGood`: the run paid for every unit feeding it along late edges and along gates from
  units outside the always-proved set) over `S`. A certificate written by a paid unit, an always-proved right one or
  the registration is its raw row's honest forming verdict, and the operand the unit reads is an always-proved right
  forming of that raw row under its noise (`CertifiedOperands`), so it is one its certificate admits, and `wt` is the
  unit's floor there. Every certificate and every value committed before the unit's coin is fixed when its noise is
  drawn (`NoLateSelection`; E1(c) per unit, as `UnitFixed`), and the noise it reads is the derivation's: an anchor the
  verifier recomputes, an always-proved right unit's output, its own in-body derivation, or a writer reaching it alone,
  which a good unit's late edge says the run paid for (`NoiseProved`). The noise is PoUW's seeded derivation; its
  grinding is inside the seeded-noise conjecture's error term, as for `UnitFixed`;
* the fold clause: the run's folded words per round are at least the words of the selected units not skipped whole,
  over `S`.

Unselected units, the padding, are in neither sum. It reads neither the draw nor the proofs: the always-proved units'
rightness is a hypothesis, which the proofs give outside their failures. It stays an assumption until the play's fixed
and folded words are defined from the committed values (the adaptive model). -/
def SelectedFixed {C B : Type} {S m s : ℕ} {J Q Pos V : Type} [Fintype C] [Fintype B] [Nonempty V] [Fintype J]
    (P : Play (Paths C m × B) S K T L U) (sl : SelectLayer C B S m s J Q Pos V)
    (mode : Paths (Paths C m × B) S → Fin S × J → Mode) (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ) :
    Prop :=
  sl.toDrawnLayer.WriterReads → sl.SelectionsSourced → sl.CertifiedOperands → sl.NoLateSelection →
    sl.NoiseProved wt → sl.RegisteredRight →
    ∀ x, P.accepts x → (∀ t ∈ sl.proved, sl.right x t.1 t.2 (sl.toDrawnLayer.view x t.1 t.2)) →
      (∑ w, ∑ j, if sl.Selected x w j ∧ sl.SelGood mode x w j then wt w j else 0) / S ≤ (P.run x).fixed ∧
        (∑ w, ∑ j, if sl.Selected x w j ∧ mode x (w, j) ≠ .whole then wt w j else 0) / S ≤ folded x

end Accounting.Assumptions.TS
