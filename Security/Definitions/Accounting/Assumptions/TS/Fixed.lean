import Definitions.Accounting.TS.Fixed
import Definitions.Accounting.TS.Play

/-!
# The two premises `UnitFixed` rests on under option A (trusted)

Option A proves `Accounting.Assumptions.TS.UnitFixed` from the fixed word of `Definitions.Accounting.TS.Fixed`
(`Accounting.Guarantees.TS.UnitFixedOfWords`). Two premises stay, each a named `Prop` of `Accounting.TS`'s vocabulary
taken as a hypothesis, never an `axiom`, its docstring opening with its claim id (`verity/core/claims/__init__.py`).
`CountsFixed` is not weaker than `UnitFixed`: it counts every fixed word, and a good fixed unit's words are fixed
words, so it asks more than `UnitFixed`'s first half. What it gains is its object: a property of the values (fixed
under option A, computed, every steering unit computed) that the labelling defines, where `UnitFixed` reads the run's
modes.

* `ModesRun`: the modes are the run's. `Accounting.TS.DrawnLayer`'s docstring states it as the protocol structure
  `UnitFixed` is claimed under; here it is a premise, per unit.
* `CountsFixed`: the labelling counts a fixed word as fixed. The play's `fixed` and `folded` are free reals
  (`Anatomy.fixed`), which mean something only through the labelling's definition of `N_right − N_adapt`, so some
  premise has to tie them to the values. This is that premise, at unit
  grain. It is `conjectured`, and retires once it is proved or the labelling is computed in a checked program.
-/

namespace Accounting.Assumptions.TS

open Accounting.TS Pouw.RelativeSecurity

/-- `accounting-ts-modes-run` (the protocol structure `UnitFixed` is claimed under, as a premise). **The modes are the
run's**: on every path, a unit the run paid for is computed (its words and the values it writes are its honest ones,
from the values the layer committed, under its noise), and a unit not skipped whole has its receipt commit the fold of
its words. It holds only where the layer's cuts are no later than the protocol's commitments: "computed from the
committed values" is true of the real run only then. -/
def ModesRun {C B : Type} {S m s : ℕ} {J Q Pos V N W : Type} [Fintype C] [Fintype B] [Nonempty V] [AddCommGroup W]
    (pf : WordLayer C B S m s J Q Pos V N W) (mode : Paths (Paths C m × B) S → Fin S × J → Mode) : Prop :=
  ∀ x u, (mode x u = .paid → pf.Computed x u) ∧ (mode x u ≠ .whole → pf.Folds x u)

open Classical in
/-- `accounting-ts-counts-fixed` (the labelling's `N_right − N_adapt` at unit grain). **The labelling counts fixed words
as fixed**: on every accepted path, the run's owned right fixed words per unit are at least the words `wt w j` of the
units whose words are fixed words (`WordLayer.FixedWords`: fixed under option A, computed, and every unit steering them
computed) over `S`, and its folded words per unit at least the words of the units whose receipts commit their folds,
over `S`. It fails if the instruction-level labelling counts as adaptive, or as not owned or not right, a word that is a
fixed word here. -/
def CountsFixed {K T L U : Type} {C B : Type} {S m s : ℕ} {J Q Pos V N W : Type} [Fintype C] [Fintype B] [Fintype J]
    [Nonempty V] [AddCommGroup W] (P : Play (Paths C m × B) S K T L U) (pf : WordLayer C B S m s J Q Pos V N W)
    (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ) : Prop :=
  ∀ x, P.accepts x →
    (∑ w, ∑ j, if pf.FixedWords wt x (w, j) then wt w j else 0) / S ≤ (P.run x).fixed ∧
      (∑ w, ∑ j, if pf.Folds x (w, j) then wt w j else 0) / S ≤ folded x

end Accounting.Assumptions.TS
