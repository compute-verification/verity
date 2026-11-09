module

public import Mathlib.Basic.ENNReal.Real
public import Definitions.Pous.Model.Audit

/-!
# PoUS's timed incompressibility

`TimedINC D Q β π`, the named assumption PoUS's timed guarantees take: `INC(β, π)` (`Definitions.Pous.Model.Inc`) with
the expander bounded. It reads only the audit game's schemes, so PoUS's guarantees (`Definitions.Pous.Guarantees`) read
it without the rest of the named assumptions.
-/

@[expose] public section

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- **`INC(β, π)` at compute bound `(D, Q)`.** As `INC`, but block `i` is rebuilt by a `(D, Q)`-bounded program
`exp W pp w i`. -/
def TimedINC (sch : Scheme M) (D Q β : ℕ) (π : ℝ≥0∞) : Prop :=
  ∀ (W : Bits sch.n) (cmp : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits β)
    (exp : Bits sch.n → Bits sch.t → Bits β → Fin sch.B → OracleAlgorithm M (Bits sch.ℓ)),
    (∀ W pp w i, (exp W pp w i).Bounded D Q) →
    pr (fun x : M.Ω × sch.Coins =>
        let pp := (sch.enc W x.1 x.2).1
        let C := (sch.enc W x.1 x.2).2
        ∀ i, (exp W pp (cmp W pp C x.1) i).run (M.answer x.1) = C i) ≤ π

end Pous
