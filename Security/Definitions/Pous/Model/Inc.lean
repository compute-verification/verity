import Mathlib.Basic.ENNReal.Real
import Definitions.Pous.Model.Audit

/-!
# `INC(β, π)` and product-form responders

The source report's names (`research/paper4/report.md` §2–3). `INC(β, π)` is whole-encoding incompressibility given
`W` and `pp`: no compressor to `β` bits lets an expander rebuild all of `C` with probability above `π`. It is
untimed: compressor and expander both see the whole primitive. `TimedINC D Q β π` bounds the expander instead:
it rebuilds each block separately, by a `(D, Q)`-bounded program from `(W, pp, w, i)`.

A product-form responder (the report's `A2'`) answers every challenged index by the same per-block function of the
same state, and `p₀ = 1 − (β − S − B)/(B·ℓ)` is the per-block success above which that responder yields a
compressor to `β` bits (report Theorem 1(ii)).
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- **`INC(β, π)` at compute bound `(D, Q)`.** As `INC`, but block `i` is rebuilt by a `(D, Q)`-bounded program
`exp W pp w i`. -/
def TimedINC (sch : Scheme M) (D Q β : ℕ) (π : ℝ≥0∞) : Prop :=
  ∀ (W : Bits sch.n) (cmp : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits β)
    (exp : Bits sch.n → Bits sch.t → Bits β → Fin sch.B → Prog M (Bits sch.ℓ)),
    (∀ W pp w i, (exp W pp w i).Bounded D Q) →
    pr (fun x : M.Ω × sch.Coins =>
        let pp := (sch.enc W x.1 x.2).1
        let C := (sch.enc W x.1 x.2).2
        ∀ i, (exp W pp (cmp W pp C x.1) i).run (M.answer x.1) = C i) ≤ π

/-- A timed product-form responder: block `i` from `(W, pp, σ)` by a `(D, Q)`-bounded program. -/
structure TimedProdResponder (sch : Scheme M) (S D Q : ℕ) : Type where
  prog : Bits sch.n → Bits sch.t → Bits S → Fin sch.B → Prog M (Bits sch.ℓ)
  bounded : ∀ W pp σ i, (prog W pp σ i).Bounded D Q

end Pous
