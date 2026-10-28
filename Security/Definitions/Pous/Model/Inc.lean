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

/-- **`INC(β, π)`**, untimed. For every `W`, compressor `Cmp(W, pp, C)` to exactly `β` bits and expander
`Exp(W, pp, w)`, both seeing the whole primitive: `Pr[Exp(W, pp, Cmp(W, pp, C)) = C] ≤ π` over setup. -/
def INC (sch : Scheme M) (β : ℕ) (π : ℝ≥0∞) : Prop :=
  ∀ (W : Bits sch.n) (cmp : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits β)
    (exp : Bits sch.n → Bits sch.t → Bits β → M.Ω → sch.Code),
    pr (fun x : M.Ω × sch.Coins =>
        let pp := (sch.enc W x.1 x.2).1
        let C := (sch.enc W x.1 x.2).2
        exp W pp (cmp W pp C x.1) x.1 = C) ≤ π

/-- **`INC(β, π)` at compute bound `(D, Q)`.** As `INC`, but block `i` is rebuilt by a `(D, Q)`-bounded program
`exp W pp w i`. -/
def TimedINC (sch : Scheme M) (D Q β : ℕ) (π : ℝ≥0∞) : Prop :=
  ∀ (W : Bits sch.n) (cmp : Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits β)
    (exp : Bits sch.n → Bits sch.t → Bits β → Fin sch.B → OracleProg M (Bits sch.ℓ)),
    (∀ W pp w i, (exp W pp w i).Bounded D Q) →
    pr (fun x : M.Ω × sch.Coins =>
        let pp := (sch.enc W x.1 x.2).1
        let C := (sch.enc W x.1 x.2).2
        ∀ i, (exp W pp (cmp W pp C x.1) i).run (M.answer x.1) = C i) ≤ π

/-- An untimed product-form responder: block `i` from `(W, pp, σ)` and the whole primitive. -/
abbrev ProdResponder (sch : Scheme M) (S : ℕ) : Type :=
  Bits sch.n → Bits sch.t → Bits S → M.Ω → Fin sch.B → Bits sch.ℓ

/-- A timed product-form responder: block `i` from `(W, pp, σ)` by a `(D, Q)`-bounded program. -/
structure TimedProdResponder (sch : Scheme M) (S D Q : ℕ) : Type where
  prog : Bits sch.n → Bits sch.t → Bits S → Fin sch.B → OracleProg M (Bits sch.ℓ)
  bounded : ∀ W pp σ i, (prog W pp σ i).Bounded D Q

/-- The `k`-challenge audit accepts an untimed product-form responder. -/
def prodAccepts (sch : Scheme M) {S : ℕ} (k : ℕ) (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (A₂ : ProdResponder sch S) (x : Outcome sch k) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  ∀ j, A₂ W pp (A₁ W pp C ω) ω (x.2.2 j) = C (x.2.2 j)

/-- The `k`-challenge audit accepts a timed product-form responder. -/
def timedProdAccepts (sch : Scheme M) {S D Q : ℕ} (k : ℕ) (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (A₂ : TimedProdResponder sch S D Q) (x : Outcome sch k) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  ∀ j, (A₂.prog W pp (A₁ W pp C ω) (x.2.2 j)).run (M.answer ω) = C (x.2.2 j)

/-- The report's `p₀ = 1 − (β − S − B)/(B·ℓ)`. -/
noncomputable def p₀ (β S B ℓ : ℕ) : ℝ :=
  1 - ((β : ℝ) - S - B) / (B * ℓ)

end Pous
