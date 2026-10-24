import Definitions.Pous.Model.Inc

/-! The proof-only part of `Definitions.Pous.Model.Inc`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

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

/-- An untimed product-form responder: block `i` from `(W, pp, σ)` and the whole primitive. -/
abbrev ProdResponder (sch : Scheme M) (S : ℕ) : Type :=
  Bits sch.n → Bits sch.t → Bits S → M.Ω → Fin sch.B → Bits sch.ℓ

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
