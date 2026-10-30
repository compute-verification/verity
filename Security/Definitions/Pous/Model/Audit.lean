import Definitions.Pous.Model.Scheme

/-!
# The timed audit game, perfect isolation

1. `W` is arbitrary (every statement quantifies over it) and fixed before setup.
2. Setup draws the primitive `ω` and the coins `r` uniformly; `(pp, C) := enc W ω r`.
3. Preprocessing `A₁` is any function of `(W, pp, C, ω)` with an `S`-bit output `σ`. Nothing else survives: this
   is perfect isolation.
4. The memory challenger draws `k` indices `I : Fin k → Fin B`, uniform and with replacement.
5. The responder returns raw blocks, from `(W, pp)` (free) and its `S`-bit state, as a `(D, Q)`-bounded `Prog`.
   The audit accepts iff every returned block equals the true one.

**Simultaneous reveal:** one program receives all of `I` and answers all `k` blocks within one `(D, Q)` budget.
**Sequential reveal:** round `j` receives `i_j`, the round index `j` and the state left by round `j - 1`, and within
`(D, Q)` returns block `i_j` and the next `S`-bit state; the next index arrives as soon as the answer is returned.

A bounded program always answers on time, so "every answer correct and on time" is "every answer correct" here.

The two responders are `SimResponderBy` and `SeqResponderBy`, over a cost bound `bd` on each program a responder runs
(and, sequentially, a state size `Sz j` at each round boundary `j`). `SimResponder` and `SeqResponder` are them at
`Prog.Bounded D Q` and a constant `S`; each other cost model's responders are them at its own bound.
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- Preprocessing: any function of `(W, pp, C)` and the whole primitive, with an `S`-bit output. -/
abbrev Preprocessor (sch : Scheme M) (S : ℕ) : Type :=
  Bits sch.n → Bits sch.t → sch.Code → M.Ω → Bits S

/-- A responder for simultaneous reveal: all `k` indices at once, one program within `bd` for the whole answer. -/
structure SimResponderBy (sch : Scheme M) (S k : ℕ) (bd : ∀ {α : Type}, Prog M α → Prop) : Type where
  prog : Bits sch.n → Bits sch.t → Bits S → (Fin k → Fin sch.B) → Prog M (Fin k → Bits sch.ℓ)
  bounded : ∀ W pp σ I, bd (prog W pp σ I)

/-- A responder for sequential reveal holding `Sz j` bits at boundary `j`: `step W pp j σ i` answers index `i` in
round `j` from the `Sz j`-bit state `σ`, and returns the block and the `Sz (j + 1)`-bit state, within `bd`. -/
structure SeqResponderBy (sch : Scheme M) (Sz : ℕ → ℕ) (bd : ∀ {α : Type}, Prog M α → Prop) : Type where
  step : Bits sch.n → Bits sch.t → (j : ℕ) → Bits (Sz j) → Fin sch.B → Prog M (Bits sch.ℓ × Bits (Sz (j + 1)))
  bounded : ∀ W pp j σ i, bd (step W pp j σ i)

/-- A responder for simultaneous reveal: all `k` indices at once, one `(D, Q)` budget for the whole answer. -/
abbrev SimResponder (sch : Scheme M) (S D Q k : ℕ) : Type := SimResponderBy sch S k (Prog.Bounded D Q)

/-- A responder for sequential reveal: `step W pp j σ i` answers index `i` in round `j` from state `σ`, and returns
the block and the next state, within `(D, Q)`. -/
abbrev SeqResponder (sch : Scheme M) (S D Q : ℕ) : Type := SeqResponderBy sch (fun _ => S) (Prog.Bounded D Q)

/-- The randomness of one audit: the primitive, the encoder's coins and the challenge indices. -/
abbrev Outcome (sch : Scheme M) (k : ℕ) : Type := M.Ω × sch.Coins × (Fin k → Fin sch.B)

/-- The blocks a sequential responder returns for the indices `is`, starting at round `j` in state `σ`. -/
def seqAnswers {ℓ S B : ℕ} (f : M.Qry → M.Ans) (step : ℕ → Bits S → Fin B → Prog M (Bits ℓ × Bits S)) :
    ℕ → List (Fin B) → Bits S → List (Bits ℓ)
  | _, [], _ => []
  | j, i :: is, σ =>
      let out := (step j σ i).run f
      out.1 :: seqAnswers f step (j + 1) is out.2

/-- The simultaneous audit accepts. -/
def simAccepts (sch : Scheme M) {S D Q k : ℕ} (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (A₂ : SimResponder sch S D Q k) (x : Outcome sch k) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  (A₂.prog W pp (A₁ W pp C ω) x.2.2).run (M.answer ω) = fun j => C (x.2.2 j)

/-- The sequential audit with `k` rounds accepts. -/
def seqAccepts (sch : Scheme M) {S D Q : ℕ} (k : ℕ) (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (A₂ : SeqResponder sch S D Q) (x : Outcome sch k) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  seqAnswers (M.answer ω) (A₂.step W pp) 0 (List.ofFn x.2.2) (A₁ W pp C ω) = List.ofFn fun j => C (x.2.2 j)

/-- How challenges are revealed (chosen by the construction). -/
inductive Reveal : Type where
  | simultaneous
  | sequential

/-- **The security target.** For every `W`, every `S`-bit preprocessing and every `(D, Q)`-bounded responder, the
`k`-challenge audit accepts with probability at most `δ + ε`. -/
def AuditSecure (sch : Scheme M) : Reveal → (S D Q k : ℕ) → (δ ε : ℝ≥0∞) → Prop
  | .simultaneous, S, D, Q, k, δ, ε =>
      ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SimResponder sch S D Q k),
        pr (simAccepts sch W A₁ A₂) ≤ δ + ε
  | .sequential, S, D, Q, k, δ, ε =>
      ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q),
        pr (seqAccepts sch k W A₁ A₂) ≤ δ + ε

end Pous
