import Pous.Protocol.Model.SecureErasure

/-!
# The sequential and secure-erasure audits over any draw of the read indices

`Pous/Protocol/Model/Audit.lean` (the sequential audit) and `Pous/Protocol/Model/SecureErasure.lean` (the secure-erasure
audit) draw their `s` read indices uniformly and with replacement. Here the indices are `rd r : Fin s → Fin B` for a
uniform draw `r : R`, and everything else is those games unchanged: the same setup, provers, rounds, accept rule and
keeping rule. With `R = Fin s → Fin B` and `rd = id` they are those games, definitionally.

**Distinct reads.** With `R = Fin s ↪ Fin B` and `rd` the embedding's function (`distinctReads`), the `s` indices are a
uniform sequence of distinct indices: a uniform `s`-subset revealed in a uniformly random order, which is what
`verity_pous.erasure.spot_indices` draws (the subset sampler, then a uniform shuffle). It needs `s ≤ B`.

**What a reviewer checks.** That `seqAcceptsDraw` and `eraseWinsDraw` are `seqAccepts` and `eraseWins` with
`x.2.2` replaced by `rd x.2.2`, and that `distinctReads` draws every sequence of `s` distinct indices with the same
probability.
-/

namespace Pous

open scoped ENNReal

variable {M : IdealModel}

/-- The randomness of one audit whose read indices come from a uniform `r : R`: the primitive, the encoder's coins and
`r`. -/
abbrev OutcomeDraw (sch : Scheme M) (R : Type) : Type := M.Ω × sch.Coins × R

/-- **Distinct reads**: a uniform embedding `Fin s ↪ Fin B`, read as its `s` distinct indices in order. -/
def distinctReads (sch : Scheme M) (s : ℕ) : (Fin s ↪ Fin sch.B) → Fin s → Fin sch.B := fun I => I

/-- The sequential audit accepts, with read indices `rd x.2.2` (`seqAccepts` with the draw `rd`). -/
def seqAcceptsDraw (sch : Scheme M) {S D Q : ℕ} {R : Type} {s : ℕ} (rd : R → Fin s → Fin sch.B) (W : Bits sch.n)
    (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q) (x : OutcomeDraw sch R) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  seqAnswers (M.answer ω) (A₂.step W pp) 0 (List.ofFn (rd x.2.2)) (A₁ W pp C ω) =
    List.ofFn fun j => C (rd x.2.2 j)

/-- **The sequential audit's target over the draw `rd`.** For every `W`, every `S`-bit preprocessing and every
`(D, Q)`-bounded sequential responder, the audit with read indices `rd r`, `r` uniform, accepts with probability at
most `δ + ε`. -/
def AuditSecureDraw (sch : Scheme M) {R : Type} [Fintype R] {s : ℕ} (rd : R → Fin s → Fin sch.B) (S D Q : ℕ)
    (δ ε : ℝ≥0∞) : Prop :=
  ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SeqResponder sch S D Q),
    pr (seqAcceptsDraw sch rd W A₁ A₂) ≤ δ + ε

/-- The randomness of one secure-erasure audit over the draw: the foreign content, the digest key, and the audit's
primitive, coins and `r`. -/
abbrev EraseOutcomeDraw (sch : Scheme M) (S_φ : ℕ) (K R : Type) : Type := Bits S_φ × K × OutcomeDraw sch R

/-- The prover wins the secure-erasure audit with read indices `rd x.2.2.2.2` (`eraseWins` with the draw `rd`): the
digest is `dig κ C`, every spot read is correct, and the foreign content is in memory at every spot read's arrival. -/
def eraseWinsDraw (sch : Scheme M) {S S_φ D Q : ℕ} {K V R : Type} {s : ℕ} (rd : R → Fin s → Fin sch.B)
    (dig : K → sch.Code → V) (W : Bits sch.n) (P : EraseProver sch S S_φ D Q K V)
    (x : EraseOutcomeDraw sch S_φ K R) : Prop :=
  let z := x.1
  let κ := x.2.1
  let ω := x.2.2.1
  let pp := (sch.enc W ω x.2.2.2.1).1
  let C := (sch.enc W ω x.2.2.2.1).2
  let out := P.fill z W pp C ω κ
  let ans := eraseAnswers (M.answer ω) (P.read.step W pp) P.keep 0 (List.ofFn (rd x.2.2.2.2)) out.2
  out.1 = dig κ C ∧ ans.map Prod.fst = List.ofFn (fun j => C (rd x.2.2.2.2 j)) ∧ ∀ a ∈ ans, a.2 = z

/-- **The secure-erasure target over the draw `rd`.** For every `W` and every prover with `S` bits of memory and
`S_φ` bits of foreign content, the audit with read indices `rd r`, `r` uniform, and digest `dig` accepts with the
foreign content kept at every spot read's arrival with probability at most `ε`. -/
def EraseSecureDraw (sch : Scheme M) {K V R : Type} [Fintype K] [Fintype R] {s : ℕ} (rd : R → Fin s → Fin sch.B)
    (dig : K → sch.Code → V) (S S_φ D Q : ℕ) (ε : ℝ≥0∞) : Prop :=
  ∀ (W : Bits sch.n) (P : EraseProver sch S S_φ D Q K V), pr (eraseWinsDraw sch rd dig W P) ≤ ε

end Pous
