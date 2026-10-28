import Definitions.Pous.P2.Cost

/-!
# P2's sequential audit with a state size per round boundary (trusted)

A round boundary is the moment between one answer and the next challenge. Boundary `j` is when the state that round
`j` reads is held; boundary `0` is the end of preprocessing. `VarResponderOn sch Sz D Q L` is `SeqResponderOn` with
`Sz j` bits at boundary `j` in place of one `S` at every boundary: preprocessing leaves a `Sz 0`-bit state, and round
`j` reads the `Sz j`-bit state and challenge `i_j` and, within `RingBounded D Q L`, returns a block and the
`Sz (j + 1)`-bit state. Nothing else survives a boundary.

`varPassesOn sch k j` is the event, over the `k`-challenge audit's randomness, that the first `j` of the `k` answers
are correct. At `Sz = fun _ => S` a `SeqResponderOn` is a `VarResponderOn`, and `varPassesOn sch k k` is
`seqAcceptsOn sch k`. `lowBoundaries Sz S k` counts the boundaries `j < k` at which the state is at most `S` bits.
-/

namespace Pous

variable {B w p : ℕ} [NeZero p]

/-- A sequential responder holding `Sz j` bits at boundary `j`: `step W pp j σ i` answers index `i` in round `j` from
the `Sz j`-bit state `σ`, and returns the block and the `Sz (j + 1)`-bit state, within `RingBounded D Q L`. -/
structure VarResponderOn (sch : Scheme (ringRO B w p)) (Sz : ℕ → ℕ) (D Q L : ℕ) : Type where
  step : Bits sch.n → Bits sch.t → (j : ℕ) → Bits (Sz j) → Fin sch.B →
    OracleAlgorithm (ringRO B w p) (Bits sch.ℓ × Bits (Sz (j + 1)))
  bounded : ∀ W pp j σ i, (step W pp j σ i).RingBounded D Q L

/-- The blocks a `VarResponderOn` returns for the indices `is`, starting at round `j` in state `σ`. -/
def varAnswers {M : IdealModel} {ℓ B : ℕ} {Sz : ℕ → ℕ} (f : M.Qry → M.Ans)
    (step : (j : ℕ) → Bits (Sz j) → Fin B → OracleAlgorithm M (Bits ℓ × Bits (Sz (j + 1)))) :
    (j : ℕ) → List (Fin B) → Bits (Sz j) → List (Bits ℓ)
  | _, [], _ => []
  | j, i :: is, σ =>
      let out := (step j σ i).run f
      out.1 :: varAnswers f step (j + 1) is out.2

/-- The first `j` of the `k` answers are correct. -/
def varPassesOn (sch : Scheme (ringRO B w p)) {Sz : ℕ → ℕ} {D Q L : ℕ} (k j : ℕ) (W : Bits sch.n)
    (A₁ : Preprocessor sch (Sz 0)) (A₂ : VarResponderOn sch Sz D Q L) (x : Outcome sch k) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  (varAnswers ((ringRO B w p).answer ω) (A₂.step W pp) 0 (List.ofFn x.2.2) (A₁ W pp C ω)).take j =
    (List.ofFn fun i => C (x.2.2 i)).take j

/-- The number of boundaries `j < k` at which the state is at most `S` bits. -/
def lowBoundaries (Sz : ℕ → ℕ) (S k : ℕ) : ℕ := ((Finset.range k).filter fun j => Sz j ≤ S).card

end Pous
