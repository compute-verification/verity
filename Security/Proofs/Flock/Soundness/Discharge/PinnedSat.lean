import Proofs.Flock.Soundness.Discharge.Pinned

/-!
# A pinned statement satisfies its statement without Δ's pinned copies

Δ's copies of the forced-zero row onto the pinned bits (`(r, r)` and `(r, zero)`, on both sides) add, at row `r`,
`μ r · (e_r + e_zero)` to `A` and to `B`, `μ r` the copies' count mod 2 (`PinRows`). In a satisfying witness whose
forced-zero row carries 0, a row with `μ r = 0` is its row without them, and a row with `μ r = 1` whose `B` without
them evaluates to 1 is `(a·z + z_r)(1 + z_r) = z_r`, so `z_r = 0` and `a·z = 0` (`pinned_rule`): its row without
them holds too. So the witness satisfies the statement without the copies, and carries 0 on every bit copied an odd
number of times (`sat_unpin`). Nothing here reads a row size, a `G` or a producer family.
-/

namespace FlockSoundness.Discharge.Pinned

theorem mulVec_pin {k : ℕ} (M A : Matrix (Fin (2 ^ k)) (Fin (2 ^ k)) (ZMod 2)) {r zero : Fin (2 ^ k)} {μ : ZMod 2}
    (h : ∀ j, M r j = A r j + μ * ((if j = r then 1 else 0) + (if j = zero then 1 else 0)))
    (v : Fin (2 ^ k) → ZMod 2) : M.mulVec v r = A.mulVec v r + μ * (v r + v zero) := by
  have e : ∀ j, M r j * v j =
      A r j * v j + μ * ((if j = r then v j else 0) + (if j = zero then v j else 0)) := by
    intro j
    rw [h]
    split_ifs <;> ring
  simp only [Matrix.mulVec, dotProduct, e, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_ite_eq',
    Finset.mem_univ, if_true]

/-- **`S`'s rows are `A`'s and `B`'s with `μ r` copies of the forced-zero row `zero` at row `r`**: `μ r · (e_r + e_zero)`
added to row `r` of each side. -/
def PinRows (S : Model.Statement) (A B : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2))
    (zero : Fin (2 ^ S.kLog)) (μ : Fin (2 ^ S.kLog) → ZMod 2) : Prop :=
  ∀ r j, S.A₀ r j = A r j + μ r * ((if j = r then 1 else 0) + (if j = zero then 1 else 0)) ∧
    S.B₀ r j = B r j + μ r * ((if j = r then 1 else 0) + (if j = zero then 1 else 0))

/-- Over GF(2): `(α + μx)(β + μx) = x`, with `β = 1` where `μ = 1`, is `αβ = x`, and `x = 0` where `μ = 1`. -/
theorem pin_row {μ α β x : ZMod 2} (h : (α + μ * x) * (β + μ * x) = x) (hβ : μ = 1 → β = 1) :
    α * β = x ∧ (μ = 1 → x = 0) := by
  have key : ∀ μ α β x : ZMod 2, (α + μ * x) * (β + μ * x) = x → (μ = 1 → β = 1) → α * β = x ∧ (μ = 1 → x = 0) := by
    decide
  exact key μ α β x h hβ

/-- **A witness of a statement with pinned copies satisfies the statement without them**, and carries 0 on each row
copied an odd number of times: where, at those rows, its forced-zero row carries 0 and `B` without the copies
evaluates to 1. A statement with no copies (`μ = 0`) needs neither. -/
theorem sat_unpin {S : Model.Statement} {A B : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)}
    {zero : Fin (2 ^ S.kLog)} {μ : Fin (2 ^ S.kLog) → ZMod 2} (h : PinRows S A B zero μ) {z : Model.Witness S.m}
    (hz : S.Satisfies z) (hzero : ∀ o r, μ r = 1 → S.block z o zero = 0)
    (hone : ∀ o r, μ r = 1 → B.mulVec (S.block z o) r = 1) :
    ({ S with A₀ := A, B₀ := B } : Model.Statement).Satisfies z ∧ ∀ o r, μ r = 1 → S.block z o r = 0 := by
  have row : ∀ o r, A.mulVec (S.block z o) r * B.mulVec (S.block z o) r = S.block z o r ∧
      (μ r = 1 → S.block z o r = 0) := by
    intro o r
    have hr := hz.1 o r
    rw [mulVec_pin S.A₀ A (fun j => (h r j).1), mulVec_pin S.B₀ B (fun j => (h r j).2)] at hr
    by_cases hμ : μ r = 1
    · rw [hzero o r hμ, add_zero] at hr
      exact pin_row hr (hone o r)
    · have h0 : μ r = 0 := by
        have key : ∀ m : ZMod 2, ¬ m = 1 → m = 0 := by decide
        exact key _ hμ
      rw [h0, zero_mul, add_zero, add_zero] at hr
      exact ⟨hr, fun h1 => absurd h1 hμ⟩
  exact ⟨⟨fun o r => (row o r).1, hz.2.1, hz.2.2⟩, fun o r hr => (row o r).2 hr⟩

/-- **The forced-zero row carries 0** in every block of a satisfying witness, when it has no copies (`μ zero = 0`) and
its `A` without them is empty. -/
theorem zero_block {S : Model.Statement} {A B : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)}
    {zero : Fin (2 ^ S.kLog)} {μ : Fin (2 ^ S.kLog) → ZMod 2} (h : PinRows S A B zero μ) (hμ0 : μ zero = 0)
    (hA0 : ∀ j, A zero j = 0) {z : Model.Witness S.m} (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl)) :
    S.block z o zero = 0 := by
  have hr := hz.1 o zero
  have ha : A.mulVec (S.block z o) zero = 0 := by
    simp only [Matrix.mulVec, dotProduct, hA0, zero_mul, Finset.sum_const_zero]
  rw [mulVec_pin S.A₀ A (fun j => (h zero j).1), hμ0, ha, zero_mul, add_zero, zero_mul] at hr
  exact hr.symm

/-- **The forced-zero row carries 0** in every block of a satisfying witness, when its `A` without the copies is empty,
whatever its own copies: at row `zero` they add `μ zero · (e_zero + e_zero)`, which is 0 over GF(2). -/
theorem zero_block' {S : Model.Statement} {A B : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)}
    {zero : Fin (2 ^ S.kLog)} {μ : Fin (2 ^ S.kLog) → ZMod 2} (h : PinRows S A B zero μ)
    (hA0 : ∀ j, A zero j = 0) {z : Model.Witness S.m} (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl)) :
    S.block z o zero = 0 := by
  have hr := hz.1 o zero
  have ha : A.mulVec (S.block z o) zero = 0 := by
    simp only [Matrix.mulVec, dotProduct, hA0, zero_mul, Finset.sum_const_zero]
  have hxx : S.block z o zero + S.block z o zero = 0 := by
    have key : ∀ x : ZMod 2, x + x = 0 := by decide
    exact key _
  rw [mulVec_pin S.A₀ A (fun j => (h zero j).1), ha, hxx, mul_zero, add_zero, zero_mul] at hr
  exact hr.symm

/-- A row that is `e_k` reads `v k`. -/
theorem mulVec_single {N : ℕ} (M : Matrix (Fin N) (Fin N) (ZMod 2)) {r k : Fin N}
    (h : ∀ j, M r j = if j = k then 1 else 0) (v : Fin N → ZMod 2) : M.mulVec v r = v k := by
  simp only [Matrix.mulVec, dotProduct, h, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, if_true]

end FlockSoundness.Discharge.Pinned
