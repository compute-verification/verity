module

public import Definitions.Flock.Statement
import Mathlib.Tactic.Ring.RingNF

/-! A witness of a pinned statement satisfies its statement without Δ's pinned copies (`sat_unpin`). -/

@[expose] public section

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

def PinRows (S : Model.Statement) (A B : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2))
    (zero : Fin (2 ^ S.kLog)) (μ : Fin (2 ^ S.kLog) → ZMod 2) : Prop :=
  ∀ r j, S.A₀ r j = A r j + μ r * ((if j = r then 1 else 0) + (if j = zero then 1 else 0)) ∧
    S.B₀ r j = B r j + μ r * ((if j = r then 1 else 0) + (if j = zero then 1 else 0))

theorem pin_row {μ α β x : ZMod 2} (h : (α + μ * x) * (β + μ * x) = x) (hβ : μ = 1 → β = 1) :
    α * β = x ∧ (μ = 1 → x = 0) := by
  have key : ∀ μ α β x : ZMod 2, (α + μ * x) * (β + μ * x) = x → (μ = 1 → β = 1) → α * β = x ∧ (μ = 1 → x = 0) := by
    decide
  exact key μ α β x h hβ

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

theorem zero_block {S : Model.Statement} {A B : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2)}
    {zero : Fin (2 ^ S.kLog)} {μ : Fin (2 ^ S.kLog) → ZMod 2} (h : PinRows S A B zero μ) (hμ0 : μ zero = 0)
    (hA0 : ∀ j, A zero j = 0) {z : Model.Witness S.m} (hz : S.Satisfies z) (o : Fin (2 ^ S.nbl)) :
    S.block z o zero = 0 := by
  have hr := hz.1 o zero
  have ha : A.mulVec (S.block z o) zero = 0 := by
    simp only [Matrix.mulVec, dotProduct, hA0, zero_mul, Finset.sum_const_zero]
  rw [mulVec_pin S.A₀ A (fun j => (h zero j).1), hμ0, ha, zero_mul, add_zero, zero_mul] at hr
  exact hr.symm

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

theorem mulVec_single {N : ℕ} (M : Matrix (Fin N) (Fin N) (ZMod 2)) {r k : Fin N}
    (h : ∀ j, M r j = if j = k then 1 else 0) (v : Fin N → ZMod 2) : M.mulVec v r = v k := by
  simp only [Matrix.mulVec, dotProduct, h, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, if_true]

end FlockSoundness.Discharge.Pinned
