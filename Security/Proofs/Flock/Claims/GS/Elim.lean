import Mathlib

/-!
# Gaussian elimination with a field-operation count

`kernel n eqs` is a nonzero solution of the homogeneous system `eqs` (each equation `e`, the coefficients of
`Σᵢ eᵢ vᵢ = 0`) in `n` unknowns, whenever there are fewer equations than unknowns (`kernel_ne_zero`, `kernel_mem`). It
eliminates one pivot per equation and back-substitutes, with no choice: the pivot is the first nonzero coefficient.

`kernelCost n eqs` counts the field operations it performs (each `+`, `−`, `×`, `÷`, negation and test against `0`),
at most `r·n·(2r+1)` for `r` equations (`kernelCost_le`).
-/

namespace Claims.GS

/-- Equation `e'` with the pivot unknown `j` of `e` eliminated, on the other unknowns: `e' − (e'ⱼ/eⱼ)·e`. -/
def reduce {F : Type} [Field F] {n : ℕ} (e : Fin (n + 1) → F) (j : Fin (n + 1)) (e' : Fin (n + 1) → F) :
    Fin n → F :=
  fun i => e' (j.succAbove i) - e' j / e j * e (j.succAbove i)

/-- The first unknown whose coefficient in `e` is nonzero. -/
def pivot {F : Type} [Field F] [DecidableEq F] {n : ℕ} (e : Fin n → F) : Option (Fin n) :=
  (List.finRange n).find? fun i => e i ≠ 0

/-- **A kernel vector**: eliminate the first equation's pivot from the rest, solve them, and back-substitute the
pivot. -/
def kernel {F : Type} [Field F] [DecidableEq F] : (n : ℕ) → List (Fin n → F) → Fin n → F
  | 0, _ => fun _ => 0
  | _ + 1, [] => Pi.single 0 1
  | n + 1, e :: es =>
    match pivot e with
    | none => kernel (n + 1) es
    | some j =>
      let w := kernel n (es.map (reduce e j))
      Fin.insertNth j (-(∑ i, e (j.succAbove i) * w i) / e j) w
termination_by _ eqs => eqs.length
decreasing_by all_goals simp

/-- **`kernel`'s field operations**: per equation, `n + 1` tests for the pivot; with a pivot, one division and `n`
products and differences per remaining equation, and `n` products, `n` sums, a negation and a division to
back-substitute. -/
def kernelCost {F : Type} [Field F] [DecidableEq F] : (n : ℕ) → List (Fin n → F) → ℕ
  | 0, _ => 0
  | _ + 1, [] => 0
  | n + 1, e :: es =>
    match pivot e with
    | none => (n + 1) + kernelCost (n + 1) es
    | some j => (n + 1) + es.length * (2 * n + 1) + (2 * n + 2) + kernelCost n (es.map (reduce e j))
termination_by _ eqs => eqs.length
decreasing_by all_goals simp

theorem pivot_none {F : Type} [Field F] [DecidableEq F] {n : ℕ} {e : Fin n → F} (h : pivot e = none) : e = 0 := by
  funext i
  have := List.find?_eq_none.1 h i (List.mem_finRange i)
  simpa using this

theorem pivot_some {F : Type} [Field F] [DecidableEq F] {n : ℕ} {e : Fin n → F} {j : Fin n} (h : pivot e = some j) :
    e j ≠ 0 := by
  simpa using List.find?_some h

/-- **`kernel` solves every equation, and is nonzero**, when there are fewer equations than unknowns. -/
theorem kernel_spec {F : Type} [Field F] [DecidableEq F] :
    ∀ (n : ℕ) (eqs : List (Fin n → F)), eqs.length < n →
      kernel n eqs ≠ 0 ∧ ∀ e ∈ eqs, ∑ i, e i * kernel n eqs i = 0
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | n + 1, [], _ => by
    refine ⟨fun h => ?_, by simp⟩
    have := congrFun h 0
    simp [kernel] at this
  | n + 1, e :: es, h => by
    rcases hp : pivot e with _ | j
    · have ih := kernel_spec (n + 1) es (by simp at h; omega)
      have hk : kernel (n + 1) (e :: es) = kernel (n + 1) es := by rw [kernel, hp]
      rw [hk]
      refine ⟨ih.1, fun e' he' => ?_⟩
      rcases List.mem_cons.1 he' with rfl | he'
      · simp [pivot_none hp]
      · exact ih.2 e' he'
    · have hej := pivot_some hp
      have ih := kernel_spec n (es.map (reduce e j)) (by simp at h ⊢; omega)
      set w := kernel n (es.map (reduce e j)) with hw
      have hk : kernel (n + 1) (e :: es) =
          Fin.insertNth j (-(∑ i, e (j.succAbove i) * w i) / e j) w := by
        rw [kernel, hp]
      rw [hk]
      refine ⟨fun h0 => ih.1 ?_, fun e' he' => ?_⟩
      · funext i
        have := congrFun h0 (j.succAbove i)
        simpa [Fin.insertNth_apply_succAbove] using this
      · rw [Fin.sum_univ_succAbove _ j]
        simp only [Fin.insertNth_apply_same, Fin.insertNth_apply_succAbove]
        rcases List.mem_cons.1 he' with rfl | he'
        · field_simp
          ring
        · have hr := ih.2 (reduce e j e') (List.mem_map.2 ⟨e', he', rfl⟩)
          simp only [reduce, sub_mul, Finset.sum_sub_distrib, mul_assoc, ← Finset.mul_sum] at hr
          rw [sub_eq_zero] at hr
          rw [hr]
          field_simp
          ring
termination_by _ eqs => eqs.length
decreasing_by all_goals simp

theorem kernel_ne_zero {F : Type} [Field F] [DecidableEq F] (n : ℕ) (eqs : List (Fin n → F))
    (h : eqs.length < n) : kernel n eqs ≠ 0 :=
  (kernel_spec n eqs h).1

theorem kernel_mem {F : Type} [Field F] [DecidableEq F] (n : ℕ) (eqs : List (Fin n → F)) (h : eqs.length < n)
    (e : Fin n → F) (he : e ∈ eqs) : ∑ i, e i * kernel n eqs i = 0 :=
  (kernel_spec n eqs h).2 e he

/-- **`kernel`'s cost**: at most `r·n·(2r+1)` field operations for `r` equations in `n` unknowns. -/
theorem kernelCost_le {F : Type} [Field F] [DecidableEq F] :
    ∀ (n : ℕ) (eqs : List (Fin n → F)), kernelCost n eqs ≤ eqs.length * n * (2 * eqs.length + 1)
  | 0, _ => by simp [kernelCost]
  | n + 1, [] => by simp [kernelCost]
  | n + 1, e :: es => by
    rcases hp : pivot e with _ | j
    · have ih := kernelCost_le (n + 1) es
      rw [kernelCost, hp]
      simp only [List.length_cons]
      nlinarith
    · have ih := kernelCost_le n (es.map (reduce e j))
      rw [kernelCost, hp]
      simp only [List.length_cons, List.length_map] at ih ⊢
      nlinarith
termination_by _ eqs => eqs.length
decreasing_by all_goals simp

end Claims.GS
