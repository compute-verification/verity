import Definitions.Pouw.NCP.Pinned
import Proofs.Pouw.Exact
import Definitions.Pouw.Guarantees.NCP

/-!
# Noise that cancels inside one checked product: proofs
-/

namespace Pouw.NCP.Proofs

open Finset Pouw.NCP

theorem cancel_entry (a b e f g h : ℤ) :
    (a + e) * (b + f) + (a + g) * (b + -h) - (a + (e + g)) * (b + (f + -h)) = a * b + (e * h - g * f) := by
  ring

theorem assoc_entry (k : ℕ) (E1 R F1 : Mat) (i j : ℕ) :
    ∑ l ∈ range k, E1 i l * mm k R F1 l j = ∑ l ∈ range k, mm k E1 R i l * F1 l j := by
  have := congrFun (congrFun (Pouw.Proofs.mmR_assoc k k E1 R F1) i) j
  simp only [Pouw.Proofs.mmR] at this
  simp only [mm]
  exact this.symm

/-- **Cancellation.** -/
theorem cancel : Pouw.Guarantees.NCP.Cancel := by
  intro k A B E1 F1 R
  funext i j
  have hassoc := assoc_entry k E1 R F1 i j
  simp only [Pi.add_apply, Pi.sub_apply]
  simp only [mm, E2, F2, E3, F3, Pi.add_apply] at hassoc ⊢
  rw [← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  simp only [cancel_entry]
  rw [Finset.sum_add_distrib, Finset.sum_sub_distrib, hassoc, sub_self, add_zero]

theorem hcat3_one {k l : ℕ} (X1 X2 X3 : Mat) (i : ℕ) (hl : l < k) : hcat3 k X1 X2 X3 i l = X1 i l := by
  simp [hcat3, hl]

theorem hcat3_two {k x : ℕ} (X1 X2 X3 : Mat) (i : ℕ) (hx : x < k) : hcat3 k X1 X2 X3 i (k + x) = X2 i x := by
  simp only [hcat3, show ¬ k + x < k by omega, show k + x < 2 * k by omega, ite_false, ite_true,
    Nat.add_sub_cancel_left]

theorem hcat3_three {k x : ℕ} (X1 X2 X3 : Mat) (i : ℕ) : hcat3 k X1 X2 X3 i (k + k + x) = X3 i x := by
  simp only [hcat3, show ¬ k + k + x < k by omega, show ¬ k + k + x < 2 * k by omega, ite_false,
    show k + k + x - 2 * k = x by omega]

theorem vcat3_one {k l : ℕ} (Y1 Y2 Y3 : Mat) (j : ℕ) (hl : l < k) : vcat3 k Y1 Y2 Y3 l j = Y1 l j := by
  simp [vcat3, hl]

theorem vcat3_two {k x : ℕ} (Y1 Y2 Y3 : Mat) (j : ℕ) (hx : x < k) : vcat3 k Y1 Y2 Y3 (k + x) j = Y2 x j := by
  simp only [vcat3, show ¬ k + x < k by omega, show k + x < 2 * k by omega, ite_false, ite_true,
    Nat.add_sub_cancel_left]

theorem vcat3_three {k x : ℕ} (Y1 Y2 Y3 : Mat) (j : ℕ) : vcat3 k Y1 Y2 Y3 (k + k + x) j = Y3 x j := by
  simp only [vcat3, show ¬ k + k + x < k by omega, show ¬ k + k + x < 2 * k by omega, ite_false,
    show k + k + x - 2 * k = x by omega]

/-- The stacked product is the three-term combination. -/
theorem stacked_eq (k : ℕ) (A B E1 F1 R : Mat) :
    mm (3 * k) (stackX k A E1 R) (stackY k B R F1)
      = mm k (A + E1) (B + F1) + mm k (A + E2 k E1 R) (B + F2 k R F1) - mm k (A + E3 k E1 R) (B + F3 k R F1) := by
  funext i j
  simp only [mm, Pi.add_apply, Pi.sub_apply]
  rw [show 3 * k = k + k + k by ring, Finset.sum_range_add, Finset.sum_range_add]
  have h1 : ∑ x ∈ range k, stackX k A E1 R i x * stackY k B R F1 x j
      = ∑ x ∈ range k, (A + E1) i x * (B + F1) x j :=
    Finset.sum_congr rfl fun x hx => by
      simp only [stackX, stackY, hcat3_one _ _ _ i (Finset.mem_range.1 hx),
        vcat3_one _ _ _ j (Finset.mem_range.1 hx)]
  have h2 : ∑ x ∈ range k, stackX k A E1 R i (k + x) * stackY k B R F1 (k + x) j
      = ∑ x ∈ range k, (A + E2 k E1 R) i x * (B + F2 k R F1) x j :=
    Finset.sum_congr rfl fun x hx => by
      simp only [stackX, stackY, hcat3_two _ _ _ i (Finset.mem_range.1 hx),
        vcat3_two _ _ _ j (Finset.mem_range.1 hx)]
  have h3 : ∑ x ∈ range k, stackX k A E1 R i (k + k + x) * stackY k B R F1 (k + k + x) j
      = -∑ x ∈ range k, (A + E3 k E1 R) i x * (B + F3 k R F1) x j := by
    rw [← Finset.sum_neg_distrib]
    exact Finset.sum_congr rfl fun x _ => by
      simp only [stackX, stackY, hcat3_three, vcat3_three, Pi.neg_apply, neg_mul]
  rw [h1, h2, h3, sub_eq_add_neg]
  simp only [Pi.add_apply]

/-- **One product.** -/
theorem stacked : Pouw.Guarantees.NCP.Stacked := fun k A B E1 F1 R =>
  ⟨stacked_eq k A B E1 F1 R, (stacked_eq k A B E1 F1 R).trans (cancel k A B E1 F1 R)⟩

/-- **int32 exactness with wraparound.** -/
theorem wrap : Pinned.Wrap := by
  intro k A B E1 F1 R i j h
  have e := congrFun (congrFun (stacked k A B E1 F1 R).2 i) j
  rw [Pouw.Proofs.mmW_eq_of_mem (by rw [e]; exact h), e]

/-- **int32 exactness for int7 operands.** -/
theorem wrapInt7 : Pinned.WrapInt7 := by
  intro k A B E1 F1 R i j hk hA hB
  exact wrap k A B E1 F1 R i j (Pouw.Proofs.int7_sum_mem k (fun l => A i l) (fun l => B l j) hk hA hB)

theorem e2_bound {k : ℕ} {E1 R : Mat} (hR : SignedPerm k R) {i : ℕ} (hE : ∀ l < k, -31 ≤ E1 i l ∧ E1 i l ≤ 31)
    {l : ℕ} (hl : l < k) : -31 ≤ E2 k E1 R i l ∧ E2 k E1 R i l ≤ 31 := by
  obtain ⟨π, σ, hσ, hR⟩ := hR
  have hc : ∀ x : Fin k, R x l = if x = π ⟨l, hl⟩ then σ ⟨l, hl⟩ else 0 := fun x => hR x ⟨l, hl⟩
  have e : E2 k E1 R i l = E1 i (π ⟨l, hl⟩) * σ ⟨l, hl⟩ := by
    simp only [E2, mm]
    rw [Finset.sum_range (fun x => E1 i x * R x l)]
    simp only [hc, mul_ite, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [e]
  have hb := hE _ (π ⟨l, hl⟩).isLt
  rcases hσ ⟨l, hl⟩ with h | h <;> rw [h] <;> constructor <;> linarith [hb.1, hb.2]

theorem f2_bound {k : ℕ} {R F1 : Mat} (hR : SignedPerm k R) {j : ℕ} (hF : ∀ l < k, -31 ≤ F1 l j ∧ F1 l j ≤ 31)
    {l : ℕ} (hl : l < k) : -31 ≤ F2 k R F1 l j ∧ F2 k R F1 l j ≤ 31 := by
  obtain ⟨π, σ, hσ, hR⟩ := hR
  have hc : ∀ x : Fin k, R l x = if x = π.symm ⟨l, hl⟩ then σ x else 0 := fun x => by
    have hiff : ((⟨l, hl⟩ : Fin k) = π x) ↔ x = π.symm ⟨l, hl⟩ :=
      ⟨fun h => by rw [h, Equiv.symm_apply_apply], fun h => by rw [h, Equiv.apply_symm_apply]⟩
    exact (hR ⟨l, hl⟩ x).trans (if_congr hiff rfl rfl)
  have e : F2 k R F1 l j = -(σ (π.symm ⟨l, hl⟩) * F1 (π.symm ⟨l, hl⟩) j) := by
    simp only [F2, mm]
    rw [Finset.sum_range (fun x => R l x * F1 x j)]
    simp only [hc, ite_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rw [e]
  have hb := hF _ (π.symm ⟨l, hl⟩).isLt
  rcases hσ (π.symm ⟨l, hl⟩) with h | h <;> rw [h] <;> constructor <;> linarith [hb.1, hb.2]

/-- **The int8 range.** -/
theorem range : Pouw.Guarantees.NCP.Range := by
  intro k A B E1 F1 R hR hE hF hA hB
  refine ⟨fun i l hl => ?_, fun l j hl => ?_⟩
  · simp only [stackX, hcat3, E3, Pi.add_apply, Pi.neg_apply]
    split_ifs with h1 h2
    · have := hA i l h1; have := hE i l h1; unfold InInt7 at *; constructor <;> linarith
    · have := hA i (l - k) (by omega); have := hE i (l - k) (by omega)
      have := e2_bound hR (hE i) (show l - k < k by omega); unfold InInt7 at *; constructor <;> linarith
    · have := hA i (l - 2 * k) (by omega); have := hE i (l - 2 * k) (by omega)
      have := e2_bound hR (hE i) (show l - 2 * k < k by omega); unfold InInt7 at *; constructor <;> linarith
  · simp only [stackY, vcat3, F3, Pi.add_apply]
    split_ifs with h1 h2
    · have := hB l h1 j; have := hF l h1 j; unfold InInt7 at *; constructor <;> linarith
    · have := hB (l - k) (by omega) j; have := hF (l - k) (by omega) j
      have := f2_bound hR (fun l' hl' => hF l' hl' j) (show l - k < k by omega); unfold InInt7 at *
      constructor <;> linarith
    · have := hB (l - 2 * k) (by omega) j; have := hF (l - 2 * k) (by omega) j
      have := f2_bound hR (fun l' hl' => hF l' hl' j) (show l - 2 * k < k by omega); unfold InInt7 at *
      constructor <;> linarith

/-- **`Range`'s hypotheses are satisfiable.** -/
theorem signedPermWitness : Pinned.SignedPermWitness := by
  intro k
  refine ⟨Equiv.refl _, fun _ => 1, fun _ => Or.inl rfl, fun i j => ?_⟩
  by_cases h : i = j <;> simp [h, Fin.val_inj]

end Pouw.NCP.Proofs
