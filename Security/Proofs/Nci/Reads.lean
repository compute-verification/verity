import Proofs.Nci.Throttling
import Models.Nci.Guarantees.Reads

/-!
# Updated matrix multiplications are throttled, in reads form

The proof of `UpdatedMatMul` (`Throttling.lean`) before it divides by `α`: the same per-IU counts (`card_products_le`,
`iu_reads`), the same charge for split entries (`sum_card_outputsOther_le`) and the same total (`sum_card_products`),
with the IUs' total reads in place of `α size(D)`. A compliant division at `(Γ, α)` reads at most `α size(D)` in total
and each of its IUs at most `K = αΓ`, which gives back `F · 2G ≤ size(D)` through `F = f(K) / (2α)` (`F_eq_readFactor`,
`updatedMatMul_of_reads`).
-/

namespace Nci.SecurityProofs

open Verity Finset

/-- **NCI's factor is the read factor over `2α`**: `F = f(K) / (2α)`. -/
theorem F_eq_readFactor (α K b_w b_a d : ℝ) : F α K b_w b_a d = readFactor K b_w b_a d / (2 * α) := by
  unfold F readFactor
  rw [one_div_mul_eq_div]

theorem readFactor_nonpos {K b_w b_a : ℝ} {d : ℕ} (hw : 0 < b_w) (hdeg : d = 0 ∨ K ≤ 0) :
    readFactor K b_w b_a d ≤ 0 := by
  unfold readFactor
  refine Real.iSup_nonpos fun μ => ?_
  rcases hdeg with rfl | hK
  · exact min_le_of_left_le (by simp)
  · refine min_le_of_right_le ?_
    rw [Real.sqrt_eq_zero'.2 (div_nonpos_of_nonneg_of_nonpos hw.le hK), mul_zero]

open Classical in
/-- **Updated matrix multiplications are throttled, in reads form** (B5). -/
theorem UpdatedMatMulReads : Guarantees.UpdatedMatMulReads := by
  intro Γ K b_w b_a D c d n P hw ha hd hP hA hPS
  have hT : (0 : ℝ) ≤ ∑ u, (novelReads D (P.gates u) : ℝ) := sum_nonneg fun _ _ => Nat.cast_nonneg _
  rcases Nat.eq_zero_or_pos c.G with hG | hG
  · rw [hG, Nat.cast_zero, zero_mul]
    exact hT
  by_cases hdeg : d = 0 ∨ K ≤ 0
  · exact (mul_nonpos_of_nonneg_of_nonpos (Nat.cast_nonneg _) (readFactor_nonpos hw hdeg)).trans hT
  rw [not_or, not_le] at hdeg
  have hdpos : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hdeg.1
  have hK : 0 < K := hdeg.2
  have hGr : (0 : ℝ) < c.G := by exact_mod_cast hG
  obtain ⟨s, hsum, hs⟩ := hPS n P hP
  have hreads := fun u => iu_reads D c Γ K b_w b_a d hd ha.le hA (P.gates u) (P.gates_subset_computed u)
    (hP u).1 (hP u).2
  have key : ∀ μ ∈ Set.Ioo (0 : ℝ) 1, min ((1 + √μ) ^ 2 * (b_w * b_a * d / K))
      (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))) * c.G ≤ ∑ u, (novelReads D (P.gates u) : ℝ) := by
    intro μ hμ
    have hY : (∑ u, (#(outputsOther c (P.gates u)) : ℝ)) ≤ 2 * ∑ u, (s u : ℝ) := by
      exact_mod_cast (sum_card_outputsOther_le c P).trans (Nat.mul_le_mul_left 2 hsum)
    have hcap' : ∀ u ∈ (univ : Finset (Fin n)), b_w * (#(c.weightsUsed (P.gates u)) : ℝ) +
        b_a * (d * (#(univ.filter (c.Full (P.gates u))) : ℝ) +
          #((c.inputsUsed (P.gates u)).filter fun x => ¬ c.Full (P.gates u) x.2.1)) ≤ K :=
      fun u _ => (hreads u).trans (hP u).2
    have hg : ∀ u ∈ (univ : Finset (Fin n)),
        (#(univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
          c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ P.gates u) : ℝ) ≤
        #(c.weightsUsed (P.gates u)) * #(univ.filter (c.Full (P.gates u))) +
          √(#(c.weightsUsed (P.gates u)) * #((c.inputsUsed (P.gates u)).filter fun x => ¬ c.Full (P.gates u) x.2.1) *
            #(outputsOther c (P.gates u))) :=
      fun u _ => card_products_le c (P.gates u)
    have h₁ : ∑ u, (b_w * (#(c.weightsUsed (P.gates u)) : ℝ) +
        b_a * (d * (#(univ.filter (c.Full (P.gates u))) : ℝ) +
          #((c.inputsUsed (P.gates u)).filter fun x => ¬ c.Full (P.gates u) x.2.1))) ≤
        ∑ u, (novelReads D (P.gates u) : ℝ) :=
      sum_le_sum fun u _ => hreads u
    have h₂ : ∑ u, (b_w * (#(c.weightsUsed (P.gates u)) : ℝ) + b_a / 2 * #(outputsOther c (P.gates u))) ≤
        ∑ u, (novelReads D (P.gates u) : ℝ) :=
      calc ∑ u, (b_w * (#(c.weightsUsed (P.gates u)) : ℝ) + b_a / 2 * #(outputsOther c (P.gates u)))
          = ∑ u, b_w * (#(c.weightsUsed (P.gates u)) : ℝ) + b_a / 2 * ∑ u, (#(outputsOther c (P.gates u)) : ℝ) := by
            rw [sum_add_distrib, mul_sum]
        _ ≤ ∑ u, b_w * (#(c.weightsUsed (P.gates u)) : ℝ) + b_a * ∑ u, (s u : ℝ) := by
            have : b_a / 2 * ∑ u, (#(outputsOther c (P.gates u)) : ℝ) ≤ b_a * ∑ u, (s u : ℝ) :=
              calc b_a / 2 * ∑ u, (#(outputsOther c (P.gates u)) : ℝ) ≤ b_a / 2 * (2 * ∑ u, (s u : ℝ)) :=
                    mul_le_mul_of_nonneg_left hY (by positivity)
                _ = b_a * ∑ u, (s u : ℝ) := by ring
            linarith
        _ = ∑ u, (b_w * (#(c.weightsUsed (P.gates u)) : ℝ) + b_a * s u) := by
            rw [sum_add_distrib, mul_sum]
        _ ≤ ∑ u, (novelReads D (P.gates u) : ℝ) := sum_le_sum fun u _ => hs u
    have h := min_mul_sum_le_reads univ hw ha hdpos hK hμ.1.le hμ.2.le (fun _ _ => Nat.cast_nonneg _)
      (fun _ _ => Nat.cast_nonneg _) (fun _ _ => Nat.cast_nonneg _) (fun _ _ => Nat.cast_nonneg _) hcap' hg h₁ h₂
    refine le_trans ?_ h
    have hprod : (c.G : ℝ) = ∑ u, (#(univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
        c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ P.gates u) : ℝ) := by
      exact_mod_cast (sum_card_products c P).symm
    rw [hprod]
  have hsup : readFactor K b_w b_a d ≤ (∑ u, (novelReads D (P.gates u) : ℝ)) / c.G := by
    have : Nonempty (Set.Ioo (0 : ℝ) 1) := ⟨⟨1 / 2, by norm_num, by norm_num⟩⟩
    refine ciSup_le fun μ => ?_
    rw [le_div_iff₀ hGr]
    have e₁ : (1 + √(μ : ℝ)) ^ 2 * b_w * b_a * d / K = (1 + √(μ : ℝ)) ^ 2 * (b_w * b_a * d / K) := by ring
    have e₂ : √2 * (1 + √(μ : ℝ)) * √(1 - μ) * b_a * √(b_w / K) =
        √2 * (1 + √(μ : ℝ)) * √(1 - μ) * (b_a * √(b_w / K)) := by ring
    rw [e₁, e₂]
    exact key μ μ.2
  rw [le_div_iff₀ hGr] at hsup
  linarith

/-- The reads form gives back `UpdatedMatMul`: a division that witnesses compliance at `(Γ, α)` has IUs of at most `Γ`
gates, each reading at most `K = αΓ`, and reads at most `α size(D)` in total. -/
theorem updatedMatMul_of_reads (h : Guarantees.UpdatedMatMulReads) : Guarantees.UpdatedMatMul := by
  intro Γ α b_w b_a D c d hα hw ha hd ⟨n, P, hP⟩ hA hPS
  have hcap : ∀ u, D.D.size (P.gates u) ≤ Γ ∧ (novelReads D (P.gates u) : ℝ) ≤ α * Γ := fun u =>
    ⟨(hP u).1, (hP u).2.trans (mul_le_mul_of_nonneg_left (by exact_mod_cast (hP u).1) hα.le)⟩
  have htotal : ∑ u, (novelReads D (P.gates u) : ℝ) ≤ α * D.D.size D.D.computed := by
    calc ∑ u, (novelReads D (P.gates u) : ℝ) ≤ ∑ u, α * (D.D.size (P.gates u) : ℝ) :=
          sum_le_sum fun u _ => (hP u).2
      _ = α * D.D.size D.D.computed := by
          rw [← mul_sum]
          congr 1
          exact_mod_cast P.sum_card_gates
  have hr := (h Γ (α * Γ) b_w b_a D c d n P hw ha hd hcap hA hPS).trans htotal
  rw [F_eq_readFactor]
  calc readFactor (α * Γ) b_w b_a d / (2 * α) * (2 * c.G) = c.G * readFactor (α * Γ) b_w b_a d / α := by
        field_simp
    _ ≤ α * D.D.size D.D.computed / α := div_le_div_of_nonneg_right hr hα.le
    _ = D.D.size D.D.computed := by field_simp

end Nci.SecurityProofs
