import Proofs.Nci.Chain
import Proofs.Nci.Inequalities
import Proofs.Nci.LoomisWhitney

/-!
# Updated matrix multiplications are throttled

The draft's proof of its theorem ("Throttling updated matrix multiplications"), over the IUs of a division `P` that
witnesses compliance, each of at most `Γ` gates and so reading at most `K = αΓ`. An IU `R` that uses `p` weight entries
and has `τ` full positions computes at most `p τ + √(p Q Y)` products (`card_products_le`): at most `p τ` at its full
positions (Lemma 1) and, by Loomis–Whitney at each multiplication and Cauchy–Schwarz over them, at most `√(p Q Y)` at
its other positions, where it uses `Q` input entries and contributes to `Y` output entries (`outputsOther`). Each of
those output entries is split (Lemma 2), so the IUs' `Y`s add up to at most twice the partial sums that
incompressibility charges (`sum_card_outputsOther_le`). The two read counts, weighted by `μ`, give
`min{c₁, c₂} G ≤ α size(D)` for every `μ`, since the IUs' products add up to `G` (`sum_card_products`), and so
`F · 2G ≤ size(D)`.
-/

namespace Nci.SecurityProofs

open Verity Finset

section Counting

variable {D : Circuit} (c : Chain D)

open Classical in
/-- The output entries `y_t^{(ℓ)}[i]` that `R` contributes products to at positions that are not full for `R`: the
draft's `Y_R`. -/
noncomputable def outputsOther (R : Finset (Fin D.N)) : Finset ((ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ)) :=
  univ.filter fun x => ¬ c.Full R x.2.1 ∧ ∃ j, c.prod x.1 x.2.1 x.2.2 j ∈ R

/-- Double counting: the pairs `(a, b)` related by `r`, counted by `a` or by `b`. -/
theorem sum_card_filter_comm {α β : Type*} [Fintype α] [Fintype β] (r : α → β → Prop) [∀ a b, Decidable (r a b)] :
    ∑ a, #(univ.filter (r a)) = ∑ b, #(univ.filter fun a => r a b) := by
  simp only [card_filter]
  exact sum_comm

open Classical in
/-- **The products of an IU** (the draft's proof of its theorem): `R` computes at most `p τ + √(p Q Y)` products, at most
`p τ` at its full positions (Lemma 1) and at most `√(p Q Y)` at the others, by Loomis–Whitney at each multiplication
and Cauchy–Schwarz over them. -/
theorem card_products_le (R : Finset (Fin D.N)) :
    ((univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
        c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ R).card : ℝ) ≤
      (c.weightsUsed R).card * (univ.filter (c.Full R)).card +
        √((c.weightsUsed R).card * ((c.inputsUsed R).filter fun x => ¬ c.Full R x.2.1).card *
          (outputsOther c R).card) := by
  rw [← card_filter_add_card_filter_not fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
    c.Full R g.2.1, filter_filter, filter_filter, Nat.cast_add]
  refine add_le_add (by exact_mod_cast (full_positions c R 0 fun _ => Nat.zero_le _).1) ?_
  let T : Fin c.L → Finset (ℕ × ℕ × ℕ) := fun ℓ =>
    (univ.filter fun x : Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) => c.prod ℓ x.1 x.2.1 x.2.2 ∈ R ∧ ¬ c.Full R x.1).image
      fun x => (x.2.1.val, x.2.2.val, x.1.val)
  have hT : #(univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
      c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ R ∧ ¬ c.Full R g.2.1) = ∑ ℓ, #(T ℓ) := by
    rw [show (univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
        c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ R ∧ ¬ c.Full R g.2.1) = univ.sigma fun ℓ =>
        univ.filter fun x : Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) => c.prod ℓ x.1 x.2.1 x.2.2 ∈ R ∧ ¬ c.Full R x.1 by
      ext ⟨ℓ, x⟩
      simp, card_sigma]
    refine sum_congr rfl fun ℓ _ => (card_image_of_injective _ fun x x' h => ?_).symm
    simp only [Prod.mk.injEq] at h
    exact Prod.ext (Fin.ext h.2.2) (Prod.ext (Fin.ext h.1) (Fin.ext h.2.1))
  have hp : ∀ ℓ ∈ (univ : Finset (Fin c.L)), #((T ℓ).image fun x => (x.1, x.2.1)) ≤ #(c.weightsUsed R) :=
    fun ℓ _ => by
      refine (card_le_card fun y hy => ?_).trans (card_image_le (f := fun w => (w.2.1.val, w.2.2.val)))
      simp only [T, image_image, mem_image, mem_filter, mem_univ, true_and, Function.comp_def] at hy
      obtain ⟨x, ⟨hx, -⟩, rfl⟩ := hy
      refine mem_image.2 ⟨⟨ℓ, x.2.1, x.2.2⟩, ?_, rfl⟩
      simp only [Chain.weightsUsed, mem_filter, mem_univ, true_and]
      exact ⟨x.1, hx⟩
  have hq : ∀ ℓ ∈ (univ : Finset (Fin c.L)), #((T ℓ).image fun x => (x.2.2, x.2.1)) ≤
      #(((c.inputsUsed R).filter fun x => ¬ c.Full R x.2.1).filter fun x => x.1 = ℓ) := fun ℓ _ => by
    refine (card_le_card fun y hy => ?_).trans (card_image_le (f := fun x => (x.2.1.val, x.2.2.val)))
    simp only [T, image_image, mem_image, mem_filter, mem_univ, true_and, Function.comp_def] at hy
    obtain ⟨x, ⟨hx, hnf⟩, rfl⟩ := hy
    refine mem_image.2 ⟨⟨ℓ, x.1, x.2.2⟩, ?_, rfl⟩
    simp only [Chain.inputsUsed, mem_filter, mem_univ, true_and, and_true]
    exact ⟨⟨x.2.1, hx⟩, hnf⟩
  have hy : ∀ ℓ ∈ (univ : Finset (Fin c.L)), #((T ℓ).image fun x => (x.2.2, x.1)) ≤
      #((outputsOther c R).filter fun x => x.1 = ℓ) := fun ℓ _ => by
    refine (card_le_card fun y hy => ?_).trans (card_image_le (f := fun x => (x.2.1.val, x.2.2.val)))
    simp only [T, image_image, mem_image, mem_filter, mem_univ, true_and, Function.comp_def] at hy
    obtain ⟨x, ⟨hx, hnf⟩, rfl⟩ := hy
    refine mem_image.2 ⟨⟨ℓ, x.1, x.2.1⟩, ?_, rfl⟩
    simp only [outputsOther, mem_filter, mem_univ, true_and, and_true]
    exact ⟨hnf, x.2.2, hx⟩
  have h := sum_card_products_le_sqrt univ T hp hq hy
  rw [← card_eq_sum_card_fiberwise fun _ _ => mem_univ _, ← card_eq_sum_card_fiberwise fun _ _ => mem_univ _,
    ← Nat.cast_sum, ← hT] at h
  exact h

open Classical in
/-- **The split entries' contributions** (the draft's proof of its theorem): every output entry an IU contributes to at a
position that is not full for it is split (Lemma 2), so its products lie in at least two IUs, and the IUs' `Y`s add up
to at most `2 ∑ (spread − 1)`. -/
theorem sum_card_outputsOther_le {n : ℕ} (P : Partition D n) :
    ∑ u, #(outputsOther c (P.gates u)) ≤
      2 * ∑ x : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ), (c.spread P x.1 x.2.1 x.2.2 - 1) := by
  have hswap : ∑ u, #(outputsOther c (P.gates u)) =
      ∑ x : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ), #(univ.filter fun u => x ∈ outputsOther c (P.gates u)) := by
    rw [← sum_card_filter_comm]
    exact sum_congr rfl fun u _ => by rw [filter_mem_eq_inter, univ_inter]
  rw [hswap, mul_sum]
  refine sum_le_sum fun ⟨ℓ, t, i⟩ _ => ?_
  dsimp only
  have hle : #(univ.filter fun u => (⟨ℓ, t, i⟩ : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ)) ∈ outputsOther c (P.gates u)) ≤
      c.spread P ℓ t i := by
    refine card_le_card fun u hu => ?_
    simp only [outputsOther, mem_filter, mem_univ, true_and] at hu ⊢
    exact hu.2
  rcases Nat.eq_zero_or_pos #(univ.filter fun u =>
      (⟨ℓ, t, i⟩ : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ)) ∈ outputsOther c (P.gates u)) with h0 | hpos
  · rw [h0]
    exact Nat.zero_le _
  obtain ⟨u, hu⟩ := card_pos.1 hpos
  simp only [outputsOther, mem_filter, mem_univ, true_and] at hu
  obtain ⟨hnf, j, hj⟩ := hu
  have h2 : 2 ≤ c.spread P ℓ t i := by
    by_contra hlt
    refine (other_positions c P u).2 ℓ t i j hj hnf ⟨u, fun j' => ?_⟩
    obtain ⟨u', hu'⟩ := Option.ne_none_iff_exists'.1
      ((P.input_iff (c.prod ℓ t i j')).not.2 (by rw [c.prod_computed]; exact Bool.false_ne_true))
    have huu : u' = u := (card_le_one (s := univ.filter fun u => ∃ j, c.prod ℓ t i j ∈ P.gates u)).1
      (by unfold Chain.spread at hlt; omega) u'
      (by simp only [mem_filter, mem_univ, true_and]; exact ⟨j', P.mem_gates.2 hu'⟩) u
      (by simp only [mem_filter, mem_univ, true_and]; exact ⟨j, hj⟩)
    rw [P.mem_gates, hu', huu]
  omega

/-- **The products add up to `G`**: each product is a computed gate, in exactly one IU. -/
theorem sum_card_products {n : ℕ} (P : Partition D n) :
    ∑ u, #(univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
      c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ P.gates u) = c.G := by
  rw [sum_card_filter_comm]
  have hone : ∀ g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ),
      #(univ.filter fun u => c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ P.gates u) = 1 := fun g => by
    obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1
      ((P.input_iff (c.prod g.1 g.2.1 g.2.2.1 g.2.2.2)).not.2 (by rw [c.prod_computed]; exact Bool.false_ne_true))
    refine card_eq_one.2 ⟨u, ?_⟩
    ext u'
    simp only [mem_filter, mem_univ, true_and, mem_singleton, P.mem_gates, hu, Option.some_inj]
    exact eq_comm
  simp only [hone, sum_const, card_univ, smul_eq_mul, mul_one, Fintype.card_sigma, Fintype.card_prod,
    Fintype.card_fin, Chain.G]
  exact sum_congr rfl fun _ _ => by ring

end Counting

open Classical in
/-- **Updated matrix multiplications are throttled** (the draft's theorem, "Throttling updated matrix
multiplications"). -/
theorem UpdatedMatMul : Guarantees.UpdatedMatMul := by
  intro Γ α b_w b_a D c d hα hw ha hd ⟨n, P, hP⟩ hA hPS
  have hcap : ∀ u, (novelReads D (P.gates u) : ℝ) ≤ α * Γ := fun u =>
    (hP u).2.trans (mul_le_mul_of_nonneg_left (by exact_mod_cast (hP u).1) hα.le)
  have htotal : ∑ u, (novelReads D (P.gates u) : ℝ) ≤ α * D.D.size D.D.computed := by
    calc ∑ u, (novelReads D (P.gates u) : ℝ) ≤ ∑ u, α * (D.D.size (P.gates u) : ℝ) :=
          sum_le_sum fun u _ => (hP u).2
      _ = α * D.D.size D.D.computed := by
          rw [← mul_sum]
          congr 1
          exact_mod_cast P.sum_card_gates
  rcases Nat.eq_zero_or_pos c.G with hG | hG
  · rw [hG, Nat.cast_zero, mul_zero, mul_zero]
    exact Nat.cast_nonneg _
  by_cases hdeg : d = 0 ∨ Γ = 0
  · have hF : F α (α * Γ) b_w b_a d ≤ 0 := by
      unfold F
      refine mul_nonpos_of_nonneg_of_nonpos (by positivity) (Real.iSup_nonpos fun μ => min_le_of_left_le ?_)
      rcases hdeg with rfl | rfl <;> simp
    exact (mul_nonpos_of_nonpos_of_nonneg hF (by positivity)).trans (Nat.cast_nonneg _)
  rw [not_or] at hdeg
  have hdpos : (0 : ℝ) < d := by exact_mod_cast Nat.pos_of_ne_zero hdeg.1
  have hK : 0 < α * Γ := mul_pos hα (by exact_mod_cast Nat.pos_of_ne_zero hdeg.2)
  have hGr : (0 : ℝ) < c.G := by exact_mod_cast hG
  obtain ⟨s, hsum, hs⟩ := hPS n P fun u => ⟨(hP u).1, hcap u⟩
  have hreads := fun u => iu_reads D c Γ (α * Γ) b_w b_a d hd ha.le hA (P.gates u) (P.gates_subset_computed u)
    (hP u).1 (hcap u)
  have key : ∀ μ ∈ Set.Ioo (0 : ℝ) 1, min ((1 + √μ) ^ 2 * (b_w * b_a * d / (α * Γ)))
      (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / (α * Γ)))) * c.G ≤ α * D.D.size D.D.computed := by
    intro μ hμ
    have hY : (∑ u, (#(outputsOther c (P.gates u)) : ℝ)) ≤ 2 * ∑ u, (s u : ℝ) := by
      exact_mod_cast (sum_card_outputsOther_le c P).trans (Nat.mul_le_mul_left 2 hsum)
    have hcap' : ∀ u ∈ (univ : Finset (Fin n)), b_w * (#(c.weightsUsed (P.gates u)) : ℝ) +
        b_a * (d * (#(univ.filter (c.Full (P.gates u))) : ℝ) +
          #((c.inputsUsed (P.gates u)).filter fun x => ¬ c.Full (P.gates u) x.2.1)) ≤ α * Γ :=
      fun u _ => (hreads u).trans (hcap u)
    have hg : ∀ u ∈ (univ : Finset (Fin n)),
        (#(univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
          c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ P.gates u) : ℝ) ≤
        #(c.weightsUsed (P.gates u)) * #(univ.filter (c.Full (P.gates u))) +
          √(#(c.weightsUsed (P.gates u)) * #((c.inputsUsed (P.gates u)).filter fun x => ¬ c.Full (P.gates u) x.2.1) *
            #(outputsOther c (P.gates u))) :=
      fun u _ => card_products_le c (P.gates u)
    have h₁ : ∑ u, (b_w * (#(c.weightsUsed (P.gates u)) : ℝ) +
        b_a * (d * (#(univ.filter (c.Full (P.gates u))) : ℝ) +
          #((c.inputsUsed (P.gates u)).filter fun x => ¬ c.Full (P.gates u) x.2.1))) ≤ α * D.D.size D.D.computed :=
      (sum_le_sum fun u _ => hreads u).trans htotal
    have h₂ : ∑ u, (b_w * (#(c.weightsUsed (P.gates u)) : ℝ) + b_a / 2 * #(outputsOther c (P.gates u))) ≤
        α * D.D.size D.D.computed :=
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
        _ ≤ α * D.D.size D.D.computed := htotal
    have h := min_mul_sum_le_reads univ hw ha hdpos hK hμ.1.le hμ.2.le (fun _ _ => Nat.cast_nonneg _)
      (fun _ _ => Nat.cast_nonneg _) (fun _ _ => Nat.cast_nonneg _) (fun _ _ => Nat.cast_nonneg _) hcap' hg h₁ h₂
    refine le_trans ?_ h
    have hprod : (c.G : ℝ) = ∑ u, (#(univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
        c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ P.gates u) : ℝ) := by
      exact_mod_cast (sum_card_products c P).symm
    rw [hprod]
  have hsup : (⨆ μ : Set.Ioo (0 : ℝ) 1, min ((1 + √(μ : ℝ)) ^ 2 * b_w * b_a * d / (α * Γ))
      (√2 * (1 + √(μ : ℝ)) * √(1 - μ) * b_a * √(b_w / (α * Γ)))) ≤ α * D.D.size D.D.computed / c.G := by
    have : Nonempty (Set.Ioo (0 : ℝ) 1) := ⟨⟨1 / 2, by norm_num, by norm_num⟩⟩
    refine ciSup_le fun μ => ?_
    rw [le_div_iff₀ hGr]
    have e₁ : (1 + √(μ : ℝ)) ^ 2 * b_w * b_a * d / (α * Γ) = (1 + √(μ : ℝ)) ^ 2 * (b_w * b_a * d / (α * Γ)) := by
      ring
    have e₂ : √2 * (1 + √(μ : ℝ)) * √(1 - μ) * b_a * √(b_w / (α * Γ)) =
        √2 * (1 + √(μ : ℝ)) * √(1 - μ) * (b_a * √(b_w / (α * Γ))) := by
      ring
    rw [e₁, e₂]
    exact key μ μ.2
  have hle : F α (α * Γ) b_w b_a d ≤ D.D.size D.D.computed / (2 * c.G) := by
    unfold F
    calc 1 / (2 * α) * _ ≤ 1 / (2 * α) * (α * D.D.size D.D.computed / c.G) :=
          mul_le_mul_of_nonneg_left hsup (by positivity)
      _ = D.D.size D.D.computed / (2 * c.G) := by
          field_simp
  rwa [le_div_iff₀ (by positivity)] at hle

/-- **Frontier training is throttled** (the draft, "From updated matrix multiplications to training"). -/
theorem Training : Guarantees.Training :=
  training_of_updatedMatMul UpdatedMatMul

end Nci.SecurityProofs
