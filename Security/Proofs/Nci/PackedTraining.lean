import Definitions.Nci.Guarantees.PackedTraining
import Proofs.Nci.Packed
import Proofs.Nci.Training

/-!
# The partial-sum law with dark gates, on a chain with several products to a gate

Lemma 2 and the split entries' count for a `PackedChain`, then `PackedIncompressibleContainedMatMulOn` and
`PackedIncompressibleContainedTrainingOn`. As for the fan-out bound, the counts are of products and entries, and
`prod_mult` is read once: the chain's products whose gates are dark number at most `k` per dark gate
(`card_products_le_dark_packed`), which is where `k` enters the training form, as `2 k #dark`. At `k = 1` both are the
per-product guarantees (`incompressibleContainedMatMulOn_of_packed`, `incompressibleContainedTrainingOn_of_packed`).
-/

namespace Nci.SecurityProofs

open Verity Finset

section Lemmas

variable {D : Circuit} {k : ℕ} (c : PackedChain D k)

/-- **Lemma 2 (other positions)**, on a packed chain: every output entry that an IU contributes products to, at a
position that is not full for it, is split. -/
theorem other_positions_packed {n : ℕ} (P : Partition D n) (u : Fin n) :
    ∀ ℓ t i j, c.prod ℓ t i j ∈ P.gates u → ¬ c.Full (P.gates u) t → c.Split P ℓ t i := by
  refine fun ℓ t i j hj ht ⟨u', hw⟩ => ht ⟨ℓ, i, ?_⟩
  have hu : u' = u := by
    have h1 := (P.mem_gates).1 (hw j)
    have h2 := (P.mem_gates).1 hj
    exact Option.some_injective _ (h1.symm.trans h2)
  exact hu ▸ hw

open Classical in
/-- **The split entries' contributions**, on a packed chain: the IUs' `Y`s add up to at most `2 ∑ (spread − 1)`. -/
theorem sum_card_outputsOther_le_packed {n : ℕ} (P : Partition D n) :
    ∑ u, #(outputsOtherPacked c (P.gates u)) ≤
      2 * ∑ x : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ), (c.spread P x.1 x.2.1 x.2.2 - 1) := by
  have hswap : ∑ u, #(outputsOtherPacked c (P.gates u)) =
      ∑ x : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ),
        #(univ.filter fun u => x ∈ outputsOtherPacked c (P.gates u)) := by
    rw [← sum_card_filter_comm]
    exact sum_congr rfl fun u _ => by rw [filter_mem_eq_inter, univ_inter]
  rw [hswap, mul_sum]
  refine sum_le_sum fun ⟨ℓ, t, i⟩ _ => ?_
  dsimp only
  have hle : #(univ.filter fun u =>
      (⟨ℓ, t, i⟩ : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ)) ∈ outputsOtherPacked c (P.gates u)) ≤
      c.spread P ℓ t i := by
    refine card_le_card fun u hu => ?_
    simp only [outputsOtherPacked, mem_filter, mem_univ, true_and] at hu ⊢
    exact hu.2
  rcases Nat.eq_zero_or_pos #(univ.filter fun u =>
      (⟨ℓ, t, i⟩ : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ)) ∈ outputsOtherPacked c (P.gates u)) with h0 | hpos
  · rw [h0]
    exact Nat.zero_le _
  obtain ⟨u, hu⟩ := card_pos.1 hpos
  simp only [outputsOtherPacked, mem_filter, mem_univ, true_and] at hu
  obtain ⟨hnf, j, hj⟩ := hu
  have h2 : 2 ≤ c.spread P ℓ t i := by
    by_contra hlt
    refine other_positions_packed c P u ℓ t i j hj hnf ⟨u, fun j' => ?_⟩
    obtain ⟨u', hu'⟩ := Option.ne_none_iff_exists'.1
      ((P.input_iff (c.prod ℓ t i j')).not.2 (by rw [c.prod_computed]; exact Bool.false_ne_true))
    have huu : u' = u := (card_le_one (s := univ.filter fun u => ∃ j, c.prod ℓ t i j ∈ P.gates u)).1
      (by unfold PackedChain.spread at hlt; omega) u'
      (by simp only [mem_filter, mem_univ, true_and]; exact ⟨j', P.mem_gates.2 hu'⟩) u
      (by simp only [mem_filter, mem_univ, true_and]; exact ⟨j, hj⟩)
    rw [P.mem_gates, hu', huu]
  omega

end Lemmas

/-- **Products off the books are in dark gates, at most `k` to a gate**: the chain's products number at most those
whose gates are in `S` plus `k` for each dark gate. This is where `prod_mult` is read. -/
theorem card_products_le_dark_packed {c d : Declaration} {k : ℕ} (M : PartialContainment c d) (ch : PackedChain c.D k) :
    ch.G ≤ (univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
      ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card + k * M.dark.card := by
  have huniv : (univ : Finset ((ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ))).card = ch.G := by
    simp only [card_univ, Fintype.card_sigma, Fintype.card_prod, Fintype.card_fin, PackedChain.G]
    exact sum_congr rfl fun _ _ => by ring
  rw [← huniv, ← card_filter_add_card_filter_not (s := univ)
    fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) => ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S]
  refine Nat.add_le_add_left (card_le_mul_card_image_of_maps_to
    (f := fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) => ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2)
    (fun p hp => ?_) k fun g _ => ?_) _
  · simp only [mem_filter, mem_univ, true_and] at hp
    exact mem_dark.2 ⟨by simpa [Circuit.computed] using ch.prod_computed p.1 p.2.1 p.2.2.1 p.2.2.2, hp⟩
  · exact (card_le_card (filter_subset_filter _ (subset_univ _))).trans (ch.prod_mult g)

open Classical in
/-- **The partial sums, counted over the pieces**, on a packed chain. -/
theorem partialSums_count_packed {c d : Declaration} {M : PartialContainment c d} {n : ℕ} {P : Partition d.D n}
    {k : ℕ} {ch : PackedChain c.D k} {b_w b_a : ℝ} (ha : 0 ≤ b_a)
    {s : Fin (Fintype.card (Fin n ⊕ {v // v ∈ M.dark})) → ℕ}
    (hsum : ∑ x : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ), (ch.spread (withDark M P) x.1 x.2.1 x.2.2 - 1) ≤ ∑ i, s i)
    (hsP : ∀ u, b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
      b_a * s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u)) ≤ novelReads d (P.gates u)) :
    ∑ u, (b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) + b_a / 2 * (outputsOtherPacked ch (piece M P u)).card) ≤
      ∑ u, (novelReads d (P.gates u) : ℝ) +
        b_a * ∑ v : {v // v ∈ M.dark}, (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ) := by
  have hY : ∑ u, ((outputsOtherPacked ch (piece M P u)).card : ℝ) ≤
      2 * (∑ u, (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u)) : ℝ) +
        ∑ v : {v // v ∈ M.dark}, (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ)) := by
    have h := (sum_card_outputsOther_le_packed ch (withDark M P)).trans (Nat.mul_le_mul_left 2 hsum)
    rw [sum_withDark M, sum_withDark M] at h
    simp only [withDark_gates_inl, withDark_gates_inr] at h
    exact_mod_cast le_trans (Nat.le_add_right _ _) h
  have e₁ : ∑ u, (b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
      b_a * s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u))) =
      ∑ u, b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
        b_a * ∑ u, (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u)) : ℝ) := by
    rw [sum_add_distrib, mul_sum]
  have e₂ : ∑ u, (b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
      b_a / 2 * (outputsOtherPacked ch (piece M P u)).card) =
      ∑ u, b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
        b_a / 2 * ∑ u, ((outputsOtherPacked ch (piece M P u)).card : ℝ) := by
    rw [sum_add_distrib, mul_sum]
  have h₃ := sum_le_sum fun u (_ : u ∈ univ) => hsP u
  rw [e₁] at h₃
  have h₄ := mul_le_mul_of_nonneg_left hY (by positivity : (0 : ℝ) ≤ b_a / 2)
  rw [e₂]
  linarith

open Classical in
/-- **Training's incompressibility gives the activations'**, on a packed chain: a small IU is a division of `c` into
it and the rest. -/
theorem activationsIncompressible_of_training_packed {c : Declaration} {k : ℕ} {ch : PackedChain c.D k}
    {T : Threading c} {Γ : ℕ} {K b_w b_a : ℝ} (hba : 0 ≤ b_a)
    (h : Assumptions.PackedTrainingIncompressible c ch T Γ K b_w b_a) :
    Assumptions.PackedActivationsIncompressible c ch Γ K b_w b_a := by
  intro R hR hΓ hK
  obtain ⟨s, -, hs⟩ := h 2 (splitOff c.D R)
  have h0 := hs 0 (by rw [splitOff_gates_zero hR]; exact ⟨hR, hΓ, hK⟩)
  rw [splitOff_gates_zero hR] at h0
  exact le_of_max_charge_left hba h0.1

open Classical in
/-- **The partial-sum law with dark gates, in reads, at most `k` products to a gate**: each dark gate is a small IU of
its own, whose partial sums are within its reads. -/
theorem PackedIncompressibleContainedMatMulOn : Guarantees.PackedIncompressibleContainedMatMulOn := by
  intro k Γ α K b_w b_a μ c d ch T dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he ⟨n, P, hP⟩ hB hX h
  obtain ⟨s, hsum, hs⟩ := h _ (withDark M P)
  have hcap := withDark_caps hα hαK hΓ hP hB hX
  have hsP : ∀ u, b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
      b_a * s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u)) ≤ novelReads d (P.gates u) := fun u => by
    have h' := le_of_max_charge_right ha.le (hs (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u))
      ⟨(withDark M P).gates_subset_computed _, (hcap _).1, (hcap _).2⟩).1
    rw [withDark_gates_inl] at h'
    exact h'.trans (by exact_mod_cast novelReads_piece hB u)
  have hsZ : ∀ v : {v // v ∈ M.dark},
      b_a * (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ) ≤ novelReads c {v.1} := fun v => by
    have h' := le_of_max_charge_right ha.le (hs (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v))
      ⟨(withDark M P).gates_subset_computed _, (hcap _).1, (hcap _).2⟩).1
    rw [withDark_gates_inr] at h'
    have : 0 ≤ b_w * ((ch.weightsUsed {v.1}).card : ℝ) := by positivity
    linarith
  have hZ : b_a * ∑ v : {v // v ∈ M.dark}, (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ) ≤
      ∑ v ∈ M.dark, (novelReads c {v} : ℝ) := by
    rw [mul_sum]
    calc ∑ v : {v // v ∈ M.dark}, b_a * (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ)
        ≤ ∑ v : {v // v ∈ M.dark}, (novelReads c {v.1} : ℝ) := sum_le_sum fun v _ => hsZ v
      _ = ∑ v ∈ M.dark, (novelReads c {v} : ℝ) := sum_coe_sort M.dark fun g => (novelReads c {g} : ℝ)
  have h₀ := matMulOn_of_counts_packed hα hK hw ha hdd hαK hμ₀ hμ₁ hd he hP hB
    (activationsIncompressible_of_training_packed ha.le h) (partialSums_count_packed ha.le hsum hsP)
  exact h₀.trans (by linarith [mul_le_mul_of_nonneg_left hZ (by linarith : (0 : ℝ) ≤ 1 - μ)])

/-- **Training is slowed, with dark gates, at most `k` products to a gate**: the chain's products off the books are in
dark gates, at most `k` to a gate (`card_products_le_dark_packed`). -/
theorem PackedIncompressibleContainedTrainingOn : Guarantees.PackedIncompressibleContainedTrainingOn := by
  intro k Γ α K b_w b_a μ κ c d ch T dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he hP hB hX h hκ
  exact training_of_counts (le_max_of_le_right (by positivity))
    (PackedIncompressibleContainedMatMulOn k Γ α K b_w b_a μ c d ch T dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he
      hP hB hX h) hκ (by exact_mod_cast card_products_le_dark_packed M ch)

/-- At `k = 1` the packed training assumption is the per-product one. -/
theorem packedTrainingIncompressible_toPacked {c : Declaration} {ch : Chain c.D} {T : Threading c} {Γ : ℕ}
    {K b_w b_a : ℝ} :
    Assumptions.PackedTrainingIncompressible c ch.toPacked T Γ K b_w b_a ↔
      Assumptions.TrainingIncompressible c ch T Γ K b_w b_a := by
  first
  | exact Iff.rfl
  | classical
    have hcard : ∀ R : Finset (Fin c.D.N),
        ((ch.toPacked.inputsUsed R).filter fun x => ¬ ch.toPacked.LinkedWhole R x.1 x.2.1 x.2.2).card =
          ((ch.inputsUsed R).filter fun x => ¬ ch.LinkedWhole R x.1 x.2.1 x.2.2).card := fun R => by
      show ((ch.inputsUsed R).filter fun x => ¬ ch.toPacked.LinkedWhole R x.1 x.2.1 x.2.2).card = _
      rw [Finset.filter_congr fun x _ => not_congr (toPacked_linkedWhole ch R x.1 x.2.1 x.2.2)]
    unfold Assumptions.PackedTrainingIncompressible Assumptions.TrainingIncompressible
    simp only [hcard]
    exact Iff.rfl

/-- **The per-product partial-sum law with dark gates is the case `k = 1`.** -/
theorem incompressibleContainedMatMulOn_of_packed (h : Guarantees.PackedIncompressibleContainedMatMulOn) :
    Guarantees.IncompressibleContainedMatMulOn :=
  fun Γ α K b_w b_a μ c d ch T dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he hP hB hX hT =>
    h 1 Γ α K b_w b_a μ c d ch.toPacked T dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he hP hB hX
      (packedTrainingIncompressible_toPacked.2 hT)

/-- **The per-product training form with dark gates is the case `k = 1`**, where `2 k #dark` is `2 #dark`. -/
theorem incompressibleContainedTrainingOn_of_packed (h : Guarantees.PackedIncompressibleContainedTrainingOn) :
    Guarantees.IncompressibleContainedTrainingOn := by
  intro Γ α K b_w b_a μ κ c d ch T dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he hP hB hX hT hκ
  have h₁ := h 1 Γ α K b_w b_a μ κ c d ch.toPacked T dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he hP hB hX
    (packedTrainingIncompressible_toPacked.2 hT) hκ
  simpa only [Nat.cast_one, one_mul] using h₁

end Nci.SecurityProofs
