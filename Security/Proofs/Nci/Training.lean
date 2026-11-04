import Definitions.Nci.Guarantees.Training
import Proofs.Nci.Threads
import Proofs.Nci.Contained
import Proofs.Nci.Packed

/-!
# Proofs of NCI's guarantees under training's incompressibility

`TrainingIncompressible` is one law over divisions of the reference circuit. A small IU `R` is a division of `c` into
`R` and the rest (`splitOff`), so the law gives `WeightsIncompressible` and `ActivationsIncompressible` on `R`; on a
division into small IUs it gives `PartialSumsIncompressible`. Conversely the three give the law: on a division with an IU
that is not small, charge every partial sum to that IU (`trainingIncompressible_iff`). The guarantees then follow from the
thread and partial-sum proofs, which read only those three.

Through a partial containment each dark gate is an IU of its own (`withDark`), so the law bounds the partial sums it
absorbs by its reads (`IncompressibleContainedMatMulOn`). At `μ = 1` that budget has no weight, and the activations' half
alone bounds the products the declaration computes (`ContainedFanOutOn`, the case `k = 1` of `PackedContainedFanOutOn`).
-/

namespace Nci.SecurityProofs

open Verity Finset

/-- The division of `C` into the gates of `R` (unit 0) and its other computed gates (unit 1). -/
def splitOff (C : Circuit) (R : Finset (Fin C.N)) : Partition C 2 where
  unit g := if C.isInput g = true then none else if g ∈ R then some 0 else some 1
  input_iff g := by
    by_cases h : C.isInput g = true
    · simp [h]
    · by_cases hR : g ∈ R <;> simp [h, hR]

theorem splitOff_unit (C : Circuit) (R : Finset (Fin C.N)) (g : Fin C.N) :
    (splitOff C R).unit g = if C.isInput g = true then none else if g ∈ R then some 0 else some 1 := rfl

theorem splitOff_gates_zero {C : Circuit} {R : Finset (Fin C.N)} (hR : R ⊆ C.computed) :
    (splitOff C R).gates 0 = R := by
  ext g
  rw [Partition.mem_gates, splitOff_unit]
  by_cases h : C.isInput g = true
  · have hg : g ∉ R := fun hg => by
      have := (mem_filter.1 (hR hg)).2
      rw [h] at this
      cases this
    simp [h, hg]
  · by_cases hg : g ∈ R <;> simp [h, hg]

section Charges

variable {b_w b_a w r : ℝ} {a b : ℕ}

theorem le_of_max_charge_left (hba : 0 ≤ b_a) (h : b_w * w + b_a * ((max a b : ℕ) : ℝ) ≤ r) :
    b_w * w + b_a * a ≤ r := by
  have hab : (a : ℝ) ≤ ((max a b : ℕ) : ℝ) := Nat.cast_le.2 (le_max_left a b)
  have := mul_le_mul_of_nonneg_left hab hba
  linarith

theorem le_of_max_charge_right (hba : 0 ≤ b_a) (h : b_w * w + b_a * ((max a b : ℕ) : ℝ) ≤ r) :
    b_w * w + b_a * b ≤ r := by
  have hab : (b : ℝ) ≤ ((max a b : ℕ) : ℝ) := Nat.cast_le.2 (le_max_right a b)
  have := mul_le_mul_of_nonneg_left hab hba
  linarith

theorem max_charge_le (ha : b_w * w + b_a * a ≤ r) (hb : b_w * w + b_a * b ≤ r) :
    b_w * w + b_a * ((max a b : ℕ) : ℝ) ≤ r := by
  rcases le_total a b with h | h
  · rwa [max_eq_right h]
  · rwa [max_eq_left h]

end Charges

section Training

variable {c : RegisteredCircuit} {ch : Chain c.D} {T : Threading c} {Γ : ℕ} {K b_w b_a : ℝ}

open Classical in
/-- The law on one small IU: divide `c` into it and the rest. -/
theorem training_piece (h : Assumptions.TrainingIncompressible c ch T Γ K b_w b_a) {R : Finset (Fin c.D.N)}
    (hR : Small c Γ K R) :
    ∃ s : ℕ,
      b_w * (ch.weightsUsed R).card +
          b_a * ((max ((ch.inputsUsed R).filter fun x => ¬ ch.LinkedWhole R x.1 x.2.1 x.2.2).card s : ℕ) : ℝ)
        ≤ novelReads c R ∧
      b_w * (T.weightsUsed R).card + b_a * ∑ t, (T.threadReads R t).card ≤ novelReads c R := by
  obtain ⟨s, -, hs⟩ := h 2 (splitOff c.D R)
  have h0 := hs 0 (by rwa [splitOff_gates_zero hR.1])
  rw [splitOff_gates_zero hR.1] at h0
  exact ⟨s 0, h0⟩

/-- **Training's incompressibility gives the weights'**. -/
theorem weightsIncompressible_of_training (h : Assumptions.TrainingIncompressible c ch T Γ K b_w b_a) :
    Assumptions.WeightsIncompressible c T Γ K b_w b_a := fun R hR => by
  obtain ⟨_, -, hW⟩ := training_piece h hR
  exact hW

/-- **Training's incompressibility gives the activations'**. -/
theorem activationsIncompressible_of_training (hba : 0 ≤ b_a)
    (h : Assumptions.TrainingIncompressible c ch T Γ K b_w b_a) :
    Assumptions.ActivationsIncompressible c ch Γ K b_w b_a := by
  intro R hR hΓ hK
  obtain ⟨_, hA, -⟩ := training_piece h (R := R) ⟨hR, hΓ, hK⟩
  exact le_of_max_charge_left hba hA

/-- **Training's incompressibility gives the partial sums'**. -/
theorem partialSumsIncompressible_of_training (hba : 0 ≤ b_a)
    (h : Assumptions.TrainingIncompressible c ch T Γ K b_w b_a) :
    Assumptions.PartialSumsIncompressible c ch Γ K b_w b_a := by
  intro n P hP
  obtain ⟨s, hsum, hs⟩ := h n P
  exact ⟨s, hsum, fun u => le_of_max_charge_right hba
    (hs u ⟨P.gates_subset_computed u, (hP u).1, (hP u).2⟩).1⟩

/-- **The three give training's incompressibility**: on a division into small IUs, partial-sum incompressibility
charges the partial sums; on one with an IU that is not small, that IU takes them all. -/
theorem trainingIncompressible_of_reads (hba : 0 ≤ b_a) (hW : Assumptions.WeightsIncompressible c T Γ K b_w b_a)
    (hA : Assumptions.ActivationsIncompressible c ch Γ K b_w b_a)
    (hPS : Assumptions.PartialSumsIncompressible c ch Γ K b_w b_a) :
    Assumptions.TrainingIncompressible c ch T Γ K b_w b_a := by
  intro n P
  by_cases hall : ∀ u, c.D.size (P.gates u) ≤ Γ ∧ (novelReads c (P.gates u) : ℝ) ≤ K
  · obtain ⟨s, hsum, hs⟩ := hPS n P hall
    exact ⟨s, hsum, fun u hu => ⟨max_charge_le (hA _ hu.1 hu.2.1 hu.2.2) (hs u), hW _ hu⟩⟩
  · push_neg at hall
    obtain ⟨u₀, hu₀⟩ := hall
    refine ⟨fun u => if u = u₀ then
        ∑ x : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ), (ch.spread P x.1 x.2.1 x.2.2 - 1) else 0,
      by simp, fun u hu => ⟨?_, hW _ hu⟩⟩
    have hne : u ≠ u₀ := by
      rintro rfl
      exact absurd hu.2.2 (not_le.2 (hu₀ hu.2.1))
    have hAu := hA _ hu.1 hu.2.1 hu.2.2
    refine max_charge_le hAu ?_
    simp only [if_neg hne, Nat.cast_zero, mul_zero, add_zero]
    exact (le_add_of_nonneg_right (mul_nonneg hba (Nat.cast_nonneg _))).trans hAu

/-- **Training's incompressibility is exactly the three** that the thread and partial-sum proofs read. -/
theorem trainingIncompressible_iff (hba : 0 ≤ b_a) :
    Assumptions.TrainingIncompressible c ch T Γ K b_w b_a ↔
      Assumptions.WeightsIncompressible c T Γ K b_w b_a ∧ Assumptions.ActivationsIncompressible c ch Γ K b_w b_a ∧
        Assumptions.PartialSumsIncompressible c ch Γ K b_w b_a :=
  ⟨fun h => ⟨weightsIncompressible_of_training h, activationsIncompressible_of_training hba h,
      partialSumsIncompressible_of_training hba h⟩,
    fun ⟨hW, hA, hPS⟩ => trainingIncompressible_of_reads hba hW hA hPS⟩

/-- **The support forms give training's incompressibility**: `ActivationsSupport` at cap `K + b_w + b_a` and
`PartialSumsSupport`, with `WeightsIncompressible`, for any committed values of the baseline. -/
theorem trainingIncompressible_of_support {θv : c.D.Transcript} (hw : 0 ≤ b_w) (hba : 0 ≤ b_a)
    (hW : Assumptions.WeightsIncompressible c T Γ K b_w b_a)
    (hA : Assumptions.ActivationsSupport c θv ch (K + b_w + b_a) b_w b_a)
    (hPS : Assumptions.PartialSumsSupport c θv ch Γ K b_w b_a) :
    Assumptions.TrainingIncompressible c ch T Γ K b_w b_a :=
  trainingIncompressible_of_reads hba hW (activationsIncompressible_of_support hw hba hA)
    (partialSumsIncompressible_of_support hPS)

end Training

theorem IncompressibleThreadedThrottled : Guarantees.IncompressibleThreadedThrottled := by
  intro c k b_w b_a r d ch T dd e M hc hbw hba hdd hP hB hT h
  exact ThreadedThrottled c k b_w b_a r d T dd e M hc hbw hba hdd hP hB hT (weightsIncompressible_of_training h)

theorem IncompressibleThreadedThrottledOn : Guarantees.IncompressibleThreadedThrottledOn := by
  intro c k b_w b_a r d ch T dd e M hc hbw hba hdd hP hB hT h
  exact ThreadedThrottledOn c k b_w b_a r d T dd e M hc hbw hba hdd hP hB hT (weightsIncompressible_of_training h)

theorem IncompressibleContainedTraining_rate :
    ∀ (Γ : ℕ) (α b_w b_a κ : ℝ) (c d : RegisteredCircuit) (ch : Chain c.D) (T : Threading c) (dd : ℕ)
      (M : Containment c.D d.D),
      0 < α → 0 < b_w → 0 < b_a → (∀ ℓ, dd ≤ ch.d ℓ) → CompliesRate Γ α d →
      Assumptions.ThetaRegistered c d (PartialContainment.ofContainment M) →
      Assumptions.TrainingIncompressible c ch T Γ (α * Γ) b_w b_a →
      κ * c.D.size c.D.computed ≤ 2 * ch.G →
        κ * F α (α * Γ) b_w b_a dd * c.D.size c.D.computed ≤ d.D.size d.D.computed := by
  intro Γ α b_w b_a κ c d ch T dd M hα hbw hba hd hP hB h hκ
  have hF := containedMatMul_of_reads Γ α b_w b_a c d ch dd M hα hbw hba hd hP hB
    (activationsIncompressible_of_training hba.le h) (partialSumsIncompressible_of_training hba.le h)
  have hF0 : 0 ≤ F α (α * Γ) b_w b_a dd := F_nonneg hα.le (by positivity) hbw.le hba.le (by positivity)
  calc κ * F α (α * Γ) b_w b_a dd * c.D.size c.D.computed
      = F α (α * Γ) b_w b_a dd * (κ * c.D.size c.D.computed) := by ring
    _ ≤ F α (α * Γ) b_w b_a dd * (2 * ch.G) := mul_le_mul_of_nonneg_left hκ hF0
    _ ≤ d.D.size d.D.computed := hF

open Classical in
/-- **The partial-sum law with dark gates, in reads**: each dark gate is a small IU of its own, whose partial sums are
within its reads. -/
theorem IncompressibleContainedMatMulOn_rate :
    ∀ (Γ : ℕ) (α K b_w b_a μ : ℝ) (c d : RegisteredCircuit) (ch : Chain c.D) (T : Threading c) (dd emax : ℕ)
      (M : PartialContainment c d),
      0 ≤ α → 0 < K → 0 < b_w → 0 < b_a → 0 < dd → 1 ≤ Γ → α * Γ ≤ K → 0 ≤ μ → μ ≤ 1 →
      (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → CompliesRate Γ α d → Assumptions.ThetaRegistered c d M →
      (∀ v ∈ M.dark, (novelReads c {v} : ℝ) ≤ K) →
      Assumptions.TrainingIncompressible c ch T Γ K b_w b_a →
        max (min ((1 + √μ) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))))
            (μ * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) *
          (Finset.univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
            ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card ≤
        α * d.D.size d.D.computed + (1 - μ) * ∑ v ∈ M.dark, (novelReads c {v} : ℝ) := by
  intro Γ α K b_w b_a μ c d ch T dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he ⟨n, P, hP⟩ hB hX h
  obtain ⟨s, hsum, hs⟩ := partialSumsIncompressible_of_training ha.le h _ (withDark M P)
    (withDark_caps hα hαK hΓ hP hB hX)
  have hsP : ∀ u, b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
      b_a * s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u)) ≤ novelReads d (P.gates u) := fun u => by
    have h' := hs (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u))
    rw [withDark_gates_inl] at h'
    exact h'.trans (by exact_mod_cast novelReads_piece hB u)
  have hsZ : ∀ v : {v // v ∈ M.dark},
      b_a * (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ) ≤ novelReads c {v.1} := fun v => by
    have h' := hs (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v))
    rw [withDark_gates_inr] at h'
    have : 0 ≤ b_w * ((ch.weightsUsed {v.1}).card : ℝ) := by positivity
    linarith
  have hZ : b_a * ∑ v : {v // v ∈ M.dark}, (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ) ≤
      ∑ v ∈ M.dark, (novelReads c {v} : ℝ) := by
    rw [mul_sum]
    calc ∑ v : {v // v ∈ M.dark}, b_a * (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ)
        ≤ ∑ v : {v // v ∈ M.dark}, (novelReads c {v.1} : ℝ) := sum_le_sum fun v _ => hsZ v
      _ = ∑ v ∈ M.dark, (novelReads c {v} : ℝ) := sum_coe_sort M.dark fun g => (novelReads c {g} : ℝ)
  have h₀ := matMulOn_of_counts hα hK hw ha hdd hαK hμ₀ hμ₁ hd he hP hB
    (activationsIncompressible_of_training ha.le h) (partialSums_count ha.le hsum hsP)
  exact h₀.trans (by linarith [mul_le_mul_of_nonneg_left hZ (by linarith : (0 : ℝ) ≤ 1 - μ)])

theorem IncompressibleContainedTrainingOn_rate :
    ∀ (Γ : ℕ) (α K b_w b_a μ κ : ℝ) (c d : RegisteredCircuit) (ch : Chain c.D) (T : Threading c) (dd emax : ℕ)
      (M : PartialContainment c d),
      0 ≤ α → 0 < K → 0 < b_w → 0 < b_a → 0 < dd → 1 ≤ Γ → α * Γ ≤ K → 0 ≤ μ → μ ≤ 1 →
      (∀ ℓ, dd ≤ ch.d ℓ) → (∀ ℓ, ch.e ℓ ≤ emax) → CompliesRate Γ α d → Assumptions.ThetaRegistered c d M →
      (∀ v ∈ M.dark, (novelReads c {v} : ℝ) ≤ K) →
      Assumptions.TrainingIncompressible c ch T Γ K b_w b_a →
      κ * c.D.size c.D.computed ≤ 2 * ch.G →
        max (min ((1 + √μ) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))))
            (μ * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) *
          (κ * c.D.size c.D.computed - 2 * M.dark.card) ≤
        2 * (α * d.D.size d.D.computed + (1 - μ) * ∑ v ∈ M.dark, (novelReads c {v} : ℝ)) := by
  intro Γ α K b_w b_a μ κ c d ch T dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he hP hB hX h hκ
  exact training_of_counts (le_max_of_le_right (by positivity))
    (IncompressibleContainedMatMulOn_rate Γ α K b_w b_a μ c d ch T dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he hP hB
      hX h) hκ (by exact_mod_cast card_products_le_dark M ch)

theorem IncompressibleContainedTraining : Guarantees.IncompressibleContainedTraining := by
  intro c k b_w b_a κ r d ch T dd M hc hk hbw hba hd hP hB h hκ
  have h' := IncompressibleContainedTraining_rate c (k / c) b_w b_a κ r d ch T dd M
    (div_pos (Nat.cast_pos.2 hk) (Nat.cast_pos.2 hc)) hbw hba hd (compliesRate_of_complies hP) hB
    (by rwa [rate_mul _ hc]) hκ
  rwa [rate_mul _ hc] at h'

theorem IncompressibleContainedMatMulOn : Guarantees.IncompressibleContainedMatMulOn := by
  intro c k K b_w b_a μ r d ch T dd emax M hK hw ha hdd hc hkK hμ₀ hμ₁ hd he hP hB hX h
  exact IncompressibleContainedMatMulOn_rate c (k / c) K b_w b_a μ r d ch T dd emax M (by positivity) hK hw ha hdd
    (by omega) (by rwa [rate_mul _ hc]) hμ₀ hμ₁ hd he (compliesRate_of_complies hP) hB hX h

theorem IncompressibleContainedTrainingOn : Guarantees.IncompressibleContainedTrainingOn := by
  intro c k K b_w b_a μ κ r d ch T dd emax M hK hw ha hdd hc hkK hμ₀ hμ₁ hd he hP hB hX h hκ
  exact IncompressibleContainedTrainingOn_rate c (k / c) K b_w b_a μ κ r d ch T dd emax M (by positivity) hK hw ha
    hdd (by omega) (by rwa [rate_mul _ hc]) hμ₀ hμ₁ hd he (compliesRate_of_complies hP) hB hX h hκ

/-- **Products done in the declaration are bounded by their fan-out**: the case `k = 1` of the packed bound, at `μ = 1`,
where the dark gates' budget has weight `1 − μ = 0` and the partial sums are not needed. -/
theorem ContainedFanOutOn : Guarantees.ContainedFanOutOn :=
  containedFanOutOn_of_packed PackedContainedFanOutOn

end Nci.SecurityProofs
