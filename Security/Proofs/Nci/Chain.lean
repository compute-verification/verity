import Definitions.Nci.Guarantees
import Proofs.Core

/-!
# Lemmas 1 and 2: what an IU reads at full and other positions

The draft's proof of its theorem ("Throttling updated matrix multiplications") counts, for each IU `R`, the weight
entries `R` uses, its full positions and the input entries it uses at its other positions. Lemma 1 bounds the products
`R` computes at its full positions, and finds at each of them a whole input vector that `R` reads (`full_positions`).
Lemma 2 says that at its other positions every input entry `R` uses is read, and every output entry `R` contributes to
is split (`other_positions`). With incompressibility, they give the reads the theorem's proof starts from (`iu_reads`).

`training_of_updatedMatMul` derives the bound for training from the bound for updated matrix multiplications.
-/

namespace Nci.SecurityProofs

open Verity Finset

section Lemmas

variable {D : Circuit} (c : Chain D)

/-- An input entry linked to an entry whole in `R` makes `R` compute a whole entry at an earlier multiplication. -/
theorem exists_whole_of_linkedWhole {R : Finset (Fin D.N)} {ℓ : Fin c.L} {t : Fin c.N} {j : Fin (c.d ℓ)}
    (h : c.LinkedWhole R ℓ t j) : ∃ ℓ' < ℓ, ∃ i, c.Whole R ℓ' t i := by
  obtain ⟨_ | k, hk⟩ := ℓ
  · simp [Chain.LinkedWhole] at h
  · exact ⟨⟨k, by omega⟩, by simp [Fin.lt_def], _, h⟩

/-- At a position that is not full for `R`, no input entry is linked to an entry whole in `R`. -/
theorem not_linkedWhole_of_not_full {R : Finset (Fin D.N)} {ℓ : Fin c.L} {t : Fin c.N} (ht : ¬ c.Full R t)
    (j : Fin (c.d ℓ)) : ¬ c.LinkedWhole R ℓ t j := fun h => by
  obtain ⟨ℓ', -, i, hi⟩ := exists_whole_of_linkedWhole c h
  exact ht ⟨ℓ', i, hi⟩

open Classical in
/-- At a full position, `R` uses at least `d` input entries that are not linked to an entry whole in `R`: those of the
first multiplication at which `R` computes a whole entry. -/
theorem le_card_fiber (R : Finset (Fin D.N)) (d : ℕ) (hd : ∀ ℓ, d ≤ c.d ℓ) {t : Fin c.N} (ht : c.Full R t) :
    d ≤ (((c.inputsUsed R).filter fun x => c.Full R x.2.1 ∧ ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).filter
      fun x => x.2.1 = t).card := by
  obtain ⟨ℓ, ⟨i, hi⟩, hmin⟩ := (wellFounded_lt (α := Fin c.L)).has_min {ℓ | ∃ i, c.Whole R ℓ t i}
    (by obtain ⟨ℓ, i, h⟩ := ht; exact ⟨ℓ, i, h⟩)
  refine (hd ℓ).trans ?_
  rw [← Finset.card_fin (c.d ℓ)]
  refine Finset.card_le_card_of_injOn (fun j => (⟨ℓ, t, j⟩ : (ℓ : Fin c.L) × Fin c.N × Fin (c.d ℓ)))
    (fun j _ => ?_) (fun j _ j' _ h => ?_)
  · simp only [Chain.inputsUsed, mem_coe, mem_filter, mem_univ, true_and, and_true]
    refine ⟨⟨i, hi j⟩, ht, fun hl => ?_⟩
    obtain ⟨ℓ', hlt, i', hi'⟩ := exists_whole_of_linkedWhole c hl
    exact hmin ℓ' ⟨i', hi'⟩ hlt
  · simp only [Sigma.mk.inj_iff, heq_eq_eq, Prod.mk.injEq, true_and] at h
    exact h

open Classical in
/-- **Lemma 1 (full positions)** (the draft, "Throttling updated matrix multiplications"): "If `R` uses `p` weight
entries and has `τ` full positions, then `R` computes at most `pτ` products at its full positions, and reads at least
`b_a d` bits of activations or partial sums for each of them." The reads are those
`Assumptions.ActivationsIncompressible` charges: at each full position, `R` uses at least `d` input entries that are not
linked to an entry whole in `R`, `d` the smallest inner dimension. -/
theorem full_positions (R : Finset (Fin D.N)) (d : ℕ) (hd : ∀ ℓ, d ≤ c.d ℓ) :
    (univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
        c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ R ∧ c.Full R g.2.1).card ≤
      (c.weightsUsed R).card * (univ.filter (c.Full R)).card ∧
    d * (univ.filter (c.Full R)).card ≤
      ((c.inputsUsed R).filter fun x => c.Full R x.2.1 ∧ ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).card := by
  constructor
  · rw [← Finset.card_product]
    refine Finset.card_le_card_of_injOn
      (fun g => ((⟨g.1, g.2.2.1, g.2.2.2⟩ : (ℓ : Fin c.L) × Fin (c.e ℓ) × Fin (c.d ℓ)), g.2.1))
      (fun g hg => ?_) (fun g _ g' _ h => ?_)
    · simp only [mem_coe, mem_filter, mem_univ, true_and] at hg
      simp only [mem_coe, mem_product, Chain.weightsUsed, mem_filter, mem_univ, true_and]
      exact ⟨⟨_, hg.1⟩, hg.2⟩
    · obtain ⟨ℓ, t, i, j⟩ := g
      obtain ⟨ℓ', t', i', j'⟩ := g'
      simp only [Prod.mk.injEq, Sigma.mk.inj_iff] at h
      obtain ⟨⟨rfl, h⟩, rfl⟩ := h
      simp only [heq_eq_eq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl⟩ := h
      rfl
  · have hfib := Finset.card_eq_sum_card_fiberwise
      (f := fun x : (ℓ : Fin c.L) × Fin c.N × Fin (c.d ℓ) => x.2.1)
      (s := (c.inputsUsed R).filter fun x => c.Full R x.2.1 ∧ ¬ c.LinkedWhole R x.1 x.2.1 x.2.2)
      (t := univ.filter (c.Full R)) (fun x hx => by simp only [mem_coe, mem_filter] at hx ⊢; exact ⟨mem_univ _, hx.2.1⟩)
    rw [hfib, mul_comm, ← smul_eq_mul, ← Finset.sum_const]
    exact Finset.sum_le_sum fun t ht => le_card_fiber c R d hd (mem_filter.1 ht).2

/-- **Lemma 2 (other positions)** (the draft, "Throttling updated matrix multiplications"): "At positions that are not
full positions of `R`, every entry of `x^{(ℓ-1)}` that `R` uses costs `R` a read of at least `b_a` bits, and every entry
of `y^{(ℓ)}` that `R` contributes products to is split." `R` is an IU of a division `P`; the read is the one
`Assumptions.ActivationsIncompressible` charges, since such an entry is not linked to an entry whole in `R`. -/
theorem other_positions {n : ℕ} (P : Partition D n) (u : Fin n) :
    (∀ ℓ t j, ⟨ℓ, t, j⟩ ∈ c.inputsUsed (P.gates u) → ¬ c.Full (P.gates u) t →
      ¬ c.LinkedWhole (P.gates u) ℓ t j) ∧
    ∀ ℓ t i j, c.prod ℓ t i j ∈ P.gates u → ¬ c.Full (P.gates u) t → c.Split P ℓ t i := by
  refine ⟨fun ℓ t j _ ht => not_linkedWhole_of_not_full c ht j, fun ℓ t i j hj ht ⟨u', hw⟩ => ht ⟨ℓ, i, ?_⟩⟩
  have hu : u' = u := by
    have h1 := (P.mem_gates).1 (hw j)
    have h2 := (P.mem_gates).1 hj
    exact Option.some_injective _ (h1.symm.trans h2)
  exact hu ▸ hw

end Lemmas

open Classical in
/-- **What an IU reads** (the draft's proof of its theorem, "Throttling updated matrix multiplications"): "By the lemmas
and incompressibility, `R` reads at least `b_w p_R + b_a (d τ_R + Q_R)` bits", for an IU `R` of at most `Γ` computed
gates that reads at most `K` bits of novel information, under `Assumptions.ActivationsIncompressible`. `p_R` is the
number of weight entries `R` uses, `τ_R` its number of full positions, `Q_R` the number of input entries it uses at
positions that are not full, and `d` the smallest inner dimension. -/
theorem iu_reads (D : Declaration) (c : Chain D.D) (Γ : ℕ) (K b_w b_a : ℝ) (d : ℕ) (hd : ∀ ℓ, d ≤ c.d ℓ)
    (hb_a : 0 ≤ b_a) (h : Assumptions.ActivationsIncompressible D c Γ K b_w b_a) (R : Finset (Fin D.D.N))
    (hR : R ⊆ D.D.computed) (hΓ : D.D.size R ≤ Γ) (hK : (novelReads D R : ℝ) ≤ K) :
    b_w * (c.weightsUsed R).card +
        b_a * (d * (univ.filter (c.Full R)).card + ((c.inputsUsed R).filter fun x => ¬ c.Full R x.2.1).card) ≤
      novelReads D R := by
  refine le_trans ?_ (h R hR hΓ hK)
  gcongr
  have hdisj : Disjoint ((c.inputsUsed R).filter fun x => c.Full R x.2.1 ∧ ¬ c.LinkedWhole R x.1 x.2.1 x.2.2)
      ((c.inputsUsed R).filter fun x => ¬ c.Full R x.2.1) :=
    Finset.disjoint_filter.2 fun _ _ h1 h2 => h2 h1.1
  have hsub : ((c.inputsUsed R).filter fun x => c.Full R x.2.1 ∧ ¬ c.LinkedWhole R x.1 x.2.1 x.2.2) ∪
      ((c.inputsUsed R).filter fun x => ¬ c.Full R x.2.1) ⊆
      (c.inputsUsed R).filter fun x => ¬ c.LinkedWhole R x.1 x.2.1 x.2.2 := by
    intro x hx
    simp only [mem_union, mem_filter] at hx ⊢
    rcases hx with ⟨hx, -, hl⟩ | ⟨hx, ht⟩
    · exact ⟨hx, hl⟩
    · exact ⟨hx, not_linkedWhole_of_not_full c ht _⟩
  have := (full_positions c R d hd).2
  have hcard := Finset.card_le_card hsub
  rw [Finset.card_union_of_disjoint hdisj] at hcard
  exact_mod_cast (Nat.add_le_add_right this _).trans hcard

/-- `F ≥ 0`: each term inside the minimum is nonnegative. -/
theorem F_nonneg {α K b_w b_a d : ℝ} (hα : 0 ≤ α) (hK : 0 ≤ K) (hw : 0 ≤ b_w) (ha : 0 ≤ b_a) (hd : 0 ≤ d) :
    0 ≤ F α K b_w b_a d := by
  unfold F
  refine mul_nonneg (by positivity) (Real.iSup_nonneg fun μ => le_min ?_ ?_)
  · have := μ.2.1
    positivity
  · have := μ.2.2
    positivity

/-- The bound for training follows from the bound for updated matrix multiplications: each compliant declaration that
implements `T` runs a chain whose `2G` is at least `κ size(T)`, so its size is at least `κ F size(T)`. -/
theorem training_of_updatedMatMul (h : Guarantees.UpdatedMatMul) : Guarantees.Training := by
  intro Γ α b_w b_a κ T d hα hw ha hT hchain
  refine le_iInf₂ fun D ⟨hImp, hC⟩ => ?_
  obtain ⟨c, hd, hκ, hA, hP⟩ := hchain D hImp hC
  have hF := h Γ α b_w b_a D c d hα hw ha hd hC hA hP
  have hF0 : 0 ≤ F α (α * Γ) b_w b_a d := F_nonneg hα.le (by positivity) hw.le ha.le (by positivity)
  have hTpos : (0 : ℝ) < T.D.size T.D.computed := by exact_mod_cast hT
  have hle : κ * F α (α * Γ) b_w b_a d ≤ (D.D.size D.D.computed : ℝ) / T.D.size T.D.computed := by
    rw [le_div_iff₀ hTpos]
    calc κ * F α (α * Γ) b_w b_a d * T.D.size T.D.computed
        = F α (α * Γ) b_w b_a d * (κ * T.D.size T.D.computed) := by ring
      _ ≤ F α (α * Γ) b_w b_a d * (2 * c.G) := mul_le_mul_of_nonneg_left hκ hF0
      _ ≤ D.D.size D.D.computed := hF
  calc ENNReal.ofReal (κ * F α (α * Γ) b_w b_a d)
      ≤ ENNReal.ofReal ((D.D.size D.D.computed : ℝ) / T.D.size T.D.computed) := ENNReal.ofReal_le_ofReal hle
    _ = (D.D.size D.D.computed : ENNReal) / T.D.size T.D.computed := by
        rw [ENNReal.ofReal_div_of_pos hTpos, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]

end Nci.SecurityProofs

theorem Nci.SecurityProofs.FullPositions : Nci.Guarantees.FullPositions := @Nci.SecurityProofs.full_positions

theorem Nci.SecurityProofs.OtherPositions : Nci.Guarantees.OtherPositions := @Nci.SecurityProofs.other_positions
