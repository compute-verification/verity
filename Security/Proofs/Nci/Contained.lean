import Models.Nci.Guarantees.Contained
import Proofs.Nci.Reads
import Proofs.Nci.Pull
import Proofs.Nci.Support

/-!
# Proofs of the partial-sum law on a reference circuit

Support form gives the reference circuit's incompressibility in reads (`activationsIncompressible_of_support`,
`partialSumsIncompressible_of_support`). Then pull a compliant division of the containing declaration back to `c`
(`pull`): each piece has no more gates than its IU and reads no more novel information (`novelReads_pull`), so the
pieces form a division of `c` into IUs of at most `Γ` gates reading at most `αΓ` each. The reads form of the law
(`UpdatedMatMulReads`), applied to `c` with that division, bounds the pieces' total reads from below by `G · f(αΓ)`;
the IUs read at least as much, and at most `α size(d)` in total.

Through a partial containment (`ContainedMatMulOn`), the pieces divide only the gates `S` that `d` computes. Adding each
dark gate as a part of its own (`withDark`) divides all of `c`, and the partial-sum count runs over that division: the
dark parts absorb partial sums, `Z` in total, which their values must hold, so `b_a Z ≤ width(dark)`. The reads count
runs over the pieces alone, each within its IU's reads. A second per-piece bound counts products by columns, at most
`e_max` per input entry used at a position that is not full (`card_products_le_columns`, `max_mul_sum_le`).
-/

namespace Nci.SecurityProofs

open Verity Finset

/-- **Support form gives partial-sum incompressibility**: an IU's gates take at most `2^novelReads` values. -/
theorem partialSumsIncompressible_of_support {D : Declaration} {θv : D.D.Transcript} {c : Chain D.D} {Γ : ℕ}
    {K b_w b_a : ℝ} (h : Assumptions.PartialSumsSupport D θv c Γ K b_w b_a) :
    Assumptions.PartialSumsIncompressible D c Γ K b_w b_a := by
  intro n P hP
  obtain ⟨s, hsum, hs⟩ := h n P hP
  refine ⟨s, hsum, fun u => ?_⟩
  have hc : (support D θv (P.gates u) : ℝ) ≤ (2 : ℝ) ^ (novelReads D (P.gates u) : ℝ) := by
    rw [Real.rpow_natCast]
    exact_mod_cast support_le D θv (P.gates_subset_computed u) (subset_refl _)
  exact (Real.rpow_le_rpow_left_iff (by norm_num : (1 : ℝ) < 2)).mp ((hs u).trans hc)

/-- The partial-sum law on a reference circuit, in reads: `UpdatedMatMul` with its assumptions on the contained
circuit `c`. -/
theorem containedMatMul_of_reads (Γ : ℕ) (α b_w b_a : ℝ) (c d : Declaration) (ch : Chain c.D) (dd : ℕ)
    (hα : 0 < α) (hw : 0 < b_w) (ha : 0 < b_a) (hd : ∀ ℓ, dd ≤ ch.d ℓ) (hC : c.Contains d) (hP : Complies Γ α d)
    (hA : Assumptions.ActivationsIncompressible c ch Γ (α * Γ) b_w b_a)
    (hPS : Assumptions.PartialSumsIncompressible c ch Γ (α * Γ) b_w b_a) :
    F α (α * Γ) b_w b_a dd * (2 * ch.G) ≤ d.D.size d.D.computed := by
  obtain ⟨M, hθ⟩ := hC
  obtain ⟨n, P, hP⟩ := hP
  have hpiece : ∀ u, (novelReads c ((pull M P).gates u) : ℝ) ≤ novelReads d (P.gates u) := fun u => by
    exact_mod_cast novelReads_pull M hθ P u
  have hcap : ∀ u, c.D.size ((pull M P).gates u) ≤ Γ ∧ (novelReads c ((pull M P).gates u) : ℝ) ≤ α * Γ := fun u =>
    ⟨(card_pull_gates_le u).trans (hP u).1,
      ((hpiece u).trans (hP u).2).trans (mul_le_mul_of_nonneg_left (by exact_mod_cast (hP u).1) hα.le)⟩
  have hreads := UpdatedMatMulReads Γ (α * Γ) b_w b_a c ch dd n (pull M P) hw ha hd hcap hA hPS
  have htotal : ch.G * readFactor (α * Γ) b_w b_a dd ≤ α * d.D.size d.D.computed :=
    calc ch.G * readFactor (α * Γ) b_w b_a dd ≤ ∑ u, (novelReads c ((pull M P).gates u) : ℝ) := hreads
      _ ≤ ∑ u, (novelReads d (P.gates u) : ℝ) := sum_le_sum fun u _ => hpiece u
      _ ≤ ∑ u, α * (d.D.size (P.gates u) : ℝ) := sum_le_sum fun u _ => (hP u).2
      _ = α * d.D.size d.D.computed := by
          rw [← mul_sum]
          congr 1
          exact_mod_cast P.sum_card_gates
  rw [F_eq_readFactor]
  calc readFactor (α * Γ) b_w b_a dd / (2 * α) * (2 * ch.G) = ch.G * readFactor (α * Γ) b_w b_a dd / α := by
        field_simp
    _ ≤ α * d.D.size d.D.computed / α := div_le_div_of_nonneg_right htotal hα.le
    _ = d.D.size d.D.computed := by field_simp

theorem ContainedMatMul : Guarantees.ContainedMatMul := by
  intro Γ α b_w b_a c d θv ch dd hα hw ha hd hC hP hA hPS
  exact containedMatMul_of_reads Γ α b_w b_a c d ch dd hα hw ha hd hC hP
    (activationsIncompressible_of_support hw.le ha.le hA) (partialSumsIncompressible_of_support hPS)

theorem ContainedTraining : Guarantees.ContainedTraining := by
  intro Γ α b_w b_a κ c d θv ch dd hα hw ha hd hC hP hA hPS hκ
  have hF := ContainedMatMul Γ α b_w b_a c d θv ch dd hα hw ha hd hC hP hA hPS
  have hF0 : 0 ≤ F α (α * Γ) b_w b_a dd := F_nonneg hα.le (by positivity) hw.le ha.le (by positivity)
  calc κ * F α (α * Γ) b_w b_a dd * c.D.size c.D.computed
      = F α (α * Γ) b_w b_a dd * (κ * c.D.size c.D.computed) := by ring
    _ ≤ F α (α * Γ) b_w b_a dd * (2 * ch.G) := mul_le_mul_of_nonneg_left hκ hF0
    _ ≤ d.D.size d.D.computed := hF

/-- Support form bounds a set's charge by its novel reads: its values are functions of what it reads. -/
theorem le_novelReads_of_support {D : Declaration} {θv : D.D.Transcript} {R : Finset (Fin D.D.N)} {x : ℝ}
    (hR : R ⊆ D.D.computed) (h : (2 : ℝ) ^ x ≤ support D θv R) : x ≤ novelReads D R := by
  have hc : (support D θv R : ℝ) ≤ (2 : ℝ) ^ (novelReads D R : ℝ) := by
    rw [Real.rpow_natCast]
    exact_mod_cast support_le D θv hR (subset_refl _)
  exact (Real.rpow_le_rpow_left_iff (by norm_num : (1 : ℝ) < 2)).mp (h.trans hc)

/-- Support form bounds a set's charge by its width: it holds no more values than its bits. -/
theorem le_width_of_support {D : Declaration} {θv : D.D.Transcript} {R : Finset (Fin D.D.N)} {x : ℝ}
    (h : (2 : ℝ) ^ x ≤ support D θv R) : x ≤ D.D.width R := by
  have hc : (support D θv R : ℝ) ≤ (2 : ℝ) ^ (D.D.width R : ℝ) := by
    rw [Real.rpow_natCast]
    exact_mod_cast support_le_width D θv R
  exact (Real.rpow_le_rpow_left_iff (by norm_num : (1 : ℝ) < 2)).mp (h.trans hc)

section Dark

variable {c d : Declaration} (M : PartialContainment c d) {n : ℕ} (P : Partition d.D n)

/-- The part of `c`'s gate `g` when each dark gate joins the pieces of `P` as a part of its own. -/
def darkUnit (g : Fin c.D.N) : Option (Fin n ⊕ {v // v ∈ M.dark}) :=
  if g ∈ M.S then (P.unit (M.emb g)).map Sum.inl else if h : g ∈ M.dark then some (Sum.inr ⟨g, h⟩) else none

variable {M P}

theorem darkUnit_eq_none (g : Fin c.D.N) : darkUnit M P g = none ↔ c.D.isInput g = true := by
  unfold darkUnit
  split_ifs with hS hX
  · have hc := M.S_computed hS
    simp only [Circuit.computed, mem_filter, mem_univ, true_and] at hc
    rw [Option.map_eq_none_iff, P.input_iff, M.computed_emb g hS, hc]
  · have hc := (mem_dark.1 hX).1
    simp only [Circuit.computed, mem_filter, mem_univ, true_and] at hc
    simp [hc]
  · refine iff_of_true rfl ?_
    by_contra hne
    exact hX (mem_dark.2 ⟨by simpa [Circuit.computed] using hne, hS⟩)

variable (M P) in
/-- **The pieces with the dark gates**: the division of `c` into the pieces of `P` and one part for each dark gate. -/
noncomputable def withDark : Partition c.D (Fintype.card (Fin n ⊕ {v // v ∈ M.dark})) where
  unit g := (darkUnit M P g).map (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}))
  input_iff g := by rw [Option.map_eq_none_iff, darkUnit_eq_none]

theorem mem_withDark_gates {i : Fin n ⊕ {v // v ∈ M.dark}} {g : Fin c.D.N} :
    g ∈ (withDark M P).gates (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) i) ↔ darkUnit M P g = some i := by
  rw [Partition.mem_gates]
  exact (Option.map_injective (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark})).injective).eq_iff
    (a := darkUnit M P g) (b := some i)

theorem withDark_gates_inl (u : Fin n) :
    (withDark M P).gates (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u)) = piece M P u := by
  ext g
  rw [mem_withDark_gates, mem_piece, Partition.mem_gates]
  unfold darkUnit
  split_ifs with hS hX <;> simp [hS]

theorem withDark_gates_inr (v : {v // v ∈ M.dark}) :
    (withDark M P).gates (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) = {v.1} := by
  ext g
  rw [mem_withDark_gates, mem_singleton]
  unfold darkUnit
  split_ifs with hS hX
  · have hne : g ≠ v.1 := fun h => (mem_dark.1 v.2).2 (h ▸ hS)
    simp [hne]
  · constructor
    · intro h
      simp only [Option.some.injEq, Sum.inr.injEq] at h
      exact congrArg Subtype.val h
    · intro h
      subst h
      rfl
  · have hne : g ≠ v.1 := fun h => hX (h ▸ v.2)
    simp [hne]

variable (M) in
theorem sum_withDark {β : Type*} [AddCommMonoid β] (f : Fin (Fintype.card (Fin n ⊕ {v // v ∈ M.dark})) → β) :
    ∑ i, f i = ∑ u, f (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u)) +
      ∑ v : {v // v ∈ M.dark}, f (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) := by
  rw [← Equiv.sum_comp (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark})) f, Fintype.sum_sum_type]

/-- **Products off the books are dark gates**: the chain's products number at most those of `S` plus the dark gates. -/
theorem card_products_le_dark (M : PartialContainment c d) (ch : Chain c.D) :
    ch.G ≤ (univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
      ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card + M.dark.card := by
  have huniv : (univ : Finset ((ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ))).card = ch.G := by
    simp only [card_univ, Fintype.card_sigma, Fintype.card_prod, Fintype.card_fin, Chain.G]
    exact sum_congr rfl fun _ _ => by ring
  rw [← huniv, ← card_filter_add_card_filter_not (s := univ)
    fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) => ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S]
  refine Nat.add_le_add_left (card_le_card_of_injOn (fun p => ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2)
    (fun p hp => ?_) fun p _ q _ h => ch.prod_inj h) _
  simp only [mem_coe, mem_filter, mem_univ, true_and] at hp
  exact mem_coe.2 (mem_dark.2 ⟨by simpa [Circuit.computed] using ch.prod_computed p.1 p.2.1 p.2.2.1 p.2.2.2, hp⟩)

end Dark

open Classical in
/-- **The partial-sum law with dark gates.** -/
theorem ContainedMatMulOn : Guarantees.ContainedMatMulOn := by
  intro Γ α K b_w b_a μ c d θv ch dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he ⟨n, P, hP⟩ hX hA hPS
  have hdpos : (0 : ℝ) < dd := by exact_mod_cast hdd
  have hcapP : ∀ u, c.D.size (piece M P u) ≤ Γ ∧ (novelReads c (piece M P u) : ℝ) ≤ K := fun u =>
    ⟨(card_piece_le u).trans (hP u).1, ((Nat.cast_le.2 (novelReads_piece u)).trans (hP u).2).trans
      ((mul_le_mul_of_nonneg_left (by exact_mod_cast (hP u).1) hα).trans hαK)⟩
  have hcap : ∀ i, c.D.size ((withDark M P).gates i) ≤ Γ ∧ (novelReads c ((withDark M P).gates i) : ℝ) ≤ K := by
    intro i
    obtain ⟨j, rfl⟩ := (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark})).surjective i
    rcases j with u | v
    · rw [withDark_gates_inl]
      exact hcapP u
    · rw [withDark_gates_inr]
      exact ⟨(card_singleton v.1).le.trans hΓ, hX v.1 v.2⟩
  obtain ⟨s, hsum, hs⟩ := hPS _ (withDark M P) hcap
  have hAI := activationsIncompressible_of_support (Γ := Γ) hw.le ha.le hA
  have hr : ∀ u, (novelReads c (piece M P u) : ℝ) ≤ novelReads d (P.gates u) := fun u => by
    exact_mod_cast novelReads_piece u
  have hiu := fun u => iu_reads c ch Γ K b_w b_a dd hd ha.le hAI (piece M P u) (piece_subset_computed u)
    (hcapP u).1 (hcapP u).2
  -- the charge of each piece, within its IU's reads, and of each dark gate, within its width
  have hsP : ∀ u, b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
      b_a * s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u)) ≤ novelReads d (P.gates u) := fun u => by
    have h := le_novelReads_of_support ((withDark M P).gates_subset_computed _)
      (hs (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u)))
    rw [withDark_gates_inl] at h
    exact h.trans (hr u)
  have hsZ : ∀ v : {v // v ∈ M.dark},
      b_a * (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ) ≤ c.D.w v.1 := fun v => by
    have h := le_width_of_support (hs (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)))
    rw [withDark_gates_inr] at h
    simp only [Circuit.width, sum_singleton] at h
    have : 0 ≤ b_w * ((ch.weightsUsed {v.1}).card : ℝ) := by positivity
    linarith
  -- the split entries' contributions, over the pieces and the dark parts
  have hY : ∑ u, ((outputsOther ch (piece M P u)).card : ℝ) ≤
      2 * (∑ u, (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u)) : ℝ) +
        ∑ v : {v // v ∈ M.dark}, (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ)) := by
    have h := (sum_card_outputsOther_le ch (withDark M P)).trans (Nat.mul_le_mul_left 2 hsum)
    rw [sum_withDark M, sum_withDark M] at h
    simp only [withDark_gates_inl, withDark_gates_inr] at h
    exact_mod_cast le_trans (Nat.le_add_right _ _) h
  have h₁ : ∑ u, (b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
      b_a * (dd * ((univ.filter (ch.Full (piece M P u))).card : ℝ) +
        ((ch.inputsUsed (piece M P u)).filter fun x => ¬ ch.Full (piece M P u) x.2.1).card)) ≤
      ∑ u, (novelReads d (P.gates u) : ℝ) :=
    sum_le_sum fun u _ => (hiu u).trans (hr u)
  have h₂ : ∑ u, (b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
      b_a / 2 * (outputsOther ch (piece M P u)).card) ≤
      ∑ u, (novelReads d (P.gates u) : ℝ) +
        b_a * ∑ v : {v // v ∈ M.dark}, (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ) := by
    have e₁ : ∑ u, (b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
        b_a * s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u))) =
        ∑ u, b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
          b_a * ∑ u, (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inl u)) : ℝ) := by
      rw [sum_add_distrib, mul_sum]
    have e₂ : ∑ u, (b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
        b_a / 2 * (outputsOther ch (piece M P u)).card) =
        ∑ u, b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
          b_a / 2 * ∑ u, ((outputsOther ch (piece M P u)).card : ℝ) := by
      rw [sum_add_distrib, mul_sum]
    have h₃ := sum_le_sum fun u (_ : u ∈ univ) => hsP u
    rw [e₁] at h₃
    have h₄ := mul_le_mul_of_nonneg_left hY (by positivity : (0 : ℝ) ≤ b_a / 2)
    rw [e₂]
    linarith
  have hW := mu_weighting_budget univ hμ₀ hμ₁ h₁ h₂
  have hM := max_mul_sum_le univ hw ha hdpos hK hμ₀ hμ₁ (le_max_left (emax : ℝ) (K / (4 * b_w * dd)))
    (le_max_right (emax : ℝ) (K / (4 * b_w * dd))) (fun _ _ => by positivity) (fun _ _ => by positivity)
    (fun _ _ => by positivity) (fun _ _ => by positivity) (fun _ _ => by positivity)
    (fun u _ => (hiu u).trans (hcapP u).2) (fun u _ => card_products_le ch (piece M P u))
    (fun u _ => by exact_mod_cast card_products_le_columns ch (piece M P u) he) hW
  have hrsum : ∑ u, (novelReads d (P.gates u) : ℝ) ≤ α * d.D.size d.D.computed :=
    calc ∑ u, (novelReads d (P.gates u) : ℝ) ≤ ∑ u, α * (d.D.size (P.gates u) : ℝ) := sum_le_sum fun u _ => (hP u).2
      _ = α * d.D.size d.D.computed := by
          rw [← mul_sum]
          congr 1
          exact_mod_cast P.sum_card_gates
  have hZ : b_a * ∑ v : {v // v ∈ M.dark}, (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ) ≤
      c.D.width M.dark := by
    rw [mul_sum]
    calc ∑ v : {v // v ∈ M.dark}, b_a * (s (Fintype.equivFin (Fin n ⊕ {v // v ∈ M.dark}) (Sum.inr v)) : ℝ)
        ≤ ∑ v : {v // v ∈ M.dark}, (c.D.w v.1 : ℝ) := sum_le_sum fun v _ => hsZ v
      _ = c.D.width M.dark := by
          rw [Circuit.width, Nat.cast_sum]
          exact sum_coe_sort M.dark fun g => (c.D.w g : ℝ)
  have hGS : ((univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
      ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card : ℝ) =
      ∑ u, ((univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
        ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ piece M P u).card : ℝ) := by
    exact_mod_cast (sum_card_filter_piece (M := M) (P := P) univ
      fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) => ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2).symm
  rw [hGS]
  refine hM.trans ?_
  have h1μ : 0 ≤ 1 - μ := by linarith
  nlinarith [mul_le_mul_of_nonneg_left hZ h1μ]

/-- The training form of a product bound `m G ≤ R`: when `κ S ≤ 2 G'` and `G' ≤ G + X`, `m (κ S − 2 X) ≤ 2 R`. -/
theorem training_of_counts {m G X G' S κ R : ℝ} (hm : 0 ≤ m) (h : m * G ≤ R) (hκ : κ * S ≤ 2 * G')
    (hG : G' ≤ G + X) : m * (κ * S - 2 * X) ≤ 2 * R := by
  nlinarith [mul_le_mul_of_nonneg_left (show κ * S - 2 * X ≤ 2 * G by linarith) hm]

/-- **Training is slowed, by the partial-sum law, with dark gates**: the chain's products off the books are dark gates
(`card_products_le_dark`). -/
theorem ContainedTrainingOn : Guarantees.ContainedTrainingOn := by
  intro Γ α K b_w b_a μ κ c d θv ch dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he hP hX hA hPS hκ
  exact training_of_counts (le_max_of_le_right (by positivity))
    (ContainedMatMulOn Γ α K b_w b_a μ c d θv ch dd emax M hα hK hw ha hdd hΓ hαK hμ₀ hμ₁ hd he hP hX hA hPS) hκ
    (by exact_mod_cast card_products_le_dark M ch)

end Nci.SecurityProofs
