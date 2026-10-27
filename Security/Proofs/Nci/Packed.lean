import Definitions.Nci.Guarantees.Packed
import Definitions.Nci.Guarantees.Training
import Proofs.Nci.Contained

/-!
# The fan-out bound on a chain with several products to a gate

Lemmas 1 and 2 and the product counts, for a `PackedChain`, and `PackedContainedFanOutOn`. None of them reads `prod_mult`:
every count is of products, the indices `(ℓ, t, i, j)`, and of the entries they use, never of the gates that hold them,
so the fan-out bound is the same at every `k`. An input entry is in at most `e_max` products, one for each row of its
multiplication, whichever gates hold them (`card_products_le_columns_packed`).

`prod_mult` is read only where products are converted into gates: the chain's products off the books number at most
`k` per dark gate, where `Chain`'s are at most one (`card_products_le_dark`, which the `κ` forms read), and a single gate's
charge is at most `k (b_w + b_a)`, where `Chain`'s is at most `b_w + b_a` (`card_prods_le_one`, which the support form
reads). Neither is on the fan-out bound's path.

At `k = 1` the packed chain is the chain (`Chain.toPacked`), the assumption is the per-product one
(`packedActivationsIncompressible_toPacked`), and `ContainedFanOutOn` is the case `k = 1` (`containedFanOutOn_of_packed`).
-/

namespace Nci.SecurityProofs

open Verity Finset

section Lemmas

variable {D : Circuit} {k : ℕ} (c : PackedChain D k)

/-- An input entry linked to an entry whole in `R` makes `R` compute a whole entry at an earlier multiplication. -/
theorem exists_whole_of_linkedWhole_packed {R : Finset (Fin D.N)} {ℓ : Fin c.L} {t : Fin c.N} {j : Fin (c.d ℓ)}
    (h : c.LinkedWhole R ℓ t j) : ∃ ℓ' < ℓ, ∃ i, c.Whole R ℓ' t i := by
  obtain ⟨_ | m, hm⟩ := ℓ
  · simp [PackedChain.LinkedWhole] at h
  · exact ⟨⟨m, by omega⟩, by simp [Fin.lt_def], _, h⟩

/-- At a position that is not full for `R`, no input entry is linked to an entry whole in `R`. -/
theorem not_linkedWhole_of_not_full_packed {R : Finset (Fin D.N)} {ℓ : Fin c.L} {t : Fin c.N} (ht : ¬ c.Full R t)
    (j : Fin (c.d ℓ)) : ¬ c.LinkedWhole R ℓ t j := fun h => by
  obtain ⟨ℓ', -, i, hi⟩ := exists_whole_of_linkedWhole_packed c h
  exact ht ⟨ℓ', i, hi⟩

open Classical in
/-- At a full position, `R` uses at least `d` input entries that are not linked to an entry whole in `R`. -/
theorem le_card_fiber_packed (R : Finset (Fin D.N)) (d : ℕ) (hd : ∀ ℓ, d ≤ c.d ℓ) {t : Fin c.N} (ht : c.Full R t) :
    d ≤ (((c.inputsUsed R).filter fun x => c.Full R x.2.1 ∧ ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).filter
      fun x => x.2.1 = t).card := by
  obtain ⟨ℓ, ⟨i, hi⟩, hmin⟩ := (wellFounded_lt (α := Fin c.L)).has_min {ℓ | ∃ i, c.Whole R ℓ t i}
    (by obtain ⟨ℓ, i, h⟩ := ht; exact ⟨ℓ, i, h⟩)
  refine (hd ℓ).trans ?_
  rw [← Finset.card_fin (c.d ℓ)]
  refine Finset.card_le_card_of_injOn (fun j => (⟨ℓ, t, j⟩ : (ℓ : Fin c.L) × Fin c.N × Fin (c.d ℓ)))
    (fun j _ => ?_) (fun j _ j' _ h => ?_)
  · simp only [PackedChain.inputsUsed, mem_coe, mem_filter, mem_univ, true_and, and_true]
    refine ⟨⟨i, hi j⟩, ht, fun hl => ?_⟩
    obtain ⟨ℓ', hlt, i', hi'⟩ := exists_whole_of_linkedWhole_packed c hl
    exact hmin ℓ' ⟨i', hi'⟩ hlt
  · simp only [Sigma.mk.inj_iff, heq_eq_eq, Prod.mk.injEq, true_and] at h
    exact h

open Classical in
/-- **Lemma 1 (full positions)**, counting products: `R` holds the gates of at most `p τ` products at its full positions,
and uses at least `d` input entries not linked to an entry whole in `R` at each of them. -/
theorem full_positions_packed (R : Finset (Fin D.N)) (d : ℕ) (hd : ∀ ℓ, d ≤ c.d ℓ) :
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
      simp only [mem_coe, mem_product, PackedChain.weightsUsed, mem_filter, mem_univ, true_and]
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
    exact Finset.sum_le_sum fun t ht => le_card_fiber_packed c R d hd (mem_filter.1 ht).2

open Classical in
/-- The output entries `y_t^{(ℓ)}[i]` that `R` contributes products to at positions that are not full for `R`. -/
noncomputable def outputsOtherPacked (R : Finset (Fin D.N)) : Finset ((ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ)) :=
  univ.filter fun x => ¬ c.Full R x.2.1 ∧ ∃ j, c.prod x.1 x.2.1 x.2.2 j ∈ R

open Classical in
/-- **The products of an IU**: `R` holds the gates of at most `p τ + √(p Q Y)` products. -/
theorem card_products_le_packed (R : Finset (Fin D.N)) :
    ((univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
        c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ R).card : ℝ) ≤
      (c.weightsUsed R).card * (univ.filter (c.Full R)).card +
        √((c.weightsUsed R).card * ((c.inputsUsed R).filter fun x => ¬ c.Full R x.2.1).card *
          (outputsOtherPacked c R).card) := by
  rw [← card_filter_add_card_filter_not fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
    c.Full R g.2.1, filter_filter, filter_filter, Nat.cast_add]
  refine add_le_add (by exact_mod_cast (full_positions_packed c R 0 fun _ => Nat.zero_le _).1) ?_
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
      simp only [PackedChain.weightsUsed, mem_filter, mem_univ, true_and]
      exact ⟨x.1, hx⟩
  have hq : ∀ ℓ ∈ (univ : Finset (Fin c.L)), #((T ℓ).image fun x => (x.2.2, x.2.1)) ≤
      #(((c.inputsUsed R).filter fun x => ¬ c.Full R x.2.1).filter fun x => x.1 = ℓ) := fun ℓ _ => by
    refine (card_le_card fun y hy => ?_).trans (card_image_le (f := fun x => (x.2.1.val, x.2.2.val)))
    simp only [T, image_image, mem_image, mem_filter, mem_univ, true_and, Function.comp_def] at hy
    obtain ⟨x, ⟨hx, hnf⟩, rfl⟩ := hy
    refine mem_image.2 ⟨⟨ℓ, x.1, x.2.2⟩, ?_, rfl⟩
    simp only [PackedChain.inputsUsed, mem_filter, mem_univ, true_and, and_true]
    exact ⟨⟨x.2.1, hx⟩, hnf⟩
  have hy : ∀ ℓ ∈ (univ : Finset (Fin c.L)), #((T ℓ).image fun x => (x.2.2, x.1)) ≤
      #((outputsOtherPacked c R).filter fun x => x.1 = ℓ) := fun ℓ _ => by
    refine (card_le_card fun y hy => ?_).trans (card_image_le (f := fun x => (x.2.1.val, x.2.2.val)))
    simp only [T, image_image, mem_image, mem_filter, mem_univ, true_and, Function.comp_def] at hy
    obtain ⟨x, ⟨hx, hnf⟩, rfl⟩ := hy
    refine mem_image.2 ⟨⟨ℓ, x.1, x.2.1⟩, ?_, rfl⟩
    simp only [outputsOtherPacked, mem_filter, mem_univ, true_and, and_true]
    exact ⟨hnf, x.2.2, hx⟩
  have h := sum_card_products_le_sqrt univ T hp hq hy
  rw [← card_eq_sum_card_fiberwise fun _ _ => mem_univ _, ← card_eq_sum_card_fiberwise fun _ _ => mem_univ _,
    ← Nat.cast_sum, ← hT] at h
  exact h

open Classical in
/-- **The products of an IU, by columns**: at positions that are not full for `R`, each input entry `R` uses is in at
most `E` products, one for each row of its multiplication, whichever gates hold them. -/
theorem card_products_le_columns_packed (R : Finset (Fin D.N)) {E : ℕ} (hE : ∀ ℓ, c.e ℓ ≤ E) :
    (univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
        c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ R).card ≤
      (c.weightsUsed R).card * (univ.filter (c.Full R)).card +
        E * ((c.inputsUsed R).filter fun x => ¬ c.Full R x.2.1).card := by
  rw [← card_filter_add_card_filter_not fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
    c.Full R g.2.1, filter_filter, filter_filter]
  refine Nat.add_le_add (full_positions_packed c R 0 fun _ => Nat.zero_le _).1 ?_
  rw [mul_comm, ← card_range E, ← card_product]
  refine card_le_card_of_injOn
    (fun g => ((⟨g.1, g.2.1, g.2.2.2⟩ : (ℓ : Fin c.L) × Fin c.N × Fin (c.d ℓ)), g.2.2.1.val))
    (fun g hg => ?_) (fun g _ g' _ h => ?_)
  · simp only [mem_coe, mem_filter, mem_univ, true_and] at hg
    simp only [mem_coe, mem_product, PackedChain.inputsUsed, mem_filter, mem_univ, true_and, mem_range]
    exact ⟨⟨⟨g.2.2.1, hg.1⟩, hg.2⟩, g.2.2.1.isLt.trans_le (hE _)⟩
  · obtain ⟨ℓ, t, i, j⟩ := g
    obtain ⟨ℓ', t', i', j'⟩ := g'
    simp only [Prod.mk.injEq, Sigma.mk.inj_iff] at h
    obtain ⟨⟨rfl, h⟩, hi⟩ := h
    simp only [heq_eq_eq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain rfl := Fin.ext hi
    rfl

end Lemmas

open Classical in
/-- **What an IU reads**, on a packed chain: `R` reads at least `b_w p_R + b_a (d τ_R + Q_R)` bits under
`Assumptions.PackedActivationsIncompressible`. -/
theorem iu_reads_packed {k : ℕ} (D : Declaration) (c : PackedChain D.D k) (Γ : ℕ) (K b_w b_a : ℝ) (d : ℕ)
    (hd : ∀ ℓ, d ≤ c.d ℓ) (hb_a : 0 ≤ b_a) (h : Assumptions.PackedActivationsIncompressible D c Γ K b_w b_a)
    (R : Finset (Fin D.D.N)) (hR : R ⊆ D.D.computed) (hΓ : D.D.size R ≤ Γ) (hK : (novelReads D R : ℝ) ≤ K) :
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
    · exact ⟨hx, not_linkedWhole_of_not_full_packed c ht _⟩
  have := (full_positions_packed c R d hd).2
  have hcard := Finset.card_le_card hsub
  rw [Finset.card_union_of_disjoint hdisj] at hcard
  exact_mod_cast (Nat.add_le_add_right this _).trans hcard

open Classical in
/-- **The partial-sum law over the pieces, with a budget for the dark gates**, on a packed chain:
`max (min c₁ c₂) (μ b_a / Λ) · G_S ≤ α size(d) + (1 − μ) B`, with `G_S` counting products, whatever `k`. -/
theorem matMulOn_of_counts_packed {k Γ : ℕ} {α K b_w b_a μ B : ℝ} {c d : Declaration} {ch : PackedChain c.D k}
    {dd emax : ℕ} {M : PartialContainment c d} {n : ℕ} {P : Partition d.D n} (hα : 0 ≤ α) (hK : 0 < K)
    (hw : 0 < b_w) (ha : 0 < b_a) (hdd : 0 < dd) (hαK : α * Γ ≤ K) (hμ₀ : 0 ≤ μ) (hμ₁ : μ ≤ 1)
    (hd : ∀ ℓ, dd ≤ ch.d ℓ) (he : ∀ ℓ, ch.e ℓ ≤ emax)
    (hP : ∀ u, d.D.size (P.gates u) ≤ Γ ∧ (novelReads d (P.gates u) : ℝ) ≤ α * d.D.size (P.gates u))
    (hB : Assumptions.BaselineRegistered c d M) (hA : Assumptions.PackedActivationsIncompressible c ch Γ K b_w b_a)
    (h₂ : ∑ u, (b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
        b_a / 2 * (outputsOtherPacked ch (piece M P u)).card) ≤ ∑ u, (novelReads d (P.gates u) : ℝ) + B) :
    max (min ((1 + √μ) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √μ) * √(1 - μ) * (b_a * √(b_w / K))))
        (μ * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) *
      (univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
        ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card ≤
    α * d.D.size d.D.computed + (1 - μ) * B := by
  have hdpos : (0 : ℝ) < dd := by exact_mod_cast hdd
  have hcapP := piece_caps (M := M) hα hαK hP hB
  have hr : ∀ u, (novelReads c (piece M P u) : ℝ) ≤ novelReads d (P.gates u) := fun u => by
    exact_mod_cast novelReads_piece hB u
  have hiu := fun u => iu_reads_packed c ch Γ K b_w b_a dd hd ha.le hA (piece M P u) (piece_subset_computed u)
    (hcapP u).1 (hcapP u).2
  have h₁ : ∑ u, (b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
      b_a * (dd * ((univ.filter (ch.Full (piece M P u))).card : ℝ) +
        ((ch.inputsUsed (piece M P u)).filter fun x => ¬ ch.Full (piece M P u) x.2.1).card)) ≤
      ∑ u, (novelReads d (P.gates u) : ℝ) :=
    sum_le_sum fun u _ => (hiu u).trans (hr u)
  have hW := mu_weighting_budget univ hμ₀ hμ₁ h₁ h₂
  have hM := max_mul_sum_le univ hw ha hdpos hK hμ₀ hμ₁ (le_max_left (emax : ℝ) (K / (4 * b_w * dd)))
    (le_max_right (emax : ℝ) (K / (4 * b_w * dd))) (fun _ _ => by positivity) (fun _ _ => by positivity)
    (fun _ _ => by positivity) (fun _ _ => by positivity) (fun _ _ => by positivity)
    (fun u _ => (hiu u).trans (hcapP u).2) (fun u _ => card_products_le_packed ch (piece M P u))
    (fun u _ => by exact_mod_cast card_products_le_columns_packed ch (piece M P u) he) hW
  have hrsum : ∑ u, (novelReads d (P.gates u) : ℝ) ≤ α * d.D.size d.D.computed :=
    calc ∑ u, (novelReads d (P.gates u) : ℝ) ≤ ∑ u, α * (d.D.size (P.gates u) : ℝ) := sum_le_sum fun u _ => (hP u).2
      _ = α * d.D.size d.D.computed := by
          rw [← mul_sum]
          congr 1
          exact_mod_cast P.sum_card_gates
  have hGS : ((univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
      ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card : ℝ) =
      ∑ u, ((univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
        ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ piece M P u).card : ℝ) := by
    exact_mod_cast (sum_card_filter_piece (M := M) (P := P) univ
      fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) => ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2).symm
  rw [hGS]
  refine hM.trans ?_
  linarith

open Classical in
/-- **Products done in the declaration are bounded by their fan-out, at most `k` products to a gate**: at `μ = 1` the
dark gates' budget has weight `1 − μ = 0`, so any budget does. -/
theorem PackedContainedFanOutOn : Guarantees.PackedContainedFanOutOn := by
  intro k Γ α K b_w b_a c d ch dd emax M hα hK hw ha hdd hαK hd he ⟨n, P, hP⟩ hB hA
  have h₂ : ∑ u, (b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
        b_a / 2 * (outputsOtherPacked ch (piece M P u)).card) ≤
      ∑ u, (novelReads d (P.gates u) : ℝ) +
        ∑ u, (b_w * ((ch.weightsUsed (piece M P u)).card : ℝ) +
          b_a / 2 * (outputsOtherPacked ch (piece M P u)).card) :=
    le_add_of_nonneg_left (sum_nonneg fun u _ => Nat.cast_nonneg _)
  have h := matMulOn_of_counts_packed (μ := 1) hα hK hw ha hdd hαK zero_le_one le_rfl hd he hP hB hA h₂
  have hm : b_a / max (emax : ℝ) (K / (4 * b_w * dd)) ≤
      max (min ((1 + √1) ^ 2 * (b_w * b_a * dd / K)) (√2 * (1 + √1) * √(1 - 1) * (b_a * √(b_w / K))))
        (1 * b_a / max (emax : ℝ) (K / (4 * b_w * dd))) :=
    le_max_of_le_right (le_of_eq (by ring))
  calc b_a / max (emax : ℝ) (K / (4 * b_w * dd)) *
        ((univ.filter fun p : (ℓ : Fin ch.L) × Fin ch.N × Fin (ch.e ℓ) × Fin (ch.d ℓ) =>
          ch.prod p.1 p.2.1 p.2.2.1 p.2.2.2 ∈ M.S).card : ℝ)
      ≤ _ := mul_le_mul_of_nonneg_right hm (Nat.cast_nonneg _)
    _ ≤ _ := h
    _ = α * d.D.size d.D.computed := by ring

/-- Under `Chain.toPacked` an input entry is linked to an entry whole in `R` exactly when it is in the chain. -/
theorem toPacked_linkedWhole {D : Circuit} (c : Chain D) (R : Finset (Fin D.N)) (ℓ : Fin c.L) (t : Fin c.N)
    (j : Fin (c.d ℓ)) : c.toPacked.LinkedWhole R ℓ t j ↔ c.LinkedWhole R ℓ t j := by
  obtain ⟨_ | m, hm⟩ := ℓ <;> first | exact Iff.rfl | simp [PackedChain.LinkedWhole, Chain.LinkedWhole]

/-- At `k = 1` the packed assumption is the per-product one. -/
theorem packedActivationsIncompressible_toPacked {D : Declaration} {c : Chain D.D} {Γ : ℕ} {K b_w b_a : ℝ} :
    Assumptions.PackedActivationsIncompressible D c.toPacked Γ K b_w b_a ↔
      Assumptions.ActivationsIncompressible D c Γ K b_w b_a := by
  first
  | exact Iff.rfl
  | classical
    have hcard : ∀ R : Finset (Fin D.D.N),
        ((c.toPacked.inputsUsed R).filter fun x => ¬ c.toPacked.LinkedWhole R x.1 x.2.1 x.2.2).card =
          ((c.inputsUsed R).filter fun x => ¬ c.LinkedWhole R x.1 x.2.1 x.2.2).card := fun R => by
      show ((c.inputsUsed R).filter fun x => ¬ c.toPacked.LinkedWhole R x.1 x.2.1 x.2.2).card = _
      rw [Finset.filter_congr fun x _ => not_congr (toPacked_linkedWhole c R x.1 x.2.1 x.2.2)]
    unfold Assumptions.PackedActivationsIncompressible Assumptions.ActivationsIncompressible
    simp only [hcard]
    exact Iff.rfl

/-- **The per-product fan-out bound is the case `k = 1`**: a chain is a packed chain with one product to a gate, and
the assumption, the products in `S` and the bound are the same under `Chain.toPacked`. -/
theorem containedFanOutOn_of_packed (h : Guarantees.PackedContainedFanOutOn) : Guarantees.ContainedFanOutOn :=
  fun Γ α K b_w b_a c d ch dd emax M hα hK hw ha hdd hαK hd he hP hB hA =>
    h 1 Γ α K b_w b_a c d ch.toPacked dd emax M hα hK hw ha hdd hαK hd he hP hB
      (packedActivationsIncompressible_toPacked.2 hA)

end Nci.SecurityProofs
