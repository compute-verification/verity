module

public import Proofs.Nci.Chain
public import Proofs.Nci.Inequalities
public import Proofs.Nci.LoomisWhitney
public import Proofs.Nci.Lemmas

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
`F · 2G ≤ size(D)`: the counts here, assembled in reads form in `Proofs.Nci.Reads` (`UpdatedMatMulReads`).
-/

@[expose] public section

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
/-- **The products of an IU, by columns**: `R` computes at most `p τ` products at its full positions (Lemma 1) and, at
the others, at most `E` for each input entry it uses there, one for each row of its multiplication, when every output
dimension is at most `E`. -/
theorem card_products_le_columns (R : Finset (Fin D.N)) {E : ℕ} (hE : ∀ ℓ, c.e ℓ ≤ E) :
    (univ.filter fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
        c.prod g.1 g.2.1 g.2.2.1 g.2.2.2 ∈ R).card ≤
      (c.weightsUsed R).card * (univ.filter (c.Full R)).card +
        E * ((c.inputsUsed R).filter fun x => ¬ c.Full R x.2.1).card := by
  rw [← card_filter_add_card_filter_not fun g : (ℓ : Fin c.L) × Fin c.N × Fin (c.e ℓ) × Fin (c.d ℓ) =>
    c.Full R g.2.1, filter_filter, filter_filter]
  refine Nat.add_le_add (full_positions c R 0 fun _ => Nat.zero_le _).1 ?_
  rw [mul_comm, ← card_range E, ← card_product]
  refine card_le_card_of_injOn
    (fun g => ((⟨g.1, g.2.1, g.2.2.2⟩ : (ℓ : Fin c.L) × Fin c.N × Fin (c.d ℓ)), g.2.2.1.val))
    (fun g hg => ?_) (fun g _ g' _ h => ?_)
  · simp only [mem_coe, mem_filter, mem_univ, true_and] at hg
    simp only [mem_coe, mem_product, Chain.inputsUsed, mem_filter, mem_univ, true_and, mem_range]
    exact ⟨⟨⟨g.2.2.1, hg.1⟩, hg.2⟩, g.2.2.1.isLt.trans_le (hE _)⟩
  · obtain ⟨ℓ, t, i, j⟩ := g
    obtain ⟨ℓ', t', i', j'⟩ := g'
    simp only [Prod.mk.injEq, Sigma.mk.inj_iff] at h
    obtain ⟨⟨rfl, h⟩, hi⟩ := h
    simp only [heq_eq_eq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    obtain rfl := Fin.ext hi
    rfl

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

end Nci.SecurityProofs
