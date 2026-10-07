import Definitions.Pouw.Dimension.Causal
import Proofs.Pouw.Dimension.Proofs

/-!
# Theorem D with causal inputs: proofs

* `sd_wordC_lt`: the four-point difference at unit `u₀` kills the words of every earlier unit.
* `sd_wordC_self`: at unit `u₀` itself it reads the checkpoint coefficient, as for `A = g(F₁)`.
* `indep_modC`: the words are independent modulo `V₀`, by induction from the last unit.
* `algBoundCausal`, `causalWitness`, `algWitnessCausal`.
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.NCP.RelationProofs

theorem pt_fst_ne {N m n k : ℕ} (bE bF : Bool) {u0 : ℕ} (i a b j : ℕ) (u' : Fin N) (h : (u' : ℕ) ≠ u0) :
    (pt N m n k bE bF u0 i a b j).1 u' = (pt N m n k false bF u0 i a b j).1 u' := by
  funext x y
  simp [pt, h]

theorem causal_pt {N m n k : ℕ} {A : Noise N m n k → ℕ → Mat} (hA : Causal A) (bF : Bool) {u0 u : ℕ}
    (i a b j : ℕ) (h : u ≤ u0) :
    A (pt N m n k true bF u0 i a b j) u = A (pt N m n k false bF u0 i a b j) u :=
  hA _ _ u rfl fun u' hu' => pt_fst_ne true bF i a b j u' (by omega)

theorem sd_wordC_lt {k m n N : ℕ} (R : Mat) {A : Noise N m n k → ℕ → Mat} (hA : Causal A) (B : ℕ → Mat)
    {u0 u : ℕ} (i a b j : ℕ) (h : u < u0) (p : ℕ) :
    sd N m n k u0 i a b j (wordC k m n N R A B u p) = 0 := by
  have hE : ∀ bF, checked k m n N R (A (pt N m n k true bF u0 i a b j)) B (pt N m n k true bF u0 i a b j) u p =
      checked k m n N R (A (pt N m n k false bF u0 i a b j)) B (pt N m n k false bF u0 i a b j) u p := by
    intro bF
    unfold checked
    rw [e1Of_pt_ne true bF i a b j h.ne, causal_pt hA bF i a b j h.le]
    rfl
  simp only [sd_apply, wordC, hE]
  ring

theorem sd_wordC_self {k m n N : ℕ} (R : Mat) {A : Noise N m n k → ℕ → Mat} (hA : Causal A) (B : ℕ → Mat)
    {u0 e a b : ℕ} (hu : u0 < N) (he : e < m * n) (ha : a < k) (hb : b < k) (p : ℕ) :
    sd N m n k u0 (e / n) a b (e % n) (wordC k m n N R A B u0 p) = (coefAt k m n R p e a b : ℚ) := by
  have hn : 0 < n := Nat.pos_of_ne_zero fun h => by simp [h] at he
  have hi : e / n < m := (Nat.div_lt_iff_lt_mul hn).2 he
  have hj : e % n < n := Nat.mod_lt e hn
  have h := fun bE bF => congrFun (checked_pt (N := N) R (A (pt N m n k bE bF u0 (e / n) a b (e % n))) B bE bF
    hu hi ha hb hj) p
  simp only [sd_apply, wordC, h, ite_true, Bool.false_eq_true, ite_false]
  rw [causal_pt hA true (e / n) a b (e % n) le_rfl, causal_pt hA false (e / n) a b (e % n) le_rfl,
    ← coef_eq R e a b p, ← mixdiff2 k (A (pt N m n k false true u0 (e / n) a b (e % n)) u0)
    (A (pt N m n k false false u0 (e / n) a b (e % n)) u0) (B u0)]
  push_cast
  ring

theorem secondDifferenceC {k m n N : ℕ} (R : Mat) {A : Noise N m n k → ℕ → Mat} (hA : Causal A) (B : ℕ → Mat)
    {U : Finset ℕ} (c : ℕ → ℕ → ℚ) (hU : U ⊆ range N)
    (hsum : (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • wordC k m n N R A B u p) ∈ V0 N m n k)
    {u0 : ℕ} (hu0 : u0 ∈ U) (hlater : ∀ u ∈ U, u0 < u → ∀ p < nonFinal k m n, c u p = 0)
    {e : ℕ} (he : e < m * n) {a : ℕ} (ha : a < k) {b : ℕ} (hb : b < k) :
    ∑ t ∈ range (3 * k / 16 - 1), c u0 (t * (m * n) + e) * (ckMat k R (16 * (t + 1)) a b : ℚ) = 0 := by
  have hN : u0 < N := mem_range.1 (hU hu0)
  have h0 := sd_V0 u0 (e / n) a b (e % n) hsum
  rw [map_sum, sum_eq_single u0 (fun u hu hne => by
      rw [map_sum]
      rcases lt_or_gt_of_ne hne with hlt | hgt
      · exact sum_eq_zero fun p _ => by rw [map_smul, sd_wordC_lt R hA B _ _ _ _ hlt, smul_zero]
      · exact sum_eq_zero fun p hp => by rw [hlater u hu hgt p (mem_range.1 hp), zero_smul, map_zero])
    (fun h => absurd hu0 h), map_sum] at h0
  simp only [map_smul, smul_eq_mul, sd_wordC_self R hA B hN he ha hb, coefAt] at h0
  push_cast at h0
  have hM : e < m * n := he
  rw [← h0, nonFinal]
  simp only [mul_ite, mul_zero]
  rw [sum_blocks (m * n) e hM (fun p => c u0 p * (ckMat k R (16 * (p / (m * n) + 1)) a b : ℚ))]
  refine sum_congr rfl fun t _ => ?_
  rw [div_blocks hM]

/-- Under independence, the non-final checked words for a causal `A` are linearly independent modulo `V₀`: induction
from the last unit. -/
theorem indep_modC {k m n N : ℕ} {R : Mat} {A : Noise N m n k → ℕ → Mat} (hA : Causal A) (B : ℕ → Mat)
    {U : Finset ℕ} (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N) (c : ℕ → ℕ → ℚ)
    (hc : (∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c u p • wordC k m n N R A B u p) ∈ V0 N m n k) :
    ∀ u ∈ U, ∀ p < nonFinal k m n, c u p = 0 := by
  have key : ∀ d, ∀ u ∈ U, N ≤ u + d → ∀ p < nonFinal k m n, c u p = 0 := by
    intro d
    induction d with
    | zero => intro u hu h; have := mem_range.1 (hU hu); omega
    | succ d ih =>
      intro u hu hud p hp
      have hlater : ∀ u' ∈ U, u < u' → ∀ p < nonFinal k m n, c u' p = 0 :=
        fun u' hu' hlt => ih u' hu' (by omega)
      have hmn : 0 < m * n := Nat.pos_of_ne_zero fun h => by simp [nonFinal, h] at hp
      have hp' : p / (m * n) < 3 * k / 16 - 1 := (Nat.div_lt_iff_lt_mul hmn).2 hp
      have hT : 0 < 3 * k / 16 - 1 := lt_of_le_of_lt (Nat.zero_le _) hp'
      have hk : 16 * (3 * k / 16 - 1) < 3 * k := by omega
      have := indepQ hind (3 * k / 16 - 1) hk (fun t => c u (t * (m * n) + p % (m * n)))
        (fun a ha b hb => secondDifferenceC R hA B c hU hc hu hlater (Nat.mod_lt _ hmn) ha hb)
        (p / (m * n)) hp'
      rwa [Nat.div_add_mod'] at this
  intro u hu
  exact key N u hu (by omega)

theorem indep_subC {k m n N : ℕ} {R : Mat} {A : Noise N m n k → ℕ → Mat} (hA : Causal A) (B : ℕ → Mat)
    {U : Finset ℕ} (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N)
    (c : ↥(U ×ˢ range (nonFinal k m n)) → ℚ)
    (hc : ∑ x, c x • wordC k m n N R A B x.1.1 x.1.2 ∈ V0 N m n k) : ∀ x, c x = 0 := by
  classical
  let c' : ℕ → ℕ → ℚ := fun u p => if h : (u, p) ∈ U ×ˢ range (nonFinal k m n) then c ⟨(u, p), h⟩ else 0
  have hsum : ∑ u ∈ U, ∑ p ∈ range (nonFinal k m n), c' u p • wordC k m n N R A B u p =
      ∑ x : ↥(U ×ˢ range (nonFinal k m n)), c x • wordC k m n N R A B x.1.1 x.1.2 := by
    rw [← sum_product (s := U) (t := range (nonFinal k m n))
        (f := fun x => c' x.1 x.2 • wordC k m n N R A B x.1 x.2),
      ← sum_coe_sort (U ×ˢ range (nonFinal k m n))]
    refine sum_congr rfl fun x _ => ?_
    have hx : ((x : ℕ × ℕ).1, (x : ℕ × ℕ).2) ∈ U ×ˢ range (nonFinal k m n) := x.2
    simp only [c', dif_pos hx]
  rintro ⟨⟨u, p⟩, hx⟩
  have := indep_modC hA B hind hU c' (hsum ▸ hc) u (mem_product.1 hx).1 p (mem_range.1 (mem_product.1 hx).2)
  rw [show c' u p = c ⟨(u, p), hx⟩ from dif_pos hx] at this
  exact this

/-- **Theorem D with causal inputs.** -/
theorem algBoundCausal : AlgBoundCausal := by
  intro k m n N R A B U L P hind hA hU hL hH hout
  have hcount := count_le P L
    (fun x : ↥(U ×ˢ range (nonFinal k m n)) => wordC k m n N R A B x.1.1 x.1.2)
    (fun c hc => indep_subC hA B hind hU c (span1_le_V0 hL hc))
    (fun x => hout x.1.1 (mem_product.1 x.2).1 x.1.2 (mem_range.1 (mem_product.1 x.2).2))
  rw [Fintype.card_coe, card_product, card_range, nonFinal] at hcount
  have hc : U.card * (m * n) * (3 * k / 16 - 1) ≤ nPriced P := by
    calc U.card * (m * n) * (3 * k / 16 - 1) = U.card * ((3 * k / 16 - 1) * (m * n)) := by ring
      _ ≤ nPriced P := hcount
  have := h16_cost hH
  omega

/-- **`Causal`'s witness.** -/
theorem causalWitness : CausalWitness := by
  intro N m n k ω ω' u _ h
  by_cases hu : u = 0
  · simp [hu]
  · show (if u = 0 then (0 : Mat) else e1Of ω (u - 1)) = if u = 0 then 0 else e1Of ω' (u - 1)
    rw [if_neg hu, if_neg hu]
    funext i l
    unfold e1Of
    by_cases hh : u - 1 < N ∧ i < m ∧ l < k
    · rw [dif_pos hh, dif_pos hh, h ⟨u - 1, hh.1⟩ (show u - 1 < u by omega)]
    · rw [dif_neg hh, dif_neg hh]

/-- The honest program for causal inputs: one priced instruction at 16 per non-final checked word. -/
noncomputable def honestC (k m n N : ℕ) (R : Mat) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (U : Finset ℕ) :
    List (Instr (Noise N m n k)) :=
  U.toList.flatMap fun u => (List.range (nonFinal k m n)).map fun p => Instr.priced 16 (wordC k m n N R A B u p)

theorem mem_honestC {k m n N : ℕ} {R : Mat} {A : Noise N m n k → ℕ → Mat} {B : ℕ → Mat} {U : Finset ℕ}
    {ins : Instr (Noise N m n k)} :
    ins ∈ honestC k m n N R A B U ↔
      ∃ u ∈ U, ∃ p < nonFinal k m n, Instr.priced 16 (wordC k m n N R A B u p) = ins := by
  simp [honestC, List.mem_flatMap, List.mem_map]

/-- **The causal bound's hypotheses hold jointly, and it is attained.** -/
theorem algWitnessCausal : AlgWitnessCausal := by
  intro k hk h16 m n N A B U _
  obtain ⟨R, hadm, hind⟩ := RelationWitness.independentWitness k hk h16
  refine ⟨R, honestC k m n N R A B U, hadm, hind, ?_, fun u hu p hp =>
    mem_slots_of_priced (mem_honestC.2 ⟨u, hu, p, hp, rfl⟩) _, ?_⟩
  · intro ins hins _
    obtain ⟨u, _, p, _, rfl⟩ := mem_honestC.1 hins
    exact le_rfl
  · have h1 : ∀ u, cost ((List.range (nonFinal k m n)).map fun p => Instr.priced 16 (wordC k m n N R A B u p)) =
        16 * nonFinal k m n := by
      intro u
      simp only [cost, List.map_map]
      rw [show (Instr.price ∘ fun p => Instr.priced 16 (wordC k m n N R A B u p)) = fun _ => 16 from rfl,
        List.map_const', List.sum_replicate, List.length_range, smul_eq_mul, mul_comm]
    rw [honestC, cost_flatMap]
    simp only [h1]
    rw [List.map_const', List.sum_replicate, length_toList, smul_eq_mul, nonFinal]
    ring

end Pouw.Dimension
