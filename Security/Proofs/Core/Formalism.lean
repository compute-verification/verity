module

public import Definitions.Core.Guarantees.Formalism
public import Proofs.Core.Circuit
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Ring

/-!
# Proofs of the working draft's Lemma 1 and Theorem 3, and of Theorem 3 at the serving blocks

* `Committed.recompute_eval`: in a run of the registered circuit, a inference block recomputed from the committed
  declaration and the run is the run: a inference block's values are determined by its unregistered inputs and `θ`;
* `length_blockProgram`: on a Boolean circuit, inference block `u`'s program has `|e| + ⌈log₂ B⌉ + |in(I) \ θ|` bits;
* `Partition.sum_blocks`: a sum over operation gates is the sum, over the inference blocks, of the sums over their gates.
-/

@[expose] public section

namespace Verity

open Finset

theorem Circuit.length_bits {C : Circuit} (hC : C.Boolean) (X : C.Trace) (l : List (Fin C.N)) :
    (C.bits X l).length = l.length := by
  induction l with
  | nil => rfl
  | cons g l ih =>
    simp only [bits, List.flatMap_cons, List.length_append, List.length_ofFn, List.length_cons] at ih ⊢
    rw [ih, hC.one_bit g]
    omega

theorem Committed.recompute_eval (R : Committed) {a : R.d.D.Trace} (ha : R.Honest a) (u : Fin R.d.n) :
    R.recompute (R.d.D.eval a) u = R.d.D.eval a := by
  funext g
  induction g using WellFoundedLT.induction with
  | _ g ih => ?_
  by_cases hθ : g ∈ R.d.θ
  · rw [recompute, R.d.D.run_fixed (b := R.val g) (by simp [hθ]), R.d.D.eval_input a (R.d.θ_input g hθ), ha g hθ]
  · by_cases hu : R.d.𝓘.unit g = some u
    · have hi : R.d.D.isInput g = false := by
        cases h : R.d.D.isInput g
        · rfl
        · rw [← R.d.𝓘.input_iff] at h
          simp [h] at hu
      rw [recompute, R.d.D.run_free (by simp [hθ, hu]), R.d.D.eval_op a hi]
      exact R.d.D.op_congr g _ _ fun x hx => ih x (R.d.D.reads_lt g x hx)
    · rw [recompute, R.d.D.run_fixed (b := R.d.D.eval a g) (by simp [hθ, hu])]

theorem length_blockProgram (R : Committed) (hC : R.d.D.Boolean) (e : List Bool) (X : R.d.D.Trace)
    (u : Fin R.d.n) :
    (blockProgram R e X u).length = e.length + Nat.clog 2 R.d.n + (R.d.unregistered (R.d.𝓘.gates u)).card := by
  simp [blockProgram, indexCode, Circuit.length_bits hC, Finset.length_sort]
  omega

theorem Partition.sum_blocks {C : Circuit} {B : ℕ} (P : Partition C B) {V : Finset (Fin C.N)}
    (hV : V ⊆ C.computed) (ω : Fin C.N → ℝ) :
    ∑ v ∈ V, ω v = ∑ u, ∑ v ∈ V ∩ P.gates u, ω v := by
  classical
  rw [← Finset.sum_biUnion]
  · congr 1
    ext g
    simp only [mem_biUnion, mem_univ, true_and, mem_inter, P.mem_gates]
    constructor
    · intro hg
      have hi : C.isInput g = false := by simpa [Circuit.computed] using hV hg
      obtain ⟨u, hu⟩ : ∃ u, P.unit g = some u :=
        Option.ne_none_iff_exists'.1 fun h => by simp [(P.input_iff g).1 h] at hi
      exact ⟨u, hg, hu⟩
    · rintro ⟨_, hg, _⟩
      exact hg
  · intro u _ v _ huv
    refine Finset.disjoint_left.2 fun g hu hv => huv ?_
    rw [mem_inter, P.mem_gates] at hu hv
    exact Option.some_injective _ (hu.2.symm.trans hv.2)

end Verity

namespace Verity.SecurityProofs

open Finset

theorem CKSize : Guarantees.CKSize := by
  intro d B c k P hP
  rw [Circuit.size, ← P.sum_card_gates]
  calc ∑ u, (P.gates u).card ≤ ∑ _u : Fin B, c := Finset.sum_le_sum fun u _ => hP.card u
    _ = c * B := by simp [mul_comm]

theorem BlockDescription : Guarantees.BlockDescription := by
  intro R c k hP ρ M r t a_blk e y he hev a ha u
  have hrun := hev u (R.d.D.eval a)
  rw [R.recompute_eval ha u] at hrun
  refine (iInf₂_le (blockProgram R e (R.d.D.eval a) u) hrun).trans ?_
  rw [length_blockProgram R hP.boolean]
  have := hP.reads u
  exact_mod_cast (by omega : e.length + Nat.clog 2 R.d.n + (R.d.unregistered (R.d.𝓘.gates u)).card ≤
    k + Nat.clog 2 R.d.n + a_blk)

theorem CountingSum : Guarantees.CountingSum := by
  intro C B P V ω q hV h
  rw [P.sum_blocks hV ω]
  exact Finset.sum_le_sum fun u _ => h u

theorem Counting : Guarantees.Counting := by
  intro d B c P V ω q hV hq h
  have hsum : ∑ v ∈ V, ω v ≤ B * q := by
    simpa using CountingSum d.D B P V ω (fun _ => q) hV h
  have h1 : (c : ℝ) / q * ∑ v ∈ V, ω v ≤ (c * B : ℕ) := by
    push_cast
    calc (c : ℝ) / q * ∑ v ∈ V, ω v ≤ c / q * (B * q) :=
          mul_le_mul_of_nonneg_left hsum (div_nonneg (Nat.cast_nonneg _) hq.le)
      _ = c * B := by field_simp
  refine ⟨h1, fun T₀ hT => ?_⟩
  rw [slowdown, show (c : ℝ) / (T₀ * q) * ∑ v ∈ V, ω v = (c / q * ∑ v ∈ V, ω v) / T₀ by ring]
  exact div_le_div_of_nonneg_right h1 hT.le

theorem ServingBlockSum : Guarantees.ServingBlockSum := by
  intro C B P S V ω qS qI qDark hV hS hI hD
  rw [P.sum_blocks hV ω, ← sum_sdiff (subset_univ S)]
  have : ∑ u ∈ S, ∑ v ∈ V ∩ P.gates u, ω v ≤ ∑ u ∈ S, min (qS u) (qI u) :=
    sum_le_sum fun u hu => le_min (hS u hu) (hI u hu)
  linarith

theorem InferenceCap : Guarantees.InferenceCap := by
  classical
  intro C nf nc Pf Pc r V U ω q hV hq s
  have hsplit : ∑ v ∈ V ∩ Pc.gates s, ω v =
      ∑ v ∈ (V ∩ Pc.gates s) \ U, ω v + ∑ v ∈ (V ∩ U) ∩ Pc.gates s, ω v := by
    have hsub : (V ∩ U) ∩ Pc.gates s ⊆ V ∩ Pc.gates s := fun v hv => by
      simp only [mem_inter] at hv ⊢
      exact ⟨hv.1.1, hv.2⟩
    rw [← sum_sdiff hsub, show (V ∩ Pc.gates s) \ ((V ∩ U) ∩ Pc.gates s) = (V ∩ Pc.gates s) \ U by
      ext v
      simp only [mem_sdiff, mem_inter]
      tauto]
  have hfine : (V ∩ Pc.gates s) \ U =
      (univ.filter fun i => r.parent i = s).biUnion fun i => (V \ U) ∩ Pf.gates i := by
    ext v
    simp only [mem_sdiff, mem_inter, mem_biUnion, mem_filter, mem_univ, true_and, Partition.mem_gates]
    constructor
    · rintro ⟨⟨hv, hs⟩, hu⟩
      have hi : C.isInput v = false := by simpa [Circuit.computed] using hV hv
      obtain ⟨i, hi'⟩ : ∃ i, Pf.unit v = some i :=
        Option.ne_none_iff_exists'.1 fun h => by simp [(Pf.input_iff v).1 h] at hi
      refine ⟨i, ?_, ⟨hv, hu⟩, hi'⟩
      have := r.sub v
      rw [hs, hi', Option.map_some] at this
      exact (Option.some_injective _ this).symm
    · rintro ⟨i, hi, ⟨hv, hu⟩, hi'⟩
      refine ⟨⟨hv, ?_⟩, hu⟩
      rw [r.sub v, hi', Option.map_some, hi]
  have hdisj : Set.PairwiseDisjoint ((univ.filter fun i => r.parent i = s) : Set (Fin nf))
      (fun i => (V \ U) ∩ Pf.gates i) := by
    intro i _ j _ hij
    refine disjoint_left.2 fun v hvi hvj => hij ?_
    rw [mem_inter, Partition.mem_gates] at hvi hvj
    exact Option.some_injective _ (hvi.2.symm.trans hvj.2)
  rw [hsplit, hfine, sum_biUnion hdisj]
  have : ∑ i ∈ univ.filter (fun i => r.parent i = s), ∑ v ∈ (V \ U) ∩ Pf.gates i, ω v ≤
      ((univ.filter fun i => r.parent i = s).card : ℝ) * q := by
    rw [← nsmul_eq_mul]
    exact sum_le_card_nsmul _ _ _ fun i _ => hq i
  linarith

end Verity.SecurityProofs
