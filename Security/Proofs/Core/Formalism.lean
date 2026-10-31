import Definitions.Core.Guarantees.Formalism
import Proofs.Core.Circuit
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# Proofs of the working draft's Lemma 1 and Theorem 3

* `Registration.recompute_eval`: in a run of the registered circuit, a proof block recomputed from the registration and
  the run is the run: a proof block's values are determined by its unregistered inputs and `θ`;
* `length_blockProgram`: on a Boolean circuit, proof block `u`'s program has `|e| + ⌈log₂ B⌉ + |in(I) \ θ|` bits;
* `Partition.sum_blocks`: a sum over operation gates is the sum, over the proof blocks, of the sums over their gates.
-/

namespace Verity

open Finset

theorem Circuit.length_bits {C : Circuit} (hC : C.Boolean) (X : C.Transcript) (l : List (Fin C.N)) :
    (C.bits X l).length = l.length := by
  induction l with
  | nil => rfl
  | cons g l ih =>
    simp only [bits, List.flatMap_cons, List.length_append, List.length_ofFn, List.length_cons] at ih ⊢
    rw [ih, hC.one_bit g]
    omega

theorem Registration.recompute_eval (R : Registration) {a : R.d.D.Transcript} (ha : R.Honest a) (u : Fin R.B) :
    R.recompute (R.d.D.eval a) u = R.d.D.eval a := by
  funext g
  induction g using WellFoundedLT.induction with
  | _ g ih => ?_
  by_cases hθ : g ∈ R.d.θ
  · rw [recompute, R.d.D.run_fixed (b := R.val g) (by simp [hθ]), R.d.D.eval_input a (R.d.θ_input g hθ), ha g hθ]
  · by_cases hu : R.P.unit g = some u
    · have hi : R.d.D.isInput g = false := by
        cases h : R.d.D.isInput g
        · rfl
        · rw [← R.P.input_iff] at h
          simp [h] at hu
      rw [recompute, R.d.D.run_free (by simp [hθ, hu]), R.d.D.eval_op a hi]
      exact R.d.D.op_congr g _ _ fun x hx => ih x (R.d.D.reads_lt g x hx)
    · rw [recompute, R.d.D.run_fixed (b := R.d.D.eval a g) (by simp [hθ, hu])]

theorem length_blockProgram (R : Registration) (hC : R.d.D.Boolean) (e : List Bool) (X : R.d.D.Transcript)
    (u : Fin R.B) :
    (blockProgram R e X u).length = e.length + Nat.clog 2 R.B + (R.d.unregistered (R.P.gates u)).card := by
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
  simp [hP.card, mul_comm]

theorem BlockDescription : Guarantees.BlockDescription := by
  intro R c k hP ρ M r t a_blk e y he hev a ha u
  have hrun := hev u (R.d.D.eval a)
  rw [R.recompute_eval ha u] at hrun
  refine (iInf₂_le (blockProgram R e (R.d.D.eval a) u) hrun).trans ?_
  rw [length_blockProgram R hP.boolean]
  have := hP.reads u
  exact_mod_cast (by omega : e.length + Nat.clog 2 R.B + (R.d.unregistered (R.P.gates u)).card ≤
    k + Nat.clog 2 R.B + a_blk)

theorem CountingSum : Guarantees.CountingSum := by
  intro C B P V ω q hV h
  rw [P.sum_blocks hV ω]
  exact Finset.sum_le_sum fun u _ => h u

theorem Counting : Guarantees.Counting := by
  intro d B c k P hP V ω q hV hq h
  have hsum : ∑ v ∈ V, ω v ≤ B * q := by
    simpa using CountingSum d.D B P V ω (fun _ => q) hV h
  have hsize : ((d.D.size d.D.computed : ℕ) : ℝ) = c * B := by
    rw [CKSize d B c k P hP]
    push_cast
    rfl
  have h1 : (c : ℝ) / q * ∑ v ∈ V, ω v ≤ d.D.size d.D.computed := by
    rw [hsize]
    calc (c : ℝ) / q * ∑ v ∈ V, ω v ≤ c / q * (B * q) :=
          mul_le_mul_of_nonneg_left hsum (div_nonneg (Nat.cast_nonneg _) hq.le)
      _ = c * B := by field_simp
  refine ⟨h1, fun T₀ hT => ?_⟩
  rw [slowdown, show (c : ℝ) / (T₀ * q) * ∑ v ∈ V, ω v = (c / q * ∑ v ∈ V, ω v) / T₀ by ring]
  exact div_le_div_of_nonneg_right h1 hT.le

end Verity.SecurityProofs
