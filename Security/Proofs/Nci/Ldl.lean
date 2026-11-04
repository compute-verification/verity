import Definitions.Nci.Guarantees.Ldl
import Proofs.Core.Circuit
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith

/-!
# Local description length on a `(c,k)`-partition: the proofs

* `localEval_eval`: a block evaluated from the circuit's evaluation is that evaluation, so the block's extraction from
  its unregistered input bits is the run's (Lemma 1).
* `sum_blocks`, `size_eq`: a sum over operation gates is the sum of its blocks' sums, and exact blocks give `|C| = c B`
  (Theorem 3).
* Corollary 4b is Lemma 1, the fan-out count `|𝓙_I| ≤ e |V_I|` and Theorem 3 with unit weights.
-/

namespace Nci.Ldl

open Verity Finset
open scoped ENNReal

theorem LDL_le {D : Decoder} {t : ℕ} {y p : List Bool} (h : D p t = some y) : LDL D t y ≤ p.length :=
  iInf₂_le p h

/-- A block recomputed from the circuit's evaluation is the evaluation. -/
theorem localEval_eval {C : Circuit} {n : ℕ} (P : Partition C n) (inp : C.Transcript) (u : Fin n) :
    P.localEval (C.eval inp) u = C.eval inp := by
  funext g
  induction g using WellFoundedLT.induction with
  | _ g ih => ?_
  by_cases hu : P.unit g = some u
  · have hc : C.isInput g = false := by
      cases h : C.isInput g
      · rfl
      · rw [← P.input_iff] at h
        rw [h] at hu
        cases hu
    rw [P.localEval_of_mem _ hu, C.eval_op inp hc]
    exact C.op_congr g _ _ fun x hx => ih x (C.reads_lt g x hx)
  · exact P.localEval_of_not_mem _ hu

/-- A sum over operation gates is the sum of the blocks' sums. -/
theorem sum_blocks {C : Circuit} {n : ℕ} (P : Partition C n) {V : Finset (Fin C.N)} (hV : V ⊆ C.computed)
    (ω : Fin C.N → ℝ) : ∑ v ∈ V, ω v = ∑ u, ∑ v ∈ V ∩ P.gates u, ω v := by
  have hfil : ∀ u, V ∩ P.gates u = V.filter fun v => P.unit v = some u := fun u => by
    ext v
    simp [Partition.mem_gates]
  simp only [hfil, Finset.sum_filter]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun v hv => ?_
  have hc : C.isInput v = false := by
    have := hV hv
    simpa [Circuit.computed] using this
  obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 fun h => by
    rw [P.input_iff] at h
    rw [h] at hc
    cases hc
  simp only [hu, Option.some.injEq]
  rw [Finset.sum_eq_single u (fun b _ hb => by simp [Ne.symm hb]) (fun h => absurd (Finset.mem_univ u) h)]
  simp

/-- Exact blocks: `|C| = c B`. -/
theorem size_eq {C : Circuit} {n c : ℕ} (P : Partition C n) (h : Exact P c) : C.size C.computed = c * n := by
  unfold Circuit.size
  rw [← P.sum_card_gates]
  simp only [Exact, Circuit.size] at h
  simp [h, mul_comm]

namespace SecurityProofs

theorem BlockDescription : Guarantees.BlockDescription := by
  intro R k D t ext aBlk inp hk hX hev u
  obtain ⟨enc, henc⟩ := hev
  obtain ⟨-, hlen, hdec⟩ := henc u _ hX
  rw [localEval_eval] at hdec
  refine (LDL_le hdec).trans ?_
  have := hk u
  exact_mod_cast (by omega : (enc u (R.d.D.eval inp)).length ≤ k + Nat.clog 2 R.d.n + aBlk)

theorem Counting : Guarantees.Counting := by
  intro C n c P V ω q T₀ hc hV hq hT hblk
  have hsum : ∑ v ∈ V, ω v ≤ n * q := by
    rw [sum_blocks P hV]
    calc ∑ u, ∑ v ∈ V ∩ P.gates u, ω v ≤ ∑ _u : Fin n, q := Finset.sum_le_sum fun u _ => hblk u
      _ = n * q := by simp
  have hsize : (C.size C.computed : ℝ) = c * n := by exact_mod_cast size_eq P hc
  have h1 : c / q * ∑ v ∈ V, ω v ≤ C.size C.computed := by
    rw [hsize]
    calc c / q * ∑ v ∈ V, ω v ≤ c / q * (n * q) :=
          mul_le_mul_of_nonneg_left hsum (div_nonneg (Nat.cast_nonneg _) hq.le)
      _ = c * n := by field_simp
  refine ⟨h1, ?_⟩
  have he : c / (T₀ * q) * ∑ v ∈ V, ω v = (c / q * ∑ v ∈ V, ω v) / T₀ := by
    field_simp
  rw [he]
  exact div_le_div_of_nonneg_right h1 hT.le

theorem FanOutBound : Guarantees.FanOutBound := by
  intro R c k ι A _ M e D t aBlk a β Λ T₀ inp hck hX hfan hβ hT hev hrate
  obtain ⟨hc, hk⟩ := hck
  set h : ℝ := k + Nat.clog 2 R.d.n + aBlk + Λ with hh
  have hV : ∀ u, β * (M.acts R.d.𝓘 u).card ≤ h := fun u => by
    have h1 := BlockDescription R k D t a aBlk inp hk hX hev u
    have h2 : ENNReal.ofReal (β * (M.acts R.d.𝓘 u).card - Λ) ≤ ((k + Nat.clog 2 R.d.n + aBlk : ℕ) : ℝ≥0∞) :=
      (hrate u).trans (by exact_mod_cast ENat.toENNReal_le.2 h1)
    rw [ENNReal.ofReal_le_natCast] at h2
    push_cast at h2
    linarith
  have hJ : ∀ u, (M.inBlock R.d.𝓘 u).card ≤ e * (M.acts R.d.𝓘 u).card := fun u =>
    Finset.card_le_mul_card_image _ e fun j _ =>
      (Finset.card_le_card (Finset.filter_subset_filter _ (Finset.filter_subset _ _))).trans (hfan j)
  have hrhs : 0 ≤ (R.d.D.size R.d.D.computed : ℝ) / T₀ := div_nonneg (Nat.cast_nonneg _) hT.le
  by_cases hpos : 0 < (e : ℝ) * h
  · set V := M.J.image M.rep
    have hq : 0 < e * h / β := div_pos hpos hβ
    have hVsub : V ⊆ R.d.D.computed := fun v hv => by
      obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hv
      simpa [Circuit.computed] using M.rep_op p hp
    have hblk : ∀ u, ∑ v ∈ V ∩ R.d.𝓘.gates u, (fun _ => (1 : ℝ)) v ≤ e * h / β := fun u => by
      have hsub : V ∩ R.d.𝓘.gates u ⊆ (M.inBlock R.d.𝓘 u).image M.rep := fun v hv => by
        obtain ⟨hv, hg⟩ := Finset.mem_inter.1 hv
        obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hv
        exact Finset.mem_image.2 ⟨p, Finset.mem_filter.2 ⟨hp, (R.d.𝓘.mem_gates).1 hg⟩, rfl⟩
      have hcard : ((V ∩ R.d.𝓘.gates u).card : ℝ) ≤ (M.inBlock R.d.𝓘 u).card :=
        by exact_mod_cast (Finset.card_le_card hsub).trans Finset.card_image_le
      have hJu : ((M.inBlock R.d.𝓘 u).card : ℝ) ≤ e * (M.acts R.d.𝓘 u).card := by exact_mod_cast hJ u
      simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
      rw [le_div_iff₀ hβ]
      have := hV u
      nlinarith [Nat.cast_nonneg (α := ℝ) e, Nat.cast_nonneg (α := ℝ) (M.acts R.d.𝓘 u).card]
    obtain ⟨-, h2⟩ := Counting R.d.D R.d.n c R.d.𝓘 V (fun _ => 1) (e * h / β) T₀ hc hVsub hq hT hblk
    have hVJ : V.card = M.J.card := Finset.card_image_of_injOn M.rep_inj
    simp only [Finset.sum_const, nsmul_eq_mul, mul_one, hVJ] at h2
    have he : (c : ℝ) / (T₀ * (e * h / β)) * M.J.card = c * β * M.J.card / (T₀ * e * h) := by
      field_simp
    rw [he] at h2
    exact h2
  · refine le_trans ?_ hrhs
    apply div_nonpos_of_nonneg_of_nonpos
    · have := Nat.cast_nonneg (α := ℝ) c
      have := Nat.cast_nonneg (α := ℝ) M.J.card
      positivity
    · have : T₀ * e * h = T₀ * (e * h) := by ring
      rw [this]
      exact mul_nonpos_of_nonneg_of_nonpos hT.le (not_lt.1 hpos)

end SecurityProofs

end Nci.Ldl
