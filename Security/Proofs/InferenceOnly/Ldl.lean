import Definitions.InferenceOnly.Guarantees.Ldl
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
* The pair bound is Lemma 1, the pair count `|𝓙_I| ≤ |W_I| |S_I| ≤ h² / (4 βw βe)` (`card_le_pairs`) and Theorem 3 with
  unit weights.
-/

namespace InferenceOnly.Ldl

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

/-- Occurrences naming distinct (weight, entry) pairs number at most `|W| |S|`, and when `βw |W| + βe |S| ≤ h` that is
at most `h² / (4 βw βe)`. -/
theorem card_le_pairs {ι W S : Type} [DecidableEq W] [DecidableEq S] (J : Finset ι) (wt : ι → W) (src : ι → S)
    (hinj : Set.InjOn (fun p => (wt p, src p)) J) {βw βe h : ℝ} (hw : 0 < βw) (he : 0 < βe)
    (hh : βw * (J.image wt).card + βe * (J.image src).card ≤ h) :
    4 * βw * βe * J.card ≤ h ^ 2 := by
  have hJ : J.card ≤ (J.image wt).card * (J.image src).card := by
    rw [← Finset.card_product]
    exact Finset.card_le_card_of_injOn _ (fun p hp => Finset.mem_product.2
      ⟨Finset.mem_image_of_mem _ hp, Finset.mem_image_of_mem _ hp⟩) hinj
  have hJr : (J.card : ℝ) ≤ (J.image wt).card * (J.image src).card := by exact_mod_cast hJ
  set x : ℝ := ((J.image wt).card : ℝ)
  set y : ℝ := ((J.image src).card : ℝ)
  have hx : 0 ≤ x := Nat.cast_nonneg _
  have hy : 0 ≤ y := Nat.cast_nonneg _
  have hsum : 0 ≤ βw * x + βe * y := by positivity
  calc 4 * βw * βe * J.card ≤ 4 * βw * βe * (x * y) := by
        have : 0 ≤ 4 * βw * βe := by positivity
        exact mul_le_mul_of_nonneg_left hJr this
    _ ≤ (βw * x + βe * y) ^ 2 := by nlinarith [sq_nonneg (βw * x - βe * y)]
    _ ≤ h ^ 2 := pow_le_pow_left₀ hsum hh 2

namespace SecurityProofs

theorem BlockDescription : Guarantees.BlockDescription := by
  intro d n k R D t ext aBlk inp hk hX hev u
  obtain ⟨enc, henc⟩ := hev
  obtain ⟨-, hlen, hdec⟩ := henc u _ hX
  rw [localEval_eval] at hdec
  refine (LDL_le hdec).trans ?_
  have := hk u
  exact_mod_cast (by omega : (enc u (d.D.eval inp)).length ≤ k + Nat.clog 2 n + aBlk)

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
  intro d n c k R ι A _ M e D t aBlk a β Λ T₀ inp hck hX hfan hβ hT hev hrate
  obtain ⟨hc, hk⟩ := hck
  set h : ℝ := k + Nat.clog 2 n + aBlk + Λ with hh
  have hV : ∀ u, β * (M.acts R.P u).card ≤ h := fun u => by
    have h1 := BlockDescription d n k R D t a aBlk inp hk hX hev u
    have h2 : ENNReal.ofReal (β * (M.acts R.P u).card - Λ) ≤ ((k + Nat.clog 2 n + aBlk : ℕ) : ℝ≥0∞) :=
      (hrate u).trans (by exact_mod_cast ENat.toENNReal_le.2 h1)
    rw [ENNReal.ofReal_le_natCast] at h2
    push_cast at h2
    linarith
  have hJ : ∀ u, (M.inBlock R.P u).card ≤ e * (M.acts R.P u).card := fun u =>
    Finset.card_le_mul_card_image _ e fun j _ =>
      (Finset.card_le_card (Finset.filter_subset_filter _ (Finset.filter_subset _ _))).trans (hfan j)
  have hrhs : 0 ≤ (d.D.size d.D.computed : ℝ) / T₀ := div_nonneg (Nat.cast_nonneg _) hT.le
  by_cases hpos : 0 < (e : ℝ) * h
  · set V := M.J.image M.rep
    have hq : 0 < e * h / β := div_pos hpos hβ
    have hVsub : V ⊆ d.D.computed := fun v hv => by
      obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hv
      simpa [Circuit.computed] using M.rep_op p hp
    have hblk : ∀ u, ∑ v ∈ V ∩ R.P.gates u, (fun _ => (1 : ℝ)) v ≤ e * h / β := fun u => by
      have hsub : V ∩ R.P.gates u ⊆ (M.inBlock R.P u).image M.rep := fun v hv => by
        obtain ⟨hv, hg⟩ := Finset.mem_inter.1 hv
        obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hv
        exact Finset.mem_image.2 ⟨p, Finset.mem_filter.2 ⟨hp, (R.P.mem_gates).1 hg⟩, rfl⟩
      have hcard : ((V ∩ R.P.gates u).card : ℝ) ≤ (M.inBlock R.P u).card :=
        by exact_mod_cast (Finset.card_le_card hsub).trans Finset.card_image_le
      have hJu : ((M.inBlock R.P u).card : ℝ) ≤ e * (M.acts R.P u).card := by exact_mod_cast hJ u
      simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
      rw [le_div_iff₀ hβ]
      have := hV u
      nlinarith [Nat.cast_nonneg (α := ℝ) e, Nat.cast_nonneg (α := ℝ) (M.acts R.P u).card]
    obtain ⟨-, h2⟩ := Counting d.D n c R.P V (fun _ => 1) (e * h / β) T₀ hc hVsub hq hT hblk
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

theorem PairBound : Guarantees.PairBound := by
  intro d n c k R ι A W S _ _ M wt src D t aBlk a βw βe ℓ T₀ inp hck hX hinj hw he hT hev hrate
  obtain ⟨hc, hk⟩ := hck
  set h : ℝ := k + Nat.clog 2 n + aBlk + ℓ with hh
  have hpair : ∀ u, 4 * βw * βe * (M.inBlock R.P u).card ≤ h ^ 2 := fun u => by
    have h1 := BlockDescription d n k R D t a aBlk inp hk hX hev u
    have h2 : ENNReal.ofReal (βw * ((M.inBlock R.P u).image wt).card + βe * ((M.inBlock R.P u).image src).card - ℓ) ≤
        ((k + Nat.clog 2 n + aBlk : ℕ) : ℝ≥0∞) :=
      (hrate u).trans (by exact_mod_cast ENat.toENNReal_le.2 h1)
    rw [ENNReal.ofReal_le_natCast] at h2
    push_cast at h2
    have hsubJ : M.inBlock R.P u ⊆ M.J := Finset.filter_subset _ _
    exact card_le_pairs (M.inBlock R.P u) wt src (hinj.mono (Finset.coe_subset.2 hsubJ)) hw he (by linarith)
  have hrhs : 0 ≤ (d.D.size d.D.computed : ℝ) / T₀ := div_nonneg (Nat.cast_nonneg _) hT.le
  by_cases hpos : h ^ 2 = 0
  · rw [hpos, mul_zero, div_zero]
    exact hrhs
  · have hsq : 0 < h ^ 2 := lt_of_le_of_ne (sq_nonneg h) (Ne.symm hpos)
    set V := M.J.image M.rep
    have hq : 0 < h ^ 2 / (4 * βw * βe) := div_pos hsq (by positivity)
    have hVsub : V ⊆ d.D.computed := fun v hv => by
      obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hv
      simpa [Circuit.computed] using M.rep_op p hp
    have hblk : ∀ u, ∑ v ∈ V ∩ R.P.gates u, (fun _ => (1 : ℝ)) v ≤ h ^ 2 / (4 * βw * βe) := fun u => by
      have hsub : V ∩ R.P.gates u ⊆ (M.inBlock R.P u).image M.rep := fun v hv => by
        obtain ⟨hv, hg⟩ := Finset.mem_inter.1 hv
        obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hv
        exact Finset.mem_image.2 ⟨p, Finset.mem_filter.2 ⟨hp, (R.P.mem_gates).1 hg⟩, rfl⟩
      have hcard : ((V ∩ R.P.gates u).card : ℝ) ≤ (M.inBlock R.P u).card :=
        by exact_mod_cast (Finset.card_le_card hsub).trans Finset.card_image_le
      simp only [Finset.sum_const, nsmul_eq_mul, mul_one]
      rw [le_div_iff₀ (by positivity)]
      have := mul_le_mul_of_nonneg_right hcard (by positivity : (0 : ℝ) ≤ 4 * βw * βe)
      linarith [hpair u]
    obtain ⟨-, h2⟩ := Counting d.D n c R.P V (fun _ => 1) (h ^ 2 / (4 * βw * βe)) T₀ hc hVsub hq hT hblk
    have hVJ : V.card = M.J.card := Finset.card_image_of_injOn M.rep_inj
    simp only [Finset.sum_const, nsmul_eq_mul, mul_one, hVJ] at h2
    have hrw : (c : ℝ) / (T₀ * (h ^ 2 / (4 * βw * βe))) * M.J.card = 4 * c * βw * βe * M.J.card / (T₀ * h ^ 2) := by
      field_simp
    rw [hrw] at h2
    exact h2

end SecurityProofs

end InferenceOnly.Ldl
