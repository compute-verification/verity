import Definitions.Nci.Guarantees.Capability.Dependency
import Proofs.Nci.Capability.Items
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Tactic.Linarith

/-!
# Local bounds: proofs of the caps from dependency

* `FeedsWithin.exists_into`: a path inside a region from a gate outside `S` to a gate `S` reads crosses into `S` at a gate
  of the region, which lies in `in(S)`;
* `ImportOrRecompute`: a finished consumer whose required gates are not all in the block has a required gate outside it,
  whose path to the finishing set enters the block at a gate of `in(S) ∩ A c`, unregistered since `A c` misses `θ`; each
  such consumer is charged to those gates, and the overlap weights bound what one gate carries by 1;
* `RecomputeByGates`: `g Σ ω ≤ Σ_c |G c| ω_c = Σ_{x ∈ S} Σ_{c ∋ x} ω_c ≤ |S|`;
* `RecomputeByState`: the state a recomputation needs lies among what the block holds, so it is within the budget;
* `WeightedProductCap`: the completed products lie in the rectangle of their slots and uses, whose weighted sum factors,
  and their slots are a selection the block holds.
-/

namespace Nci.Covering.FeedsWithin

open Verity Finset

theorem exists_into {C : Circuit} {R S : Finset (Fin C.N)} {g t : Fin C.N} (h : FeedsWithin C R g t)
    (hg : g ∉ S) (ht : ∃ y ∈ S, t ∈ C.reads y) : ∃ x ∈ R, x ∈ C.into S := by
  induction h with
  | refl htR =>
    obtain ⟨y, hy, hty⟩ := ht
    exact ⟨_, htR, mem_filter.2 ⟨mem_univ _, hg, y, hy, hty⟩⟩
  | @step g' h' t' hgR hgh _ ih =>
    by_cases hh : h' ∈ S
    · exact ⟨g', hgR, mem_filter.2 ⟨mem_univ _, hg, h', hh, hgh⟩⟩
    · exact ih hh ht

end Nci.Covering.FeedsWithin

namespace Nci.SecurityProofs

open Verity Finset Nci.Capability

theorem ImportOrRecompute : Guarantees.ImportOrRecompute := by
  intro R κ _ K ω hK hω u
  classical
  set F := R.P.gates u
  set Sv := K.servedIn F
  set S := R.d.unregistered F
  rw [← sum_filter_add_sum_filter_not Sv (fun c => K.G c ⊆ F)]
  suffices h : ∑ c ∈ Sv.filter (fun c => ¬ K.G c ⊆ F), ω c ≤ S.card by linarith
  have hx : ∀ c ∈ Sv.filter (fun c => ¬ K.G c ⊆ F), ∃ x ∈ S, x ∈ K.A c := by
    intro c hc
    obtain ⟨hcS, hcG⟩ := mem_filter.1 hc
    obtain ⟨F', hF', hF'F⟩ := (mem_filter.1 hcS).2
    obtain ⟨g, hgG, hgF⟩ := not_subset.1 hcG
    obtain ⟨a, ha, hfeeds⟩ := hK c F' hF' g hgG
    obtain ⟨-, -, y, hy, hay⟩ := mem_filter.1 (mem_inter.1 ha).1
    obtain ⟨x, hxA, hxI⟩ := hfeeds.exists_into hgF ⟨y, hF'F hy, hay⟩
    exact ⟨x, mem_sdiff.2 ⟨hxI, fun hθ => disjoint_left.1 (K.A_θ c) hxA hθ⟩, hxA⟩
  have hterm : ∀ c x, 0 ≤ if x ∈ K.A c then ω c else 0 := fun c x => by
    split_ifs
    · exact hω.1 c
    · exact le_rfl
  calc ∑ c ∈ Sv.filter (fun c => ¬ K.G c ⊆ F), ω c
      ≤ ∑ c ∈ Sv.filter (fun c => ¬ K.G c ⊆ F), ∑ x ∈ S, if x ∈ K.A c then ω c else 0 := by
        refine sum_le_sum fun c hc => ?_
        obtain ⟨x, hx, hxA⟩ := hx c hc
        have := single_le_sum (f := fun x => if x ∈ K.A c then ω c else 0) (fun x _ => hterm c x) hx
        simpa [hxA] using this
    _ = ∑ x ∈ S, ∑ c ∈ Sv.filter (fun c => ¬ K.G c ⊆ F), if x ∈ K.A c then ω c else 0 := sum_comm
    _ ≤ ∑ x ∈ S, ∑ c ∈ univ.filter (fun c => x ∈ K.A c), ω c := by
        refine sum_le_sum fun x _ => ?_
        rw [sum_filter (fun c => x ∈ K.A c)]
        exact sum_le_sum_of_subset_of_nonneg (subset_univ _) fun c _ _ => hterm c x
    _ ≤ ∑ _x ∈ S, (1 : ℝ) := sum_le_sum fun x _ => hω.2 x
    _ = S.card := by simp

theorem RecomputeByGates : Guarantees.RecomputeByGates := by
  intro d κ _ K ω g hω hg J S
  classical
  set J' := J.filter fun c => K.G c ⊆ S
  have hterm : ∀ (c : κ) (x : Fin d.D.N) (T : Finset (Fin d.D.N)), 0 ≤ if x ∈ T then ω c else 0 := fun c x T => by
    split_ifs
    · exact hω.1 c
    · exact le_rfl
  calc g * ∑ c ∈ J', ω c = ∑ c ∈ J', g * ω c := mul_sum _ _ _
    _ ≤ ∑ c ∈ J', ((K.G c).card : ℝ) * ω c :=
        sum_le_sum fun c _ => mul_le_mul_of_nonneg_right (hg c) (hω.1 c)
    _ = ∑ c ∈ J', ∑ x ∈ S, if x ∈ K.G c then ω c else 0 := by
        refine sum_congr rfl fun c hc => ?_
        rw [← sum_filter, filter_mem_eq_inter, inter_eq_right.2 (mem_filter.1 hc).2, sum_const, nsmul_eq_mul]
    _ = ∑ x ∈ S, ∑ c ∈ J', if x ∈ K.G c then ω c else 0 := sum_comm
    _ ≤ ∑ x ∈ S, ∑ c ∈ univ.filter (fun c => x ∈ K.A c), ω c := by
        refine sum_le_sum fun x _ => ?_
        rw [sum_filter (fun c => x ∈ K.A c)]
        calc ∑ c ∈ J', (if x ∈ K.G c then ω c else 0)
            ≤ ∑ c ∈ J', (if x ∈ K.A c then ω c else 0) := by
              refine sum_le_sum fun c _ => ?_
              by_cases hxG : x ∈ K.G c
              · simp [hxG, K.G_sub c hxG]
              · simpa [hxG] using hterm c x (K.A c)
          _ ≤ ∑ c, (if x ∈ K.A c then ω c else 0) :=
              sum_le_sum_of_subset_of_nonneg (subset_univ _) fun c _ _ => hterm c x (K.A c)
    _ ≤ ∑ _x ∈ S, (1 : ℝ) := sum_le_sum fun x _ => hω.2 x
    _ = S.card := by simp

theorem RecomputeByState : Guarantees.RecomputeByState := by
  intro R ι W κ K h hW hK u c hc
  obtain ⟨U, σ, hsel, hσ, hlt⟩ := hK c
  exact absurd (hW u U σ hsel fun i hi => (hσ i hi).trans (holds_mono hc)) (not_le.2 hlt)

theorem WeightedProductCap : Guarantees.WeightedProductCap := by
  intro R B W ω h hS hω hW u
  classical
  set P := completedIn B (R.P.gates u)
  have hP : ∀ q ∈ P, q ∈ B.P ∧ B.m q ⊆ R.P.gates u := fun q hq => mem_filter.1 hq
  obtain ⟨σ, hσ, hσ'⟩ :=
    exists_selection W P Prod.fst B.m (R.P.gates u) (fun q hq => hS q (hP q hq).1) (fun q hq => (hP q hq).2)
  have hU := hW u _ σ hσ hσ'
  have hV : 0 ≤ ∑ j ∈ served B (R.P.gates u), ω j := sum_nonneg fun j _ => hω j
  calc ∑ q ∈ P, W.price q.1 * ω q.2
      ≤ ∑ q ∈ (P.image Prod.fst) ×ˢ (P.image Prod.snd), W.price q.1 * ω q.2 := by
        refine sum_le_sum_of_subset_of_nonneg (fun q hq => ?_) fun q _ _ => mul_nonneg (W.price_nonneg _) (hω _)
        exact mem_product.2 ⟨mem_image_of_mem _ hq, mem_image_of_mem _ hq⟩
    _ = (∑ i ∈ P.image Prod.fst, W.price i) * ∑ j ∈ served B (R.P.gates u), ω j := by
        rw [sum_product, sum_mul_sum]
        rfl
    _ ≤ h * ∑ j ∈ served B (R.P.gates u), ω j := mul_le_mul_of_nonneg_right hU hV

end Nci.SecurityProofs
