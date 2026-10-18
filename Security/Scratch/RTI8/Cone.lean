import Mathlib

/-!
RT-I8 r2 scratch (never lands): `Covers` for a share that charges each bad unit its cone.

Units `U`, nonnegative weights `wt`, an edge relation `R` (a unit feeds another), `bad` units (not held, or wrong at
their committed values). A unit is good when no unit of its upstream cone along `R` (itself included) is bad. The cone
weight of `u` is the weight of every unit `u` reaches. Then the units that aren't good weigh at most the bad units' cone
weights, so a share that charges each skipped unit its cone covers every unit that isn't good, whatever `R` is. With
fewer edges (`R' ≤ R`) more units are good, so E1(c) per unit at `R'` implies it at `R`.
-/

namespace RTI8R2

open Classical

variable {U : Type} [Fintype U]

def Good (R : U → U → Prop) (bad : U → Prop) (v : U) : Prop := ∀ u, Relation.ReflTransGen R u v → ¬ bad u

noncomputable def coneWt (R : U → U → Prop) (wt : U → ℝ) (u : U) : ℝ :=
  ∑ v, if Relation.ReflTransGen R u v then wt v else 0

theorem covers_of_cone (R : U → U → Prop) (bad : U → Prop) (wt : U → ℝ) (hw : ∀ u, 0 ≤ wt u) :
    (∑ v, if Good R bad v then 0 else wt v) ≤ ∑ u, if bad u then coneWt R wt u else 0 := by
  have key : ∀ v, (if Good R bad v then 0 else wt v) ≤
      ∑ u, if bad u then (if Relation.ReflTransGen R u v then wt v else 0) else 0 := by
    intro v
    by_cases hg : Good R bad v
    · rw [if_pos hg]
      exact Finset.sum_nonneg fun u _ => by split_ifs <;> simp [hw v]
    · rw [if_neg hg]
      obtain ⟨u, hu, hb⟩ : ∃ u, Relation.ReflTransGen R u v ∧ bad u := by
        by_contra h
        push_neg at h
        exact hg fun u hr hb => h u hr hb
      calc wt v = if bad u then (if Relation.ReflTransGen R u v then wt v else 0) else 0 := by
            rw [if_pos hb, if_pos hu]
        _ ≤ _ := Finset.single_le_sum (f := fun u => if bad u then
              (if Relation.ReflTransGen R u v then wt v else 0) else 0)
            (fun u _ => by split_ifs <;> simp [hw v]) (Finset.mem_univ u)
  calc (∑ v, if Good R bad v then 0 else wt v)
      ≤ ∑ v, ∑ u, if bad u then (if Relation.ReflTransGen R u v then wt v else 0) else 0 :=
        Finset.sum_le_sum fun v _ => key v
    _ = ∑ u, ∑ v, if bad u then (if Relation.ReflTransGen R u v then wt v else 0) else 0 := Finset.sum_comm
    _ = ∑ u, if bad u then coneWt R wt u else 0 := by
        refine Finset.sum_congr rfl fun u _ => ?_
        by_cases hb : bad u
        · simp only [hb, if_true, coneWt]
        · simp [hb]

theorem rtg_mono {R R' : U → U → Prop} (hR : ∀ a b, R' a b → R a b) {u v : U}
    (hr : Relation.ReflTransGen R' u v) : Relation.ReflTransGen R u v := by
  induction hr with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hbc ih => exact Relation.ReflTransGen.tail ih (hR _ _ hbc)

theorem good_mono {R R' : U → U → Prop} (hR : ∀ a b, R' a b → R a b) (bad : U → Prop) (v : U)
    (h : Good R bad v) : Good R' bad v :=
  fun u hr => h u (rtg_mono hR hr)

/-- E1(c) per unit at the smaller edge set implies it at the larger: its good units are fewer. -/
theorem unitFixed_mono {R R' : U → U → Prop} (hR : ∀ a b, R' a b → R a b) (bad : U → Prop) (wt : U → ℝ)
    (hw : ∀ u, 0 ≤ wt u) (fixed : ℝ) (h : (∑ v, if Good R' bad v then wt v else 0) ≤ fixed) :
    (∑ v, if Good R bad v then wt v else 0) ≤ fixed :=
  le_trans (Finset.sum_le_sum fun v _ => by
    by_cases hg : Good R bad v
    · simp [hg, good_mono hR bad v hg]
    · simp only [hg, if_false]; split_ifs <;> simp [hw v]) h

/-- The cone of a unit along same-stage edges is at most `F` times its weight when every chain within a stage has at
most `F` units of at most its weight: the share's surcharge over the skipped work. -/
theorem coneWt_le (R : U → U → Prop) (wt : U → ℝ) (hw : ∀ u, 0 ≤ wt u) (u : U) (F : ℕ)
    (hcard : (Finset.univ.filter fun v => Relation.ReflTransGen R u v).card ≤ F)
    (hle : ∀ v, Relation.ReflTransGen R u v → wt v ≤ wt u) : coneWt R wt u ≤ F * wt u := by
  unfold coneWt
  rw [← Finset.sum_filter]
  calc (∑ v ∈ Finset.univ.filter fun v => Relation.ReflTransGen R u v, wt v)
      ≤ ∑ _v ∈ Finset.univ.filter fun v => Relation.ReflTransGen R u v, wt u :=
        Finset.sum_le_sum fun v hv => hle v (Finset.mem_filter.mp hv).2
    _ = (Finset.univ.filter fun v => Relation.ReflTransGen R u v).card * wt u := by
        rw [Finset.sum_const, nsmul_eq_mul]
    _ ≤ F * wt u := mul_le_mul_of_nonneg_right (by exact_mod_cast hcard) (hw u)

end RTI8R2

#print axioms RTI8R2.covers_of_cone
#print axioms RTI8R2.unitFixed_mono
#print axioms RTI8R2.coneWt_le
