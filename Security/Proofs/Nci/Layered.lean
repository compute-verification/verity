import Definitions.Nci.Guarantees.Layered
import Proofs.Nci.Threads

/-!
# Proofs that keyed circuits are threaded

Fix a small subcircuit `R` and a thread `t`. Following keys from any value of `t` in `R` reaches either a value of `t`
that `R` reads (where the path leaves `R`) or a stop in `R` (`walk`). Each leg of a root in `R` does so too, and two legs
never reach the same value, so a root of `t` in `R` gives `d ≤ #(t's reads) + #(t's lookups in R)`, by induction down
the roots (`root_reads`). The lookups of `t` in `R` read disjoint columns of novel inputs, so `R` holds at most `E` of
them when it reads less than `(E + 1) · b` bits (`lookupsIn_bits`). A subcircuit that reads fewer than `d − E` values of
`t` thus holds no root of `t`, and each of `t`'s counted multiplications in `R` is charged to a read (at most `e` each)
or to a lookup (at most `eLk` each), with a lookup costing a read of its token (`threaded_of_layered`).
-/

namespace Nci.SecurityProofs

open Verity Finset

theorem threadMults_subset {c : Declaration} (T : Threading c) (R : Finset (Fin c.D.N)) (t : Fin T.m) :
    T.threadMults R t ⊆ T.M := by
  intro g hg
  rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter] at hg
  exact hg.1.1

section Layered

variable {c : Declaration} {T : Threading c} {d e eLk b : ℕ}

/-- The lookups of thread `t` in `R`. -/
def lookupsIn (L : Layered T d e eLk b) (R : Finset (Fin c.D.N)) (t : Fin T.m) : Finset (Fin c.D.N) :=
  R.filter fun v => v ∈ L.lookups ∧ T.thread v = some t

theorem mem_lookupsIn {L : Layered T d e eLk b} {R : Finset (Fin c.D.N)} {t : Fin T.m} {v : Fin c.D.N} :
    v ∈ lookupsIn L R t ↔ v ∈ R ∧ v ∈ L.lookups ∧ T.thread v = some t := by
  unfold lookupsIn
  exact mem_filter

open Classical in
/-- The counted multiplications of thread `t` in `R` whose activation's key path reaches `u`. -/
noncomputable def fiber (L : Layered T d e eLk b) (R : Finset (Fin c.D.N)) (t : Fin T.m) (u : Fin c.D.N) :
    Finset (Fin c.D.N) :=
  (T.threadMults R t).filter fun g => KeyPath L.key (L.roots ∪ L.lookups) (T.act g) u

theorem mem_fiber {L : Layered T d e eLk b} {R : Finset (Fin c.D.N)} {t : Fin T.m} {u g : Fin c.D.N} :
    g ∈ fiber L R t u ↔ g ∈ T.threadMults R t ∧ KeyPath L.key (L.roots ∪ L.lookups) (T.act g) u := by
  classical
  unfold fiber
  exact mem_filter

/-- A key path only goes down. -/
theorem path_le (L : Layered T d e eLk b) {a u : Fin c.D.N} (h : KeyPath L.key (L.roots ∪ L.lookups) a u) :
    u ≤ a := by
  induction h with
  | refl a => exact le_rfl
  | step _ hk _ ih => exact ih.trans (c.D.reads_lt _ _ (L.key_reads _ _ hk)).le

/-- A key path stays in its thread. -/
theorem path_thread (L : Layered T d e eLk b) {a u : Fin c.D.N} (h : KeyPath L.key (L.roots ∪ L.lookups) a u) :
    T.thread u = T.thread a := by
  induction h with
  | refl a => rfl
  | step _ hk _ ih => exact ih.trans (L.key_thread _ _ hk)

/-- Following keys from a value of thread `t` in `R` reaches a value of `t` that `R` reads, or a stop in `R`. -/
theorem walk (L : Layered T d e eLk b) (R : Finset (Fin c.D.N)) (hR : R ⊆ c.D.computed) (t : Fin T.m) :
    ∀ a, a ∈ R → T.thread a = some t → ∃ u, KeyPath L.key (L.roots ∪ L.lookups) a u ∧
      (u ∈ T.threadReads R t ∨ (u ∈ R ∧ u ∈ L.roots ∪ L.lookups)) := by
  intro a
  induction a using WellFoundedLT.induction with
  | _ a ih =>
    intro haR hat
    by_cases hs : a ∈ L.roots ∪ L.lookups
    · exact ⟨a, KeyPath.refl a, Or.inr ⟨haR, hs⟩⟩
    · obtain ⟨k, hk⟩ := Option.isSome_iff_exists.1 (L.keyed a t (mem_filter.1 (hR haR)).2 hat hs)
      have hkr : k ∈ c.D.reads a := L.key_reads a k hk
      have hkt : T.thread k = some t := (L.key_thread a k hk).trans hat
      by_cases hkR : k ∈ R
      · obtain ⟨u, hku, hu⟩ := ih k (c.D.reads_lt a k hkr) hkR hkt
        exact ⟨u, KeyPath.step hs hk hku, hu⟩
      · refine ⟨k, KeyPath.step hs hk (KeyPath.refl k),
          Or.inl (mem_filter.2 ⟨mem_sdiff.2 ⟨?_, L.thread_θ k t hkt⟩, hkt⟩)⟩
        simp only [Circuit.into, mem_filter, mem_univ, true_and]
        exact ⟨hkR, a, haR, hkr⟩

/-- **A root costs `d`**: a subcircuit that holds a root of thread `t` reads `d` values of `t` or holds lookups of `t`
to make up the difference. -/
theorem root_reads (L : Layered T d e eLk b) (R : Finset (Fin c.D.N)) (hR : R ⊆ c.D.computed) (t : Fin T.m) :
    ∀ r, r ∈ R → r ∈ L.roots → T.thread r = some t →
      d ≤ (T.threadReads R t).card + (lookupsIn L R t).card := by
  classical
  intro r
  induction r using WellFoundedLT.induction with
  | _ r ih =>
    intro hrR hroot hrt
    have hleg : ∀ p ∈ L.legs r, ∃ u, KeyPath L.key (L.roots ∪ L.lookups) p u ∧
        (u ∈ T.threadReads R t ∪ lookupsIn L R t ∨ (u ∈ R ∧ u ∈ L.roots ∧ u < r)) := by
      intro p hp
      have hpr : p ∈ c.D.reads r := L.legs_reads r hroot hp
      have hpt : T.thread p = some t := (L.legs_thread r hroot p hp).trans hrt
      by_cases hpR : p ∈ R
      · obtain ⟨u, hpu, hu⟩ := walk L R hR t p hpR hpt
        refine ⟨u, hpu, ?_⟩
        rcases hu with hu | ⟨huR, hus⟩
        · exact Or.inl (mem_union_left _ hu)
        · rcases mem_union.1 hus with hu' | hu'
          · exact Or.inr ⟨huR, hu', (path_le L hpu).trans_lt (c.D.reads_lt r p hpr)⟩
          · exact Or.inl (mem_union_right _ (mem_lookupsIn.2 ⟨huR, hu', (path_thread L hpu).trans hpt⟩))
      · refine ⟨p, KeyPath.refl p,
          Or.inl (mem_union_left _ (mem_filter.2 ⟨mem_sdiff.2 ⟨?_, L.thread_θ p t hpt⟩, hpt⟩))⟩
        simp only [Circuit.into, mem_filter, mem_univ, true_and]
        exact ⟨hpR, r, hrR, hpr⟩
    choose! f hf using hleg
    by_cases hlow : ∃ p ∈ L.legs r, f p ∉ T.threadReads R t ∪ lookupsIn L R t
    · obtain ⟨p, hp, hpA⟩ := hlow
      rcases (hf p hp).2 with h | ⟨huR, hu, hur⟩
      · exact absurd h hpA
      · exact ih (f p) hur huR hu ((path_thread L (hf p hp).1).trans ((L.legs_thread r hroot p hp).trans hrt))
    · have hall : ∀ p ∈ L.legs r, f p ∈ T.threadReads R t ∪ lookupsIn L R t := fun p hp => by
        by_contra h
        exact hlow ⟨p, hp, h⟩
      have hinj : Set.InjOn f (L.legs r : Set (Fin c.D.N)) := by
        intro p hp q hq hpq
        by_contra hne
        have hq' := (hf q (mem_coe.1 hq)).1
        rw [← hpq] at hq'
        exact L.legs_disjoint r hroot p (mem_coe.1 hp) q (mem_coe.1 hq) hne (f p) (hf p (mem_coe.1 hp)).1 hq'
      calc d ≤ (L.legs r).card := L.legs_card r hroot
        _ ≤ (T.threadReads R t ∪ lookupsIn L R t).card :=
            card_le_card_of_injOn f (fun p hp => mem_coe.2 (hall p (mem_coe.1 hp))) hinj
        _ ≤ (T.threadReads R t).card + (lookupsIn L R t).card := card_union_le _ _

/-- **Lookups cost bits**: the lookups of a thread in `R` read disjoint columns of novel inputs, `b` bits each. -/
theorem lookupsIn_bits (L : Layered T d e eLk b) (R : Finset (Fin c.D.N)) (hR : R ⊆ c.D.computed) (t : Fin T.m) :
    (lookupsIn L R t).card * b ≤ novelReads c R := by
  classical
  have hsub : (lookupsIn L R t).biUnion L.col ⊆ c.D.into R \ c.θ := by
    refine biUnion_subset.2 fun v hv => ?_
    intro g hg
    obtain ⟨hvR, hvl, -⟩ := mem_lookupsIn.1 hv
    obtain ⟨hin, hθ⟩ := L.col_input v hvl g hg
    refine mem_sdiff.2 ⟨?_, hθ⟩
    simp only [Circuit.into, mem_filter, mem_univ, true_and]
    refine ⟨fun hgR => ?_, v, hvR, L.col_reads v hvl hg⟩
    have := (mem_filter.1 (hR hgR)).2
    rw [hin] at this
    cases this
  have hdisj : Set.PairwiseDisjoint (lookupsIn L R t : Set (Fin c.D.N)) L.col := by
    intro v hv v' hv' hne
    obtain ⟨-, hvl, hvt⟩ := mem_lookupsIn.1 (mem_coe.1 hv)
    obtain ⟨-, hvl', hvt'⟩ := mem_lookupsIn.1 (mem_coe.1 hv')
    exact L.col_disjoint v hvl v' hvl' hne (hvt.trans hvt'.symm)
  calc (lookupsIn L R t).card * b = ∑ _v ∈ lookupsIn L R t, b := by rw [sum_const, smul_eq_mul]
    _ ≤ ∑ v ∈ lookupsIn L R t, c.D.width (L.col v) :=
        sum_le_sum fun v hv => L.col_bits v (mem_lookupsIn.1 hv).2.1
    _ = c.D.width ((lookupsIn L R t).biUnion L.col) := by
        unfold Circuit.width
        rw [sum_biUnion hdisj]
    _ ≤ novelReads c R := by
        unfold novelReads Circuit.width
        exact sum_le_sum_of_subset hsub

/-- **Keyed circuits are threaded.** -/
theorem threaded_of_layered (L : Layered T d e eLk b) (Γ : ℕ) (S : ℝ) (E : ℕ) (hS : S < ((E + 1) * b : ℕ)) :
    T.Threaded Γ S (d - E) (e + E * eLk) := by
  classical
  intro R hR t hlt
  have hV : (lookupsIn L R t).card ≤ E := by
    by_contra h
    have h1 : (E + 1) * b ≤ (lookupsIn L R t).card * b := Nat.mul_le_mul_right b (by omega)
    have h2 : (((E + 1) * b : ℕ) : ℝ) ≤ (novelReads c R : ℝ) :=
      Nat.cast_le.2 (h1.trans (lookupsIn_bits L R hR.1 t))
    exact absurd (hR.2.2.trans_lt hS) (not_lt.2 h2)
  have hnoroot : ∀ r ∈ R, r ∈ L.roots → T.thread r ≠ some t := by
    intro r hrR hr hrt
    have := root_reads L R hR.1 t r hrR hr hrt
    omega
  have hcover : ∀ g ∈ T.threadMults R t, ∃ u, (u ∈ T.threadReads R t ∨ u ∈ lookupsIn L R t) ∧
      KeyPath L.key (L.roots ∪ L.lookups) (T.act g) u := by
    intro g hg
    have hg' := hg
    rw [Threading.threadMults, mem_filter, Threading.mults, mem_inter] at hg'
    obtain ⟨⟨hgM, hgR⟩, hgt⟩ := hg'
    by_cases haR : T.act g ∈ R
    · obtain ⟨u, hau, hu⟩ := walk L R hR.1 t _ haR hgt
      refine ⟨u, ?_, hau⟩
      rcases hu with hu | ⟨huR, hus⟩
      · exact Or.inl hu
      · have hut : T.thread u = some t := (path_thread L hau).trans hgt
        rcases mem_union.1 hus with hr | hl
        · exact absurd hut (hnoroot u huR hr)
        · exact Or.inr (mem_lookupsIn.2 ⟨huR, hl, hut⟩)
    · refine ⟨T.act g, Or.inl (mem_filter.2 ⟨mem_sdiff.2 ⟨?_, L.thread_θ _ t hgt⟩, hgt⟩), KeyPath.refl _⟩
      simp only [Circuit.into, mem_filter, mem_univ, true_and]
      exact ⟨haR, g, hgR, T.act_reads g hgM⟩
  have hsub : T.threadMults R t ⊆
      (T.threadReads R t).biUnion (fiber L R t) ∪ (lookupsIn L R t).biUnion (fiber L R t) := by
    intro g hg
    obtain ⟨u, hu, hp⟩ := hcover g hg
    rcases hu with hu | hu
    · exact mem_union_left _ (mem_biUnion.2 ⟨u, hu, mem_fiber.2 ⟨hg, hp⟩⟩)
    · exact mem_union_right _ (mem_biUnion.2 ⟨u, hu, mem_fiber.2 ⟨hg, hp⟩⟩)
  have hfib : ∀ u, fiber L R t u ⊆ T.M := by
    intro u g hg
    exact threadMults_subset T R t (mem_fiber.1 hg).1
  have hX : ∀ u, (fiber L R t u).card ≤ e := fun u =>
    L.reach u _ (hfib u) fun g hg => (mem_fiber.1 hg).2
  have hXl : ∀ v ∈ lookupsIn L R t, (fiber L R t v).card ≤ eLk := fun v hv =>
    L.reach_lookups v (mem_lookupsIn.1 hv).2.1 _ (hfib v) fun g hg => (mem_fiber.1 hg).2
  have hcount : (T.threadMults R t).card ≤ e * (T.threadReads R t).card + eLk * (lookupsIn L R t).card :=
    calc (T.threadMults R t).card
        ≤ ((T.threadReads R t).biUnion (fiber L R t) ∪ (lookupsIn L R t).biUnion (fiber L R t)).card :=
          card_le_card hsub
      _ ≤ ((T.threadReads R t).biUnion (fiber L R t)).card + ((lookupsIn L R t).biUnion (fiber L R t)).card :=
          card_union_le _ _
      _ ≤ ∑ u ∈ T.threadReads R t, (fiber L R t u).card + ∑ v ∈ lookupsIn L R t, (fiber L R t v).card :=
          add_le_add card_biUnion_le card_biUnion_le
      _ ≤ ∑ _u ∈ T.threadReads R t, e + ∑ _v ∈ lookupsIn L R t, eLk :=
          add_le_add (sum_le_sum fun u _ => hX u) (sum_le_sum hXl)
      _ = e * (T.threadReads R t).card + eLk * (lookupsIn L R t).card := by
          rw [sum_const, sum_const, smul_eq_mul, smul_eq_mul]
          ring
  have hread : (lookupsIn L R t).card = 0 ∨ 0 < (T.threadReads R t).card := by
    rcases Nat.eq_zero_or_pos (lookupsIn L R t).card with h | h
    · exact Or.inl h
    · right
      obtain ⟨v, hv⟩ := card_pos.1 h
      obtain ⟨hvR, hvl, hvt⟩ := mem_lookupsIn.1 hv
      obtain ⟨k, hkr, hkin, hkt⟩ := L.lookup_input v hvl
      have hkt' : T.thread k = some t := hkt.trans hvt
      refine card_pos.2 ⟨k, mem_filter.2 ⟨mem_sdiff.2 ⟨?_, L.thread_θ k t hkt'⟩, hkt'⟩⟩
      simp only [Circuit.into, mem_filter, mem_univ, true_and]
      refine ⟨fun hkR => ?_, v, hvR, hkr⟩
      have := (mem_filter.1 (hR.1 hkR)).2
      rw [hkin] at this
      cases this
  refine hcount.trans ?_
  rcases hread with h0 | hpos
  · rw [h0, mul_zero, add_zero]
    exact Nat.mul_le_mul_right _ (Nat.le_add_right _ _)
  · rw [add_mul]
    refine Nat.add_le_add_left ?_ _
    calc eLk * (lookupsIn L R t).card ≤ eLk * E := Nat.mul_le_mul_left _ hV
      _ = E * eLk * 1 := by ring
      _ ≤ E * eLk * (T.threadReads R t).card := Nat.mul_le_mul_left _ (by omega)

end Layered

/-- **Keyed circuits are threaded.** -/
theorem ThreadedOfLayered : Guarantees.ThreadedOfLayered := by
  intro c T d e eLk b L Γ S E hS
  exact threaded_of_layered L Γ S E hS

/-- **Training is slowed, for keyed circuits.** -/
theorem LayeredTraining : Guarantees.LayeredTraining := by
  intro Γ α b_w b_a κ c dcl T d e eLk b E M L hα hbw hba hE hd hB hS hW hκ
  exact ThreadedTraining Γ α b_w b_a κ c dcl T (d - E) (e + E * eLk) M hα hbw hba (Nat.sub_pos_of_lt hE) hd hB
    (threaded_of_layered L Γ (α * Γ) E hS) hW hκ

/-- **The reference circuit is threaded, at Kimi K2's shapes.** -/
theorem ReferenceThreaded : Guarantees.ReferenceThreaded := by
  intro c T hL Γ S hS
  unfold Assumptions.ReferenceLayered at hL
  obtain ⟨L⟩ := hL
  exact threaded_of_layered L Γ S 7 (hS.trans_lt (by norm_num))

/-- **Training is slowed, at Kimi K2's shapes.** -/
theorem ReferenceTraining : Guarantees.ReferenceTraining := by
  intro Γ α b_w b_a κ c dcl T M hL hα hbw hba hd hB hS hW hκ
  unfold Assumptions.ReferenceLayered at hL
  obtain ⟨L⟩ := hL
  exact LayeredTraining Γ α b_w b_a κ c dcl T _ _ _ _ 7 M L hα hbw hba (by norm_num) hd hB
    (hS.trans_lt (by norm_num)) hW hκ

end Nci.SecurityProofs
