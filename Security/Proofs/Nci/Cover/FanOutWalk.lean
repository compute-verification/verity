import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Proofs.Nci.Cover.FanOutPlan
import Proofs.Nci.Cover.Paths

/-!
# The fan-out proof: walks along keys, and what they charge

`walk` follows keys back from a held exit of a value. A value the block doesn't read has that exit computed, and its
key's link into it lies in the value, which meets no input of the block, so the link's first gate, an exit of the key,
is in the block and precedes the exit (`entry`); induction on the exit finishes it. `readSet_card_le` bounds the values
a block reads by its inputs outside `θ`, `lookupCost` the lookups of one token it holds, through their disjoint columns,
and `charge` a token's whole products by its spent values when the block holds none of its roots: each is charged to
the value its activation's walk ends at, which `reach` bounds.
-/

namespace Nci.Cover

namespace Follows

variable {Val : Type*} {key : Val → Option Val}

/-- Following keys through a larger stop set follows them through a smaller one. -/
theorem mono {S S' : Finset Val} (hS : S' ⊆ S) {v u : Val} (h : Follows key S v u) : Follows key S' v u := by
  induction h with
  | refl v => exact Follows.refl v
  | step hv hk _ ih => exact Follows.step (fun h => hv (hS h)) hk ih

/-- Following keys from a stop goes nowhere. -/
theorem eq_of_mem {S : Finset Val} {v u : Val} (hv : v ∈ S) (h : Follows key S v u) : u = v := by
  cases h with
  | refl => rfl
  | step hv' _ _ => exact absurd hv hv'

/-- With no keys, following keys goes nowhere. -/
theorem eq_of_key_none {S : Finset Val} (hkey : ∀ v, key v = none) {v u : Val} (h : Follows key S v u) : u = v := by
  cases h with
  | refl => rfl
  | step _ hk _ => simp [hkey] at hk

end Follows

namespace SecurityProofs

theorem mem_of_mem_exits {C : Circuit} {X : Finset (Fin C.N)} {g : Fin C.N} (h : g ∈ C.exits X) : g ∈ X :=
  (Finset.mem_filter.1 h).1

/-- Indices increase along a path. -/
theorem isPath_le {C : Circuit} {π : List (Fin C.N)} {x y : Fin C.N} (h : C.IsPath π x y) : x ≤ y := by
  obtain ⟨hx, hy, hπ⟩ := h
  suffices H : ∀ x y, π.head? = some x → π.getLast? = some y → x ≤ y from H x y hx hy
  clear hx hy
  induction hπ with
  | nil => intro x y hx; simp at hx
  | singleton a =>
    intro x y hx hy
    simp only [List.head?_cons, List.getLast?_singleton, Option.some.injEq] at hx hy
    rw [← hx, ← hy]
  | @cons_cons a b l hr _ ih =>
    intro x y hx hy
    simp only [List.head?_cons, Option.some.injEq] at hx
    rw [List.getLast?_cons_cons] at hy
    rw [← hx]
    exact (C.pred_lt b a hr).le.trans (ih b y rfl hy)

section

variable {C : Circuit} {J L Tok Val : Type} [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V f : ℕ}

theorem mem_readSet {K : Keyed M Tok Val d aOut c V f} {I : Finset (Fin C.N)} {v : Val} :
    v ∈ K.readSet I ↔ v ∈ K.vals ∧ (K.gates v ∩ C.inputs I).Nonempty :=
  Finset.mem_filter

theorem mem_heldSet {K : Keyed M Tok Val d aOut c V f} {I : Finset (Fin C.N)} {v : Val} :
    v ∈ K.heldSet I ↔
      v ∈ K.vals ∧ ¬ (K.gates v ∩ C.inputs I).Nonempty ∧ (C.exits (K.gates v) ∩ C.computed I).Nonempty :=
  Finset.mem_filter

end

theorem walk : FanOutPlan.Walk := by
  intro C J L Tok Val _ M d aOut c V f K I
  suffices H : ∀ n : ℕ, ∀ v ∈ K.vals, ∀ h ∈ C.exits (K.gates v), h ∈ I → h.val < n →
      ∃ u ∈ K.vals, K.tok u = K.tok v ∧ Follows K.key (K.stops ∪ K.readSet I) v u ∧
        (u ∈ K.readSet I ∨ (u ∈ K.stops ∩ K.heldSet I ∧ ∃ g ∈ C.exits (K.gates u) ∩ C.computed I, g ≤ h)) from
    fun v hv h hh hhI => H _ v hv h hh hhI (Nat.lt_succ_self _)
  intro n
  induction n with
  | zero => exact fun _ _ _ _ _ hn => absurd hn (Nat.not_lt_zero _)
  | succ n ih =>
    intro v hv h hh hhI hn
    by_cases hr : v ∈ K.readSet I
    · exact ⟨v, hv, rfl, Follows.refl v, Or.inl hr⟩
    have hnr : ¬ (K.gates v ∩ C.inputs I).Nonempty := fun hne => hr (mem_readSet.2 ⟨hv, hne⟩)
    have hhc : h ∈ C.computed I :=
      mem_computed.2 ⟨hhI, fun hin => hnr ⟨h, Finset.mem_inter.2 ⟨mem_of_mem_exits hh, hin⟩⟩⟩
    by_cases hs : v ∈ K.stops
    · have hheld : v ∈ K.heldSet I := mem_heldSet.2 ⟨hv, hnr, h, Finset.mem_inter.2 ⟨hh, hhc⟩⟩
      exact ⟨v, hv, rfl, Follows.refl v, Or.inr ⟨Finset.mem_inter.2 ⟨hs, hheld⟩, h,
        Finset.mem_inter.2 ⟨hh, hhc⟩, le_rfl⟩⟩
    obtain ⟨κ, hκ, hkey, htok, hlink⟩ := K.keyed v hv hs
    obtain ⟨h', hh', x, hx, hh'x, π, hπ, hπv⟩ := hlink h hh
    have hxc : x ∈ C.computed I := by
      by_contra hxc
      obtain ⟨g, hg, hgI⟩ := entry C I π x h hπ hhI hxc
      exact hnr ⟨g, Finset.mem_inter.2 ⟨hπv g hg, hgI⟩⟩
    have hh'I : h' ∈ I := pred_subset_of_mem_computed hxc hh'x
    have hlt : h' < h := lt_of_lt_of_le (C.pred_lt x h' hh'x) (isPath_le hπ)
    obtain ⟨u, hu, hut, hfu, hend⟩ :=
      ih κ hκ h' hh' hh'I (Nat.lt_of_lt_of_le hlt (Nat.le_of_lt_succ hn))
    refine ⟨u, hu, hut.trans htok, Follows.step (fun hm => ?_) hkey hfu, ?_⟩
    · rcases Finset.mem_union.1 hm with hm | hm
      · exact hs hm
      · exact hr hm
    · rcases hend with hend | ⟨hus, g, hg, hgle⟩
      · exact Or.inl hend
      · exact Or.inr ⟨hus, g, hg, hgle.trans hlt.le⟩

theorem readSet_card_le : FanOutPlan.ReadSetCardLe := by
  intro C J L Tok Val _ M d aOut c V f K I
  have hne : ∀ v ∈ K.readSet I, (K.gates v ∩ C.inputs I).Nonempty := fun v hv => (mem_readSet.1 hv).2
  have hdisj : ((K.readSet I : Finset Val) : Set Val).PairwiseDisjoint fun v => K.gates v ∩ C.inputs I :=
    (K.gates_disjoint.subset (Finset.coe_subset.2 (Finset.filter_subset _ _))).mono
      fun v => Finset.inter_subset_left
  refine (Finset.card_le_card_biUnion hdisj hne).trans (Finset.card_le_card ?_)
  intro x hx
  obtain ⟨v, hv, hx⟩ := Finset.mem_biUnion.1 hx
  obtain ⟨hxg, hxi⟩ := Finset.mem_inter.1 hx
  exact Finset.mem_sdiff.2 ⟨hxi, Finset.disjoint_left.1 (K.gates_θ v (mem_readSet.1 hv).1) hxg⟩

theorem lookupCost : FanOutPlan.LookupCost := by
  intro C J L Tok Val _ _ M d aOut c V f K s k E bcol hCI hk I hI Y hY htok
  have hcols := hCI I hI.1 Y hY htok
  have hcap : ((C.inputs I \ C.θ).card : ℝ) ≤ k := by exact_mod_cast hI.2
  have hsum : (Y.biUnion K.col).card = ∑ v ∈ Y, (K.col v).card := by
    refine Finset.card_biUnion fun v hv v' hv' hne => ?_
    exact K.col_disjoint v (Finset.mem_inter.1 (hY hv)).1 v' (Finset.mem_inter.1 (hY hv')).1 hne
      (htok v hv v' hv')
  have hV : Y.card * V ≤ (Y.biUnion K.col).card := by
    rw [hsum]
    calc Y.card * V = ∑ _v ∈ Y, V := by rw [Finset.sum_const, smul_eq_mul]
      _ ≤ ∑ v ∈ Y, (K.col v).card := Finset.sum_le_sum fun v hv => K.col_card v (Finset.mem_inter.1 (hY hv)).1
  by_contra hlt
  rw [not_le] at hlt
  have hE : (E : ℝ) + 1 ≤ Y.card := by exact_mod_cast hlt
  have hV' : (Y.card : ℝ) * V ≤ (Y.biUnion K.col).card := by exact_mod_cast hV
  have hk0 : (0 : ℝ) ≤ k := Nat.cast_nonneg _
  have hV0 : (0 : ℝ) ≤ V := Nat.cast_nonneg _
  rcases le_or_gt bcol 0 with hb | hb
  · have : bcol * V * (E + 1) ≤ 0 :=
      mul_nonpos_of_nonpos_of_nonneg (mul_nonpos_of_nonpos_of_nonneg hb hV0) (by positivity)
    linarith
  · have h1 : bcol * V * (E + 1) ≤ bcol * V * Y.card := mul_le_mul_of_nonneg_left hE (mul_nonneg hb.le hV0)
    have h2 : bcol * (Y.card * V) ≤ bcol * (Y.biUnion K.col).card := mul_le_mul_of_nonneg_left hV' hb.le
    nlinarith

theorem charge : FanOutPlan.Charge := by
  intro C J L Tok Val _ _ M d aOut c V f K I i hnoroot B hB htokB
  classical
  have hpick : ∀ p ∈ B, ∃ u, (K.Spent I u ∧ K.tok u = i) ∧ Follows K.key (K.roots ∪ K.lookups) (K.act p) u := by
    intro p hp
    have hpw := hB hp
    obtain ⟨hpP, hpI⟩ := Finset.mem_filter.1 hpw
    obtain ⟨h, hh, x, hx, hhx, -⟩ := K.act_link p hpP
    have hhI : h ∈ I := pred_subset_of_mem_computed (hpI hx) hhx
    obtain ⟨u, -, hut, hfu, hend⟩ := walk C M d aOut c V f K I (K.act p) (K.act_mem p hpP) h hh hhI
    have hfu' : Follows K.key (K.roots ∪ K.lookups) (K.act p) u := hfu.mono Finset.subset_union_left
    rcases hend with hur | ⟨hus, -⟩
    · exact ⟨u, ⟨⟨hur, K.act p, Or.inl ⟨p, hpw, rfl⟩, hfu⟩, hut.trans (htokB p hp)⟩, hfu'⟩
    · exfalso
      obtain ⟨hus, huh⟩ := Finset.mem_inter.1 hus
      rcases Finset.mem_union.1 hus with hur | hul
      · exact hnoroot u (Finset.mem_inter.2 ⟨hur, huh⟩) (hut.trans (htokB p hp))
      · exact K.reach_lookups p hpP u hul hfu'
  rcases B.eq_empty_or_nonempty with rfl | ⟨p₀, -⟩
  · exact ⟨∅, fun u hu => absurd hu (Finset.notMem_empty u), by simp⟩
  have : Nonempty Val := ⟨K.act p₀⟩
  choose! φ hφ using hpick
  refine ⟨B.image φ, fun u hu => ?_, ?_⟩
  · obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hu
    exact (hφ p hp).1
  · refine Finset.card_le_mul_card_image B aOut fun u _ => K.reach u _ (fun p hp => ?_) fun p hp => ?_
    · exact (Finset.mem_filter.1 (hB (Finset.mem_filter.1 hp).1)).1
    · obtain ⟨hpB, rfl⟩ := Finset.mem_filter.1 hp
      exact (hφ p hpB).2

end SecurityProofs

end Nci.Cover
