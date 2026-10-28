import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Proofs.Nci.Cover.Area
import Proofs.Nci.Cover.FanOutWalk
import Definitions.Nci.Guarantees.CoverPartialSum

/-!
# The partial-sum bounds: partial sums are read

`label_acc` labels the gates of one accumulation by blocks of a cover that compute them, from the top down: the top
takes any block computing it, and a gate below it takes the label of a gate it feeds on its path to the top when that
block computes it too, and otherwise any block computing it. A fresh label below the top is a gate of the accumulation
that the reader's block is handed, and distinct gates give distinct pairs of a block and an input of it, so the labels
number at most one more than those pairs. `PartialSumsRead` assigns each counted product to the label of its output gate
and adds the pairs over the entries, whose accumulations are disjoint.
-/

namespace Nci.Cover.SecurityProofs

/-- A path from `x` to another gate leaves `x` through a gate of the path that reads `x`. -/
theorem exists_reader_of_isPath {C : Circuit} {π : List (Fin C.N)} {x y : Fin C.N} (hπ : C.IsPath π x y)
    (hxy : x ≠ y) : ∃ h ∈ π, x ∈ C.pred h := by
  obtain ⟨hx, hy, hc⟩ := hπ
  match π, hx, hy, hc with
  | [], hx, _, _ => simp at hx
  | [a], hx, hy, _ =>
    simp only [List.head?_cons, List.getLast?_singleton, Option.some.injEq] at hx hy
    exact absurd (hx.symm.trans hy) hxy
  | a :: b :: l, hx, _, hc =>
    simp only [List.head?_cons, Option.some.injEq] at hx
    subst hx
    exact ⟨b, List.mem_cons_of_mem _ List.mem_cons_self, (List.isChain_cons_cons.1 hc).1⟩

/-- **Labelling an accumulation**: if every gate of `A` has predecessors and reaches `T` inside `A`, then a cover labels
each gate of `A` with a block computing it, using at most `1 + Σ_I #(inputs I ∩ A)` labels. -/
theorem label_acc {C : Circuit} {𝓘 : Finset (Finset (Fin C.N))} (h𝓘 : C.Covers 𝓘) {A : Finset (Fin C.N)}
    {T : Fin C.N} (hA : ∀ g ∈ A, (C.pred g).Nonempty) (hT : ∀ g ∈ A, C.ReachesIn A g T) :
    ∃ lab : Fin C.N → Finset (Fin C.N), (∀ g ∈ A, lab g ∈ 𝓘 ∧ g ∈ C.computed (lab g)) ∧
      (A.image lab).card ≤ 1 + ∑ I ∈ 𝓘, (C.inputs I ∩ A).card := by
  classical
  set S := (𝓘 ×ˢ A).filter fun x => x.2 ∈ C.inputs x.1 with hSdef
  have hS : S.card = ∑ I ∈ 𝓘, (C.inputs I ∩ A).card := by
    rw [hSdef, Finset.card_filter, Finset.sum_product]
    refine Finset.sum_congr rfl fun I _ => ?_
    rw [← Finset.card_filter]
    congr 1
    ext g
    simp only [Finset.mem_filter, Finset.mem_inter]
    exact and_comm
  have hle : ∀ g ∈ A, g ≤ T := fun g hg => by
    obtain ⟨π, hπ, -⟩ := hT g hg
    exact isPath_le hπ
  suffices H : ∀ m n, n + m = C.N → ∃ lab : Fin C.N → Finset (Fin C.N), ∃ F ⊆ S,
      (∀ x ∈ F, n ≤ x.2.val) ∧ (∀ g ∈ A, n ≤ g.val → lab g ∈ 𝓘 ∧ g ∈ C.computed (lab g)) ∧
      ((A.filter fun g => n ≤ g.val).image lab).card ≤ 1 + F.card by
    obtain ⟨lab, F, hFS, -, hlab, hcard⟩ := H C.N 0 (zero_add _)
    refine ⟨lab, fun g hg => hlab g hg (Nat.zero_le _), ?_⟩
    rw [Finset.filter_true_of_mem fun _ _ => Nat.zero_le _] at hcard
    rw [← hS]
    exact hcard.trans (Nat.add_le_add_left (Finset.card_le_card hFS) 1)
  intro m
  induction m with
  | zero =>
    intro n hn
    refine ⟨fun _ => ∅, ∅, Finset.empty_subset _, fun x hx => absurd hx (Finset.notMem_empty x),
      fun g _ hg => absurd hg (by omega), ?_⟩
    rw [Finset.filter_false_of_mem fun g _ => by omega]
    simp
  | succ m ih =>
    intro n hn
    obtain ⟨lab, F, hFS, hFn, hlab, hcard⟩ := ih (n + 1) (by omega)
    have hnN : n < C.N := by omega
    set g : Fin C.N := ⟨n, hnN⟩ with hgdef
    have hne : ∀ g' : Fin C.N, n ≤ g'.val → g' ≠ g → n + 1 ≤ g'.val := fun g' h1 h2 => by
      have : g'.val ≠ n := fun h => h2 (Fin.ext h)
      omega
    by_cases hgA : g ∉ A
    · have hfilt : (A.filter fun g' => n ≤ g'.val) = A.filter fun g' => n + 1 ≤ g'.val := by
        refine Finset.filter_congr fun g' hg' => ⟨fun h => hne g' h fun he => hgA (he ▸ hg'), fun h => by omega⟩
      refine ⟨lab, F, hFS, fun x hx => (hFn x hx).trans' (Nat.le_succ n), fun g' hg' h => ?_, hfilt ▸ hcard⟩
      exact hlab g' hg' (hne g' h fun he => hgA (he ▸ hg'))
    rw [not_not] at hgA
    have hins : (A.filter fun g' => n ≤ g'.val) = insert g (A.filter fun g' => n + 1 ≤ g'.val) := by
      ext g'
      simp only [Finset.mem_filter, Finset.mem_insert]
      constructor
      · rintro ⟨hg'A, hg'n⟩
        by_cases he : g' = g
        · exact Or.inl he
        · exact Or.inr ⟨hg'A, hne g' hg'n he⟩
      · rintro (rfl | ⟨hg'A, hg'n⟩)
        · exact ⟨hgA, le_rfl⟩
        · exact ⟨hg'A, by omega⟩
    have hgnot : g ∉ A.filter fun g' => n + 1 ≤ g'.val := fun h => by
      have := (Finset.mem_filter.1 h).2
      simp [hgdef] at this
    -- Relabelling `g` with `J` adds at most `J` to the labels.
    have himage : ∀ J, ((A.filter fun g' => n ≤ g'.val).image (Function.update lab g J)) =
        insert J ((A.filter fun g' => n + 1 ≤ g'.val).image lab) := by
      intro J
      rw [hins, Finset.image_insert, Function.update_self]
      congr 1
      refine Finset.image_congr fun x hx => ?_
      exact Function.update_of_ne (fun he : x = g => hgnot (he ▸ hx)) J lab
    have hlab' : ∀ J ∈ 𝓘, g ∈ C.computed J → ∀ g' ∈ A, n ≤ g'.val →
        Function.update lab g J g' ∈ 𝓘 ∧ g' ∈ C.computed (Function.update lab g J g') := by
      intro J hJ hgJ g' hg' h
      by_cases he : g' = g
      · subst he
        rw [Function.update_self]
        exact ⟨hJ, hgJ⟩
      · rw [Function.update_of_ne he]
        exact hlab g' hg' (hne g' h he)
    obtain ⟨J₀, hJ₀, hgJ₀⟩ := h𝓘 g (hA g hgA)
    by_cases hgT : g = T
    · have hempty : (A.filter fun g' => n + 1 ≤ g'.val) = ∅ := Finset.filter_false_of_mem fun g' hg' h => by
        have := hle g' hg'
        rw [← hgT, Fin.le_def] at this
        simp only [hgdef] at this
        omega
      refine ⟨Function.update lab g J₀, F, hFS, fun x hx => (hFn x hx).trans' (Nat.le_succ n),
        hlab' J₀ hJ₀ hgJ₀, ?_⟩
      rw [himage, hempty, Finset.image_empty, insert_empty_eq, Finset.card_singleton]
      exact Nat.le_add_right 1 _
    obtain ⟨π, hπ, hπA⟩ := hT g hgA
    obtain ⟨h, hhπ, hgh⟩ := exists_reader_of_isPath hπ hgT
    have hhA := hπA h hhπ
    have hgh' : n + 1 ≤ h.val := C.pred_lt h g hgh
    have hhm : h ∈ A.filter fun g' => n + 1 ≤ g'.val := Finset.mem_filter.2 ⟨hhA, hgh'⟩
    obtain ⟨hJh, hhc⟩ := hlab h hhA hgh'
    have hgI : g ∈ lab h := pred_subset_of_mem_computed hhc hgh
    by_cases hgc : g ∈ C.computed (lab h)
    · refine ⟨Function.update lab g (lab h), F, hFS, fun x hx => (hFn x hx).trans' (Nat.le_succ n),
        hlab' (lab h) hJh hgc, ?_⟩
      rw [himage, Finset.insert_eq_of_mem (Finset.mem_image_of_mem lab hhm)]
      exact hcard
    · have hgin : g ∈ C.inputs (lab h) := by
        by_contra hgin
        exact hgc (mem_computed.2 ⟨hgI, hgin⟩)
      have hnew : (lab h, g) ∉ F := fun hm => by
        have := hFn _ hm
        simp [hgdef] at this
      refine ⟨Function.update lab g J₀, insert (lab h, g) F, ?_, ?_, hlab' J₀ hJ₀ hgJ₀, ?_⟩
      · refine Finset.insert_subset ?_ hFS
        exact Finset.mem_filter.2 ⟨Finset.mem_product.2 ⟨hJh, hgA⟩, hgin⟩
      · intro x hx
        rcases Finset.mem_insert.1 hx with rfl | hx
        · exact le_rfl
        · exact (hFn x hx).trans' (Nat.le_succ n)
      · rw [himage, Finset.card_insert_of_notMem hnew]
        exact (Finset.card_insert_le _ _).trans (by omega)

theorem PartialSumsRead : Guarantees.PartialSumsRead := by
  intro C J L Tok Val Mat Row Col _ _ _ _ _ M d aOut c V w K Q 𝓘 h𝓘
  classical
  have hlab : ∀ E, ∃ lab : Fin C.N → Finset (Fin C.N), E ∈ M.P.image Q.entry →
      (∀ g ∈ Q.acc E, lab g ∈ 𝓘 ∧ g ∈ C.computed (lab g)) ∧
      ((Q.acc E).image lab).card ≤ 1 + ∑ I ∈ 𝓘, (C.inputs I ∩ Q.acc E).card := by
    intro E
    by_cases hE : E ∈ M.P.image Q.entry
    · obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hE
      obtain ⟨lab, h1, h2⟩ := label_acc h𝓘 (Q.acc_computed p hp) (Q.acc_top p hp)
      exact ⟨lab, fun _ => ⟨h1, h2⟩⟩
    · exact ⟨fun _ => ∅, fun h => absurd h hE⟩
  choose lab hlab using hlab
  have hacc : ∀ p ∈ M.P, M.o p ∈ Q.acc (Q.entry p) := fun p hp => Q.acc_prod p hp (o_mem M (M.P_subset hp))
  refine ⟨fun p => lab (Q.entry p) (M.o p), fun p hp => (hlab _ (Finset.mem_image_of_mem _ hp)).1 _ (hacc p hp), ?_⟩
  have hdisj : ∀ E ∈ M.P.image Q.entry, ∀ E' ∈ M.P.image Q.entry, E ≠ E' → Disjoint (Q.acc E) (Q.acc E') := by
    intro E hE E' hE' hne
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hE
    obtain ⟨p', hp', rfl⟩ := Finset.mem_image.1 hE'
    exact Q.acc_disjoint p hp p' hp' hne
  calc ∑ E ∈ M.P.image Q.entry, (((M.P.filter fun p => Q.entry p = E).image fun p => lab (Q.entry p) (M.o p)).card - 1)
      ≤ ∑ E ∈ M.P.image Q.entry, ∑ I ∈ 𝓘, (C.inputs I ∩ Q.acc E).card := by
        refine Finset.sum_le_sum fun E hE => ?_
        have hsub : ((M.P.filter fun p => Q.entry p = E).image fun p => lab (Q.entry p) (M.o p)) ⊆
            (Q.acc E).image (lab E) := by
          intro I hI
          obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hI
          obtain ⟨hpP, hpE⟩ := Finset.mem_filter.1 hp
          rw [hpE]
          exact Finset.mem_image_of_mem _ (hpE ▸ hacc p hpP)
        have := (Finset.card_le_card hsub).trans (hlab E hE).2
        omega
    _ = ∑ I ∈ 𝓘, ∑ E ∈ M.P.image Q.entry, (C.inputs I ∩ Q.acc E).card := Finset.sum_comm
    _ = ∑ I ∈ 𝓘, (C.inputs I ∩ Q.accGates).card := by
        refine Finset.sum_congr rfl fun I _ => ?_
        rw [Cube.accGates, Finset.inter_biUnion, Finset.card_biUnion fun E hE E' hE' hne =>
          Finset.disjoint_of_subset_left Finset.inter_subset_right
            (Finset.disjoint_of_subset_right Finset.inter_subset_right (hdisj E hE E' hE' hne))]

end Nci.Cover.SecurityProofs
