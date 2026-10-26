import Mathlib.Combinatorics.Enumerative.DoubleCounting
import Definitions.Nci.Guarantees.CoverPartialSum
import Proofs.Nci.Cover.FanOutRootCost
import Proofs.Nci.Cover.FanOutValue

/-!
# The partial-sum bounds: activation reads from the keyed structure

`KeyedActivationReads` is `ActivationReads` at `f = 1` and `a = min(d − E, n)`, from the keyed structure. Fix a block
`I` handed fewer than `a` gates of a token `t`.

* `not_held_root`: `I` holds no root of `t`. A held root spends values of `t` and holds lookups of it, `d` in all
  (`rootCost`), at most `E` of them lookups (`lookupCost`), and the spent values are read values of `t`, each with its
  own gate among `I`'s inputs (`card_le_tokReads`).
* `act_card_le_tokReads`: so the walk back from each activation of `t` that `I` uses ends at a value of `t` that `I`
  reads (`walk`; a held stop would be a root of `t` or a lookup, which `reach_lookups` excludes), and distinct
  activations end at distinct values (`Keyed.keyDisjoint`).
* An entry of `t` all of whose products `I` computes whole has `n ≥ a` products with distinct activations (`Wide`,
  `cell_inj`, `stack`), more than `I` reads.
-/

namespace Nci.Cover.SecurityProofs

section

variable {C : Circuit} {J L Tok Val : Type} [DecidableEq Val] {M : MulBlocks C J L} {d aOut c V w : ℕ}
  {K : Keyed M Tok Val d aOut c V w}

/-- A block's read values of one token are at most its reads of the token. -/
theorem card_le_tokReads [DecidableEq Tok] {I : Finset (Fin C.N)} {t : Tok} {X : Finset Val}
    (hX : X ⊆ K.readSet I) (ht : ∀ u ∈ X, K.tok u = t) : X.card ≤ K.tokReads I t := by
  have hne : ∀ u ∈ X, (K.gates u ∩ C.inputs I).Nonempty := fun u hu => (mem_readSet.1 (hX hu)).2
  have hdisj : (X : Set Val).PairwiseDisjoint fun u => K.gates u ∩ C.inputs I :=
    (K.gates_disjoint.subset fun u hu => (mem_readSet.1 (hX hu)).1).mono fun u => Finset.inter_subset_left
  refine (Finset.card_le_card_biUnion hdisj hne).trans (Finset.card_le_card ?_)
  intro g hg
  obtain ⟨u, hu, hg⟩ := Finset.mem_biUnion.1 hg
  obtain ⟨hgu, hgi⟩ := Finset.mem_inter.1 hg
  exact Finset.mem_inter.2
    ⟨hgi, Finset.mem_biUnion.2 ⟨u, Finset.mem_filter.2 ⟨(mem_readSet.1 (hX hu)).1, ht u hu⟩, hgu⟩⟩

/-- A block handed fewer than `d − E` gates of a token holds none of its roots. -/
theorem not_held_root [DecidableEq J] [DecidableEq L] [DecidableEq Tok] {s k E : ℕ} {bcol : ℝ}
    (hCI : Assumptions.ColumnsIncompressible K s bcol) (hk : (k : ℝ) < bcol * V * (E + 1))
    {I : Finset (Fin C.N)} (hI : C.InferenceBlock s k I) {t : Tok} (ht : K.tokReads I t < d - E) :
    ∀ r ∈ K.roots ∩ K.heldSet I, K.tok r ≠ t := by
  intro r hr hrt
  obtain ⟨X, Y, hX, hY, hd⟩ := rootCost C M d aOut c V w K I r hr
  have hYE : Y.card ≤ E := lookupCost C M d aOut c V w K s k E bcol hCI hk I hI Y (fun u hu => (hY u hu).1)
    fun v hv v' hv' => (hY v hv).2.trans (hY v' hv').2.symm
  have hXt : X.card ≤ K.tokReads I t :=
    card_le_tokReads (fun u hu => (hX u hu).1.1) fun u hu => (hX u hu).2.trans hrt
  omega

/-- **One read per activation**: a block that holds no root of `t` reads a distinct value of `t` for each distinct
activation of the products of `t` it computes whole. -/
theorem act_card_le_tokReads [DecidableEq J] [DecidableEq Tok] {I : Finset (Fin C.N)} {t : Tok}
    (hnoroot : ∀ r ∈ K.roots ∩ K.heldSet I, K.tok r ≠ t) :
    (((M.whole I).filter fun p => K.tok (K.act p) = t).image K.act).card ≤ K.tokReads I t := by
  refine (Finset.card_le_card_of_forall_subsingleton (fun v u => Follows K.key (K.roots ∪ K.lookups) v u)
    (t := (K.readSet I).filter fun u => K.tok u = t) ?_ ?_).trans
    (card_le_tokReads (Finset.filter_subset _ _) fun u hu => (Finset.mem_filter.1 hu).2)
  · intro v hv
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hv
    obtain ⟨hpw, hpt⟩ := Finset.mem_filter.1 hp
    obtain ⟨hpP, hpI⟩ := Finset.mem_filter.1 hpw
    obtain ⟨h, hh, x, hx, hhx, -⟩ := K.act_link p hpP
    have hhI : h ∈ I := pred_subset_of_mem_computed (hpI hx) hhx
    obtain ⟨u, -, hut, hfu, hend⟩ := walk C M d aOut c V w K I (K.act p) (K.act_mem p hpP) h hh hhI
    have hfu' : Follows K.key (K.roots ∪ K.lookups) (K.act p) u := hfu.mono Finset.subset_union_left
    rcases hend with hur | ⟨hus, -⟩
    · exact ⟨u, Finset.mem_filter.2 ⟨hur, hut.trans hpt⟩, hfu'⟩
    · exfalso
      obtain ⟨hus, huh⟩ := Finset.mem_inter.1 hus
      rcases Finset.mem_union.1 hus with hur | hul
      · exact hnoroot u (Finset.mem_inter.2 ⟨hur, huh⟩) (hut.trans hpt)
      · exact K.reach_lookups p hpP u hul hfu'
  · intro u _ v hv v' hv'
    obtain ⟨hvS, hvu⟩ := hv
    obtain ⟨hv'S, hv'u⟩ := hv'
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hvS
    obtain ⟨p', hp', rfl⟩ := Finset.mem_image.1 hv'S
    obtain ⟨hpw, hpt⟩ := Finset.mem_filter.1 hp
    obtain ⟨hp'w, hp't⟩ := Finset.mem_filter.1 hp'
    by_contra hne
    exact K.keyDisjoint p (Finset.mem_filter.1 hpw).1 p' (Finset.mem_filter.1 hp'w).1 hne (hpt.trans hp't.symm) u hvu
      hv'u

end

theorem KeyedActivationReads : Guarantees.KeyedActivationReads := by
  intro C J L Tok Val Mat Row Col _ _ _ _ _ _ M d aOut c V w K Q s k E n bcol hCI hk hW I hI t ht
  rw [one_mul] at ht
  have hact := act_card_le_tokReads (not_held_root hCI hk hI (lt_of_lt_of_le ht (min_le_left _ _)))
  refine ⟨fun p hp hpt => ?_, by rw [one_mul]; exact hact⟩
  by_contra hall
  push Not at hall
  have hF : (M.P.filter fun p' => Q.entry p' = Q.entry p).image K.act ⊆
      ((M.whole I).filter fun p => K.tok (K.act p) = t).image K.act := by
    refine Finset.image_subset_image fun p' hp' => ?_
    obtain ⟨hp'P, hp'E⟩ := Finset.mem_filter.1 hp'
    refine Finset.mem_filter.2 ⟨hall p' hp'P hp'E, ?_⟩
    exact (congrArg (fun e : Mat × Row × Tok => e.2.2) hp'E : K.tok (K.act p') = K.tok (K.act p)).trans hpt
  have hinj : ((M.P.filter fun p' => Q.entry p' = Q.entry p).image K.act).card =
      (M.P.filter fun p' => Q.entry p' = Q.entry p).card := by
    refine Finset.card_image_of_injOn fun p1 hp1 p2 hp2 h => ?_
    obtain ⟨hp1P, hp1E⟩ := Finset.mem_filter.1 (Finset.mem_coe.1 hp1)
    obtain ⟨hp2P, hp2E⟩ := Finset.mem_filter.1 (Finset.mem_coe.1 hp2)
    have hE : Q.entry p1 = Q.entry p2 := hp1E.trans hp2E.symm
    simp only [Cube.entry, Prod.mk.injEq] at hE
    exact Q.cell_inj p1 hp1P p2 hp2P hE.1 hE.2.1 (Q.stack p1 hp1P p2 hp2P h).2 hE.2.2
  have hn := hW p hp
  have hsub := Finset.card_le_card hF
  have hmin := min_le_right (d - E) n
  omega

end Nci.Cover.SecurityProofs
