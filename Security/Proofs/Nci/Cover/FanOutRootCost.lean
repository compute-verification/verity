import Proofs.Nci.Cover.FanOutWalk

/-!
# The fan-out proof: a held root costs `d`

`rootCost` (`FanOutPlan.RootCost`): a set of gates that holds a root of a token spends values of that token, and holds
lookups of it, `d` in all.

Induction on a bound `n` for a computed exit `g < n` of the held root. If the block holds a root of the same token with
a computed exit below `g`, the induction gives the claim for that root, and so for the token. Otherwise take each leg:
its link into the root ends at `g`, and the link's path lies in the root, which meets no input of the block, so its
first gate is computed (`entry`) and the leg's exit before it is in the block, below `g`. `walk` takes the leg from that
exit to a value the block reads, which it spends through that leg, or to a held stop with an exit below `g`, which is a
lookup since no root of the token is held there. The legs' key paths are disjoint (`legs_disjoint`), so distinct legs
end at distinct values, and the ends number at least `d`.
-/

namespace Nci.Cover.SecurityProofs

theorem rootCost : FanOutPlan.RootCost := by
  intro C J L Tok Val _ _ M d aOut c V _ K I
  suffices H : ∀ n : ℕ, ∀ r ∈ K.roots ∩ K.heldSet I, ∀ g ∈ C.exits (K.gates r) ∩ C.computed I, g.val < n →
      ∃ X Y : Finset Val, (∀ u ∈ X, K.Spent I u ∧ K.tok u = K.tok r) ∧
        (∀ u ∈ Y, u ∈ K.lookups ∩ K.heldSet I ∧ K.tok u = K.tok r) ∧ d ≤ X.card + Y.card by
    intro r hr
    obtain ⟨g, hg⟩ := (Finset.mem_filter.1 (Finset.mem_inter.1 hr).2).2.2
    exact H _ r hr g hg (Nat.lt_succ_self _)
  intro n
  induction n with
  | zero => exact fun _ _ _ _ hn => absurd hn (Nat.not_lt_zero _)
  | succ n ih =>
    intro r hr g hg hgn
    obtain ⟨hrR, hrH⟩ := Finset.mem_inter.1 hr
    have hrnr : ¬ (K.gates r ∩ C.inputs I).Nonempty := (Finset.mem_filter.1 hrH).2.1
    obtain ⟨hgx, hgc⟩ := Finset.mem_inter.1 hg
    by_cases hlow : ∃ u ∈ K.roots ∩ K.heldSet I, K.tok u = K.tok r ∧
        ∃ g' ∈ C.exits (K.gates u) ∩ C.computed I, g' < g
    · obtain ⟨u, hu, htok, g', hg', hlt⟩ := hlow
      obtain ⟨X, Y, hX, hY, hXY⟩ := ih u hu g' hg' (Nat.lt_of_lt_of_le hlt (Nat.le_of_lt_succ hgn))
      exact ⟨X, Y, fun x hx => htok ▸ hX x hx, fun y hy => htok ▸ hY y hy, hXY⟩
    have hleg : ∀ q, ∃ u, q ∈ K.legs r → K.tok u = K.tok r ∧ Follows K.key (K.stops ∪ K.readSet I) q u ∧
        (u ∈ K.readSet I ∨ u ∈ K.lookups ∩ K.heldSet I) := by
      intro q
      by_cases hq : q ∈ K.legs r
      swap
      · exact ⟨q, fun h => absurd h hq⟩
      obtain ⟨hqv, hqt⟩ := K.legs_mem r hrR q hq
      obtain ⟨h, hh, x, hx, hhx, π, hπ, hπr⟩ := K.legs_link r hrR g hgx q hq
      have hxc : x ∈ C.computed I := by
        by_contra hxc
        obtain ⟨g', hg', hg'I⟩ := entry C I π x g hπ (computed_subset C I hgc) hxc
        exact hrnr ⟨g', Finset.mem_inter.2 ⟨hπr g' hg', hg'I⟩⟩
      have hhI : h ∈ I := pred_subset_of_mem_computed hxc hhx
      have hlt : h.val < g.val := Nat.lt_of_lt_of_le (C.pred_lt x h hhx) (isPath_le hπ)
      obtain ⟨u, -, hut, hfu, hend⟩ := walk C M d aOut c V _ K I q hqv h hh hhI
      refine ⟨u, fun _ => ⟨hut.trans hqt, hfu, ?_⟩⟩
      rcases hend with hend | ⟨hus, g', hg', hle⟩
      · exact Or.inl hend
      · obtain ⟨hus, huh⟩ := Finset.mem_inter.1 hus
        rcases Finset.mem_union.1 hus with huR | huL
        · exact absurd ⟨u, Finset.mem_inter.2 ⟨huR, huh⟩, hut.trans hqt, g', hg', Nat.lt_of_le_of_lt hle hlt⟩ hlow
        · exact Or.inr (Finset.mem_inter.2 ⟨huL, huh⟩)
    choose f hf using hleg
    have hinj : Set.InjOn f (K.legs r) := by
      intro q hq q' hq' hqq
      by_contra hne
      have h1 : Follows K.key (K.roots ∪ K.lookups) q (f q') :=
        hqq ▸ Follows.mono Finset.subset_union_left (hf q hq).2.1
      have h2 : Follows K.key (K.roots ∪ K.lookups) q' (f q') :=
        Follows.mono Finset.subset_union_left (hf q' hq').2.1
      exact K.legs_disjoint r hrR q hq q' hq' hne (f q') h1 h2
    refine ⟨((K.legs r).image f).filter (· ∈ K.readSet I), ((K.legs r).image f).filter (· ∉ K.readSet I),
      ?_, ?_, ?_⟩
    · intro u hu
      obtain ⟨hu, hur⟩ := Finset.mem_filter.1 hu
      obtain ⟨q, hq, rfl⟩ := Finset.mem_image.1 hu
      exact ⟨⟨hur, q, Or.inr ⟨r, hr, hq⟩, (hf q hq).2.1⟩, (hf q hq).1⟩
    · intro u hu
      obtain ⟨hu, hur⟩ := Finset.mem_filter.1 hu
      obtain ⟨q, hq, rfl⟩ := Finset.mem_image.1 hu
      exact ⟨(hf q hq).2.2.resolve_left hur, (hf q hq).1⟩
    · rw [Finset.card_filter_add_card_filter_not, Finset.card_image_of_injOn hinj]
      exact K.legs_card r hrR

end Nci.Cover.SecurityProofs
