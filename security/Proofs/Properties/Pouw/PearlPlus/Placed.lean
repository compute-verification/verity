module

public import Properties.Pouw.PearlPlusPlaced
public import Proofs.Pouw.PearlPlus.Draw
public import Proofs.Properties.Verity.Tied

/-!
# Pearl+'s cells in the tied circuit (`CreditedArePlaced`, `GivenUpByBlock`)

* **A credited cell is a real cell** (`real_of_credited`): its row is real by the band rule (`UnitRecord.credited`),
  and its column lies in a tile's block of columns, which `Scheme.tiles` cuts below the call's width
  (`lt_of_mem_tiles`).
* **A real cell of a run held to W is at W's entries** (`cellAt_of_real`): `TieMeans` gives, for its call, the domains
  and W's registrations of the A rows and the y, and, the weights being tied, of the weight's B rows.
* So where the tied circuit's cells are the run's real cells at W's entries, a credited cell is one it ties, placed at
  its home (`Tying.place`); `Verity.ValueTie.Tying.fromBelow` puts its 32 gates in that block (`creditedArePlaced`).
  Every given-up cell is one it ties too, and a home that reads the A row alone counts them as the records do
  (`givenUpByBlock`).
-/

@[expose] public section

open Pouw.PearlPlus.Draw (tiles_getElem? nblocks block)

namespace Properties.Pouw.PearlPlus

open PouwCheck.PearlPlus

theorem lt_of_mem_block {len w b j : ℕ} (h : j ∈ block len w b) : j < len := by
  unfold block at h
  rw [List.mem_range'_1] at h
  generalize b * w = t at h
  omega

theorem lt_of_mem_tiles {s : Scheme} {m n : ℕ} {rows cols : List ℕ} {j : ℕ}
    (h : (rows, cols) ∈ s.tiles m n) (hj : j ∈ cols) : j < n := by
  obtain ⟨x, hx⟩ := List.mem_iff_getElem?.1 h
  rw [tiles_getElem?] at hx
  by_cases hlt : x < nblocks m s.tileM * nblocks n s.tileN
  · simp only [hlt, ↓reduceIte, Option.some.injEq, Prod.mk.injEq] at hx
    obtain ⟨-, rfl⟩ := hx
    exact lt_of_mem_block hj
  · simp [hlt] at hx

theorem real_of_credited {passes : List Pass} {c : Cell} (hc : c ∈ creditedCells passes) :
    c ∈ realCells passes := by
  rw [creditedCells, List.mem_toFinset] at hc
  obtain ⟨⟨p, pi⟩, hp, hc1⟩ := List.mem_flatMap.1 hc
  obtain ⟨⟨r, pos⟩, hr, hc2⟩ := List.mem_flatMap.1 hc1
  obtain ⟨⟨rows, cols⟩, hrc, hc3⟩ := List.mem_flatMap.1 hc2
  obtain ⟨i, hi, hc4⟩ := List.mem_flatMap.1 hc3
  obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hc4
  have hi' : i < r.realRows := by
    have h := (List.mem_filter.1 (show i ∈ List.filter _ rows from hi)).2
    simp only [Bool.and_eq_true] at h
    simpa using h.1.1
  have hj' : j < r.matmul.n := lt_of_mem_tiles hrc (List.mem_filter.1 (show j ∈ List.filter _ cols from hj)).1
  rw [realCells, List.mem_toFinset]
  exact List.mem_flatMap.2 ⟨(p, pi), hp, List.mem_flatMap.2 ⟨(r, pos), hr,
    List.mem_flatMap.2 ⟨i, List.mem_range.2 hi', List.mem_map.2 ⟨j, List.mem_range.2 hj', rfl⟩⟩⟩⟩

theorem cellAt_of_real (htm : TieMeans) {R : Run} (hR : R.HeldToW) {c : Cell} (hc : c ∈ realCells R.passes) :
    ∃ (p : Pass) (r : UnitRecord) (da db dy : PouwCommit.RowsDomain) (rootA rootB yRoot : ByteArray),
      Verity.ValueTie.CellAt R c p r da db dy rootA rootB yRoot := by
  obtain ⟨hload, tied, htie, hw⟩ := hR
  obtain ⟨-, -, -, -, hcall, -⟩ := htm R.coins R.manifests R.passes R.w R.rounds R.dropped tied hload htie
  rw [realCells, List.mem_toFinset] at hc
  obtain ⟨⟨p, pi⟩, hp, hc1⟩ := List.mem_flatMap.1 hc
  obtain ⟨⟨r, pos⟩, hr, hc2⟩ := List.mem_flatMap.1 hc1
  obtain ⟨i, hi, hc3⟩ := List.mem_flatMap.1 hc2
  obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hc3
  simp only [List.mem_zipIdx_iff_getElem?] at hp hr
  obtain ⟨da, dy, yRoot, hda, hdy, -, hA, hY, hB⟩ := hcall pi p pos r hp hr
  obtain ⟨db, hdb, hB⟩ := hB hw
  refine ⟨p, r, da, db, dy, p.rootsA[pos]!, (p.weightRoots.lookup r.matmul.weight).getD ByteArray.empty, yRoot, ?_⟩
  unfold Verity.ValueTie.CellAt
  exact ⟨hp, hr, List.mem_range.1 hi, List.mem_range.1 hj, hda, hdb, hdy, hA, hB, hY⟩

theorem creditedArePlaced : CreditedArePlaced := by
  intro htm R C θ L T m Q home hR hT c hc
  have hcell : c ∈ T.cells := (hT c).2 (cellAt_of_real htm hR (real_of_credited hc))
  have hp : T.place home c = some (home c, (L.yAt c).map ⟨T.emb, T.emb_inj⟩) := by
    unfold Verity.ValueTie.Tying.place
    rw [ite_eq_left hcell]
  obtain ⟨hb, -, -⟩ := T.fromBelow Q home ∅
  exact ⟨_, hp, hb c (home c) _ hp⟩

theorem givenUpByBlock : GivenUpByBlock := by
  intro htm R C θ L T m home hr hR hT hh u
  have hP : ∀ S : Finset Cell, (S.filter fun c => home c = u) = S.filter fun c => hr (c.1, c.2.1, c.2.2.1) = u :=
    fun S => Finset.filter_congr fun c _ => by rw [hh c]
  refine ⟨?_, fun U => ?_⟩
  · have hsub : givenUpCells R.passes ⊆ T.cells := fun c hc =>
      (hT c).2 (cellAt_of_real htm hR (Finset.mem_sdiff.1 hc).1)
    rw [Finset.inter_eq_right.2 hsub, hP, givenUpAt]
  · calc (U.filter fun c => home c = u).card
        ≤ (((givenUpCells R.passes).filter fun c => home c = u) ∪
            ((U \ givenUpCells R.passes).filter fun c => home c = u)).card := by
          apply Finset.card_le_card
          intro c hc
          rw [Finset.mem_filter] at hc
          rw [Finset.mem_union, Finset.mem_filter, Finset.mem_filter, Finset.mem_sdiff]
          by_cases hg : c ∈ givenUpCells R.passes
          · exact Or.inl ⟨hg, hc.2⟩
          · exact Or.inr ⟨⟨hc.1, hg⟩, hc.2⟩
      _ ≤ ((givenUpCells R.passes).filter fun c => home c = u).card +
            ((U \ givenUpCells R.passes).filter fun c => home c = u).card := Finset.card_union_le _ _
      _ = givenUpAt R.passes hr u + ((U \ givenUpCells R.passes).filter fun c => home c = u).card := by
          rw [hP (givenUpCells R.passes), givenUpAt]

end Properties.Pouw.PearlPlus
