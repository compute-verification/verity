import SampledProofs.WiresSpan

/-! **`check` is sound against its unrolled reference** (`check_sound`): whenever `check` accepts a circuit's wires,
`checkRef`, which expands every reader row the wires cover, accepts them too, with the summary's wires the circuit's. -/

namespace SampledProofs.Wires

theorem firstWire_eq_none {c : Circuit} {f : Wire → Option Code} :
    firstWire c f = none ↔ ∀ w ∈ c.wires, f w = none := by
  unfold firstWire
  constructor
  · intro h w hw
    cases hf : c.wires.zipIdx.find? (fun (w, _) => (f w).isSome) with
    | none =>
      obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hw
      have hm : (c.wires[i], i) ∈ c.wires.zipIdx := List.mem_zipIdx_iff_getElem?.2 (by simp)
      simpa using List.find?_eq_none.1 hf _ hm
    | some p =>
      have hp := List.find?_some hf
      rw [hf] at h
      simp only [Option.bind_some, Option.map_eq_none_iff] at h
      simp [h] at hp
  · intro h
    have : c.wires.zipIdx.find? (fun (w, _) => (f w).isSome) = none := by
      refine List.find?_eq_none.2 fun ⟨w, i⟩ hm => ?_
      have hw : w ∈ c.wires := List.mem_of_getElem? (List.mem_zipIdx_iff_getElem?.1 hm)
      simp [h w hw]
    rw [this]; rfl

theorem shape_cut {c : Circuit} {w : Wire} {rt wt : Template} (h : shape c w = .ok (rt, wt)) :
    1 ≤ w.cuts ∧ w.units % w.cuts = 0 := by
  unfold shape at h
  split at h
  · split at h
    · dsimp only at h
      split at h
      · cases h
      · rename_i hc
        simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at hc
        exact ⟨hc.1.1.1, hc.2⟩
    · cases h
  · cases h

theorem readerUnit_lt {w : Wire} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0) {i : Nat} (hi : i < w.units) :
    i / w.cuts < w.readerUnits := by
  unfold Wire.readerUnits
  have h2 : w.units / w.cuts * w.cuts = w.units := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd)
  exact (Nat.div_lt_iff_lt_mul (by omega)).2 (by rw [h2]; exact hi)

theorem boundsRef_of {w : Wire} {rt wt : Template} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0)
    (h : boundsOk w rt wt = true) : boundsRef w rt wt = true := by
  simp only [boundsOk, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨h1, h2⟩, h3⟩ := h
  simp only [boundsRef, Wire.entries, List.all_map, List.all_eq_true, List.mem_range, Function.comp_apply,
    Wire.readerAt, Bool.and_eq_true, decide_eq_true_eq]
  intro i hi
  have hu := readerUnit_lt hc hd hi
  have hs := w.writerAt_mem_span i hi
  omega

theorem wireCodeRef_of {c : Circuit} {w : Wire} (h : wireCode c w = none) :
    1 ≤ w.cuts ∧ w.units % w.cuts = 0 ∧ wireCodeRef c w = none := by
  unfold wireCode at h
  unfold wireCodeRef
  split at h
  · cases h
  · rename_i rt wt hs
    have ⟨hc, hd⟩ := shape_cut hs
    refine ⟨hc, hd, ?_⟩
    split at h
    · simp [boundsRef_of hc hd ‹_›]
    · cases h

theorem entries_keys (w : Wire) : w.entries.map (·.1) =
    (List.range w.units).map fun i => (w.reader, w.start + i / w.cuts, w.port, w.cut0 + i % w.cuts) := by
  simp [Wire.entries, Wire.readerAt]

/-- A wire covers each of its reader rows once. -/
theorem entries_nodup (w : Wire) : (w.entries.map (·.1)).Nodup := by
  rw [entries_keys, List.Nodup, List.pairwise_map]
  refine (List.nodup_range).imp ?_
  intro a b hab heq
  simp only [Prod.mk.injEq] at heq
  obtain ⟨-, h1, -, h2⟩ := heq
  apply hab
  have e1 : a / w.cuts = b / w.cuts := by omega
  have e2 : a % w.cuts = b % w.cuts := by omega
  rw [← Nat.div_add_mod a w.cuts, ← Nat.div_add_mod b w.cuts, e1, e2]

/-- Two disjoint wires cover no reader row in common. -/
theorem entries_disjoint {a b : Wire} (hca : 1 ≤ a.cuts) (ha : a.units % a.cuts = 0) (hcb : 1 ≤ b.cuts)
    (hb : b.units % b.cuts = 0) (h : disjoint a b = true) :
    ∀ x ∈ a.entries.map (·.1), ∀ y ∈ b.entries.map (·.1), x ≠ y := by
  rw [entries_keys, entries_keys]
  simp only [List.mem_map, List.mem_range]
  rintro _ ⟨i, hi, rfl⟩ _ ⟨j, hj, rfl⟩ heq
  have ui := readerUnit_lt hca ha hi
  have uj := readerUnit_lt hcb hb hj
  have ri := Nat.mod_lt i (by omega : 0 < a.cuts)
  have rj := Nat.mod_lt j (by omega : 0 < b.cuts)
  simp only [Prod.mk.injEq] at heq
  obtain ⟨e1, e2, e3, e4⟩ := heq
  simp only [disjoint, Bool.or_eq_true, bne_iff_ne, ne_eq, decide_eq_true_eq] at h
  rcases h with ((((h | h) | h) | h) | h) | h
  · exact h e1
  · exact h e3
  all_goals
    generalize i / a.cuts = x at *
    generalize j / b.cuts = y at *
    generalize i % a.cuts = x' at *
    generalize j % b.cuts = y' at *
    omega

theorem pairwiseDisjoint_iff : ∀ ws : List Wire, pairwiseDisjoint ws = true ↔ ws.Pairwise (fun a b => disjoint a b = true)
  | [] => by simp [pairwiseDisjoint]
  | w :: ws => by
    simp [pairwiseDisjoint, List.pairwise_cons, List.all_eq_true, pairwiseDisjoint_iff ws]

theorem distinctRows_of {ws : List Wire} (hw : ∀ w ∈ ws, 1 ≤ w.cuts ∧ w.units % w.cuts = 0)
    (h : pairwiseDisjoint ws = true) : distinctRows (ws.flatMap Wire.entries) = true := by
  simp only [distinctRows, decide_eq_true_eq]
  rw [List.map_flatMap, List.Nodup, List.pairwise_flatMap]
  refine ⟨fun w _ => entries_nodup w, ?_⟩
  refine ((pairwiseDisjoint_iff ws).1 h).imp_of_mem fun {a b} ha hb hab => ?_
  exact entries_disjoint (hw a ha).1 (hw a ha).2 (hw b hb).1 (hw b hb).2 hab

/-- **`check` is sound against the unrolled reference**: what it accepts, `checkRef` accepts, its entries those of the
summary's wires, which are the circuit's. -/
theorem check_sound {c : Circuit} {S : Summary} (h : check c = .ok S) :
    S.wires = c.wires ∧ checkRef c = .ok (c.wires.flatMap Wire.entries) := by
  unfold check at h
  split at h
  · cases h
  · rename_i hcut
    split at h
    · cases h
    · rename_i hf
      split at h
      · cases h
      · rename_i hov
        cases h
        have hall := firstWire_eq_none.1 hf
        have hw : ∀ w ∈ c.wires, 1 ≤ w.cuts ∧ w.units % w.cuts = 0 := fun w hw =>
          ⟨(wireCodeRef_of (hall w hw)).1, (wireCodeRef_of (hall w hw)).2.1⟩
        have hf' : firstWire c (wireCodeRef c) = none :=
          firstWire_eq_none.2 fun w hw => (wireCodeRef_of (hall w hw)).2.2
        have hd := distinctRows_of hw (by simpa using hov)
        refine ⟨rfl, ?_⟩
        unfold checkRef
        simp [hcut, hf', hd]

end SampledProofs.Wires
