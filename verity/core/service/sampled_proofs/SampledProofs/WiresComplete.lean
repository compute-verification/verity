import SampledProofs.WiresSound

/-! **`check` is complete against its unrolled reference** (`check_complete`): whatever `checkRef` accepts, `check`
accepts, for wires whose radices are 1 or more (as every wire `Wire.ofJson` reads is). With `check_sound` the two decide
the same circuits: the span is attained (a reader index reaches the least writer and one the greatest), and two wires
of one port whose rectangles meet wire a reader row twice. -/

namespace SampledProofs.Wires

theorem digit_low (r : Nat) (s : Int) (h1 : 1 ≤ r) :
    (((if 0 ≤ s then 0 else r - 1 : Nat)) : Int) * s = min 0 (s * ((r : Int) - 1)) := by
  split
  · rename_i hs
    rw [Int.min_eq_left (Int.mul_nonneg hs (by omega))]
    simp
  · rename_i hs
    rw [Int.min_eq_right (Int.mul_nonpos_of_nonpos_of_nonneg (by omega) (by omega)), Int.mul_comm]
    congr 1
    omega

theorem digit_high (r : Nat) (s : Int) (h1 : 1 ≤ r) :
    (((if 0 ≤ s then r - 1 else 0 : Nat)) : Int) * s = max 0 (s * ((r : Int) - 1)) := by
  split
  · rename_i hs
    rw [Int.max_eq_right (Int.mul_nonneg hs (by omega)), Int.mul_comm]
    congr 1
    omega
  · rename_i hs
    rw [Int.max_eq_left (Int.mul_nonpos_of_nonpos_of_nonneg (by omega) (by omega))]
    simp

theorem index_cons {r i' n : Nat} {d : Nat} (hd : d < r) (hi : i' < n) :
    d + r * i' < r * n ∧ (d + r * i') % r = d ∧ (d + r * i') / r = i' := by
  have : r * (i' + 1) ≤ r * n := Nat.mul_le_mul_left r hi
  rw [Nat.mul_succ] at this
  refine ⟨by omega, ?_, ?_⟩
  · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hd]
  · rw [Nat.add_mul_div_left _ _ (by omega), Nat.div_eq_of_lt hd, Nat.zero_add]

theorem accum_low : ∀ ps : List (Nat × Int), (∀ p ∈ ps, 1 ≤ p.1) → ∃ i, i < prodR ps ∧ accum ps i = lowOf ps
  | [], _ => ⟨0, Nat.one_pos, rfl⟩
  | (r, s) :: ps, h => by
    have h1 : 1 ≤ r := h (r, s) (List.mem_cons_self ..)
    obtain ⟨i', hi', he⟩ := accum_low ps fun p hp => h p (List.mem_cons_of_mem _ hp)
    have hd : (if 0 ≤ s then 0 else r - 1) < r := by split <;> omega
    obtain ⟨hlt, hmod, hdiv⟩ := index_cons hd hi'
    refine ⟨_, hlt, ?_⟩
    simp only [accum, lowOf]
    rw [hmod, hdiv, he, digit_low r s h1]

theorem accum_high : ∀ ps : List (Nat × Int), (∀ p ∈ ps, 1 ≤ p.1) → ∃ i, i < prodR ps ∧ accum ps i = highOf ps
  | [], _ => ⟨0, Nat.one_pos, rfl⟩
  | (r, s) :: ps, h => by
    have h1 : 1 ≤ r := h (r, s) (List.mem_cons_self ..)
    obtain ⟨i', hi', he⟩ := accum_high ps fun p hp => h p (List.mem_cons_of_mem _ hp)
    have hd : (if 0 ≤ s then r - 1 else 0) < r := by split <;> omega
    obtain ⟨hlt, hmod, hdiv⟩ := index_cons hd hi'
    refine ⟨_, hlt, ?_⟩
    simp only [accum, highOf]
    rw [hmod, hdiv, he, digit_high r s h1]

theorem Wire.pairs_pos (w : Wire) (h : ∀ r ∈ w.radix, 1 ≤ r) : ∀ p ∈ w.pairs, 1 ≤ p.1 := fun _ hp =>
  h _ (List.of_mem_zip (List.mem_reverse.1 hp)).1

/-- **The span is attained**: some reader index reads the least writer, and some the greatest. -/
theorem Wire.span_attained (w : Wire) (h : ∀ r ∈ w.radix, 1 ≤ r) :
    (∃ i, i < w.units ∧ w.writerAt i = w.span.1) ∧ (∃ i, i < w.units ∧ w.writerAt i = w.span.2) := by
  obtain ⟨i, hi, he⟩ := accum_low w.pairs (w.pairs_pos h)
  obtain ⟨j, hj, hf⟩ := accum_high w.pairs (w.pairs_pos h)
  exact ⟨⟨i, hi, by simp [Wire.writerAt, Wire.span, he]⟩, ⟨j, hj, by simp [Wire.writerAt, Wire.span, hf]⟩⟩

theorem boundsOk_of_ref {w : Wire} {rt wt : Template} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0)
    (hr : ∀ r ∈ w.radix, 1 ≤ r) (h : boundsRef w rt wt = true) : boundsOk w rt wt = true := by
  simp only [boundsRef, Wire.entries, List.all_map, List.all_eq_true, List.mem_range, Function.comp_apply,
    Wire.readerAt, Bool.and_eq_true, decide_eq_true_eq] at h
  obtain ⟨⟨i, hi, hlo⟩, ⟨j, hj, hhi⟩⟩ := w.span_attained hr
  have h5 : w.units / w.cuts * w.cuts = w.units := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd)
  have hlast := h (w.units - 1) (by omega)
  have hq : w.readerUnits - 1 ≤ (w.units - 1) / w.cuts := by
    refine (Nat.le_div_iff_mul_le (by omega)).2 ?_
    rw [Nat.sub_mul, Nat.one_mul]
    unfold Wire.readerUnits
    omega
  have hru : 1 ≤ w.readerUnits := by
    unfold Wire.readerUnits
    exact Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.zero_mul] at h5; omega
  have := (h i hi).1.2
  have := (h j hj).2
  simp only [boundsOk, Bool.and_eq_true, decide_eq_true_eq]
  omega

theorem wireCode_of_ref {c : Wiring} {w : Wire} (hr : ∀ r ∈ w.radix, 1 ≤ r) (h : wireCodeRef c w = none) :
    wireCode c w = none := by
  unfold wireCodeRef at h
  unfold wireCode
  split at h
  · cases h
  · rename_i rt wt hs
    have ⟨hc, hd⟩ := shape_cut hs
    split at h
    · simp [boundsOk_of_ref hc hd hr ‹_›]
    · cases h

/-- A reader row a wire covers is among its expanded rows. -/
theorem mem_keys {w : Wire} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0) {u r : Nat} (hu1 : w.start ≤ u)
    (hu2 : u < w.start + w.readerUnits) (hr1 : w.cut0 ≤ r) (hr2 : r < w.cut0 + w.cuts) :
    (w.reader, u, w.port, r) ∈ w.entries.map (·.1) := by
  obtain ⟨k, hk⟩ : ∃ k, w.indexOf w.reader u w.port r = some k := by
    unfold Wire.indexOf
    simp only [beq_self_eq_true, Bool.true_and, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨(u - w.start) * w.cuts + (r - w.cut0), by simp [hu1, hu2, hr1, hr2]⟩
  have hlt := indexOf_lt hd hk
  have := (indexOf_iff hc hd w.reader u w.port r k hlt).2 hk
  simp only [beq_iff_eq] at this
  rw [entries_keys, List.mem_map]
  exact ⟨k, List.mem_range.2 hlt, this⟩

theorem readerUnits_pos {w : Wire} (hd : w.units % w.cuts = 0) (hu : 1 ≤ w.units) :
    1 ≤ w.readerUnits := by
  have h5 : w.units / w.cuts * w.cuts = w.units := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd)
  unfold Wire.readerUnits
  exact Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.zero_mul] at h5; omega

/-- Two wires that share no expanded reader row are disjoint. -/
theorem disjoint_of_entries {a b : Wire} (hca : 1 ≤ a.cuts) (hda : a.units % a.cuts = 0) (hua : 1 ≤ a.units)
    (hcb : 1 ≤ b.cuts) (hdb : b.units % b.cuts = 0) (hub : 1 ≤ b.units)
    (h : ∀ x ∈ a.entries.map (·.1), ∀ y ∈ b.entries.map (·.1), x ≠ y) : disjoint a b = true := by
  cases hdis : disjoint a b with
  | true => rfl
  | false =>
    exfalso
    simp only [disjoint, Bool.or_eq_false_iff, bne_eq_false_iff_eq, decide_eq_false_iff_not, Nat.not_le] at hdis
    obtain ⟨⟨⟨⟨⟨e1, e2⟩, h1⟩, h2⟩, h3⟩, h4⟩ := hdis
    have ra := readerUnits_pos hda hua
    have rb := readerUnits_pos hdb hub
    have xa := mem_keys (u := max a.start b.start) (r := max a.cut0 b.cut0) hca hda (by omega) (by omega)
      (by omega) (by omega)
    have xb := mem_keys (u := max a.start b.start) (r := max a.cut0 b.cut0) hcb hdb (by omega) (by omega)
      (by omega) (by omega)
    rw [← e1, ← e2] at xb
    exact h _ xa _ xb rfl

theorem prodR_pos : ∀ ps : List (Nat × Int), (∀ p ∈ ps, 1 ≤ p.1) → 1 ≤ prodR ps
  | [], _ => Nat.le_refl 1
  | (r, s) :: ps, h => by
    have h1 : 1 ≤ r := h (r, s) (List.mem_cons_self ..)
    have := prodR_pos ps fun p hp => h p (List.mem_cons_of_mem _ hp)
    show 1 ≤ r * prodR ps
    exact Nat.mul_le_mul h1 this

/-- **`check` is complete against the unrolled reference**: what `checkRef` accepts, `check` accepts, for wires whose
radices are 1 or more. -/
theorem check_complete {c : Wiring} {es : List Entry} (hr : ∀ w ∈ c.wires, ∀ r ∈ w.radix, 1 ≤ r)
    (h : checkRef c = .ok es) : check c = .ok ⟨c.wires, c.wires.map (rowWords c)⟩ := by
  unfold checkRef at h
  split at h
  · cases h
  · rename_i hcut
    split at h
    · cases h
    · rename_i hf
      dsimp only at h
      split at h
      · cases h
      · rename_i hov
        have hall := firstWire_eq_none.1 hf
        have hw : ∀ w ∈ c.wires, 1 ≤ w.cuts ∧ w.units % w.cuts = 0 ∧ 1 ≤ w.units := fun w hw =>
          match hs : shape c w with
          | .ok (_, _) => ⟨(shape_cut hs).1, (shape_cut hs).2, prodR_pos _ (w.pairs_pos (hr w hw))⟩
          | .error e => by have := hall w hw; simp [wireCodeRef, hs] at this
        have hf' : firstWire c (wireCode c) = none :=
          firstWire_eq_none.2 fun w hw' => wireCode_of_ref (hr w hw') (hall w hw')
        have hd : pairwiseDisjoint c.wires = true := by
          simp only [Bool.not_eq_true', Bool.not_eq_false, distinctRows, decide_eq_true_eq] at hov
          rw [List.map_flatMap, List.Nodup, List.pairwise_flatMap] at hov
          refine (pairwiseDisjoint_iff _).2 (hov.2.imp_of_mem fun {a b} ha hb hab => ?_)
          exact disjoint_of_entries (hw a ha).1 (hw a ha).2.1 (hw a ha).2.2 (hw b hb).1 (hw b hb).2.1 (hw b hb).2.2 hab
        unfold check
        simp [hcut, hf', hd]

end SampledProofs.Wires
