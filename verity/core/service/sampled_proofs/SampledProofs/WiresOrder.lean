module

public import SampledProofs.WiresSpan

/-! **The order's closed form against its entries** (`Wires` "The order"). An order's blocks rank each template's units
monotonically (`blockOf_mono`), so a wire whose first and last reader units lie in one block has every reader unit in it,
and a wire whose greatest writer's block ranks lower has every writer's block lower: `orderedWire` implies
`orderedWireRef` (`orderedWire_ref`). The converse is `orderedWire_of_ref`'s, beside `check_complete`. -/

@[expose] public section

namespace SampledProofs.Wires

theorem readerUnit_lt {w : Wire} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0) {i : Nat} (hi : i < w.units) :
    i / w.cuts < w.readerUnits := by
  unfold Wire.readerUnits
  have h2 : w.units / w.cuts * w.cuts = w.units := Nat.div_mul_cancel (Nat.dvd_of_mod_eq_zero hd)
  exact (Nat.div_lt_iff_lt_mul (by omega)).2 (by rw [h2]; exact hi)

theorem blockOf_go_ge {t : String} {u : Nat} : ∀ {o : List (String × Nat)} {k off b : Nat},
    blockOf.go t u o k off = some b → k ≤ b
  | [], _, _, _, h => by simp [blockOf.go] at h
  | (s, n) :: rest, k, off, b, h => by
    simp only [blockOf.go] at h
    split at h
    · split at h
      · cases h; exact Nat.le_refl _
      · exact Nat.le_of_succ_le (blockOf_go_ge h)
    · exact Nat.le_of_succ_le (blockOf_go_ge h)

theorem blockOf_go_mono {t : String} {x y : Nat} (hxy : x ≤ y) : ∀ {o : List (String × Nat)} {k off b : Nat},
    blockOf.go t y o k off = some b → ∃ a, blockOf.go t x o k off = some a ∧ a ≤ b
  | [], _, _, _, h => by simp [blockOf.go] at h
  | (s, n) :: rest, k, off, b, h => by
    simp only [blockOf.go] at h ⊢
    split at h
    · rename_i hs
      simp only [hs, ↓reduceIte]
      split at h
      · cases h
        exact ⟨k, by simp only [show x < off + n by omega, ↓reduceIte], Nat.le_refl _⟩
      · by_cases hx : x < off + n
        · exact ⟨k, by simp only [hx, ↓reduceIte], Nat.le_of_succ_le (blockOf_go_ge h)⟩
        · simp only [hx, ↓reduceIte]
          exact blockOf_go_mono hxy h
    · rename_i hs
      simp only [hs, Bool.false_eq_true, ↓reduceIte]
      exact blockOf_go_mono hxy h

/-- **A template's blocks rank in unit order**: a unit below one in block `b` lies in a block of rank at most `b`. -/
theorem blockOf_mono {o : List (String × Nat)} {t : String} {x y b : Nat} (hxy : x ≤ y) (h : blockOf o t y = some b) :
    ∃ a, blockOf o t x = some a ∧ a ≤ b :=
  blockOf_go_mono hxy h

/-- **The closed form implies the entries**: a wire in order by its first and last reader units and its greatest writer
has each reader unit in the first one's block and each writer's block lower. -/
theorem orderedWire_ref {o : List (String × Nat)} {w : Wire} (hc : 1 ≤ w.cuts) (hd : w.units % w.cuts = 0)
    (h : orderedWire o w = true) : orderedWireRef o w = true := by
  unfold orderedWire at h
  split at h
  · rename_i a a' b ha ha' hb
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
    obtain ⟨rfl, hba⟩ := h
    simp only [orderedWireRef, Wire.entries, List.all_map, List.all_eq_true, List.mem_range, Function.comp_apply,
      Wire.readerAt]
    intro i hi
    have hu := readerUnit_lt hc hd hi
    have hs := w.writerAt_mem_span i hi
    have hq : w.start + i / w.cuts ≤ w.start + w.readerUnits - 1 := by
      generalize i / w.cuts = q at hu ⊢
      omega
    obtain ⟨au, hau, hle⟩ := blockOf_mono hq ha'
    obtain ⟨as, has, hle'⟩ := blockOf_mono (Nat.le_add_right w.start (i / w.cuts)) hau
    rw [ha] at has
    cases has
    obtain ⟨bv, hbv, hle''⟩ := blockOf_mono (Int.toNat_le_toNat hs.2) hb
    rw [hau, ha, hbv]
    simp only [beq_iff_eq, Bool.and_eq_true, decide_eq_true_eq]
    omega
  · simp at h

end SampledProofs.Wires
