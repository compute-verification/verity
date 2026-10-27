import Proofs.Flock.Soundness.Discharge.HiddenExec.PinnedOne
import Proofs.Flock.Soundness.Discharge.Inst.Order
import Proofs.Flock.Soundness.ExecLookup

/-!
# Every pinned bit of an accepted statement is a linear row

`HmOut.checkPinned` holds each pinned output word's net to a slot of a net past the rows' circuits and each of its 16
bits' rows to `HmOut.linearRow`: past the net's input rows, below its constant, outside a lookup's product rows, its `B`
the constant alone. A net `HmRow.parse` accepted is built by `Net.ofRows` or `Lookup.build`, whose `B` entry is the
count of the column in its row (`sparseEntry_ofRows`), so such a row's local `B` (`entB`, the table side included) is
the constant alone. With `HmOut.outCol` the column of `outRow`'s row in `outSlot`'s slot (`outCol_slot`), every pinned
column is a `LinearBit` (`pinned_linear`), which `sat_pinned` reads. Nothing here reads a row size, a `G` or a producer
family.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 FlockSoundness.Discharge.Inst

/-- **An untyped circuit's output column is `outRow`'s row in `outSlot`'s slot.** -/
theorem outCol_slot {c : Circuit} (ht : c.typed = none) (g o : ℕ) :
    HmOut.outCol c g o = c.slotCol (HmOut.outRow c o).1 g (HmOut.outSlot c o) + (HmOut.outRow c o).2 := by
  unfold HmOut.outCol HmOut.outRow HmOut.outSlot
  simp only [ht, Option.isSome_none, Bool.false_eq_true, ↓reduceIte]
  split_ifs <;> rfl

/-- **What `HmOut.linearRow` checks.** -/
theorem linearRow_spec {net : Net} {i : ℕ} (h : HmOut.linearRow net i = true) :
    net.b.row i = #[net.constPos] ∧ net.inWords * WORD ≤ i ∧ i < net.constPos ∧
      ∀ j, lookupEntry net.lookup i j = 0 := by
  unfold HmOut.linearRow at h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at h
  obtain ⟨⟨⟨hb, hlo⟩, hhi⟩, hlk⟩ := h
  refine ⟨hb, hlo, hhi, fun j => ?_⟩
  cases hl : net.lookup with
  | none => rfl
  | some l =>
    rw [hl] at hlk
    have hn : ¬ (l.prod0 ≤ i ∧ i < l.prod0 + l.bits * 2 ^ (l.n - l.lo)) := by
      rintro ⟨h1, h2⟩
      simp [h1, h2] at hlk
    simp only [lookupEntry]
    split_ifs with hc
    · exact absurd ⟨hc.1, hc.2.1⟩ hn
    · rfl

/-- **A linear row's local `B` is the constant alone**, at a net `Net.ofRows` or `Lookup.build` built. -/
theorem entB_linear {c : Circuit} {n i : ℕ}
    (hk : Built (netOf c n) ∨ ∃ nm tb nb lo bits, Lookup.build nm tb nb lo bits = .ok (netOf c n))
    (h : HmOut.linearRow (netOf c n) i = true) (j : ℕ) :
    entB c n i j = if j = (netOf c n).constPos then 1 else 0 := by
  obtain ⟨hb, -, -, hlk⟩ := linearRow_spec h
  have hs : sparseEntry (netOf c n).b i j = ((((netOf c n).b.row i).toList.count j : ℕ) : ZMod 2) := by
    rcases hk with hk | ⟨nm, tb, nb, lo, bits, hk⟩
    · exact sparseEntry_built_b hk i j
    · obtain ⟨-, -, -, -, -, d1, d2, r3, -, -, -, -, -, -, hB, -⟩ := build_spec hk
      exact sparseEntry_ofRows hB i j
  unfold entB
  rw [hlk, add_zero, hs, hb, count_singleton_zmod]

/-- **What `HmOut.checkPinned` checks of each pinned output word**: its net past the rows' circuits, its slot one of the
net's, and each of its 16 bits' rows a linear row. -/
theorem pinned_word {c : Circuit} (h : HmOut.checkPinned c = .ok ()) {o : ℕ} (h1 : HmOut.pinnedFrom c ≤ o)
    (h2 : o < HmOut.pinnedFrom c + HmOut.pinnedWords c) :
    (HmOut.outRow c o).1 < c.nets.size ∧ (HmOut.outRow c o).1 ≠ c.shaNet ∧ (HmOut.outRow c o).1 ≠ c.hmNet ∧
      HmOut.outSlot c o < c.perVu.getD (HmOut.outRow c o).1 0 ∧
      ∀ t < 16, HmOut.linearRow (netOf c (HmOut.outRow c o).1) ((HmOut.outRow c o).2 + t) = true := by
  unfold HmOut.checkPinned at h
  split at h
  · rename_i e he
    have hk : HmRow.pinnedCount c.metaJ = 0 := by simp [HmRow.pinnedCount, he]
    rw [Hidden.pinnedWords_zero hk] at h2
    omega
  · dsimp only at h
    split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · rename_i hb
          have hm : o ∈ List.range' (HmOut.pinnedFrom c) (HmOut.pinnedWords c) := List.mem_range'_1.2 ⟨h1, h2⟩
          have hf := List.any_eq_false.1 (Bool.eq_false_iff.2 hb) o hm
          rcases hro : HmOut.outRow c o with ⟨n, r⟩
          rw [hro] at hf
          dsimp only at hf
          rw [Bool.not_eq_true, Bool.or_eq_false_iff, Bool.not_eq_eq_eq_not, Bool.not_false] at hf
          obtain ⟨hA, hrow⟩ := hf
          simp only [Bool.and_eq_true, decide_eq_true_eq, bne_iff_ne, ne_eq] at hA
          obtain ⟨⟨⟨hn, hsha⟩, hhm⟩, hs⟩ := hA
          refine ⟨hn, hsha, hhm, hs, fun t ht => ?_⟩
          have := List.any_eq_false.1 hrow t (List.mem_range.2 ht)
          simpa [FlockSoundness.Discharge.Inst.netOf] using this

/-- **Every pinned bit of a statement `HmOut.checkPinned` accepts is a linear row**, at an untyped circuit whose nets
`Net.ofRows` or `Lookup.build` built (`ParseFacts.nets`). -/
theorem pinned_linear {c : Circuit} (h : HmOut.checkPinned c = .ok ()) (ht : c.typed = none)
    (hnets : ∀ x ∈ c.nets.toList, Built x.2 ∨ ∃ nm tb n lo bits, Lookup.build nm tb n lo bits = .ok x.2) :
    ∀ x ∈ Hidden.pinnedCols c, LinearBit c x := by
  intro x hx
  unfold Hidden.pinnedCols at hx
  simp only [List.mem_flatMap, List.mem_range, List.mem_map] at hx
  obtain ⟨g, hg, o, ho, t, ht16, rfl⟩ := hx
  obtain ⟨hn, hsha, hhm, hs, hrow⟩ := pinned_word h (o := HmOut.pinnedFrom c + o) (by omega) (by omega)
  have hl := hrow t ht16
  obtain ⟨-, hlo, hhi, -⟩ := linearRow_spec hl
  have hk : Built (netOf c (HmOut.outRow c (HmOut.pinnedFrom c + o)).1) ∨
      ∃ nm tb nb lo bits, Lookup.build nm tb nb lo bits = .ok (netOf c (HmOut.outRow c (HmOut.pinnedFrom c + o)).1) := by
    have hmem := Array.getElem_mem_toList (xs := c.nets) hn
    have he : netOf c (HmOut.outRow c (HmOut.pinnedFrom c + o)).1 = c.nets[(HmOut.outRow c (HmOut.pinnedFrom c + o)).1].2 := by
      simp [netOf, Array.getD_eq_getD_getElem?, hn]
    rw [he]
    exact hnets _ hmem
  refine ⟨g, hg, (HmOut.outRow c (HmOut.pinnedFrom c + o)).1, HmOut.outSlot c (HmOut.pinnedFrom c + o),
    (HmOut.outRow c (HmOut.pinnedFrom c + o)).2 + t, ?_, hn, hsha, hhm, hs, hlo, hhi, entB_linear hk hl⟩
  rw [pos_code, outCol_slot ht, Nat.add_assoc]

end FlockSoundness.Discharge.HiddenExec
