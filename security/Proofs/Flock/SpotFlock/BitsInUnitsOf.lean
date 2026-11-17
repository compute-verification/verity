module

public import Proofs.Flock.SpotFlock.InsideOf
public import Proofs.Flock.SpotFlock.NeededBitsOf

/-! Each bit is a gate of its unit's proof block (`bitsInUnits`), the fact `inside_of` takes, with no hypothesis. -/

@[expose] public section

namespace Flock.Bridge.NeededBitsOf

open Claims.Sampling Properties.Spot

section key

variable (c : Flock.Circuit) (n : ℕ)

theorem free_src {g : ℕ} (hg : g < (Spot.Bits.keyProg c n).inputs) (hκ : Spot.Bits.consts c n g = none) :
    ∃ u b, u < n ∧ b < (bitsRows c).nIn ∧ g = 2 + u * (bitsRows c).nIn + b ∧
      Spot.Bits.owner (Spot.Bits.keyProg c n) g = some u := by
  have hne : Spot.Bits.owner (Spot.Bits.keyProg c n) g ≠ none := by
    intro ho
    have hκ' := hκ
    unfold Spot.Bits.consts at hκ'
    split at hκ'
    · cases hκ'
    · split at hκ'
      · cases hκ'
      · rename_i h2
        exact h2 (by simp [hg, ho])
  obtain ⟨u, ho⟩ := Option.ne_none_iff_exists'.1 hne
  obtain ⟨g', hg', hw⟩ := exists_of_owner ho
  have hin := hg
  rw [keyProg_inputs] at hin
  rcases (args_of_unit c n hg' hw).2 with h0 | h0 | ⟨col, b, hb, h0⟩ | ⟨i, hi, h0⟩
  · rw [h0, consts_zero] at hκ
    cases hκ
  · rw [h0, consts_one] at hκ
    cases hκ
  · exact ⟨u, b, (args_of_unit c n hg' hw).1, keyIdx_lt hb, h0, ho⟩
  · exact absurd hin (by omega)

end key

section bitsIn

variable {ms : Spot.Bits.Members}

theorem src_in_unit {m g : ℕ} {r : Spot.Bits.RowPos} (hm : m < ms.length)
    (hg : g < (Spot.Bits.keyProg ms[m].1 ms[m].2.size).inputs)
    (hκ : Spot.Bits.consts ms[m].1 ms[m].2.size g = none)
    (hr : Spot.Bits.rowBit ms[m].1 ms[m].2.size g = some r) :
    r.leaf < ms[m].2.size ∧ ∃ b, (membersPart ms).blocks[unitBase ms m + r.leaf]? = some b ∧
      Spot.Bits.base ms m + g ∈ b := by
  obtain ⟨u, b, hu, hb, hgu, ho⟩ := free_src _ _ hg hκ
  have hleaf : r.leaf = u := by
    rw [rowBit_leaf hr, hgu, Nat.add_assoc, Nat.add_sub_cancel_left, div_units hb]
  rw [hleaf]
  refine ⟨hu, _, Members.getElem?_blocks_membersPart ms hm hu, List.mem_map.2 ⟨g, ?_, rfl⟩⟩
  unfold Spot.Bits.blockOf
  refine List.mem_append.2 (Or.inr (List.mem_map.2
    ⟨g, mem_insOf.2 ⟨lt_size_of_lt hg, row?_of_lt hg, hκ, ho⟩, ?_⟩))
  rw [posOf_of_none (row?_of_lt hg), off_eq_of_le_inputs (Nat.le_of_lt hg)]

theorem written_in_unit {m u k j t x : ℕ} {r : Spot.Bits.RowPos} (hm : m < ms.length)
    (hx : Spot.Bits.outGate ms[m].1 ms[m].2.size u k j t = some x)
    (hr : Spot.Bits.outRowBit ms[m].1 u k j t = some r) :
    r.leaf < ms[m].2.size ∧ ∃ b, (membersPart ms).blocks[unitBase ms m + r.leaf]? = some b ∧
      Spot.Bits.base ms m + Spot.Bits.posOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) x ∈ b := by
  obtain ⟨hu, -, -, -, i, hi, rfl⟩ := outGate_some _ _ hx
  have hxu := unitOf_row _ _ hu hi
  obtain ⟨a, b, hab⟩ := row?_of_unitOf hxu
  rw [outRowBit_leaf hr]
  refine ⟨hu, _, Members.getElem?_blocks_membersPart ms hm hu, List.mem_map.2 ⟨_, ?_, rfl⟩⟩
  unfold Spot.Bits.blockOf
  refine List.mem_append.2 (Or.inl ?_)
  rw [posOf_of_some hab]
  exact mem_unitGates_of (unitOf_eq_some hxu).2.1 hxu (by rw [gateSize_of_some hab]; omega)

end bitsIn

end Flock.Bridge.NeededBitsOf

namespace SpotFlock

open Claims.Sampling
open Properties.Spot (bitsProgs membersPart memberProg unitBase)

theorem bitsInUnits (h : Held) : BitsInUnits h := by
  intro x t rp hx
  rcases hx with hs | hw
  · obtain ⟨m, hm, g, hg, hκ, r, hr, he⟩ := _root_.Flock.Bridge.NeededBitsOf.of_mem_sourceBits hs
    have e1 : x = Spot.Bits.base h.ms m + g := congrArg Prod.fst he
    have e2 : t = m := congrArg (fun e => e.2.1) he
    have e3 : rp = r := congrArg (fun e => e.2.2) he
    rw [e1, e2, e3]
    exact ⟨hm, _root_.Flock.Bridge.NeededBitsOf.src_in_unit hm hg hκ hr⟩
  · obtain ⟨m, hm, u, k, j, t', x', r, hx', hr, he⟩ := _root_.Flock.Bridge.NeededBitsOf.of_mem_writtenBits hw
    have e1 : x = Spot.Bits.base h.ms m +
        Spot.Bits.posOf (Spot.Bits.keyProg h.ms[m].1 h.ms[m].2.size) x' := congrArg Prod.fst he
    have e2 : t = m := congrArg (fun e => e.2.1) he
    have e3 : rp = r := congrArg (fun e => e.2.2) he
    rw [e1, e2, e3]
    exact ⟨hm, _root_.Flock.Bridge.NeededBitsOf.written_in_unit hm hx' hr⟩

theorem inside_of_rowed {h : Held} (hR : MembersRowed h) (hO : WrittenOnce h) : Inside h :=
  inside_of (neededBits_of_rowed hR) (sideBySide_of_held h) (bitsInUnits h) hO

end SpotFlock
