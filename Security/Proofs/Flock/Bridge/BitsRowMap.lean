import Proofs.Flock.Bridge.BitsKeys
import Proofs.Flock.Bridge.BitsRows
import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
import Proofs.Flock.Soundness.Discharge.Hidden.Rows
import Proofs.Flock.Soundness.Discharge.Hidden.Delta

/-!
# Bits' row map is the bridge's (`SampledProofs.Bits.rowBit`, `outRowBit`)

`SampledProofs.Bits.rowBit c n g` is where source gate `g` of Bits' key program is read, and `outRowBit c u k j t` where
unit `u`'s instance `k` commits bit `t` of its returned output leaf `j`. The bridge's row bits are `Layout.keyRowOf` at
an input gate and `HiddenExec.outBitRow` at a returned output's bit; `rowPosOf` reads one as Bits' position: the port's
name, the unit's leaf and the bit.

* `keyIdx_self`, `keyIdx_lt`: a key is its own key, below the inputs, so the gate a unit's input column reads is a
  source gate `rowBit` places.
* `rowBit_keyRowOf`: at a source gate that is its own key and whose key is the zero or a row bit (`RowKey`), Bits'
  commitment position is the bridge's row bit, and both are none off the source gates.
* `outAt_mono`, `outPos_eq`, `outRowBit_outBitRow`: where each output row holds one output port (`outGroup_single`),
  Bits' output row of a word is its port's (`outPortOf`), so Bits' position of a returned output's bit is the bridge's.
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge
open FlockSoundness.Discharge.Layout (keyN keyB keyRowOf portRowW RowKey)
open FlockSoundness.Discharge.HiddenExec (outBitRow outBitRow_spec outWordOf outPortOf outPortOf_spec)

variable {c : Flock.Circuit} {n : ℕ}

variable (c) in
/-- **A bridge row bit as Bits' commitment position**: bit `k` of unit `u`'s leaf of row port `q`. -/
def rowPosOf (r : Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) : SampledProofs.Bits.RowPos :=
  ⟨c.ports[r.2.1].name, r.1.val, r.2.2.val⟩

/-! ## Source gates -/

/-- **A key is its own key** (`SampledProofs.Bits.keyIdx`): the first input of a source has that source. -/
theorem keyIdx_self {B x b : ℕ} (h : SampledProofs.Bits.keyIdx c B x = some b) :
    SampledProofs.Bits.keyIdx c B b = some b := by
  unfold SampledProofs.Bits.keyIdx at h ⊢
  dsimp only at h ⊢
  split_ifs at h with h0
  have hb := List.find?_some h
  simp only [decide_eq_true_eq] at hb
  rw [ite_eq_right (by rw [hb]; exact h0), hb]
  exact h

theorem keyIdx_lt {B x b : ℕ} (h : SampledProofs.Bits.keyIdx c B x = some b) : b < B := by
  unfold SampledProofs.Bits.keyIdx at h
  dsimp only at h
  split_ifs at h
  exact List.mem_range.1 (List.mem_of_find?_eq_some h)

/-- **Bits' row map at a source gate is the bridge's**: at gate `2 + u·B + b` whose input `b` is its own key (`keyIdx`)
and whose key is the zero or a row bit (`RowKey`), Bits' commitment position is the bridge's row bit (`keyRowOf`); both
are none off the source gates. -/
theorem rowBit_keyRowOf {hU : Built c.unit} (hu : c.unit.unitLog ≤ 32) (g : Fin (keyN c hU n))
    (h : 2 ≤ g.val → g.val < 2 + n * keyB c hU →
      SampledProofs.Bits.keyIdx c (keyB c hU) ((g.val - 2) % keyB c hU) = some ((g.val - 2) % keyB c hU) ∧
        RowKey c (portRowW c) ((g.val - 2) % keyB c hU)) :
    SampledProofs.Bits.rowBit c n g.val = (keyRowOf c hU n (portRowW c) g).map (rowPosOf c) := by
  have hB : ((SampledProofs.Bits.Rows.ofNet c.unit).stack c.upv).nIn = keyB c hU :=
    (rowsAgree_circuitRows c hU hu).1
  unfold SampledProofs.Bits.rowBit keyRowOf
  dsimp only
  rw [hB]
  by_cases hr : 2 ≤ g.val ∧ g.val < 2 + n * keyB c hU
  · obtain ⟨hself, hk⟩ := h hr.1 hr.2
    have hb : keyB c hU ≠ 0 := fun h0 => by rw [h0, Nat.mul_zero] at hr; omega
    rw [dite_eq_left hr, ite_eq_right]
    · unfold RowKey at hk
      rw [srcKey_eq] at hk ⊢
      generalize SampledProofs.Bits.srcKey c (SampledProofs.Bits.Index.of c) ((g.val - 2) % keyB c hU) = s at hk ⊢
      cases s with
      | msg q byte bit =>
        obtain h0 | ⟨q', byte', bit', he, hq, hw, hb8⟩ := hk
        · exact absurd h0 (by simp [srcOf])
        · simp only [srcOf, Aliased.Src.msg.injEq] at he
          obtain ⟨rfl, rfl, rfl⟩ := he
          simp only [srcOf, SampledProofs.Bits.srcPos, hq, hw, hb8, and_self, ↓reduceDIte, Option.map_some, rowPosOf,
            Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hq, Option.getD_some]
          rfl
      | _ => rfl
    · simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, bne_iff_ne, ne_eq, not_or, not_not]
      exact ⟨⟨⟨by omega, hb⟩, by omega⟩, hself⟩
  · rw [dite_eq_right hr, ite_eq_left]
    · rfl
    · rcases Nat.lt_or_ge g.val 2 with h1 | h1
      · simp [h1]
      · have h2 : 2 + n * keyB c hU ≤ g.val := by omega
        simp [h2]

/-! ## Returned outputs -/

variable (c) in
theorem outAt_mono {a b : ℕ} (h : a ≤ b) : Hidden.outAt c a ≤ Hidden.outAt c b := by
  obtain ⟨d, rfl⟩ := Nat.exists_eq_add_of_le h
  unfold Hidden.outAt
  rw [List.take_add, List.map_append, List.sum_append]
  exact Nat.le_add_right _ _

/-- **Bits' output row of a word is its port's**: where each output row holds one output port (`outGroup_single`), the
first output row whose run of words holds word `x` is `x`'s port (`outPortOf`), the last starting at or before it. -/
theorem outPos_eq (hr : c.outRowPorts = Array.replicate c.outPorts.size 1) {x : ℕ} (hx : x < c.outWords) (u t : ℕ) :
    SampledProofs.Bits.outPos c u x t = some ⟨(c.ports.getD (HmOut.nIn c + outPortOf c x) default).name, u,
      16 * (x - Hidden.outAt c (outPortOf c x)) + t⟩ := by
  obtain ⟨hlt, hle, hw⟩ := outPortOf_spec c hx
  have hs : c.outRowPorts.size = c.outPorts.size := by rw [hr, Array.size_replicate]
  have hf : (List.range c.outRowPorts.size).find? (fun o => decide ((HmOut.outGroup c o).2.2.1 ≤ x) &&
      decide (x < (HmOut.outGroup c o).2.2.1 + (HmOut.outGroup c o).2.2.2)) = some (outPortOf c x) := by
    rw [List.find?_range_eq_some]
    refine ⟨?_, List.mem_range.2 (hs ▸ hlt), fun j hj => ?_⟩
    · rw [Hidden.outGroup_single hr hlt]
      simp only [Bool.and_eq_true, decide_eq_true_eq]
      omega
    · have hjs : j < c.outPorts.size := by omega
      rw [Hidden.outGroup_single hr hjs]
      have h1 := Hidden.outAt_succ c hjs
      have h2 := outAt_mono c (show j + 1 ≤ outPortOf c x by omega)
      have e : c.outPorts[j]!.2 = (c.outPorts.getD j default).2 := by
        rw [getElem!_pos c.outPorts j hjs, Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hjs,
          Option.getD_some]
      simp only [Bool.not_eq_true', Bool.and_eq_false_iff, decide_eq_false_iff_not, Nat.not_le, Nat.not_lt]
      omega
  unfold SampledProofs.Bits.outPos
  simp only [hf, Option.map_some, Hidden.outGroup_single hr hlt]

/-- **Bits' position of a returned output's bit is the bridge's**: where each output row holds one output port, at an
output leaf `j` of instance `k` whose word is an output word and whose bit `t` is a bit of its row (`outBitRow`). -/
theorem outRowBit_outBitRow (hr : c.outRowPorts = Array.replicate c.outPorts.size 1) (u : Fin n) {k j t : ℕ}
    (hj : j < (c.leavesOut.getD k #[]).size) (hw : outWordOf c (k, j, t) < c.outWords)
    (hb : outBitRow c u (k, j, t) ≠ none) :
    SampledProofs.Bits.outRowBit c u.val k j t = (outBitRow c u (k, j, t)).map (rowPosOf c) := by
  have ew : outWordOf c (k, j, t) = (c.leavesOut.getD k #[])[j] := getElem!_pos _ j hj
  obtain ⟨⟨u', q, b⟩, hs⟩ := Option.ne_none_iff_exists'.1 hb
  obtain ⟨rfl, hq, hk⟩ := outBitRow_spec hs
  rw [ew] at hq hk hw
  rw [hs, Option.map_some]
  unfold SampledProofs.Bits.outRowBit
  rw [Array.getElem?_eq_getElem hj, Option.bind_some, outPos_eq hr hw]
  simp only [rowPosOf, Option.some.injEq, SampledProofs.Bits.RowPos.mk.injEq, ← hk, ← hq, and_true]
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem q.isLt, Option.getD_some]
  rfl

end Flock.Bridge
