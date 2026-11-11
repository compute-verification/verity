module

public import Proofs.Flock.Bridge.BitsKeys
public import Proofs.Flock.Bridge.BitsRows
public import Proofs.Flock.Bridge.LeafMap
public import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
public import Proofs.Flock.Soundness.Discharge.Hidden.Rows
public import Proofs.Flock.Soundness.Discharge.Hidden.Delta

/-!
# Bits' row map is the bridge's (`Spot.Bits.rowBit`, `outRowBit`)

`Spot.Bits.rowBit c n g` is where source gate `g` of Bits' key program is read, and `outRowBit c u k j t` where
unit `u`'s instance `k` commits bit `t` of its returned output leaf `j`. The bridge's row bits are `Layout.keyRowOf` at
an input gate and `HiddenExec.outBitRow` at a returned output's bit; `rowPosOf` reads one as Bits' position: the port's
name, the unit's leaf and the bit.

* `rowPosOf_leaf`: at a public file without shared rows, that leaf is the bridge's leaf map's (`LeafMap.ofPublic`). At a
  shared-rows file Bits still names leaf `u`, and the refs enter below it (`Public.digest`).
* `keyIdx_self`, `keyIdx_lt`: a key is its own key, below the inputs, so the gate a unit's input column reads is a
  source gate `rowBit` places.
* `wire_inGate`: a column of a unit's own instance on a source gate is an input column of the unit, and the gate's
  input is its own key, with the column's source.
* `rowBit_keyRowOf`: at a source gate that is its own key and whose key is the zero or a row bit (`RowKey`), Bits'
  commitment position is the bridge's row bit, and both are none off the source gates.
* `rowOfH_of_input`, `rowBit_wire`: at an input gate the bridge's row bit (`rowOfH`) is its key's, and at a gate a
  unit's row reads, in a circuit whose read inputs have row keys (`KeysRow`), Bits' position is that row bit.
* `outAt_mono`, `outPos_eq`, `outRowBit_outBitRow`: where each output row holds one output port (`outGroup_single`),
  Bits' output row of a word is its port's (`outPortOf`), so Bits' position of a returned output's bit is the bridge's.
* `outRowBit_outRowOf`: that position is the bridge's output-row bit of the unit's computed gate (`outRowOf`).
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Discharge
open FlockSoundness.Discharge.Layout (keyN keyB keyRowOf portRowW RowKey)
open FlockSoundness.Discharge.HiddenExec (outBitRow outBitRow_spec outWordOf outPortOf outPortOf_spec)

variable {c : Flock.Circuit} {n : ℕ}

variable (c) in
/-- **A bridge row bit as Bits' commitment position**: bit `k` of unit `u`'s leaf of row port `q`. -/
def rowPosOf (r : Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) : Spot.Bits.RowPos :=
  ⟨c.ports[r.2.1].name, r.1.val, r.2.2.val⟩

/-- **Bits' leaf is the bridge's leaf map's** at a public file without shared rows (`LeafMap.ident`): unit `u` reads its
own leaf `u` of every port. Bits neither refuses a shared-rows file nor reads its refs: it places unit `u` at leaf `u`
there too, and the refs enter below it, where `Public.digest` reads unit `u`'s row of port `q` at its table's row
`refs[u·np + q]`, which the bridge's leaf map reads (`LeafMap.ofTables`). That file is outside this theorem. -/
theorem rowPosOf_leaf {pub : Flock.Public} (ht : pub.tables = none) (nl : ℕ)
    (r : Fin n × (q : Fin c.ports.size) × Fin (portRowW c q)) :
    (rowPosOf c r).leaf = (LeafMap.ofPublic c pub nl).ref r.2.1.val r.1.val := by
  rw [LeafMap.ofPublic_none nl ht]
  rfl

/-! ## Source gates -/

/-- **A key is its own key** (`Spot.Bits.keyIdx`): the first input of a source has that source. -/
theorem keyIdx_self {B x b : ℕ} (h : Spot.Bits.keyIdx c B x = some b) :
    Spot.Bits.keyIdx c B b = some b := by
  unfold Spot.Bits.keyIdx at h ⊢
  dsimp only at h ⊢
  split_ifs at h with h0
  have hb := List.find?_some h
  simp only [decide_eq_true_eq] at hb
  rw [ite_eq_right (by rw [hb]; exact h0), hb]
  exact h

theorem keyIdx_lt {B x b : ℕ} (h : Spot.Bits.keyIdx c B x = some b) : b < B := by
  unfold Spot.Bits.keyIdx at h
  dsimp only at h
  split_ifs at h
  exact List.mem_range.1 (List.mem_of_find?_eq_some h)

/-- **A column of a unit's own instance on a source gate** `2 ≤ g < 2 + n·B` is an input column of the unit: the
gate's unit index is the unit's, and its input is its own key, with the column's source (`Layout.wire_inGateP` at
`keyProg`). -/
theorem wire_inGate {hU : Built c.unit} (outs : Finset (Fin (keyN c hU n))) {u : Fin n}
    (x : Fin ((Placed.keyProg c hU n).rows u).w)
    (h2 : 2 ≤ ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val)
    (hlt : ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val < 2 + n * keyB c hU) :
    x.val < ((Placed.keyProg c hU n).rows u).nIn ∧
      (((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val - 2) / keyB c hU = u.val ∧
      Spot.Bits.keyIdx c (keyB c hU)
          ((((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val - 2) % keyB c hU) =
        some ((((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val - 2) % keyB c hU) ∧
      Aliased.srcKey c ((((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val - 2) % keyB c hU) =
        Aliased.srcKey c x.val := by
  have e := congrArg FlockSoundness.Rows.nIn
    (Placed.rows_circuitProg c hU n (keyB c hU) (Aliased.keyIdx c (keyB c hU)) u)
  have hx : x.val < ((Placed.keyProg c hU n).rows u).nIn := by
    by_contra hx
    by_cases hone : x = ((Placed.keyProg c hU n).rows u).one
    · rw [hone, Const.Prog.instC_wire_one] at h2
      have h0 : (Const.Prog.oneAt (Placed.keyProg c hU n) u).val = 0 :=
        Layout.oneAt_uniform_val (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
          (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) (one := Placed.oneGate n (keyB c hU))
          (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) n u
      omega
    · have hne : x.val ≠ ((Placed.keyProg c hU n).rows u).nIn + ((Placed.keyProg c hU n).rows u).nComp :=
        fun he => hone (Fin.ext he)
      have hxw := x.isLt
      have hi : x.val - ((Placed.keyProg c hU n).rows u).nIn < ((Placed.keyProg c hU n).rows u).nComp := by
        change x.val < ((Placed.keyProg c hU n).rows u).nIn + ((Placed.keyProg c hU n).rows u).nComp + 1 at hxw
        omega
      have hx' : x = ((Placed.keyProg c hU n).rows u).comp ⟨_, hi⟩ :=
        Fin.ext (by simp only [FlockSoundness.Rows.comp]; omega)
      have hm := (Const.Prog.instC (Placed.keyProg c hU n) outs u).mem ⟨_, hi⟩
      rw [← hx'] at hm
      have hnone :
          (Placed.keyProg c hU n).unit ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x) = none :=
        Layout.unit_uniform_input (k := Placed.inGates n (keyB c hU)) (R := Placed.circuitRows c hU)
          (ins := Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU)))
          (one := Placed.oneGate n (keyB c hU))
          (hone := Placed.insOf_ne_one n (keyB c hU) (Aliased.keyIdx c (keyB c hU))) n _ (by
            show _ < 2 + n * keyB c hU; omega)
      have : (Placed.keyProg c hU n).unit ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x) = some u := hm
      rw [hnone] at this
      cases this
  have hxB : x.val < keyB c hU := lt_of_lt_of_eq hx e
  refine ⟨hx, ?_⟩
  have hv : (Const.Prog.insAt (Placed.keyProg c hU n) u x.val).val =
      (Placed.insOf n (keyB c hU) (Aliased.keyIdx c (keyB c hU)) u.val ⟨x.val, hxB⟩).val :=
    Placed.insAt_uniform n u x.val hxB
  rw [Const.Prog.instC_wire_input _ _ _ x hx, hv,
    Placed.insOf_of_lt (nU := n) (B := keyB c hU) (src := Aliased.keyIdx c (keyB c hU)) u.isLt] at h2 ⊢
  rcases hb : Aliased.keyIdx c (keyB c hU) ⟨x.val, hxB⟩ with _ | b
  · rw [hb] at h2
    exact absurd h2 (by simp [Placed.srcGate, Placed.zeroGate])
  · simp only [Placed.srcGate]
    have hB : 0 < keyB c hU := lt_of_le_of_lt (Nat.zero_le _) hxB
    have hm : (2 + u.val * keyB c hU + b.val - 2) % keyB c hU = b.val := by
      rw [show 2 + u.val * keyB c hU + b.val - 2 = b.val + u.val * keyB c hU by omega,
        Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b.isLt]
    have hd : (2 + u.val * keyB c hU + b.val - 2) / keyB c hU = u.val := by
      rw [show 2 + u.val * keyB c hU + b.val - 2 = b.val + u.val * keyB c hU by omega,
        Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt b.isLt, Nat.zero_add]
    have hk : Spot.Bits.keyIdx c (keyB c hU) x.val = some b.val := by
      rw [keyIdx_eq c hxB, hb]
      rfl
    exact ⟨hd, by rw [hm]; exact keyIdx_self hk, by rw [hm, Layout.keyIdx_some hb]⟩

/-- **Bits' row map at a source gate is the bridge's**: at gate `2 + u·B + b` whose input `b` is its own key (`keyIdx`)
and whose key is the zero or a row bit (`RowKey`), Bits' commitment position is the bridge's row bit (`keyRowOf`); both
are none off the source gates. -/
theorem rowBit_keyRowOf {hU : Built c.unit} (hu : c.unit.unitLog ≤ 32) (g : Fin (keyN c hU n))
    (h : 2 ≤ g.val → g.val < 2 + n * keyB c hU →
      Spot.Bits.keyIdx c (keyB c hU) ((g.val - 2) % keyB c hU) = some ((g.val - 2) % keyB c hU) ∧
        RowKey c (portRowW c) ((g.val - 2) % keyB c hU)) :
    Spot.Bits.rowBit c n g.val = (keyRowOf c hU n (portRowW c) g).map (rowPosOf c) := by
  have hB : ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn = keyB c hU :=
    (rowsAgree_circuitRows c hU hu).1
  unfold Spot.Bits.rowBit keyRowOf
  dsimp only
  rw [hB]
  by_cases hr : 2 ≤ g.val ∧ g.val < 2 + n * keyB c hU
  · obtain ⟨hself, hk⟩ := h hr.1 hr.2
    have hb : keyB c hU ≠ 0 := fun h0 => by rw [h0, Nat.mul_zero] at hr; omega
    rw [dite_eq_left hr, ite_eq_right]
    · unfold RowKey at hk
      rw [srcKey_eq] at hk ⊢
      generalize Spot.Bits.srcKey c (Spot.Bits.Index.of c) ((g.val - 2) % keyB c hU) = s at hk ⊢
      cases s with
      | msg q byte bit =>
        obtain h0 | ⟨q', byte', bit', he, hq, hw, hb8⟩ := hk
        · exact absurd h0 (by simp [srcOf])
        · simp only [srcOf, Aliased.Src.msg.injEq] at he
          obtain ⟨rfl, rfl, rfl⟩ := he
          simp only [srcOf, Spot.Bits.srcPos, hq, hw, hb8, and_self, ↓reduceDIte, Option.map_some, rowPosOf,
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

/-- **At an input gate the bridge's row bit is its key's** (`rowOfH` is `keyRowOf`): an input gate is no unit's, so no
returned output's. -/
theorem rowOfH_of_input {hU : Built c.unit} (outs : Finset (Fin (keyN c hU n))) {g : Fin (keyN c hU n)}
    (hg : (Placed.keyProg c hU n).unit g = none) :
    HiddenExec.rowOfH c hU n outs g = keyRowOf c hU n (portRowW c) g := by
  rcases hk : keyRowOf c hU n (portRowW c) g with _ | r
  · rw [HiddenExec.rowOfH_of_none outs hk]
    unfold HiddenExec.outRowOf
    rw [hg]
  · exact HiddenExec.rowOfH_of_key outs hk

/-- **Bits' row map at a gate a unit reads is the bridge's**: a column of a unit's own instance that one of its rows
reads, in a circuit whose read inputs have row keys (`KeysRow`). -/
theorem rowBit_wire {hU : Built c.unit} (hu : c.unit.unitLog ≤ 32) (hkeys : Layout.KeysRow c hU (portRowW c))
    (outs : Finset (Fin (keyN c hU n))) {u : Fin n} (i : Fin ((Placed.keyProg c hU n).rows u).nComp)
    (x : Fin ((Placed.keyProg c hU n).rows u).w)
    (hx : x ∈ ((Placed.keyProg c hU n).rows u).a i ++ ((Placed.keyProg c hU n).rows u).b i) :
    Spot.Bits.rowBit c n ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val =
      (keyRowOf c hU n (portRowW c) ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x)).map (rowPosOf c) :=
  rowBit_keyRowOf hu _ fun h2 hlt => by
    obtain ⟨hxn, -, hself, hs⟩ := wire_inGate outs x h2 hlt
    refine ⟨hself, ?_⟩
    have hk := Layout.keysRow_keyProg hkeys u i x hx hxn
    unfold RowKey at hk ⊢
    rw [hs]
    exact hk

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
    Spot.Bits.outPos c u x t = some ⟨(c.ports.getD (HmOut.nIn c + outPortOf c x) default).name, u,
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
  unfold Spot.Bits.outPos
  simp only [hf, Option.map_some, Hidden.outGroup_single hr hlt]

/-- **Bits' position of a returned output's bit is the bridge's**: where each output row holds one output port, at an
output leaf `j` of instance `k` whose word is an output word and whose bit `t` is a bit of its row (`outBitRow`). -/
theorem outRowBit_outBitRow (hr : c.outRowPorts = Array.replicate c.outPorts.size 1) (u : Fin n) {k j t : ℕ}
    (hj : j < (c.leavesOut.getD k #[]).size) (hw : outWordOf c (k, j, t) < c.outWords)
    (hb : outBitRow c u (k, j, t) ≠ none) :
    Spot.Bits.outRowBit c u.val k j t = (outBitRow c u (k, j, t)).map (rowPosOf c) := by
  have ew : outWordOf c (k, j, t) = (c.leavesOut.getD k #[])[j] := getElem!_pos _ j hj
  obtain ⟨⟨u', q, b⟩, hs⟩ := Option.ne_none_iff_exists'.1 hb
  obtain ⟨rfl, hq, hk⟩ := outBitRow_spec hs
  rw [ew] at hq hk hw
  rw [hs, Option.map_some]
  unfold Spot.Bits.outRowBit
  rw [Array.getElem?_eq_getElem hj, Option.bind_some, outPos_eq hr hw]
  simp only [rowPosOf, Option.some.injEq, Spot.Bits.RowPos.mk.injEq, ← hk, ← hq, and_true]
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem q.isLt, Option.getD_some]
  rfl

/-- **Bits' position of a returned output's bit is the bridge's output-row bit** of the unit's computed gate on that
column (`outRowOf`), where each output row holds one output port. -/
theorem outRowBit_outRowOf {hU : Built c.unit} (hr : c.outRowPorts = Array.replicate c.outPorts.size 1)
    (outs : Finset (Fin (keyN c hU n))) {u : Fin n} (x : Fin ((Placed.keyProg c hU n).rows u).w)
    (hgu : (Placed.keyProg c hU n).unit ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x) = some u)
    (hxn : ((Placed.keyProg c hU n).rows u).nIn ≤ x.val) {u' j t : ℕ} (hoc : Layout.OutColAt c hU x.val u' j t)
    (hw : outWordOf c (u', j, t) < c.outWords) (hb : outBitRow c u (u', j, t) ≠ none) :
    Spot.Bits.outRowBit c u.val u' j t =
      (HiddenExec.outRowOf c hU n outs ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x)).map
        (rowPosOf c) := by
  rw [HiddenExec.outRowOf_wire outs x hgu hxn, Layout.outDecode_of hoc]
  exact outRowBit_outBitRow hr u hoc.2.1 hw hb

end Flock.Bridge
