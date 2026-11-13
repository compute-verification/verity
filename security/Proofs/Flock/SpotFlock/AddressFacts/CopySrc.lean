module

public import Proofs.Flock.SpotFlock.AddressFacts.CopyOf
public import Proofs.Flock.SpotFlock.AddressFacts.WrittenOf
public import Proofs.Flock.SpotFlock.LayoutAt
public import Proofs.Flock.SpotFlock.WireMapRewires
public import Proofs.Flock.Bridge.BitsRowMap
public import Proofs.Flock.Bridge.BitsKeyProg
public import Proofs.Flock.Soundness.Discharge.Layout.KeysRow

/-!
# The copy at a source bit

`copiesRef_src`: `CopiesAt`'s source half where the second stage's design evaluates and each member names its ports
once. At a source bit of member `t`'s unit `k`, the column `pickAny` chooses copies the bit's port bit (`CopiesRef`'s
input case).

A source bit is Bits' commitment position of a source gate `g = 2 + k·B + b` whose input `b` is its own key
(`rowBit`), the gate's index its position (`posOf_of_lt_inputs`). Unit `k`'s input column `b` is on that gate
(`exists_wire_src`), so `pickAny` chooses a column on it (`pickAny_of_wire`), with no condition that the unit reads it.
Bits' position there is the bridge's row bit (`rowBit_keyRowOf`), whose port and bit the column's key is
(`Layout.keyRowOf_wire`), and with each port named once that port is the address map's (`portNamed_eq`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy FlockSoundness FlockSoundness.Discharge

/-! ## Bits' source positions -/

theorem srcPos_leaf {c : _root_.Flock.Circuit} {u : ℕ} {k : Spot.Bits.Src} {r : Spot.Bits.RowPos}
    (hs : Spot.Bits.srcPos c u k = some r) : r.leaf = u := by
  unfold Spot.Bits.srcPos at hs
  split at hs
  · cases hs
    rfl
  · cases hs

theorem msgAt_bit8 {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {x t : ℕ} {wide : Bool} {p byte bit : ℕ}
    (hm : Spot.Bits.msgAt c ix x t wide = .msg p byte bit) : bit < 8 := by
  unfold Spot.Bits.msgAt at hm
  dsimp only at hm
  split at hm
  · injection hm with hp hb hbit
    subst hbit
    omega
  · cases hm

theorem srcAt_bit8 {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {k p' t p byte bit : ℕ}
    (hs : Spot.Bits.srcAt c ix k p' t = .msg p byte bit) : bit < 8 := by
  unfold Spot.Bits.srcAt at hs
  dsimp only at hs
  split at hs
  · split at hs
    · cases hs
    · exact msgAt_bit8 hs
  · split at hs
    · exact msgAt_bit8 hs
    · split at hs
      · cases hs
      · cases hs

theorem srcKey_bit8 {c : _root_.Flock.Circuit} {ix : Spot.Bits.Index} {x p byte bit : ℕ}
    (hs : Spot.Bits.srcKey c ix x = .msg p byte bit) : bit < 8 := by
  unfold Spot.Bits.srcKey at hs
  dsimp only at hs
  split at hs
  · exact srcAt_bit8 hs
  · cases hs

/-- **What a source gate's commitment position says of the gate**: it is `2 + u·B + b` for a unit `u` below `n`, its
input `b` is its own key, a message bit, and the position's leaf is `u`. -/
theorem rowBit_facts {c : _root_.Flock.Circuit} {n g : ℕ} {r : Spot.Bits.RowPos}
    (hr : Spot.Bits.rowBit c n g = some r) :
    2 ≤ g ∧ 0 < ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn ∧
      g < 2 + n * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn ∧
      Spot.Bits.keyIdx c ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn
          ((g - 2) % ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn) =
        some ((g - 2) % ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn) ∧
      r.leaf = (g - 2) / ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn ∧
      ∃ p byte bit, Spot.Bits.srcKey c (Spot.Bits.Index.of c)
        ((g - 2) % ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn) = .msg p byte bit := by
  unfold Spot.Bits.rowBit at hr
  dsimp only at hr
  split at hr
  · cases hr
  · rename_i hc
    simp only [Bool.or_eq_true, decide_eq_true_eq, beq_iff_eq, bne_iff_ne, ne_eq, not_or, not_not, not_lt,
      not_le] at hc
    obtain ⟨p, byte, bit, hk, -, -⟩ := srcPos_msg hr
    refine ⟨by omega, by omega, by omega, ?_, srcPos_leaf hr, p, byte, bit, hk⟩
    exact hc.2

/-- A source bit is a member's source gate with Bits' commitment position, at the gate's position. -/
theorem sourceBits_mem {h : Held} {x t : ℕ} {r : Spot.Bits.RowPos} (hs : (x, t, r) ∈ sourceBits h.ms) :
    ∃ ht : t < h.ms.length, ∃ g, Spot.Bits.rowBit h.ms[t].1 h.ms[t].2.size g = some r ∧
      x = Spot.Bits.base h.ms t + g := by
  unfold sourceBits at hs
  obtain ⟨m, hm, hx⟩ := List.mem_flatMap.1 hs
  have hm' := List.mem_range.1 hm
  obtain ⟨g, -, hgx⟩ := List.mem_filterMap.1 hx
  have hci : h.ms.getD m default = h.ms[m] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm', Option.getD_some]
  rw [hci] at hgx
  cases hκ : Spot.Bits.consts h.ms[m].1 h.ms[m].2.size g with
  | some b =>
    rw [hκ] at hgx
    simp at hgx
  | none =>
    rw [hκ] at hgx
    have hgx' : (Spot.Bits.rowBit h.ms[m].1 h.ms[m].2.size g).map (fun r => (Spot.Bits.base h.ms m + g, m, r)) =
        some (x, t, r) := hgx
    obtain ⟨r', hr', he⟩ := Option.map_eq_some_iff.1 hgx'
    have h1 : Spot.Bits.base h.ms m + g = x := congrArg (fun p => p.1) he
    have h2 : m = t := congrArg (fun p => p.2.1) he
    have h3 : r' = r := congrArg (fun p => p.2.2) he
    subst h2 h3
    exact ⟨hm', g, hr', h1.symm⟩

/-! ## Unit columns on source gates -/

/-- **A unit's input column is on its own source gate**: input `b`, its own key, of unit `u` reads gate
`2 + u·B + b`. -/
theorem exists_wire_src {c : _root_.Flock.Circuit} {hU : FlockSoundness.Built c.unit} {n : ℕ}
    (outs : Finset (Fin (Layout.keyN c hU n))) (u : Fin n) {b : ℕ} (hb : b < Layout.keyB c hU)
    (hself : Aliased.keyIdx c (Layout.keyB c hU) ⟨b, hb⟩ = some ⟨b, hb⟩) :
    ∃ x : Fin ((Placed.keyProg c hU n).rows u).w,
      ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val = 2 + u.val * Layout.keyB c hU + b := by
  have e := congrArg FlockSoundness.Rows.nIn
    (Placed.rows_circuitProg c hU n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)) u)
  have hx : b < ((Placed.keyProg c hU n).rows u).nIn := by
    rw [e]
    exact hb
  have hw : b < ((Placed.keyProg c hU n).rows u).w := by
    show b < ((Placed.keyProg c hU n).rows u).nIn + ((Placed.keyProg c hU n).rows u).nComp + 1
    omega
  refine ⟨⟨b, hw⟩, ?_⟩
  have hv : (Const.Prog.insAt (Placed.keyProg c hU n) u b).val =
      (Placed.insOf n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)) u.val ⟨b, hb⟩).val :=
    Placed.insAt_uniform n u b hb
  rw [Const.Prog.instC_wire_input _ _ _ ⟨b, hw⟩ hx, hv,
    Placed.insOf_of_lt (nU := n) (B := Layout.keyB c hU) (src := Aliased.keyIdx c (Layout.keyB c hU)) u.isLt, hself]
  simp only [Placed.srcGate]

variable {h : Held}

/-- **`pickAny` chooses a column on a wire wherever the unit has one**: `picks_pickAny` from the column alone. -/
theorem pickAny_of_wire {t : ℕ} (ht : t < h.ms.length) (hU : FlockSoundness.Built h.ms[t].1.unit) {k : ℕ}
    (hk : k < h.ms[t].2.size) (y : Fin (keyC h t ht hU).N)
    (hy : ∃ cc : Fin ((Placed.keyProg h.ms[t].1 hU h.ms[t].2.size).rows ⟨k, hk⟩).w,
      (Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
        (Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨k, hk⟩).wire cc = y) :
    ∃ cc : Fin ((Placed.keyProg h.ms[t].1 hU h.ms[t].2.size).rows ⟨k, hk⟩).w,
      (Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
        (Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨k, hk⟩).wire cc = y ∧
      pickAny h t k (Spot.Bits.base h.ms t + _root_.Flock.Bridge.posOf (keyC h t ht hU) y) = cc.val := by
  classical
  have hex : ∃ y', Spot.Bits.base h.ms t + _root_.Flock.Bridge.posOf (keyC h t ht hU) y' =
      Spot.Bits.base h.ms t + _root_.Flock.Bridge.posOf (keyC h t ht hU) y := ⟨y, rfl⟩
  have hyc : hex.choose = y := _root_.Flock.Bridge.posOf_injective (Nat.add_left_cancel hex.choose_spec)
  have hc : ∃ cc : Fin ((Placed.keyProg h.ms[t].1 hU h.ms[t].2.size).rows ⟨k, hk⟩).w,
      (Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
        (Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨k, hk⟩).wire cc = hex.choose := by
    rw [hyc]
    exact hy
  refine ⟨hc.choose, hc.choose_spec.trans hyc, ?_⟩
  unfold pickAny
  rw [dite_eq_left ht, dite_eq_left hU, dite_eq_left hk, dite_eq_left hex, dite_eq_left hc]

/-! ## The copy -/

/-- **`CopiesAt`'s source half**: where the design evaluates and each member names its ports once, at a source bit of
member `t`'s unit `k` the column `pickAny` chooses has the bit's port bit as its key. -/
theorem copiesRef_src {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) (hO : PortsOnce h)
    {i t k : ℕ} {rp : Spot.Bits.RowPos} (hs : (i, t, rp) ∈ sourceBits h.ms) (hleaf : rp.leaf = k) :
    ∃ m, h.ms[t]? = some m ∧ CopiesRef m.1 (pickAny h t k i) (portNamed h t rp.row) rp.bit := by
  subst hleaf
  obtain ⟨ht, g, hg, rfl⟩ := sourceBits_mem hs
  obtain ⟨-, hU, hu⟩ := templateOk_facts h (templateOk_of_two h hd ht) ht
  refine ⟨h.ms[t], List.getElem?_eq_getElem ht, ?_⟩
  have hm : memberCircuit h t = h.ms[t].1 := (by simp [Held.ms] : h.ms[t].1 = stmtOf (h.xS₀ t)).symm
  have hR := _root_.Flock.Bridge.rowsAgree_circuitRows h.ms[t].1 hU hu
  have hB : ((Spot.Bits.Rows.ofNet h.ms[t].1.unit).stack h.ms[t].1.upv).nIn = Layout.keyB h.ms[t].1 hU := hR.1
  obtain ⟨h2, hB0, hlt, hself, hleaf', p, byte, bit, hkey⟩ := rowBit_facts hg
  rw [hB] at hB0 hlt hself hleaf' hkey
  have hk : rp.leaf < h.ms[t].2.size := by
    rw [hleaf']
    exact (Nat.div_lt_iff_lt_mul hB0).2 (by omega)
  have hb0 : (g - 2) % Layout.keyB h.ms[t].1 hU < Layout.keyB h.ms[t].1 hU := Nat.mod_lt _ hB0
  have hselfA : Aliased.keyIdx h.ms[t].1 (Layout.keyB h.ms[t].1 hU) ⟨_, hb0⟩ = some ⟨_, hb0⟩ := by
    have hs' := hself
    rw [_root_.Flock.Bridge.keyIdx_eq _ hb0] at hs'
    obtain ⟨z, hz, hzv⟩ := Option.map_eq_some_iff.1 hs'
    rw [hz]
    exact congrArg some (Fin.ext hzv)
  have hg2 : 2 + rp.leaf * Layout.keyB h.ms[t].1 hU + (g - 2) % Layout.keyB h.ms[t].1 hU = g := by
    rw [hleaf']
    have := Nat.div_add_mod' (g - 2) (Layout.keyB h.ms[t].1 hU)
    omega
  have hB' : (_root_.Flock.Bridge.bitsRows h.ms[t].1).nIn = Layout.keyB h.ms[t].1 hU := hR.1
  have hgN : g < Layout.keyN h.ms[t].1 hU h.ms[t].2.size := by
    rw [_root_.Flock.Bridge.keyN_eq hR, hB']
    omega
  let y : Fin (keyC h t ht hU).N := ⟨g, hgN⟩
  have hpos : _root_.Flock.Bridge.posOf (keyC h t ht hU) y = g := by
    rw [_root_.Flock.Bridge.posOf_keyProg (c := h.ms[t].1) (n := h.ms[t].2.size) (hU := hU) hu y]
    apply _root_.Flock.Bridge.Members.posOf_of_lt_inputs
    show g < 2 + h.ms[t].2.size * ((Spot.Bits.Rows.ofNet h.ms[t].1.unit).stack h.ms[t].1.upv).nIn
    rw [hB]
    exact hlt
  obtain ⟨x0, hx0⟩ := exists_wire_src (Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨rp.leaf, hk⟩ hb0 hselfA
  have hx0' : ((Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
      (Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨rp.leaf, hk⟩).wire x0).val =
      2 + rp.leaf * Layout.keyB h.ms[t].1 hU + (g - 2) % Layout.keyB h.ms[t].1 hU := hx0
  have hwy : (Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
      (Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨rp.leaf, hk⟩).wire x0 = y := Fin.ext (hx0'.trans hg2)
  obtain ⟨cc, hcc, hpick⟩ := pickAny_of_wire ht hU hk y ⟨x0, hwy⟩
  have hp : p < h.ms[t].1.ports.size := by
    have hpb := srcKey_bits hkey
    by_contra hq
    have h0 : h.ms[t].1.ports.getD p default = default := by simp [Array.getD, hq]
    rw [h0] at hpb
    exact Nat.not_lt_zero _ hpb
  have hkeyR : Layout.RowKey h.ms[t].1 (Layout.portRowW h.ms[t].1) ((g - 2) % Layout.keyB h.ms[t].1 hU) := by
    refine Or.inr ⟨p, byte, bit, ?_, hp, ?_, srcKey_bit8 hkey⟩
    · rw [_root_.Flock.Bridge.srcKey_eq, hkey]
      rfl
    · have hpb := srcKey_bits hkey
      have h16 := bits_of_two hd ht (p := p) (by rw [hm]; exact hp)
      rw [hm, getElem!_pos h.ms[t].1.ports p hp] at h16
      have hgd : h.ms[t].1.ports.getD p default = h.ms[t].1.ports[p] := by simp [Array.getD, hp]
      rw [hgd, h16] at hpb
      show 8 * byte + bit < 16 * h.ms[t].1.ports[p].words
      exact hpb
  have hrk := _root_.Flock.Bridge.rowBit_keyRowOf (n := h.ms[t].2.size) (hU := hU) hu y (fun _ _ => ⟨hself, hkeyR⟩)
  have hrk' : (Layout.keyRowOf h.ms[t].1 hU h.ms[t].2.size (Layout.portRowW h.ms[t].1) y).map
      (_root_.Flock.Bridge.rowPosOf h.ms[t].1) = some rp := hrk.symm.trans hg
  obtain ⟨⟨u', q, b⟩, hr0, hrp⟩ := Option.map_eq_some_iff.1 hrk'
  rw [← hcc] at hr0
  obtain ⟨-, hccn, hsk⟩ := Layout.keyRowOf_wire (Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) cc hr0
  have hpn : portNamed h t rp.row = q.val := by
    refine portNamed_eq (hO t ht) (by rw [hm]; exact q.isLt) ?_
    rw [hm, getElem!_pos h.ms[t].1.ports q.val q.isLt, ← hrp]
    rfl
  have hbit : rp.bit = b.val := by
    rw [← hrp]
    rfl
  rw [show Spot.Bits.base h.ms t + g = Spot.Bits.base h.ms t + _root_.Flock.Bridge.posOf (keyC h t ht hU) y by
    rw [hpos], hpick, hpn, hbit]
  refine Or.inl ⟨?_, q.isLt, hsk⟩
  have e := congrArg FlockSoundness.Rows.nIn
    (Placed.rows_circuitProg h.ms[t].1 hU h.ms[t].2.size (Layout.keyB h.ms[t].1 hU)
      (Aliased.keyIdx h.ms[t].1 (Layout.keyB h.ms[t].1 hU)) ⟨rp.leaf, hk⟩)
  rw [e] at hccn
  have e2 : (Placed.circuitRows h.ms[t].1 hU).nIn = h.ms[t].1.upv * (h.ms[t].1.unit.inWords * _root_.Flock.WORD) := by
    show h.ms[t].1.upv * (FlockSoundness.Rows.ofNet hU).nIn = _
    rw [Layout.ofNet_nIn_eq (c := h.ms[t].1) (hU := hU)]
  exact lt_of_lt_of_eq hccn e2

end SpotFlock
