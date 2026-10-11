import Proofs.Flock.Recursive.Stage
import Proofs.Flock.VBridge.InnerUnit
import Proofs.Flock.VBridge.Keyed
import Proofs.Flock.VBridge.PartS
import Flock.RecAlgebra

/-!
# VBridge G item 3: V*'s port columns, decoded from the registered reads

At an accepted outer proof every layer is `Good` (`ZkOuter.Good`): no wrong unit, and each wire a unit reads at a
registered read carries the registrant's bit there (`RegAgree`). This file reads those bits through the session's layout
(`layShHJ`) into the columns of a copy of V*'s unit, so that the unit theorems (`unit_innerUnit`, `keyProg_recOpen`) apply
to the layer itself. Only `RegAgree` is read: the unit theorems hold at any transcript's local evaluation.

* `fill`: the layer with every wire at a registered read set to the registrant's bit. It agrees with the layer on every
  unit's wires (`fill_io`), so every column a copy computes, its constant included, is the layer's (`copyVal_fill`,
  `fill_one`). An input column no row reads is no wire of the unit, so `RegAgree` says nothing of it in the layer; in
  `fill` it carries the registrant's bit.
* The layout (`lay_pos`, `lay_bit`): wire `g`'s commit string is `posK`'s and its bit `bitK`'s, so `fill` at bit `k` of
  unit `u`'s row `q` (`keyRowOf`) is the registrant's value there (`fill_row`). A copy's input column whose Δ key is
  message bit `k` of row port `q` (`Aliased.srcKey`) is on that gate (`keyRowOf_keyInst`).
* `RegBit u j col b`: copy `j`'s input column `col` of unit `u` has the key of bit `k` of a row `q` that is a registered
  read, and the registrant's value of `u`'s row `q` has `b` at bit `k`. Then the copy's column holds `b` in `fill`
  (`copyVal_regBit`), so `InCols` and `ColsHold` hold there for the registrant's values (`inCols_fill`, `colsHold_fill`).
* `decode_innerUnit`, `decode_recOpen`: at a layer agreeing with the registrant and carrying 1 on the unit's constant,
  the unit theorems' outputs hold in the layer's own columns, from `RegBit` at the unit's input columns.
* `decode_part`: `decode_innerUnit` at a part of S (`PartS.res_partS`): the part's ports read the whole `w` at `wI` and
  the verifier's coefficients `vOf` (`verifierRows`' first component, `vOf_eq`) at `V`, as `res_partS`' `hw`, `hv` ask.

Which columns of V*'s units are registered reads, and the registrant's values there, are facts about the concrete V*
statement; they enter as `RegBit` hypotheses about one unit.
-/

namespace FlockVBridge.Decode

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.OfVStar

/-! ## The layer at the registrant's bits -/

section Fill

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

open Classical in
/-- **The layer `bits` at the registrant's bits**: every wire at a registered read carries the registrant's bit there. -/
noncomputable def fill (bits : Fin (Layout.keyN c hU nv) → Bool) : Fin (Layout.keyN c hU nv) → Bool := fun g =>
  if zo.rd.isRead (zo.lay.pos g) then zo.lay.bit (zo.rd.wv (zo.lay.pos g)) g else bits g

theorem fill_read {bits : Fin (Layout.keyN c hU nv) → Bool} {g : Fin (Layout.keyN c hU nv)}
    (h : zo.rd.isRead (zo.lay.pos g)) : fill zo bits g = zo.lay.bit (zo.rd.wv (zo.lay.pos g)) g := by
  unfold fill
  exact ite_eq_left h

/-- `fill` is the layer on every unit's wires, where the layer agrees with the registrant. -/
theorem fill_io {bits : Fin (Layout.keyN c hU nv) → Bool} (hA : zo.RegAgree bits) {u : Fin nv}
    {g : Fin (Layout.keyN c hU nv)} (hg : g ∈ ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u) :
    fill zo bits g = bits g := by
  unfold fill
  split_ifs with h
  · exact (hA u g hg h).symm
  · rfl

/-- A copy's column past its inputs is a computed row of the stack. -/
theorem stackCol_comp (R : Rows) (k : ℕ) {j : ℕ} (hj : j < k) (x : Fin R.w) (hx : R.nIn ≤ x.val) :
    ∃ i, R.stackCol k hj x = (R.stack k).comp i := by
  by_cases h2 : x.val < R.nIn + R.nComp
  · refine ⟨⟨j * (R.nComp + 1) + (1 + (x.val - R.nIn)), Rows.mul_add_lt hj (by omega)⟩, ?_⟩
    rw [stack_comp R k hj ⟨x.val - R.nIn, by omega⟩]
    congr 1
    apply Fin.ext
    show x.val = R.nIn + (x.val - R.nIn)
    omega
  · refine ⟨⟨j * (R.nComp + 1) + 0, Rows.mul_add_lt hj (by omega)⟩, ?_⟩
    rw [stack_comp_one R k hj]
    congr 1
    apply Fin.ext
    have hlt := x.isLt
    simp only [Rows.w] at hlt
    show x.val = R.nIn + R.nComp
    omega

/-- **`fill` computes what the layer computes**: a copy's columns past its inputs are the layer's. -/
theorem copyVal_fill {bits : Fin (Layout.keyN c hU nv) → Bool} (hA : zo.RegAgree bits) {u : Fin nv} {R : Rows} {k : ℕ}
    (I : ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).IsRowsUnit u (R.stack k)) {j : ℕ} (hj : j < k)
    {col : ℕ} (hcol : R.nIn ≤ col) : copyVal I (fill zo bits) hj col = copyVal I bits hj col := by
  unfold copyVal colVal
  split_ifs with hw
  · obtain ⟨i, hi⟩ := stackCol_comp R k hj ⟨col, hw⟩ hcol
    have key : ∀ x, x = (R.stack k).comp i →
        ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).localEval (fill zo bits) u (I.wire x) =
          ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).localEval bits u (I.wire x) := by
      rintro x rfl
      exact Partition.localEval_congr _ (fun g hg => fill_io zo hA hg) _ _ rfl (I.mem i)
    exact key _ hi
  · rfl

/-- **`fill` carries the layer's constant**: every copy's constant row reads it, so it is a wire of the unit. -/
theorem fill_one {bits : Fin (Layout.keyN c hU nv) → Bool} (hA : zo.RegAgree bits) {u : Fin nv} {R : Rows} {k : ℕ}
    (I : ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).IsRowsUnit u (R.stack k)) {j : ℕ} (hj : j < k) :
    fill zo bits (I.wire (R.stack k).one) = bits (I.wire (R.stack k).one) := by
  refine fill_io zo hA (I.mem_io_of_read (i := ⟨j * (R.nComp + 1) + 0, Rows.mul_add_lt hj (by omega)⟩) ?_ ?_)
  · rw [stack_a_one R k hj]
    simp
  · refine I.outside _ fun i h => ?_
    have := congrArg Fin.val h
    have hi := i.isLt
    simp only [Rows.one, Rows.comp] at this
    omega

/-! ## The layout -/

/-- The session's commit string of each wire is `posK`'s. -/
theorem lay_pos (g : Fin (Layout.keyN c hU nv)) : zo.lay.pos g = posK c hU nv g := by
  dsimp only [ZkOuter.lay, ZkHidden.layShHJ, HiddenExec.keyLayoutH, Layout.Readers.layout, Layout.ofBits,
    HiddenExec.keyReadersH]
  unfold posK
  rcases h : HiddenExec.rowOfH c hU nv (Layout.outsOf c hU nv) g with _ | ⟨u, q, k⟩
  · exact Layout.ofRows_pos_none _ h
  · exact Layout.ofRows_pos_some _ h

/-- The session's bit of each wire is `bitK`'s. -/
theorem lay_bit (v : ValK c) (g : Fin (Layout.keyN c hU nv)) : zo.lay.bit v g = bitK c hU nv v g := by
  dsimp only [ZkOuter.lay, ZkHidden.layShHJ, HiddenExec.keyLayoutH, Layout.Readers.layout, Layout.ofBits,
    HiddenExec.keyReadersH, Layout.Readers.ofRows]
  unfold bitK idxK
  rcases HiddenExec.rowOfH c hU nv (Layout.outsOf c hU nv) g with _ | ⟨u, q, k⟩ <;> rfl

end Fill

section Keys

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ}

/-- **An input of a unit's own instance with a row bit's key is that row bit**: when input `x`'s Δ key is message bit
`k` of row port `q`, its gate is bit `k` of unit `u`'s row `q` (`keyRowOf`). -/
theorem keyRowOf_instC (outs : Finset (Fin (Layout.keyN c hU n))) (u : Fin n)
    (x : Fin ((Placed.keyProg c hU n).rows u).w) (hx : x.val < Layout.keyB c hU) {q : Fin c.ports.size} {k : ℕ}
    (hk : k < Layout.portRowW c q) (hkey : Aliased.srcKey c x.val = .msg q.val (k / 8) (k % 8)) :
    Layout.keyRowOf c hU n (Layout.portRowW c) ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x) =
      some (u, ⟨q, ⟨k, hk⟩⟩) := by
  have e := congrArg FlockSoundness.Rows.nIn
    (Placed.rows_circuitProg c hU n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)) u)
  have hxn : x.val < ((Placed.keyProg c hU n).rows u).nIn := lt_of_lt_of_eq hx e.symm
  obtain ⟨b, hb⟩ : ∃ b, Aliased.keyIdx c (Layout.keyB c hU) ⟨x.val, hx⟩ = some b := by
    rcases h : Aliased.keyIdx c (Layout.keyB c hU) ⟨x.val, hx⟩ with _ | b
    · rw [Aliased.keyIdx_eq_none] at h
      rw [h] at hkey
      cases hkey
    · exact ⟨b, rfl⟩
  have hv : (Const.Prog.insAt (Placed.keyProg c hU n) u x.val).val =
      (Placed.insOf n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)) u.val ⟨x.val, hx⟩).val :=
    Placed.insAt_uniform n u x.val hx
  have hval : ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val = 2 + u.val * Layout.keyB c hU + b.val := by
    rw [Const.Prog.instC_wire_input _ _ _ x hxn, hv,
      Placed.insOf_of_lt (nU := n) (B := Layout.keyB c hU) (src := Aliased.keyIdx c (Layout.keyB c hU)) u.isLt, hb]
    rfl
  have hB : 0 < Layout.keyB c hU := lt_of_le_of_lt (Nat.zero_le _) hx
  have hm : (((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val - 2) % Layout.keyB c hU = b.val := by
    rw [hval, show 2 + u.val * Layout.keyB c hU + b.val - 2 = b.val + u.val * Layout.keyB c hU by omega,
      Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt b.isLt]
  have hun : (u.val + 1) * Layout.keyB c hU ≤ n * Layout.keyB c hU := Nat.mul_le_mul_right _ u.isLt
  rw [Nat.succ_mul] at hun
  have h2 : 2 ≤ ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val := by rw [hval]; omega
  have hlt : ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val < 2 + n * Layout.keyB c hU := by
    rw [hval]
    have := b.isLt
    omega
  have hkey' : Aliased.srcKey c ((((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x).val - 2) %
      Layout.keyB c hU) = .msg q.val (k / 8) (k % 8) := by
    rw [hm, Layout.keyIdx_some hb]
    exact hkey
  have hw : 8 * (k / 8) + k % 8 < Layout.portRowW c q := by
    have := Nat.div_add_mod k 8
    omega
  rcases hr : Layout.keyRowOf c hU n (Layout.portRowW c) ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire x)
    with _ | ⟨u', q', k'⟩
  · exact absurd hr (Layout.keyRowOf_ne_none h2 hlt hkey' q.isLt hw (Nat.mod_lt _ (by norm_num)))
  · obtain ⟨hu, -, hkey2⟩ := Layout.keyRowOf_wire outs x hr
    rw [hkey] at hkey2
    injection hkey2 with hq hb1 hb2
    subst hu
    obtain rfl : q' = q := Fin.ext hq.symm
    obtain rfl : k' = ⟨k, hk⟩ := Fin.ext (by
      have := Nat.div_add_mod k 8
      have := Nat.div_add_mod k'.val 8
      show k'.val = k
      omega)
    rfl

/-- **`keyInst`'s input with a row bit's key is that row bit** (`keyRowOf_instC`). -/
theorem keyRowOf_keyInst (outs : Finset (Fin (Layout.keyN c hU n))) (u : Fin n)
    (x : Fin ((Rows.ofNet hU).stack c.upv).w) (hx : x.val < ((Rows.ofNet hU).stack c.upv).nIn)
    {q : Fin c.ports.size} {k : ℕ} (hk : k < Layout.portRowW c q)
    (hkey : Aliased.srcKey c x.val = .msg q.val (k / 8) (k % 8)) :
    Layout.keyRowOf c hU n (Layout.portRowW c) ((keyInst hU n outs u).wire x) = some (u, ⟨q, ⟨k, hk⟩⟩) := by
  have hw := Const.Prog.castW_wire
    (Placed.rows_circuitProg c hU n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)) u)
    (Const.Prog.rowsWireC (Placed.keyProg c hU n) u) x
  change (keyInst hU n outs u).wire x = (Const.Prog.instC (Placed.keyProg c hU n) outs u).wire _ at hw
  rw [hw]
  exact keyRowOf_instC outs u _ hx hk hkey

end Keys

/-! ## A copy's input columns at the registrant's values -/

section Reg

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- The registered reads, at V*'s commit strings. -/
def regRead (p : PosK c hU nv) : Prop := zo.rd.isRead p

/-- The registrant's values, at V*'s commit strings. -/
noncomputable def regVal (p : PosK c hU nv) : ValK c := zo.rd.wv p

/-- **Copy `j`'s input column `col` of unit `u` is a registered bit `b`**: its Δ key (`Aliased.srcKey` at the stacked
input `j · nIn + col`) is bit `k` of row port `q`, unit `u`'s row `q` is a registered read, and the registrant's value
there has `b` at bit `k`. -/
def RegBit (u : Fin nv) (j col : ℕ) (b : Bool) : Prop :=
  ∃ (_ : col < (Rows.ofNet hU).nIn) (q : Fin c.ports.size) (k : ℕ) (hk : k < Layout.portRowW c q),
    Aliased.srcKey c (j * (Rows.ofNet hU).nIn + col) = .msg q.val (k / 8) (k % 8) ∧
      regRead zo (.inl (u, q)) ∧
      (regVal zo (.inl (u, q))).2 ⟨k, lt_of_lt_of_le hk ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ = b

/-- **`fill` at a registered row bit is the registrant's**: at bit `k` of unit `u`'s row `q` (`keyRowOf`), a registered
read, `fill` is the registrant's value of the row at bit `k`. -/
theorem fill_row (bits : Fin (Layout.keyN c hU nv) → Bool) {g : Fin (Layout.keyN c hU nv)} {u : Fin nv}
    {q : Fin c.ports.size} {k : ℕ} {hk : k < Layout.portRowW c q}
    (hr : Layout.keyRowOf c hU nv (Layout.portRowW c) g = some (u, ⟨q, ⟨k, hk⟩⟩)) (hread : regRead zo (.inl (u, q))) :
    fill zo bits g =
      (regVal zo (.inl (u, q))).2 ⟨k, lt_of_lt_of_le hk ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ := by
  have hr' := HiddenExec.rowOfH_of_key (Layout.outsOf c hU nv) hr
  have hp : posK c hU nv g = .inl (u, q) := by
    unfold posK
    rw [hr']
    try rfl
  have hread' : zo.rd.isRead (zo.lay.pos g) := by
    rw [lay_pos zo, hp]
    exact hread
  rw [fill_read zo hread']
  refine (lay_bit zo _ g).trans ?_
  have e : bitK c hU nv (zo.rd.wv (zo.lay.pos g)) g = bitK c hU nv (regVal zo (.inl (u, q))) g := by
    show bitK c hU nv (regVal zo (zo.lay.pos g)) g = _
    rw [lay_pos zo g, hp]
  refine e.trans ?_
  unfold bitK idxK
  rw [hr']
  try rfl

/-- **A registered bit is the copy's column in `fill`.** -/
theorem copyVal_regBit (bits : Fin (Layout.keyN c hU nv) → Bool) {u : Fin nv} {j : ℕ} (hj : j < c.upv) {col : ℕ}
    {b : Bool} (h : RegBit zo u j col b) :
    copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) (fill zo bits) hj col = b := by
  obtain ⟨hcol, q, k, hk, hkey, hread, hval⟩ := h
  have hx : ((Rows.ofNet hU).stackCol c.upv hj ⟨col, by simp only [Rows.w]; omega⟩).val =
      j * (Rows.ofNet hU).nIn + col := by
    rw [Rows.stackCol, dite_eq_left hcol]
  have hr := keyRowOf_keyInst (Layout.outsOf c hU nv) u
    ((Rows.ofNet hU).stackCol c.upv hj ⟨col, by simp only [Rows.w]; omega⟩)
    (by rw [hx]; exact Rows.mul_add_lt hj hcol) hk (by rw [hx]; exact hkey)
  exact (copyVal_input _ _ hj hcol).trans ((fill_row zo bits hr hread).trans hval)

/-- **`InCols` at the registrant's values**: in `fill`, copy `j`'s columns hold `w` on the row ports and `vv` on `v`'s
port, where those columns are registered bits of `w`'s and `vv`'s. -/
theorem inCols_fill (bits : Fin (Layout.keyN c hU nv) → Bool) {u : Fin nv} {j : ℕ} (hj : j < c.upv) (S : VStar.RepS)
    (w vv : ℕ → Flock.F128) (hw : ∀ q < 128 * S.gf.nw, RegBit zo u j (rowCol S.ports 0 q) (wbit w q))
    (hv : ∀ q < 128 * S.gf.nv, RegBit zo u j (vBase S + q) (wbit vv q)) :
    InCols S w vv (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) (fill zo bits) hj) :=
  ⟨fun q hq => copyVal_regBit zo bits hj (hw q hq), fun q hq => copyVal_regBit zo bits hj (hv q hq)⟩

/-- **`ColsHold` at the registrant's values**: in `fill`, copy `j`'s columns from `base` hold the bytes `B`, where those
columns are registered bits of `B`'s. -/
theorem colsHold_fill (bits : Fin (Layout.keyN c hU nv) → Bool) {u : Fin nv} {j : ℕ} (hj : j < c.upv) (base : ℕ)
    (B : ByteArray) (h : ∀ t < 8 * B.size, RegBit zo u j (base + t) (Hm.bitL B.data.toList t)) :
    ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) (fill zo bits) hj) base B :=
  fun t ht => copyVal_regBit zo bits hj (h t ht)

/-! ## The units' outputs in the layer -/

theorem inRows_le_outStart (st : Flock.HmNets.St) (inBits outBits : Array ℕ) :
    Hm.inRowsOf inBits ≤ Hm.outStart st inBits outBits := by
  have := Flock.HmNets.le_roundUp (Hm.inRowsOf inBits + st.rows.length) (Hm.spanOf outBits)
    (Flock.HmNets.groupWords_pos _)
  unfold Hm.outStart
  omega

/-- Output columns from `base` past the inputs hold in the layer what they hold in `fill`. -/
theorem colsHold_of_fill {bits : Fin (Layout.keyN c hU nv) → Bool} (hA : zo.RegAgree bits) {u : Fin nv} {j : ℕ}
    (hj : j < c.upv) {base : ℕ} (hbase : (Rows.ofNet hU).nIn ≤ base) {B : ByteArray}
    (h : ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) (fill zo bits) hj) base B) :
    ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj) base B := fun t ht =>
  (copyVal_fill zo hA _ hj (by omega)).symm.trans (h t ht)

/-- **V*'s `InnerRepCheck` unit computes its residuals in a layer agreeing with the registrant** (`unit_innerUnit` at
`fill`). Let the verifier build `c`'s unit net from the layout of `VStar.innerUnit S name` (`netOf`), and let the layer
`bits` agree with the registrant (`RegAgree`, half of `Good`) and carry 1 on unit `u`'s constant gate. If copy `j`'s
input columns are registered bits of `w` on the row ports and of `vv` on `v`'s port (`RegBit`), `w`, `vv` zero past `nw`,
`nv` elements, the copy's output columns in `bits` hold each residual's value (`ResCols`). -/
theorem decode_innerUnit {bits : Fin (Layout.keyN c hU nv) → Bool} (hA : zo.RegAgree bits) (u : Fin nv) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nv) u) = true)
    {S : VStar.RepS} {name : String} {U : VStar.UnitC} (hUn : VStar.innerUnit S name = .ok U)
    (hS : ∀ r ∈ S.gf.res, OpsInRange S.gf r) {l : Flock.HmNets.Laid} (hl : U.laid = .ok l) {name' : String}
    (hn : Flock.HmNets.netOf name' l = .ok c.unit) (hu : c.unit.unitLog ≤ 32) (w vv : ℕ → Flock.F128)
    (hw0 : ∀ e, S.gf.nw ≤ e → w e = 0) (hv0 : ∀ e, S.gf.nv ≤ e → vv e = 0)
    (hw : ∀ q < 128 * S.gf.nw, RegBit zo u j (rowCol S.ports 0 q) (wbit w q))
    (hv : ∀ q < 128 * S.gf.nv, RegBit zo u j (vBase S + q) (wbit vv q)) :
    ResCols S.gf w vv (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj)
      (Hm.outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16)) := by
  have h1 : fill zo bits ((keyInst hU nv (Layout.outsOf c hU nv) u).wire ((Rows.ofNet hU).stack c.upv).one) = true := by
    rw [fill_one zo hA _ hj, keyInst_one]
    exact hone
  obtain ⟨hr, hz⟩ := unit_innerUnit hU (keyInst hU nv (Layout.outsOf c hU nv) u) (fill zo bits) hj h1 hUn hS hl hn hu
    w vv hw0 hv0 (inCols_fill zo bits hj S w vv hw hv)
  have hbase : (Rows.ofNet hU).nIn ≤
      Hm.outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16) := by
    rw [Layout.ofNet_nIn_eq, inWords_laid hl hn]
    exact inRows_le_outStart _ _ _
  refine ⟨fun r hr' k hk => ?_, fun x hx1 hx2 => ?_⟩
  · exact (copyVal_fill zo hA _ hj (by omega)).symm.trans (hr r hr' k hk)
  · exact (copyVal_fill zo hA _ hj (by omega)).symm.trans (hz x hx1 hx2)

/-- **V*'s laid-out opening computes the climb and the consistency term in a layer agreeing with the registrant**
(`keyProg_recOpen` at `fill`): when copy `j`'s input columns are registered bits of the row, the siblings, `acc`, the coefficients and the
position's bits (`RegBit`), the copy's output columns in `bits` hold `laid_recOpen`'s outputs. -/
theorem decode_recOpen {bits : Fin (Layout.keyN c hU nv) → Bool} (hA : zo.RegAgree bits) (u : Fin nv) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nv) u) = true)
    (ms : Flock.MerkleScheme) (hms : ∀ l r, ms.node l r = Flock.Sha512.hash (l ++ r)) (LANES H : ℕ) (hH : H ≤ 32)
    (p : Ports) (outBits : Array ℕ) {name : String}
    (hl : Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit)
    (hu : c.unit.unitLog ≤ 32) (hout : (recOpenOuts LANES H p).length ≤ outBits.foldl (· + ·) 0)
    (r a cf : Array Flock.F128) (rest : ByteArray) (idx : ℕ) (ss : List ByteArray)
    (hr : r.size = LANES) (ha : a.size = 2) (hss : ss.length = H)
    (hrowN : p.row = 8 * (Flock.MerkleScheme.rowBytes r).size) (hauxN : 512 * H ≤ p.aux)
    (haccN : p.acc = 8 * (Flock.MerkleScheme.rowBytes a ++ rest).size)
    (hcoefN : p.coef = 8 * (Flock.MerkleScheme.rowBytes cf).size) (hdirsN : H ≤ p.dirs)
    (hrow : ∀ t < 8 * (Flock.MerkleScheme.rowBytes r).size,
      RegBit zo u j (0 + t) (Hm.bitL (Flock.MerkleScheme.rowBytes r).data.toList t))
    (haux : ∀ i (hi : i < ss.length), ss[i].size = 64 ∧
      ∀ t < 8 * ss[i].size, RegBit zo u j (p.row + 512 * i + t) (Hm.bitL ss[i].data.toList t))
    (hacc : ∀ t < 8 * (Flock.MerkleScheme.rowBytes a ++ rest).size,
      RegBit zo u j (p.row + p.aux + t) (Hm.bitL (Flock.MerkleScheme.rowBytes a ++ rest).data.toList t))
    (hcoef : ∀ t < 8 * (Flock.MerkleScheme.rowBytes cf).size,
      RegBit zo u j (p.row + p.aux + p.acc + t) (Hm.bitL (Flock.MerkleScheme.rowBytes cf).data.toList t))
    (hdir : ∀ h < H, RegBit zo u j (p.row + p.aux + p.acc + p.coef + h) (idx.testBit h)) :
    ColsHold (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj)
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits)
      ((Flock.climb ms (Flock.Sha512.hash (Flock.MerkleScheme.rowBytes r)) idx ss ++ ⟨Array.replicate 64 0⟩) ++
        (Flock.MerkleScheme.rowBytes #[consTerm LANES r a cf 0, consTerm LANES r a cf 1] ++ rest)) := by
  have hone' : fill zo bits (Const.Prog.oneAt (Placed.keyProg c hU nv) u) = true := by
    rw [← keyInst_one hU nv (Layout.outsOf c hU nv) u, fill_one zo hA _ hj, keyInst_one]
    exact hone
  have hO := keyProg_recOpen hU nv (Layout.outsOf c hU nv) (fill zo bits) u hj hone' ms hms LANES H hH p outBits hl hu
    hout r a cf rest idx ss hr ha hss hrowN hauxN haccN hcoefN hdirsN (colsHold_fill zo bits hj 0 _ hrow)
    (fun i hi => ⟨(haux i hi).1, colsHold_fill zo bits hj _ _ (haux i hi).2⟩) (colsHold_fill zo bits hj _ _ hacc)
    (colsHold_fill zo bits hj _ _ hcoef) (fun h hh => copyVal_regBit zo bits hj (hdir h hh))
  refine colsHold_of_fill zo hA hj ?_ hO
  rw [Layout.ofNet_nIn_eq, inWords_recOpen hl]
  exact inRows_le_outStart _ _ _

end Reg

/-! ## A part of S, at the whole ports -/

/-- Port elements `x` at the indices `idx`, zero past them. -/
def sub (idx : Array ℕ) (x : ℕ → Flock.F128) (k : ℕ) : Flock.F128 := if h : k < idx.size then x idx[k] else 0

theorem sub_lt (idx : Array ℕ) (x : ℕ → Flock.F128) : ∀ k (hk : k < idx.size), sub idx x k = x idx[k] :=
  fun _ hk => dite_eq_left hk

theorem sub_ge {idx : Array ℕ} (x : ℕ → Flock.F128) {m : ℕ} (h : idx.size ≤ m) : ∀ e, m ≤ e → sub idx x e = 0 :=
  fun _ he => dite_eq_right (by omega)

/-- **The verifier's coefficients** at a rep's coins: `verifierRows`' first component, zero past it (and everywhere when
the verifier refuses). -/
def vOf (sh : Flock.Shape) (rounds : Array (Array Flock.F128)) (extras : Array Flock.Claim) (comb : Array Flock.F128) :
    ℕ → Flock.F128 :=
  match Flock.RecAlgebra.verifierRows sh rounds extras comb with
  | .ok (v, _, _) => fun k => v.getD k 0
  | .error _ => fun _ => 0

theorem vOf_eq {sh : Flock.Shape} {rounds : Array (Array Flock.F128)} {extras : Array Flock.Claim}
    {comb : Array Flock.F128} {v : Array Flock.F128} {info} {kappa} (h : Flock.RecAlgebra.verifierRows sh rounds extras comb = .ok (v, info, kappa)) :
    vOf sh rounds extras comb = fun k => v.getD k 0 := by
  unfold vOf
  rw [h]
  try rfl

section Part

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **A part of S computes its residuals at the whole ports, in a layer agreeing with the registrant** (`decode_innerUnit` at `sub`). When copy
`j` of unit `u` is V*'s `InnerRepCheck` unit of a part `S` whose row ports are registered bits of the whole message
elements `w` at the part's indices `wI`, and whose `v` port is registered bits of the verifier's coefficients (`vOf`) at
`V`, its output columns in `bits` hold the part's residuals at part ports `w'`, `v'` that read `w` at `wI` and `vOf` at
`V`: `PartS.res_partS`' `hw` and `hv` at `wI = wIdx sizes P`. -/
theorem decode_part {bits : Fin (Layout.keyN c hU nv) → Bool} (hA : zo.RegAgree bits) (u : Fin nv) {j : ℕ}
    (hj : j < c.upv) (hone : bits (Const.Prog.oneAt (Placed.keyProg c hU nv) u) = true)
    {S : VStar.RepS} {name : String} {U : VStar.UnitC} (hUn : VStar.innerUnit S name = .ok U)
    (hS : ∀ r ∈ S.gf.res, OpsInRange S.gf r) {l : Flock.HmNets.Laid} (hl : U.laid = .ok l) {name' : String}
    (hn : Flock.HmNets.netOf name' l = .ok c.unit) (hu : c.unit.unitLog ≤ 32)
    (sh : Flock.Shape) (rounds : Array (Array Flock.F128)) (extras : Array Flock.Claim) (comb : Array Flock.F128)
    (w : ℕ → Flock.F128) (wI V : Array ℕ) (hwI : wI.size ≤ S.gf.nw) (hV : V.size ≤ S.gf.nv)
    (hw : ∀ q < 128 * S.gf.nw, RegBit zo u j (rowCol S.ports 0 q) (wbit (sub wI w) q))
    (hv : ∀ q < 128 * S.gf.nv, RegBit zo u j (vBase S + q) (wbit (sub V (vOf sh rounds extras comb)) q)) :
    ∃ w' v' : ℕ → Flock.F128, (∀ k (hk : k < wI.size), w' k = w wI[k]) ∧
      (∀ k (hk : k < V.size), v' k = vOf sh rounds extras comb V[k]) ∧
      ResCols S.gf w' v' (copyVal (keyInst hU nv (Layout.outsOf c hU nv) u) bits hj)
        (Hm.outStart U.st (Array.replicate U.inWords 16) (Array.replicate U.outWords 16)) :=
  ⟨sub wI w, sub V (vOf sh rounds extras comb), sub_lt wI w, sub_lt V _,
    decode_innerUnit zo hA u hj hone hUn hS hl hn hu _ _ (sub_ge w hwI) (sub_ge _ hV) hw hv⟩

end Part

end FlockVBridge.Decode
