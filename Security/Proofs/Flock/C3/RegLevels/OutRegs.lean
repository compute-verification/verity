import Proofs.Flock.C3.RegLevels.OpensRegs
import Proofs.Flock.C3.PubRows.Check
import Proofs.Flock.VBridge.Family.Openings

/-!
# n11: `OpensTopAt`'s registered outputs, from the unit's output rows

`opensTopAt_of_regs` takes `haout` (and `hT`): copy `j`'s output columns registered (`RegOut`). An output column of a
unit's instance is a computed committed gate (`Layout.outsOf`), which the hidden-output layout places on a bit of an
output row (`HiddenExec.rowOfH`, `outRowOf`): copy `j`'s output `o`, bit `t`, is bit `16 k + t` of output row
`nIn + jo`, where the copy's output word is word `k` of output port `jo` (`HiddenExec.outBitRow`). So:

* `regOut_of_outBitRow`: where that row is a registered read, the column is registered, its bit the registrant's;
* `OutsLaid`, `outBitRow_laid`: at V*'s output layout (`out` then `acc_out`, 64 words each, one output row each, copy
  `j` returning words `0..127` in order; `rec_outer.lowering`), bit `b < 1024` of output `jo < 2` is bit `b` of row
  `nIn + jo`. `RecOpen.checkAt` does not read the layout (`check_congr`: a circuit's name, unit, rows and input leaves
  only), so it is a binder here;
* `outCols_of_laid`, `regOuts_of_laid`: at a unit laid out by `layout` (`recOpenLaid`, or the verifier's
  `RecOpen.unitLaid`), output `jo`'s bit `b` is column `outStart + 1024 jo + b`, a computed column, registered where
  row `nIn + jo` is a registered read;
* `haout_of_laid`: `opensTopAt_of_regs`' `haout`, where the reads file names `acc_out` (`accOut_of_checkStmt`, by
  `checkAcc`, `own_of_checkAcc`) and the session registers each row of a port the reads file names (`hread`, as
  `regReads_of_check`'s);
* `accOutRegs_of_check`: at the verifier's unit, the `acc_out` columns registered as the bits of the row the `acc_out`
  read holds, as `accOut_of_regOut` takes them (`AccOut.lean`'s `accOutL_of_keysS`).
-/

namespace FlockC3.RegLevels

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge FlockVBridge.AccChain

/-! ## V*'s output rows -/

/-- **V*'s output layout at copy `j` of an R level's unit** (`rec_outer.lowering`): the outputs are `out` then
`acc_out`, 64 words each, copy `j` returns words `0..127` in order, and each is one output row, `out` the first past the
input rows. A fact of V*'s circuit file (`RecVStar.checkFile`), which `RecOpen.checkAt` does not read. -/
def OutsLaid (c : Flock.Circuit) (j : ℕ) : Prop :=
  c.outPorts = #[("out", 64), ("acc_out", 64)] ∧ c.leavesOut.getD j #[] = Array.range 128 ∧
    ∃ hq : HmOut.nIn c + 1 < c.ports.size,
      (c.ports[HmOut.nIn c]'(Nat.lt_of_succ_lt hq)).name = "out" ∧
      (c.ports[HmOut.nIn c]'(Nat.lt_of_succ_lt hq)).words = 64 ∧
      c.ports[HmOut.nIn c + 1].name = "acc_out" ∧ c.ports[HmOut.nIn c + 1].words = 64

theorem outBitRow_eq {c : Flock.Circuit} {n : ℕ} (u : Fin n) {d : ℕ × ℕ × ℕ} {q k : ℕ} (hq : q < c.ports.size)
    (hqv : HmOut.nIn c + HiddenExec.outPortOf c (HiddenExec.outWordOf c d) = q)
    (hkv : 16 * (HiddenExec.outWordOf c d - Hidden.outAt c (HiddenExec.outPortOf c (HiddenExec.outWordOf c d))) +
      d.2.2 = k) (hk : k < Layout.portRowW c ⟨q, hq⟩) :
    HiddenExec.outBitRow c u d = some (u, ⟨⟨q, hq⟩, ⟨k, hk⟩⟩) := by
  subst hqv hkv
  unfold HiddenExec.outBitRow
  rw [dite_eq_left hq, dite_eq_left hk]

/-- **Bit `b` of V*'s output `jo` is bit `b` of output row `nIn + jo`** (`OutsLaid`): copy `j`'s output word
`64 jo + b / 16` is word `b / 16` of output port `jo`. -/
theorem outBitRow_laid {c : Flock.Circuit} {j : ℕ} (h : OutsLaid c j) {n : ℕ} (u : Fin n) {jo b : ℕ} (hjo : jo < 2)
    (hb : b < 1024) :
    ∃ hq : HmOut.nIn c + jo < c.ports.size, ∃ hk : b < Layout.portRowW c ⟨_, hq⟩,
      HiddenExec.outBitRow c u (j, 64 * jo + b / 16, b % 16) = some (u, ⟨⟨_, hq⟩, ⟨b, hk⟩⟩) := by
  obtain ⟨hop, hlo, hq1, -, hw0, -, hw1⟩ := h
  have hq : HmOut.nIn c + jo < c.ports.size := by omega
  have hW : HiddenExec.outWordOf c (j, 64 * jo + b / 16, b % 16) = 64 * jo + b / 16 := by
    show (c.leavesOut.getD j #[])[64 * jo + b / 16]! = 64 * jo + b / 16
    rw [hlo, getElem!_pos (Array.range 128) (64 * jo + b / 16) (by rw [Array.size_range]; omega)]
    simp
  have hs : c.outPorts.size = 2 := by rw [hop] <;> rfl
  have hA0 : Hidden.outAt c 0 = 0 := by simp [Hidden.outAt]
  have hA1 : Hidden.outAt c 1 = 64 := by unfold Hidden.outAt; rw [hop] <;> rfl
  have hN : c.outWords = 128 := by
    rw [← HiddenExec.outAt_size c, hs]
    unfold Hidden.outAt
    rw [hop] <;> rfl
  have h64 : (c.outPorts.getD 0 default).2 = 64 := by rw [hop] <;> rfl
  have hP : HiddenExec.outPortOf c (64 * jo + b / 16) = jo := by
    obtain ⟨hlt, hle, hsub⟩ := HiddenExec.outPortOf_spec c (o := 64 * jo + b / 16) (by rw [hN]; omega)
    rw [hs] at hlt
    rcases (by omega : HiddenExec.outPortOf c (64 * jo + b / 16) = 0 ∨ HiddenExec.outPortOf c (64 * jo + b / 16) = 1)
      with h0 | h1
    · rw [h0, hA0, h64] at hsub
      omega
    · rw [h1, hA1] at hle
      omega
  have hw : c.ports[HmOut.nIn c + jo].words = 64 := by
    rcases (by omega : jo = 0 ∨ jo = 1) with rfl | rfl
    · exact hw0
    · exact hw1
  have hk : b < Layout.portRowW c ⟨_, hq⟩ := by
    show b < 16 * c.ports[HmOut.nIn c + jo].words
    rw [hw]
    omega
  refine ⟨hq, hk, outBitRow_eq u hq (by rw [hW, hP]) ?_ hk⟩
  rw [hW, hP]
  show 16 * (64 * jo + b / 16 - Hidden.outAt c jo) + b % 16 = b
  rcases (by omega : jo = 0 ∨ jo = 1) with rfl | rfl
  · rw [hA0]; omega
  · rw [hA1]; omega

/-! ## A copy's output columns on output rows -/

section Copy

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **A copy's output column on a registered output row is registered** (`RegOut`), as the registrant's bit there:
copy `j`'s computed column `outCol j' t'` is a returned output of the unit (`Layout.OutColAt`, so in `outsOf` and
committed), not an input row's bit (`keyRowOf_wire`), so its commit string is the output row `outBitRow` gives
(`rowOfH_of_none`, `outRowOf_wire`). -/
theorem regOut_of_outBitRow {u : Fin nv} {j : ℕ} (hj : j < c.upv) {j' t' : ℕ}
    (hj' : j' < (c.leavesOut.getD j #[]).size) (ht' : t' < 16)
    (h1 : (Rows.ofNet hU).nIn ≤ Layout.outCol c j' t')
    (h2 : Layout.outCol c j' t' < (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp)
    {q : Fin c.ports.size} {k : Fin (Layout.portRowW c q)}
    (hrow : HiddenExec.outBitRow c u (j, j', t') = some (u, ⟨q, k⟩)) (hread : Decode.regRead zo (.inl (u, q))) :
    RegOut zo u hj (Layout.outCol c j' t') ((Decode.regVal zo (.inl (u, q))).2
      (Fin.castLE ((Finset.le_sup (f := Layout.portRowW c) (Finset.mem_univ q)).trans (Nat.le_succ _)) k)) := by
  have hcol : Layout.outCol c j' t' < (Rows.ofNet hU).w := by
    simp only [FlockSoundness.Rows.w]
    omega
  refine ⟨hcol, ?_⟩
  have hxv : ((Rows.ofNet hU).stackCol c.upv hj ⟨_, hcol⟩).val = c.upv * (Rows.ofNet hU).nIn +
      (j * ((Rows.ofNet hU).nComp + 1) + (1 + (Layout.outCol c j' t' - (Rows.ofNet hU).nIn))) := by
    simp only [FlockSoundness.Rows.stackCol, show ¬ Layout.outCol c j' t' < (Rows.ofNet hU).nIn by omega, h2,
      ↓reduceDIte]
  obtain ⟨i, hi⟩ := Decode.stackCol_comp (Rows.ofNet hU) c.upv hj ⟨_, hcol⟩ h1
  generalize (Rows.ofNet hU).stackCol c.upv hj ⟨_, hcol⟩ = x at hxv hi ⊢
  have eR := Placed.rows_circuitProg c hU nv (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)) u
  obtain ⟨x', hxv', hw⟩ : ∃ x' : Fin ((Placed.keyProg c hU nv).rows u).w, x'.val = x.val ∧
      (keyInst hU nv (Layout.outsOf c hU nv) u).wire x =
        (Const.Prog.instC (Placed.keyProg c hU nv) (Layout.outsOf c hU nv) u).wire x' :=
    ⟨Fin.cast (congrArg FlockSoundness.Rows.w eR).symm x, rfl,
      Const.Prog.castW_wire eR (Const.Prog.rowsWireC (Placed.keyProg c hU nv) u) x⟩
  have eN : ((Placed.keyProg c hU nv).rows u).nIn = c.upv * (Rows.ofNet hU).nIn :=
    (congrArg FlockSoundness.Rows.nIn eR).trans rfl
  have hge : ((Placed.keyProg c hU nv).rows u).nIn ≤ x'.val := by
    rw [eN, hxv', hxv]
    omega
  have hgu : (Placed.keyProg c hU nv).unit
      ((Const.Prog.instC (Placed.keyProg c hU nv) (Layout.outsOf c hU nv) u).wire x') = some u := by
    rw [← hw, hi]
    exact (keyInst hU nv (Layout.outsOf c hU nv) u).mem i
  have hoc : Layout.OutColAt c hU x'.val j j' t' := ⟨hj, hj', ht', h1, h2, by rw [hxv', hxv]⟩
  have hkey : Layout.keyRowOf c hU nv (Layout.portRowW c)
      ((Const.Prog.instC (Placed.keyProg c hU nv) (Layout.outsOf c hU nv) u).wire x') = none := by
    rcases hk : Layout.keyRowOf c hU nv (Layout.portRowW c)
        ((Const.Prog.instC (Placed.keyProg c hU nv) (Layout.outsOf c hU nv) u).wire x') with _ | ⟨u', q', k'⟩
    · rfl
    · have := (Layout.keyRowOf_wire (Layout.outsOf c hU nv) x' hk).2.1
      omega
  have hrowH : HiddenExec.rowOfH c hU nv (Layout.outsOf c hU nv)
      ((Const.Prog.instC (Placed.keyProg c hU nv) (Layout.outsOf c hU nv) u).wire x') = some (u, ⟨q, k⟩) := by
    rw [HiddenExec.rowOfH_of_none _ hkey, HiddenExec.outRowOf_wire _ x' hgu hge, Layout.outDecode_of hoc, hrow]
  have hp : posK c hU nv ((Const.Prog.instC (Placed.keyProg c hU nv) (Layout.outsOf c hU nv) u).wire x') =
      .inl (u, q) := by
    unfold posK
    simp only [hrowH]
  have hmem : (Const.Prog.instC (Placed.keyProg c hU nv) (Layout.outsOf c hU nv) u).wire x' ∈
      Layout.outsOf c hU nv :=
    (@Finset.mem_filter _ _ (Classical.decPred _) _ _).2 ⟨Finset.mem_univ _, u, x', rfl, j, j', t', hoc⟩
  -- `layShHJ`'s `Pos` and `keyN` are `PosK` and the key circuit's `N` only past `instances` transparency, where `rw`
  -- matches and checks its motive, so `lay_pos`, `lay_bit` and `.inl (u, q)` enter by unification instead
  have hpos := (Decode.lay_pos zo _).trans hp
  have hv : zo.rd.wv (zo.lay.pos ((Const.Prog.instC (Placed.keyProg c hU nv) (Layout.outsOf c hU nv) u).wire x')) =
      Decode.regVal zo (.inl (u, q)) := congrArg zo.rd.wv hpos
  rw [hw]
  refine ⟨FlockSoundness.Audit.Partition.mem_committed_of_output _ hmem, (congrArg zo.rd.isRead hpos).mpr hread, ?_⟩
  refine (Decode.lay_bit zo _ _).trans ((congrArg (fun v => bitK c hU nv v _) hv).trans ?_)
  simp only [bitK, idxK, hrowH]

/-- **A unit laid out by `layout` returns its outputs from `outStart`**: the output group's first column, within the
computed columns, past at least the outputs' bits. Both `recOpenLaid` and the verifier's `RecOpen.unitLaid` (its
inputs one 16-bit port per word) are such layouts. -/
theorem outCols_of_laid {st : Flock.HmNets.St} {i outBits : Array ℕ} {outs : Array Flock.HmNets.Lin} {name : String}
    (hl : Flock.HmNets.netOf name (Flock.HmNets.layout st i outBits outs) = .ok c.unit) :
    (c.unit.outGroups.getD 0 default).col * Flock.WORD = Hm.outStart st i outBits ∧
      (Rows.ofNet hU).nIn ≤ Hm.outStart st i outBits ∧
      Hm.outStart st i outBits + outBits.foldl (· + ·) 0 ≤ (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp := by
  obtain ⟨-, -, -, -, hcp, -, hog, -⟩ := Flock.Net.ofRows_ok hl
  obtain ⟨h1, -⟩ := Flock.HmNets.layout_rows st i outBits outs
  have hob := Flock.HmNets.le_groupWords (outBits.foldl (· + ·) 0)
  have hw : c.unit.inWords * Flock.WORD = Hm.inRowsOf i := by
    obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, hw, -⟩ := Flock.Net.ofRows_ok hl
    rw [hw, Hm.layout_inGroups]
    simp [Hm.inRowsOf]
  refine ⟨?_, ?_, ?_⟩
  · rw [hog, Hm.layout_outGroups]
    exact Hm.outStart_word _ _ _
  · rw [FlockSoundness.Discharge.Layout.ofNet_nIn_eq (hU := hU), hw]
    exact Decode.inRows_le_outStart _ _ _
  · rw [show (Rows.ofNet hU).nIn + (Rows.ofNet hU).nComp = c.unit.constPos from Rows.ofNet_one hU, hcp, h1]
    unfold Hm.outStart Hm.inRowsOf Hm.spanOf
    omega

/-- **V*'s output `jo`'s columns are registered where its output row is a registered read** (`regOut_of_outBitRow` at
`outBitRow_laid`): at a unit laid out by `layout` with at least V*'s 2048 output bits, at V*'s output layout
(`OutsLaid`), column `outStart + 1024 jo + b` of copy `j` is registered as bit `b` of the registrant's row
`nIn + jo`. -/
theorem regOuts_of_laid {u : Fin nv} {j : ℕ} (hj : j < c.upv) {st : Flock.HmNets.St} {i outBits : Array ℕ}
    {outs : Array Flock.HmNets.Lin} {name : String}
    (hl : Flock.HmNets.netOf name (Flock.HmNets.layout st i outBits outs) = .ok c.unit)
    (hob : 2048 ≤ outBits.foldl (· + ·) 0) (hlaid : OutsLaid c j) {jo : ℕ} (hjo : jo < 2)
    (hread : ∀ hq : HmOut.nIn c + jo < c.ports.size, Decode.regRead zo (.inl (u, ⟨_, hq⟩))) {b : ℕ}
    (hb : b < 1024) :
    ∃ hq : HmOut.nIn c + jo < c.ports.size, ∃ hk : b < Layout.portRowW c ⟨_, hq⟩,
      RegOut zo u hj (Hm.outStart st i outBits + 1024 * jo + b)
        ((Decode.regVal zo (.inl (u, ⟨_, hq⟩))).2 (Fin.castLE ((Finset.le_sup (f := Layout.portRowW c)
          (Finset.mem_univ (⟨HmOut.nIn c + jo, hq⟩ : Fin c.ports.size))).trans (Nat.le_succ _)) ⟨b, hk⟩)) := by
  obtain ⟨hcW, hIn, hEnd⟩ := outCols_of_laid (hU := hU) hl
  obtain ⟨hq, hk, hrow⟩ := outBitRow_laid hlaid u hjo hb
  have hj' : 64 * jo + b / 16 < (c.leavesOut.getD j #[]).size := by
    rw [hlaid.2.1, Array.size_range]
    omega
  have hcol : Layout.outCol c (64 * jo + b / 16) (b % 16) = Hm.outStart st i outBits + 1024 * jo + b := by
    unfold Layout.outCol
    rw [hcW]
    omega
  refine ⟨hq, hk, ?_⟩
  rw [← hcol]
  exact regOut_of_outBitRow zo hj hj' (Nat.mod_lt _ (by decide)) (by rw [hcol]; omega) (by rw [hcol]; omega) hrow
    (hread hq)

/-- **`opensTopAt_of_regs`' `haout`** (`regOuts_of_laid` at `acc_out`, output 1): where the reads file names `acc_out`
(`accOut_of_checkStmt`) and the session registers each row of a port the reads file names (`hread`), each of the first
256 columns of copy `j`'s `acc_out` half is registered. -/
theorem haout_of_laid {u : Fin nv} {j : ℕ} (hj : j < c.upv) {LANES H : ℕ} {p : Ports} {outBits : Array ℕ}
    {name : String} (hl : Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit)
    (hob : 2048 ≤ outBits.foldl (· + ·) 0) (hlaid : OutsLaid c j) (hacc : ∃ pt ∈ zo.own, pt.name = "acc_out")
    (hread : ∀ q : Fin c.ports.size, (∃ pt ∈ zo.own, pt.name = c.ports[q].name) → zo.rd.isRead (.inl (u, q))) :
    ∀ t < 256, ∃ b, RegOut zo u hj
      (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + 1024 + t) b := by
  intro t ht
  have hl' := hlaid
  obtain ⟨-, -, hq1, -, -, hn1, -⟩ := hl'
  obtain ⟨pt, hpt, hpn⟩ := hacc
  obtain ⟨hq, hk, h⟩ := regOuts_of_laid zo hj hl hob hlaid (jo := 1) (by decide)
    (fun hq => hread ⟨_, hq⟩ ⟨pt, hpt, hpn.trans hn1.symm⟩) (b := t) (by omega)
  rw [Nat.mul_one] at h
  exact ⟨_, h⟩

/-- **`accOut_of_regOut`'s registered columns at the verifier's unit** (`regOuts_of_laid` at `acc_out`, output 1, at
`RecOpen.unitLaid`, `RecOpen.check_ok`): at a unit the verifier parsed as `rec-open/l<LANES>-h<H>`, with V*'s output
layout at copy `j` (`OutsLaid`), where unit `u`'s `acc_out` row (`nIn + 1`) is a registered read whose first 256 bits
are those of `A`'s bytes (`hval`), each of the first 256 columns of copy `j`'s `acc_out` half is registered as `A`'s
bit. -/
theorem accOutRegs_of_check (hrec : Flock.RecOpen.check c = .ok ()) {LANES H : ℕ}
    (hp : Flock.RecOpen.params c.unit.name = some (LANES, H)) {u : Fin nv} {j : ℕ} (hj : j < c.upv)
    (hlaid : OutsLaid c j) (hq : HmOut.nIn c + 1 < c.ports.size) (hread : zo.rd.isRead (.inl (u, ⟨_, hq⟩)))
    {A : Array F128}
    (hval : ∀ b < 256, ∀ hb : b < Layout.portRowW c ⟨_, hq⟩,
      (Decode.regVal zo (.inl (u, ⟨_, hq⟩))).2 ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (f := Layout.portRowW c)
          (Finset.mem_univ (⟨HmOut.nIn c + 1, hq⟩ : Fin c.ports.size))).trans (Nat.le_succ _))⟩ =
        Hm.bitL (MerkleScheme.rowBytes A).data.toList b) :
    ∀ t < 256, RegOut zo u hj (Family.outCol LANES H + 1024 + t) (Hm.bitL (MerkleScheme.rowBytes A).data.toList t) := by
  intro t ht
  obtain ⟨-, hk, h⟩ := regOuts_of_laid zo hj (Flock.RecOpen.check_ok hrec hp).2
    (by have := foldl_wordPorts 2048; omega) hlaid (jo := 1) (by decide) (fun _ => hread) (b := t) (by omega)
  have h' : RegOut zo u hj (Family.outCol LANES H + 1024 * 1 + t) _ := h
  convert h' using 1 <;> first | omega | exact (hval t ht hk).symm

end Copy

/-! ## `acc_out` in the reads file -/

/-- **Each of a statement's `rec-acc` ports is a port of the reads file `checkAcc` accepted.** -/
theorem own_of_checkAcc {s : VStar.Stmt} {own : Array Flock.Registered.Port}
    {rows : Option (Array Flock.Registered.Port)} (h : RecVStar.checkAcc s own rows = .ok ()) :
    ∀ nm ∈ RecVStar.accPorts s, ∃ pt ∈ own, pt.name = nm := by
  unfold RecVStar.checkAcc at h
  obtain ⟨_, h, -⟩ := FlockSoundness.Refine.ebind_ok h
  refine FlockC3.PubRows.rows_forIn_each _ ?_ _ h
  intro nm r hr
  split at hr
  · rename_i p hf
    split at hr
    · cases hr
    · refine ⟨⟨p, Array.mem_of_find?_eq_some hf, by simpa using Array.find?_some hf⟩, ?_⟩
      cases hr
      rfl
  · cases hr

/-- **An R level's checked session has `acc_out` in its reads file** (`acc_of_checkStmt`'s twin): `checkStmt` ran
`checkAcc` on V*'s statement `j`, R's level `l`, whose `rec-acc` ports are `acc` and `acc_out`. -/
theorem accOut_of_checkStmt {i : RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f) rows =
      .ok ()) {l LANES H q : ℕ}
    (hj : ∀ ss : Array VStar.Stmt, RecVStar.statements i = .ok ss → ss[j]? = some (.recOpenStacked l LANES H q)) :
    ∃ pt ∈ own, pt.name = "acc_out" := by
  obtain ⟨ss, s, hss, hs, hacc, -⟩ := checkAcc_of_checkStmt h
  obtain rfl : s = .recOpenStacked l LANES H q := Option.some.inj (hs.symm.trans (hj ss hss))
  refine own_of_checkAcc hacc "acc_out" ?_
  rw [show RecVStar.accPorts (.recOpenStacked l LANES H q) = ["acc", "acc_out"] from rfl]
  simp

end FlockC3.RegLevels

#print axioms FlockC3.RegLevels.regOuts_of_laid

#print axioms FlockC3.RegLevels.haout_of_laid

#print axioms FlockC3.RegLevels.accOutRegs_of_check
