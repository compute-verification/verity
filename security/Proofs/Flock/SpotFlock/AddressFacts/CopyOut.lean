module

public import Proofs.Flock.SpotFlock.AddressFacts.CopySrc
public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
public import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
public import Proofs.Flock.Soundness.Discharge.Placed.Setup

/-!
# The copy at a written bit

`copiesRef_out`: `CopiesAt`'s written half where the second stage's design evaluates and each member names its ports
once. At a written bit of member `t`'s unit `u`, the column `pickAny` chooses is the returned output's computed column
(`OutRef`, the output case of `CopiesRef`), and the bit's port bit is the output word's.

A written bit is Bits' commitment position of the key program's gate on bit `b` of returned output leaf `j` of unit
`u`'s instance `k` (`Spot.Bits.outGate`). `uniform` puts a unit's computed column on its own gate past the earlier
units' (`wire_comp_uniform`), so that gate is the unit's column `OutColAt` names (`outGate_wire`), and no other
column of the unit is on it (`RowsWire.shared`). The design evaluates only where every member's circuit is in
`scopeOkH`'s scope (`membersScoped`): its outputs come from the unit net, and each output row holds one output port
(`OutRowsS`), so Bits' output row of the leaf's word is the word's output port and its bit the word's bit
(`outRowBit_ref`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy FlockSoundness FlockSoundness.Discharge

/-! ## A unit's computed columns -/

theorem gates_eq (k : ℕ) (R : FlockSoundness.Rows) : ∀ n, Placed.gates k R n = k + n * R.nComp
  | 0 => by simp [Placed.gates]
  | n + 1 => by
    rw [Placed.gates, gates_eq k R n]
    ring

section uniform

variable {k : ℕ} {R : FlockSoundness.Rows} {ins : ℕ → Fin R.nIn → Fin k} {one : Fin k} {hone : ∀ u c, ins u c ≠ one}

/-- **`uniform` puts unit `u`'s computed column `x` on its own gate**: the `(x − nIn)`th past the earlier units'. -/
theorem wire_comp_uniform : ∀ (n : ℕ) (u : Fin n) (x : Fin ((Placed.uniform k R ins one hone n).rows u).w),
    R.nIn ≤ x.val → x.val < R.nIn + R.nComp →
    ((Const.Prog.rowsWireC (Placed.uniform k R ins one hone n) u).wire x).val =
      Placed.gates k R u.val + (x.val - R.nIn)
  | 0, u, _, _, _ => u.elim0
  | n + 1, u, x, h1, h2 => by
    induction u using Fin.lastCases with
    | last =>
      change ((Const.Prog.rowsWireC (FlockSoundness.Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _)
        (Fin.last n)).wire x).val = _
      rw [Const.Prog.rowsWireC_last, Const.Prog.castW_wire]
      change (FlockSoundness.Prog.colWire R _ _ _).val = _
      rw [FlockSoundness.Prog.colWire_val]
      split_ifs with ha hb
      · exact absurd ha (by simp; omega)
      · simp
      · exact absurd hb (by simp; omega)
    | cast v =>
      change ((Const.Prog.rowsWireC (FlockSoundness.Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _)
        v.castSucc).wire x).val = _
      rw [Const.Prog.rowsWireC_castSucc, Const.Prog.castW_wire]
      change (Fin.castAdd R.nComp ((Const.Prog.rowsWireC (Placed.uniform k R ins one hone n) v).wire _)).val = _
      rw [Fin.val_castAdd]
      refine (wire_comp_uniform n v _ ?_ ?_).trans ?_
      · simpa using h1
      · simpa using h2
      · simp

end uniform

/-- **Unit `u`'s computed column `x` of the circuit's program is on gate `2 + n·B + u·C + (x − B)`**, `B` and `C` the
template instance rows' inputs and computed rows. -/
theorem keyProg_wire_comp {c : _root_.Flock.Circuit} {hU : FlockSoundness.Built c.unit} {n : ℕ}
    (outs : Finset (Fin (Layout.keyN c hU n))) (u : Fin n) (x : ℕ) (hx : x < ((Placed.keyProg c hU n).rows u).w)
    (h1 : (Placed.circuitRows c hU).nIn ≤ x)
    (h2 : x < (Placed.circuitRows c hU).nIn + (Placed.circuitRows c hU).nComp) :
    ((Const.Prog.instC (Placed.keyProg c hU n) outs u).wire ⟨x, hx⟩).val =
      2 + n * Layout.keyB c hU + u.val * (Placed.circuitRows c hU).nComp + (x - (Placed.circuitRows c hU).nIn) := by
  have e := wire_comp_uniform (k := Placed.inGates n (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
    (ins := Placed.insOf n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)))
    (one := Placed.oneGate n (Layout.keyB c hU))
    (hone := Placed.insOf_ne_one n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU))) n u ⟨x, hx⟩ h1 h2
  rw [gates_eq] at e
  exact e

/-! ## A returned output's gate -/

/-- **What the key program's gate on a returned output's bit says**: the unit, instance, leaf and bit are in range, the
leaf's column is a computed row of the unit net, and the gate is the instance's row on it (`Spot.Bits.rowGate`). -/
theorem outGate_spec {c : _root_.Flock.Circuit} {n u k j b g : ℕ} (hg : Spot.Bits.outGate c n u k j b = some g) :
    u < n ∧ k < c.upv ∧ j < (c.leavesOut.getD k #[]).size ∧ b < 16 ∧
      (Spot.Bits.Rows.ofNet c.unit).nIn ≤ Spot.Bits.outCol c j b ∧
      Spot.Bits.outCol c j b < (Spot.Bits.Rows.ofNet c.unit).nIn + (Spot.Bits.Rows.ofNet c.unit).nComp ∧
      g = Spot.Bits.rowGate (Spot.Bits.Rows.ofNet c.unit) (2 + n * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn)
        ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nComp u k (Spot.Bits.outCol c j b) := by
  unfold Spot.Bits.outGate at hg
  dsimp only at hg
  split at hg
  · rename_i hcond
    simp only [Bool.and_eq_true, decide_eq_true_eq] at hcond
    obtain ⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h4⟩, h5⟩, h6⟩ := hcond
    exact ⟨h1, h2, h3, h4, h5, h6, (Option.some.inj hg).symm⟩
  · cases hg

/-- **A returned output's gate is its unit's column `OutColAt` names**: unit `u`'s computed column of instance `k`'s
leaf `j`, bit `b`, is on the gate `Spot.Bits.outGate` gives. -/
theorem outGate_wire {c : _root_.Flock.Circuit} (hU : FlockSoundness.Built c.unit) (hu : c.unit.unitLog ≤ 32)
    {n u k j b g : ℕ} (outs : Finset (Fin (Layout.keyN c hU n))) (hg : Spot.Bits.outGate c n u k j b = some g) :
    ∃ hun : u < n, ∃ x : Fin ((Placed.keyProg c hU n).rows ⟨u, hun⟩).w,
      Layout.OutColAt c hU x.val k j b ∧ ((Placed.keyProg c hU n).rows ⟨u, hun⟩).nIn ≤ x.val ∧
      ((Const.Prog.instC (Placed.keyProg c hU n) outs ⟨u, hun⟩).wire x).val = g := by
  obtain ⟨hun, hk, hj, hb, hlo, hhi, rfl⟩ := outGate_spec hg
  have hR := _root_.Flock.Bridge.rowsAgree_circuitRows c hU hu
  have eIn := congrArg FlockSoundness.Rows.nIn
    (Placed.rows_circuitProg c hU n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)) ⟨u, hun⟩)
  have eComp := congrArg FlockSoundness.Rows.nComp
    (Placed.rows_circuitProg c hU n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)) ⟨u, hun⟩)
  have cIn : (Placed.circuitRows c hU).nIn = c.upv * (FlockSoundness.Rows.ofNet hU).nIn := rfl
  have cComp : (Placed.circuitRows c hU).nComp = c.upv * ((FlockSoundness.Rows.ofNet hU).nComp + 1) := rfl
  have sIn : (Spot.Bits.Rows.ofNet c.unit).nIn = (FlockSoundness.Rows.ofNet hU).nIn := rfl
  have sComp : (Spot.Bits.Rows.ofNet c.unit).nComp = (FlockSoundness.Rows.ofNet hU).nComp := rfl
  have hB : ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn = Layout.keyB c hU := hR.1
  have hC : ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nComp = (Placed.circuitRows c hU).nComp := hR.2.1
  rw [sIn, sComp] at hlo hhi
  have hmul := Nat.mul_le_mul_right ((FlockSoundness.Rows.ofNet hU).nComp + 1) (show k + 1 ≤ c.upv by omega)
  have e1 : (k + 1) * ((FlockSoundness.Rows.ofNet hU).nComp + 1) =
      k * ((FlockSoundness.Rows.ofNet hU).nComp + 1) + ((FlockSoundness.Rows.ofNet hU).nComp + 1) := by ring
  obtain ⟨X, hX⟩ : ∃ X, X = c.upv * (FlockSoundness.Rows.ofNet hU).nIn +
      (k * ((FlockSoundness.Rows.ofNet hU).nComp + 1) +
        (1 + (Spot.Bits.outCol c j b - (FlockSoundness.Rows.ofNet hU).nIn))) := ⟨_, rfl⟩
  have hX1 : (Placed.circuitRows c hU).nIn ≤ X := by omega
  have hX2 : X < (Placed.circuitRows c hU).nIn + (Placed.circuitRows c hU).nComp := by omega
  have hXw : X < ((Placed.keyProg c hU n).rows ⟨u, hun⟩).w := by
    show X < ((Placed.keyProg c hU n).rows ⟨u, hun⟩).nIn + ((Placed.keyProg c hU n).rows ⟨u, hun⟩).nComp + 1
    rw [eIn, eComp]
    omega
  refine ⟨hun, ⟨X, hXw⟩, ⟨hk, hj, hb, hlo, hhi, hX⟩, by rw [eIn]; exact hX1, ?_⟩
  rw [keyProg_wire_comp outs ⟨u, hun⟩ X hXw hX1 hX2]
  unfold Spot.Bits.rowGate
  rw [hB, hC, sIn, sComp]
  omega

/-! ## A returned output's port -/

/-- A word some output row holds is an output word, where each output row holds one output port. -/
theorem outPos_lt {c : _root_.Flock.Circuit} (hrep : c.outRowPorts = Array.replicate c.outPorts.size 1)
    {u x b : ℕ} {r : Spot.Bits.RowPos} (ho : Spot.Bits.outPos c u x b = some r) : x < c.outWords := by
  unfold Spot.Bits.outPos at ho
  dsimp only at ho
  obtain ⟨o, hf, -⟩ := Option.map_eq_some_iff.1 ho
  have hmem := List.mem_range.1 (List.mem_of_find?_eq_some hf)
  have hp := List.find?_some hf
  simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
  have hs : c.outRowPorts.size = c.outPorts.size := by rw [hrep, Array.size_replicate]
  have ho' : o < c.outPorts.size := by omega
  rw [FlockSoundness.Discharge.Hidden.outGroup_single hrep ho'] at hp
  dsimp only at hp
  have h1 := FlockSoundness.Discharge.Hidden.outAt_succ c ho'
  have h2 := _root_.Flock.Bridge.outAt_mono c (show o + 1 ≤ c.outPorts.size by omega)
  rw [FlockSoundness.Discharge.HiddenExec.outAt_size] at h2
  have e : c.outPorts[o]!.2 = (c.outPorts.getD o default).2 := by
    rw [getElem!_pos c.outPorts o ho', Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem ho', Option.getD_some]
  omega

/-- **Bits' position of a returned output's bit is its output port's bit**, where each output row holds one output
port: bit `16 k' + b` of output row `nIn + jo`, the leaf's word being word `k'` of output port `jo`. -/
theorem outRowBit_ref {c : _root_.Flock.Circuit} (hrep : c.outRowPorts = Array.replicate c.outPorts.size 1)
    {u k j b : ℕ} {r : Spot.Bits.RowPos} (hj : j < (c.leavesOut.getD k #[]).size)
    (hr : Spot.Bits.outRowBit c u k j b = some r) :
    ∃ jo k', jo < c.outPorts.size ∧ k' < (c.outPorts.getD jo default).2 ∧
      FlockSoundness.Discharge.Hidden.outAt c jo + k' = (c.leavesOut.getD k #[])[j]! ∧
      r = ⟨(c.ports.getD (_root_.Flock.HmOut.nIn c + jo) default).name, u, 16 * k' + b⟩ := by
  unfold Spot.Bits.outRowBit at hr
  rw [Array.getElem?_eq_getElem hj, Option.bind_some] at hr
  rw [getElem!_pos (c.leavesOut.getD k #[]) j hj]
  have hwlt := outPos_lt hrep hr
  obtain ⟨hlt, hle, hk⟩ := FlockSoundness.Discharge.HiddenExec.outPortOf_spec c hwlt
  rw [_root_.Flock.Bridge.outPos_eq hrep hwlt] at hr
  exact ⟨_, _, hlt, hk, by omega, (Option.some.inj hr).symm⟩

/-! ## The members' scope -/

variable {h : Held}

/-- Where the design evaluates, each member's circuit is in `scopeOkH`'s scope. -/
theorem scopeOkH_member {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) {t : ℕ}
    (ht : t < h.ms.length) : FlockSoundness.Discharge.Layout.scopeOkH (memberCircuit h t) = true := by
  rw [← FlockSoundness.Discharge.FailClosed.scopeOkH_eq]
  have h1 := (scoped_wiring_of_two h hd).1
  unfold membersScoped at h1
  exact List.all_eq_true.1 h1 t (List.mem_range.2 ht)

/-- **Where the design evaluates, each member's rows are covered** (`RowsFitS`): its statement circuit is a committed
circuit unrolled, in `scopeOkH`'s scope (`HiddenExec.rowsFit_of_committed`). -/
theorem rowsFit_member {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) {t : ℕ}
    (ht : t < h.ms.length) : FlockSoundness.Discharge.Hidden.Any.RowsFitS (memberCircuit h t) := by
  have hsc := scopeOkH_member hd ht
  have hok := templateOk_of_two h hd ht
  unfold templateOk at hok
  cases hr : FlockDeploy.readOf true (h.xS₀ t) with
  | none => rw [hr] at hok; cases hok
  | some r =>
    rw [hr] at hok
    simp only [Bool.and_eq_true] at hok
    obtain ⟨⟨⟨⟨⟨hho, hty⟩, hps⟩, -⟩, -⟩, -⟩ := hok
    have hty' : r.tags.typed = false := by simpa using hty
    have hm : memberCircuit h t = r.stmt := by simp [memberCircuit, stmtOf, hr]
    rw [hm] at hsc ⊢
    obtain ⟨c, hc⟩ : ∃ c, _root_.Flock.HmOut.parseCommit r.tags r.I.circuitFile r.I.tables = .ok c := by
      cases hD : _root_.Flock.Desc.parseStmt r.tags r.I.circuitFile r.I.tables with
      | error e =>
        rw [hD] at hps
        cases hps
      | ok D =>
        have h1 := _root_.Flock.Desc.parseStmt_stmt hD
        unfold _root_.Flock.HmOut.parseStmt at h1
        unfold _root_.Flock.HmOut.parseCommit
        cases hi : r.tags.isTmpl <;> simp only [hi, Bool.false_eq_true, ↓reduceIte] at h1 ⊢
        · exact ⟨_, h1⟩
        · cases hp : _root_.Flock.HmOut.parseTmpl r.tags r.I.circuitFile r.I.tables with
          | error e =>
            rw [hp] at h1
            cases h1
          | ok c => exact ⟨c, rfl⟩
    have hcirc : r.circ = c := by
      simp only [_root_.Flock.Readable.Read.circ, _root_.Flock.ProvedScope.circuitOf, hho, ↓reduceIte, hc]
    change FlockSoundness.Discharge.Layout.scopeOkH (_root_.Flock.HmOut.unrolled r.circ) = true at hsc
    show FlockSoundness.Discharge.Hidden.Any.RowsFitS (_root_.Flock.HmOut.unrolled r.circ)
    rw [hcirc] at hsc ⊢
    have hrb := FlockSoundness.Discharge.Layout.scopeOkH_rowsB hsc
    rw [FlockSoundness.Discharge.HiddenExec.v1PortsL_unrolled, FlockSoundness.Discharge.HiddenExec.bitRowsL_unrolled]
      at hrb
    exact (FlockSoundness.Discharge.HiddenExec.rowsFit_unrolled c).2
      (FlockSoundness.Discharge.HiddenExec.rowsFit_of_committed hty'
        (FlockSoundness.Discharge.Hidden.committed_of_commit hc) hrb)

/-! ## The copy -/

/-- A written bit is a returned output's gate with Bits' commitment position, at the gate's position. -/
theorem writtenBits_mem {i t : ℕ} {r : Spot.Bits.RowPos} (hy : (i, t, r) ∈ writtenBits h.ms) :
    ∃ ht : t < h.ms.length, ∃ u k j b g, Spot.Bits.outGate h.ms[t].1 h.ms[t].2.size u k j b = some g ∧
      Spot.Bits.outRowBit h.ms[t].1 u k j b = some r ∧
      i = Spot.Bits.base h.ms t + Spot.Bits.posOf (Spot.Bits.keyProg h.ms[t].1 h.ms[t].2.size) g := by
  unfold writtenBits at hy
  obtain ⟨m, hm, hy⟩ := List.mem_flatMap.1 hy
  have hm' := List.mem_range.1 hm
  have hci : h.ms.getD m default = h.ms[m] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hm', Option.getD_some]
  dsimp only at hy
  simp only [hci] at hy
  obtain ⟨u, -, hy⟩ := List.mem_flatMap.1 hy
  obtain ⟨k, -, hy⟩ := List.mem_flatMap.1 hy
  obtain ⟨j, -, hy⟩ := List.mem_flatMap.1 hy
  obtain ⟨b, -, hy⟩ := List.mem_filterMap.1 hy
  cases hg : Spot.Bits.outGate h.ms[m].1 h.ms[m].2.size u k j b with
  | none =>
    rw [hg] at hy
    simp at hy
  | some g =>
    cases hr : Spot.Bits.outRowBit h.ms[m].1 u k j b with
    | none =>
      rw [hg, hr] at hy
      simp at hy
    | some r' =>
      rw [hg, hr] at hy
      simp only [Option.map_some, Option.some.injEq, Prod.mk.injEq] at hy
      obtain ⟨h1, rfl, rfl⟩ := hy
      exact ⟨hm', u, k, j, b, g, hg, hr, h1.symm⟩

/-- **`CopiesAt`'s written half**: where the design evaluates and each member names its ports once, at a written bit
of member `t`'s unit `k` the column `pickAny` chooses is the returned output's computed column, its port bit the
bit's. -/
theorem copiesRef_out {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) (hO : PortsOnce h)
    {i t k : ℕ} {rp : Spot.Bits.RowPos} (hy : (i, t, rp) ∈ writtenBits h.ms) (hleaf : rp.leaf = k) :
    ∃ m, h.ms[t]? = some m ∧ CopiesRef m.1 (pickAny h t k i) (portNamed h t rp.row) rp.bit := by
  subst hleaf
  obtain ⟨ht, u, kk, j, b, g, hg, hr, rfl⟩ := writtenBits_mem hy
  obtain ⟨-, hU, hu⟩ := templateOk_facts h (templateOk_of_two h hd ht) ht
  refine ⟨h.ms[t], List.getElem?_eq_getElem ht, ?_⟩
  have hm : memberCircuit h t = h.ms[t].1 := (by simp [Held.ms] : h.ms[t].1 = stmtOf (h.xS₀ t)).symm
  have hfit := rowsFit_member hd ht
  rw [hm] at hfit
  obtain ⟨⟨hrep, hports⟩, -⟩ := hfit
  have hsc : FlockSoundness.Discharge.Layout.scopeOkH h.ms[t].1 = true := by
    rw [← hm]
    exact scopeOkH_member hd ht
  have hon := (FlockSoundness.Discharge.Layout.scopeOkH_spec hU hsc).2.1
  obtain ⟨hun, x, hoc, hxn, hxg⟩ := outGate_wire hU hu (Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) hg
  obtain ⟨jo, k', hjo, hk', hsum, rfl⟩ := outRowBit_ref hrep hoc.2.1 hr
  have hgN : g < Layout.keyN h.ms[t].1 hU h.ms[t].2.size := lt_of_eq_of_lt hxg.symm
    ((Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
      (Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨u, hun⟩).wire x).isLt
  let y : Fin (keyC h t ht hU).N := ⟨g, hgN⟩
  have hwy : (Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
      (Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨u, hun⟩).wire x = y := Fin.ext hxg
  obtain ⟨cc, hcc, hpick⟩ := pickAny_of_wire ht hU hun y ⟨x, hwy⟩
  have hccx : cc = x := by
    rcases (Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
      (Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨u, hun⟩).shared cc x (hcc.trans hwy.symm) with h' | h'
    · exact h'
    · exact absurd h'.2 (by omega)
  have hpos : _root_.Flock.Bridge.posOf (keyC h t ht hU) y =
      Spot.Bits.posOf (Spot.Bits.keyProg h.ms[t].1 h.ms[t].2.size) g :=
    _root_.Flock.Bridge.posOf_keyProg (c := h.ms[t].1) (n := h.ms[t].2.size) (hU := hU) hu y
  have hq : _root_.Flock.HmOut.nIn h.ms[t].1 + jo < h.ms[t].1.ports.size := (hports jo hjo).1
  have hpn : portNamed h t (h.ms[t].1.ports.getD (_root_.Flock.HmOut.nIn h.ms[t].1 + jo) default).name =
      _root_.Flock.HmOut.nIn h.ms[t].1 + jo := by
    refine portNamed_eq (hO t ht) (by rw [hm]; exact hq) ?_
    rw [hm, getElem!_pos h.ms[t].1.ports _ hq]
    simp [Array.getD, hq]
  show CopiesRef h.ms[t].1
    (pickAny h t u (Spot.Bits.base h.ms t + Spot.Bits.posOf (Spot.Bits.keyProg h.ms[t].1 h.ms[t].2.size) g))
    (portNamed h t (h.ms[t].1.ports.getD (_root_.Flock.HmOut.nIn h.ms[t].1 + jo) default).name) (16 * k' + b)
  rw [← hpos, hpick, hccx, hpn]
  exact Or.inr ⟨hon, hU, kk, j, b, jo, k', hoc.1, hoc.2.1, hoc.2.2.1, hoc.2.2.2.1, hoc.2.2.2.2.1, hoc.2.2.2.2.2,
    hjo, hk', hsum, rfl, rfl⟩

end SpotFlock
