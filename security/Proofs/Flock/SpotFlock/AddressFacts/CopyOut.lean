module

public import Proofs.Flock.SpotFlock.AddressFacts.CopySrc
public import Proofs.Flock.SpotFlock.AddressFacts.WidthOf
public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
public import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
public import Proofs.Flock.Soundness.Discharge.Placed.Setup

/-! `CopiesAt`'s written half where the second stage's design evaluates and each member names its ports once
(`copiesRef_out`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy FlockSoundness FlockSoundness.Discharge

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

theorem outGate_wire {c : _root_.Flock.Circuit} (hU : FlockSoundness.Built c.unit) (hu : c.unit.unitLog ≤ 32)
    {n u k j b g : ℕ} (outs : Finset (Fin (Layout.keyN c hU n))) (hg : Spot.Bits.outGate c n u k j b = some g) :
    ∃ hun : u < n, ∃ x : Fin ((Placed.keyProg c hU n).rows ⟨u, hun⟩).w,
      Layout.OutColAt c hU x.val k j b ∧ ((Placed.keyProg c hU n).rows ⟨u, hun⟩).nIn ≤ x.val ∧
      ((Const.Prog.instC (Placed.keyProg c hU n) outs ⟨u, hun⟩).wire x).val = g := by
  obtain ⟨hun, hk, hj, hb, h1, h2, rfl⟩ := outGate_spec hg
  have hR := _root_.Flock.Bridge.rowsAgree_circuitRows c hU hu
  have hB : ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn = (Placed.circuitRows c hU).nIn := hR.1
  have hC : ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nComp = (Placed.circuitRows c hU).nComp := hR.2.1
  have bI : ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn = c.upv * (Spot.Bits.Rows.ofNet c.unit).nIn := rfl
  have bM : ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nComp =
      c.upv * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) := rfl
  have hkM := Nat.mul_le_mul_right ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) (show k + 1 ≤ c.upv from hk)
  rw [Nat.succ_mul] at hkM
  obtain ⟨xv, hxv⟩ : ∃ xv, xv = c.upv * (Spot.Bits.Rows.ofNet c.unit).nIn +
      (k * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) +
        (1 + (Spot.Bits.outCol c j b - (Spot.Bits.Rows.ofNet c.unit).nIn))) := ⟨_, rfl⟩
  have hx1 : (Placed.circuitRows c hU).nIn ≤ xv := by omega
  have hx2 : xv < (Placed.circuitRows c hU).nIn + (Placed.circuitRows c hU).nComp := by omega
  have eR : (Placed.keyProg c hU n).rows ⟨u, hun⟩ = Placed.circuitRows c hU :=
    Placed.rows_circuitProg c hU n _ _ ⟨u, hun⟩
  have hxw : xv < ((Placed.keyProg c hU n).rows ⟨u, hun⟩).w := by
    rw [congrArg FlockSoundness.Rows.w eR]
    simp only [FlockSoundness.Rows.w]
    omega
  have e1 : n * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn = n * (Placed.circuitRows c hU).nIn := by rw [hB]
  have e2 : u * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nComp = u * (Placed.circuitRows c hU).nComp := by
    rw [hC]
  refine ⟨hun, ⟨xv, hxw⟩, ⟨hk, hj, hb, h1, h2, hxv⟩,
    Nat.le_trans (Nat.le_of_eq (congrArg FlockSoundness.Rows.nIn eR)) hx1, ?_⟩
  have hw := _root_.Flock.Bridge.rowsWireC_keyProg_comp (c := c) (n := n) (hU := hU) ⟨u, hun⟩ ⟨xv, hxw⟩ hx1 hx2
  refine hw.trans ?_
  show 2 + n * (Placed.circuitRows c hU).nIn + u * (Placed.circuitRows c hU).nComp +
      (xv - (Placed.circuitRows c hU).nIn) =
    2 + n * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn + u * ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nComp +
      (k * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) +
        (1 + (Spot.Bits.outCol c j b - (Spot.Bits.Rows.ofNet c.unit).nIn)))
  omega

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

variable {h : Held}

theorem scopeOkH_member {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) {t : ℕ}
    (ht : t < h.ms.length) : FlockSoundness.Discharge.Layout.scopeOkH (memberCircuit h t) = true := by
  rw [← FlockSoundness.Discharge.FailClosed.scopeOkH_eq]
  have h1 := (scoped_wiring_of_two h hd).1
  unfold membersScoped at h1
  exact List.all_eq_true.1 h1 t (List.mem_range.2 ht)

theorem member_commit {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) {t : ℕ}
    (ht : t < h.ms.length) :
    ∃ (r : _root_.Flock.Readable.Read) (c : _root_.Flock.Circuit), r.tags.typed = false ∧
      _root_.Flock.HmOut.parseCommit r.tags r.I.circuitFile r.I.tables = .ok c ∧
      memberCircuit h t = _root_.Flock.HmOut.unrolled c := by
  have hok := templateOk_of_two h hd ht
  unfold templateOk at hok
  cases hr : FlockDeploy.readOf true (h.xS₀ t) with
  | none => rw [hr] at hok; cases hok
  | some r =>
    rw [hr] at hok
    simp only [Bool.and_eq_true] at hok
    obtain ⟨⟨⟨⟨⟨hho, hty⟩, hps⟩, -⟩, -⟩, -⟩ := hok
    have hm : memberCircuit h t = r.stmt := by simp [memberCircuit, stmtOf, hr]
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
    refine ⟨r, c, by simpa using hty, hc, ?_⟩
    rw [hm, ← hcirc]
    rfl

theorem rowsFit_member {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) {t : ℕ}
    (ht : t < h.ms.length) : FlockSoundness.Discharge.Hidden.Any.RowsFitS (memberCircuit h t) := by
  obtain ⟨r, c, hty, hc, hm⟩ := member_commit hd ht
  have hsc := scopeOkH_member hd ht
  rw [hm] at hsc ⊢
  have hrb := FlockSoundness.Discharge.Layout.scopeOkH_rowsB hsc
  rw [FlockSoundness.Discharge.HiddenExec.v1PortsL_unrolled, FlockSoundness.Discharge.HiddenExec.bitRowsL_unrolled]
    at hrb
  exact (FlockSoundness.Discharge.HiddenExec.rowsFit_unrolled c).2
    (FlockSoundness.Discharge.HiddenExec.rowsFit_of_committed hty
      (FlockSoundness.Discharge.Hidden.committed_of_commit hc) hrb)

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

theorem copiesAt_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) (hO : PortsOnce h) :
    CopiesAt h := by
  rintro t k i rp ⟨hleaf, ⟨hf, -⟩ | ⟨-, hf, -, -⟩⟩
  · exact copiesRef_src hd hO (List.mem_of_find?_eq_some hf) hleaf
  · exact copiesRef_out hd hO (List.mem_of_find?_eq_some hf) hleaf

theorem widthFact_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) (hO : PortsOnce h)
    (W : Wiring h) : WidthFact h (addressOf h) :=
  widthFact_of_wiring W (bitsInPorts_of_once hd hO)

theorem copyFact_of_two {s : AuditStatement} {d : Design.Two} {ω : ℕ → ℕ} {keep : ℕ → Bool}
    (hd : two h s = some d) (hO : PortsOnce h) (W : Wiring h) (hrow : RowFact h ω keep (addressOf h)) :
    CopyFact h (addressOf h) :=
  copyFact_of_wiring W (bitsInPorts_of_once hd hO) hrow (copiesAt_of_two hd hO)

end SpotFlock
