module

public import Proofs.Flock.SpotFlock.RowAtFits
public import Proofs.Flock.SpotFlock.WiringRows
public import Proofs.Flock.SpotFlock.OutputsWritten
public import Proofs.Flock.SpotFlock.BitsInUnitsOf

/-!
# The per-template decision gives every wired source bit a gate to go to

`wiresMove_of_rows`: where the design evaluates and `wiresMoveRows` (`Code`) holds, `wiresMove` holds. The design
decides `wiresMoveRows` (a clause of `wiringOk`), so `wiresMove` holds wherever it evaluates (`wiresMove_of_two`).
For a source bit at a port a wire names and no link names (`wireTarget_of_rows`):

* the bit is the first entry at its gate (`source_find`): a gate is one member's, at one place of it (`pos_eq`);
* its unit is one of its member's (`bitsInUnits`), so it is a position of a whole cell, and that position reads one
  of the rows a replay block holds (`cellRow_of_fits`, through `RowAtFits.rowAt_of_fits`);
* the writer's unit it reads at is one of the writer's (`writer_unit_lt`);
* the writer's unit writes the bit (`outputsWritten_words_of_two`): the bit is inside the writer's output port's words,
  which the decision reads off one unit's input columns (`sourceBits_cell`);
* the boundary record holds no written bit at a port no link names (`returnedWhole_of_once`).
-/

@[expose] public section

namespace Flock.Bridge.NeededBitsOf

open Claims.Sampling Properties.Spot

/-- **A source bit is the first entry at its gate**: a gate is one member's, at one place of its program
(`pos_eq`), and a source gate has one commitment position. -/
theorem source_find {ms : Spot.Bits.Members} {x t : ℕ} {r : Spot.Bits.RowPos}
    (hs : (x, t, r) ∈ _root_.SpotFlock.sourceBits ms) :
    (_root_.SpotFlock.sourceBits ms).find? (·.1 == x) = some (x, t, r) := by
  obtain ⟨m, hm, g, hg, -, r', hr', he⟩ := of_mem_sourceBits hs
  have e1 : x = Spot.Bits.base ms m + g := congrArg Prod.fst he
  have e2 : t = m := congrArg (fun e => e.2.1) he
  have e3 : r = r' := congrArg (fun e => e.2.2) he
  have hall : ∀ e ∈ _root_.SpotFlock.sourceBits ms, (e.1 == x) = true → e = (x, t, r) := by
    intro e he' hpe
    obtain ⟨m', hm', g', hg', -, r'', hr'', rfl⟩ := of_mem_sourceBits he'
    have hpe' : Spot.Bits.base ms m' + g' = Spot.Bits.base ms m + g := (beq_iff_eq.1 hpe).trans e1
    obtain ⟨hmm, hgg⟩ := pos_eq ms hm' hm (lt_len_of_input _ hg') (lt_len_of_input _ hg) hpe'
    rw [e1, e2, e3]
    subst hmm
    subst hgg
    rw [hr''] at hr'
    cases hr'
    rfl
  obtain ⟨e, hfe, rfl⟩ := find?_of_all (p := fun e : ℕ × ℕ × Spot.Bits.RowPos => e.1 == x)
    (Q := fun e => e = (x, t, r)) ⟨_, hs, beq_self_eq_true _⟩ hall
  exact hfe

end Flock.Bridge.NeededBitsOf

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-! ## A source bit's row and bit are one of its circuit's -/

/-- A source's commitment position has the same row and bit at every unit. -/
theorem srcPos_cell {c : _root_.Flock.Circuit} {u : ℕ} {s : Spot.Bits.Src} {rp : Spot.Bits.RowPos}
    (hs : Spot.Bits.srcPos c u s = some rp) :
    (Spot.Bits.srcPos c 0 s).map (fun x : Spot.Bits.RowPos => (x.row, x.bit)) = some (rp.row, rp.bit) := by
  have key : ∀ v, (Spot.Bits.srcPos c v s).map (fun x : Spot.Bits.RowPos => (x.row, x.bit)) =
      (Spot.Bits.srcPos c u s).map (fun x : Spot.Bits.RowPos => (x.row, x.bit)) := by
    intro v
    cases s <;> first | rfl | simp [Spot.Bits.srcPos]
  exact (key 0).trans (congrArg (Option.map fun x : Spot.Bits.RowPos => (x.row, x.bit)) hs)

/-- **A source gate's row and bit are one of those its circuit's units read.** -/
theorem cell_mem_sourceCells {c : _root_.Flock.Circuit} {n g : ℕ} {rp : Spot.Bits.RowPos}
    (hr : Spot.Bits.rowBit c n g = some rp) : (rp.row, rp.bit) ∈ sourceCells c := by
  unfold Spot.Bits.rowBit at hr
  dsimp only at hr
  split at hr
  · cases hr
  · rename_i hc
    have hB : ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn ≠ 0 := by
      intro h0
      apply hc
      first | simp [h0] | (rw [h0]; simp)
    unfold sourceCells
    try dsimp only
    exact List.mem_filterMap.2 ⟨(g - 2) % ((Spot.Bits.Rows.ofNet c.unit).stack c.upv).nIn,
      List.mem_range.2 (Nat.mod_lt _ (Nat.pos_of_ne_zero hB)), srcPos_cell hr⟩

variable (h : Held)

variable {h} in
/-- **A source bit's row and bit are one of those its member's circuit's units read.** -/
theorem sourceBits_cell {x t : ℕ} {r : Spot.Bits.RowPos} (hs : (x, t, r) ∈ sourceBits h.ms) :
    (r.row, r.bit) ∈ sourceCells (memberCircuit h t) := by
  obtain ⟨m, hm, g, -, -, r', hr', he⟩ := _root_.Flock.Bridge.NeededBitsOf.of_mem_sourceBits hs
  have e2 : t = m := congrArg (fun e => e.2.1) he
  have e3 : r = r' := congrArg (fun e => e.2.2) he
  rw [e2, e3, memberCircuit_ms h hm]
  exact cell_mem_sourceCells hr'

/-! ## What the decision's clauses say -/

/-- Member `t`'s units are the held ones. -/
theorem units_ms {t : ℕ} (ht : t < h.ms.length) : h.ms[t].2 = h.vus.getD t #[] := by
  simp [Held.ms]

/-- The first template of a name is one of the members. -/
theorem heldWriter_lt {name : String} {tw : ℕ} (htw : heldWriter h name = some tw) : tw < h.ms.length := by
  have h1 : (List.range h.ms.length).find? (fun t => h.names.getD t "" == name) = some tw := htw
  exact List.mem_range.1 (List.mem_of_find?_eq_some h1)

variable {h} in
/-- What `outBit` decides. -/
theorem outBit_spec {c : _root_.Flock.Circuit} {wport : String} {bit : ℕ} (hb : outBit c wport bit = true) :
    ∃ o, o < c.outPorts.size ∧ (c.ports.getD (_root_.Flock.HmOut.nIn c + o) default).name = wport ∧
      bit < 16 * (c.outPorts.getD o default).2 := by
  unfold outBit at hb
  obtain ⟨o, ho, hc⟩ := List.any_eq_true.1 hb
  obtain ⟨h1, h2⟩ := Bool.and_eq_true_iff.1 hc
  exact ⟨o, List.mem_range.1 ho, eq_of_beq h1, of_decide_eq_true h2⟩

/-- At a row a wire names and no link does, `bitGoes` is the wire's writer's `outBit`. -/
theorem bitGoes_spec {t : ℕ} {row : String} {bit : ℕ} {w : Spot.SecondStage.Wire} {tw : ℕ}
    (hl : heldLink h (h.names.getD t "") row = none) (hw : heldWire h (h.names.getD t "") row = some w)
    (htw : heldWriter h w.writer = some tw) (hg : bitGoes h t row bit = true) :
    outBit (memberCircuit h tw) w.wport bit = true := by
  have h1 : bitGoes h t row bit = wireBit h w bit := by
    unfold bitGoes
    rw [hl, hw]
  have h2 : wireBit h w bit = outBit (memberCircuit h tw) w.wport bit := by
    unfold wireBit
    rw [htw]
  rw [h1, h2] at hg
  exact hg

/-- What `cellGoes` decides. -/
theorem cellGoes_spec {t tw : ℕ} {w : Spot.SecondStage.Wire} {r : Spot.SecondStage.Rows}
    (hg : cellGoes h t tw w r = true) :
    heldLink h (h.names.getD tw "") w.wport = none ∧ 0 < h.reg.ns.getD t 0 ∧
      (h.vus.getD t #[]).size % h.reg.ns.getD t 0 = 0 ∧
      (h.vus.getD t #[]).size / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 ≤ (h.vus.getD tw #[]).size ∧
      r.per ≤ h.reg.ns.getD tw 0 ∧ Spot.SecondStage.fits r.runs (h.reg.ns.getD t 0) = true ∧
      ∀ x ∈ r.runs, x.row + (x.count - 1) * x.step < r.per := by
  unfold cellGoes at hg
  obtain ⟨hg, h7⟩ := Bool.and_eq_true_iff.1 hg
  obtain ⟨hg, h6⟩ := Bool.and_eq_true_iff.1 hg
  obtain ⟨hg, h5⟩ := Bool.and_eq_true_iff.1 hg
  obtain ⟨hg, h4⟩ := Bool.and_eq_true_iff.1 hg
  obtain ⟨hg, h3⟩ := Bool.and_eq_true_iff.1 hg
  obtain ⟨h1, h2⟩ := Bool.and_eq_true_iff.1 hg
  exact ⟨Option.isNone_iff_eq_none.1 h1, of_decide_eq_true h2, of_decide_eq_true h3, of_decide_eq_true h4,
    of_decide_eq_true h5, h6, fun x hx => of_decide_eq_true (Array.all_eq_true_iff_forall_mem.1 h7 x hx)⟩

/-- What `wireGoes` decides. -/
theorem wireGoes_spec {t : ℕ} {row : String} {w : Spot.SecondStage.Wire} (hg : wireGoes h t row w = true) :
    ∃ r tw, heldRows h (h.names.getD t "") row = some r ∧ heldWriter h w.writer = some tw ∧
      cellGoes h t tw w r = true := by
  unfold wireGoes at hg
  cases hr : heldRows h (h.names.getD t "") row with
  | none =>
    rw [hr] at hg
    have hf : false = true := hg
    cases hf
  | some r =>
    cases htw : heldWriter h w.writer with
    | none =>
      rw [hr, htw] at hg
      have hf : false = true := hg
      cases hf
    | some tw =>
      rw [hr, htw] at hg
      exact ⟨r, tw, rfl, rfl, hg⟩

/-- At a row a wire names and no link does, `rowGoes` is the wire's `wireGoes`. -/
theorem rowGoes_wire {t : ℕ} {row : String} {w : Spot.SecondStage.Wire}
    (hl : heldLink h (h.names.getD t "") row = none) (hw : heldWire h (h.names.getD t "") row = some w) :
    rowGoes h t row = wireGoes h t row w := by
  unfold rowGoes
  rw [hl, hw]

variable {h} in
/-- **A position of a cell the runs hold reads one of the rows a replay block holds**, where each run's last row is
one of them: the bisection finds the position's run (`rowAt_of_fits`), and its row is at most the run's last. -/
theorem cellRow_of_fits {r : Spot.SecondStage.Rows} {n c : ℕ} (hf : Spot.SecondStage.fits r.runs n = true)
    (hrun : ∀ x ∈ r.runs, x.row + (x.count - 1) * x.step < r.per) (hc : c < n) :
    ∃ x, x < r.per ∧ cellRow r c = some x := by
  obtain ⟨i, hi, hlo, hhi, hrow⟩ := RowAtFits.rowAt_of_fits hf hc
  have hmem : r.runs[i]! ∈ r.runs := by
    rw [getElem!_pos r.runs i hi]
    exact Array.getElem_mem hi
  have h1 := hrun _ hmem
  have h2 : (c - r.runs[i]!.unit) * r.runs[i]!.step ≤ (r.runs[i]!.count - 1) * r.runs[i]!.step :=
    Nat.mul_le_mul_right _ (by omega)
  have hx : r.runs[i]!.row + (c - r.runs[i]!.unit) * r.runs[i]!.step < r.per := by omega
  refine ⟨_, hx, ?_⟩
  unfold cellRow
  rw [hrow]
  show (if r.runs[i]!.row + (c - r.runs[i]!.unit) * r.runs[i]!.step < r.per then
    some (r.runs[i]!.row + (c - r.runs[i]!.unit) * r.runs[i]!.step) else none) = _
  exact ite_eq_left_of_eq_true _ _ (eq_true hx)

/-- The writer's unit a reader's unit reads at is one of the writer's: unit `leaf` of `U` units in whole cells of
`n`, at row `x` of the `per ≤ nw` a replay block holds, reads unit `leaf / n * nw + x` of the writer's
`Uw ≥ U / n * nw`. -/
theorem writer_unit_lt {leaf U n nw Uw x per : ℕ} (hleaf : leaf < U) (hdiv : U % n = 0)
    (hcnt : U / n * nw ≤ Uw) (hper : per ≤ nw) (hx : x < per) : leaf / n * nw + x < Uw := by
  have h0 := Nat.div_add_mod U n
  have h1 : leaf / n < U / n := Nat.div_lt_of_lt_mul (by omega)
  have h2 : (leaf / n + 1) * nw ≤ U / n * nw := Nat.mul_le_mul_right nw h1
  rw [Nat.add_mul, Nat.one_mul] at h2
  omega

/-- **A wired source bit has a gate to go to**, where the design evaluates and the per-template decision holds
(`wiresMoveRows`): the bit's unit reads its row at one of the writer's units (`cellRow_of_fits`, `writer_unit_lt`),
which writes the bit (`outputsWritten_words_of_two`), and the boundary record holds no written bit at a port no link
names (`returnedWhole_of_once`). -/
theorem wireTarget_of_rows {s : AuditStatement} {d : Design.Two} (hd : two h s = some d)
    (hc : wiresMoveRows h = true) {x t : ℕ} {r : Spot.Bits.RowPos} (hs : (x, t, r) ∈ sourceBits h.ms)
    {w : Spot.SecondStage.Wire} (hl : heldLink h (h.names.getD t "") r.row = none)
    (hw : heldWire h (h.names.getD t "") r.row = some w) : ∃ y, wireTarget h x = some y := by
  obtain ⟨ht, hleaf, -⟩ := bitsInUnits h x t r (Or.inl hs)
  rw [units_ms h ht] at hleaf
  have hrow := (sourceBits_row hs).2
  have hcell := sourceBits_cell hs
  unfold wiresMoveRows at hc
  have hct := List.all_eq_true.1 hc t (List.mem_range.2 ht)
  obtain ⟨hc1, hc2⟩ := Bool.and_eq_true_iff.1 hct
  have hrg : rowGoes h t r.row = true := List.all_eq_true.1 hc1 r.row hrow
  have hbg : bitGoes h t r.row r.bit = true := List.all_eq_true.1 hc2 (r.row, r.bit) hcell
  rw [rowGoes_wire h hl hw] at hrg
  obtain ⟨rr, tw, hrr, htw, hcg⟩ := wireGoes_spec h hrg
  obtain ⟨hnl, hpos, hdiv, hcnt, hper, hfit, hrun⟩ := cellGoes_spec h hcg
  have htwl := heldWriter_lt h htw
  obtain ⟨o, ho, hname, hbit⟩ := outBit_spec (bitGoes_spec h hl hw htw hbg)
  obtain ⟨c, hcp, hcr⟩ := cellRow_of_fits hfit hrun (Nat.mod_lt r.leaf hpos)
  have hu : wireUnit h t r.row r.leaf =
      some (tw, w.wport, r.leaf / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + c) := by
    have h1 := wireCell_eq h hl hw hrr htw (r.leaf % h.reg.ns.getD t 0)
    rw [hcr] at h1
    unfold wireUnit
    rw [h1]
    try rfl
  have hul : r.leaf / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + c < (h.vus.getD tw #[]).size :=
    writer_unit_lt hleaf hdiv hcnt hper hcp
  rw [← units_ms h htwl] at hul
  rw [memberCircuit_ms h htwl] at ho hname hbit
  obtain ⟨gx, hgx⟩ := outputsWritten_words_of_two h hd htwl hul ho hbit
  rw [hname] at hgx
  obtain ⟨⟨zx, zt, zr⟩, hz, hzm, hzt, hzr⟩ := _root_.Flock.Bridge.NeededBitsOf.find?_of_all
    (p := fun z : ℕ × ℕ × Spot.Bits.RowPos => z.2.1 == tw &&
      z.2.2 == Spot.Bits.RowPos.mk w.wport (r.leaf / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + c) r.bit)
    (Q := fun b => b ∈ writtenBits h.ms ∧ b.2.1 = tw ∧
      b.2.2 = Spot.Bits.RowPos.mk w.wport (r.leaf / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + c) r.bit)
    ⟨_, hgx, Bool.and_eq_true_iff.2 ⟨beq_self_eq_true _, beq_self_eq_true _⟩⟩
    (fun b hb hpb => ⟨hb, eq_of_beq (Bool.and_eq_true_iff.1 hpb).1, eq_of_beq (Bool.and_eq_true_iff.1 hpb).2⟩)
  have hzt' : zt = tw := hzt
  have hzr' : zr = Spot.Bits.RowPos.mk w.wport (r.leaf / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + c) r.bit := hzr
  have hret : returned h zx = false := by
    refine returned_of_whole (returnedWhole_of_once (writtenOnce_held h)) hzm ?_
    rw [hzt', hzr']
    exact hnl
  refine ⟨zx, ?_⟩
  unfold wireTarget
  refine Option.bind_eq_some_iff.2 ⟨(x, t, r), _root_.Flock.Bridge.NeededBitsOf.source_find hs, ?_⟩
  refine Option.bind_eq_some_iff.2
    ⟨(tw, w.wport, r.leaf / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + c), hu, ?_⟩
  show (if (heldLink h (h.names.getD tw "") w.wport).isSome then none
    else (((writtenBits h.ms).find? fun z => z.2.1 == tw &&
      z.2.2 == Spot.Bits.RowPos.mk w.wport (r.leaf / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + c) r.bit).map
        (·.1)).filter fun g => !returned h g) = some zx
  rw [hnl]
  show ((((writtenBits h.ms).find? fun z => z.2.1 == tw &&
      z.2.2 == Spot.Bits.RowPos.mk w.wport (r.leaf / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + c) r.bit).map
        (·.1)).filter fun g => !returned h g) = some zx
  refine Option.filter_eq_some_iff.2 ⟨congrArg (Option.map (·.1)) hz, ?_⟩
  show (!returned h zx) = true
  rw [hret]
  rfl

/-- **Every wired source bit has a gate to go to** (`wiresMove`, stated bit by bit), where the design evaluates and
the per-template decision holds (`wiresMoveRows`). -/
theorem wiresMove_of_rows {s : AuditStatement} {d : Design.Two} (hd : two h s = some d)
    (hc : wiresMoveRows h = true) : wiresMove h = true := by
  unfold wiresMove
  rw [List.all_eq_true]
  rintro ⟨x, t, r⟩ hs
  show ((heldLink h (h.names.getD t "") r.row).isSome || (heldWire h (h.names.getD t "") r.row).isNone ||
    (wireTarget h x).isSome) = true
  cases hl : heldLink h (h.names.getD t "") r.row with
  | some l => rfl
  | none =>
    cases hw : heldWire h (h.names.getD t "") r.row with
    | none => rfl
    | some w =>
      obtain ⟨y, hy⟩ := wireTarget_of_rows h hd hc hs hl hw
      rw [hy]
      rfl

/-- **Where the design evaluates, every wired source bit has a gate to go to**: the design decides it once per
template (`wiring_of_two`, `wiresMoveRows`), which gives it bit by bit (`wiresMove_of_rows`). -/
theorem wiresMove_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) : WiresMove h :=
  wiresMove_of_rows h hd (wiring_of_two h hd).2.2.2.1

end SpotFlock
