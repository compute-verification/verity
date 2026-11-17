module

public import Proofs.Flock.SpotFlock.NeededBits
public import Proofs.Flock.SpotFlock.WiredCode

/-! What the address facts ask of the held wiring, each condition decided on the held wiring and circuits or stated as a `Prop` of the members' programs, with no run, and what each gives. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart memberProg)

variable (h : Held)

variable {h} in
theorem wiresAlike_spec (hc : wiresAlike h = true) {t : ℕ} (ht : t < h.ms.length) {port : String}
    {w : Spot.SecondStage.Wire} (hw : heldWire h (h.names.getD t "") port = some w) {tw : ℕ}
    (htw : heldWriter h w.writer = some tw) :
    ((memberCircuit h t).ports[portNamed h t port]!).words =
        ((memberCircuit h tw).ports[portNamed h tw w.wport]!).words ∧
      _root_.Flock.PortRoots.schemaOf (memberCircuit h t) (portNamed h t port) =
        _root_.Flock.PortRoots.schemaOf (memberCircuit h tw) (portNamed h tw w.wport) := by
  have hw' : (h.wires.getD #[]).find? (fun w => w.template == h.names.getD t "" && w.port == port) = some w := hw
  have hm : w ∈ h.wires.getD #[] := Array.mem_of_find?_eq_some hw'
  have hf0 := Array.find?_some hw'
  have hf : (w.template == h.names.getD t "" && w.port == port) = true := hf0
  obtain ⟨hwt, hwp⟩ := Bool.and_eq_true_iff.1 hf
  have hp : w.port = port := eq_of_beq hwp
  have h1 : ((List.range h.ms.length).all fun t => !(w.template == h.names.getD t "") || wireAlike h w t) = true :=
    Array.all_eq_true_iff_forall_mem.1 hc w hm
  have h2 : (!(w.template == h.names.getD t "") || wireAlike h w t) = true :=
    List.all_eq_true.1 h1 t (List.mem_range.2 ht)
  rw [hwt, Bool.not_true, Bool.false_or] at h2
  unfold wireAlike at h2
  rw [htw, hp] at h2
  simp only [Bool.and_eq_true, beq_iff_eq] at h2
  exact h2

variable {h} in
theorem returned_of_whole (hc : returnedWhole h = true) {y t : ℕ} {r : Spot.Bits.RowPos}
    (hy : (y, t, r) ∈ writtenBits h.ms) (hl : heldLink h (h.names.getD t "") r.row = none) :
    returned h y = false := by
  have h1 : ((heldLink h (h.names.getD t "") r.row).isSome || !returned h y) = true :=
    List.all_eq_true.1 hc (y, t, r) hy
  rw [hl] at h1
  cases hr : returned h y with
  | false => rfl
  | true =>
    rw [hr] at h1
    simp at h1

variable {h} in
theorem unwired_of_written (hc : writtenUnwired h = true) {y t : ℕ} {r : Spot.Bits.RowPos}
    (hy : (y, t, r) ∈ writtenBits h.ms) (hl : heldLink h (h.names.getD t "") r.row = none) :
    heldWire h (h.names.getD t "") r.row = none := by
  have h1 : ((heldLink h (h.names.getD t "") r.row).isSome ||
      (heldWire h (h.names.getD t "") r.row).isNone) = true := List.all_eq_true.1 hc (y, t, r) hy
  rw [hl] at h1
  cases hw : heldWire h (h.names.getD t "") r.row with
  | none => rfl
  | some w =>
    rw [hw] at h1
    simp at h1

def WrittenOnce : Prop :=
  ∀ a ∈ writtenBits h.ms, ∀ b ∈ writtenBits h.ms, a.1 = b.1 → a = b

variable {h} in
theorem written_find (hO : WrittenOnce h) {y t : ℕ} {r : Spot.Bits.RowPos} (hy : (y, t, r) ∈ writtenBits h.ms) :
    (writtenBits h.ms).find? (·.1 == y) = some (y, t, r) := by
  cases hf : (writtenBits h.ms).find? (·.1 == y) with
  | none =>
    have hno := List.find?_eq_none.1 hf (y, t, r) hy
    simp at hno
  | some b =>
    have hb : b ∈ writtenBits h.ms := List.mem_of_find?_eq_some hf
    have hb1 : b.1 = y := by simpa using List.find?_some hf
    rw [hO b hb (y, t, r) hy hb1]

variable {h} in
theorem returnedWhole_of_once (hO : WrittenOnce h) : returnedWhole h = true := by
  unfold returnedWhole
  rw [List.all_eq_true]
  intro y hy
  show ((heldLink h (h.names.getD y.2.1 "") y.2.2.row).isSome || !returned h y.1) = true
  cases hr : returned h y.1 with
  | false => simp
  | true =>
    unfold returned at hr
    obtain ⟨y', hy', hc⟩ := List.any_eq_true.1 hr
    obtain ⟨h1, h2⟩ := Bool.and_eq_true_iff.1 hc
    have he : y'.1 = y.1 := eq_of_beq h1
    rw [hO y' hy' y hy he] at h2
    rw [h2]
    rfl

def RowsArePorts : Prop :=
  (∀ s ∈ sourceBits h.ms, ∃ q, q < (memberCircuit h s.2.1).ports.size ∧
      (memberCircuit h s.2.1).ports[q]!.name = s.2.2.row) ∧
    ∀ y ∈ writtenBits h.ms, ∃ q, q < (memberCircuit h y.2.1).ports.size ∧
      (memberCircuit h y.2.1).ports[q]!.name = y.2.2.row

def PortsOnce : Prop :=
  ∀ t < h.ms.length, ((memberCircuit h t).ports.toList.map (·.name)).Nodup

variable {h} in
theorem portsOnce_of (hc : portsOnce h = true) : PortsOnce h := by
  intro t ht
  unfold portsOnce at hc
  exact of_decide_eq_true (List.all_eq_true.1 hc t (List.mem_range.2 ht))

theorem portsOnce_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) : PortsOnce h :=
  portsOnce_of (wiring_of_two h hd).2.2.2.2.2.2.2.2.2.1

variable {h} in
theorem portNamed_spec {t : ℕ} {name : String}
    (hq : ∃ q, q < (memberCircuit h t).ports.size ∧ (memberCircuit h t).ports[q]!.name = name) :
    (List.range (memberCircuit h t).ports.size).find? (fun q => (memberCircuit h t).ports[q]!.name == name) =
        some (portNamed h t name) ∧
      portNamed h t name < (memberCircuit h t).ports.size ∧
      (memberCircuit h t).ports[portNamed h t name]!.name = name := by
  obtain ⟨q, hq, hn⟩ := hq
  cases hf : (List.range (memberCircuit h t).ports.size).find?
      (fun q => (memberCircuit h t).ports[q]!.name == name) with
  | none =>
    have hno := List.find?_eq_none.1 hf q (List.mem_range.2 hq)
    simp [hn] at hno
  | some p =>
    have hp : portNamed h t name = p := by simp only [portNamed, hf, Option.getD_some]
    rw [hp]
    exact ⟨rfl, List.mem_range.1 (List.mem_of_find?_eq_some hf), by simpa using List.find?_some hf⟩

def PortsMember : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → r.sts[0]!.c.ports = (memberCircuit h (templateOf h.ms r.u)).ports

def OwnReads : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ own reg, r.ins.registered = some own → reg ∈ own →
    heldLink h (h.names.getD (templateOf h.ms r.u) "") reg.name = none →
    heldWire h (h.names.getD (templateOf h.ms r.u) "") reg.name = none →
    ∀ rec, jsonMsg r.m₁ = some rec → ((Spot.Record.byPort rec).find? (·.1 == reg.value)).isSome = true

def Inside : Prop :=
  ∀ r : CheckedRun h.stage, r.Ok h → ∀ i, r.s.prog.Needs r.b i → returned h (wireMap h i) = false →
    (bitsProgs h.ms).gates[wireMap h i]? ≠ some .input →
      wireMap h i ∈ cutBlock (wireMap h) r.rb ∧
        (r.s.prog.wired (wireMap h) (returned h)).Boundary (cutBlock (wireMap h) r.rb) (wireMap h i) = false

variable {h} in
theorem topological_of_two {s : AuditStatement} {d : Design.Two} (hd : two h s = some d) :
    ∀ ci ∈ h.ms, (memberProg ci.1 ci.2.size).Topological := by
  have hT : ∀ ci ∈ h.ms, FlockSoundness.Built ci.1.unit ∧ ci.1.unit.unitLog ≤ 32 := fun ci hci => by
    obtain ⟨t, hlt, rfl⟩ := List.getElem_of_mem hci
    exact (templateOk_facts h (templateOk_of_two h hd hlt) hlt).2
  exact fun ci hci =>
    Flock.Bridge.Members.topological_memberProg ci.1 (hU := (hT ci hci).1) (hT ci hci).2 ci.2.size

variable {h} in
theorem source_none_of_written (htop : ∀ ci ∈ h.ms, (memberProg ci.1 ci.2.size).Topological) {y t : ℕ}
    {r : Spot.Bits.RowPos} (hy : (y, t, r) ∈ writtenBits h.ms) :
    (sourceBits h.ms).find? (·.1 == y) = none := by
  cases hf : (sourceBits h.ms).find? (·.1 == y) with
  | none => rfl
  | some s =>
    exfalso
    obtain ⟨sx, st, sr⟩ := s
    have hs : (sx, st, sr) ∈ sourceBits h.ms := List.mem_of_find?_eq_some hf
    have hsx : sx = y := by simpa using List.find?_some hf
    obtain ⟨hst, g, hg, hκ, hx⟩ := mem_sourceBits hs
    obtain ⟨ht, a, ha, he⟩ := mem_writtenBits hy
    have hin := (Flock.Bridge.Members.bitsProgs_input h.ms htop hst hg hκ).1
    obtain ⟨⟨ws, hand⟩, -, -⟩ := Flock.Bridge.Members.bitsProgs_written h.ms htop ht ha
    rw [← hx, hsx, he, hand] at hin
    cases hin

variable {h} in
theorem written_not_input (htop : ∀ ci ∈ h.ms, (memberProg ci.1 ci.2.size).Topological) {y t : ℕ}
    {r : Spot.Bits.RowPos} (hy : (y, t, r) ∈ writtenBits h.ms) :
    (bitsProgs h.ms).gates[y]? ≠ some .input := by
  obtain ⟨ht, a, ha, he⟩ := mem_writtenBits hy
  obtain ⟨⟨ws, hand⟩, -, -⟩ := Flock.Bridge.Members.bitsProgs_written h.ms htop ht ha
  rw [he, hand]
  intro hne
  cases hne

variable {h} in
theorem wireUnit_wire {t : ℕ} {port : String} {u : ℕ} {y : ℕ × String × ℕ} (hy : wireUnit h t port u = some y) :
    ∃ w, heldLink h (h.names.getD t "") port = none ∧ heldWire h (h.names.getD t "") port = some w ∧
      heldWriter h w.writer = some y.1 ∧ y.2.1 = w.wport := by
  unfold wireUnit at hy
  obtain ⟨c, hc, rfl⟩ := Option.map_eq_some_iff.1 hy
  obtain ⟨w, rr, hl, hw, -, htw, hp, -⟩ := wireCell_wire h hc
  exact ⟨w, hl, hw, htw, hp⟩

variable {h} in
theorem wireTarget_at {x y t : ℕ} {r : Spot.Bits.RowPos}
    (hf : (sourceBits h.ms).find? (·.1 == x) = some (x, t, r)) (hxy : wireTarget h x = some y) :
    ∃ tw wport u, wireUnit h t r.row r.leaf = some (tw, wport, u) ∧
      heldLink h (h.names.getD tw "") wport = none ∧
      (y, tw, Spot.Bits.RowPos.mk wport u r.bit) ∈ writtenBits h.ms ∧ returned h y = false := by
  unfold wireTarget at hxy
  obtain ⟨s, hs, hxy1⟩ := Option.bind_eq_some_iff.1 hxy
  rw [hf] at hs
  cases hs
  obtain ⟨⟨tw, wport, u⟩, hu, hxy2⟩ := Option.bind_eq_some_iff.1 hxy1
  have hu' : wireUnit h t r.row r.leaf = some (tw, wport, u) := hu
  have hxy3 : (if (heldLink h (h.names.getD tw "") wport).isSome then none
      else (((writtenBits h.ms).find? fun z =>
        z.2.1 == tw && z.2.2 == Spot.Bits.RowPos.mk wport u r.bit).map (·.1)).filter fun g =>
          !returned h g) = some y := hxy2
  cases hv : heldLink h (h.names.getD tw "") wport with
  | some l =>
    rw [hv] at hxy3
    simp at hxy3
  | none =>
    rw [hv] at hxy3
    have hxy4 : (((writtenBits h.ms).find? fun z =>
        z.2.1 == tw && z.2.2 == Spot.Bits.RowPos.mk wport u r.bit).map (·.1)).filter
          (fun g => !returned h g) = some y := hxy3
    obtain ⟨hxy5, hret⟩ := Option.filter_eq_some_iff.1 hxy4
    have hret' : returned h y = false := by
      cases hr : returned h y with
      | false => rfl
      | true =>
        have hret2 : (!returned h y) = true := hret
        rw [hr] at hret2
        cases hret2
    obtain ⟨⟨zx, zt, zr⟩, hz, hzy⟩ := Option.map_eq_some_iff.1 hxy5
    have hzx : zx = y := hzy
    have hzp := List.find?_some hz
    have hzt : zt = tw := eq_of_beq (Bool.and_eq_true_iff.1 hzp).1
    have hzr : zr = Spot.Bits.RowPos.mk wport u r.bit := eq_of_beq (Bool.and_eq_true_iff.1 hzp).2
    have hm : (zx, zt, zr) ∈ writtenBits h.ms := List.mem_of_find?_eq_some hz
    rw [hzx, hzt, hzr] at hm
    exact ⟨tw, wport, u, hu', hv, hm, hret'⟩

end SpotFlock
