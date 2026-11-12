module

public import Proofs.Flock.SpotFlock.WiresTie
public import Proofs.Flock.SpotFlock.WrittenAt
public import Proofs.Flock.SpotFlock.RowsFit

/-!
# What round 2's check of the wires accepts moves every wired source bit

`SecondWiresMove` (`WiresTie`) says that at a registration whose two-stage design evaluates (`two`), where round 2's
check of the wires passes and the three conditions it does not decide hold (`rowsUnder`, `namesOnce`, `wiresWide`),
every source bit at a port a wire names and no link names has a gate to go to (`wiresMove`). This file proves it.

For a source bit of member `t`'s unit `u` at such a port, with the wire `w`:

* `wire_checked`: the check places `t`'s units in the port's rows (`SecondStage.fits`), holds the rows a replay
  block to the units a replay block of the template of the writer's name, and finds no link at the writer's port.
  Where each name is one template's (`names_inj`), that template is the wire's writer as the wiring map finds it
  (`heldWriter_of_names`).
* `room_of_two`: where the design evaluates, unit `u` is in a replay block, `u / n_t`, and every member has room for
  its units of that block.
* `rowAt_lt_of_fits` (`RowsFit`) with `rows_under`: the unit's cell position has a row, one of the writer's units a
  replay block, so the wiring names a unit of the writer (`wireUnit`), and that unit is one of the writer's.
* `wide_wire`: the bit is under the `bits` of a port of the reader's circuit file of the wire's name, so the
  writer's circuit file writes it on the wire's output row (`writesBit`), in every unit (`outGate_unit`,
  `outRowBit_leaf`): a written bit of the writer's unit at the bit's commitment position.
* `writtenBits_inj` (`WrittenAt`): every written bit at that gate is that one, at a port no link names, so the
  boundary record holds none there (`returned`) and the gate is the source bit's (`wireTarget_eq`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling
open Properties.Spot (bitsProgs membersPart memberProg filesReplay unitBase)
open Flock.Bridge.Members

variable (h : Held)

/-! ## Names -/

/-- **Where each name is one template's, two templates of one name are one.** -/
theorem names_inj {q : ℕ} (hn : namesOnce (partsAt h q) = true) {a b : ℕ} (ha : a < h.ms.length)
    (hb : b < h.ms.length) (he : h.names.getD a "" = h.names.getD b "") : a = b := by
  have h0 : ((partsAt h q).toList.map (·.template)).Nodup := of_decide_eq_true hn
  have h1 : (partsAt h q).toList.map (·.template) = (List.range h.ms.length).map fun t => h.names.getD t "" := by
    unfold partsAt
    rw [List.toList_toArray, List.map_map]
    rfl
  rw [h1] at h0
  have hpw := List.pairwise_iff_getElem.1 h0
  have hlen : ((List.range h.ms.length).map fun t => h.names.getD t "").length = h.ms.length := by
    rw [List.length_map, List.length_range]
  rcases Nat.lt_trichotomy a b with hab | hab | hab
  · exfalso
    have hne := hpw a b (by rw [hlen]; exact ha) (by rw [hlen]; exact hb) hab
    simp only [List.getElem_map, List.getElem_range] at hne
    exact hne he
  · exact hab
  · exfalso
    have hne := hpw b a (by rw [hlen]; exact hb) (by rw [hlen]; exact ha) hab
    simp only [List.getElem_map, List.getElem_range] at hne
    exact hne he.symm

/-- **Where each name is one template's, the first template of a template's name is that template.** -/
theorem heldWriter_of_names {q : ℕ} (hn : namesOnce (partsAt h q) = true) {t : ℕ} (ht : t < h.ms.length) :
    heldWriter h (h.names.getD t "") = some t := by
  obtain ⟨t', ht', hle⟩ := heldWriter_le h ht rfl
  have hfind : (List.range h.ms.length).find? (fun a => h.names.getD a "" == h.names.getD t "") = some t' := ht'
  have hp := List.find?_some hfind
  have hname : h.names.getD t' "" = h.names.getD t "" := eq_of_beq hp
  rw [ht', names_inj h hn (Nat.lt_of_le_of_lt hle ht) ht hname]

/-- **Each template's circuit file by its name**, where each name is one template's. -/
theorem find?_circuitsOf {q : ℕ} (hn : namesOnce (partsAt h q) = true) {t : ℕ} (ht : t < h.ms.length) :
    (circuitsOf h).find? (·.1 == h.names.getD t "") = some (h.names.getD t "", h.ms[t].1) := by
  have key : (circuitsOf h).find? (·.1 == h.names.getD t "") =
      (heldWriter h (h.names.getD t "")).map fun a => (h.names.getD a "", (h.ms.getD a default).1) := by
    unfold circuitsOf
    rw [List.find?_toArray, List.find?_map]
    rfl
  have hci : h.ms.getD t default = h.ms[t] := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem ht, Option.getD_some]
  rw [key, heldWriter_of_names h hn ht, Option.map_some, hci]

/-! ## What the check of the wires and the three conditions give of a wire -/

/-- A part that passes is a template's. -/
theorem any_partsAt {q : ℕ} {f : Spot.SecondStage.Part → Bool} (hf : (partsAt h q).any f = true) :
    ∃ t, t < h.ms.length ∧ f (partAt h q t) = true := by
  unfold partsAt at hf
  rw [List.any_toArray] at hf
  obtain ⟨p, hp, hfp⟩ := List.any_eq_true.1 hf
  obtain ⟨t, ht, rfl⟩ := List.mem_map.1 hp
  exact ⟨t, List.mem_range.1 ht, hfp⟩

/-- **What round 2's check of the wires holds of a wire**, where each name is one template's: the reader's port has
rows that place the reader's units a replay block (`SecondStage.fits`); the rows a replay block are the units a
replay block of the wire's writer, the first template of the writer's name; and no link names the writer's output
port. `own` is what the verifier holds of a run, of which the check reads the wires, the rows and the links. -/
theorem wire_checked {q : ℕ} {own : Spot.SecondStage.Own} (hw : own.wires = h.wires) (hrw : own.rows = h.rows)
    (hlk : own.links = h.links) (hn : namesOnce (partsAt h q) = true)
    (hc : Spot.SecondStage.wiresOk (partsAt h q) own = true) {t : ℕ} (ht : t < h.ms.length) {port : String}
    {w : Spot.SecondStage.Wire} (hwire : heldWire h (h.names.getD t "") port = some w) :
    ∃ r tw, heldRows h (h.names.getD t "") port = some r ∧
      Spot.SecondStage.fits r.runs (h.reg.ns.getD t 0) = true ∧ tw < h.ms.length ∧
      heldWriter h w.writer = some tw ∧ r.per = h.reg.ns.getD tw 0 ∧
      heldLink h (h.names.getD tw "") w.wport = none := by
  have hwm : w ∈ h.wires.getD #[] := Array.mem_of_find?_eq_some hwire
  have hwp := Array.find?_some hwire
  have htmpl : w.template = h.names.getD t "" := eq_of_beq (Bool.and_eq_true_iff.1 hwp).1
  have hport : w.port = port := eq_of_beq (Bool.and_eq_true_iff.1 hwp).2
  have hrows : Spot.SecondStage.rowsOf own w.template w.port = heldRows h (h.names.getD t "") port := by
    unfold Spot.SecondStage.rowsOf heldRows
    rw [hrw, hport, htmpl]
  unfold Spot.SecondStage.wiresOk at hc
  rw [hw] at hc
  cases hws : h.wires with
  | none =>
    rw [hws] at hc
    simp at hc
  | some ws =>
    rw [hws] at hc hwm
    have hwm' : w ∈ ws := hwm
    have hcw := Array.all_eq_true_iff_forall_mem.1 hc w hwm'
    simp only [Bool.and_eq_true] at hcw
    obtain ⟨⟨⟨hA, hB⟩, hC⟩, -⟩ := hcw
    obtain ⟨ta, hta, hfa⟩ := any_partsAt h hA
    have hfa' := Bool.and_eq_true_iff.1 hfa
    have hna : h.names.getD ta "" = w.template := eq_of_beq hfa'.1
    have hta' : ta = t := names_inj h hn hta ht (hna.trans htmpl)
    have hplaced : Spot.SecondStage.placed own (partAt h q t) w.port = true := by
      rw [← hta']
      exact hfa'.2
    unfold Spot.SecondStage.placed at hplaced
    have hrows' : Spot.SecondStage.rowsOf own (partAt h q t).template w.port =
        heldRows h (h.names.getD t "") port := by
      rw [← hrows, htmpl]
      rfl
    rw [hrows'] at hplaced
    cases hr : heldRows h (h.names.getD t "") port with
    | none =>
      rw [hr] at hplaced
      simp at hplaced
    | some r =>
      rw [hr] at hplaced
      have hfits : Spot.SecondStage.fits r.runs (h.reg.ns.getD t 0) = true := hplaced
      obtain ⟨tb, htb, hfb⟩ := any_partsAt h hB
      have hfb' := Bool.and_eq_true_iff.1 hfb
      have hnb : h.names.getD tb "" = w.writer := eq_of_beq hfb'.1
      have hany : (Spot.SecondStage.rowsOf own w.template w.port).any (·.per == (partAt h q tb).n) = true := hfb'.2
      rw [hrows, hr] at hany
      have hper : r.per = h.reg.ns.getD tb 0 := eq_of_beq hany
      have hwr : heldWriter h w.writer = some tb := by
        rw [← hnb]
        exact heldWriter_of_names h hn htb
      have hlink : heldLink h (h.names.getD tb "") w.wport = none := by
        have hC' : Spot.SecondStage.linkOf own w.writer w.wport = none := Option.isNone_iff_eq_none.1 hC
        unfold Spot.SecondStage.linkOf at hC'
        rw [hlk] at hC'
        rw [hnb]
        exact hC'
      exact ⟨r, tb, rfl, hfits, htb, hwr, hper, hlink⟩

/-- **Where each run's last row is one of its replay block's rows** (`rowsUnder`), each run of a wired port's rows
has its last unit's row under `per`. -/
theorem rows_under {own : Spot.SecondStage.Own} (hw : own.wires = h.wires) (hrw : own.rows = h.rows)
    (hu : rowsUnder own = true) {t : ℕ} {port : String} {w : Spot.SecondStage.Wire} {r : Spot.SecondStage.Rows}
    (hwire : heldWire h (h.names.getD t "") port = some w) (hr : heldRows h (h.names.getD t "") port = some r) :
    ∀ x ∈ r.runs, x.row + (x.count - 1) * x.step < r.per := by
  have hwm : w ∈ h.wires.getD #[] := Array.mem_of_find?_eq_some hwire
  have hwp := Array.find?_some hwire
  have htmpl : w.template = h.names.getD t "" := eq_of_beq (Bool.and_eq_true_iff.1 hwp).1
  have hport : w.port = port := eq_of_beq (Bool.and_eq_true_iff.1 hwp).2
  have hrows : Spot.SecondStage.rowsOf own w.template w.port = some r := by
    unfold Spot.SecondStage.rowsOf
    rw [hrw, hport, htmpl]
    exact hr
  unfold rowsUnder at hu
  rw [hw] at hu
  have h1 := Array.all_eq_true_iff_forall_mem.1 hu w hwm
  rw [hrows] at h1
  have h2 : (r.runs.all fun x => decide (x.row + (x.count - 1) * x.step < r.per)) = true := h1
  intro x hx
  exact of_decide_eq_true (Array.all_eq_true_iff_forall_mem.1 h2 x hx)

/-- **Where each reader's port is no wider than what its writer writes** (`wiresWide`) and each name is one
template's, the writer's circuit file writes, on the wire's output row, every bit under the `bits` of each port of
the wire's name in the reader's circuit file. -/
theorem wide_wire {q : ℕ} {own : Spot.SecondStage.Own} (hw : own.wires = h.wires)
    (hn : namesOnce (partsAt h q) = true) (hwd : wiresWide (circuitsOf h) own = true) {t tw : ℕ}
    (ht : t < h.ms.length) (htw : tw < h.ms.length) {port : String} {w : Spot.SecondStage.Wire}
    (hwire : heldWire h (h.names.getD t "") port = some w) (hwr : h.names.getD tw "" = w.writer) :
    ∀ p ∈ h.ms[t].1.ports, p.name = port → ∀ β < p.bits, writesBit h.ms[tw].1 w.wport β = true := by
  have hwm : w ∈ h.wires.getD #[] := Array.mem_of_find?_eq_some hwire
  have hwp := Array.find?_some hwire
  have htmpl : w.template = h.names.getD t "" := eq_of_beq (Bool.and_eq_true_iff.1 hwp).1
  have hport : w.port = port := eq_of_beq (Bool.and_eq_true_iff.1 hwp).2
  unfold wiresWide at hwd
  rw [hw] at hwd
  have h1 := Array.all_eq_true_iff_forall_mem.1 hwd w hwm
  rw [htmpl, ← hwr, find?_circuitsOf h hn ht, find?_circuitsOf h hn htw] at h1
  have h2 : (h.ms[t].1.ports.all fun p => p.name != w.port ||
      (List.range p.bits).all fun β => writesBit h.ms[tw].1 w.wport β) = true := h1
  intro p hp hname β hβ
  have h3 := Array.all_eq_true_iff_forall_mem.1 h2 p hp
  rw [Bool.or_eq_true] at h3
  rcases h3 with h3 | h3
  · rw [hname, hport] at h3
    simp at h3
  · exact List.all_eq_true.1 h3 β (List.mem_range.2 hβ)

/-- **A bit a circuit file writes on an output row** (`writesBit`): bit `β % 16` of a returned output leaf of an
instance, with a gate in the key program of one unit, committed at bit `β` of the row at unit 0. -/
theorem writesBit_spec {c : _root_.Flock.Circuit} {port : String} {β : ℕ} (hb : writesBit c port β = true) :
    ∃ k j, k < c.upv ∧ j < (c.leavesOut.getD k #[]).size ∧ (Spot.Bits.outGate c 1 0 k j (β % 16)).isSome ∧
      Spot.Bits.outRowBit c 0 k j (β % 16) = some (Spot.Bits.RowPos.mk port 0 β) := by
  unfold writesBit at hb
  obtain ⟨k, hk, hb⟩ := List.any_eq_true.1 hb
  obtain ⟨j, hj, hb⟩ := List.any_eq_true.1 hb
  rw [Bool.and_eq_true, decide_eq_true_eq] at hb
  exact ⟨k, j, List.mem_range.1 hk, List.mem_range.1 hj, hb.1, hb.2⟩

/-! ## A unit's replay block -/

/-- **Where the two-stage design evaluates, a member's unit is in a replay block with room for every member**: unit
`u` of member `t` is one of the `n_t` units of replay block `u / n_t`, and each member `t'` has its `n_{t'}` units
of that block. The scope holds every proof block to a replay block (`scopeOk`), and the layout gives each replay
block's units room (`filesOk`). -/
theorem room_of_two {s : AuditStatement} (hs : (two h s).isSome) {t u : ℕ} (ht : t < h.ms.length)
    (hu : u < h.ms[t].2.size) :
    0 < h.reg.ns.getD t 0 ∧
      ∀ t' (ht' : t' < h.ms.length), (u / h.reg.ns.getD t 0 + 1) * h.reg.ns.getD t' 0 ≤ h.ms[t'].2.size := by
  obtain ⟨d, hd⟩ := Option.isSome_iff_exists.1 hs
  obtain ⟨l, -, -, hde, hscope, hfiles⟩ := two_eq h hd
  have hpart : d.part = filesReplay h.ms h.reg.ns l := congrArg Design.Two.part hde
  unfold scopeOk at hscope
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, beq_iff_eq] at hscope
  obtain ⟨⟨⟨⟨⟨⟨⟨-, hpo⟩, -⟩, hrep⟩, -⟩, -⟩, -⟩, -⟩ := hscope
  obtain ⟨hnd, -, -, hne, -⟩ := partitionsOk_sound hpo
  have hfit := (filesOk_sound hfiles).2.2.1
  have hv := unitBase_add_lt h.ms ht hu
  obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hrep _ hv)
  rw [hpart] at hρ
  have hρlt : ρ < l.blocks.size := by
    have hρ' := hρ
    unfold Claims.Sampling.replayOf at hρ'
    have hmem := List.mem_of_find?_eq_some hρ'
    rw [List.mem_range, length_blocks_filesReplay] at hmem
    exact hmem
  obtain ⟨hroom, t', ht', p, hp, hv'⟩ :=
    exists_of_mem_slotFiles h.ms hfit ((replayOf_filesReplay h.ms hnd hne hfit hv hρlt).1 hρ)
  have hx' : ρ * h.reg.ns.getD t' 0 + p < h.ms[t'].2.size :=
    Nat.lt_of_lt_of_le (rank_mul_add_lt (Nat.lt_add_one ρ) hp) (hroom t' ht')
  have e := stratumOf_membersPart h.ms ht' hx'
  rw [hv', stratumOf_membersPart h.ms ht hu] at e
  subst e
  have hu' : u = ρ * h.reg.ns.getD t 0 + p := by omega
  have hpos : 0 < h.reg.ns.getD t 0 := by omega
  have hdiv : u / h.reg.ns.getD t 0 = ρ := by
    rw [hu', Nat.add_comm (ρ * h.reg.ns.getD t 0) p, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hp,
      Nat.zero_add]
  rw [hdiv]
  exact ⟨hpos, hroom⟩

/-! ## The gate a wired source bit goes to -/

/-- **The gate a source bit goes to, from the unit its wire names**: where the wiring sends the source bit's unit
to unit `u'` of template `tw` at an output port no link names, the gate is the first written bit of `tw` at that
row, unit and bit, where the boundary record holds no written bit there. -/
theorem wireTarget_eq {x t : ℕ} {r : Spot.Bits.RowPos} {tw u' : ℕ} {wport : String}
    (hs : (x, t, r) ∈ sourceBits h.ms) (hunit : wireUnit h t r.row r.leaf = some (tw, wport, u'))
    (hlink : heldLink h (h.names.getD tw "") wport = none) :
    wireTarget h x = (((writtenBits h.ms).find? fun z =>
      z.2.1 == tw && z.2.2 == Spot.Bits.RowPos.mk wport u' r.bit).map (·.1)).filter fun g => !returned h g := by
  unfold wireTarget
  rw [find?_sourceBits hs]
  show ((wireUnit h t r.row r.leaf).bind fun y =>
    if (heldLink h (h.names.getD y.1 "") y.2.1).isSome then none
    else (((writtenBits h.ms).find? fun z =>
      z.2.1 == y.1 && z.2.2 == Spot.Bits.RowPos.mk y.2.1 y.2.2 r.bit).map (·.1)).filter fun g =>
        !returned h g) = _
  rw [hunit]
  show (if (heldLink h (h.names.getD tw "") wport).isSome then none
    else (((writtenBits h.ms).find? fun z =>
      z.2.1 == tw && z.2.2 == Spot.Bits.RowPos.mk wport u' r.bit).map (·.1)).filter fun g =>
        !returned h g) = _
  rw [hlink]
  rfl

/-- **A wired source bit has a gate to go to**, where the two-stage design evaluates, round 2's check of the wires
passes and the three conditions hold: at a port a wire names and no link names, the wiring map sends the source bit
to a gate (`wireTarget`). `own` is what the verifier holds of a run, of which the checks read the wires, the rows
and the links, the auditor's copies. -/
theorem wireTarget_of_checked {s : AuditStatement} (htwo : (two h s).isSome) {q : ℕ} {own : Spot.SecondStage.Own}
    (hw : own.wires = h.wires) (hrw : own.rows = h.rows) (hlk : own.links = h.links)
    (hc : Spot.SecondStage.wiresOk (partsAt h q) own = true) (hu : rowsUnder own = true)
    (hn : namesOnce (partsAt h q) = true) (hwd : wiresWide (circuitsOf h) own = true) {x t : ℕ}
    {r : Spot.Bits.RowPos} (hs : (x, t, r) ∈ sourceBits h.ms) (hl : heldLink h (h.names.getD t "") r.row = none)
    {w : Spot.SecondStage.Wire} (hwire : heldWire h (h.names.getD t "") r.row = some w) :
    ∃ y, wireTarget h x = some y := by
  obtain ⟨ht, g, -, hrb, -⟩ := mem_sourceBits_rowBit hs
  obtain ⟨hleaf, qp, hrow, hbit⟩ := rowBit_spec hrb
  obtain ⟨rr, tw, hr, hfits, htw, hwr, hper, hlink⟩ := wire_checked h hw hrw hlk hn hc ht hwire
  have hunder := rows_under h hw hrw hu hwire hr
  obtain ⟨hpos, hroom⟩ := room_of_two h htwo ht hleaf
  obtain ⟨x', hx', hx'lt⟩ := rowAt_lt_of_fits hfits (Nat.mod_lt r.leaf hpos) hunder
  have hcell : cellRow rr (r.leaf % h.reg.ns.getD t 0) = some x' := by
    unfold cellRow
    rw [hx', Option.bind_some, ite_eq_left hx'lt]
  obtain ⟨u', hu'⟩ : ∃ u', u' = r.leaf / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + x' := ⟨_, rfl⟩
  have hunit : wireUnit h t r.row r.leaf = some (tw, w.wport, u') := by
    unfold wireUnit
    rw [wireCell_eq h hl hwire hr hwr, hcell, hu']
    rfl
  have hu'lt : u' < h.ms[tw].2.size := by
    rw [hu']
    exact Nat.lt_of_lt_of_le (rank_mul_add_lt (Nat.lt_add_one _) (by rw [← hper]; exact hx'lt)) (hroom tw htw)
  have hfind : (List.range h.ms.length).find? (fun a => h.names.getD a "" == w.writer) = some tw := hwr
  have hfp := List.find?_some hfind
  have hnw : h.names.getD tw "" = w.writer := eq_of_beq hfp
  have hwb := wide_wire h hw hn hwd ht htw hwire hnw _ (port_mem_of_bits hbit) hrow.symm r.bit hbit
  obtain ⟨k, j, hk, hj, hgate, hpos0⟩ := writesBit_spec hwb
  obtain ⟨a, ha⟩ := outGate_unit hgate hu'lt
  have hrbit : Spot.Bits.outRowBit h.ms[tw].1 u' k j (r.bit % 16) =
      some (Spot.Bits.RowPos.mk w.wport u' r.bit) := by
    rw [outRowBit_leaf, hpos0]
    rfl
  have hmem : (Spot.Bits.base h.ms tw + Spot.Bits.posOf (Spot.Bits.keyProg h.ms[tw].1 h.ms[tw].2.size) a, tw,
      Spot.Bits.RowPos.mk w.wport u' r.bit) ∈ writtenBits h.ms :=
    mem_writtenBits_iff.2 ⟨htw, u', k, j, r.bit % 16, a, hu'lt, hk, hj, Nat.mod_lt _ (by omega), ha, hrbit, rfl⟩
  cases hf : (writtenBits h.ms).find? (fun z =>
      z.2.1 == tw && z.2.2 == Spot.Bits.RowPos.mk w.wport u' r.bit) with
  | none =>
    have hno := List.find?_eq_none.1 hf _ hmem
    simp at hno
  | some z =>
    obtain ⟨zy, zt, zr⟩ := z
    have hzp := List.find?_some hf
    have hzt : zt = tw := eq_of_beq (Bool.and_eq_true_iff.1 hzp).1
    have hzr : zr = Spot.Bits.RowPos.mk w.wport u' r.bit := eq_of_beq (Bool.and_eq_true_iff.1 hzp).2
    have hzm : (zy, zt, zr) ∈ writtenBits h.ms := List.mem_of_find?_eq_some hf
    rw [hzt, hzr] at hzm
    have hret : returned h zy = false := by
      unfold returned
      rw [List.any_eq_false]
      rintro ⟨yy, yt, yr⟩ hy hyc
      rw [Bool.and_eq_true] at hyc
      have hyy : yy = zy := eq_of_beq hyc.1
      rw [hyy] at hy
      obtain ⟨h1, h2⟩ := writtenBits_inj hy hzm
      have hyl : (heldLink h (h.names.getD yt "") yr.row).isSome = true := hyc.2
      have hnone : heldLink h (h.names.getD yt "") yr.row = none := by
        rw [h1, h2]
        exact hlink
      rw [hnone] at hyl
      exact Bool.false_ne_true hyl
    refine ⟨zy, ?_⟩
    rw [wireTarget_eq h hs hunit hlink, hf]
    show Option.filter (fun g => !returned h g) (some zy) = some zy
    rw [Option.filter_some, hret]
    rfl

/-- **What the check of the wires accepts, under the three conditions, moves every wired source bit**
(`SecondWiresMove`). -/
theorem secondWiresMove : SecondWiresMove := by
  intro h s st rec draw c₀ htwo hchk hrows hnames hwide
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
      obtain ⟨y, hy⟩ := wireTarget_of_checked h htwo (own := ownOf h st rec draw c₀) rfl rfl rfl hchk hrows
        hnames hwide hs hl hw
      rw [hy]
      rfl

end SpotFlock
