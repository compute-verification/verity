module

public import Proofs.Flock.Claims.Sampling.Rewire
public import Proofs.Flock.SpotFlock.AddressesAt
public import Proofs.Flock.SpotFlock.EntriesAt
public import Proofs.Flock.SpotFlock.Sessions

/-!
# The wiring round 2 checks, as a map on the members' gates

Round 2's statement check (`Spot.SecondStage.readOk`) holds each registered read of a session to a row: at a port a
wire of the auditor's copies names, the row of the reading unit's writer unit in the same sampled replay block
(`wiredRow`); at a port a link names, the row of the boundary port the link names (`linkedRow`), or, where the interior
record registers a value of the link's name, that value's row. This file states that wiring from the auditor's copies
alone (`Held`: its `wires`, `links`, `rows`, `names` and the registration's cell sizes), with no run, and proves that it
is the one an accepted run's registered reads are held to.

* **The lookups** (`heldLink`, `heldWire`, `heldRows`, `heldWriter`): round 2's own (`linkOf`, `wireOf`, `rowsOf`,
  `writerOf`) at the auditor's copies; they read nothing of a run (`writerOf_ownOf`).
* **A cell** is a template's units in one replay block. `cellRow` is the row a cell position reads inside its replay
  block's rows; `wireCell` sends a reading cell position to its writer's template, output port and cell position, and
  `linkCell` to the boundary port the link names, its rows a replay block and the row among them.
* **A member's unit** `u` of template `t` is position `u % n_t` of replay block `u / n_t`'s cell (`filesBlockOf`).
  `wireUnit` sends it to its writer's unit in the same replay block, and `linkUnit` to the boundary port's row
  (`wireUnit_block`, `linkUnit_block`).
* **The gates** (`wireTarget`, `wireMap`): a source bit at a wired port goes to the gate on the same bit of its writer
  unit's output row, where the writer's key program has one, no link names the writer's output port and the boundary
  record holds no written bit at that gate (`wireTarget_spec`); every other gate to itself. A linked port's source
  bits stay: the boundary port is no gate of the members' programs, and the auditor's copies name no unit that writes
  it. `wiredProg` is the members' programs read through that map, returning the written bits the boundary record holds
  (`Prog.wired` at `returned`: those at an output port a link names), so no gate the map moves a source bit to is
  returned (`returned_wireMap`).
* **Two conditions on the auditor's copies, decided**: every wired source bit has a gate to go to (`wiresMove`,
  `wireTarget_of_wiresMove`), and no unit is wired to itself (`noSelfWire`, `wireUnit_ne_self`).
* **A returned output wired inside** (`noReturnedWired`, `writer_not_linked`): where a link names a wire's writer's
  output port, round 2 holds the writer's session to the link's record there and the wired reader to the interior
  record's port, and holds neither entry to the other; the map keeps the two gates apart, and the condition that
  there is no such port is decided from the auditor's copies.
* **What round 2 checks** (`wired_read`, `linked_read`): in an accepted run, each registered read of a template's
  session at a wired port is of the writer's output port of the interior record, in a tree of the writer's cell size a
  sampled block, each reading slot at the row `wireRow` names; and at a linked port it is of the link's value, each
  slot at the row `linkRow` names in the boundary port, or `linkRowIn` names in the interior record's value of that
  name. Which of the two a linked read is held to is the interior record's, which the run states, not the auditor's
  copies. The row a slot reads at a wired port and the unit its gates go to are one cell position of the writer
  (`wireRow_unit`).
* **One slot a gate** (`addresses_one_way`): under `Addresses`, every accepted run that needs a gate reads it as a
  registered value, or every one reads it off the interior record, since an address has one slot.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Properties.Spot (bitsProgs membersPart)
open Lean (Json)

variable (h : Held)

/-! ## Round 2's lookups, at the auditor's copies -/

/-- **The link that names template `t`'s session port `port`**, as round 2 finds it (`SecondStage.linkOf`). -/
def heldLink (t port : String) : Option Spot.SecondStage.Link :=
  (h.links.getD #[]).find? fun l => l.template == t && l.port == port

/-- **The wire that names template `t`'s session port `port`**, as round 2 finds it (`SecondStage.wireOf`). -/
def heldWire (t port : String) : Option Spot.SecondStage.Wire :=
  (h.wires.getD #[]).find? fun w => w.template == t && w.port == port

/-- **The rows template `t`'s units read at port `port`**, as round 2 finds them (`SecondStage.rowsOf`). -/
def heldRows (t port : String) : Option Spot.SecondStage.Rows :=
  h.rows.find? fun r => r.template == t && r.port == port

/-- **The template named `name`**: the first of that name, whose part round 2 takes for a wire's writer
(`SecondStage.writerOf`). -/
def heldWriter (name : String) : Option ℕ :=
  (List.range h.ms.length).find? fun t => h.names.getD t "" == name

/-- **A wire's writer's part is the part of the first template of the writer's name**, in every run. -/
theorem writerOf_ownOf (st : Roots) (rec smp : Json) (c₀ : FlockDeploy.Coin) (w : Spot.SecondStage.Wire) :
    Spot.SecondStage.writerOf (ownOf h st rec smp c₀) w =
      (heldWriter h w.writer).map (partAt h (h.reg.sample (src c₀)).size) := by
  have key : (((List.range h.ms.length).map (partAt h (h.reg.sample (src c₀)).size)).toArray).find?
      (fun p => p.template == w.writer) =
        (heldWriter h w.writer).map (partAt h (h.reg.sample (src c₀)).size) := by
    rw [List.find?_toArray, List.find?_map]
    rfl
  exact key

/-! ## Cells -/

/-- **The row a cell position reads inside its replay block's rows**, at rows `r`: its unit's (`SecondStage.rowAt`),
one of the `per` the replay block holds; none where the rows give it none. -/
def cellRow (r : Spot.SecondStage.Rows) (c : ℕ) : Option ℕ :=
  (Spot.SecondStage.rowAt r.runs c).bind fun x => if x < r.per then some x else none

/-- **Where position `c` of template `t`'s cell reads at a port a wire names and no link does**: the writer's
template, its output port, and the writer's position in its own cell of the same replay block. -/
def wireCell (t : ℕ) (port : String) (c : ℕ) : Option (ℕ × String × ℕ) :=
  match heldLink h (h.names.getD t "") port, heldWire h (h.names.getD t "") port,
      heldRows h (h.names.getD t "") port with
  | none, some w, some r => (heldWriter h w.writer).bind fun tw => (cellRow r c).map fun x => (tw, w.wport, x)
  | _, _, _ => none

/-- **Where position `c` of template `t`'s cell reads at a port a link names**: the boundary port the link names, its
rows a replay block, and the row among them. -/
def linkCell (t : ℕ) (port : String) (c : ℕ) : Option (String × ℕ × ℕ) :=
  match heldLink h (h.names.getD t "") port, heldRows h (h.names.getD t "") port with
  | some l, some r => (cellRow r c).map fun x => (l.value, r.per, x)
  | _, _ => none

theorem wireCell_eq {t : ℕ} {port : String} {w : Spot.SecondStage.Wire} {r : Spot.SecondStage.Rows} {tw : ℕ}
    (hl : heldLink h (h.names.getD t "") port = none) (hw : heldWire h (h.names.getD t "") port = some w)
    (hr : heldRows h (h.names.getD t "") port = some r) (htw : heldWriter h w.writer = some tw) (c : ℕ) :
    wireCell h t port c = (cellRow r c).map fun x => (tw, w.wport, x) := by
  have h1 : wireCell h t port c =
      (heldWriter h w.writer).bind fun tw => (cellRow r c).map fun x => (tw, w.wport, x) := by
    unfold wireCell
    rw [hl, hw, hr]
  rw [h1, htw]
  rfl

theorem linkCell_eq {t : ℕ} {port : String} {l : Spot.SecondStage.Link} {r : Spot.SecondStage.Rows}
    (hl : heldLink h (h.names.getD t "") port = some l) (hr : heldRows h (h.names.getD t "") port = some r) (c : ℕ) :
    linkCell h t port c = (cellRow r c).map fun x => (l.value, r.per, x) := by
  unfold linkCell
  rw [hl, hr]

/-! ## Units -/

/-- **Where unit `u` of member `t` reads at a wired port**: its writer's template, output port and unit. Unit `u` is
position `u % n_t` of replay block `u / n_t`'s cell, and its writer's unit the writer's position in the same replay
block's cell of the writer's template. -/
def wireUnit (t : ℕ) (port : String) (u : ℕ) : Option (ℕ × String × ℕ) :=
  (wireCell h t port (u % h.reg.ns.getD t 0)).map fun y =>
    (y.1, y.2.1, u / h.reg.ns.getD t 0 * h.reg.ns.getD y.1 0 + y.2.2)

/-- **Where unit `u` of member `t` reads at a linked port**: the boundary port the link names and the row, its replay
block's rows from `(u / n_t) · per`. -/
def linkUnit (t : ℕ) (port : String) (u : ℕ) : Option (String × ℕ) :=
  (linkCell h t port (u % h.reg.ns.getD t 0)).map fun y => (y.1, u / h.reg.ns.getD t 0 * y.2.1 + y.2.2)

/-- **A unit's writer, by replay block**: unit `c + ρ · n_t` of member `t`, position `c < n_t` of replay block `ρ`'s
cell, reads its writer's unit of the same replay block, `ρ · n_w` plus the writer's position `wireCell` gives `c`. -/
theorem wireUnit_block {t : ℕ} (port : String) (ρ : ℕ) {c : ℕ} (hc : c < h.reg.ns.getD t 0) :
    wireUnit h t port (c + ρ * h.reg.ns.getD t 0) =
      (wireCell h t port c).map fun y => (y.1, y.2.1, ρ * h.reg.ns.getD y.1 0 + y.2.2) := by
  have hpos : 0 < h.reg.ns.getD t 0 := Nat.lt_of_le_of_lt (Nat.zero_le c) hc
  unfold wireUnit
  rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hc,
    Nat.zero_add]

/-- **A unit's boundary row, by replay block**: unit `c + ρ · n_t` of member `t` reads, at a linked port, row
`ρ · per` plus the row `linkCell` gives `c`. -/
theorem linkUnit_block {t : ℕ} (port : String) (ρ : ℕ) {c : ℕ} (hc : c < h.reg.ns.getD t 0) :
    linkUnit h t port (c + ρ * h.reg.ns.getD t 0) =
      (linkCell h t port c).map fun y => (y.1, ρ * y.2.1 + y.2.2) := by
  have hpos : 0 < h.reg.ns.getD t 0 := Nat.lt_of_le_of_lt (Nat.zero_le c) hc
  unfold linkUnit
  rw [Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hc, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hc,
    Nat.zero_add]

/-- **A cell position a wire sends somewhere**: no link names its port, a wire does, the port has rows, the wire's
writer is a template's name, and the position has a row among them. -/
theorem wireCell_wire {t : ℕ} {port : String} {c : ℕ} {y : ℕ × String × ℕ} (hy : wireCell h t port c = some y) :
    ∃ w r, heldLink h (h.names.getD t "") port = none ∧ heldWire h (h.names.getD t "") port = some w ∧
      heldRows h (h.names.getD t "") port = some r ∧ heldWriter h w.writer = some y.1 ∧ y.2.1 = w.wport ∧
      cellRow r c = some y.2.2 := by
  unfold wireCell at hy
  cases hl : heldLink h (h.names.getD t "") port with
  | some l =>
    rw [hl] at hy
    cases hy
  | none =>
    cases hw : heldWire h (h.names.getD t "") port with
    | none =>
      rw [hl, hw] at hy
      cases hy
    | some w =>
      cases hr : heldRows h (h.names.getD t "") port with
      | none =>
        rw [hl, hw, hr] at hy
        cases hy
      | some r =>
        rw [hl, hw, hr] at hy
        have hy' : ((heldWriter h w.writer).bind fun tw => (cellRow r c).map fun x => (tw, w.wport, x)) = some y := hy
        obtain ⟨tw, htw, hy2⟩ := Option.bind_eq_some_iff.1 hy'
        obtain ⟨x, hx, hy3⟩ := Option.map_eq_some_iff.1 hy2
        subst hy3
        exact ⟨w, r, rfl, rfl, rfl, htw, rfl, hx⟩

/-- **No unit is wired to itself**, decided: no cell position of a template reads, at a port a wire names, that
position of the same template. A unit's wire then leaves its proof block. -/
def noSelfWire : Bool :=
  (List.range h.ms.length).all fun t => (h.wires.getD #[]).all fun w =>
    (List.range (h.reg.ns.getD t 0)).all fun c =>
      match wireCell h t w.port c with
      | some y => !(y.1 == t && y.2.2 == c)
      | none => true

/-- Where no unit is wired to itself, no cell position's wire names the position itself. -/
theorem wireCell_ne_self (hc : noSelfWire h = true) {t : ℕ} (ht : t < h.ms.length) {port : String} {c : ℕ}
    (hcn : c < h.reg.ns.getD t 0) (wport : String) : wireCell h t port c ≠ some (t, wport, c) := by
  intro hcell
  obtain ⟨w, r, -, hw, -, -, -, -⟩ := wireCell_wire h hcell
  have hwm : w ∈ h.wires.getD #[] := Array.mem_of_find?_eq_some hw
  have hwp := Array.find?_some hw
  have hport : w.port = port := eq_of_beq (Bool.and_eq_true_iff.1 hwp).2
  have h1 := List.all_eq_true.1 hc t (List.mem_range.2 ht)
  have h2 := Array.all_eq_true_iff_forall_mem.1 h1 w hwm
  have h3 := List.all_eq_true.1 h2 c (List.mem_range.2 hcn)
  simp [hport, hcell] at h3

/-- **Where no unit is wired to itself, no member's unit's wire names the unit itself**, so the gate a source bit of
the unit goes to is no gate of the unit. -/
theorem wireUnit_ne_self (hc : noSelfWire h = true) {t : ℕ} (ht : t < h.ms.length) (port wport : String) {u : ℕ}
    (hn : 0 < h.reg.ns.getD t 0) : wireUnit h t port u ≠ some (t, wport, u) := by
  intro hu
  unfold wireUnit at hu
  obtain ⟨⟨tw, wp, x⟩, hcell, hy⟩ := Option.map_eq_some_iff.1 hu
  have h1 : tw = t := congrArg Prod.fst hy
  have h3 : u / h.reg.ns.getD t 0 * h.reg.ns.getD tw 0 + x = u := congrArg (fun z => z.2.2) hy
  rw [h1] at hcell h3
  have hx : x = u % h.reg.ns.getD t 0 :=
    Nat.add_left_cancel (h3.trans (Nat.div_add_mod' u (h.reg.ns.getD t 0)).symm)
  rw [hx] at hcell
  exact wireCell_ne_self h hc ht (Nat.mod_lt _ hn) wp hcell

/-! ## Gates -/

/-- **Every source bit**: each source gate a unit reads that has a commitment position (`Bits.rowBit`), with its
position in the members' programs, its member, and that commitment position at the member's own units. -/
def sourceBits (ms : Spot.Bits.Members) : List (ℕ × ℕ × Spot.Bits.RowPos) :=
  (List.range ms.length).flatMap fun m =>
    let ci := ms.getD m default
    (List.range (Spot.Bits.keyProg ci.1 ci.2.size).inputs).filterMap fun g =>
      if (Spot.Bits.consts ci.1 ci.2.size g).isSome then none
      else (Spot.Bits.rowBit ci.1 ci.2.size g).map fun r => (Spot.Bits.base ms m + g, m, r)

/-- **Every written bit**: each returned output leaf bit's gate (`Bits.outGate`), with its position in the members'
programs, its member, and its commitment position at the member's own units (`Bits.outRowBit`). -/
def writtenBits (ms : Spot.Bits.Members) : List (ℕ × ℕ × Spot.Bits.RowPos) :=
  (List.range ms.length).flatMap fun m =>
    let ci := ms.getD m default
    let kp := Spot.Bits.keyProg ci.1 ci.2.size
    (List.range ci.2.size).flatMap fun u => (List.range ci.1.upv).flatMap fun k =>
      (List.range (ci.1.leavesOut.getD k #[]).size).flatMap fun j => (List.range 16).filterMap fun b =>
        match (Spot.Bits.outGate ci.1 ci.2.size u k j b).map (Spot.Bits.posOf kp), Spot.Bits.outRowBit ci.1 u k j b with
        | some x, some r => some (Spot.Bits.base ms m + x, m, r)
        | _, _ => none

/-- **No returned output is wired inside**, decided: no wire's writer's output port is a port a link names. A link
names a template's output port where its replay block returns that output, and round 2 then holds the writer's own
session to the link's record at that port and a reader wired to it to the interior record's port, two entries it
does not hold to each other. -/
def noReturnedWired : Bool :=
  (h.wires.getD #[]).all fun w => (heldLink h w.writer w.wport).isNone

/-- Where no returned output is wired inside, no link names a wire's writer's output port. -/
theorem writer_not_linked (hc : noReturnedWired h = true) {t port : String} {w : Spot.SecondStage.Wire}
    (hw : heldWire h t port = some w) : heldLink h w.writer w.wport = none := by
  have hm : w ∈ h.wires.getD #[] := Array.mem_of_find?_eq_some hw
  have hn : (heldLink h w.writer w.wport).isNone = true := Array.all_eq_true_iff_forall_mem.1 hc w hm
  exact Option.isNone_iff_eq_none.1 hn

/-- **A gate the boundary record holds**: a written bit at an output port a link names for its template, which its
replay block returns. Every other written bit is one the interior record alone holds. -/
def returned (x : ℕ) : Bool :=
  (writtenBits h.ms).any fun y => y.1 == x && (heldLink h (h.names.getD y.2.1 "") y.2.2.row).isSome

/-- **The gate a source bit is wired to**: a source bit at a port a wire names goes to the gate on the same bit of
its writer unit's output row (`wireUnit`), where the writer's program has a gate there, no link names the writer's
output port and the boundary record holds no written bit at that gate; none for every other position. At a gate the
boundary record holds, the reader's source bit and the writer's gate stay two gates: round 2 holds them to two
entries. -/
def wireTarget (x : ℕ) : Option ℕ :=
  ((sourceBits h.ms).find? (·.1 == x)).bind fun s =>
    (wireUnit h s.2.1 s.2.2.row s.2.2.leaf).bind fun y =>
      if (heldLink h (h.names.getD y.1 "") y.2.1).isSome then none
      else (((writtenBits h.ms).find? fun z =>
        z.2.1 == y.1 && z.2.2 == Spot.Bits.RowPos.mk y.2.1 y.2.2 s.2.2.bit).map (·.1)).filter fun g =>
          !returned h g

/-- **The wiring map**: a source bit goes to the gate it is wired to (`wireTarget`), every other position to
itself. -/
def wireMap (x : ℕ) : ℕ := (wireTarget h x).getD x

/-- **The members' programs read through the wiring map**, returning the gates the boundary record holds
(`Prog.wired`). -/
def wiredProg : Prog := (bitsProgs h.ms).wired (wireMap h) (returned h)

theorem wireMap_of_target {x y : ℕ} (hxy : wireTarget h x = some y) : wireMap h x = y := by
  simp only [wireMap, hxy, Option.getD_some]

theorem wireMap_of_no_target {x : ℕ} (hx : wireTarget h x = none) : wireMap h x = x := by
  simp only [wireMap, hx, Option.getD_none]

/-- A position that is no source bit goes to itself. -/
theorem wireMap_of_not_source {x : ℕ} (hx : (sourceBits h.ms).find? (·.1 == x) = none) : wireMap h x = x :=
  wireMap_of_no_target h (by simp only [wireTarget, hx, Option.bind_none])

/-- **The gate a source bit goes to is its writer unit's gate at the wire's port, on the same bit**: the source bit
is member `t`'s at commitment position `r`, the wiring sends unit `r.leaf` at port `r.row` to unit `u` of template
`tw` at its output port `wport` (`wireUnit`), which no link names, the gate is `tw`'s written bit at row `wport`,
leaf `u`, bit `r.bit`, and the boundary record holds no written bit at it. -/
theorem wireTarget_spec {x y : ℕ} (hxy : wireTarget h x = some y) :
    ∃ t r tw wport u, (x, t, r) ∈ sourceBits h.ms ∧ wireUnit h t r.row r.leaf = some (tw, wport, u) ∧
      heldLink h (h.names.getD tw "") wport = none ∧
      (y, tw, Spot.Bits.RowPos.mk wport u r.bit) ∈ writtenBits h.ms ∧ returned h y = false := by
  unfold wireTarget at hxy
  obtain ⟨⟨sx, st, sr⟩, hs, hxy1⟩ := Option.bind_eq_some_iff.1 hxy
  obtain ⟨⟨tw, wport, u⟩, hu, hxy2⟩ := Option.bind_eq_some_iff.1 hxy1
  have hsp := List.find?_some hs
  have hsx : sx = x := eq_of_beq hsp
  have hsm : (x, st, sr) ∈ sourceBits h.ms := by
    rw [← hsx]
    exact List.mem_of_find?_eq_some hs
  have hxy3 : (if (heldLink h (h.names.getD tw "") wport).isSome then none
      else (((writtenBits h.ms).find? fun z =>
        z.2.1 == tw && z.2.2 == Spot.Bits.RowPos.mk wport u sr.bit).map (·.1)).filter fun g =>
          !returned h g) = some y := hxy2
  cases hv : heldLink h (h.names.getD tw "") wport with
  | some l =>
    rw [hv] at hxy3
    simp at hxy3
  | none =>
    rw [hv] at hxy3
    have hxy4 : (((writtenBits h.ms).find? fun z =>
        z.2.1 == tw && z.2.2 == Spot.Bits.RowPos.mk wport u sr.bit).map (·.1)).filter
          (fun g => !returned h g) = some y := by
      first
        | exact hxy3
        | simpa using hxy3
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
    have hzr : zr = Spot.Bits.RowPos.mk wport u sr.bit := eq_of_beq (Bool.and_eq_true_iff.1 hzp).2
    have hm : (zx, zt, zr) ∈ writtenBits h.ms := List.mem_of_find?_eq_some hz
    rw [hzx, hzt, hzr] at hm
    exact ⟨st, sr, tw, wport, u, hsm, hu, hv, hm, hret'⟩

/-- **No gate the wiring moves a source bit to is one the boundary record holds**: the wired program does not return
it. -/
theorem returned_wireMap {x : ℕ} (hx : wireMap h x ≠ x) : returned h (wireMap h x) = false := by
  cases hv : wireTarget h x with
  | none => exact absurd (wireMap_of_no_target h hv) hx
  | some y =>
    rw [wireMap_of_target h hv]
    obtain ⟨_, _, _, _, _, _, _, _, _, hret⟩ := wireTarget_spec h hv
    exact hret

/-- **Every wired source bit has a gate to go to**, decided: each source bit at a port a wire names and no link names
has one (`wireTarget`). A wired source bit with none stays an input gate, which round 2 still reads off the interior
record. -/
def wiresMove : Bool :=
  (sourceBits h.ms).all fun s =>
    (heldLink h (h.names.getD s.2.1 "") s.2.2.row).isSome ||
      (heldWire h (h.names.getD s.2.1 "") s.2.2.row).isNone || (wireTarget h s.1).isSome

/-- Where every wired source bit has a gate to go to, a source bit at a port a wire names and no link names has
one. -/
theorem wireTarget_of_wiresMove (hc : wiresMove h = true) {x t : ℕ} {r : Spot.Bits.RowPos}
    (hs : (x, t, r) ∈ sourceBits h.ms) (hl : heldLink h (h.names.getD t "") r.row = none)
    {w : Spot.SecondStage.Wire} (hw : heldWire h (h.names.getD t "") r.row = some w) :
    ∃ y, wireTarget h x = some y := by
  have hall : ((heldLink h (h.names.getD t "") r.row).isSome ||
      (heldWire h (h.names.getD t "") r.row).isNone || (wireTarget h x).isSome) = true :=
    List.all_eq_true.1 hc (x, t, r) hs
  rw [hl, hw] at hall
  cases hv : wireTarget h x with
  | some y => exact ⟨y, rfl⟩
  | none =>
    rw [hv] at hall
    simp at hall

/-! ## The rows a run's slots read -/

/-- **The row slot `j` of part `p` reads at template `t`'s wired port**, as the wiring gives it: the writer's cell
size times the slot's replay block's place in the sample, plus the writer's position in its cell. -/
def wireRow (t : ℕ) (port : String) (p : Spot.SecondStage.Part) (j : ℕ) : Option ℕ :=
  (wireCell h t port (p.unit j)).map fun y => p.span j * h.reg.ns.getD y.1 0 + y.2.2

/-- **The row a slot reads at a wired port and the unit its gates go to are one cell position of the writer**: where
the wiring gives slot `j` of part `p` the row `row`, the slot's cell position is wired to position `x` of template
`tw`'s cell at its output port `wport`; the row is that position in the slot's sampled block (`p.span j`), and the
unit the wiring names for the slot's unit of replay block `ρ` is that position in replay block `ρ`. -/
theorem wireRow_unit {t : ℕ} {port : String} {p : Spot.SecondStage.Part} {j row : ℕ} (ρ : ℕ)
    (hc : p.unit j < h.reg.ns.getD t 0) (hrow : wireRow h t port p j = some row) :
    ∃ tw wport x, wireCell h t port (p.unit j) = some (tw, wport, x) ∧
      row = p.span j * h.reg.ns.getD tw 0 + x ∧
      wireUnit h t port (p.unit j + ρ * h.reg.ns.getD t 0) = some (tw, wport, ρ * h.reg.ns.getD tw 0 + x) := by
  unfold wireRow at hrow
  obtain ⟨⟨tw, wport, x⟩, hcell, hr⟩ := Option.map_eq_some_iff.1 hrow
  have hu := wireUnit_block h port ρ hc
  rw [hcell] at hu
  exact ⟨tw, wport, x, hcell, hr.symm, hu⟩

/-- **The row slot `j` of part `p` reads at template `t`'s linked port in the boundary port**, at replay sample
`S`: the slot's replay block's rows, from `S[i] · per`. -/
def linkRow (t : ℕ) (port : String) (S : Array ℕ) (p : Spot.SecondStage.Part) (j : ℕ) : Option ℕ :=
  (S[p.span j]?).bind fun b => (linkCell h t port (p.unit j)).map fun y => b * y.2.1 + y.2.2

/-- **The row slot `j` of part `p` reads at template `t`'s linked port in the interior record's value of the link's
name**: the slot's replay block's place in the sample, times `per`, plus the row. -/
def linkRowIn (t : ℕ) (port : String) (p : Spot.SecondStage.Part) (j : ℕ) : Option ℕ :=
  (linkCell h t port (p.unit j)).map fun y => p.span j * y.2.1 + y.2.2

theorem wiredRow_cell (p : Spot.SecondStage.Part) (r : Spot.SecondStage.Rows) (j : ℕ) :
    Spot.SecondStage.wiredRow p r j = (cellRow r (p.unit j)).map fun x => p.span j * r.per + x := by
  unfold Spot.SecondStage.wiredRow cellRow
  cases Spot.SecondStage.rowAt r.runs (p.unit j) with
  | none => rfl
  | some x =>
    show (if x < r.per then some (p.span j * r.per + x) else none) =
      (if x < r.per then some x else none).map fun x => p.span j * r.per + x
    by_cases hlt : x < r.per
    · rw [ite_eq_left hlt, ite_eq_left hlt]
      rfl
    · rw [ite_eq_right hlt, ite_eq_right hlt]
      rfl

theorem linkedRow_cell (own : Spot.SecondStage.Own) (p : Spot.SecondStage.Part) (r : Spot.SecondStage.Rows) (j : ℕ) :
    Spot.SecondStage.linkedRow own p r j =
      (own.sample[p.span j]?).bind fun b => (cellRow r (p.unit j)).map fun x => b * r.per + x := by
  unfold Spot.SecondStage.linkedRow cellRow
  cases own.sample[p.span j]? with
  | none => rfl
  | some b =>
    cases Spot.SecondStage.rowAt r.runs (p.unit j) with
    | none => rfl
    | some x =>
      show (if x < r.per then some (b * r.per + x) else none) =
        (if x < r.per then some x else none).map fun x => b * r.per + x
      by_cases hlt : x < r.per
      · rw [ite_eq_left hlt, ite_eq_left hlt]
        rfl
      · rw [ite_eq_right hlt, ite_eq_right hlt]
        rfl

/-- **At a wired port, the row round 2 holds a slot to is the one the wiring names**, where the writer's cell size is
the rows' `per`, which round 2 checks. -/
theorem wireRow_eq {t : ℕ} {port : String} {w : Spot.SecondStage.Wire} {r : Spot.SecondStage.Rows} {tw : ℕ}
    (hl : heldLink h (h.names.getD t "") port = none) (hw : heldWire h (h.names.getD t "") port = some w)
    (hr : heldRows h (h.names.getD t "") port = some r) (htw : heldWriter h w.writer = some tw)
    (hn : h.reg.ns.getD tw 0 = r.per) (p : Spot.SecondStage.Part) :
    wireRow h t port p = Spot.SecondStage.wiredRow p r := by
  funext j
  unfold wireRow
  rw [wireCell_eq h hl hw hr htw, wiredRow_cell]
  cases cellRow r (p.unit j) with
  | none => rfl
  | some x =>
    show some (p.span j * h.reg.ns.getD tw 0 + x) = some (p.span j * r.per + x)
    rw [hn]

/-- **At a linked port, the boundary row round 2 holds a slot to is the one the link names.** -/
theorem linkRow_eq {t : ℕ} {port : String} {l : Spot.SecondStage.Link} {r : Spot.SecondStage.Rows}
    (hl : heldLink h (h.names.getD t "") port = some l) (hr : heldRows h (h.names.getD t "") port = some r)
    (own : Spot.SecondStage.Own) (p : Spot.SecondStage.Part) :
    linkRow h t port own.sample p = Spot.SecondStage.linkedRow own p r := by
  funext j
  unfold linkRow
  rw [linkCell_eq h hl hr, linkedRow_cell]
  cases own.sample[p.span j]? with
  | none => rfl
  | some b => cases cellRow r (p.unit j) <;> rfl

/-- **At a linked port read off the interior record, the row round 2 holds a slot to is the one the link names.** -/
theorem linkRowIn_eq {t : ℕ} {port : String} {l : Spot.SecondStage.Link} {r : Spot.SecondStage.Rows}
    (hl : heldLink h (h.names.getD t "") port = some l) (hr : heldRows h (h.names.getD t "") port = some r)
    (p : Spot.SecondStage.Part) : linkRowIn h t port p = Spot.SecondStage.wiredRow p r := by
  funext j
  unfold linkRowIn
  rw [linkCell_eq h hl hr, wiredRow_cell]
  cases cellRow r (p.unit j) <;> rfl

/-! ## What round 2 holds a linked read to -/

/-- **A linked read reads the link's value at its unit's row**: at a port a link names, a read `readOk` passes is of
the link's value; where the interior record registers a value of that name, at that value's root, domain and leaves,
`Q · per` of them, each slot at its unit's row of its sampled block's (`wiredRow`); else at the boundary port of that
name, at its root, domain and leaves, `per` a replay block of the boundary's population, each slot at its unit's row
of its replay block's (`linkedRow`). -/
theorem readOk_linked {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Json} {l : Spot.SecondStage.Link} (hl : Spot.SecondStage.linkOf own p.template port = some l)
    (hok : Spot.SecondStage.readOk own named p port e = true) :
    Spot.getD e "value" = .str l.value ∧ ∃ r, Spot.SecondStage.rowsOf own p.template port = some r ∧
      ((∃ x, Spot.SecondStage.registeredEntry own.interior l.value = some x ∧
          Spot.SecondStage.sameAt e x = true ∧
          Spot.SecondStage.placedOk own.sample.size p e (own.sample.size * r.per)
            (Spot.SecondStage.wiredRow p r) = true) ∨
        (Spot.SecondStage.registeredEntry own.interior l.value = none ∧ ∃ b m,
          (l.value, b) ∈ Spot.Record.byPort own.boundary ∧
          Spot.nat? (Spot.getD own.boundary "population") = some m ∧ Spot.SecondStage.sameAt e b = true ∧
          Spot.SecondStage.placedOk own.sample.size p e (m * r.per)
            (Spot.SecondStage.linkedRow own p r) = true)) := by
  unfold Spot.SecondStage.readOk at hok
  rw [hl] at hok
  cases hv : Spot.getD e "value" with
  | str v =>
    rw [hv] at hok
    simp only [Bool.and_eq_true, beq_iff_eq] at hok
    obtain ⟨rfl, hm⟩ := hok
    refine ⟨rfl, ?_⟩
    cases hri : Spot.SecondStage.registeredEntry own.interior l.value with
    | some x =>
      cases hro : Spot.SecondStage.rowsOf own p.template port with
      | none => simp [hri, hro] at hm
      | some ro =>
        simp only [hri, hro, Bool.and_eq_true] at hm
        exact ⟨ro, rfl, Or.inl ⟨x, rfl, hm.1, hm.2⟩⟩
    | none =>
      cases hf : (Spot.Record.byPort own.boundary).find? (·.1 == l.value) with
      | none => simp [hri, hf] at hm
      | some kb =>
        obtain ⟨k, b⟩ := kb
        cases hpop : Spot.nat? (Spot.getD own.boundary "population") with
        | none => simp [hri, hf, hpop] at hm
        | some m =>
          cases hro : Spot.SecondStage.rowsOf own p.template port with
          | none => simp [hri, hf, hpop, hro] at hm
          | some ro =>
            simp only [hri, hf, hpop, hro, Bool.and_eq_true] at hm
            have hk : k = l.value := by simpa using Array.find?_some hf
            subst hk
            exact ⟨ro, rfl, Or.inr ⟨rfl, b, m, Array.mem_of_find?_eq_some hf, rfl, hm.1, hm.2⟩⟩
  | _ => rw [hv] at hok; simp at hok

/-! ## An accepted run's registered reads -/

/-- **Round 2 passes every registered read of an accepted run**: template `t`'s session statement, as round 2 reads
it, has each entry of its registered reads pass `SecondStage.readOk` against the template's part, at what the
verifier holds of the run. -/
theorem readOk_of_stmtOk {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ st rec smp w S, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some smp ∧
      FlockDeploy.decode (stmtAt h m₂ t) = some w ∧ stmtJsonOf h t w = some S ∧
      ∀ port e, (port, e) ∈ Spot.entries (Spot.getD S.registered "ports") →
        Spot.SecondStage.readOk (ownOf h st rec smp c₀) (Spot.SecondStage.byTemplate rec)
          (partAt h (ownOf h st rec smp c₀).sample.size t) port e = true := by
  have hat := stmtOkAt_of h hok ht
  obtain ⟨s, hx⟩ := parse_of_stmtOkAt h hat
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, hc⟩ := readsOk_true h hat.2
  obtain ⟨-, -, -, -, -, -, hreads, -, -, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  refine ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, fun port e hm => ?_⟩
  have hrd : Spot.SecondStage.readOk (ownOf h st rec smp c₀) (Spot.SecondStage.byTemplate rec)
      (partAt h (ownOf h st rec smp c₀).sample.size t) port e = true :=
    Array.all_eq_true_iff_forall_mem.1 hreads _ hm
  exact hrd

/-- **In an accepted run, a registered read at a wired port is at the row the wiring names.** Template `t`'s session
statement, as round 2 reads it, has each entry `e` of its registered reads at a port `port` that a wire `wr` of the
auditor's copies names and no link does read the writer's output port of the interior record `rec`, at that port's
root, domain and leaves: the writer `tw` is the first template of the wire's writer's name, its cell size is the
rows' `per`, the tree holds that many rows a sampled block, and each slot that reads is one of the template's part
and reads the row `wireRow` names, its writer's position in the same sampled block's cell. -/
theorem wired_read {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ st rec smp w S, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some smp ∧
      FlockDeploy.decode (stmtAt h m₂ t) = some w ∧ stmtJsonOf h t w = some S ∧
      ∀ port e wr, (port, e) ∈ Spot.entries (Spot.getD S.registered "ports") →
        heldLink h (h.names.getD t "") port = none → heldWire h (h.names.getD t "") port = some wr →
        ∃ tw r xe, heldWriter h wr.writer = some tw ∧ heldRows h (h.names.getD t "") port = some r ∧
          h.reg.ns.getD tw 0 = r.per ∧
          Spot.getD e "value" =
            .str (Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) wr.writer wr.wport) ∧
          (Spot.SecondStage.portOf (Spot.SecondStage.byTemplate rec) wr.writer wr.wport, xe) ∈
            Spot.Record.byPort rec ∧
          Spot.SecondStage.sameAt e xe = true ∧
          Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
            ((h.reg.sample (src c₀)).size * h.reg.ns.getD tw 0)
            (wireRow h t port (partAt h (h.reg.sample (src c₀)).size t)) = true := by
  obtain ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, hall⟩ := readOk_of_stmtOk h hok ht
  refine ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, fun port e wr hm hl hwr => ?_⟩
  obtain ⟨xe, pw, r, hv, hmem, hpw, hr, hn, hsame, hpl⟩ :=
    Spot.SecondStage.readOk_wired (own := ownOf h st rec smp c₀)
      (named := Spot.SecondStage.byTemplate rec) (p := partAt h (ownOf h st rec smp c₀).sample.size t)
      (port := port) (e := e) (w := wr) hl hwr (hall port e hm)
  rw [writerOf_ownOf] at hpw
  obtain ⟨tw, htw, rfl⟩ := Option.map_eq_some_iff.1 hpw
  have hn' : h.reg.ns.getD tw 0 = r.per := hn
  have hr' : heldRows h (h.names.getD t "") port = some r := hr
  have hpl' : Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
      ((h.reg.sample (src c₀)).size * r.per)
      (Spot.SecondStage.wiredRow (partAt h (h.reg.sample (src c₀)).size t) r) = true := hpl
  refine ⟨tw, r, xe, htw, hr', hn', hv, hmem, hsame, ?_⟩
  rw [wireRow_eq h hl hwr hr' htw hn', hn']
  exact hpl'

/-- **In an accepted run, a registered read at a linked port is at the row the link names.** Template `t`'s session
statement, as round 2 reads it, has each entry `e` of its registered reads at a port `port` that a link `l` of the
auditor's copies names read the link's value: where the interior record `rec` registers a value of that name, that
value, each reading slot at the row `linkRowIn` names; else the boundary port of that name of the run's statement,
`per` rows a replay block of its population, each reading slot at the row `linkRow` names, its replay block's. Which
of the two holds is the interior record's, not the auditor's copies'. -/
theorem linked_read {x m₁ m₂ : Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ st rec smp w S, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some smp ∧
      FlockDeploy.decode (stmtAt h m₂ t) = some w ∧ stmtJsonOf h t w = some S ∧
      ∀ port e l, (port, e) ∈ Spot.entries (Spot.getD S.registered "ports") →
        heldLink h (h.names.getD t "") port = some l →
        Spot.getD e "value" = .str l.value ∧ ∃ r, heldRows h (h.names.getD t "") port = some r ∧
          ((∃ xe, Spot.SecondStage.registeredEntry rec l.value = some xe ∧ Spot.SecondStage.sameAt e xe = true ∧
              Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
                ((h.reg.sample (src c₀)).size * r.per)
                (linkRowIn h t port (partAt h (h.reg.sample (src c₀)).size t)) = true) ∨
            (Spot.SecondStage.registeredEntry rec l.value = none ∧ ∃ b m,
              (l.value, b) ∈ Spot.Record.byPort st.record ∧
              Spot.nat? (Spot.getD st.record "population") = some m ∧ Spot.SecondStage.sameAt e b = true ∧
              Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
                (m * r.per)
                (linkRow h t port (h.reg.sample (src c₀)) (partAt h (h.reg.sample (src c₀)).size t)) = true)) := by
  obtain ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, hall⟩ := readOk_of_stmtOk h hok ht
  refine ⟨st, rec, smp, w, S, hst, hrec, hsmp, hw, hS, fun port e l hm hl => ?_⟩
  obtain ⟨hv, r, hr, hcase⟩ :=
    readOk_linked (own := ownOf h st rec smp c₀) (named := Spot.SecondStage.byTemplate rec)
      (p := partAt h (ownOf h st rec smp c₀).sample.size t) (port := port) (e := e) (l := l) hl (hall port e hm)
  have hr' : heldRows h (h.names.getD t "") port = some r := hr
  refine ⟨hv, r, hr', ?_⟩
  rcases hcase with ⟨xe, hxe, hsame, hpl⟩ | ⟨hnone, b, m, hb, hpop, hsame, hpl⟩
  · refine Or.inl ⟨xe, hxe, hsame, ?_⟩
    have hpl' : Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
        ((h.reg.sample (src c₀)).size * r.per)
        (Spot.SecondStage.wiredRow (partAt h (h.reg.sample (src c₀)).size t) r) = true := hpl
    rw [linkRowIn_eq h hl hr']
    exact hpl'
  · refine Or.inr ⟨hnone, b, m, hb, hpop, hsame, ?_⟩
    have hrow : linkRow h t port (h.reg.sample (src c₀)) (partAt h (h.reg.sample (src c₀)).size t) =
        Spot.SecondStage.linkedRow (ownOf h st rec smp c₀) (partAt h (h.reg.sample (src c₀)).size t) r :=
      linkRow_eq h hl hr' (ownOf h st rec smp c₀) (partAt h (h.reg.sample (src c₀)).size t)
    have hpl' : Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
        (m * r.per)
        (Spot.SecondStage.linkedRow (ownOf h st rec smp c₀) (partAt h (h.reg.sample (src c₀)).size t) r) =
          true := hpl
    rw [hrow]
    exact hpl'

/-! ## One slot a gate -/

/-- **Under `Addresses`, a gate is read one way**: every accepted run's checked unit that needs it reads it as a
registered value, or every one reads it off the interior record. An address has one slot, which a registered read asks
not sized and the interior record's port sized; so no address holds at a gate one unit's session reads as a registered
value and another's off the interior record. -/
theorem addresses_one_way (hA : Addresses h) (g : ℕ) : ∃ S : Address,
    (∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → r.ReadsRegistered h S g) ∨
      ∀ r : CheckedRun h.stage, r.Ok h → r.s.prog.Needs r.b g → r.ReadsInterior h S g := by
  obtain ⟨S, hS⟩ := hA g
  refine ⟨S, ?_⟩
  by_cases hs : S.slot.sized = true
  · refine Or.inr fun r hr hg => ?_
    obtain ⟨-, -, hrd⟩ := hS r hr hg
    rcases hrd with ⟨reg, -, -, -, -, -, hf, -⟩ | hI
    · exact absurd (hf.symm.trans hs) (by decide)
    · exact hI
  · refine Or.inl fun r hr hg => ?_
    obtain ⟨-, -, hrd⟩ := hS r hr hg
    rcases hrd with hR | ⟨-, -, -, -, -, ht, -⟩
    · exact hR
    · exact absurd ht hs

end SpotFlock
