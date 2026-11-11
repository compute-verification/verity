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
  `wireUnit` sends it to its writer's unit in the same replay block, and `linkUnit` to the boundary port's row.
* **The gates** (`wireMap`): a source bit at a wired port goes to the gate on the same bit of its writer unit's output
  row, where the writer's key program has one and no link names the writer's output port; every other gate to
  itself. A linked port's source bits stay: the boundary port is no gate of the members' programs, and the auditor's
  copies name no unit that writes it. `wiredProg` is the members' programs read through that map (`Prog.rewire`).
* **A returned output wired inside** (`noReturnedWired`, `writer_not_linked`): where a link names a wire's writer's
  output port, round 2 holds the writer's session to the link's record there and the wired reader to the interior
  record's port, and holds neither entry to the other; the map keeps the two gates apart, and the condition that
  there is no such port is decided from the auditor's copies.
* **What round 2 checks** (`wired_read`, `linked_read`): in an accepted run, each registered read of a template's
  session at a wired port is of the writer's output port of the interior record, in a tree of the writer's cell size a
  sampled block, each reading slot at the row `wireRow` names; and at a linked port it is of the link's value, each
  slot at the row `linkRow` names in the boundary port, or `linkRowIn` names in the interior record's value of that
  name. Which of the two a linked read is held to is the interior record's, which the run states, not the auditor's
  copies.
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
theorem writerOf_ownOf (st : Roots) (rec draw : Json) (c₀ : FlockDeploy.Coin) (w : Spot.SecondStage.Wire) :
    Spot.SecondStage.writerOf (ownOf h st rec draw c₀) w =
      (heldWriter h w.writer).map (partAt h (h.reg.sample (src c₀)).size) := by
  have key : (((List.range h.ms.length).map (partAt h (h.reg.sample (src c₀)).size)).toArray).find?
      (fun p => p.template == w.writer) =
        (heldWriter h w.writer).map (partAt h (h.reg.sample (src c₀)).size) := by
    first
      | (rw [List.find?_toArray, List.find?_map] <;> rfl)
      | (simp [heldWriter, partAt, Function.comp_def])
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
    first
      | (unfold wireCell; rw [hl, hw, hr] <;> rfl)
      | (simp only [wireCell, hl, hw, hr])
      | (simp [wireCell, hl, hw, hr])
  rw [h1, htw] <;> rfl

theorem linkCell_eq {t : ℕ} {port : String} {l : Spot.SecondStage.Link} {r : Spot.SecondStage.Rows}
    (hl : heldLink h (h.names.getD t "") port = some l) (hr : heldRows h (h.names.getD t "") port = some r) (c : ℕ) :
    linkCell h t port c = (cellRow r c).map fun x => (l.value, r.per, x) := by
  first
    | (unfold linkCell; rw [hl, hr] <;> rfl)
    | (simp only [linkCell, hl, hr])
    | (simp [linkCell, hl, hr])

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
  first
    | exact Option.isNone_iff_eq_none.1 hn
    | simpa using hn

/-- **The wiring map**: a source bit at a port a wire names goes to the gate on the same bit of its writer unit's
output row (`wireUnit`), where the writer's program has a gate there and no link names the writer's output port;
every other position to itself. At a writer's output port a link names, the reader's source bit and the writer's
gate stay two gates: round 2 holds them to two entries. -/
def wireMap (x : ℕ) : ℕ :=
  match (sourceBits h.ms).find? (·.1 == x) with
  | some (_, t, r) =>
    (match wireUnit h t r.row r.leaf with
      | some (tw, wport, u) =>
        if (heldLink h (h.names.getD tw "") wport).isSome then x
        else (((writtenBits h.ms).find? fun y =>
          y.2.1 == tw && y.2.2 == Spot.Bits.RowPos.mk wport u r.bit).map (·.1)).getD x
      | none => x)
  | none => x

/-- **The members' programs read through the wiring map.** -/
def wiredProg : Prog := (bitsProgs h.ms).rewire (wireMap h)

/-- A position that is no source bit goes to itself. -/
theorem wireMap_of_not_source {x : ℕ} (hx : (sourceBits h.ms).find? (·.1 == x) = none) : wireMap h x = x := by
  first
    | (unfold wireMap; rw [hx] <;> rfl)
    | (simp only [wireMap, hx])
    | (simp [wireMap, hx])

/-! ## The rows a run's slots read -/

/-- **The row slot `j` of part `p` reads at template `t`'s wired port**, as the wiring gives it: the writer's cell
size times the slot's replay block's place in the sample, plus the writer's position in its cell. -/
def wireRow (t : ℕ) (port : String) (p : Spot.SecondStage.Part) (j : ℕ) : Option ℕ :=
  (wireCell h t port (p.unit j)).map fun y => p.span j * h.reg.ns.getD y.1 0 + y.2.2

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
    · first | (rw [if_pos hlt, if_pos hlt] <;> rfl) | simp [hlt]
    · first | (rw [if_neg hlt, if_neg hlt] <;> rfl) | simp [hlt]

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
      · first | (rw [if_pos hlt, if_pos hlt] <;> rfl) | simp [hlt]
      · first | (rw [if_neg hlt, if_neg hlt] <;> rfl) | simp [hlt]

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
theorem readOk_of_stmtOk {x m₁ m₂ : Msg} {c₀ c₁ : FlockDeploy.Coin} (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ}
    (ht : t < h.ms.length) :
    ∃ st rec draw w S, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
      FlockDeploy.decode (stmtAt h m₂ t) = some w ∧ stmtJsonOf h t w = some S ∧
      ∀ port e, (port, e) ∈ Spot.entries (Spot.getD S.registered "ports") →
        Spot.SecondStage.readOk (ownOf h st rec draw c₀) (Spot.SecondStage.byTemplate rec)
          (partAt h (ownOf h st rec draw c₀).sample.size t) port e = true := by
  have hat := stmtOkAt_of h hok ht
  obtain ⟨s, hx⟩ := parse_of_stmtOkAt h hat
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, hc⟩ := readsOk_true h hat.2
  obtain ⟨-, -, -, -, -, -, hreads, -, -, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  refine ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, fun port e hm => ?_⟩
  have hrd : Spot.SecondStage.readOk (ownOf h st rec draw c₀) (Spot.SecondStage.byTemplate rec)
      (partAt h (ownOf h st rec draw c₀).sample.size t) port e = true :=
    Array.all_eq_true_iff_forall_mem.1 hreads _ hm
  exact hrd

/-- **In an accepted run, a registered read at a wired port is at the row the wiring names.** Template `t`'s session
statement, as round 2 reads it, has each entry `e` of its registered reads at a port `port` that a wire `wr` of the
auditor's copies names and no link does read the writer's output port of the interior record `rec`, at that port's
root, domain and leaves: the writer `tw` is the first template of the wire's writer's name, its cell size is the
rows' `per`, the tree holds that many rows a sampled block, and each slot that reads is one of the template's part
and reads the row `wireRow` names, its writer's position in the same sampled block's cell. -/
theorem wired_read {x m₁ m₂ : Msg} {c₀ c₁ : FlockDeploy.Coin} (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ}
    (ht : t < h.ms.length) :
    ∃ st rec draw w S, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
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
  obtain ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, hall⟩ := readOk_of_stmtOk h hok ht
  refine ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, fun port e wr hm hl hwr => ?_⟩
  obtain ⟨xe, pw, r, hv, hmem, hpw, hr, hn, hsame, hpl⟩ :=
    Spot.SecondStage.readOk_wired (own := ownOf h st rec draw c₀)
      (named := Spot.SecondStage.byTemplate rec) (p := partAt h (ownOf h st rec draw c₀).sample.size t)
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
theorem linked_read {x m₁ m₂ : Msg} {c₀ c₁ : FlockDeploy.Coin} (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ}
    (ht : t < h.ms.length) :
    ∃ st rec draw w S, decode x = some st ∧ jsonMsg m₁ = some rec ∧ secondDraw h c₀ c₁ = some draw ∧
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
  obtain ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, hall⟩ := readOk_of_stmtOk h hok ht
  refine ⟨st, rec, draw, w, S, hst, hrec, hdraw, hw, hS, fun port e l hm hl => ?_⟩
  obtain ⟨hv, r, hr, hcase⟩ :=
    readOk_linked (own := ownOf h st rec draw c₀) (named := Spot.SecondStage.byTemplate rec)
      (p := partAt h (ownOf h st rec draw c₀).sample.size t) (port := port) (e := e) (l := l) hl (hall port e hm)
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
        Spot.SecondStage.linkedRow (ownOf h st rec draw c₀) (partAt h (h.reg.sample (src c₀)).size t) r :=
      linkRow_eq h hl hr' (ownOf h st rec draw c₀) (partAt h (h.reg.sample (src c₀)).size t)
    have hpl' : Spot.SecondStage.placedOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) e
        (m * r.per)
        (Spot.SecondStage.linkedRow (ownOf h st rec draw c₀) (partAt h (h.reg.sample (src c₀)).size t) r) =
          true := hpl
    rw [hrow]
    exact hpl'

end SpotFlock
