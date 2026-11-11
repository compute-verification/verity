module

public import Proofs.Properties.Nci.Record
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.String.Defs
import all Init.Data.Repr
import all Init.Data.List.Sort.Basic
import all Init.Data.List.Control
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.WF

/-!
# Instances against the text's tie in `Served`

A scratch file, never merged: what the two clauses that hold a check's slots and held rows to the record's text say
and leave out.

* **The clauses are of the instance's two texts.** The Program and the parsed text `Served` binds are the ones the
  instance's texts decode and parse to, so a served check's slots are `NciCheck.RecordText.slot` of them and each row
  it holds is one `NciCheck.RecordText.declared` declares, for that reason (`tied_of_served`). Two checks of one
  instance give every unit one slot (`slot_unique`), and where both hold a row they hold it for one declared reason
  (`held_declared_unique`); a check that holds a row for another reason is served by no circuit
  (`not_served_of_undeclared`). An opening's pin is not part of that reason (`opened_pin`).
* **Where the readers refuse.** On a text that does not name the Program, `templatesOf` refuses, so `slot` and
  `declared` are none everywhere (`slot_noProgram`, `declared_noProgram`), and a served check has no slot and holds
  no row (`served_noProgram`). So a check with a slot, or one that holds a row, is served on no such text
  (`not_served_of_slot`, `not_served_of_held`): on the text of no wire that is every check with a slot, whether it
  holds a row or not (`not_served_slotted`), and every check that holds every row (`not_served_hold_all`).
* **Fewer rows held.** A served record is served with no row held and θ empty (`served_hold_none`).
-/

@[expose] public section

namespace NciCheck.Properties.Record.Tie

open Lean (Json)
open Flock.Program (Prog Scope Ref Ty Defn)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row Held Declared wiresOf)

deriving instance DecidableEq for Spot.Wires.Port
deriving instance DecidableEq for Spot.Wires.Template
deriving instance DecidableEq for Spot.Wires.Wire
deriving instance DecidableEq for Spot.Wires.Wiring
deriving instance DecidableEq for Spot.Wires.Refusal
deriving instance DecidableEq for Spot.Wires.Summary

theorem ok_of_toOption {ε α : Type} {x : Except ε α} {a : α} (h : x.toOption = some a) : x = .ok a := by
  cases x <;> simp_all [Except.toOption]

/-! ### The text, read by the kernel -/

/-- The wiring `Served` reads of a text given as its characters, none where the parse or the reading refuses. -/
def readTie (cs : List Char) : Option Spot.Wires.Wiring :=
  (Flock.JsonParse.parseList cs).toOption.bind fun doc => (Spot.Wires.Wiring.ofJson (wiresOf doc)).toOption

/-- The text's `program`, none where the text does not parse. -/
def readProgram (cs : List Char) : Option (Option String) :=
  (Flock.JsonParse.parseList cs).toOption.map fun doc => NciCheck.RecordText.stringAt doc "program"

theorem tie_of {cs : List Char} {W : Spot.Wires.Wiring} (h : readTie cs = some W) :
    ∃ doc, Flock.JsonParse.parse (String.ofList cs) = .ok doc ∧ Spot.Wires.Wiring.ofJson (wiresOf doc) = .ok W := by
  unfold readTie at h
  obtain ⟨doc, hdoc, hW⟩ := Option.bind_eq_some_iff.1 h
  refine ⟨doc, ?_, ok_of_toOption hW⟩
  rw [Flock.JsonParse.parse_eq, String.toList_ofList]
  exact ok_of_toOption hdoc

/-- The `program` of a text that parses is the one `readProgram` reads. -/
theorem program_of {cs : List Char} {doc : Json} {p : Option String}
    (hparse : Flock.JsonParse.parse (String.ofList cs) = .ok doc) (hp : readProgram cs = some p) :
    NciCheck.RecordText.stringAt doc "program" = p := by
  rw [Flock.JsonParse.parse_eq, String.toList_ofList] at hparse
  unfold readProgram at hp
  rw [hparse] at hp
  first
    | exact Option.some.inj hp
    | simpa [Except.toOption] using hp

/-- `{"wires":[]}` -/
def recordEmpty : List Char :=
  ['{', '"', 'w', 'i', 'r', 'e', 's', '"', ':', '[', ']', '}']

theorem tie_empty : readTie recordEmpty = some ⟨[], [], [], none⟩ := by decide +kernel

theorem noProgram_empty : readProgram recordEmpty = some none := by decide +kernel

theorem check_empty : Spot.Wires.check ⟨[], [], [], none⟩ = .ok ⟨[], []⟩ :=
  ok_of_toOption (by decide +kernel)

/-! ### Where the readers refuse -/

/-- A text that names no Program has no templates, whatever the Program. -/
theorem templatesOf_noProgram {P : Prog} {doc : Json} (h : NciCheck.RecordText.stringAt doc "program" = none) :
    ∃ e, NciCheck.RecordText.templatesOf P doc = .error e := by
  cases hres : NciCheck.RecordText.templatesOf P doc with
  | error e => exact ⟨e, rfl⟩
  | ok ts =>
    exfalso
    unfold NciCheck.RecordText.templatesOf at hres
    rw [h] at hres
    first
      | exact Except.noConfusion hres
      | cases hres
      | simp at hres

/-- So it gives no unit a slot. -/
theorem slot_noProgram {P : Prog} {doc : Json} (h : NciCheck.RecordText.stringAt doc "program" = none)
    (tid : String) (u : ℕ) : NciCheck.RecordText.slot P doc tid u = none := by
  obtain ⟨e, he⟩ := templatesOf_noProgram (P := P) h
  first
    | (unfold NciCheck.RecordText.slot; rw [he]; done)
    | (unfold NciCheck.RecordText.slot; rw [he]; rfl)
    | simp [NciCheck.RecordText.slot, he]

/-- And declares no row. -/
theorem declared_noProgram {P : Prog} {doc : Json} (h : NciCheck.RecordText.stringAt doc "program" = none)
    (r : Row) : NciCheck.RecordText.declared P doc r = none := by
  obtain ⟨e, he⟩ := templatesOf_noProgram (P := P) h
  first
    | (unfold NciCheck.RecordText.declared; rw [he]; done)
    | (unfold NciCheck.RecordText.declared; rw [he]; rfl)
    | (unfold NciCheck.RecordText.declared; rw [he]; cases NciCheck.RecordText.wiring doc <;> rfl)
    | simp [NciCheck.RecordText.declared, he]

/-- **The clauses are of the instance's two texts.** Where the instance's Program decodes to `P` and its record
parses to `doc`, a served check's slots are the text's, and each row it holds is one the text declares, for that
reason. -/
theorem tied_of_served {F : Form} {c : F.Case} {d : Verity.RegisteredCircuit} (hd : Served F c d) {P : Prog}
    {doc : Json} (hP : Flock.Program.decode (F.inst c).program.toUTF8 = .ok P)
    (hdoc : Flock.JsonParse.parse (F.inst c).record = .ok doc) :
    F.slot c = NciCheck.RecordText.slot P doc ∧
      ∀ r h, F.held c r = some h → NciCheck.RecordText.declared P doc r = some h.declared := by
  obtain ⟨P', doc', W, S, rs, at_, hdec, hparse, -, -, -, hslot, hheld, -⟩ := hd
  obtain rfl : P' = P := Except.ok.inj (hdec.symm.trans hP)
  obtain rfl : doc' = doc := Except.ok.inj (hparse.symm.trans hdoc)
  exact ⟨hslot, hheld⟩

/-- **On a text that names no Program, a served check has no slot and holds no row.** -/
theorem served_noProgram {F : Form} {c : F.Case} {d : Verity.RegisteredCircuit} {cs : List Char}
    (hrec : (F.inst c).record = String.ofList cs) (hp : readProgram cs = some none) (hd : Served F c d) :
    (∀ tid u, F.slot c tid u = none) ∧ ∀ r, F.held c r = none := by
  obtain ⟨P, doc, W, S, rs, at_, -, hparse, -, -, -, hslot, hheld, -⟩ := hd
  rw [hrec] at hparse
  have hprog := program_of hparse hp
  refine ⟨fun tid u => ?_, fun r => ?_⟩
  · rw [hslot]
    exact slot_noProgram hprog tid u
  · cases hr : F.held c r with
    | none => rfl
    | some h =>
      have hdecl := hheld r h hr
      rw [declared_noProgram hprog] at hdecl
      exact absurd hdecl (by simp)

/-- A check with a slot is served on no text that names no Program. -/
theorem not_served_of_slot {F : Form} {c : F.Case} {cs : List Char} (hrec : (F.inst c).record = String.ofList cs)
    (hp : readProgram cs = some none) {tid : String} {u s : ℕ} (hs : F.slot c tid u = some s)
    (d : Verity.RegisteredCircuit) : ¬ Served F c d := fun hd => by
  have h := (served_noProgram hrec hp hd).1 tid u
  rw [hs] at h
  exact absurd h (by simp)

/-- A check that holds a row is served on no text that names no Program. -/
theorem not_served_of_held {F : Form} {c : F.Case} {cs : List Char} (hrec : (F.inst c).record = String.ofList cs)
    (hp : readProgram cs = some none) {r : Row} {x : Held} (hh : F.held c r = some x)
    (d : Verity.RegisteredCircuit) : ¬ Served F c d := fun hd => by
  have h := (served_noProgram hrec hp hd).2 r
  rw [hh] at h
  exact absurd h (by simp)

/-! ### The two checks of one text -/

/-- A check that reads every instance as the text of no wire beside the Program text `p`, every unit in the slot
`sl` and every row held as `hd`. -/
def formTie (p : String) (sl : Option ℕ) (hd : Option Held) : Form where
  Case := Unit
  read := fun _ _ => .ok ()
  inst := fun _ => { program := p, record := String.ofList recordEmpty, gamma := 0, num := 0, den := 1, pins := [] }
  wiring := fun _ => { templates := [], cuts := [], wires := [], order := none }
  slot := fun _ _ _ => sl
  held := fun _ _ => hd
  summarize := fun _ => ⟨0, 0⟩
  rate := fun _ _ _ _ => none
  layout := fun _ => none

/-- **Neither check of the text of no wire is served**: with every unit in a slot, the check that holds the row and
the check that holds nothing are both refused, whatever the Program's text and the circuit. -/
theorem not_served_slotted (p : String) (s : ℕ) (hd : Option Held) (d : Verity.RegisteredCircuit) :
    ¬ Served (formTie p (some s) hd) () d :=
  not_served_of_slot (cs := recordEmpty) (tid := "") (u := 0) rfl noProgram_empty rfl d

/-- The check that holds a row is refused by its held rows alone, with no unit in a slot. -/
theorem not_served_holding (p : String) (x : Held) (d : Verity.RegisteredCircuit) :
    ¬ Served (formTie p none (some x)) () d :=
  not_served_of_held (cs := recordEmpty) (r := default) rfl noProgram_empty rfl d

/-! ### Held rows -/

/-- Two checks of one instance that hold one row hold it for one declared reason. -/
theorem held_declared_unique {F G : Form} {c : F.Case} {c' : G.Case} {d d' : Verity.RegisteredCircuit}
    (hi : F.inst c = G.inst c') (hd : Served F c d) (hd' : Served G c' d') {r : Row} {x x' : Held}
    (hx : F.held c r = some x) (hx' : G.held c' r = some x') : x.declared = x'.declared := by
  obtain ⟨P, doc, W, S, rs, at_, hdec, hparse, -, -, -, -, hheld, -⟩ := hd
  rw [hi] at hdec hparse
  have h := (tied_of_served hd' hdec hparse).2 r x' hx'
  rw [hheld r x hx] at h
  exact Option.some.inj h

/-- Two checks of one instance that are both served give every unit one slot. -/
theorem slot_unique {F G : Form} {c : F.Case} {c' : G.Case} {d d' : Verity.RegisteredCircuit}
    (hi : F.inst c = G.inst c') (hd : Served F c d) (hd' : Served G c' d') : F.slot c = G.slot c' := by
  obtain ⟨P, doc, W, S, rs, at_, hdec, hparse, -, -, -, hslot, -⟩ := hd
  rw [hi] at hdec hparse
  exact hslot.trans (tied_of_served hd' hdec hparse).1.symm

/-- A check that holds a row for a reason its text does not declare is served by no circuit. -/
theorem not_served_of_undeclared {F : Form} {c : F.Case} {P : Prog} {doc : Json}
    (hP : Flock.Program.decode (F.inst c).program.toUTF8 = .ok P)
    (hdoc : Flock.JsonParse.parse (F.inst c).record = .ok doc) {r : Row} {x : Held} (hx : F.held c r = some x)
    (hn : NciCheck.RecordText.declared P doc r ≠ some x.declared) (d : Verity.RegisteredCircuit) :
    ¬ Served F c d :=
  fun hd => hn ((tied_of_served hd hP hdoc).2 r x hx)

/-- An opening's pin is not part of the reason the text declares. -/
theorem opened_pin (name p a b : String) : (Held.opened name p a).declared = (Held.opened name p b).declared := rfl

/-- The check `F`, holding the rows `h`. -/
def rehold (F : Form) (h : F.Case → Row → Option Held) : Form :=
  { F with held := h }

/-- **Fewer rows held.** A record that lays out a circuit lays out the same wired circuit with no row held and no
gate in θ, so that every input gate is fresh, whatever its text declares. -/
theorem served_hold_none (F : Form) (c : F.Case) (d : Verity.RegisteredCircuit) (hd : Served F c d) :
    ∃ d' : Verity.RegisteredCircuit, Served (rehold F fun _ _ => none) c d' ∧ d'.D = d.D ∧ d'.θ = ∅ := by
  obtain ⟨P, doc, W, S, rs, at_, hdec, hparse, hof, hcuts, hwires, hslot, -, hcheck, heval, htmpl, hown, hal, hfill,
    -, hlay, -, hdiv⟩ := hd
  refine ⟨Verity.RegisteredCircuit.ofPartition d.D ∅ (fun g hg => absurd hg (Finset.notMem_empty g)) d.n d.𝓘,
    ⟨P, doc, W, S, rs, at_, hdec, hparse, hof, hcuts, hwires, hslot, fun r h hr => ?_, hcheck, heval, htmpl, hown,
      hal, hfill, fun r hr => ?_, hlay, fun x => ?_, hdiv⟩, rfl, rfl⟩
  · exact absurd hr (by simp [rehold])
  · exact absurd hr (by simp [rehold])
  · constructor
    · intro hx
      exact absurd hx (Finset.notMem_empty x)
    · rintro ⟨r, l, lo, hi, hheld, -⟩
      exact absurd hheld (by simp [rehold])

/-- Every row held: on a text that names no Program no circuit is served, whatever the check was. -/
theorem not_served_hold_all (F : Form) (c : F.Case) {cs : List Char} (hrec : (F.inst c).record = String.ofList cs)
    (hp : readProgram cs = some none) (d : Verity.RegisteredCircuit) :
    ¬ Served (rehold F fun _ _ => some (Held.read "")) c d :=
  not_served_of_held (F := rehold F fun _ _ => some (Held.read "")) (r := default) hrec hp rfl d

end NciCheck.Properties.Record.Tie
