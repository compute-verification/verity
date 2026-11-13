module

public import Proofs.Properties.Nci.RecordAligned
public import Proofs.Properties.Nci.RecordCircuit
public import Proofs.Properties.Nci.RecordForm
public import Proofs.Properties.Nci.RecordInputs
public import Proofs.Properties.Nci.RecordRows
public import Proofs.Properties.Nci.RecordTemplates
public import Proofs.Properties.Nci.RecordWired

/-!
# The record check's own form lays out a circuit

`form_laidOut` (`Properties.Record.LaidOut`): every record `read` accepts lays out a registered circuit
(`Properties.Record.Served`), clause by clause.

* The texts. The Program is the instance's, decoded (`decode_of_read`), and the record's text parses
  (`inst_of_read`); the wiring's cuts and wires, the slots and the reasons rows are held are the text's
  (`wiring_of_read`, `slot_of_read`, `declared_of_heldAt`).
* The wiring. It passes `Spot.Wires.check` (`wired_of_read`), the template query accepts its template ids and its
  templates are the Program's (`templates_of_read`); every unit's rows are its own input gates (`ownRows_holds`);
  the wires pass the check's `unwhole` on the leaves the Program gives each template (`unwhole_of_read`), so every
  wire lands whole and every bit of a wired row is its writer's (`aligned`, `filled`); and no row the check holds has
  a writer (`held_unwired`).
* The circuit. What a unit returns and what its gates read are its own (`returnsOwn`, `operandsOwn`), a unit's gates
  are computed and every gate has its place (`unitsComputed`, `gatesOwn`), and the wires rank the units (`ranked`):
  so the Program's gates no wire covers are a circuit (`laysOut_exists`). θ is the gates of the rows held and the
  blocks are the slots (`names_divides`, on `slot_of_unit`); the registered circuit is that circuit with them
  (`Verity.RegisteredCircuit.ofPartition`).
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open RecordText
open Lean (Json)

/-- `front` reads each template's leaves off its own Program. -/
theorem front_leaves {j tr : Json} {f : Front} (h : front j tr = .ok f) :
    f.leaves = f.ts.map (leavesOf f.P (passedOn f.P)) := by
  unfold front at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rfl

end NciCheck.Record.SecurityProofs

namespace NciCheck.Properties.Record.SecurityProofs

open Lean (Json)
open NciCheck.Record (unwhole leavesOf passedOn)
open NciCheck.Record.SecurityProofs (form read_front finish_text front_text front_leaves inst_of_read
  program_of_read wiring_of_read slot_of_read declared_of_heldAt)

/-- **The wires of a record `read` accepts pass the check's `unwhole`**, on the templates the text gives and the
leaves the Program gives each. -/
theorem unwhole_of_read {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c) :
    NciCheck.RecordText.templatesOf c.f.P c.f.doc = .ok c.f.ts ∧
      unwhole c.f.ts (c.f.ts.map (leavesOf c.f.P (passedOn c.f.P))) c.wired.wires = none := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, -⟩ := finish_text hc
  refine ⟨front_text hf, ?_⟩
  rw [← front_leaves hf]
  exact (finish_case hc).2.1

/-- **A record `read` accepts lays out a registered circuit**: the wired circuit of its Program, with θ the gates of
the rows the check holds and the blocks the slots. -/
theorem served_of_read {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c) :
    ∃ d, Served form c d := by
  have hP := readProg_of_read h
  obtain ⟨-, -, -, -, hown, -⟩ := program_of_read h
  obtain ⟨text, htext, hcuts, hwires⟩ := wiring_of_read h
  have hS := wired_of_read h
  obtain ⟨rs, hrs, hT⟩ := templates_of_read h
  obtain ⟨hts, hwh⟩ := unwhole_of_read h
  have hO := ownRows_holds hP hown hrs
  have hG := gatesOwn hP hrs
  obtain ⟨ρ, hρ⟩ := ranked hS
  obtain ⟨D, at_, hL⟩ := laysOut_exists c.f.P rs c.W c.wired hO (returnsOwn hP hrs hT hS) (unitsComputed hP hrs)
    (operandsOwn hP hown hrs hT hS) hG ρ hρ
  obtain ⟨θ, hθ, U, hN, hD⟩ := names_divides hL hO hG (fun r => (c.heldAt r).isSome) c.slot c.S
    (fun tid u sc hu => slot_of_unit h hrs hu)
  exact ⟨Verity.RegisteredCircuit.ofPartition D θ hθ c.S U, c.f.P, c.f.doc, text, c.wired, rs, at_,
    decode_of_read h, (inst_of_read h).2, htext, hcuts, hwires, slot_of_read h,
    fun r hd hh => declared_of_heldAt h hh, hS, hrs, hT, hO, aligned hP hown hrs hT hS hts hwh,
    filled hP hrs hT hS hts hwh, fun r hh => held_unwired h r hh, hL, hN, hD⟩

end NciCheck.Properties.Record.SecurityProofs

namespace NciCheck.Record.SecurityProofs

/-- **The record check's own form lays out a circuit** (`Properties.Record.LaidOut`). -/
theorem form_laidOut : Properties.Record.LaidOut form := by
  intro j tr c h
  have h' : read j tr = .ok c := h
  exact Properties.Record.SecurityProofs.served_of_read h'

end NciCheck.Record.SecurityProofs
