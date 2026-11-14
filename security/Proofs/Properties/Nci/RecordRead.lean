module

public import NciCheck.Record
public import Spot.Bits

/-!
# What the record check holds is the record's text's

`Record.read` reads an execution phase's circuit record in two steps, `front` and `finish`, and what it holds of a
record it accepts it holds as `RecordText`'s readers give it, of the check's own Program and text (`Case.f`'s `P`
and `doc`). A statement about the check may then name those readers in place of the check.

* `inst_of_read`: the Program and the text are the instance's, decoded and parsed.
* `wiring_of_read`: the wiring's cuts and wires are the text's (`RecordText.wiresOf`, as `Spot.Wires.Wiring.ofJson`
  reads it).
* `slot_of_read`: a unit's block is the text's (`RecordText.slot`).
* `declared_of_heldAt`: a row the check holds is one the text declares, for the reason it is held, the pin aside
  (`RecordText.declared`). The check may hold fewer rows than the text declares.

`bitsOf_eq`: a type's bits as the check reads them, from the type's structure (`Flock.Sigma.bits`), are its leaves'
widths summed (`Spot.Bits.bitsOf`), on every type.
-/

@[expose] public section

namespace NciCheck.Record.SecurityProofs

open RecordText
open Lean (Json)
open Flock.Program (Ty)

private theorem foldl_add (l : List Nat) (a : Nat) : l.foldl (· + ·) a = a + l.sum := by
  induction l generalizing a with
  | nil => simp
  | cons x xs ih => simp [ih, Nat.add_assoc]

/-- **A type's bits, read from its structure, are its leaves' widths summed.** -/
theorem bitsOf_eq (t : Ty) : Spot.Bits.bitsOf t = Flock.Sigma.bits t := by
  rw [Spot.Bits.bitsOf, ← Array.foldl_toList, foldl_add, Nat.zero_add, Flock.Sigma.widths_sum]

/-- A stage whose refusal is recoded gave what it returns. -/
theorem recode_ok {α : Type} {code : String} {x : Except String α} {a : α} (h : recode code x = .ok a) :
    x = .ok a := by
  cases x <;> simp_all [recode]

/-- The record's wiring keeps the text's cuts and wires. -/
theorem wiringOf_text {f : Front} {text W : Spot.Wires.Wiring} (h : wiringOf f text = .ok W) :
    W.cuts = text.cuts ∧ W.wires = text.wires := by
  unfold wiringOf at h
  split at h
  · cases h; exact ⟨rfl, rfl⟩
  · cases h

/-- `finish` keeps its front, and reads the grids, the text's wiring and the declarations as the text's readers do. -/
theorem finish_text {f : Front} {pin : String → Option String} {c : Case} (h : finish f pin = .ok c) :
    c.f = f ∧ gridsOf f.ts f.doc = .ok (c.S, c.grids) ∧
      ∃ text, wiring f.doc = .ok text ∧ inputsOf f.ts text f.doc = .ok c.x ∧
        c.W.cuts = text.cuts ∧ c.W.wires = text.wires := by
  unfold finish at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  split at h
  · cases h
  · rename_i text htext
    split at h
    · cases h
    · rename_i W hW
      have hcw := wiringOf_text hW
      split at h
      · split at h
        · cases h
        · split at h
          · cases h
          · rename_i x hx
            split at h
            · cases h
            split at h
            · cases h
            · rename_i g hg
              split at h
              · cases h
              · cases h
                exact ⟨rfl, hg, text, recode_ok htext, hx, hcw.1, hcw.2⟩
      · cases h

/-- `front` reads its templates as `templatesOf` does, of its own Program and record. -/
theorem front_text {j tr : Json} {f : Front} (h : front j tr = .ok f) : templatesOf f.P f.doc = .ok f.ts := by
  unfold front at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  assumption

/-- `front`'s Program and record are the instance's, decoded and parsed. -/
theorem front_inst {j tr : Json} {f : Front} (h : front j tr = .ok f) :
    decodeProgram f.i.program = .ok f.P ∧ Flock.JsonParse.parse f.i.record = .ok f.doc := by
  unfold front at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  exact ⟨by assumption, by assumption⟩

/-- What `front` checks of the Program it reads: no reference out of order, the root laid out, input nodes at the root
only, every body walkable, and every unit's rows its own. -/
theorem front_program {j tr : Json} {f : Front} (h : front j tr = .ok f) :
    f.P.orderViolation = none ∧ Desc.rootLaidOut f.P = true ∧ Desc.inputsAtRoot f.P = true ∧
      Desc.walkable f.P = true ∧ ownRows f.P = true := by
  unfold front at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat (split at h <;> try cases h)
  rename_i P _ ho hl hw _ _ _ _ _ _ _ _ _ _ _ hown
  show P.orderViolation = none ∧ Desc.rootLaidOut P = true ∧ Desc.inputsAtRoot P = true ∧
    Desc.walkable P = true ∧ ownRows P = true
  simp only [Bool.not_eq_true', Bool.not_eq_false, Bool.and_eq_true] at hl hw hown
  refine ⟨?_, hl.1, hl.2, hw, hown⟩
  cases hov : P.orderViolation with
  | none => rfl
  | some x => rw [hov] at ho; exact absurd rfl ho

/-- A Program whose templates the text's reader gives has no root node of a composite keyed as an input id. -/
theorem templatesOf_keyed {P : Flock.Program.Prog} {doc : Json} {ts : List Tmpl} (h : templatesOf P doc = .ok ts) :
    keyedAsInput P = false := by
  unfold templatesOf at h
  simp (config := { iota := false }) only [bind, Except.bind, pure, Except.pure, throw, throwThe,
    MonadExceptOf.throw] at h
  split at h
  · cases h
  · split at h
    · cases h
    · rename_i hk
      simpa using hk

/-- A record `read` accepts passed `front`, then `finish` on that front. -/
theorem read_front {j tr : Json} {c : Case} (h : read j tr = .ok c) :
    ∃ f pin, front j tr = .ok f ∧ finish f pin = .ok c := by
  unfold read at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i f hf
    exact ⟨f, _, hf, h⟩

/-- Where `heldAt` gives a reason, the declarations declare the row to hold it, without its pin. -/
theorem heldAt_declared {c : Case} {r : Row} {hd : Held} (hh : c.heldAt r = some hd) :
    c.x.declared r = some hd.declared := by
  unfold Case.heldAt at hh
  unfold Inputs.declared
  split at hh
  · cases hh
  · rename_i fill hfill
    rw [hfill]
    simp only [Option.bind_some]
    split at hh
    · rename_i p hdec
      split at hh
      · cases hh; simpa [Held.declared] using hdec
      · cases hh
    · rename_i name param root hdec
      split at hh
      · cases hh; simpa [Held.declared] using hdec
      · cases hh
    · rename_i name param i hdec
      split at hh
      · cases hpin : (c.x.openings[i]?.bind fun o => c.pins.lookup o.reader) with
        | none => simp [hpin] at hh
        | some pin => simp [hpin] at hh; cases hh; simpa [Held.declared] using hdec
      · cases hh
    · cases hh

/-- **The Program and the text are the instance's**, on a record `read` accepts. -/
theorem inst_of_read {j tr : Json} {c : Case} (h : read j tr = .ok c) :
    decodeProgram c.f.i.program = .ok c.f.P ∧ Flock.JsonParse.parse c.f.i.record = .ok c.f.doc := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, -⟩ := finish_text hc
  exact front_inst hf

/-- **What `read` checks of the Program**, on a record it accepts: no reference out of order, the root laid out,
input nodes at the root only and none of a composite keyed as an input id, every body walkable, and every unit's
rows its own. -/
theorem program_of_read {j tr : Json} {c : Case} (h : read j tr = .ok c) :
    c.f.P.orderViolation = none ∧ Desc.rootLaidOut c.f.P = true ∧ Desc.inputsAtRoot c.f.P = true ∧
      Desc.walkable c.f.P = true ∧ ownRows c.f.P = true ∧ keyedAsInput c.f.P = false := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, -⟩ := finish_text hc
  obtain ⟨h1, h2, h3, h4, h5⟩ := front_program hf
  exact ⟨h1, h2, h3, h4, h5, templatesOf_keyed (front_text hf)⟩

/-- **The wiring's cuts and wires are the text's**, on a record `read` accepts. -/
theorem wiring_of_read {j tr : Json} {c : Case} (h : read j tr = .ok c) :
    ∃ text, Spot.Wires.Wiring.ofJson (wiresOf c.f.doc) = .ok text ∧ c.W.cuts = text.cuts ∧ c.W.wires = text.wires := by
  obtain ⟨f, pin, -, hc⟩ := read_front h
  obtain ⟨rfl, -, text, ht, -, hcuts, hwires⟩ := finish_text hc
  exact ⟨text, ht, hcuts, hwires⟩

/-- **A unit's block is the text's**, on a record `read` accepts. -/
theorem slot_of_read {j tr : Json} {c : Case} (h : read j tr = .ok c) : c.slot = slot c.f.P c.f.doc := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, hg, -⟩ := finish_text hc
  funext tid u
  simp [Case.slot, slot, front_text hf, hg]

/-- **A held row is declared by the text**, for the reason it is held, on a record `read` accepts. -/
theorem declared_of_heldAt {j tr : Json} {c : Case} (h : read j tr = .ok c) {r : Row} {hd : Held}
    (hh : c.heldAt r = some hd) : declared c.f.P c.f.doc r = some hd.declared := by
  obtain ⟨f, pin, hf, hc⟩ := read_front h
  obtain ⟨rfl, -, text, ht, hx, -⟩ := finish_text hc
  have hd' : declared c.f.P c.f.doc r = c.x.declared r := by simp [declared, front_text hf, ht, hx]
  rw [hd']
  exact heldAt_declared hh

end NciCheck.Record.SecurityProofs
