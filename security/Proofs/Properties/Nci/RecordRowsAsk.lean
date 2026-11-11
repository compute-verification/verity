module

public import Proofs.Properties.Nci.RecordPrim
public import Proofs.Properties.Nci.RecordRead
public import Proofs.Properties.Nci.RecordRowBits
public import Proofs.Properties.Nci.RecordWired

/-!
# The rows' and the declarations' statements, asked

Statements only, each to be proved in a file of its own set: `RecordRows.lean` (R1 to R5) and `RecordInputs.lean`
(I1 to I3). A statement here is the form the two assemblies read; a proof may take fewer hypotheses than it lists,
never more. What a unit of a primitive gives them is proved, in `RecordPrim.lean`: `prim_unit_operands`.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row Tmpl)
open NciCheck.Properties.SecurityProofs
open Lean (Json)

variable {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}

/-! ## R: a unit's rows (`RecordRows.lean`) -/

/-- **R1. A leaf of a unit's parameter is a leaf of one of the unit's rows**: the row its cut puts it in. -/
theorem rowLeaf_of_param
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) {tid : String} {u : ℕ} {sc : Scope}
    (hu : UnitAt P rs tid u sc) {i t g : ℕ} (hi : i < (P.defs[sc.d]!).params.size)
    (ht : t < ((P.defs[sc.d]!).params[i]!).2.leaves)
    (hg : Flock.Program.resolve P sc (⟨true, i, t⟩ : Ref) = .ok g) :
    ∃ r l lo hi, r.tid = tid ∧ r.unit = u ∧ RowLeaf P rs W r l g lo hi := by
  sorry

/-- **R2. What a unit returns is its own** (`ReturnsOwn`). -/
theorem returnsOwn (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) : ReturnsOwn P rs W := by
  sorry

/-- **R3. What a gate of a unit reads is the unit's own** (`OperandsOwn`). -/
theorem operandsOwn (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) : OperandsOwn P rs W := by
  sorry

/-- **R4. Every unit's rows are its own input gates** (`OwnRows`), of a Program whose root passes the check's
`ownRows`. -/
theorem ownRows_holds (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) : OwnRows P rs W := by
  sorry

/-- **R5. The wires rank the units** (`Ranked`), of a wiring `Spot.Wires.check` accepts. -/
theorem ranked (hS : Spot.Wires.check W = .ok S) : ∃ ρ : String × ℤ → ℕ, Ranked S ρ := by
  sorry

/-! ## I: the declarations (`RecordInputs.lean`) -/

/-- **I1. The wires' declarations stand first** in the declarations `inputsOf` reads, in the wires' order. -/
theorem inputsOf_wires {ts : List Tmpl} {doc : Json} {x : NciCheck.RecordText.Inputs}
    (h : NciCheck.RecordText.inputsOf ts W doc = .ok x) :
    ∃ rest, x.decls = (W.wires.zipIdx.map fun (w, i) =>
      ({ w, bits := 16 * NciCheck.RecordText.rowWords ts W w, fill := .wire i } : NciCheck.RecordText.Decl)) ++
        rest := by
  sorry

/-- **I2. The summary the check keeps is `Spot.Wires.check`'s** of the wiring it keeps. -/
theorem wired_of_read {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c) :
    Spot.Wires.check c.W = .ok c.wired := by
  sorry

/-- **I3. No row the check holds has a writer.** -/
theorem held_unwired {j tr : Json} {c : NciCheck.Record.Case} (h : NciCheck.Record.read j tr = .ok c) (r : Row)
    (hh : (c.heldAt r).isSome = true) : c.wired.writerOf r.tid r.unit r.port r.row = none := by
  sorry

end NciCheck.Properties.Record.SecurityProofs
