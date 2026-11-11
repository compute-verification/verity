module

public import Proofs.Properties.Nci.LeafWidthsBody
public import Proofs.Properties.Nci.LeafWidthsResolve
public import Proofs.Properties.Nci.RecordLanding
public import Proofs.Properties.Nci.RecordPrim
public import Proofs.Properties.Nci.RecordRead
public import Proofs.Properties.Nci.RecordWhole

/-!
# The widths' and the bits' statements, asked

Statements only, each to be proved in a file of its own set: `RecordWide.lean` (K1, K2) and `RecordAligned.lean`
(K3, K4). A statement here is the form the two assemblies read; a proof may take fewer hypotheses than it lists,
never more. Proved, and read by them: what a unit of a primitive gives (`RecordPrim.lean`: `unit_param_ok`,
`outLeaf_prim`), and that a member's argument leaf is as wide as the callee's parameter leaf
(`LeafWidthsResolve.lean`: `arg_wide`, which K1 reads at the root).
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row Tmpl templatesOf)
open NciCheck.Record (unwhole leavesOf passedOn)
open NciCheck.Properties.SecurityProofs
open Lean (Json)

variable {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}

/-! ## K1, K2: widths (`RecordWide.lean`) -/

/-- **K1. A gate is as wide as the leaf that names it** (`LeafWide`). -/
theorem leafWide (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) : LeafWide P rs W := by
  sorry

/-- **K2. A wired row holds its output** (`OutFits`). -/
theorem outFits (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) : OutFits P rs W S := by
  sorry

/-! ## K3, K4: a wire's two rows (`RecordAligned.lean`) -/

/-- **K3. Every wire lands whole** (`Aligned`), of a record whose wires pass the check's `unwhole`. -/
theorem aligned (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) {doc : Json} {ts : List Tmpl}
    (hts : templatesOf P doc = .ok ts)
    (hwh : unwhole ts (ts.map (leavesOf P (passedOn P))) S.wires = none) : Aligned P rs W S := by
  sorry

/-- **K4. A wired row is its writer's bits** (`Filled`), of a record whose wires pass the check's `unwhole`. -/
theorem filled (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) (W.templates.map (·.1)).toArray = .ok rs)
    (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) {doc : Json} {ts : List Tmpl}
    (hts : templatesOf P doc = .ok ts)
    (hwh : unwhole ts (ts.map (leavesOf P (passedOn P))) S.wires = none) : Filled P rs W S := by
  sorry

end NciCheck.Properties.Record.SecurityProofs
