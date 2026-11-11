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
never more.
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

/-! ## Assumed by name: in work, in `RecordPrim.lean` -/

/-- **A leaf of a unit's parameter resolves**, whatever the unit's Definition. -/
theorem unit_param_ok {ts : Array String} (hP : ReadProg P) (hO : NciCheck.Record.ownRows P = true)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) {i t : ℕ} (hi : i < (P.defs[sc.d]!).params.size)
    (ht : t < ((P.defs[sc.d]!).params[i]!).2.leaves) :
    ∃ g, Flock.Program.resolve P sc (⟨true, i, t⟩ : Ref) = .ok g := by
  sorry

/-- **The output rows of a unit of a primitive are one row of one leaf**: the unit's gate, at the bits of the
value its Definition returns. -/
theorem outLeaf_prim {ts : Array String} (hP : ReadProg P)
    (hrs : Flock.TemplateQuery.evaluate (Flock.TemplateQuery.Desc.ofProg P) ts = .ok rs) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) (hp : (P.defs[sc.d]!).prim = true) :
    ∃ w, (P.defs[sc.d]!).ret = .value w ∧ OutLeaf P sc ("o" ++ toString 0) 0 sc.off 0 w ∧
      ∀ out l g lo hi, OutLeaf P sc out l g lo hi →
        out = "o" ++ toString 0 ∧ l = 0 ∧ g = sc.off ∧ lo = 0 ∧ hi = w := by
  sorry

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
