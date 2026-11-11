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
(K3, K4), and one asked of `LeafWidthsResolve.lean` (`arg_wide`, which K1 reads at the root). A statement here is
the form the two assemblies read; a proof may take fewer hypotheses than it lists, never more. What a unit of a
primitive gives them is proved, in `RecordPrim.lean`: `unit_param_ok`, `outLeaf_prim`.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program
open NciCheck.Properties.SecurityProofs.Resolve (Good RefOk BodyOk)

/-! ## Asked of the width check's soundness (`LeafWidthsResolve.lean`) -/

/-- **A member's argument leaf is as wide as the callee's parameter leaf, for any callee.** Member `m` of node `k`
of an activation's body reads leaf `j` of its parameter `i` at a reference of the node's argument `i`: reference
`j` where the node is a call, reference `Resolve.memberIdx _ m j` where it is a batch. The leaf that reference
names in the activation's body is as wide as leaf `j` of the callee's parameter, the callee a composite or a
primitive: nothing here reads the callee's body. -/
theorem arg_wide (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (hW : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyWide (P.defs[d]!).body)
    (parent : Scope) (hp : Good P parent) (k : ℕ) (hk : k < (P.defs[parent.d]!).body.nodes.size) (m : ℕ)
    (hm : m < ((P.defs[parent.d]!).body.nodes[k]!).members) (i : ℕ)
    (hi : i < (P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params.size) (j : ℕ)
    (hj : j < ((P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params[i]!).2.leaves) :
    (((P.defs[parent.d]!).body.nodes[k]!).form = .call →
      Widths.wAt (tyOf P parent ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get j))
          ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get j).j =
        Widths.wAt ((P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params[i]!).2 j) ∧
    (((P.defs[parent.d]!).body.nodes[k]!).form = .batch →
      Widths.wAt (tyOf P parent ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get
            (Resolve.memberIdx (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j)))
          ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get
            (Resolve.memberIdx (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j)).j =
        Widths.wAt ((P.defs[((P.defs[parent.d]!).body.nodes[k]!).fn]!).params[i]!).2 j) := by
  sorry

end NciCheck.Properties.SecurityProofs.LeafWidths

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
