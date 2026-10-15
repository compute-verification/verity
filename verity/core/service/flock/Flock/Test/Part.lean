/-
`RecVStar.checkPart` where it fails closed before the verifier's part is computed: a part's statement over a unit not
named `rec-residuals/m<m>-k<k_log>-c<claims>-p<j>`, at a shape with no `fast100` schedule (m = 6), or whose META's
`holo` is not a boolean, is refused; a level's statement passes whatever its unit. The whole S's row-port names
(`portNames`) are one per port, `m0` first and the running sums last. The verifier's part at a real shape
takes minutes to evaluate interpreted, so the comparison itself is held to V*'s honest parts by the flock suite's
`test_f1_the_session_meets` (V*'s statements `alg/p0`, `alg/p1` at m = 27, through `checkCircuit`).
-/
import Flock.RecVStar

namespace Flock.Test.Part

open Flock Flock.RecVStar

def inner : VStar.Stmt := .inner 0 "rec-residuals/m23-k22-c8-p0" default

def circ (name : String) (mj : Lean.Json := Lean.Json.null) : Circuit :=
  { (default : Circuit) with nets := #[(name, { (default : Net) with name })], unitNet := 0, metaJ := mj }

def ok (s : VStar.Stmt) (c : Circuit) : Bool := (checkPart s c).isOk

#guard ok (.recOpen 0 8 1 2) (circ "rec-open/l8-h1")
#guard ok (.recOpen 0 8 1 2) (circ "rec-residuals/m6-k6-c2-p0")
#guard !ok inner (circ "rec-open/l8-h1")
#guard !ok inner (circ "rec-residuals/m23-k22-c8")
#guard !ok inner (circ "rec-residuals/m6-k6-c2-p0")
#guard !ok inner (circ "rec-residuals/m23-k22-c8-p0" (Lean.Json.mkObj [("holo", Lean.Json.str "yes")]))

#guard match RecResiduals.shape? 23 22 8 with
  | some sh => let ns := portNames sh
    ns.size == (RecResiduals.portSizes sh).size && ns[0]! == "m0" && ns.toList.drop (ns.size - 2) == ["acc0", "acc1"]
  | none => false

end Flock.Test.Part
