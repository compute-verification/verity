/-
`RecVStar.checkTemplate`: the verifier's bindings' `program_digests` names V*'s template of the statement, the descriptor id
core's codec names its Definition by (`RecVStar.templateId`, each pinned below as `codec.descriptor_id` writes it). A part
relabelled under another template (another part's S, a level's, or the same S with a port renamed) is refused, and so is
a level under its neighbour's `H`, however consistently the bindings carry the label.
-/
import Flock.RecVStar

namespace Flock.Test.Template

open Lean Flock Flock.RecVStar

/-- `rec_residuals.definition(S)`'s S: one row port `m0` of one element, one v element, one residual. -/
def S : VStar.RepS :=
  { gf := { nw := 1, nv := 1, ops := [[.w 0], [.v 0]], res := [{ mul := [(0, 1)], add := [.wt 0 3] }] },
    ports := [("m0", 1)] }

/-- `descriptor_id(rec_residuals.definition(S))`. -/
def S_ID : String :=
  "InnerRepCheck_v2{S={\"dict\":{\"gf\":{\"dict\":{\"nv\":1,\"nw\":1,\"ops\":[[[\"w\",0]],[[\"v\",0]]]," ++
  "\"res\":[{\"dict\":{\"add\":[[\"wt\",0,3]],\"mul\":[[0,1]]}}]}},\"ports\":[[\"m0\",1]]}}}"

def part (S : VStar.RepS) : VStar.Stmt := .inner 0 "rec-residuals/m23-k22-c8-p0" { S, res := [0], v := [0] }

def openingS : RecOpen.GfS := { ops := [[.w 0], [.v 0]], res := [{ mul := [(0, 1)], add := [.wt 0 3] }] }

def opening : VStar.Stmt := .opening 22 2 0 openingS [("m0", 1)] 1

def level : VStar.Stmt := .recOpen 2 16 1 142

def bind (key : String) : VStar.Bindings :=
  { partition := "", program := "", programDigests := Json.mkObj [(key, Json.str (String.ofList (List.replicate 128 'e')))] }

def ok (s : VStar.Stmt) (f : VStar.Bindings) : Bool := (checkTemplate s f).isOk

def idOf (s : VStar.Stmt) : String := match templateId s with
  | .ok t => t
  | .error e => s!"refused: {e}"

def why (s : VStar.Stmt) (f : VStar.Bindings) : String := match checkTemplate s f with
  | .ok () => "passed"
  | .error e => e

#guard idOf level == "RecOpen_v4{LANES=16,H=1}"
#guard idOf (.recOpenStacked 0 16 1 2) == "RecOpen_v4{LANES=16,H=1}"
#guard idOf (.sparse { nu := 4, ell := 6, sLog := 2, tLog := 1 } 0) == "RecSparse_v1{NU=4,ELL=6,S_LOG=2,T_LOG=1}"
#guard idOf (.cap 0 128) == "RecCap_v1"
#guard idOf (part S) == S_ID
#guard idOf opening == S_ID

#guard ok (part S) (bind S_ID)
#guard ok opening (bind S_ID)
#guard ok level (bind "RecOpen_v4{LANES=16,H=1}")
#guard ok (.cap 3 8) (bind "RecCap_v1")
-- one part's template relabelled: another S (its port renamed), a level's, or the honest S under a later version
#guard !ok (part S) (bind (S_ID.replace "m0" "m1"))
#guard !ok (part { S with ports := [("m1", 1)] }) (bind S_ID)
#guard !ok (part S) (bind "RecOpen_v4{LANES=16,H=1}")
#guard !ok (part S) (bind (S_ID.replace "InnerRepCheck_v2" "InnerRepCheck_v3"))
#guard !ok level (bind "RecOpen_v4{LANES=16,H=2}")
#guard !ok (.cap 0 128) (bind "RecOpen_v4{LANES=16,H=1}")
#guard !ok level { bind "RecOpen_v4{LANES=16,H=1}" with
  programDigests := Json.mkObj [("RecOpen_v4{LANES=16,H=1}", "e"), ("RecCap_v1", "e")] }
#guard !ok level { bind "RecOpen_v4{LANES=16,H=1}" with programDigests := Json.arr #[] }
#guard why (part S) (bind (S_ID.replace "m0" "m1")) ==
  ("rec-residuals/m23-k22-c8-p0: the verifier's bindings' program_digests names " ++
    S_ID.replace "m0" "m1" ++ ", not V*'s template of the statement " ++ S_ID)

end Flock.Test.Template
