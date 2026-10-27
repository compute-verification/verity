/-
`RecVStar.checkHolo`: the parsed circuit's META `holo` is V*'s file's. A part's statement says `holo` exactly over a holo
inner session, and no other statement ever does: a plain session's part with META `holo`, a holo session's part
without it, any level with it and a `holo` that is not a boolean are refused.
-/
import Flock.RecVStar

namespace Flock.Test.Holo

open Flock Flock.RecVStar

def inner : VStar.Stmt := .inner 0 "rec-residuals/m23-k22-c8-p0" default

def level : VStar.Stmt := .recOpen 0 8 1 2

def circ (mj : Lean.Json := Lean.Json.null) : Circuit :=
  { (default : Circuit) with metaJ := mj }

def holoMeta (v : Lean.Json) : Lean.Json := Lean.Json.mkObj [("holo", v)]

def ok (s : VStar.Stmt) (c : Circuit) (holo : Bool) : Bool := (checkHolo s c holo).isOk

#guard ok inner (circ) false
#guard ok inner (circ (holoMeta (.bool false))) false
#guard ok inner (circ (holoMeta (.bool true))) true
#guard !ok inner (circ (holoMeta (.bool true))) false
#guard !ok inner (circ) true
#guard ok level (circ) true
#guard ok level (circ) false
#guard !ok level (circ (holoMeta (.bool true))) true
#guard !ok inner (circ (holoMeta (.str "yes"))) false

end Flock.Test.Holo
