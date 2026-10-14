/-
`RecVStar.checkMsgNames` on a part (`.inner`) and a part of R's opening (`.opening`), each with row ports `m0` and `acc0`:
a circuit whose ports are the statement's passes, and one with a port `m1` the statement has not is refused by name, at
both forms; a port not named as a message round's (`acc1`, `v`) is never refused, nor is any port of another statement.
A circuit naming a port twice is refused at every form, the statement's own ports included.
-/
import Flock.RecVStar

namespace Flock.Test.MsgNames

open Flock Flock.RecVStar

def circ (names : List String) : Circuit :=
  { (default : Circuit) with ports := (names.map fun nm => ({ name := nm, words := 64, chunks := 1, nb := 1 } : Port)).toArray }

def part : VStar.Stmt := .inner 0 "rec-residuals/m22-k6-c4-p0" { (default : VStar.Part) with S := { gf := default, ports := [("m0", 8), ("acc0", 8)] } }

def partAt (k : Nat) : VStar.Stmt :=
  .inner 0 "rec-residuals/m22-k6-c4-p0"
    { (default : VStar.Part) with S := { gf := default, ports := [(s!"m{k}", 8), ("acc0", 8)] } }

def opening : VStar.Stmt := .opening 22 2 0 { ops := [], res := [] } [("m0", 8), ("acc0", 8)] 1

def refused (s : VStar.Stmt) (names : List String) : Bool := (checkMsgNames s (circ names)).toOption.isNone

#guard !refused part ["m0", "acc0", "v", "out"]
#guard !refused (partAt 23) ["m23", "acc0"]
#guard refused (partAt 24) ["m24", "acc0"]
#guard !refused opening ["m0", "acc0"]
#guard refused part ["m0", "m1", "acc0"]
#guard refused opening ["m0", "m1"]
#guard msgUnnamed part (circ ["m0", "m1", "m12"]) == some "m1"
#guard !refused part ["m0", "acc1"]
#guard !refused (.recOpen 0 4 2 4) ["m0", "m1"]
#guard refused part ["m0", "acc0", "v", "m0"]
#guard refused opening ["m0", "acc0", "acc0"]
#guard refused (.recOpen 0 4 2 4) ["out", "v", "out"]
#guard !refused (.recOpen 0 4 2 4) ["out", "v"]

end Flock.Test.MsgNames
