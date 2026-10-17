/-
`RecVStar.checkMsgNames` on a part (`.inner`) and a part of R's opening (`.opening`), each with row ports `m0` and `acc0`:
a circuit whose ports are the statement's passes, and one with a port `m1` the statement has not is refused by name, at
both forms; a port not named as a message round's (`acc1`, `v`) is never refused, nor is any port of another statement.
A circuit naming a port twice is refused at every form, the statement's own ports included. A part's port `m<k>` past its
rep's message rounds is refused even where the statement has it: at `m22-k6-c4` the rep has 46 (`m0` … `m45`, Ligerito's
included), at `m27-k24-c8` 80, whose part 1 the honest prover names through `m79`; at a shape with no fast100 schedule every
`m<k>` is refused. The count is V*'s spec's ports (`VStar.Algebra.rounds`) at every fast100 `m`. Over a holo inner session
(`holo`) the rep has one round more, h's (`holoRepRounds`), as V*'s holo spec has one port more: `m46` passes at
`m22-k6-c4`, `m47` does not.
-/
import Flock.RecVStar

namespace Flock.Test.MsgNames

open Flock Flock.RecVStar

def circ (names : List String) : Circuit :=
  { (default : Circuit) with ports := (names.map fun nm => ({ name := nm, words := 64, chunks := 1, nb := 1 } : Port)).toArray }

def part : VStar.Stmt := .inner 0 "rec-residuals/m22-k6-c4-p0" { (default : VStar.Part) with S := { gf := default, ports := [("m0", 8), ("acc0", 8)] } }

def partOf (name : String) (k : Nat) : VStar.Stmt :=
  .inner 0 name { (default : VStar.Part) with S := { gf := default, ports := [(s!"m{k}", 8), ("acc0", 8)] } }

def partAt : Nat → VStar.Stmt := partOf "rec-residuals/m22-k6-c4-p0"

def opening : VStar.Stmt := .opening 22 2 0 { ops := [], res := [] } [("m0", 8), ("acc0", 8)] 1

def refused (s : VStar.Stmt) (names : List String) (holo : Bool := false) : Bool :=
  (checkMsgNames s (circ names) holo).toOption.isNone

#guard !refused part ["m0", "acc0", "v", "out"]
#guard !refused (partAt 24) ["m24", "acc0"]
#guard !refused (partAt 45) ["m45", "acc0"]
#guard refused (partAt 46) ["m46", "acc0"]
#guard !refused (partOf "rec-residuals/m27-k24-c8-p1" 51) ["m51", "acc0"]
#guard !refused (partOf "rec-residuals/m27-k24-c8-p1" 79) ["m79", "acc0"]
#guard refused (partOf "rec-residuals/m27-k24-c8-p1" 80) ["m80", "acc0"]
#guard !refused (partOf "rec-residuals/m27-k22-c8-p1" 77) ["m77", "acc0"]
#guard refused (partOf "rec-residuals/m27-k22-c8-p1" 78) ["m78", "acc0"]
#guard refused (partOf "rec-residuals/m21-k6-c4-p0" 0) ["m0", "acc0"]
#guard (List.range 14).all fun d => [(6, 2), (6, 4), (22, 8), (24, 8)].all fun (k, c) =>
  innerMsgRounds (partOf (RecResiduals.unitName (22 + d) k c 0) 0) ==
    (fast100 (22 + d)).map fun s => (VStar.Algebra.rounds ⟨22 + d, k, c⟩ s).size
#guard !refused (partAt 46) ["m46", "acc0"] (holo := true)
#guard refused (partAt 47) ["m47", "acc0"] (holo := true)
#guard (List.range 14).all fun d => [(6, 2), (6, 4), (22, 8), (24, 8)].all fun (k, c) =>
  innerMsgRounds (partOf (RecResiduals.unitName (22 + d) k c 0) 0) true ==
    (innerMsgRounds (partOf (RecResiduals.unitName (22 + d) k c 0) 0)).map (· + 1)
#guard (VStar.Algebra.specHolo ⟨22, 6, 4⟩).toOption.map (·.ports.length - 2) == innerMsgRounds (partAt 0) true
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
