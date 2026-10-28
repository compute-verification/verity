/-
`HmIn.derivedGroups` and `HmIn.checkDerived`: the verifier derives a public input itself only for a one-word port of seed
`instance` and derivation `mod:G` (`G` as written by `toString`, `0 < G ≤ 2^16`), every port so; it then holds instance
`i`'s word of port `k` to `i mod G_k`.
-/
import Flock.HmRow

namespace Flock.Test.DerivedPos

open Lean

def port (name seed derivation : String) : Json :=
  Json.mkObj [("port", name), ("words", (1 : Nat)), ("seed", seed), ("derivation", derivation)]

def ports (ps : Array (String × String × String)) (words : Nat := 1) : HmIn.Ports :=
  { json := Json.arr (ps.map fun (n, s, d) => port n s d), ports := ps.map fun (n, _, _) => (n, words) }

#guard HmIn.derivedGroups (ports #[("pos", "instance", "mod:4")]) == some #[4]
#guard HmIn.derivedGroups (ports #[("pos", "instance", "mod:4"), ("b", "instance", "mod:65536")]) == some #[4, 65536]
#guard HmIn.derivedGroups (ports #[]) == none
#guard HmIn.derivedGroups (ports #[("pos", "instance", "mod:4"), ("t", "token", "mod:4")]) == none
#guard HmIn.derivedGroups (ports #[("pos", "instance", "mod:0")]) == none
#guard HmIn.derivedGroups (ports #[("pos", "instance", "mod:04")]) == none
#guard HmIn.derivedGroups (ports #[("pos", "instance", "mod:65537")]) == none
#guard HmIn.derivedGroups (ports #[("pos", "instance", "div:4")]) == none
#guard HmIn.derivedGroups (ports #[("pos", "instance", "mod:4")] (words := 2)) == none

def pub (outputs : Array Nat) (population : Option (Array Nat) := none) : Public :=
  { (default : Public) with
    n := outputs.size, outputs := outputs, population := population.map fun ws => (ws.size, #[], ws) }

def two := ports #[("pos", "instance", "mod:2"), ("q", "instance", "mod:3")]

def passes (pi : HmIn.Ports) (p : Public) (gs : Array Nat) : Bool := (HmIn.checkDerived pi p gs).toBool

#guard passes (ports #[("pos", "instance", "mod:4")]) (pub #[0, 1, 2, 3, 0, 1]) #[4]
#guard !passes (ports #[("pos", "instance", "mod:4")]) (pub #[0, 1, 2, 3, 0, 2]) #[4]
#guard passes two { (default : Public) with n := 4, outputs := #[0, 0, 1, 1, 0, 2, 1, 0] } #[2, 3]
#guard !passes two { (default : Public) with n := 4, outputs := #[0, 0, 1, 1, 0, 2, 1, 1] } #[2, 3]
#guard passes (ports #[("pos", "instance", "mod:2")]) (pub #[7] (some #[0, 1, 0, 1])) #[2]
#guard !passes (ports #[("pos", "instance", "mod:2")]) (pub #[0, 1, 0, 1] (some #[0, 1, 1])) #[2]

end Flock.Test.DerivedPos
