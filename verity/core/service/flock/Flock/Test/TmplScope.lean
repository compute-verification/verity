/-
`ProvedScope.circuitRules` on a templated circuit (`Circuit.tmpl`), on its description (`scopeOkT`): `Flock.Test.Tmpl`'s
unit at one `--zk` table with hidden outputs passes with or without the verifier's own registered reads, and with them a
shared-row public file (`sharedRowsOk`); without them a shared-row file is refused, and so are two tables, a session
without `--zk`, public outputs and outputs a tail stage computes (META `out_net`).
-/
import Flock.ProvedScope
import Flock.Test.Tmpl

namespace Flock.Test.TmplScope

open Flock Flock.Test.Tmpl

def unit : Except String TmplUnit := do
  TmplUnit.read (← Tmpl.parse tmplLines) .lenient

def tags (hidden : Bool) : Tags := { Tags.circuitTmpl with hiddenOutputs := hidden, merkleLeaf := "hm96-sha512/v1" }

/-- Whether `circuitRules` passes the unit's templated circuit (changed by `f`) at `--zk` `zk`, `count` tables, the
verifier's own registered reads `reg`, and a shared-row public file where `shared`. -/
def passes (zk : Bool) (count : Nat) (reg shared : Bool) (hidden : Bool := true) (f : Circuit → Circuit := id) :
    Option Bool :=
  unit.toOption.map fun tu =>
    let pub : Public := if shared then { (default : Public) with tables := some default } else default
    (ProvedScope.circuitRules (tags hidden) zk count reg (f { (default : Circuit) with tmpl := some tu }) pub).toOption.isSome

#guard unit.toOption.isSome
#guard passes true 1 false false == some true
#guard passes true 1 true false == some true
#guard passes true 1 true true == some true
#guard passes true 1 false true == some false
#guard passes true 2 true false == some false
#guard passes true 2 true true == some false
#guard passes false 1 true false == some false
#guard passes true 1 true false (hidden := false) == some false
#guard passes true 1 true true (f := fun c => { c with outNet := 1 }) == some false

end Flock.Test.TmplScope
