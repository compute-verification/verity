module

public import Properties.Pouw.Credit
public import Proofs.Pouw.Period.Accept
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.Array.QSort.Basic
import all Init.Data.List.Sort.Basic
import all Init.Data.Nat.Bitwise.Basic
import all Init.Data.Repr
import all Init.Data.String.Defs

/-!
# The lanes window's strata and law

The compiled `strata` and `law` run on the lanes window in the kernel (`decide +kernel`, through the `DecidableEq` the
sample's structures derive here), past `strata`'s key sort, which the kernel doesn't reduce (`keys_lanes`, as
`Proofs.Pouw.Period.Accept` reads H1's items). Their loops, sorts and the template's digits are core's, so this module
imports those in full. `Proofs.Pouw.Credit` reads the results from here: with core's definitions in full there, its
`Nat.choose` bounds exceed their budget.
-/

@[expose] public section

namespace Properties.Pouw.Credit.SecurityProofs

open Lean (toJson)

deriving instance DecidableEq for Flock.Draw.Stratum
deriving instance DecidableEq for Flock.Draw.WorkStratum

theorem ok_of_toOption {ε α : Type} {x : Except ε α} {a : α} (h : x.toOption = some a) : x = .ok a := by
  cases x <;> simp_all [Except.toOption]

theorem keys_lanes : PouwCheck.keys lanesLayout = some PouwCheck.Credit.LAYOUT_KEYS :=
  Properties.Pouw.Period.SecurityProofs.Accept.keys_of (by decide) (by decide +kernel)

theorem strata_lanes : PouwCheck.Credit.strata lanesLayout = .ok lanesStrata := by
  apply ok_of_toOption
  unfold PouwCheck.Credit.strata
  rw [keys_lanes]
  decide +kernel

theorem law_lanes : PouwCheck.Credit.law lanesStrata (toJson lanesK) = .ok lanesLaw :=
  ok_of_toOption (by decide +kernel)

end Properties.Pouw.Credit.SecurityProofs
