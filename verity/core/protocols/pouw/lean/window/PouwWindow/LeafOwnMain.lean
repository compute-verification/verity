import PouwWindow.RegFile

/-!
`pouw-leaf-own REGISTRATION PRODUCED REGISTERED R C K (LANE | combine)`: the leaf verifier's check of one drawn-leaf
statement's `--registered` file against the accepted registration and produced record (`PouwWindow.Lanes.leafOwn?`), at
the `R × C` tile of contraction `K`, every value's rows `hm96-sha512/row/v2`. It prints `OWN {"accepted", "why"?}` and
exits 0 where it accepts, 1 where it refuses and 2 on a usage error.
-/

open Lean (Json)

def readJson (path : String) : IO Json := do
  match Json.parse (← IO.FS.readFile path) with
  | .ok j => pure j
  | .error e => throw (IO.userError s!"{path}: {e}")

def usage (why : String) : IO UInt32 := do
  IO.eprintln why
  pure 2

def main (args : List String) : IO UInt32 := do
  let [reg, prod, file, r, c, k, stmt] := args
    | usage "usage: pouw-leaf-own REGISTRATION PRODUCED REGISTERED R C K (LANE | combine)"
  let (some r, some c, some k) := (r.toNat?, c.toNat?, k.toNat?)
    | usage "R, C and K are natural numbers"
  let some lane := (if stmt == "combine" then some none else stmt.toNat?.map some : Option (Option Nat))
    | usage s!"the statement is a lane's index or combine, not {stmt}"
  match PouwWindow.Lanes.leafOwn? (← readJson reg) (← readJson prod) (← readJson file) ⟨r, c, k⟩ lane
      Flock.HmRow.ROW_SCHEMA_V2 with
  | .ok () =>
    IO.println s!"OWN {(Json.mkObj [("accepted", Json.bool true)]).compress}"
    return 0
  | .error e =>
    IO.println s!"OWN {(Json.mkObj [("accepted", Json.bool false), ("why", Json.str e)]).compress}"
    return 1
