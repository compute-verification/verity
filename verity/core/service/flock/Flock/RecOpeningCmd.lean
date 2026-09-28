import Flock.HmRow
import Flock.RecOpening

/-!
`flock-rec-opening`'s command: the verifier's registered-opening units (`Flock.RecOpening`), which
`test_rec_opening_unit` holds to `rec_residuals`' builders.

    flock-rec-opening structure M CLAIMS     the opening's S and its parts, one JSON line
    flock-rec-opening check CIRCUIT ...      `RecOpening.check` on each circuit file, one JSON line each

`structure` prints S as `rec_algebra.structure` writes it (`{"nw", "nv", "ops", "res"}`), each row port's elements
(`rec_residuals.spec`'s ports, unnamed) and each part's S, row ports' elements and v elements' number. `check` parses each
file as `verity/flock-circuit` and prints `{"circuit", "ok"}`, with `"error"` when refused.
-/

namespace Flock.RecOpeningCmd

open Lean

def termJ : RecAlgebra.Term → Json
  | .w i => .arr #[.str "w", toJson i]
  | .v j => .arr #[.str "v", toJson j]
  | .wt i u => .arr #[.str "wt", toJson i, toJson u]

def termJ' : RecOpen.Term → Json
  | .w i => .arr #[.str "w", toJson i]
  | .v j => .arr #[.str "v", toJson j]
  | .wt i u => .arr #[.str "wt", toJson i, toJson u]

def pairJ (ab : Nat × Nat) : Json := .arr #[toJson ab.1, toJson ab.2]

def wholeJ (S : RecAlgebra.GfS) : Json :=
  Json.mkObj [("nw", toJson S.nw), ("nv", toJson S.nv), ("ops", .arr (S.ops.map fun o => .arr (o.map termJ))),
    ("res", .arr (S.res.map fun r => Json.mkObj [("mul", .arr (r.mul.map pairJ)), ("add", .arr (r.add.map termJ))]))]

def partJ (S : RecOpen.GfS) (ns : List Nat) (nv : Nat) : Json :=
  Json.mkObj [("ports", toJson ns), ("nv", toJson nv),
    ("ops", .arr (S.ops.toArray.map fun o => .arr (o.toArray.map termJ'))),
    ("res", .arr (S.res.toArray.map fun r =>
      Json.mkObj [("mul", .arr (r.mul.toArray.map pairJ)), ("add", .arr (r.add.toArray.map termJ'))]))]

def structureJ (m c : Nat) : Except String Json := do
  let some sh := RecOpening.shape? m c | throw s!"no opening at m {m}, claims {c}"
  let S ← RecOpening.structureOf sh
  let n := (← RecResiduals.groups S (RecOpening.portSizes sh)).size
  let parts ← (List.range n).mapM fun j => do
    let (P, ns, nv) ← RecOpening.part sh j
    pure (partJ P ns nv)
  pure (Json.mkObj [("gf", wholeJ S), ("ports", toJson (RecOpening.portSizes sh)), ("parts", .arr parts.toArray)])

def checkOne (path : String) : IO (Bool × Json) := do
  let b ← IO.FS.readBinFile path
  let r : V Unit := do RecOpening.check (← HmRow.parse Tags.circuit b [])
  pure <| match r with
    | .ok () => (true, Json.mkObj [("circuit", .str path), ("ok", .bool true)])
    | .error e => (false, Json.mkObj [("circuit", .str path), ("ok", .bool false), ("error", .str e)])

def cmd (args : List String) : IO UInt32 := do
  match args with
  | ["structure", m, c] =>
    match m.toNat?, c.toNat? with
    | some m, some c => match structureJ m c with
      | .ok j => IO.println j.compress; return 0
      | .error e => IO.eprintln e; return 1
    | _, _ => IO.eprintln "structure M CLAIMS: two numbers"; return 2
  | "check" :: files =>
    let mut rc : UInt32 := 0
    for f in files do
      let (ok, j) ← checkOne f
      IO.println j.compress
      if !ok then rc := 1
    return rc
  | _ => IO.eprintln "usage: flock-rec-opening (structure M CLAIMS | check CIRCUIT ...)"; return 2

end Flock.RecOpeningCmd
