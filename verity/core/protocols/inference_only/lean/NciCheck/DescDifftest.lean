import NciCheck.Desc
import NciCheck.Difftest

/-!
# The difftest of the description-sized check: against the unrolled one, and against the Python reference

`scripts/desc_difftest_vectors.py` writes instances of `NciCheck.Desc` (θ and the owners per root node), each one expanded
gate by gate where it is well-formed, and `scripts/nci_reference.py`'s `check_desc` on each. `NciCheck/DescDifftestMain.lean`
fails unless `NciCheck.Desc.check` gives Python's result on every instance and `NciCheck.check`'s on every expansion: per
IU its size and novel reads, the verdict and the first failing IU, or the first malformation's code. The Lean audit runs
it (`runs` in `lean-audit.json`): compiled evaluation, a test and not a proof.
-/

namespace NciCheck.DescDifftest

open Lean (Json)

/-- How `a` differs from `b`, one line per field of `NciCheck.Difftest.fields`. -/
def compare (i : Nat) (what : String) (a b : Json) : List String :=
  NciCheck.Difftest.fields.filterMap fun k =>
    let x := a.getObjValD k
    let y := b.getObjValD k
    if x == y then none else some s!"instance {i}: {k}: Desc {x.compress}, {what} {y.compress}"

/-- How instance line `i` differs: `NciCheck.Desc.check` against Python's result and, on an expansion, against
`NciCheck.check` on it. -/
def differences (i : Nat) (inst exp python : String) : List String :=
  match Json.parse inst, Json.parse exp, Json.parse python with
  | .error e, _, _ => [s!"instance {i}: not JSON: {e}"]
  | _, .error e, _ => [s!"instance {i}: the expansion doesn't parse: {e}"]
  | _, _, .error e => [s!"instance {i}: Python's results don't parse: {e}"]
  | .ok j, .ok x, .ok py =>
    let desc := Desc.check j
    compare i "Python" desc py ++ (if x.isNull then [] else compare i "NciCheck.check" desc (check x))

/-- `lake env lean --run NciCheck/DescDifftestMain.lean INSTANCES EXPANDED PYTHON N`: fails unless the three files have
`N` lines each and no line differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, exp, py, n] := args
    | IO.eprintln "usage: NciCheck/DescDifftestMain.lean INSTANCES EXPANDED PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let e ← lines exp
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size ∨ a.size != e.size then
    IO.eprintln s!"desc difftest: {a.size} instances, {e.size} expansions and {b.size} results, not {n}"
    return 1
  let tasks := (List.range a.size).map fun i => Task.spawn fun _ => differences i a[i]! e[i]! b[i]!
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"desc difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end NciCheck.DescDifftest
