import NciCheck.Check

/-!
# The difftest: the NCI check against the Python reference, on random declarations

`scripts/difftest_vectors.py` generates instances (declarations as descriptor v1 Programs with θ, divisions, `Γ` and
`α`) and runs the Python reference on them (`scripts/nci_reference.py`). `NciCheck/DifftestMain.lean` reads both files
and fails unless `NciCheck.check` gives every result: per IU its size and novel reads, the verdict and the first failing
IU with its rule, or the code of the first malformation. The Lean audit runs it (`runs` in `lean-audit.json`): compiled
evaluation, a test and not a proof.
-/

namespace NciCheck.Difftest

open Lean (Json)

/-- The results' fields, in the order they are compared. -/
def fields : List String := ["error", "gates", "ius", "complies", "failure"]

/-- How the Lean check's results for instance line `i` differ from Python's, one line per field that differs. -/
def differences (i : Nat) (inst python : String) : List String :=
  match Json.parse inst, Json.parse python with
  | .error e, _ => [s!"instance {i}: not JSON: {e}"]
  | _, .error e => [s!"instance {i}: Python's results don't parse: {e}"]
  | .ok j, .ok py =>
    let lean := check j
    fields.filterMap fun k =>
      let a := lean.getObjValD k
      let b := py.getObjValD k
      if a == b then none else some s!"instance {i}: {k}: Lean {a.compress}, Python {b.compress}"

/-- `lake env lean --run NciCheck/DifftestMain.lean INSTANCES PYTHON N`: every instance line of `INSTANCES` checked, each
compared with the same line of `PYTHON`. Fails unless both files have `N` lines and none differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: NciCheck/DifftestMain.lean INSTANCES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"difftest: {a.size} instances and {b.size} results, not {n}"
    return 1
  let tasks := (a.zip b).toList.zipIdx.map fun ((x, y), i) => Task.spawn fun _ => differences i x y
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end NciCheck.Difftest
