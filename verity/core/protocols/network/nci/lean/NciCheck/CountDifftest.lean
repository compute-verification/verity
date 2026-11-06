module

public import NciCheck.Count

/-!
# The count mode's difftest, against the Python reference

`scripts/count_difftest_vectors.py` writes instances of `NciCheck.Count` (the Program, θ by root node, the block
Definitions, the gate table and `(c, k, n)`) and `scripts/nci_reference.py`'s `check_count` on each, which unrolls each
block Definition's activations where `NciCheck.Count` reads one summary per Definition. `NciCheck/CountDifftestMain.lean`
fails unless `NciCheck.Count.check` gives Python's result on every instance: per block Definition its Boolean gates, fresh
bits and blocks, the total, the verdict and the first failure, or the first refusal's code. The Lean audit runs it (`runs`
in `lean-audit.json`): compiled evaluation, a test and not a proof.
-/

@[expose] public section

namespace NciCheck.CountDifftest

open Lean (Json)

/-- The fields compared: a refusal's code, or the verdict's. -/
def fields : List String := ["error", "blocks", "total", "complies", "failure"]

/-- How instance line `i` differs from Python's result, one line per field of `fields`. -/
def differences (i : Nat) (inst python : String) : List String :=
  match Json.parse inst, Json.parse python with
  | .error e, _ => [s!"instance {i}: not JSON: {e}"]
  | _, .error e => [s!"instance {i}: Python's results don't parse: {e}"]
  | .ok j, .ok py =>
    let lean := Count.check j
    fields.filterMap fun k =>
      let a := lean.getObjValD k
      let b := py.getObjValD k
      if a == b then none else some s!"instance {i}: {k}: Lean {a.compress}, Python {b.compress}"

/-- `lake env lean --run NciCheck/CountDifftestMain.lean INSTANCES PYTHON N`: fails unless both files have `N` lines and
no line differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args
    | IO.eprintln "usage: NciCheck/CountDifftestMain.lean INSTANCES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"count difftest: {a.size} instances and {b.size} results, not {n}"
    return 1
  let tasks := (List.range a.size).map fun i => Task.spawn fun _ => differences i a[i]! b[i]!
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"count difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end NciCheck.CountDifftest
