import NciCheck.Baseline
import NciCheck.DescDifftest

/-!
# The difftest of θ's provenance: against the Python reference, and against the unrolled check

`scripts/baseline_difftest_vectors.py` writes cases, each `{"instance", "trusted"}` (an instance of `NciCheck.Desc` with
θ's `provenance`, and the trusted input beside it), each expanded gate by gate where the policy check gives it a verdict
and its provenance is one entry per θ entry, and `scripts/nci_reference.py`'s `check_baseline` on each.
`NciCheck/BaselineDifftestMain.lean` fails unless `NciCheck.Baseline.check` gives Python's result on every case and
`NciCheck.Baseline.checkUnrolled`'s on every expansion: per IU its size and novel reads, the verdict and the first failing
IU, or the first refusal's code. The Lean audit runs it (`runs` in `lean-audit.json`): compiled evaluation, a test and not
a proof.
-/

namespace NciCheck.BaselineDifftest

open Lean (Json)

/-- How case line `i` differs: `NciCheck.Baseline.check` against Python's result and, on an expansion, against
`NciCheck.Baseline.checkUnrolled` on it. -/
def differences (i : Nat) (case exp python : String) : List String :=
  match Json.parse case, Json.parse exp, Json.parse python with
  | .error e, _, _ => [s!"case {i}: not JSON: {e}"]
  | _, .error e, _ => [s!"case {i}: the expansion doesn't parse: {e}"]
  | _, _, .error e => [s!"case {i}: Python's results don't parse: {e}"]
  | .ok c, .ok x, .ok py =>
    let out := Baseline.check (c.getObjValD "instance") (c.getObjValD "trusted")
    DescDifftest.compare i "Python" out py ++
      (if x.isNull then []
       else DescDifftest.compare i "Baseline.checkUnrolled" out
        (Baseline.checkUnrolled (x.getObjValD "instance") (x.getObjValD "trusted")))

/-- `lake env lean --run NciCheck/BaselineDifftestMain.lean CASES EXPANDED PYTHON N`: fails unless the three files have
`N` lines each and no line differs. -/
def main (args : List String) : IO UInt32 := do
  let [cases, exp, py, n] := args
    | IO.eprintln "usage: NciCheck/BaselineDifftestMain.lean CASES EXPANDED PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines cases
  let e ← lines exp
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size ∨ a.size != e.size then
    IO.eprintln s!"baseline difftest: {a.size} cases, {e.size} expansions and {b.size} results, not {n}"
    return 1
  let tasks := (List.range a.size).map fun i => Task.spawn fun _ => differences i a[i]! e[i]! b[i]!
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"baseline difftest: {a.size} cases compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end NciCheck.BaselineDifftest
