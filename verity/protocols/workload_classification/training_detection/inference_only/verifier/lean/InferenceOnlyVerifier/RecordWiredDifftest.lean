module

public import InferenceOnlyVerifier.RecordWired
public import InferenceOnlyVerifier.RecordDifftest

/-!
# The record check's wired reading's difftest, against the Python reference

`scripts/record_wired_difftest_vectors.py` writes cases (an instance, the trusted input, the rows whose kinds are
asked, and whether the root has an edge) and, on each, `scripts/inference_only_record_reference.py`'s wired reading,
which visits every row of every unit and resolves every leaf of every edge's row to its gate.
`RecordWiredDifftestMain.lean`, beside this file, fails unless, on every case:

* `RecordWired.checkDetail` gives Python's `check_detail` with `wired`: the summary's counts, the verdict and the
  failure, or the first refusal's code;
* `RecordWired.kinds` gives Python's `kinds` with `wired`, the class of each row asked, or the refusal's code;
* where the root has no edge, `RecordWired` gives what `Record` gives: `Record.checkDetail`'s counts, verdict and
  refusal's code, and `Record.kinds`.

Compiled evaluation, a test and not a proof.
-/

@[expose] public section

namespace InferenceOnlyVerifier.RecordWiredDifftest

open Lean (Json)
open InferenceOnlyVerifier.CountDifftest (compare)
open InferenceOnlyVerifier.RecordDifftest (rowsDiffer)

/-- The fields of the whole verdict compared: a refusal's code, or the summary's counts and the verdict. -/
def detailFields : List String := ["error", "blocks", "gates", "inputs", "fresh", "cross_wires", "free", "ingress",
  "unregistered", "openings", "wired_rows", "wired_bits", "complies", "failure"]

/-- How case line `i` differs from Python's results on it. -/
def differences (i : Nat) (case py : String) : List String :=
  match Json.parse case, Json.parse py with
  | .error e, _ => [s!"instance {i}: not JSON: {e}"]
  | _, .error e => [s!"instance {i}: Python's results don't parse: {e}"]
  | .ok c, .ok p =>
    let inst := c.getObjValD "instance"
    let tr := c.getObjValD "trusted"
    let rows := c.getObjValD "rows"
    let detail := RecordWired.checkDetail inst tr
    let kinds := RecordWired.kinds inst tr rows
    compare i "detail" detailFields detail (p.getObjValD "detail") ++
      rowsDiffer "kinds" i kinds (p.getObjValD "kinds") ++
      if c.getObjValD "edge" == Json.bool false then
        compare i "unwired detail" detailFields detail (Record.checkDetail inst tr) ++
          rowsDiffer "unwired kinds" i kinds (Record.kinds inst tr rows)
      else []

/-- `lake env lean --run InferenceOnlyVerifier/RecordWiredDifftestMain.lean CASES PYTHON N`: fails unless the two files
have `N` lines and no line differs. -/
def main (args : List String) : IO UInt32 := do
  let [cases, python, n] := args
    | IO.eprintln "usage: InferenceOnlyVerifier/RecordWiredDifftestMain.lean CASES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines cases
  let b ← lines python
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"record wired difftest: {a.size} cases and {b.size} results, not {n}"
    return 1
  let tasks := (List.range a.size).map fun i => Task.spawn fun _ => differences i a[i]! b[i]!
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"record wired difftest: {a.size} cases compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end InferenceOnlyVerifier.RecordWiredDifftest
