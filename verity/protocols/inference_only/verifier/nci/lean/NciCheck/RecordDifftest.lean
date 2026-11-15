module

public import NciCheck.RecordCount
public import NciCheck.CountDifftest

/-!
# The record check's difftest, against the Python reference

`scripts/record_difftest_vectors.py` writes cases of `Record` (an instance, the trusted input, the rows whose
kinds are asked, and the count's limits) and, on each, `scripts/nci_record_reference.py`'s results, which visit every
row of every unit where `Record` reads the record's description. `NciCheck/RecordDifftestMain.lean` fails
unless, on every case:

* `Record.checkDetail` gives Python's `check_detail`: the summary's counts, the verdict and the failure, or
  the first refusal's code, with its message where it has one;
* `Record.check` gives Python's `check`, the verdict or the refusal's code and message and no other field;
* `Record.kinds` gives Python's `kinds`, the class of each row asked, or the refusal's code;
* `Record.exported` gives Python's `exported`, of each row asked its unit's slot and why the row is θ, or
  the refusal's code;
* `RecordCount.checkDetail` gives Python's `count`: a block's Boolean gates, the verdict and the failure, or
  the first refusal's code;
* the templates `Record.front` reads have the closed rows of Python's `closed`, template by template, or the
  two refuse with one code: what each Definition passes on (`Record.passedOn`), held on every template
  whether or not a wire fills its rows;
* **the same circuit**: where the case states the instance of SPOT's `wires` piece that Python builds from its own
  layout of each template, SPOT's reader (`Spot.Wires.Wiring.ofJson`) reads from it the wiring it reads from the
  form's own (`Record.piece`), and the form refuses the record's wires exactly where SPOT's reader or its
  check (`Spot.Wires.check`) refuses that instance.

The Lean audit runs it (`runs` in `lean-audit.json`): compiled evaluation, a test and not a proof.
-/

@[expose] public section

namespace NciCheck.RecordDifftest

open Lean (Json)
open NciCheck.CountDifftest (compare unpublishable)

/-- The fields of the whole verdict compared: a refusal's code and message, or the summary's counts and the verdict. -/
def detailFields : List String := ["error", "message", "blocks", "gates", "inputs", "fresh", "cross_wires", "free",
  "ingress", "unregistered", "openings", "wired_rows", "wired_bits", "complies", "failure"]

/-- The fields compared of what the form would publish: a refusal's code and message, or the judgment. -/
def publicFields : List String := ["error", "message", "complies"]

/-- The fields of the count compared: a refusal's code, or a block's Boolean gates and the verdict. -/
def countFields : List String := ["error", "bool_gates", "complies", "failure"]

/-- How Lean's list of `what` (the rows' kinds, what the form exports of them, or the templates' closed rows) differs
from Python's on case line `i`: both the same list, or both a refusal of one code. -/
def rowsDiffer (what : String) (i : Nat) (lean py : Json) : List String :=
  let same := match lean, py with
    | .arr a, .arr b => a == b
    | .arr _, _ | _, .arr _ => false
    | _, _ => lean.getObjValD "error" == py.getObjValD "error" && lean.getObjValD "error" != Json.null
  if same then [] else [s!"instance {i}: {what}: Lean {lean.compress}, Python {py.compress}"]

/-- Each template's closed rows as the form reads them (`Record.Leaves.closed`), `[template, rows]` each, or the
refusal of a stage as far as `own-rows`. -/
def closedOf (inst tr : Json) : Json :=
  match Record.front inst tr with
  | .ok f =>
    Json.arr (f.leaves.map fun l => Json.arr #[Json.str l.tid, Json.arr (l.closed.map Json.str).toArray]).toArray
  | .error code => NciCheck.refusal code

/-- The wiring SPOT's reader reads from the `wires` piece's instance `j`, shown; `refused` where it refuses. -/
def shown (j : Json) : String :=
  match Spot.Wires.Wiring.ofJson j with
  | .ok W => s!"{repr W}"
  | .error _ => "refused"

/-- Whether SPOT's reader and its check take the `wires` piece's instance `j`. -/
def taken (j : Json) : Bool :=
  match Spot.Wires.Wiring.ofJson j with
  | .ok W => match Spot.Wires.check W with
    | .ok _ => true
    | .error _ => false
  | .error _ => false

/-- **The same circuit** on case line `i`, where the case states SPOT's instance `spot` (the module docstring);
`code` is the form's refusal, `null` for none. -/
def circuitDiffers (i : Nat) (inst tr spot code : Json) : List String :=
  if spot == Json.null then [] else
  match Record.front inst tr with
  | .error e => [s!"instance {i}: same circuit: the form refuses the case before its wires ({e})"]
  | .ok f =>
    let mine := shown (Record.piece f)
    let theirs := shown spot
    (if mine == theirs then [] else
      [s!"instance {i}: same circuit: the form's wiring is {mine.take 600}, SPOT's instance gives {theirs.take 600}"]) ++
    (if (code == Json.str "wires") == !taken spot then [] else
      [s!"instance {i}: same circuit: the form's refusal is {code.compress}, and SPOT " ++
        (if taken spot then "takes" else "refuses") ++ " its instance"])

/-- How case line `i` differs from Python's results on it. -/
def differences (i : Nat) (case py : String) : List String :=
  match Json.parse case, Json.parse py with
  | .error e, _ => [s!"instance {i}: not JSON: {e}"]
  | _, .error e => [s!"instance {i}: Python's results don't parse: {e}"]
  | .ok c, .ok p =>
    let inst := c.getObjValD "instance"
    let tr := c.getObjValD "trusted"
    let detail := Record.checkDetail inst tr
    let pub := Record.check inst tr
    compare i "detail" detailFields detail (p.getObjValD "detail") ++
      compare i "public" publicFields pub (p.getObjValD "public") ++
      unpublishable i "Lean" pub ++ unpublishable i "Python" (p.getObjValD "public") ++
      rowsDiffer "kinds" i (Record.kinds inst tr (c.getObjValD "rows")) (p.getObjValD "kinds") ++
      rowsDiffer "exported" i (Record.exported inst tr (c.getObjValD "rows")) (p.getObjValD "exported") ++
      compare i "count" countFields (RecordCount.checkDetail inst tr (c.getObjValD "count")) (p.getObjValD "count") ++
      rowsDiffer "closed" i (closedOf inst tr) (p.getObjValD "closed") ++
      circuitDiffers i inst tr (c.getObjValD "spot") (detail.getObjValD "error")

/-- `lake env lean --run NciCheck/RecordDifftestMain.lean CASES PYTHON N`: fails unless the two files have `N` lines
and no line differs. -/
def main (args : List String) : IO UInt32 := do
  let [cases, python, n] := args
    | IO.eprintln "usage: NciCheck/RecordDifftestMain.lean CASES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines cases
  let b ← lines python
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"record difftest: {a.size} cases and {b.size} results, not {n}"
    return 1
  let tasks := (List.range a.size).map fun i => Task.spawn fun _ => differences i a[i]! b[i]!
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"record difftest: {a.size} cases compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end NciCheck.RecordDifftest
