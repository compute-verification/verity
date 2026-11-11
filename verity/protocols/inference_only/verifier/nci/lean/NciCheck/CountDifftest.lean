module

public import NciCheck.Count

/-!
# The count mode's difftest, against the Python reference

`scripts/count_difftest_vectors.py` writes instances of `NciCheck.Count` (the Program, θ by root node, the block
Definitions, the gate table and `(c, k, n)`) and, on each, `scripts/nci_reference.py`'s `check_count_detail` and
`check_count`, which unroll each block Definition's activations where `NciCheck.Count` reads one summary per Definition.
`NciCheck/CountDifftestMain.lean` fails unless both forms give Python's on every instance: `NciCheck.Count.checkDetail`,
per block Definition its Boolean gates, fresh bits and blocks, the total, the verdict and the first failure, or the
first refusal's code; and `NciCheck.Count.check`, what the count publishes, the verdict or the refusal's code and no
other field. The Lean audit runs it (`runs` in `lean-audit.json`): compiled evaluation, a test and not a proof.
-/

@[expose] public section

namespace NciCheck.CountDifftest

open Lean (Json)

/-- The fields of the whole verdict compared: a refusal's code, or the verdict's. -/
def detailFields : List String := ["error", "blocks", "total", "complies", "failure"]

/-- The fields compared of what the count publishes: a refusal's code, or the judgment against `n`. -/
def publicFields : List String := ["error", "complies"]

/-- The fields a published result may hold: `publicFields` and a refusal's `message`, which Python doesn't write. -/
def publishable : List String := "message" :: publicFields

/-- How Lean's `lean` differs from Python's `py` in `form` on instance line `i`, one line per field of `fields`. -/
def compare (i : Nat) (form : String) (fields : List String) (lean py : Json) : List String :=
  fields.filterMap fun k =>
    let a := lean.getObjValD k
    let b := py.getObjValD k
    if a == b then none else some s!"instance {i}: {form} {k}: Lean {a.compress}, Python {b.compress}"

/-- The fields of `o` that a published result may not hold, as a line each for `who` on instance line `i`. -/
def unpublishable (i : Nat) (who : String) (o : Json) : List String :=
  match o with
  | .obj kvs => (kvs.foldl (init := []) fun acc k _ => k :: acc).reverse.filterMap fun k =>
      if publishable.contains k then none else some s!"instance {i}: public: {who} publishes {k}"
  | _ => [s!"instance {i}: public: {who}'s result is not an object"]

/-- How instance line `i` differs from Python's two results, the whole verdict's (`detail`) and the published one's
(`pub`). -/
def differences (i : Nat) (inst detail pub : String) : List String :=
  match Json.parse inst, Json.parse detail, Json.parse pub with
  | .error e, _, _ => [s!"instance {i}: not JSON: {e}"]
  | _, .error e, _ => [s!"instance {i}: Python's detail results don't parse: {e}"]
  | _, _, .error e => [s!"instance {i}: Python's public results don't parse: {e}"]
  | .ok j, .ok d, .ok p =>
    let lean := Count.check j
    compare i "detail" detailFields (Count.checkDetail j) d ++ compare i "public" publicFields lean p ++
      unpublishable i "Lean" lean ++ unpublishable i "Python" p

/-- `lake env lean --run NciCheck/CountDifftestMain.lean INSTANCES DETAIL PUBLIC N`: fails unless the three files have
`N` lines and no line differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, det, pub, n] := args
    | IO.eprintln "usage: NciCheck/CountDifftestMain.lean INSTANCES DETAIL PUBLIC N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines det
  let c ← lines pub
  if n.toNat? != some a.size ∨ a.size != b.size ∨ a.size != c.size then
    IO.eprintln s!"count difftest: {a.size} instances, {b.size} detail and {c.size} public results, not {n}"
    return 1
  let tasks := (List.range a.size).map fun i => Task.spawn fun _ => differences i a[i]! b[i]! c[i]!
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"count difftest: {a.size} instances compared in both forms, {found.length} differences"
  return if found.isEmpty then 0 else 1

end NciCheck.CountDifftest
