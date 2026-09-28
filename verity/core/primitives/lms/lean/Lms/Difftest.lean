import Lms.Verify
import Lean.Data.Json

/-!
# The difftest: LMS verification against `lms.py`, on generated keys and signatures

`scripts/difftest_vectors.py` writes instances, each a parameter set, a public key, a message and a signature's bytes,
and `lms.py`'s results on each. `Lms/DifftestMain.lean` reads both files and fails unless `evaluate`, which computes the
same results with `Lms.Verify`'s definitions, gives every one of them. The Lean audit runs it (`runs` in
`lean-audit.json`): compiled evaluation, a test and not a proof. `evaluate` gives:
* `params`: whether `Params` admits (N, W, H, toy);
* `decode`: whether `Signature.decode` reads the signature's bytes (`null` without parameters);
* `digits`: the chain lengths of the signature's message hash under the key's I (`null` without a signature);
* `root`: `root_of` under the key's I, `null` where it refuses;
* `verify`: `verify`'s verdict, `null` where it refuses (a public key of another length) or there is no signature.
-/

namespace Lms.Difftest

open Lean Flock

def bytesOf (j : Json) : Except String ByteArray := do unhex (← j.getStr?)

/-- What `Lms.Verify`'s definitions give on an instance, in the shape of the Python's results. -/
def evaluate (j : Json) : Except String Json := do
  let #[n, w, h, toy] ← j.getObjValAs? (Array Json) "params" | throw "not [N, W, H, toy]"
  let pub ← bytesOf (← j.getObjVal? "public")
  let message ← bytesOf (← j.getObjVal? "message")
  let signature ← bytesOf (← j.getObjVal? "signature")
  match Params.make (← n.getNat?) (← w.getNat?) (← h.getNat?) (← toy.getBool?) with
  | .error _ => pure (Json.mkObj [("params", false), ("decode", .null), ("digits", .null), ("root", .null),
      ("verify", .null)])
  | .ok p =>
    match Signature.decode p signature with
    | .error _ => pure (Json.mkObj [("params", true), ("decode", false), ("digits", .null), ("root", .null),
        ("verify", .null)])
    | .ok sig =>
      let I := pub.extract 0 16
      let ok {α : Type} (f : α → Json) : Except String α → Json
        | .ok a => f a
        | .error _ => .null
      pure <| Json.mkObj [
        ("params", true), ("decode", true),
        ("digits", toJson (digits p (messageHash p I sig.q sig.C message))),
        ("root", ok (Json.str ∘ hexOf) (rootOf p I message sig)),
        ("verify", ok toJson (verify p pub message sig))]

/-- The results' fields, in the order they are compared. -/
def fields : List String := ["params", "decode", "digits", "root", "verify"]

/-- How the Lean side's results for instance line `i` differ from Python's, one line per field that differs. -/
def differences (i : Nat) (inst python : String) : List String :=
  match Json.parse inst >>= evaluate, Json.parse python with
  | .error e, _ => [s!"instance {i}: the Lean side rejects it: {e}"]
  | _, .error e => [s!"instance {i}: Python's results don't parse: {e}"]
  | .ok lean, .ok py =>
    let found := fields.filterMap fun k =>
      let a := lean.getObjValD k
      let b := py.getObjValD k
      if a == b then none else some s!"instance {i}: {k}: Lean {a.compress}, Python {b.compress}"
    if found.isEmpty && lean != py then [s!"instance {i}: Python's results have fields Lean doesn't give"] else found

/-- `lake env lean --run Lms/DifftestMain.lean INSTANCES PYTHON N`: every instance line of `INSTANCES` evaluated, each
compared with the same line of `PYTHON`. Fails unless both files have `N` lines and none differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: Lms/DifftestMain.lean INSTANCES PYTHON N"; return 2
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

end Lms.Difftest
