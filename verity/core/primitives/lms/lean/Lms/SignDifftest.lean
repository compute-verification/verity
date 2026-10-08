import Lms.Sign
import Lean.Data.Json

/-!
# The sign difftest: LMS keys and signing against `lms.py`, on generated seeds

`scripts/sign_vectors.py` writes instances, each a parameter set, a key made by `keygen` from a seed or by `key` from an
identifier and a seed, and leaves and messages to sign at, and `lms.py`'s results on each. `Lms/SignDifftestMain.lean`
reads both files and fails unless `evaluate`, which computes the same results with `Lms.Sign`'s and `Lms.Verify`'s
definitions, gives every one of them. The Lean audit runs it (`runs` in `lean-audit.json`): compiled evaluation, a test
and not a proof. `evaluate` gives:
* `params`: whether `Params` admits (N, W, H, toy);
* `public`: the key's public key, `null` where `keygen` or `key` refuses;
* `signatures`: each `signAt`'s encoding with `verify`'s verdict on it, `null` where `signAt` refuses the leaf.
-/

namespace Lms.SignDifftest

open Lean Flock

def bytesOf (j : Json) : Except String ByteArray := do unhex (← j.getStr?)

/-- What `Lms.Sign`'s definitions give on an instance, in the shape of the Python's results. -/
def evaluate (j : Json) : Except String Json := do
  let #[n, w, h, toy] ← j.getObjValAs? (Array Json) "params" | throw "not [N, W, H, toy]"
  let seed ← bytesOf (← j.getObjVal? "seed")
  let signs ← (← j.getObjValAs? (Array Json) "signs").mapM fun s => do
    let #[q, m] ← s.getArr? | throw s!"not [q, message]: {s.compress}"
    pure ((← q.getNat?), ← bytesOf m)
  let refused (params : Bool) := Json.mkObj [("params", params), ("public", .null), ("signatures", .null)]
  let .ok p := Params.make (← n.getNat?) (← w.getNat?) (← h.getNat?) (← toy.getBool?) | pure (refused false)
  let made ← match ← j.getObjValAs? String "make" with
    | "keygen" => pure (keygen p seed)
    | "key" => pure (key p (← bytesOf (← j.getObjVal? "I")) seed)
    | m => throw s!"no key maker {m}"
  let .ok k := made | pure (refused true)
  let pub := k.publicKey
  pure <| Json.mkObj [("params", true), ("public", .str (hexOf pub)),
    ("signatures", .arr (signs.map fun (q, m) => match signAt k q m with
      | .ok sig => .arr #[.str (hexOf sig.encode), toJson ((verify p pub m sig).toOption.getD false)]
      | .error _ => .null))]

/-- The results' fields, in the order they are compared. -/
def fields : List String := ["params", "public", "signatures"]

/-- How the Lean side's results for instance line `i` differ from Python's, one line per field that differs. -/
def differences (i : Nat) (inst python : String) : List String :=
  match Json.parse inst >>= evaluate, Json.parse python with
  | .error e, _ => [s!"instance {i}: the Lean side rejects it: {e}"]
  | _, .error e => [s!"instance {i}: Python's results don't parse: {e}"]
  | .ok lean, .ok py =>
    let found := fields.filterMap fun k =>
      let a := lean.getObjValD k
      let b := py.getObjValD k
      if a == b then none else some s!"instance {i}: {k}: Lean {a.compress.take 300}, Python {b.compress.take 300}"
    if found.isEmpty && lean != py then [s!"instance {i}: Python's results have fields Lean doesn't give"] else found

/-- `lake env lean --run Lms/SignDifftestMain.lean INSTANCES PYTHON N`: every instance line of `INSTANCES` evaluated, each
compared with the same line of `PYTHON`. Fails unless both files have `N` lines and none differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: Lms/SignDifftestMain.lean INSTANCES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"sign difftest: {a.size} instances and {b.size} results, not {n}"
    return 1
  let tasks := (a.zip b).toList.zipIdx.map fun ((x, y), i) => Task.spawn fun _ => differences i x y
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"sign difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end Lms.SignDifftest
