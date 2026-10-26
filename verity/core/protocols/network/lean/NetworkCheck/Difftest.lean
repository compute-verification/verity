import NetworkCheck.Certificate
import Lean.Data.Json

/-!
# The difftest: the network certificate's encodings against `certificate.py`, on generated instances

`scripts/difftest_vectors.py` writes instances and the Python's results on each. `NetworkCheck/DifftestMain.lean` reads
both files and fails unless `evaluate`, which computes the same results with `NetworkCheck.Certificate`'s definitions,
gives every one of them, byte for byte. The Lean audit runs it (`runs` in `lean-audit.json`): compiled evaluation, a test
and not a proof.

An instance is a window's grid parameters and Σ_sync (or none), its grid, the link, direction, window, status and
deadline of its record, and the queries on it: a frame list (`frames`, the grid's real frames or another), the frames
to open, `verify_frame` queries, record heads given field by field, signed digests, certificate messages and nodes.
`evaluate` gives, with `null` where Python raises `ValueError`:
* `depth` and `empty`, `E_0 … E_D`;
* `real_frames` of the grid; `leaves`, `frame_leaf` of each frame of `frames`; `root`, `frames_root` of `frames`;
  `opens`, `open_frame` of `frames` at each k; `verifies`, each query's `verify_frame`;
* `manifest`, the canonical JSON `parameters_digest` hashes, and `parameters`, the digest;
* `record_head`, the encoding and digest of `record_head` of the grid under that digest; `heads`, each given head's;
* `digests`, each `certificate_digest` of a salt and a record digest;
* `messages`, each `certificate_message`, sessions of 31 and 33 bytes among them, which both refuse; `nodes`, each
  `node`.
-/

namespace NetworkCheck.Difftest

open Lean Flock

def hexJson (b : ByteArray) : Json := .str (hexOf b)

def bytesOf (j : Json) : Except String ByteArray := do unhex (← j.getStr?)

def arrOf (j : Json) : Except String (Array Json) := j.getArr?

def slotOf : Json → Except String (Option ByteArray)
  | .null => pure none
  | j => some <$> bytesOf j

def reasonOf : Json → Except String (Option String)
  | .null => pure none
  | j => some <$> j.getStr?

def paramsOf (j : Json) : Except String Params := do
  pure { buckets := ← j.getObjValAs? Nat "T", allocation := ← j.getObjValAs? Nat "r",
         frameBytes := ← j.getObjValAs? Nat "B", bucketMs := ← j.getObjValAs? Nat "tau",
         queueCapacity := ← j.getObjValAs? Nat "Q" }

def syncSetOf : Json → Except String (Option SyncSet)
  | .null => pure none
  | j => do
    let rhos ← (← arrOf (← j.getObjVal? "rhos")).mapM fun r => do
      let #[n, d] ← arrOf r | throw s!"not [num, den]: {r.compress}"
      pure ((← n.getInt?), (← d.getNat?))
    pure (some { deltas := ← j.getObjValAs? (Array Int) "deltas", rhos })

def headOf (j : Json) : Except String RecordHead := do
  pure { link := ← j.getObjValAs? String "link", direction := ← Direction.ofValue (← j.getObjValAs? String "direction"),
         window := ← j.getObjValAs? Int "window", buckets := ← j.getObjValAs? Int "T",
         allocation := ← j.getObjValAs? Int "r", frameBytes := ← j.getObjValAs? Int "B",
         bucketMs := ← j.getObjValAs? Int "tau", status := ← reasonOf (← j.getObjVal? "status"),
         deadline := ← j.getObjValAs? Bool "deadline", parameters := ← bytesOf (← j.getObjVal? "parameters"),
         counts := ← j.getObjValAs? (Array Int) "counts", root := ← bytesOf (← j.getObjVal? "root") }

/-- A result, `null` where the definition refuses (Python's `ValueError`). -/
def orNull {α : Type} (f : α → Json) : Except String α → Json
  | .ok a => f a
  | .error _ => .null

def headJson (h : Except String RecordHead) : Json :=
  orNull id do
    let h ← h
    pure (.arr #[hexJson (← h.encode), hexJson (← h.digest)])

/-- What `NetworkCheck.Certificate`'s definitions give on an instance, in the shape of the Python's results. -/
def evaluate (j : Json) : Except String Json := do
  let p ← paramsOf (← j.getObjVal? "params")
  let ss ← syncSetOf (← j.getObjVal? "sync_set")
  let grid ← (← arrOf (← j.getObjVal? "grid")).mapM fun r => do (← arrOf r).mapM slotOf
  let frames ← (← arrOf (← j.getObjVal? "frames")).mapM bytesOf
  let link ← j.getObjValAs? String "link"
  let direction ← Direction.ofValue (← j.getObjValAs? String "direction")
  let window ← j.getObjValAs? Int "window"
  let status ← reasonOf (← j.getObjVal? "status")
  let deadline ← j.getObjValAs? Bool "deadline"
  let opens ← j.getObjValAs? (Array Int) "opens"
  let verifies ← (← arrOf (← j.getObjVal? "verifies")).mapM fun q => do
    let #[root, k, frame, path] ← arrOf q | throw s!"not [root, k, frame, path]: {q.compress}"
    pure (← bytesOf root, ← k.getInt?, ← bytesOf frame, ← (← arrOf path).mapM bytesOf)
  let heads ← (← arrOf (← j.getObjVal? "heads")).mapM headOf
  let digests ← (← arrOf (← j.getObjVal? "digests")).mapM fun d => do
    let #[salt, record] ← arrOf d | throw s!"not [salt, record]: {d.compress}"
    pure (← bytesOf salt, ← bytesOf record)
  let messages ← (← arrOf (← j.getObjVal? "messages")).mapM fun m => do
    let #[i, s, l, d, w, g] ← arrOf m
      | throw s!"not [key_id, session, link, direction, window, digest]: {m.compress}"
    pure (← bytesOf i, ← bytesOf s, ← l.getStr?, ← Direction.ofValue (← d.getStr?), ← w.getInt?, ← bytesOf g)
  let nodes ← (← arrOf (← j.getObjVal? "nodes")).mapM fun n => do
    let #[a, b] ← arrOf n | throw s!"not [left, right]: {n.compress}"
    pure (← bytesOf a, ← bytesOf b)
  let parameters := parametersDigest p ss
  pure <| Json.mkObj [
    ("depth", toJson (depth p)),
    ("empty", .arr ((List.range (depth p + 1)).toArray.map (hexJson ∘ empty))),
    ("real_frames", orNull (.arr ∘ Array.map hexJson) (realFrames grid p)),
    ("leaves", .arr (frames.map fun f => orNull hexJson (frameLeaf f p.frameBytes))),
    ("root", orNull hexJson (framesRoot frames p)),
    ("opens", .arr (opens.map fun k => orNull (.arr ∘ Array.map hexJson) (openFrame frames p k))),
    ("verifies", .arr (verifies.map fun (r, k, f, path) => toJson (verifyFrame r p k f path))),
    ("manifest", orNull Json.str (canon (parametersManifest p ss))),
    ("parameters", orNull hexJson parameters),
    ("record_head", headJson do recordHead link direction window grid status p (← parameters) deadline),
    ("heads", .arr (heads.map (headJson ∘ pure))),
    ("digests", .arr (digests.map fun (salt, record) => hexJson (certificateDigest salt record))),
    ("messages", .arr (messages.map fun (i, s, l, d, w, g) => orNull hexJson (certificateMessage i s l d w g))),
    ("nodes", .arr (nodes.map fun (a, b) => hexJson (node a b)))]

/-- The results' fields, in the order they are compared. -/
def fields : List String :=
  ["depth", "empty", "real_frames", "leaves", "root", "opens", "verifies", "manifest", "parameters", "record_head",
   "heads", "digests", "messages", "nodes"]

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

/-- `lake env lean --run NetworkCheck/DifftestMain.lean INSTANCES PYTHON N`: every instance line of `INSTANCES` evaluated,
each compared with the same line of `PYTHON`. Fails unless both files have `N` lines and none differs. -/
def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: NetworkCheck/DifftestMain.lean INSTANCES PYTHON N"; return 2
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

end NetworkCheck.Difftest
