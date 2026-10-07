import Flock.Canon
import PousCheck.Merkle

/-!
# The difftest: the PoUS check against the Python verifier, on generated instances

`scripts/difftest_vectors.py` writes instances and the Python's answers on each, one canonical JSON line per instance in
two files. `evaluate` computes the answers from the instance with `PousCheck`'s definitions, and `main` fails unless
every line's canonical JSON is the Python's line, byte for byte. The Lean audit runs it (`runs` in `lean-audit.json`):
a test, not a proof.

* `merkle`: `Merkle.leaf` or `Merkle.packageLeaf` of every block, `Merkle.root`, the SHA-512 of every `Merkle.path`,
  `Merkle.pathLength` of every leaf, `Merkle.verifyPath` on each check the Python ran (`checkInput` rebuilds its input
  from its name and arguments), and the leaf of each sample block.
-/

namespace PousCheck.Difftest

open Lean Flock

def hex (b : ByteArray) : Json := .str (hexOf b)

def num (n : Nat) : Json := .num (JsonNumber.fromNat n)

def bytesOf (j : Json) (ctx : String) : V ByteArray := do unhex (← J.asStr j ctx)

def bytesArr (j : Json) (ctx : String) : V (Array ByteArray) := do (← J.asArr j ctx).mapM (bytesOf · ctx)

/-- A check's index, leaf and path from its name and arguments, as the script's `check_input` builds them. -/
def checkInput (name : String) (i a b : Nat) (leaves : Array ByteArray) (p : Array ByteArray) (root : ByteArray) :
    Nat × ByteArray × Array ByteArray :=
  let leaf := leaves[i]?.getD .empty
  let first := p[0]?.getD .empty
  match name with
  | "flipped" =>
    let s := p[a]?.getD .empty
    (i, leaf, p.set! a (s.set! 0 (s.get! 0 ^^^ (1 <<< b).toUInt8)))
  | "short" => (i, leaf, p.pop)
  | "swapped" => (i, first, p.set! 0 leaf)
  | "odd-sibling" => (i, leaf, p.set! 0 (first.extract 0 63))
  | "long" => (i, leaf, p.push root)
  | "index" | "past" => (a, leaf, p)
  | _ => (i, leaf, p)

/-- The checks the Python ran, each recomputed: its name and arguments are the Python's, its verdict this one's. -/
def merkleChecks (py : Json) (leaves : Array ByteArray) (root : ByteArray) : V Json := do
  let paths := (Array.range leaves.size).map (Merkle.path leaves)
  let checks ← (← J.arr py "checks" "python").mapM fun c => do
    let #[name, i, a, b, _] := (← J.asArr c "check") | throw "merkle: a check is [name, leaf, a, b, verdict]"
    let (name, i, a, b) := (← J.asStr name "name", ← J.asNat i "leaf", ← J.asNat a "a", ← J.asNat b "b")
    let (j, lf, q) := checkInput name i a b leaves (paths[i]?.getD #[]) root
    pure (Json.arr #[.str name, num i, num a, num b, .bool (Merkle.verifyPath root leaves.size j lf q)])
  pure (.arr checks)

/-- Every answer of a `merkle` instance; `py` supplies only the checks' names and arguments. -/
def merkle (inst py : Json) : V Json := do
  let salt ← bytesOf (← J.get inst "salt" "merkle") "salt"
  let bps ← J.nat inst "blocks_per_segment" "merkle"
  let blocks ← bytesArr (← J.get inst "blocks" "merkle") "blocks"
  let leafOf ← match ← J.str inst "leaf" "merkle" with
    | "spec" => pure Merkle.leaf
    | "package" => pure Merkle.packageLeaf
    | k => throw s!"merkle: leaf {k}"
  let leaves := blocks.mapIdx fun i b => leafOf salt (i / bps) (i % bps) b
  let n := leaves.size
  let root ← match Merkle.root leaves with
    | some r => pure r
    | none => throw "merkle: no leaf"
  let paths := (Array.range n).map (Merkle.path leaves)
  let samples ← (← J.arr inst "samples" "merkle").mapM fun s => do
    let #[sj, vj, bj] := (← J.asArr s "sample") | throw "merkle: a sample is [s, v, block]"
    pure (hex (leafOf salt (← J.asNat sj "s") (← J.asNat vj "v") (← bytesOf bj "block")))
  pure <| Json.mkObj [
    ("leaves", .arr (leaves.map hex)), ("root", hex root),
    ("paths_sha512", hex (Sha512.hash (← canon (.arr (paths.map fun p => .arr (p.map hex)))).toUTF8)),
    ("checks", ← merkleChecks py leaves root),
    ("path_lengths", .arr ((Array.range n).map fun i => match Merkle.pathLength n i with
      | some h => num h
      | none => .null)),
    ("samples", .arr samples)]

def evaluate (inst py : Json) : V Json := do
  match ← J.str inst "kind" "instance" with
  | "merkle" => merkle inst py
  | k => throw s!"unknown instance kind {k}"

/-- The differences on one line: none, or why its answer is not the Python's. -/
def differences (i : Nat) (instLine pyLine : String) : List String :=
  match (do
    let inst ← Json.parse instLine
    let py ← Json.parse pyLine
    let out ← evaluate inst py
    let mine ← canon out
    pure (if mine == pyLine then none else some mine) : Except String (Option String)) with
  | .ok none => []
  | .ok (some mine) =>
    let pos := ((mine.toList.zip pyLine.toList).takeWhile fun (a, b) => a == b).length
    [s!"instance {i}: from byte {pos}, Lean {((mine.drop (pos - min pos 80)).take 240).toString}\n  Python {((pyLine.drop (pos - min pos 80)).take 240).toString}"]
  | .error e => [s!"instance {i}: {e}"]

def main (args : List String) : IO UInt32 := do
  let [insts, py, n] := args | IO.eprintln "usage: PousCheck/DifftestMain.lean INSTANCES PYTHON N"; return 2
  let lines (p : String) : IO (Array String) := do
    return ((← IO.FS.readFile p).splitOn "\n").filter (! ·.isEmpty) |>.toArray
  let a ← lines insts
  let b ← lines py
  if n.toNat? != some a.size ∨ a.size != b.size then
    IO.eprintln s!"difftest: {a.size} instances and {b.size} answers, not {n}"
    return 1
  let tasks := (a.zip b).toList.zipIdx.map fun ((x, y), i) => Task.spawn fun _ => differences i x y
  let found := tasks.flatMap Task.get
  for d in found.take 50 do IO.eprintln d
  IO.println s!"difftest: {a.size} instances compared, {found.length} differences"
  return if found.isEmpty then 0 else 1

end PousCheck.Difftest
