import Flock.Canon
import PousCheck.Merkle
import PousCheck.Challenge
import PousCheck.Wire
import PousCheck.Certificate
import PousCheck.Audit

/-!
# The difftest: the PoUS check against the Python verifier, on generated instances

`scripts/difftest_vectors.py` writes instances and the Python's answers on each, one canonical JSON line per instance in
two files. `evaluate` computes the answers from the instance with `PousCheck`'s definitions, and `main` fails unless
every line's canonical JSON is the Python's line, byte for byte. The Lean audit runs it (`runs` in `lean-audit.json`):
a test, not a proof.

* `merkle`: `Merkle.leaf` or `Merkle.packageLeaf` of every block, `Merkle.root`, the SHA-512 of every `Merkle.path`,
  `Merkle.pathLength` of every leaf, `Merkle.verifyPath` on each check the Python ran (`checkInput` rebuilds its input
  from its name and arguments), and the leaf of each sample block.
* `randomness`: `Randomness.derive` of a random source, domain and context, its streams and uniforms over random indices,
  and P2's audit id, mode-0 key, indices and nonces and mode-1 round indices (`PousCheck.Challenge`).
* `audit`: `Audit.check` on a session (frames, measured latencies, the verifier's secret, beacons, seen pairs, its own
  root of a private setup and its held `W`): its `FINAL` frame, the fault's code when there is none, or the refusal.
  An accepted `FINAL` also gives the session's `Certificate.assess`: its verdict, coverage and StorageProfile.
* `storage`: the certificate gate and StorageProfile (`PousCheck.Certificate`) at a store, `k` and `Δ + RTT`.
* `sha512`: the SHA-512 of labelled bytes, against the pins the spec's §7 states for them.

`transcript` is not a kind of the run: `pous-audit-difftest serve` answers it for `interop.py`, which holds it to
`wire.py` on the spec's transcripts.
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

/-- A part as the script writes it: `["b", hex]`, `["s", string]` or `["i", n]`. -/
def partOf (j : Json) : V Randomness.Part := do
  match ← J.asArr j "part" with
  | #[.str "b", b] => pure (.bytes (← bytesOf b "part"))
  | #[.str "s", s] => pure (.str (← J.asStr s "part"))
  | #[.str "i", n] => pure (.int (← J.asNat n "part"))
  | _ => throw "randomness: a part is [b|s|i, value]"

def indexOf (j : Json) : V (List Randomness.Part) := do pure (← (← J.asArr j "index").mapM partOf).toList

/-- A `randomness` instance's answers: the derived key, its streams and its uniforms, and P2's mode-0 audit id, key,
indices and nonces and mode-1 round indices (`Challenge.auditId`, `Challenge.auditKey`, `Challenge.roundKey`). -/
def randomness (inst : Json) : V Json := do
  let source ← bytesOf (← J.get inst "source" "randomness") "source"
  let context ← (← J.arr inst "context" "randomness").mapM fun kv => do
    let #[k, v] := (← J.asArr kv "context") | throw "randomness: a context pair is [key, part]"
    pure (← J.asStr k "key", ← partOf v)
  let key := Randomness.derive source (← J.str inst "domain" "randomness") context.toList
  let streams ← (← J.arr inst "streams" "randomness").mapM fun s => do
    let #[i, n] := (← J.asArr s "stream") | throw "randomness: a stream is [index, n]"
    pure (hex (Randomness.stream key (← indexOf i) (← J.asNat n "n")))
  let uniforms ← (← J.arr inst "uniforms" "randomness").mapM fun u => do
    let #[n, i] := (← J.asArr u "uniform") | throw "randomness: a uniform is [n, index]"
    match Randomness.uniform key (← J.asNat n "n") (← indexOf i) with
    | some x => pure (num x)
    | none => throw "randomness: uniform ran out"
  let p2 ← J.get inst "p2" "randomness"
  let b name := do bytesOf (← J.get p2 name "p2") name
  let (src, epoch, root, salt) := (← b "source", ← b "epoch", ← b "root", ← b "salt")
  let (counter, blocks) := (← J.nat p2 "counter" "p2", ← J.nat p2 "blocks" "p2")
  let aid := Challenge.auditId src epoch counter
  let akey := Challenge.auditKey src aid epoch root salt
  let rounds := List.range (← J.nat p2 "rounds" "p2")
  let beacons ← bytesArr (← J.get p2 "beacons" "p2") "beacons"
  pure <| Json.mkObj [
    ("key", hex key), ("streams", .arr streams), ("uniforms", .arr uniforms),
    ("audit_id", hex aid), ("audit_key", hex akey),
    ("indices", .arr (rounds.map fun j => match Challenge.index akey blocks j with
      | some x => num x
      | none => .null).toArray),
    ("nonces", .arr (rounds.map fun j => hex (Challenge.nonce akey j)).toArray),
    ("round_indices", .arr (beacons.mapIdx fun j r => match Challenge.index (Challenge.roundKey r aid epoch root j) blocks j with
      | some x => num x
      | none => .null))]

/-- A JSON field that may be `null` or absent. -/
def optional {α : Type} (j : Json) (k : String) (f : Json → V α) : V (Option α) :=
  match j.getObjValD k with
  | .null => pure none
  | v => some <$> f v

/-- A session's record: its frames, measured latencies, and the verifier's secret, beacons, every pair it has used (a
list, required), and the root it built from `W` and the `W` it holds, when it has them. -/
def session (inst : Json) : V (Array ByteArray × Array Nat × Wire.Verifier) := do
  let frames ← bytesArr (← J.get inst "frames" "audit") "frames"
  let elapsed ← (← J.arr inst "elapsed_ns" "audit").mapM (J.asNat · "elapsed_ns")
  let source ← optional inst "verifier_source" (bytesOf · "verifier_source")
  let beacons ← optional inst "beacons" fun j => do (← J.asArr j "beacons").mapM fun b => do
    let #[r, v] := (← J.asArr b "beacon") | throw "audit: a beacon is [round, value]"
    pure (← J.asNat r "round", ← bytesOf v "beacon")
  let seen ← (← J.arr inst "seen" "audit").mapM fun p => do
    let #[e, a] := (← J.asArr p "pair") | throw "audit: a seen pair is [epoch, audit id]"
    pure (← bytesOf e "epoch", ← bytesOf a "audit id")
  let privateRoot ← optional inst "private_root" (bytesOf · "private_root")
  let wBytes ← optional inst "w_bytes" (bytesOf · "w_bytes")
  pure (frames, elapsed, ⟨source, beacons, seen, privateRoot, wBytes⟩)

/-- An `audit` instance's answer: `Audit.check`'s `FINAL` frame and assessment, the code of the fault that leaves no
`FINAL`, or the verifier's refusal of its inputs. -/
def audit (inst : Json) : V Json := do
  let (frames, elapsed, v) ← session inst
  pure <| match Audit.check frames elapsed v with
    | .ok (final, storage) => Json.mkObj [("final", hex final), ("storage", storage)]
    | .error (.fault e) => Json.mkObj [("error", num e.code)]
    | .error (.refused r) => Json.mkObj [("refused", .str r)]

def fault (e : Wire.Fault) : Json := Json.mkObj [("fault", num e.code), ("message", .str e.message)]

def stop : Wire.Stop → Json
  | .fault e => fault e
  | .refused r => Json.mkObj [("refused", .str r)]

/-- A transcript's outputs for `interop.py`, as the wire's two implementations state them: the setup's (`Wire.parseSetup`,
`Wire.setupFailure`: the epoch id, the declared root and the identity it binds, how many blocks it carried, its failure
code, and where the audit begins) and the audit's (`Audit.check`: the `FINAL` frame, its verdict, code, record count and
transcript hash), or the fault or refusal each stops at. -/
def transcript (inst : Json) : V Json := do
  let (frames, elapsed, v) ← session inst
  let setup := match Wire.parseSetup frames with
    | .ok (s, cursor) => match Wire.judgeable s v with
      | .ok () => Json.mkObj [("blocks", num s.blocks.size), ("commitment", hex s.commitment), ("cursor", num cursor),
          ("epoch_id", hex s.epoch), ("failure", num (Wire.setupFailure s v.privateRoot)), ("manifest", hex s.manifest),
          ("root", hex s.root), ("salt", hex s.salt)]
      | .error r => stop r
    | .error e => fault e
  -- The transcript hash follows the envelope's 10 bytes, the epoch id, the audit id, verdict, code, round count, Δ and RTT.
  let o := 10 + Wire.D + Wire.S + 1 + 1 + 2 + 8 + 8
  let audit := match Audit.check frames elapsed v with
    | .ok (final, _) => match Wire.parseFinal final with
      | .ok (verdict, code, records) => Json.mkObj [("code", num code), ("final", hex final), ("records", num records),
          ("transcript_hash", hex (final.extract o (o + Wire.D))), ("verdict", .bool verdict)]
      | .error e => fault e
    | .error e => stop e
  pure (Json.mkObj [("audit", audit), ("setup", setup)])

/-- A `storage` instance's answer: the certificate gate and the StorageProfile of an accepted audit of `k` challenges
within `Δ + RTT` of a store (`Certificate.gate`, `profileJson`), or `null` when `P2Params` refuses the store. -/
def storage (inst : Json) : V Json := do
  let n k := J.nat inst k "storage"
  let q : Wire.Params := ⟨← n "w", ← n "c0", ← n "blocks_per_segment", ← n "segments", "", 0⟩
  let (k, delta, rtt) := (← n "k", ← n "delta", ← n "rtt")
  pure <| match Certificate.Store.ofParams q with
    | none => .null
    | some s => Json.mkObj [("covered", .bool (Certificate.gate s k delta rtt).2.1.isEmpty),
      ("profile", Certificate.profileJson s k delta rtt)]

/-- A `sha512` instance's answer: the SHA-512 of each item's bytes, by its label (the spec's §7 pins). -/
def sha512 (inst : Json) : V Json := do
  let items ← (← J.arr inst "items" "sha512").mapM fun i => do
    let #[label, b] := (← J.asArr i "item") | throw "sha512: an item is [label, bytes]"
    pure (← J.asStr label "label", hex (Sha512.hash (← bytesOf b "bytes")))
  pure (Json.mkObj items.toList)

def evaluate (inst py : Json) : V Json := do
  match ← J.str inst "kind" "instance" with
  | "merkle" => merkle inst py
  | "randomness" => randomness inst
  | "audit" => audit inst
  | "sha512" => sha512 inst
  | "storage" => storage inst
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
