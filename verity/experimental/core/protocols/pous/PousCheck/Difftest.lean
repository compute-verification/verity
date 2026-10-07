import PousCheck.Setup

/-!
# The setup difftest: `PousCheck` against the Python reference, on generated instances and the spec's vectors

`scripts/setup_difftest.py` writes instances and the answers of `verity.experimental.protocols.pous` (`schemes/p2.py`,
`setup.py`, `protocol.py`) on each; `PousCheck/DifftestMain.lean` reads both and fails unless `answer`, which computes
the same from this package's definitions, gives every one of them. The kinds:

* `maps`: `P2.d`, `P2.sigma` and `P2.decodeValue` on canonical values (Python's `_Field.sq`, `sigma`);
* `keys`: `Keys.packagePrefix` and `Expander.keys` (`expand_prefix`, `sha512_keys`): K, t and which candidate t is, or
  the salt's refusal;
* `decode`: `Setup.decodeSegment` (`decode_segment`): the payload, or the kind of refusal;
* `commit`: `Commit.commit` and `Commit.manifest` (`setup.commit`, `canonical_json_bytes(params.manifest())`); `tagged`:
  `Commit.taggedSha512` (`identity.tagged_sha512`), and `Prim.Sha512` absorbing the payload in the given pieces against
  `hashlib.sha512`;
* `setup`: `Setup.checkEncoding` (`protocol.verifier_key(…, mode=PUBLIC)`): accepted, or the kind of refusal. Python's
  answer also carries `vk`'s root, which is compared once `PousCheck.Merkle` supplies `Setup.Tree` (`ROOT_PENDING`).

`vectorProblems` checks the spec's §7 file (`benchmarks/pous/p2_v1/vectors.json`, schema `p2-test-vectors/v4`) against
the SHA-512 §7 pins, each entry's own `sha512` (of its canonical JSON without it), and each vector's K, t, its rejection
index, c and m under the spec's prefix (`Keys.specPrefix`), and the σ rejection cases.

The audit runs it (`runs` in `lean-audit.json`): compiled evaluation, a test and not a proof.
-/

namespace PousCheck.Difftest

open Lean PousCheck

abbrev V := Except String

def str (j : Json) (k : String) : V String := Flock.J.str j k "instance"
def nat (j : Json) (k : String) : V Nat := Flock.J.nat j k "instance"
def bytes (j : Json) (k : String) : V ByteArray := do Flock.unhex (← str j k)
def natHex (j : Json) (k : String) : V Nat := do return Prim.natOfBE (← bytes j k)
def hexW (x n : Nat) : Json := .str (Flock.hexOf (Prim.beOfNat x n))
def hex (b : ByteArray) : Json := .str (Flock.hexOf b)
def err (kind : String) : Json := Json.mkObj [("error", .str kind)]

def params (j : Json) : V P2.Params := do
  let p ← Flock.J.get j "params" "instance"
  P2.Params.validate ⟨← nat p "w", ← nat p "c0", ← nat p "blocks_per_segment", ← nat p "segments"⟩

def blockList (j : Json) : V (Array ByteArray) := do (← Flock.J.arr j "blocks" "instance").mapM fun b => do
  Flock.unhex (← Flock.J.asStr b "block")

/-- Python's answer keys that Lean does not compute yet: `vk`'s root, until `PousCheck.Merkle` supplies the tree. -/
def ROOT_PENDING : List String := ["root"]

/-- This package's answer to one instance, in the form of Python's line. -/
def answer (i : Json) : V Json := do
  match ← str i "kind" with
  | "maps" =>
    let q : P2.Params := ⟨← nat i "w", ← nat i "c0", 1, 1⟩
    let (p, n) := (q.p, q.blockBytes)
    let (z, r, k, t) := (← natHex i "z", ← natHex i "r", ← natHex i "k", ← natHex i "t")
    if z ≥ p || r ≥ p || t ≥ p then throw "maps: a value is not canonical"
    return Json.mkObj [("d", hexW (P2.d p z) n), ("sigma", hexW (P2.sigma p r k) n),
      ("decode", hexW (P2.decodeValue p k t z) n)]
  | "keys" =>
    let q ← P2.Params.validate ⟨← nat i "w", ← nat i "c0", 2 ^ 16, 2 ^ 20⟩
    match Keys.packagePrefix q (← bytes i "salt") with
    | .error _ => return err "salt"
    | .ok pre =>
      let (k, t, a) ← (Keys.Expander.ofPrefix pre).keys q (← nat i "s") (← nat i "j")
      return Json.mkObj [("K", hexW k q.blockBytes), ("t", hexW t q.blockBytes), ("t_index", .num a)]
  | "decode" =>
    match Setup.decodeSegment (← params i) (← bytes i "salt") (← nat i "s") (← blockList i) with
    | .ok payload => return Json.mkObj [("payload", hex payload)]
    | .error r => return err r.kind
  | "commit" =>
    let q ← params i
    return Json.mkObj [("commitment", hex (← Commit.commit q (← bytes i "w"))),
      ("manifest", .str (String.fromUTF8! (← Commit.manifest q)))]
  | "tagged" =>
    let payload ← bytes i "payload"
    let pieces ← (← Flock.J.arr i "pieces" "instance").mapM (Flock.J.asNat · "piece")
    let (st, off) := pieces.foldl (init := (Prim.Sha512.init, 0)) fun (st, off) n =>
      (st.update (payload.extract off (off + n)), off + n)
    if off != payload.size then throw "tagged: the pieces are not the payload"
    if Flock.Sha512.hash payload != st.finalize then throw "tagged: Prim.Sha512 is not Flock.Sha512.hash"
    return Json.mkObj [("digest", hex (Commit.taggedSha512 (← str i "tag") payload)), ("sha512", hex st.finalize)]
  | "setup" =>
    let encoded ← (← Flock.J.arr i "encoded" "instance").mapM fun seg => do
      (← Flock.J.asArr seg "segment").mapM fun b => do Flock.unhex (← Flock.J.asStr b "block")
    match Setup.checkEncoding (← params i) (← bytes i "commitment") (← bytes i "salt") encoded with
    | .ok () => return Json.mkObj [("verdict", .str "ok")]
    | .error r => return err r.kind
  | k => throw s!"unknown kind {k}"

def without (keys : List String) : Json → Json
  | .obj kvs => .obj (keys.foldl (fun m k => m.erase k) kvs)
  | j => j

/-- The differences on instance `n`, empty when Lean gives Python's answer. -/
def differences (n : Nat) (inst py : String) : List String :=
  match (do
    let i ← Json.parse inst
    let want ← Flock.canon (without ROOT_PENDING (← Json.parse py))
    let got ← Flock.canon (← answer i)
    pure (← str i "kind", got, want) : V (String × String × String)) with
  | .error e => [s!"instance {n}: {e}"]
  | .ok (kind, got, want) => if got == want then [] else [s!"instance {n} ({kind}): Lean {got.take 300}, Python {want.take 300}"]

/-- §7's SHA-512 of `vectors.json`. -/
def VECTORS_SHA512 : String :=
  "959911fd371111fe20c4cee6fc532ff205bbf377962c73ac539a4d2de9b22121287e6e996d9ae4bfbef9f26fcd67ac78620873702c8b2899bc41a21bdd76e853"

/-- §7's entry hashes, by width. -/
def ENTRY_SHA512 : List (Nat × String) := [
  (64, "c34350fa0e0b8ae7c521cd7df833c160df1498adc938a99feb6166c2f37b3e6a2ca07239b9ab71fdd979b02efc8c8951d68f32d108abbde5cc44bac3137241e1"),
  (96, "df8941e52d73848baa7862cd3f20ea6f7adc1da79149db154b0be9e0d28be8905b95aac079cd98e14b56f8077ba6cfb173e0472df7e83b40138c6f16a871ff1f"),
  (128, "adf7cd300e2d4ea93643e94d563b02c80e29d88d65a34a55269f9e37da6b71bde6dce1c9e4c04c561dd103a8dc4c3955ee83ecabef20ad717de4672e8014fcae"),
  (16448, "2a227a512683d67624bca51ba087ec05a4dda85c5e93b67c50c03349a7823f50f84b98f4f2f9e92d9c87ddc41ab0e5fd8d2211472709a05093e7c0121a86811c")]

/-- One §7 vector: its entry hash, keys, codeword and payload. -/
def vectorProblem (v : Json) : V (Option String) := do
  let id ← str v "id"
  let q : P2.Params := ⟨← nat v "w", ← nat v "c0", 2 ^ 16, 8⟩
  let entry := Flock.hexOf (Prim.Sha512.hash (← Flock.canon (without ["sha512"] v)).toUTF8)
  if entry != (← str v "sha512") then return some s!"{id}: the entry's sha512 is not its canonical JSON's"
  if ENTRY_SHA512.lookup q.w != some entry then return some s!"{id}: the entry's sha512 is not §7's"
  let e := Keys.Expander.ofPrefix (Keys.specPrefix q (← bytes v "model_commitment") (← bytes v "setup_salt"))
  let (k, t, a) ← e.keys q (← nat v "segment") (← nat v "block")
  let b := if q.w == W_BITS then 16432 else 8 * (q.blockBytes - 1)
  let (c, m) := (← natHex v "c", ← natHex v "m")
  let checks := [("K", k == (← natHex v "K")), ("t", t == (← natHex v "t")), ("t_rejection_index", a == (← nat v "t_rejection_index")),
    ("c < p", c < q.p), ("m < 2^b", m < 2 ^ b), ("D(c) = m", P2.decodeValue q.p k t c == m),
    ("decoded_m", m == (← natHex v "decoded_m"))]
  return (checks.find? (! ·.2)).map fun (what, _) => s!"{id}: {what}"

/-- One σ rejection case: `r ⊕ K ≥ p`, so `σ_K(r) = r`. -/
def sigmaProblem (v : Json) : V (Option String) := do
  let q : P2.Params := ⟨← nat v "w", ← nat v "c0", 1, 1⟩
  let (r, k) := (← natHex v "r", ← natHex v "K")
  let ok := (← natHex v "p") == q.p && (r ^^^ k) == (← natHex v "r_xor_K") && r ^^^ k ≥ q.p
    && P2.sigma q.p r k == (← natHex v "sigma") && (← str v "branch") == "reject-xor-and-return-r"
  let id ← str v "id"
  return if ok then none else some s!"{id}: σ's rejection branch"

/-- The problems with the spec's §7 file, and how many vectors and σ cases it held. -/
def vectorProblems (raw : ByteArray) : V (List String × Nat × Nat) := do
  let mut out : List String := []
  if Flock.hexOf (Prim.Sha512.hash raw) != VECTORS_SHA512 then out := out ++ ["vectors.json: not §7's SHA-512"]
  let j ← Json.parse (String.fromUTF8! raw)
  for (k, want) in [("schema", "p2-test-vectors/v4"), ("protocol", "p2/v4"), ("randomness", "verity/randomness/derive/v2")] do
    if (← str j k) != want then out := out ++ [s!"vectors.json: {k} is not {want}"]
  let vs ← Flock.J.arr j "vectors" "vectors.json"
  let ss ← Flock.J.arr j "sigma_rejection_vectors" "vectors.json"
  for v in vs do if let some p ← vectorProblem v then out := out ++ [p]
  for v in ss do if let some p ← sigmaProblem v then out := out ++ [p]
  return (out, vs.size, ss.size)

def main (args : List String) : IO UInt32 := do
  let [insts, py, vectors, n] := args
    | IO.eprintln "usage: PousCheck/DifftestMain.lean INSTANCES PYTHON VECTORS N"; return 2
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
  match vectorProblems (← IO.FS.readBinFile vectors) with
  | .error e => IO.eprintln s!"vectors.json: {e}"; return 1
  | .ok (problems, nv, ns) =>
    for p in problems do IO.eprintln p
    IO.println s!"vectors.json: {nv} vectors and {ns} σ rejection cases checked, {problems.length} problems"
    return if found.isEmpty && problems.isEmpty then 0 else 1

end PousCheck.Difftest
