/-
# The value commitments' salts, the ZK server's (live PROTOCOL.md §10.2)

Every hm96 row leaf whose `b ‖ c` or root goes public is the ZK server's. The serving stack hands the server each row's
unsalted 64-byte digest `x`, and the server derives the leaf's 192-byte salt from its own 64-byte key `k`, the commit
session `s` it draws once every digest of the session is in, the table `j` and the leaf index `i` (`saltOf`):

    y = SHA-512(k ‖ L ‖ s ‖ be32 j ‖ be64 i ‖ 0) ‖ SHA-512(… ‖ 1) ‖ SHA-512(… ‖ 2),   L = SHA-512("verity/zk-server/value-salt/v1\0")

It answers each leaf's `b ‖ c` (`Hm96.Default512.commitString`), the frame-v3-sha512 root over the leaves (the verifier's
own `FV3.Sha512.leafAs` and `FV3.root`), and a tag per table (`tagOf`): a MAC under `k` over the table's session, index,
domain, leaf count, schema and root, by which the server knows a table it committed without keeping a record of it.
Every salt message is 205 bytes and every tag message 333, under different constants, so no two share an input.

* `parseCommit`, `commitAnswer`: a commit request's tables, and their leaves, roots and tags at the session `s`.
* `parseRefs`, `saltsAnswer`: the salts of the rows a refs file names, every table ref's tag checked (`TableRef.valid`).
* `checkStatement`: every hm96 tree of a statement's public file is a table the server committed. The session loop
  serves no statement whose value commitments it did not make (`Session.run`, `--salt-key`, `--commitments`).
* `loadKey`: the key file.

No request takes a salt from the stack: a commit request's body is its digests and nothing else (`commitBody`). A
statement whose leaves were made under salts of the stack's has roots no tag of the server's names, so the loop refuses
it. A prover's witness salts must then be the server's for its proof to pass, since each leaf's `c` is the SHA-512 of
its salt.
-/
import Flock.Hm96
import Flock.FV3
import Flock.Registered
import Flock.Comp
import Flock.JsonParse

namespace Flock.Firewall.Salts

open Lean (Json)

def KEY_BYTES : Nat := 64
def SESSION_BYTES : Nat := 64
def DIGEST_BYTES : Nat := 64
def BC_BYTES : Nat := 128
def SALT_BYTES : Nat := HmRow.SALT_BYTES

/-- A table ref's bytes: `s ‖ be32 j ‖ D ‖ be64 N ‖ u8 schema ‖ ROOT ‖ TAG`, `schema` the index of the table's row schema
in `Registered.SCHEMAS`. -/
def TABLE_REF_BYTES : Nat := 64 + 4 + 64 + 8 + 1 + 64 + 64
/-- A row ref's bytes: `be32 t ‖ be64 i`, leaf `i` of table ref `t`. -/
def ROW_REF_BYTES : Nat := 12

private def label (s : String) : ByteArray := Sha512.hash (s.toUTF8.push 0)

def SALT_LABEL : ByteArray := label "verity/zk-server/value-salt/v1"
def TAG_LABEL : ByteArray := label "verity/zk-server/value-commitment/v1"

/-- **The salt of leaf `i` of table `j` of the commit session `s`** under the key `k`: three SHA-512 blocks in counter mode,
each `Hash.Keyed`'s keyed digest `H(k ‖ L ‖ top)` with `top = s ‖ be32 j ‖ be64 i ‖ ctr`. -/
def saltOf (k s : ByteArray) (j i : Nat) : ByteArray :=
  let pre := k ++ SALT_LABEL ++ s ++ be32 j.toUInt32 ++ be64 i.toUInt64
  Sha512.hash (pre.push 0) ++ Sha512.hash (pre.push 1) ++ Sha512.hash (pre.push 2)

/-- A committed table: its frame-v3-sha512 domain id, its leaf count and its row schema's index in `Registered.SCHEMAS`. -/
structure Table where
  dom : ByteArray
  n : Nat
  schema : Nat
  deriving Inhabited

def Table.schemaName (t : Table) : String := Registered.SCHEMAS.getD t.schema ""

/-- **The tag of table `j` of the commit session `s`**, of root `root`, under the key `k`. -/
def tagOf (k s : ByteArray) (j : Nat) (t : Table) (root : ByteArray) : ByteArray :=
  Sha512.hash (k ++ TAG_LABEL ++ s ++ be32 j.toUInt32 ++ t.dom ++ be64 t.n.toUInt64 ++ ByteArray.mk #[t.schema.toUInt8] ++ root)

/-- Whether `a` and `b` are equal, every byte compared whatever the first difference. -/
def ctEq (a b : ByteArray) : Bool :=
  a.size == b.size && (List.range a.size).foldl (fun acc i => acc ||| (a.get! i ^^^ b.get! i)) (0 : UInt8) == 0

/-- The big-endian number of the `len` bytes of `b` at `o`. -/
def beAt (b : ByteArray) (o len : Nat) : Nat := (List.range len).foldl (fun acc i => 256 * acc + (b.get! (o + i)).toNat) 0

/-! ### Commit -/

/-- **A commit request's tables**, from its line's words after `commit`: `J`, then `D N S` for each of the `J` tables, `D`
the table's frame-v3-sha512 domain id (128 lowercase hex digits), `N` its leaf count (at least one) and `S` its row
schema, one of `Registered.SCHEMAS`. -/
def parseCommit (ws : List String) : Except String (Array Table) := do
  let j :: rest := ws | throw "a commit request with no table count"
  let some J := j.toNat? | throw s!"a table count `{j}` that is not a number"
  if J == 0 || J ≥ 2 ^ 32 then throw s!"a commit request of {J} tables, where it takes 1 to 2^32 − 1"
  go J rest #[]
where
  go : Nat → List String → Array Table → Except String (Array Table)
    | 0, [], acc => pure acc
    | 0, _ :: _, _ => throw "a commit request with more words than its tables take"
    | k + 1, d :: n :: s :: rest, acc => do
      if d.length != 2 * DIGEST_BYTES then throw s!"a domain id `{d}` that is not {2 * DIGEST_BYTES} hex digits"
      let dom ← unhex d
      let some n := n.toNat? | throw s!"a leaf count `{n}` that is not a number"
      if n == 0 || n ≥ 2 ^ 64 then throw s!"a table of {n} leaves, where it takes 1 to 2^64 − 1"
      let some schema := Registered.SCHEMAS.findIdx? (· == s) | throw s!"a row schema `{s}`, none of {Registered.SCHEMAS}"
      go k rest (acc.push { dom, n, schema })
    | _ + 1, _, _ => throw "a commit request with fewer words than its tables take"

/-- The bytes a commit request's body is, after its line: every table's digests, 64 bytes a leaf, and nothing else. -/
def commitBody (ts : Array Table) : Nat := DIGEST_BYTES * ts.foldl (· + ·.n) 0

/-- Table `j` of the commit session `s` under the key `k`, its digests from byte `o` of `xs`: its root, its tag and its
leaves' `b ‖ c` in leaf order. -/
def commitTable (k s : ByteArray) (j : Nat) (t : Table) (xs : ByteArray) (o : Nat) : ByteArray × ByteArray × ByteArray :=
  let bcs := (Array.range t.n).map fun i =>
    Hm96.Default512.commitString (xs.extract (o + DIGEST_BYTES * i) (o + DIGEST_BYTES * (i + 1))) (saltOf k s j i)
  let leaves := (Array.range t.n).map fun i => FV3.Sha512.leafAs t.schemaName t.dom i (Hm96.Default512.treeLeaf bcs[i]!)
  let root := FV3.root t.dom leaves Sha512.hash
  (root, tagOf k s j t root, bcs.foldl (· ++ ·) .empty)

/-- **A commit session's tables** `ts` at the session `s`, their digests `xs` table-major: each table's root, tag and
leaves' `b ‖ c`, in table order. -/
def commitAnswer (k s : ByteArray) (ts : Array Table) (xs : ByteArray) : Array (ByteArray × ByteArray × ByteArray) :=
  let offs := (ts.foldl (fun (acc : Array Nat × Nat) t => (acc.1.push acc.2, acc.2 + DIGEST_BYTES * t.n)) (#[], 0)).1
  (Array.range ts.size).map fun j => commitTable k s j ts[j]! xs offs[j]!

/-! ### Refs and salts -/

/-- A table ref: a table the server committed, as its commit answered it. -/
structure TableRef where
  s : ByteArray
  j : Nat
  t : Table
  root : ByteArray
  tag : ByteArray
  deriving Inhabited

/-- The table ref at byte `o` of `b` (`TABLE_REF_BYTES` of them there). -/
def TableRef.parse (b : ByteArray) (o : Nat) : Except String TableRef := do
  let schema := (b.get! (o + 140)).toNat
  if schema ≥ Registered.SCHEMAS.length then throw s!"a table ref of schema {schema}, none of {Registered.SCHEMAS}'s"
  let n := beAt b (o + 132) 8
  if n == 0 then throw "a table ref of no leaf"
  return { s := b.extract o (o + 64), j := beAt b (o + 64) 4, t := { dom := b.extract (o + 68) (o + 132), n, schema },
           root := b.extract (o + 141) (o + 205), tag := b.extract (o + 205) (o + 269) }

/-- Whether `r`'s tag is the server's for its session, table, domain, leaf count, schema and root, under the key `k`. -/
def TableRef.valid (k : ByteArray) (r : TableRef) : Bool := ctEq r.tag (tagOf k r.s r.j r.t r.root)

/-- **A refs file**: `be32 T ‖ be32 R`, then `T` table refs, then `R` row refs, each row ref's table `t < T` and its leaf
below that table's leaf count. -/
def parseRefs (b : ByteArray) : Except String (Array TableRef × Array (Nat × Nat)) := do
  if b.size < 8 then throw s!"a refs file of {b.size} bytes, shorter than its counts"
  let T := beAt b 0 4
  let R := beAt b 4 4
  let want := 8 + TABLE_REF_BYTES * T + ROW_REF_BYTES * R
  if b.size != want then throw s!"a refs file of {b.size} bytes, where {T} table refs and {R} row refs take {want}"
  let ts ← (Array.range T).mapM fun t => TableRef.parse b (8 + TABLE_REF_BYTES * t)
  let o := 8 + TABLE_REF_BYTES * T
  let rs := (Array.range R).map fun r => (beAt b (o + ROW_REF_BYTES * r) 4, beAt b (o + ROW_REF_BYTES * r + 4) 8)
  let past := fun ((t, i) : Nat × Nat) => match ts[t]? with
    | some r => i ≥ r.t.n
    | none => true
  if let some (t, i) := rs.find? past then throw s!"a row ref to leaf {i} of table ref {t}, past it"
  return (ts, rs)

/-- **The salts of the rows a refs file names**, in its row refs' order, under the key `k`; refused whole where any table
ref's tag is not the server's. -/
def saltsAnswer (k refs : ByteArray) : Except String ByteArray := do
  let (ts, rs) ← parseRefs refs
  if let some t := (List.range ts.size).find? fun t => !(ts[t]!.valid k) then
    throw s!"table ref {t}: its tag is not the ZK server's for its session, table, domain, leaf count, schema and root"
  return rs.foldl (fun acc (t, i) => let r := ts[t]!; acc ++ saltOf k r.s r.j i) .empty

/-! ### A statement's value commitments -/

/-- A statement's hm96 trees, from its public file's header: for each root of `frame_v3` whose schema is one of
`Registered.SCHEMAS`, its port's name, its table (domain id, leaf count: its `shared_rows` table's, else `instances`)
and its root. -/
def treesOf (header : Json) : Except String (Array (String × Table × ByteArray)) := do
  let ctx := "public file.frame_v3"
  let fv ← J.get header "frame_v3" "public file"
  let .obj kvs ← J.get fv "roots" ctx | throw s!"{ctx}.roots: not an object"
  let schemas ← J.get fv "schemas" ctx
  let doms ← J.get fv "domain_ids" ctx
  let n ← J.nat header "instances" "public file"
  let shared := (header.getObjVal? "shared_rows").toOption
  let mut out := #[]
  for (name, r) in kvs.toArray do
    let .ok sj := schemas.getObjVal? name | continue
    let .ok sname := sj.getStr? | continue
    let some schema := Registered.SCHEMAS.findIdx? (· == sname) | continue
    let root ← unhex (← J.asStr r s!"{ctx}.roots.{name}")
    let dom ← unhex (← J.str doms name s!"{ctx}.domain_ids")
    let count ← match shared with
      | some s => J.nat s name "public file.shared_rows"
      | none => pure n
    out := out.push (name, { dom, n := count, schema }, root)
  return out

/-- **Whether a statement's value commitments are the server's**: every hm96 tree of its public file `publicFile`
(`treesOf`) is a table one of the refs file `refs`'s table refs names, at its domain, leaf count, schema and root, with
the server's tag under the key `k`. -/
def checkStatement (k refs publicFile : ByteArray) : Except String Unit := do
  let (rs, _) ← parseRefs refs
  let some nl := findByte publicFile 10 | throw "public file: no header line"
  let some text := String.fromUTF8? (publicFile.extract 0 nl) | throw "public file: its header is not UTF-8"
  let trees ← treesOf (← JsonParse.parse text)
  for (name, t, root) in trees do
    unless rs.any fun r => r.t.dom == t.dom && r.t.n == t.n && r.t.schema == t.schema && r.root == root && r.valid k do
      throw s!"row {name}: its root is no value commitment of the ZK server's (no commitment of its domain, leaf count, schema and root carries the server's tag)"

/-! ### The key -/

/-- **The salt key in `path`**, exactly its 64 bytes. With `create`, where `path` does not exist, 64 bytes drawn from the
operating system, in a file created there that its owner alone may read and write, set so before the key is written. -/
def loadKey (path : System.FilePath) (create : Bool) : IO (Except String ByteArray) := do
  if (← path.pathExists) then
    let b ← IO.FS.readBinFile path
    if b.size != KEY_BYTES then return .error s!"--salt-key {path}: {b.size} bytes, where a salt key is {KEY_BYTES}"
    return .ok b
  if !create then return .error s!"--salt-key {path}: no such file (`flock-firewall values` creates it)"
  let k ← IO.getRandomBytes KEY_BYTES.toUSize
  let h ← IO.FS.Handle.mk path .writeNew
  IO.setAccessRights path { user := { read := true, write := true } }
  h.write k
  h.flush
  return .ok k

end Flock.Firewall.Salts
