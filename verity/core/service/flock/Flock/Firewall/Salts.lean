/-
# The value commitments' salts, the ZK server's (live PROTOCOL.md §10.2)

Every hm96 row leaf whose `b ‖ c` or root goes public is the ZK server's. The serving stack hands the server each row's
unsalted 64-byte digest `x`, and the server derives the leaf's 192-byte salt from its own 64-byte key `k`, the commit
session `s` it samples once every digest of the session is in, the table `j` and the leaf index `i` (`saltOf`):

    y = SHA-512(k ‖ L ‖ s ‖ be32 j ‖ be64 i ‖ 0) ‖ SHA-512(… ‖ 1) ‖ SHA-512(… ‖ 2),   L = SHA-512("verity/zk-server/value-salt/v1\0")

It answers each leaf's `b ‖ c` (`Hm96.Default512.commitString`), the frame-v3-sha512 root over the leaves (the verifier's
own `FV3.Sha512.leafAs` and `FV3.root`), and a tag per table (`tagOf`): a MAC under `k` over the table's session, index,
domain, leaf count, schema and root, by which the server knows a table it committed without keeping a record of it.
Every salt message is 205 bytes and every tag message 333, under different constants, so no two share an input.

* `parseCommit`, `commitAnswer`: a commit request's tables, and their leaves, roots and tags at the session `s`.
* `interiorAnswer`, `Held`, `keyTagOf`: an interior commit's (below).
* `parseRefs`, `saltsAnswer`: the salts of the rows a refs file names, every table ref's tag checked (`TableRef.valid`).
* `Epoch`, `recordOf`: the commit sessions the service opened since its last registration, and a registration's
  record: its number `n` on the ZK server's record, the number `m` of sessions its epoch opened, the registered
  session's table refs and a record tag under `k` over all three (`recordTagOf`; an interior session's under its own
  key). A registration closes the epoch, so the stack registers one session of the `m` it had and no other; the
  service of record opens one an epoch, so `m` is 1 (`Values.answer`). The number is the record's: registrations are
  numbered 0, 1, 2, … in the order the ZK server's record holds them, with no gap (`core-firewall values`).
* `treesOfFile`, `readsOf`: a statement's hm96 trees, and the registrations they read: for each tree, the first
  registration on the record that holds a table at its domain, leaf count, schema and root, its record tag checked
  (`RegistrationRecord.valid`). The release side admits each registration once, oldest first, at the first statement
  that reads it, and a statement none of whose trees a registration of the server's holds is refused
  (`Core.Firewall.admitReads`, `core-firewall session --salt-key`).
* `loadKey`: the key file.

**Interior commits.** A two-stage audit's interior registration holds values that may follow the auditor's first-stage
sample (`Values.answer`'s `interior`). Nothing the service answers of it reads `k`. Once the request is whole the
service draws from the operating system the session `s`, each leaf's 192-byte salt and the session's own 64-byte key
`ks`, and holds the salts and `ks` by session while it runs (`Held`). The tables' tags and the registration's record
tag are under `ks`. At `register S` the service writes `ks` on the registration's line of the ZK server's record, with
`ks`'s tag under `k` over `ks` and the record (`keyTagOf`). The release side checks the record tag under `ks` only where
that tag is the server's (`RegistrationRecord.valid`), so a writer of the record without `k` adds no interior
registration it admits, and a record with no key is checked under `k`. `ks` and its tag are witness: they stay on the
ZK server's record, which only the ZK server's own user reads, and are never released.

No request takes a salt from the stack: a commit request's body is its digests and nothing else (`commitBody`). A
statement whose leaves were made under salts of the stack's has roots no record of the server's holds, so the release
side refuses it. A prover's witness salts must then be the server's for its proof to pass, since each leaf's `c` is the
SHA-512 of its salt.

**Grinding.** A stack that commits its values in `m` sessions and registers the one whose public roots it likes chooses
among `m` outcomes, so it leaks at most `log₂ m` bits through the choice (`Verity.ValueSalts`,
`Verity.Guarantees.RegisteredLeakage`). The service of record fixes `m` at 1: it refuses a commit while its epoch's
session is not yet registered, so the stack chooses neither `m` nor a session, the record's `m` says so, and the term is
0. A stack that registers more than once and passes over a registration chooses which ones it uses: each one passed over
is a failure on its stream's budget, so across a stream the choice is among the patterns of at most `b` passed over
(`Verity.Guarantees.RegisteredStreamLeakage`, every `m i` 1).

**The key is the ZK server's alone** (`zk-server-own-user`, `Verity.Assumptions.ZkServerOwnUser`): the value service and
the session loop run under the ZK server's own OS user, which no worker and no serving stack shares, so nothing the stack
runs reads the key, the epoch or the record. The tags, the record tags and the count `m` mean what they say only under
it.
-/
module

public import Flock.Hm96
public import Flock.FV3
public import Flock.Registered
public import Flock.Comp
public import Flock.JsonParse

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

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
def KEY_TAG_LABEL : ByteArray := label "verity/zk-server/interior-key-tag/v1"

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

/-- Table `j` of the commit session `s`, leaf `i`'s salt `salt i` and its tag under the key `tk`, its digests from byte
`o` of `xs`: its root, its tag and its leaves' `b ‖ c` in leaf order. -/
def commitTableWith (salt : Nat → ByteArray) (tk s : ByteArray) (j : Nat) (t : Table) (xs : ByteArray) (o : Nat) :
    ByteArray × ByteArray × ByteArray :=
  let bcs := (Array.range t.n).map fun i =>
    Hm96.Default512.commitString (xs.extract (o + DIGEST_BYTES * i) (o + DIGEST_BYTES * (i + 1))) (salt i)
  let leaves := (Array.range t.n).map fun i => FV3.Sha512.leafAs t.schemaName t.dom i (Hm96.Default512.treeLeaf bcs[i]!)
  let root := FV3.root t.dom leaves Sha512.hash
  (root, tagOf tk s j t root, bcs.foldl (· ++ ·) .empty)

/-- Table `j` of the commit session `s` under the key `k`, its digests from byte `o` of `xs`: its root, its tag and its
leaves' `b ‖ c` in leaf order. -/
def commitTable (k s : ByteArray) (j : Nat) (t : Table) (xs : ByteArray) (o : Nat) : ByteArray × ByteArray × ByteArray :=
  commitTableWith (saltOf k s j) k s j t xs o

/-- Each table's first leaf among its session's leaves, table-major, and the session's leaf count. -/
def leafOffsets (ts : Array Table) : Array Nat × Nat :=
  ts.foldl (fun (acc : Array Nat × Nat) t => (acc.1.push acc.2, acc.2 + t.n)) (#[], 0)

/-- **A commit session's tables** `ts` at the session `s`, their digests `xs` table-major: each table's root, tag and
leaves' `b ‖ c`, in table order. -/
def commitAnswer (k s : ByteArray) (ts : Array Table) (xs : ByteArray) : Array (ByteArray × ByteArray × ByteArray) :=
  let lo := (leafOffsets ts).1
  (Array.range ts.size).map fun j => commitTable k s j ts[j]! xs (DIGEST_BYTES * lo[j]!)

/-- The salt `ys` holds for leaf `i` of the table whose first leaf is `lo` among its session's. -/
def heldSalt (ys : ByteArray) (lo i : Nat) : ByteArray := ys.extract (SALT_BYTES * (lo + i)) (SALT_BYTES * (lo + i + 1))

/-- **An interior commit session's tables** `ts` at the session `s`, under the session's own key `ks`, their digests
`xs` and their leaves' salts `ys`, both table-major: each table's root, its tag under `ks` and its leaves' `b ‖ c`, in
table order. -/
def interiorAnswer (ks s ys : ByteArray) (ts : Array Table) (xs : ByteArray) :
    Array (ByteArray × ByteArray × ByteArray) :=
  let lo := (leafOffsets ts).1
  (Array.range ts.size).map fun j =>
    commitTableWith (heldSalt ys lo[j]!) ks s j ts[j]! xs (DIGEST_BYTES * lo[j]!)

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

/-- An interior commit session's key and salts, which the service holds while it runs: its session, its own key, each
table's first leaf among the session's leaves, and the leaves' salts, table-major. -/
structure Held where
  s : ByteArray
  key : ByteArray
  lo : Array Nat
  ys : ByteArray
  deriving Inhabited

/-- **The salts of the rows a refs file names**, in its row refs' order, under the key `k`: a table whose tag is the
server's under `k` takes the salts derived from `k`, and one of an interior session `held` holds, whose tag is under
that session's own key, the salts `held` holds for it. Refused whole where any table ref's tag is the server's under
neither: an interior session's key and salts are the service's only while it runs. -/
def saltsAnswer (k : ByteArray) (held : Array Held) (refs : ByteArray) : Except String ByteArray := do
  let (ts, rs) ← parseRefs refs
  let src ← (Array.range ts.size).mapM fun t => show Except String (Option (ByteArray × Nat)) from do
    let r := ts[t]!
    if r.valid k then return none
    if let some h := held.find? (·.s == r.s) then
      if r.valid h.key then return some (h.ys, h.lo[r.j]!)
    throw (s!"table ref {t}: its tag is not the ZK server's for its session, table, domain, leaf count, schema and " ++
      "root (under its key, or an interior session's own while the service that drew it runs)")
  return rs.foldl (fun acc (t, i) => let r := ts[t]!; acc ++ match src[t]! with
    | none => saltOf k r.s r.j i
    | some (ys, lo) => heldSalt ys lo i) .empty

/-! ### Registrations -/

def REG_LABEL : ByteArray := label "verity/zk-server/registration/v1"

/-- A registration record's bytes before its table refs: `be64 n ‖ be64 m ‖ be32 T`. -/
def RECORD_HEAD_BYTES : Nat := 20
def RECORD_TAG_BYTES : Nat := 64

/-- A table ref's bytes, `TableRef.parse`'s layout. -/
def TableRef.bytes (r : TableRef) : ByteArray :=
  r.s ++ be32 r.j.toUInt32 ++ r.t.dom ++ be64 r.t.n.toUInt64 ++ ByteArray.mk #[r.t.schema.toUInt8] ++ r.root ++ r.tag

/-- A commit session the service opened: its sample, each of its tables with the root it answered, in table order, and
an interior session's own key, drawn from the operating system (none for a commit session). -/
structure Opened where
  s : ByteArray
  tables : Array (Table × ByteArray)
  own : Option ByteArray := none
  deriving Inhabited

/-- The key the session `o`'s tags and record tag are under: its own where it is an interior session, else `k`. -/
def Opened.key (k : ByteArray) (o : Opened) : ByteArray := o.own.getD k

/-- **The value service's epoch**: the commit sessions it opened since its last registration, or since it started, in the
order it drew them, one at most where it numbers registrations (`Values.answer`). A registration closes it, so a session
registers only in the epoch that opened it, and only once. -/
structure Epoch where
  opened : Array Opened := #[]
  deriving Inhabited

/-- **The record tag** of registration `n`, whose epoch opened `m` commit sessions, over its `T` table refs' bytes
`trefs`, under the key `k`. -/
def recordTagOf (k : ByteArray) (n m T : Nat) (trefs : ByteArray) : ByteArray :=
  Sha512.hash (k ++ REG_LABEL ++ be64 n.toUInt64 ++ be64 m.toUInt64 ++ be32 T.toUInt32 ++ trefs)

/-- **The tag of an interior session's own key** `ks`, beside its registration's record `record` on the ZK server's
record, under the key `k`: `SHA-512(k ‖ L ‖ ks ‖ record)`, `L = SHA-512("verity/zk-server/interior-key-tag/v1\0")`,
`ks` of `KEY_BYTES` bytes, so the message splits into key and record one way only. By it the release side knows the key
the record's tags are under is one the service drew; it stays on the ZK server's record, and nothing released reads
it. -/
def keyTagOf (k ks record : ByteArray) : ByteArray := Sha512.hash (k ++ KEY_TAG_LABEL ++ ks ++ record)

/-- **The record of registration `n`**, the session `o` of an epoch that opened `m` commit sessions, under the key `k`:
`be64 n ‖ be64 m ‖ be32 T ‖` the session's `T` table refs, each with its tag, `‖` the record tag, the tags and the
record tag under the session's key (`Opened.key`). `n` and `m` are public: `n` is the registration's place among the ZK
server's, and `m` the number of sessions among which the stack chose the one it registers. -/
def recordOf (k : ByteArray) (n m : Nat) (o : Opened) : ByteArray :=
  let tk := o.key k
  let trefs := (Array.range o.tables.size).foldl (fun acc j =>
    let (t, root) := o.tables[j]!
    acc ++ ({ s := o.s, j, t, root, tag := tagOf tk o.s j t root } : TableRef).bytes) .empty
  be64 n.toUInt64 ++ be64 m.toUInt64 ++ be32 o.tables.size.toUInt32 ++ trefs ++ recordTagOf tk n m o.tables.size trefs

/-- A registration's record: its number, its epoch's session count, its table refs, their bytes, its record tag and
its bytes whole. -/
structure RegistrationRecord where
  n : Nat
  m : Nat
  tables : Array TableRef
  trefs : ByteArray
  tag : ByteArray
  bytes : ByteArray
  deriving Inhabited

/-- Whether `r`'s record tag is the server's for its number, session count and table refs. Where the ZK server's record
holds no key beside it (`own` none), the tag is checked under the key `k`. Where it holds an interior session's own key
`ks` and that key's tag, the tag is checked under `ks`, and only once `ks` is `KEY_BYTES` long and its tag is the
server's under `k` for `ks` and the record (`keyTagOf`). -/
def RegistrationRecord.valid (k : ByteArray) (r : RegistrationRecord) (own : Option (ByteArray × ByteArray)) : Bool :=
  let tag := fun key => ctEq r.tag (recordTagOf key r.n r.m r.tables.size r.trefs)
  match own with
  | none => tag k
  | some (ks, kt) => ks.size == KEY_BYTES && ctEq kt (keyTagOf k ks r.bytes) && tag ks

/-- **A registration's record**, exactly `be64 n ‖ be64 m ‖ be32 T ‖ T table refs ‖ tag`, `T` at least one. -/
def RegistrationRecord.parse (b : ByteArray) : Except String RegistrationRecord := do
  if b.size < RECORD_HEAD_BYTES then throw s!"a registration record of {b.size} bytes, shorter than its counts"
  let T := beAt b 16 4
  if T == 0 then throw "a registration record that holds no table"
  let e := RECORD_HEAD_BYTES + TABLE_REF_BYTES * T + RECORD_TAG_BYTES
  if b.size != e then throw s!"a registration record of {b.size} bytes, where {T} table refs take {e}"
  let tables ← (Array.range T).mapM fun t => TableRef.parse b (RECORD_HEAD_BYTES + TABLE_REF_BYTES * t)
  return { n := beAt b 0 8, m := beAt b 8 8, tables, trefs := b.extract RECORD_HEAD_BYTES (e - RECORD_TAG_BYTES),
           tag := b.extract (e - RECORD_TAG_BYTES) e, bytes := b }

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

/-- The hm96 trees of the public file `publicFile` (`treesOf` on its header line). -/
def treesOfFile (publicFile : ByteArray) : Except String (Array (String × Table × ByteArray)) := do
  let some nl := findByte publicFile 10 | throw "public file: no header line"
  let some text := String.fromUTF8? (publicFile.extract 0 nl) | throw "public file: its header is not UTF-8"
  treesOf (← JsonParse.parse text)

/-- **The registrations a statement reads**, in order and each once: for each of its hm96 trees `trees`, the first of the
registrations `regs` (registration `n`'s record at `n`, with the interior session's own key and its tag where the ZK
server's record holds them) that holds a table at the tree's domain, leaf count, schema and root, its record tag checked
under the key `k`, or under the interior session's own key that `k` vouches for (`RegistrationRecord.valid`). Refused
where a record does not read as one, where no registration holds a tree (the stack committed it under salts of its own,
or never registered it), or where the one that holds it does not carry the server's record tag. -/
def readsOf (k : ByteArray) (regs : Array (ByteArray × Option (ByteArray × ByteArray)))
    (trees : Array (String × Table × ByteArray)) : Except String (Array Nat) := do
  let rs ← (Array.range regs.size).mapM fun n =>
    (RegistrationRecord.parse regs[n]!.1).mapError (s!"registration {n}: {·}")
  let mut reads : Array Nat := #[]
  for (name, t, root) in trees do
    let holds := fun (r : RegistrationRecord) =>
      r.tables.any fun x => x.t.dom == t.dom && x.t.n == t.n && x.t.schema == t.schema && x.root == root
    let some n := (List.range rs.size).find? (holds rs[·]!)
      | throw s!"row {name}: its root is no value commitment of the ZK server's (no registration on its record holds a table of its domain, leaf count, schema and root)"
    unless rs[n]!.valid k regs[n]!.2 do
      throw s!"registration {n}: its record tag is not the ZK server's for its number, session count and table refs"
    unless reads.contains n do reads := reads.push n
  return reads.qsort (· < ·)

/-! ### The key -/

/-- **The salt key in `path`**, exactly its 64 bytes. With `create`, where `path` does not exist, 64 bytes sampled from the
operating system, in a file created there that its owner alone may read and write, set so before the key is written. -/
def loadKey (path : System.FilePath) (create : Bool) : IO (Except String ByteArray) := do
  if (← path.pathExists) then
    let b ← IO.FS.readBinFile path
    if b.size != KEY_BYTES then return .error s!"--salt-key {path}: {b.size} bytes, where a salt key is {KEY_BYTES}"
    return .ok b
  if !create then return .error s!"--salt-key {path}: no such file (`core-firewall values` creates it)"
  let k ← IO.getRandomBytes KEY_BYTES.toUSize
  let h ← IO.FS.Handle.mk path .writeNew
  IO.setAccessRights path { user := { read := true, write := true } }
  h.write k
  h.flush
  return .ok k

end Flock.Firewall.Salts
