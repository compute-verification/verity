import Flock.Hm96
import Flock.HmRow
import Randomness
import Flock.FV3

/-!
# The memory challenger's records and coins

`verity/core/protocols/pous/memory_challenger.py`'s formats, whose module docstring is the spec: the coin tree
(`coinDigest`, `coinLeaf`, `CoinTree.make`, `CoinTree.draw`, `CoinTree.opening`, `verifyCoin`), the session commitment
and its header, the round record (`Round.value`), the records root, the signed message, and `blockDigest`. A Python
exception is an `Except.error` here, and where the Python's caller reads an exception as "no" (`verify_coin`), so does the
Lean. Integers are big-endian.
-/

namespace MemoryChallenger

open Flock

abbrev E := Except String

def PROTOCOL : String := "verity-pous/memory-challenger/v1"
def COIN_DOMAIN : String := "verity/pous/memory-challenger-coins/v1"
def COIN_TREE : String := "verity-pous/coin-tree/hm96-sha512/v1"
def RECORDS_TREE : String := "verity-pous/round-records/frame-v3-sha512/v1"
def COIN_TAG : ByteArray := "verity-pous/memory-challenger/coin/v1".toUTF8.push 0
def COIN_BINDING : ByteArray := "verity-pous/memory-challenger/coin-tree/v1".toUTF8.push 0
def RECORD_BINDING : ByteArray := "verity-pous/memory-challenger/round-records/v1".toUTF8.push 0
def SESSION_TAG : ByteArray := "verity-pous/memory-challenger/session/v1".toUTF8.push 0
def COIN_SCHEMA : String := "pous/coin/hm96-sha512/v1"
def RECORD_SCHEMA : String := "pous/round-record/v2"
def TEST_PREFIX : String := "test/"
def COMPLETE : String := "complete"
def LATE : String := "late"
def CLOCK : String := "CLOCK_MONOTONIC ns"

/-- The stream a coin index reads, in bytes: `Randomness.uniform` reads a bounded prefix of the Python's unbounded stream.
A draw over fewer than 2^64 blocks takes 16-byte words, so this is 256 words, each rejected with probability below
2^-64. -/
def DRAW_BYTES : Nat := 4096

/-- `x.to_bytes(n, "big")`, refused at `2^(8n)` and above, as Python's `OverflowError`. -/
def beN (n x : Nat) (what : String) : E ByteArray :=
  if x < 2 ^ (8 * n) then pure ⟨(Array.range n).map fun i => (x >>> (8 * (n - 1 - i))).toUInt8⟩
  else throw s!"{what} = {x} does not fit in {n} bytes"

/-- `block_digest`: the block's `sha512/row/v2` digest, `SHA-512(P(8N) ‖ block)`. -/
def blockDigest (block : ByteArray) : ByteArray := Sha512.hash (HmRow.rowPrefixV2 (8 * block.size) ++ block)

/-- `_domain(tag, session, count)`: binding the first 32 bytes of `SHA-512(tag ‖ session)`, owner −1 (the boundary). -/
def domain (tag session : ByteArray) (count : Nat) : FV3.Sha512.Domain :=
  .make ((Sha512.hash (tag ++ session)).extract 0 32) (-1) 0 count

/-! ## Coins -/

def HM96 : Hm96 := Hm96.sha512

/-- `HM96.check_key`: 256 bytes, bit 2047 (bit 7 of the last byte) zero. -/
def checkKey (key : ByteArray) : E Unit :=
  if key.size == HM96.keyBytes && key.get! (HM96.keyBytes - 1) &&& 0x80 == 0 then pure ()
  else throw s!"an {HM96.name} key is {HM96.keyBytes} bytes with bit {HM96.keyBits} and above zero"

def checkSalt (salt : ByteArray) : E Unit :=
  if salt.size == HM96.saltBytes then pure () else throw s!"an {HM96.name} salt is {HM96.saltBytes} bytes"

/-- `coin_digest`: `x_j = SHA-512(COIN_TAG ‖ session ‖ u32 j ‖ u64 index)`. -/
def coinDigest (session : ByteArray) (j index : Nat) : E ByteArray := do
  pure (Sha512.hash (COIN_TAG ++ session ++ (← beN 4 j "j") ++ (← beN 8 index "index")))

/-- `coin_leaf`: `HM96.tree_leaf(key, HM96.commit_string(key, x_j, salt))`. -/
def coinLeaf (key session : ByteArray) (j index : Nat) (salt : ByteArray) : E ByteArray := do
  let x ← coinDigest session j index
  checkKey key
  checkSalt salt
  pure (HM96.treeLeaf key (HM96.commitString key x salt))

structure CoinOpening where
  j : Nat
  index : Nat
  salt : ByteArray
  path : Array ByteArray
  deriving BEq, Inhabited

/-- The session's `k` coins over `blocks` blocks, committed on the device before the session and released after it. -/
structure CoinTree where
  session : ByteArray
  blocks : Nat
  indices : Array Nat
  key : ByteArray
  salts : Array ByteArray
  levels : Array (Array ByteArray)
  root : ByteArray

/-- `CoinTree(session, blocks, indices, key, salts)`, refused as the Python refuses it. -/
def CoinTree.make (session : ByteArray) (blocks : Nat) (indices : Array Nat) (key : ByteArray) (salts : Array ByteArray) :
    E CoinTree := do
  if session.size ≠ 32 then throw "a session id is 32 bytes"
  if blocks < 1 || indices.isEmpty || salts.size != indices.size || indices.any (· ≥ blocks) then
    throw "a coin tree has one salt per index, every index a block"
  checkKey key
  let d := domain COIN_BINDING session indices.size
  let hm ← (Array.range indices.size).mapM fun j => coinLeaf key session j indices[j]! salts[j]!
  let levels := d.levels ((Array.range hm.size).map fun j => d.leaf j j COIN_SCHEMA hm[j]!)
  pure ⟨session, blocks, indices, key, salts, levels, levels.back!.getD 0 ByteArray.empty⟩

/-- `derive(source, COIN_DOMAIN, {"session", "blocks", "k"})` and its `uniform(blocks, j)` for `j < k`: the indices.
`derive` refuses an empty source (a `TypeError`), and `uniform` no blocks. -/
def drawIndices (source session : ByteArray) (blocks k : Nat) : E (Array Nat) := do
  if source.isEmpty then throw "source must be nonempty bytes: a key holder's secret"
  let key := FlockSoundness.Randomness.derive source COIN_DOMAIN
    [("session", .bytes session), ("blocks", .int blocks), ("k", .int k)]
  if k > 0 && blocks = 0 then throw "uniform(n) needs n > 0"
  (Array.range k).mapM fun j =>
    match FlockSoundness.Randomness.uniform key blocks [.int j] DRAW_BYTES with
    | some i => pure i
    | none => throw s!"coin {j}'s stream ran out at {DRAW_BYTES} bytes"

/-- `CoinTree.draw(source, session, blocks, k, key=key, salts=salts)`: the key and salts as given (the driver takes them
from the world's coins). -/
def CoinTree.draw (source session : ByteArray) (blocks k : Nat) (key : ByteArray) (salts : Array ByteArray) : E CoinTree := do
  CoinTree.make session blocks (← drawIndices source session blocks k) key salts

def CoinTree.k (t : CoinTree) : Nat := t.indices.size

def CoinTree.opening (t : CoinTree) (j : Nat) : CoinOpening :=
  ⟨j, t.indices[j]!, t.salts[j]!, FV3.Sha512.path t.levels j⟩

/-! ## The session -/

/-- What the memory challenger publishes before the session. Every field is in the signed header. -/
structure SessionCommitment where
  session : ByteArray
  coinRoot : ByteArray
  coinKey : ByteArray
  blocks : Nat
  k : Nat
  limitNs : Nat
  blockBytes : Nat
  binding : ByteArray
  deriving BEq, Inhabited

/-- `verify_coin`: the opening opens round `j`'s coin under the commitment's coin tree, to a block index. -/
def verifyCoin (c : SessionCommitment) (o : CoinOpening) : Bool :=
  o.j < c.k && o.index < c.blocks &&
    match coinLeaf c.coinKey c.session o.j o.index o.salt with
    | .error _ => false
    | .ok leaf => (domain COIN_BINDING c.session c.k).verifyOpening c.coinRoot c.k o.j leaf COIN_SCHEMA o.path

/-- One round's record. -/
structure Round where
  j : Nat
  index : Nat
  sendNs : Nat
  receiptNs : Nat
  digest : ByteArray
  deriving BEq, Inhabited

def Round.elapsedNs (r : Round) : Int := (r.receiptNs : Int) - r.sendNs

/-- `Round.value`: `u32 j ‖ u64 index ‖ u64 send_ns ‖ u64 receipt_ns ‖ H(block)`. -/
def Round.value (r : Round) : E ByteArray := do
  pure ((← beN 4 r.j "j") ++ (← beN 8 r.index "index") ++ (← beN 8 r.sendNs "send_ns") ++ (← beN 8 r.receiptNs "receipt_ns")
    ++ r.digest)

/-- `records_root`: the frame-v3-sha512 root over the round records, the empty root for none. -/
def recordsRoot (session : ByteArray) (rounds : Array Round) : E ByteArray := do
  let d := domain RECORD_BINDING session rounds.size
  let values ← rounds.mapM Round.value
  pure (d.root ((Array.range values.size).map fun r => d.leaf r r RECORD_SCHEMA values[r]!))

private def hex4 (n : Nat) : String :=
  let d := fun (k : Nat) => let x := (n >>> (4 * k)) % 16; if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)
  String.ofList [d 3, d 2, d 1, d 0]

/-- A string as Python's `json.dumps` writes it (`ensure_ascii`): `"` and `\` escaped, `\b \f \n \r \t` short, every
other character outside `' '…'~'` as `\uXXXX` (above U+FFFF a surrogate pair), lowercase. -/
def pyJsonStr (s : String) : String := Id.run do
  let mut out := "\""
  for c in s.toList do
    let n := c.toNat
    out := match c with
      | '"' => out ++ "\\\""
      | '\\' => out ++ "\\\\"
      | '\n' => out ++ "\\n"
      | '\r' => out ++ "\\r"
      | '\t' => out ++ "\\t"
      | c => if n == 8 then out ++ "\\b" else if n == 12 then out ++ "\\f"
             else if 0x20 ≤ n && n ≤ 0x7e then out.push c
             else if n < 0x10000 then out ++ "\\u" ++ hex4 n
             else let m := n - 0x10000; out ++ "\\u" ++ hex4 (0xd800 + (m >>> 10)) ++ "\\u" ++ hex4 (0xdc00 + (m &&& 0x3ff))
  return out ++ "\""

/-- The header's fields, in key order, as `SessionCommitment.header()` and `message` write them. -/
def SessionCommitment.headerFields (c : SessionCommitment) : List (String × String) :=
  [("binding", pyJsonStr (hexOf c.binding)), ("block_bytes", toString c.blockBytes), ("blocks", toString c.blocks),
   ("clock", pyJsonStr CLOCK), ("coin_key_sha512", pyJsonStr (hexOf (Sha512.hash c.coinKey))),
   ("coin_root", pyJsonStr (hexOf c.coinRoot)), ("coin_tree", pyJsonStr COIN_TREE), ("k", toString c.k),
   ("limit_ns", toString c.limitNs), ("protocol", pyJsonStr PROTOCOL), ("records_tree", pyJsonStr RECORDS_TREE),
   ("session", pyJsonStr (hexOf c.session))]

/-- The signed header: the commitment's header with how the session ended and its number of rounds, as
`json.dumps(sort_keys=True, separators=(",", ":"))`. -/
def header (c : SessionCommitment) (ended : String) (rounds : Nat) : String :=
  let fields := (c.headerFields ++ [("ended", pyJsonStr ended), ("rounds", toString rounds)]).toArray.qsort (·.1 < ·.1)
  "{" ++ ",".intercalate (fields.toList.map fun (k, v) => pyJsonStr k ++ ":" ++ v) ++ "}"

/-- `message`: what the session's one signature signs, `SHA-512(SESSION_TAG ‖ u64 len ‖ header ‖ records root)`. -/
def message (c : SessionCommitment) (ended : String) (rounds : Array Round) : E ByteArray := do
  let head := (header c ended rounds.size).toUTF8
  let root ← recordsRoot c.session rounds
  pure (Sha512.hash (SESSION_TAG ++ be64 head.size.toUInt64 ++ head ++ root))

end MemoryChallenger
