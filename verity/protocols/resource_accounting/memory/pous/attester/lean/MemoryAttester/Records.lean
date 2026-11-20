module

public import Flock.HmRow
public import Randomness
public import Flock.FV3
public import Commitments.SessionCoins

/-!
# The memory challenger's records and coins

`verity/protocols/accounting/pous/device/memory_challenger.py`'s formats, whose module docstring is the spec: the coin tree
(`coinDigest`, `coinLeaf`, `CoinTree.make`, `CoinTree.draw`, `CoinTree.opening`, `verifyCoin`), the session commitment
and its header, the round record (`Round.value`), the records digest, the signed message, and `blockDigest`. A Python
exception is an `Except.error` here, and where the Python's caller reads an exception as "no" (`verify_coin`), so does the
Lean. Integers are big-endian. The coin's digest and leaf, the opening's check (`verifyCoin`), the tags and the signed header
are `Flock.SessionCoins`'s (the commitments' package), which a supplied function's body reads too; the names here are its.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace MemoryAttester

open Flock
open Flock.SessionCoins (COIN_BINDING COIN_SCHEMA)

abbrev E := Except String

def COIN_DOMAIN : String := "verity/pous/memory-challenger-coins/v1"
def SESSION_TAG : ByteArray := "verity-pous/memory-challenger/session/v2".toUTF8.push 0
/-- The session record's salt, the device's own draw a session: the signed digest hides the rows it covers. -/
def RECORD_SALT_BYTES : Nat := 64
def RECORD_BYTES : Nat := 92
def TEST_PREFIX : String := "test/"

/-- The stream a coin index reads, in bytes: `Randomness.uniform` reads a bounded prefix of the Python's unbounded stream.
A sample over fewer than 2^64 blocks takes 16-byte words, so this is 256 words, each rejected with probability below
2^-64. -/
def DRAW_BYTES : Nat := 4096

def beN (n x : Nat) (what : String) : E ByteArray := SessionCoins.beN n x what

/-- `block_digest`: the block's `sha512/row/v2` digest, `SHA-512(P(8N) ‖ block)`. -/
def blockDigest (block : ByteArray) : ByteArray := Sha512.hash (HmRow.rowPrefixV2 (8 * block.size) ++ block)

def domain (tag session : ByteArray) (count : Nat) : FV3.Sha512.Domain := SessionCoins.domain tag session count

/-! ## Coins -/

def HM96 : Hm96 := SessionCoins.HM96

def checkKey (key : ByteArray) : E Unit := SessionCoins.checkKey key

def checkSalt (salt : ByteArray) : E Unit := SessionCoins.checkSalt salt

def coinDigest (session : ByteArray) (j index : Nat) : E ByteArray := SessionCoins.coinDigest session j index

def coinLeaf (key session : ByteArray) (j index : Nat) (salt : ByteArray) : E ByteArray :=
  SessionCoins.coinLeaf key session j index salt

structure CoinOpening where
  j : Nat
  index : Nat
  salt : ByteArray
  path : Array ByteArray
  deriving BEq, Inhabited

/-- The session's `k` coins over `blocks` blocks, committed before the session. -/
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
def sampleIndices (source session : ByteArray) (blocks k : Nat) : E (Array Nat) := do
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
  CoinTree.make session blocks (← sampleIndices source session blocks k) key salts

def CoinTree.k (t : CoinTree) : Nat := t.indices.size

def CoinTree.opening (t : CoinTree) (j : Nat) : CoinOpening :=
  ⟨j, t.indices[j]!, t.salts[j]!, FV3.Sha512.path t.levels j⟩

/-! ## The session -/

/-- What the memory challenger publishes before the session. Every field is in the signed header. -/
structure ChallengePhaseCommitment where
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
def verifyCoin (c : ChallengePhaseCommitment) (o : CoinOpening) : Bool :=
  SessionCoins.verify c.session c.coinRoot c.coinKey c.blocks c.k o.j o.index o.salt o.path

/-- One round's record. -/
structure Round where
  j : Nat
  index : Nat
  sendNs : Nat
  receiptNs : Nat
  digest : ByteArray
  deriving BEq, Inhabited

def Round.elapsedNs (r : Round) : Int := (r.receiptNs : Int) - r.sendNs

/-- `Round.value`: `u32 j ‖ u64 index ‖ u64 send_ns ‖ u64 receipt_ns ‖ H(block)`, `RECORD_BYTES` bytes; a digest of another
length is refused, so the records' stream splits one way. -/
def Round.value (r : Round) : E ByteArray :=
  if r.digest.size ≠ 64 then throw "a round's digest is 64 bytes" else do
  pure ((← beN 4 r.j "j") ++ (← beN 8 r.index "index") ++ (← beN 8 r.sendNs "send_ns") ++ (← beN 8 r.receiptNs "receipt_ns")
    ++ r.digest)

/-- The round records' values in round order, as one stream. -/
def recordsStream (rounds : Array Round) : E ByteArray := do
  pure ((← rounds.mapM Round.value).foldl (· ++ ·) ByteArray.empty)

/-- `records_digest`: the records stream's `blockDigest`. -/
def recordsDigest (rounds : Array Round) : E ByteArray := do
  pure (blockDigest (← recordsStream rounds))

def pyJsonStr (s : String) : String := SessionCoins.pyJsonStr s

/-- The signed header: the commitment's header with how the session ended and its number of rounds. -/
def header (c : ChallengePhaseCommitment) (ended : String) (rounds : Nat) : String :=
  SessionCoins.header c.session c.coinRoot (Sha512.hash c.coinKey) c.binding c.blocks c.k c.limitNs c.blockBytes ended rounds

/-- `message`: what the session's one signature signs, `SHA-512(SESSION_TAG ‖ salt ‖ u64 len ‖ header ‖ records digest)`,
the salt `RECORD_SALT_BYTES` bytes, so it ends where the tag says and the header's length begins. The record's salt
is the memory attester's own draw, `RECORD_SALT_BYTES` (64) bytes from its coins source as the session opens
(`challenger_party.Challenger._coins`, `wipe_challenger.MemoryAttesterParty.start`; `os.urandom` in
`MemoryAttester.__init__` when none is given; in Lean the device's own coins, `Session.Params.salt`), outside the
session's commitment and before its first index is revealed, and read once, when `close()` signs; it makes the signed
digest m a hiding commitment to the guessable header and records, and nothing in soundness rests on its being random or
fresh: a salt the prover could choose would still leave the signature binding the same records. -/
def message (c : ChallengePhaseCommitment) (salt : ByteArray) (ended : String) (rounds : Array Round) : E ByteArray := do
  if salt.size ≠ RECORD_SALT_BYTES then throw s!"a session record's salt is {RECORD_SALT_BYTES} bytes"
  let head := (header c ended rounds.size).toUTF8
  let digest ← recordsDigest rounds
  pure (Sha512.hash (SESSION_TAG ++ salt ++ be64 head.size.toUInt64 ++ head ++ digest))

end MemoryAttester
