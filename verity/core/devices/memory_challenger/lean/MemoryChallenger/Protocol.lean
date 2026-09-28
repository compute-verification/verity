import CoreProtocol.Protocol
import CoreProtocol.Cli
import MemoryChallenger.Verdict

/-!
# `verdict` as a core protocol

The memory challenger's verdict behind the core protocol interface (`Core.Protocol`), its bytes canonical JSON:

* parameters `{"expected": {"<i>": hex, …}, "trusted": [hex, …]}`: `expected["<i>"]` is `block_digest` of block `i` of
  `C`, keyed by `i`'s decimal digits (a block with no entry has no answer, so the digests of the blocks the rounds
  name are enough, and C's whole table never has to be read), `trusted` the memory challengers' public keys;
* the commitment: `SessionCommitment.to_json()` (its header and `coin_key`), its digest SHA-512 of those bytes;
* the evidence: `{"openings": [CoinOpening.to_json(), …], "session": SignedSession.to_json()}`;
* the check: `verdict`. On acceptance the public output is the verdict alone, `{"accepted": true}` (`Verdict.public`),
  and there is no profile (`null`); a rejection's reason is the verdict's reasons joined by `"; "`.

Each input is the canonical JSON of the object it decodes to and nothing else: a field missing, an extra one, a byte
string not lowercase hex, an integer negative, or a constant of the header not the protocol's is refused. `protocol` is
S1's (`SCHEMES`); `testProtocol`, under an id of its own, accepts the test signers' schemes and only them, as the Python's
tests pass `pous_test_signer.SCHEMES`.
-/

namespace MemoryChallenger

open Lean (Json)
open Flock

private def hexJ (b : ByteArray) : Json := .str (hexOf b)
private def natJ (n : Nat) : Json := Lean.toJson n

def SessionCommitment.toJson (c : SessionCommitment) : Json :=
  Json.mkObj [("protocol", .str PROTOCOL), ("coin_tree", .str COIN_TREE), ("records_tree", .str RECORDS_TREE),
    ("session", hexJ c.session), ("coin_root", hexJ c.coinRoot), ("coin_key_sha512", hexJ (Sha512.hash c.coinKey)),
    ("blocks", natJ c.blocks), ("k", natJ c.k), ("limit_ns", natJ c.limitNs), ("block_bytes", natJ c.blockBytes),
    ("binding", hexJ c.binding), ("clock", .str CLOCK), ("coin_key", hexJ c.coinKey)]

def Round.toJson (r : Round) : Json :=
  Json.mkObj [("j", natJ r.j), ("index", natJ r.index), ("send_ns", natJ r.sendNs), ("receipt_ns", natJ r.receiptNs),
    ("digest", hexJ r.digest)]

def CoinOpening.toJson (o : CoinOpening) : Json :=
  Json.mkObj [("j", natJ o.j), ("index", natJ o.index), ("salt", hexJ o.salt), ("path", .arr (o.path.map hexJ))]

def SignedSession.toJson (s : SignedSession) : Json :=
  Json.mkObj [("commitment", s.commitment.toJson), ("ended", .str s.ended), ("rounds", .arr (s.rounds.map Round.toJson)),
    ("scheme", .str s.scheme), ("public_key", hexJ s.publicKey), ("signature", hexJ s.signature)]

/-- `C`'s digests by block index, for the blocks a caller has them for. -/
abbrev Digests := Array (Nat × ByteArray)

def Digests.toJson (d : Digests) : Json := Json.mkObj (d.toList.map fun (i, h) => (toString i, hexJ h))

def Digests.find (d : Digests) (i : Nat) : Option ByteArray := (d.find? (·.1 == i)).map (·.2)

structure Params where
  expected : Digests
  trusted : Array ByteArray

def Params.toJson (p : Params) : Json :=
  Json.mkObj [("expected", p.expected.toJson), ("trusted", .arr (p.trusted.map hexJ))]

structure Evidence where
  session : SignedSession
  openings : Array CoinOpening

def Evidence.toJson (e : Evidence) : Json :=
  Json.mkObj [("openings", .arr (e.openings.map CoinOpening.toJson)), ("session", e.session.toJson)]

def hexField (j : Json) (k ctx : String) : V ByteArray := do
  match unhex (← J.str j k ctx) with
  | .ok b => pure b
  | .error e => throw s!"{ctx}.{k}: {e}"

def hexArr (j : Json) (k ctx : String) : V (Array ByteArray) := do
  (← J.arr j k ctx).mapM fun x => do
    match unhex (← J.asStr x s!"{ctx}.{k}") with
    | .ok b => pure b
    | .error e => throw s!"{ctx}.{k}: {e}"

def objs (j : Json) (k ctx : String) {α} (f : Json → V α) : V (Array α) := do (← J.arr j k ctx).mapM f

/-- An object of digests keyed by block index, each key a natural's decimal digits with no leading zero. -/
def Digests.ofJson (j : Json) (k ctx : String) : V Digests := do
  match ← J.get j k ctx with
  | .obj kvs => kvs.toArray.mapM fun (key, v) => do
    let some i := key.toNat? | throw s!"{ctx}.{k}: the key {key} is not a block index"
    if toString i != key then throw s!"{ctx}.{k}: the key {key} is not a block index"
    match unhex (← J.asStr v s!"{ctx}.{k}") with
    | .ok b => pure (i, b)
    | .error e => throw s!"{ctx}.{k}: {e}"
  | _ => throw s!"{ctx}.{k}: not an object"

def SessionCommitment.ofJson (j : Json) : V SessionCommitment := do
  let ctx := "commitment"
  pure { session := ← hexField j "session" ctx, coinRoot := ← hexField j "coin_root" ctx, coinKey := ← hexField j "coin_key" ctx,
         blocks := ← J.nat j "blocks" ctx, k := ← J.nat j "k" ctx, limitNs := ← J.nat j "limit_ns" ctx,
         blockBytes := ← J.nat j "block_bytes" ctx, binding := ← hexField j "binding" ctx }

def Round.ofJson (j : Json) : V Round := do
  pure { j := ← J.nat j "j" "round", index := ← J.nat j "index" "round", sendNs := ← J.nat j "send_ns" "round",
         receiptNs := ← J.nat j "receipt_ns" "round", digest := ← hexField j "digest" "round" }

def CoinOpening.ofJson (j : Json) : V CoinOpening := do
  pure { j := ← J.nat j "j" "opening", index := ← J.nat j "index" "opening", salt := ← hexField j "salt" "opening",
         path := ← hexArr j "path" "opening" }

def SignedSession.ofJson (j : Json) : V SignedSession := do
  let ctx := "session"
  pure { commitment := ← SessionCommitment.ofJson (← J.get j "commitment" ctx), ended := ← J.str j "ended" ctx,
         rounds := ← objs j "rounds" ctx Round.ofJson, scheme := ← J.str j "scheme" ctx,
         publicKey := ← hexField j "public_key" ctx, signature := ← hexField j "signature" ctx }

/-- Decode `j` with `f`, then require `j` to be exactly the canonical JSON of what it decoded to. -/
def exactly {α} (f : Json → V α) (enc : α → Json) (j : Json) : V α := do
  let a ← f j
  if (← canon (enc a)) != (← canon j) then throw "not the canonical JSON of what it holds (a field extra, or a constant not the protocol's)"
  pure a

def decodeParams : Json → V Params :=
  exactly (fun j => do pure ⟨← Digests.ofJson j "expected" "params", ← hexArr j "trusted" "params"⟩) Params.toJson

def decodeCommitment (b : ByteArray) : V SessionCommitment := do
  exactly SessionCommitment.ofJson SessionCommitment.toJson (← Core.canonicalJson b)

def decodeEvidence (b : ByteArray) : V Evidence := do
  exactly (fun j => do pure ⟨← SignedSession.ofJson (← J.get j "session" "evidence"),
                             ← objs j "openings" "evidence" CoinOpening.ofJson⟩) Evidence.toJson (← Core.canonicalJson b)

/-- The commitment's digest: SHA-512 of its canonical JSON, the bytes it was decoded from. -/
def commitmentDigest (c : SessionCommitment) : ByteArray :=
  match canon c.toJson with
  | .ok s => Sha512.hash s.toUTF8
  | .error _ => ByteArray.empty

/-- The verdict as an outcome: accepted, with nothing but that public, or rejected with its reasons. -/
def outcome (schemes : Schemes) (p : Params) (c : SessionCommitment) (e : Evidence) : Core.Outcome Unit Unit :=
  let v := verdict schemes p.trusted.toList p.expected.find e.session c e.openings
  if v.accepted then .accept () () else .reject ("; ".intercalate v.reasons.toList)

def mkProtocol (id : String) (schemes : Schemes) : Core.Protocol where
  id := id
  Params := Params
  Commitment := SessionCommitment
  Evidence := Evidence
  Public := Unit
  Profile := Unit
  params := decodeParams
  commitment := decodeCommitment
  evidence := decodeEvidence
  digest := commitmentDigest
  check := outcome schemes
  publicJson := fun _ => Json.mkObj [("accepted", .bool true)]
  profileJson := fun _ => Json.null

/-- `verity-pous/memory-challenger/v1` under the accepted schemes. -/
def protocol : Core.Protocol := mkProtocol PROTOCOL SCHEMES

/-- The same check under the test signers' schemes alone, its id `test/verity-pous/memory-challenger/v1`. -/
def testProtocol : Core.Protocol := mkProtocol (TEST_PREFIX ++ PROTOCOL) TEST_SCHEMES

end MemoryChallenger
