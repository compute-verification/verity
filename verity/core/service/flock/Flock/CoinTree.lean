/-
`coin-tree/hm96-sha512/v2`, the keyed coin tree (`docs/coin-tree-v2.md`, §3 and §7): the verifier's part. The prover's
`Hello` carries its fresh 32-byte nonce in `coins`; the server answers with its root and a fresh 256-byte hm96 key for the
session's tree, drawn after the nonce, never reused and never revealed before that answer. The record keeps
`coin_tree = {key, nonce, root, spec}`.

The verifier checks the record, not the openings: the commitment protects the prover (§7). It checks R7 with the prover's
nonce, the record's `coin_tree`, and, across the sessions it verifies together, that no key repeats. Nonces are not
compared across sessions: a repeated prover nonce is not a soundness matter.
-/
import Flock.Canon
import Flock.JsonParse

namespace Flock.CoinTree

open Lean (Json)

def SCHEME : String := "coin-tree/hm96-sha512/v2"

/-- The committed schedule's shape: the streams in leaf order, `2^roundsLog` rounds each, at most `block` coins a round. -/
structure Spec where
  streams : Array String
  roundsLog : Nat
  block : Nat
  deriving BEq, Inhabited, Repr

/-- `Spec::to_json`: the record's `spec`. -/
def Spec.toJson (s : Spec) : Json :=
  Json.mkObj [("block", Lean.toJson s.block), ("rounds_log", Lean.toJson s.roundsLog), ("scheme", Json.str SCHEME),
    ("streams", Json.arr (s.streams.map Json.str))]

/-- `Hello`'s `coins` (§3): the spec with the prover's nonce. -/
def Spec.coins (s : Spec) (nonce : String) : Json := s.toJson.setObjVal! "nonce" (Json.str nonce)

/-- The tree's spec, when a `Hello` names this scheme in `coins`. -/
def ofHello (hello : String) : Option Spec := do
  let c ← ((JsonParse.parse hello).toOption.bind (·.getObjVal? "coins" |>.toOption))
  if (c.getObjValAs? String "scheme").toOption != some SCHEME then none
  let streams ← (c.getObjValAs? (Array String) "streams").toOption
  return ⟨streams, ← (c.getObjValAs? Nat "rounds_log").toOption, ← (c.getObjValAs? Nat "block").toOption⟩

/-- `n` lowercase hex digits. -/
def isHex (n : Nat) (s : String) : Bool := s.length == n && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

/-- **R7 with the prover's nonce** (§3): the received `Hello` must be the verifier's own `Hello` object with the received
`coins.nonce` put in, serialized canonically, byte for byte; the nonce must be 64 lowercase hex digits. The nonce. -/
def checkHello (own : Json) (received : String) : V String := do
  let some j := (JsonParse.parse received).toOption | throw "S2/R7: the prover's Hello is not JSON"
  let nonce := ((j.getObjVal? "coins").toOption.bind (·.getObjValAs? String "nonce" |>.toOption)).getD ""
  if !isHex 64 nonce then throw "S2/R7: Hello's coins.nonce is not 64 lowercase hex digits"
  let coins := (own.getObjVal? "coins").toOption.getD Json.null
  if (← canon (own.setObjVal! "coins" (coins.setObjVal! "nonce" (Json.str nonce)))) != received then
    throw "S2/R7: the session parameters are not the verifier's"
  return nonce

/-- **The record's `coin_tree`** (§7): exactly `key`, `nonce`, `root` and `spec`; the nonce `Hello`'s; the key 256 bytes
with bit 2047 zero (bit 7 of byte 255), the root 64 bytes, both lowercase hex; the spec the verifier's. The key. -/
def checkRecord (spec : Spec) (nonce : String) (ct : Json) : V String := do
  let names := match ct with
    | .obj kvs => kvs.toArray.map fun (kv : String × Json) => kv.1
    | _ => #[]
  if names.qsort (· < ·) != #["key", "nonce", "root", "spec"] then throw "S2: coin_tree is not {key, nonce, root, spec}"
  let str := fun (k : String) => (ct.getObjValAs? String k).toOption.getD ""
  if str "nonce" != nonce then throw "S2: coin_tree.nonce is not Hello's nonce"
  let key := str "key"
  if !isHex 512 key then throw "S2: coin_tree.key is not 256 bytes of lowercase hex"
  if !"01234567".contains (key.toList.getD 510 'f') then throw "S2: coin_tree.key has bit 2047 set"
  if !isHex 128 (str "root") then throw "S2: coin_tree.root is not 64 bytes of lowercase hex"
  if (← canon ((ct.getObjVal? "spec").toOption.getD Json.null)) != (← canon spec.toJson) then
    throw "S2: coin_tree.spec is not the verifier's"
  return key

/-- **The session key is fresh** (the red team's condition on the session key): never reused. Refuse a key that one of the sessions
verified before this one used. -/
def checkFresh (seen : Array String) (key : String) : V Unit :=
  if seen.contains key then throw "S2: coin_tree.key is an earlier session's; a session key is never reused" else pure ()

end Flock.CoinTree
