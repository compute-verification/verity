import Difftest
import Lms.Sign
import MemoryChallenger.Protocol

/-!
# The difftest's Lean half

The functions `tests/` holds to `memory_challenger.py` through `verity_difftest.Port`, each over JSON in the formats of
`MemoryChallenger.Protocol` (byte strings lowercase hex, integers naturals): the crypto and record formats, and LMS
signing at a leaf and verification. `memory-challenger-difftest` serves them (`Difftest.serve`). The verdict's own
difftest drives `memory-challenger-verdict`, the executable its callers run.
-/

namespace MemoryChallenger.Difftest

open Lean (Json)
open Flock

private def hex (j : Json) (k : String) : V ByteArray := do
  match unhex (← J.str j k "input") with
  | .ok b => pure b
  | .error e => throw s!"{k}: {e}"

private def hexes (j : Json) (k : String) : V (Array ByteArray) := do
  (← J.arr j k "input").mapM fun x => do
    match unhex (← J.asStr x k) with
    | .ok b => pure b
    | .error e => throw s!"{k}: {e}"

private def hexJ (b : ByteArray) : Json := .str (hexOf b)

private def rounds (j : Json) : V (Array Round) := do (← J.arr j "rounds" "input").mapM Round.ofJson

def coinDigestFn (j : Json) : V Json := do
  pure (hexJ (← coinDigest (← hex j "session") (← J.nat j "j" "input") (← J.nat j "index" "input")))

def coinTreeFn (j : Json) : V Json := do
  let t ← CoinTree.draw (← hex j "source") (← hex j "session") (← J.nat j "blocks" "input") (← J.nat j "k" "input")
    (← hex j "key") (← hexes j "salts")
  pure (Json.mkObj [("indices", .arr (t.indices.map Lean.toJson)), ("root", hexJ t.root),
    ("openings", .arr ((Array.range t.k).map fun i => (t.opening i).toJson))])

def verifyCoinFn (j : Json) : V Json := do
  pure (.bool (verifyCoin (← SessionCommitment.ofJson (← J.get j "commitment" "input"))
    (← CoinOpening.ofJson (← J.get j "opening" "input"))))

def recordsRootFn (j : Json) : V Json := do pure (hexJ (← recordsRoot (← hex j "session") (← rounds j)))

def headerFn (j : Json) : V Json := do
  pure (.str (header (← SessionCommitment.ofJson (← J.get j "commitment" "input")) (← J.str j "ended" "input")
    (← J.nat j "rounds" "input")))

def messageFn (j : Json) : V Json := do
  pure (hexJ (← message (← SessionCommitment.ofJson (← J.get j "commitment" "input")) (← J.str j "ended" "input") (← rounds j)))

def blockDigestFn (j : Json) : V Json := do pure (hexJ (blockDigest (← hex j "block")))

def mcMessageFn (j : Json) : V Json := do
  pure (hexJ (← memoryChallengerMessage (← hex j "I") (← hex j "session") (← hex j "record")))

/-- `pous_test_signer.TestSigner(secret, height)`'s key: `keygen` at `(64, 1, height)` of the first 64 bytes of
`SHA-512("verity-pous/test-signer/v0" ‖ secret)`. -/
def testKey (secret : ByteArray) (height : Nat) : E Lms.PrivateKey := do
  if ¬(1 ≤ height ∧ height ≤ 10) then throw s!"a test signer's height is in range(1, 11), not {height}"
  Lms.keygen ⟨64, 1, height⟩ ((Sha512.hash ("verity-pous/test-signer/v0".toUTF8 ++ secret)).extract 0 64)

/-- A test signer's key and its signature of `record` for `session` at leaf `q`, as `LmsSigner.sign` makes it with its
leaf state at `q`. -/
def signAtFn (j : Json) : V Json := do
  let k ← testKey (← hex j "secret") (← J.nat j "height" "input")
  let m ← memoryChallengerMessage k.I (← hex j "session") (← hex j "record")
  let s ← Lms.signAt k (← J.nat j "q" "input") m
  pure (Json.mkObj [("public_key", hexJ k.publicKey), ("signature", hexJ s.encode)])

/-- `SCHEMES[scheme]` or `pous_test_signer.SCHEMES[scheme]` on `(public_key, session, record, signature)`. -/
def lmsVerifyFn (j : Json) : V Json := do
  let scheme ← J.str j "scheme" "input"
  let some p := (SCHEMES ++ TEST_SCHEMES).lookup scheme | throw s!"no scheme {scheme}"
  pure (.bool (← lmsVerify p (← hex j "public_key") (← hex j "session") (← hex j "record") (← hex j "signature")))

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("coin_digest", coinDigestFn), ("coin_tree", coinTreeFn), ("verify_coin", verifyCoinFn),
  ("records_root", recordsRootFn), ("header", headerFn), ("message", messageFn), ("block_digest", blockDigestFn),
  ("mc_message", mcMessageFn), ("sign_at", signAtFn), ("lms_verify", lmsVerifyFn)]

end MemoryChallenger.Difftest
