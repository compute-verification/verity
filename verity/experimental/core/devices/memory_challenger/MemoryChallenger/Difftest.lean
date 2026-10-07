import Difftest
import MemoryChallenger.Protocol
import MemoryChallenger.Step

/-!
# The difftest's Lean half

The functions `tests/` holds to `memory_challenger.py` through `verity_difftest.Port`, each over JSON in the formats of
`MemoryChallenger.Protocol` (byte strings lowercase hex, integers naturals): the crypto and record formats, and the
verdict with its reasons. `memory-challenger-difftest` serves them (`Difftest.serve`).
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

/-- `verdict` under the test signers' schemes, on a protocol case's decoded parameters, commitment and evidence: the
whole `Verdict`, reasons and counts too. -/
def verdictFn (j : Json) : V Json := do
  let p ← decodeParams (← J.get j "params" "input")
  let c ← decodeCommitment (← unhex (← J.str j "commitment" "input"))
  let e ← decodeEvidence (← unhex (← J.str j "evidence" "input"))
  let v := verdict TEST_SCHEMES p.trusted.toList (fun i => p.expected[i]?) e.session c e.openings
  pure (Json.mkObj [("accepted", .bool v.accepted), ("reasons", .arr (v.reasons.map Json.str)),
    ("late", Lean.toJson v.late), ("wrong", Lean.toJson v.wrong)])

private def event (j : Json) : V Event := do
  if let .ok now := j.getObjValAs? Nat "tick" then return .tick now
  if let .ok r := j.getObjVal? "received" then return .received (← J.nat r "now" "received") (← hex r "bytes")
  if (j.getObjVal? "closed").isOk then return .closed
  if let .ok q := j.getObjValAs? Nat "reserved" then return .reserved (.ok q)
  if let .ok e := j.getObjValAs? String "refused" then return .reserved (.error e)
  throw "an event is {tick}, {received: {now, bytes}}, {closed}, {reserved} or {refused}"

def outJ : Out → Json
  | .commitment c => Json.mkObj [("commitment", c.toJson)]
  | .session s os => Json.mkObj [("openings", .arr (os.map CoinOpening.toJson)), ("session", s.toJson)]

private def actionJ : Action → Json
  | .send b => Json.mkObj [("send", hexJ b)]
  | .publish o => Json.mkObj [("publish", outJ o)]
  | .reserve => Json.mkObj [("reserve", .null)]

/-- A session's `Config`: its coins `CoinTree.draw`n, its signer a test signer's key (`testKey`). -/
def configOf (c : Json) : V Config := do
  let coins ← CoinTree.draw (← hex c "source") (← hex c "session") (← J.nat c "blocks" "config") (← J.nat c "k" "config")
    (← hex c "key") (← hexes c "salts")
  let height ← J.nat c "height" "config"
  Config.make coins (← J.nat c "limit_ns" "config") (← J.nat c "block_bytes" "config") (← hex c "binding")
    (← testKey (← hex c "secret") height) (testScheme height)

/-- `step` over a session's events: each event's actions, and whether the session ended unsigned. -/
def stepFn (j : Json) : V Json := do
  let cfg ← configOf (← J.get j "config" "input")
  let (s, out) := run cfg (← (← J.arr j "events" "input").toList.mapM event)
  pure (Json.mkObj [("actions", .arr (out.map fun acts => .arr (acts.map actionJ).toArray)),
    ("refused", .bool s.refusal.isSome)])

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("coin_digest", coinDigestFn), ("coin_tree", coinTreeFn), ("verify_coin", verifyCoinFn),
  ("records_root", recordsRootFn), ("header", headerFn), ("message", messageFn), ("block_digest", blockDigestFn),
  ("mc_message", mcMessageFn), ("sign_at", signAtFn), ("lms_verify", lmsVerifyFn), ("verdict", verdictFn),
  ("step", stepFn)]

end MemoryChallenger.Difftest
