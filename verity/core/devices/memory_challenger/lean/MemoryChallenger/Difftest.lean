import Difftest
import Lms.Sign
import Attest.Period
import MemoryChallenger.Protocol
import MemoryChallenger.WipeJudge

/-!
# The difftest's Lean half

The functions `tests/` holds to `memory_challenger.py` and `streamed_wipe.py` through `verity_difftest.Port`, each over JSON
in the formats of `MemoryChallenger.Protocol` (byte strings lowercase hex, integers naturals) and
`MemoryChallenger.WipeJudge`: the crypto and record formats, LMS signing at a leaf and verification, the streamed wipe's
signed message and egress count, and a device's HSS role digests and checks (`Lms.Hss`, `Attest.Period`: the
`range-u64` domain id, the period tree's root, the verifier's acceptance of a period and the liveness check).
`memory-challenger-difftest` serves them (`Difftest.serve`). Each verdict's own difftest drives the executable its callers
run, `memory-challenger-verdict` and `streamed-wipe-verdict`.
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

/-- `streamed_wipe.message(commitment, ended, fill_start_ns, fill_end_ns, batches, egress_bits)`. -/
def wipeMessageFn (j : Json) : V Json := do
  let c ← StreamedWipe.Commitment.ofJson (← J.get j "commitment" "input")
  let batches ← (← J.arr j "batches" "input").mapM StreamedWipe.Batch.ofJson
  let intAt := fun (k : String) => do J.asInt (← J.get j k "input") k
  pure (hexJ (← StreamedWipe.message c (← J.str j "ended" "input") (← intAt "fill_start_ns") (← intAt "fill_end_ns")
    batches (← intAt "egress_bits")))

/-- `streamed_wipe.egress_units(egress_bits)`. -/
def wipeEgressUnitsFn (j : Json) : V Json := do
  pure (Lean.toJson (StreamedWipe.egressUnits (← J.asInt (← J.get j "egress_bits" "input") "egress_bits")))

/-- `lms.party_key_message(I, deployment, party, lower)`. -/
def partyKeyMessageFn (j : Json) : V Json := do
  pure (hexJ (← Lms.partyKeyMessage (← hex j "I") (← hex j "deployment") (← J.str j "party" "input") (← hex j "lower")))

/-- `lms.period_root_message(I, deployment, party, attachment, link, period, count, root)`. -/
def periodRootMessageFn (j : Json) : V Json := do
  pure (hexJ (← Lms.periodRootMessage (← hex j "I") (← hex j "deployment") (← J.str j "party" "input")
    (← hex j "attachment") (← J.str j "link" "input") (← J.nat j "period" "input") (← J.nat j "count" "input")
    (← hex j "root")))

/-- `lms.liveness_message(I, deployment, party, attachment, link, period, block)`. -/
def livenessMessageFn (j : Json) : V Json := do
  pure (hexJ (← Lms.livenessMessage (← hex j "I") (← hex j "deployment") (← J.str j "party" "input")
    (← hex j "attachment") (← J.str j "link" "input") (← J.nat j "period" "input") (← J.nat j "block" "input")))

/-- `lms.Params(64, W, H)`. -/
private def params (j : Json) : V Lms.Params := do Lms.Params.make 64 (← J.nat j "W" "input") (← J.nat j "H" "input") false

/-- `lms.hss_verify(Params(64, W, H), identity, deployment, party, HssSignature.decode(signature), message)`. -/
def hssVerifyFn (j : Json) : V Json := do
  let p ← params j
  let s ← Lms.HssSignature.decode p (← hex j "signature")
  pure (.bool (← Lms.hssVerify p (← hex j "identity") (← hex j "deployment") (← J.str j "party" "input") s
    (← hex j "message")))

/-- `merkle.CommitmentDomain(binding, owner, RangeU64Domain(0, count), "sha512").domain_id` (`Attest.domainU64`), refused
where `merkle` refuses: a binding not 64 bytes, `owner < −2`, or `owner + 2` or `count` past a u64. -/
def domainU64Fn (j : Json) : V Json := do
  let binding ← hex j "binding"
  let owner ← J.asInt (← J.get j "owner" "input") "owner"
  let count ← J.nat j "count" "input"
  if binding.size ≠ 64 then throw "a range-u64 domain's binding is 64 bytes"
  if owner < -2 ∨ owner + 2 ≥ 2 ^ 64 then throw "owner is at least -2 and owner + 2 a u64"
  if count ≥ 2 ^ 64 then throw "a range-u64 domain's count is a u64"
  pure (hexJ (Attest.domainU64 binding owner count).id)

/-- `attest.root(records)`. -/
def attestRootFn (j : Json) : V Json := do pure (hexJ (Attest.root (← hexes j "records")))

private def readOf (x : Json) : V Attest.Read := do
  pure ⟨← J.nat x "position" "read", ← hex x "record", ← hexes x "path"⟩

/-- `Attest.accepts` at `Params(64, W, H)`: the party's lower key and its certificate, the signed root with the lower
key's signature of it, each signature decoded at the parameters, and the reads. -/
def attestCheckFn (j : Json) : V Json := do
  let p ← params j
  let k : Attest.PartyKey := ⟨← hex j "lower", ← Lms.Signature.decode p (← hex j "certificate")⟩
  let s : Attest.SignedRoot := ⟨← hex j "attachment", ← J.str j "link" "input", ← J.nat j "period" "input",
    ← J.nat j "count" "input", ← hex j "root", ← Lms.Signature.decode p (← hex j "signature")⟩
  let reads ← (← J.arr j "reads" "input").mapM readOf
  pure (.bool (← Attest.accepts p (← hex j "identity") (← hex j "deployment") (← J.str j "party" "input") k s reads))

/-- `Attest.checkLiveness` at `Params(64, W, H)`. -/
def attestLivenessFn (j : Json) : V Json := do
  pure (.bool (← Attest.checkLiveness (← params j) (← hex j "identity") (← hex j "deployment")
    (← J.str j "party" "input") (← hex j "attachment") (← J.str j "link" "input") (← J.nat j "period" "input")
    (← J.nat j "block" "input") (← hex j "signature")))

def table : String → Option _root_.Difftest.Fn := _root_.Difftest.table [
  ("coin_digest", coinDigestFn), ("coin_tree", coinTreeFn), ("verify_coin", verifyCoinFn),
  ("records_root", recordsRootFn), ("header", headerFn), ("message", messageFn), ("block_digest", blockDigestFn),
  ("mc_message", mcMessageFn), ("sign_at", signAtFn), ("lms_verify", lmsVerifyFn), ("wipe_message", wipeMessageFn),
  ("wipe_egress_units", wipeEgressUnitsFn), ("party_key_message", partyKeyMessageFn),
  ("period_root_message", periodRootMessageFn), ("liveness_message", livenessMessageFn), ("hss_verify", hssVerifyFn),
  ("domain_u64", domainU64Fn), ("attest_root", attestRootFn), ("attest_check", attestCheckFn),
  ("attest_liveness", attestLivenessFn)]

end MemoryChallenger.Difftest
