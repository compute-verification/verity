import CoreProtocol.Cli
import CoreProtocol.Ratio
import PousCheck.Certificate
import PousCheck.Run

/-!
# A PoUS run as a core protocol

`PousCheck.Run.checkAt` behind the core interface (`Core.Protocol`), PoUS's component of onsite Verity
(`verity/core/protocols/onsite`), each byte string canonical JSON:

* params `{"point": {"k", "limit_ns", "params": {"blocks_per_segment", "c0", "segments", "w"}}, "session_id": hex,
  "trusted": [hex, …]}`: the point the run is checked at, the id the auditor designated for the period's session, and
  the verifier's list of memory-challenger keys, which never comes from the run's record;
* the commitment: the setup's public outputs (`SetupProtocol.Public.toJson`), which the prover published before any
  challenge; its digest SHA-512 of those bytes;
* the evidence: `{"digests": hex, "session": Run.ChallengerRecord.toJson, "setup": {"commitment": hex,
  "evidence": hex}}`, the memory challenger's table of `C*`, the session it signed, and the setup's own commitment and
  evidence as its case carries them (`SetupProtocol`'s `w_commitment ‖ salt` and `C*` framed);
* the check: `accepts`, the verifier's whole decision, its first refusal the reason: setup accepts `C*` against the
  server's commitment and salt at the point's parameters and publishes exactly the commitment (`tie`), and `checkAt`
  accepts the run against it. A run committed to any other setup, or to outputs no accepted setup published, is refused.
  At P2's deployed point under S1's LMS, `accepts` is `SetupProtocol.protocol.check q c C = .accept pub ()` and
  `Run.check trusted pub digests r sid = .ok ()` (`accepts_deployed`), the body of `Pous.EndToEnd.Accepted`, the
  decision `Pous.Period.PeriodHeld` is about; `protocol_accepts` reads the protocol's acceptance as `accepts`.

On acceptance it publishes the storage it certifies and the error that holds at, `{"certified", "eps", "s_bits"}`
(`certify`): when a certificate covers the point (`certificate`: P2's registered one, `Certificate.p2`, at its
`(w, c0, blocks per segment)` and segment count, else P2's proved family, `familyCertificate`, at the registered width
and prime for `2^8` to `2^23` blocks, else the per-period guarantee, `wideCertificate`, there for `2^19` to `2^27`
blocks; `k` at least its own and `Δ + RTT` within its rounds), `s_bits` is `⌊ρ·blocks·w⌋` and `eps` is
`δ + 2^eps_log2`: a node holding fewer bits when the challenges arrived passes with probability at most `eps`, under
the certificate's claims. Off it, `s_bits` is 0 and `eps` 0, the vacuous statement, and the profile names each gap.
At Demo B's store, 1038 segments of `2^16` blocks, that is `Pous.Period.PeriodHeldDemo`'s 1,060,008,348,186 bits at
`1% + 2^-128`.

`protocol` (`verity/pous/p2/v4/run`) accepts S1's LMS (`SCHEMES`) only; `testProtocol`, under an id of its own, the test
signers' schemes and only them.
-/

namespace PousCheck.RunProtocol

open Lean (Json JsonNumber)
open Flock MemoryChallenger

def ID : String := "verity/pous/p2/v4/run"

structure Params where
  point : Run.Point
  sessionId : ByteArray
  trusted : Array ByteArray

private def natJ (n : Nat) : Json := .num (JsonNumber.fromNat n)
private def hexJ (b : ByteArray) : Json := .str (hexOf b)

def pointJson (pt : Run.Point) : Json :=
  let q := pt.params
  Json.mkObj [("k", natJ pt.k), ("limit_ns", natJ pt.limitNs), ("params", Json.mkObj [
    ("blocks_per_segment", natJ q.blocksPerSegment), ("c0", natJ q.c0), ("segments", natJ q.segments), ("w", natJ q.w)])]

def Params.toJson (p : Params) : Json :=
  Json.mkObj [("point", pointJson p.point), ("session_id", hexJ p.sessionId), ("trusted", .arr (p.trusted.map hexJ))]

def decodeParams : Json → V Params :=
  exactly (fun j => do
    pure ⟨← Run.Point.ofJson (← J.get j "point" "params"), ← hexField j "session_id" "params", ← hexArr j "trusted" "params"⟩)
    Params.toJson

def decodeCommitment (b : ByteArray) : V SetupProtocol.Public := do Run.decodeSetup (← Core.canonicalJson b)

/-- The setup's own inputs, as its case carries them: the server's commitment to `W` with the verifier's salt
(`SetupProtocol.Commitment`), and `C*` segment by segment. -/
structure SetupEvidence where
  commitment : SetupProtocol.Commitment
  store : Array (Array ByteArray)

def SetupEvidence.toJson (s : SetupEvidence) : Json :=
  Json.mkObj [("commitment", hexJ (s.commitment.w ++ s.commitment.salt)),
    ("evidence", hexJ (SetupProtocol.encodeEvidence s.store))]

/-- The setup's commitment and evidence, each read by `SetupProtocol`'s own decoder. -/
def SetupEvidence.ofJson (j : Json) : V SetupEvidence := do
  let ctx := "evidence.setup"
  let c ← match SetupProtocol.commitment (← hexField j "commitment" ctx) with
    | .ok c => pure c
    | .error e => throw s!"{ctx}.commitment: {e}"
  let store ← match SetupProtocol.evidence (← hexField j "evidence" ctx) with
    | .ok s => pure s
    | .error e => throw s!"{ctx}.evidence: {e}"
  pure ⟨c, store⟩

structure Evidence where
  digests : ByteArray
  session : Run.ChallengerRecord
  setup : SetupEvidence

def Evidence.toJson (e : Evidence) : Json :=
  Json.mkObj [("digests", hexJ e.digests), ("session", e.session.toJson), ("setup", e.setup.toJson)]

def decodeEvidence (b : ByteArray) : V Evidence := do
  exactly (fun j => do pure ⟨← hexField j "digests" "evidence",
      ← Run.ChallengerRecord.ofJson (← J.get j "session" "evidence"),
      ← SetupEvidence.ofJson (← J.get j "setup" "evidence")⟩)
    Evidence.toJson (← Core.canonicalJson b)

/-- SHA-512 of the setup's public outputs as canonical JSON, the bytes they decoded from. -/
def digest (pub : SetupProtocol.Public) : ByteArray :=
  match canon pub.toJson with
  | .ok s => Sha512.hash s.toUTF8
  | .error _ => ByteArray.empty

/-- `p2.family_certificate`: `P2SlackFamilyUncond`'s `λ = 128` row for a store at the registered width and prime of any
block count from `2^8` to `2^23` (`p2.CERT_FAMILY_BLOCKS`), in any number of segments: the theorem is about
`m1p B 16448 (2^16448 − 21065) b` for every such `B`. -/
def familyCertificate (w c0 blocks : Nat) : Option Certificate.CapacityBound :=
  if w == Certificate.POINT.1 && c0 == Certificate.POINT.2.1 && 2 ^ 8 ≤ blocks && blocks ≤ 2 ^ 23 then
    some { k := Certificate.CERT_K, rounds := 0, work := 2 ^ 20, callNs := Certificate.ROOT_CALL_NS,
           theoremName := "Pous.SecurityProofs.P2SlackFamilyUncond", segments := none, claims := Certificate.P2_CLAIMS }
  else none

/-- `p2.WIDE_CLAIMS`: `P2_CLAIMS`, and one claim for each premise `periodHeldAt` takes at `N` blocks
(`P2RingRoundSecureWide`, `RingDeviceAt`). -/
def WIDE_CLAIMS : List String :=
  Certificate.P2_CLAIMS ++ ["p2-ring-round-secure-wide", "pous-ring-device-at"]

/-- `p2.wide_certificate`: `Pous.Period.SecurityProofs.periodHeldAt` for a store at the registered width and prime of
any block count from `2^19` to `2^27` (`p2.CERT_WIDE_BLOCKS`), in any number of segments: an accepted period with no
break while the node held fewer than `⌊ρ·N·16448⌋` bits has probability at most `δ + ε_max`, read at `N` = the blocks. -/
def wideCertificate (w c0 blocks : Nat) : Option Certificate.CapacityBound :=
  if w == Certificate.POINT.1 && c0 == Certificate.POINT.2.1 && 2 ^ 19 ≤ blocks && blocks ≤ 2 ^ 27 then
    some { k := Certificate.CERT_K, rounds := 0, work := 2 ^ 20, callNs := Certificate.ROOT_CALL_NS,
           theoremName := "Pous.Period.SecurityProofs.periodHeldAt", segments := none, claims := WIDE_CLAIMS }
  else none

/-- `period.certificate`: P2's registered certificate (`Certificate.p2`) when the store is its point, segment count and
all, else P2's proved family's (`familyCertificate`), else the per-period guarantee's (`wideCertificate`), else none. -/
def certificate (q : Run.Point) : Option Certificate.CapacityBound :=
  let p := q.params
  let rest := (familyCertificate p.w p.c0 p.blocks).orElse fun _ => wideCertificate p.w p.c0 p.blocks
  match Certificate.p2 p.w p.c0 p.blocksPerSegment with
  | some c => if c.segments == some p.segments then some c else rest
  | none => rest

/-- What an accepted run certifies at `pt`: the certificate when it covers the point, and why it doesn't otherwise. -/
def gaps (pt : Run.Point) : Option Certificate.CapacityBound × List String :=
  let q := pt.params
  match certificate pt with
  | none => (none, [s!"P2 has no certificate at (w, c0, blocks) = ({q.w}, {q.c0}, {q.blocks})"])
  | some c => (some c, c.refusals pt.k pt.limitNs)

/-- `δ + 2^eps_log2`, a certificate's error, exactly. -/
def CapacityBound.eps (c : Certificate.CapacityBound) : Core.Ratio :=
  let e := 2 ^ (-c.epsLog2).toNat
  (Core.Ratio.mk' (c.delta.1 * e + c.delta.2) (c.delta.2 * e)).getD ⟨1, 1⟩

/-- The public outputs: whether the certificate covers the point, the error, and the storage it certifies, in bits. -/
structure Public where
  certified : Bool
  eps : Core.Ratio
  sBits : Nat

def certify (pt : Run.Point) : Public :=
  match gaps pt with
  | (some c, []) => ⟨true, CapacityBound.eps c, pt.params.blocks * pt.params.w * c.rho.1 / c.rho.2⟩
  | _ => ⟨false, ⟨0, 1⟩, 0⟩

def Public.toJson (p : Public) : Json :=
  Json.mkObj [("certified", .bool p.certified), ("eps", .str p.eps.toString), ("s_bits", natJ p.sBits)]

/-- The profile: the store and the audit the run was checked at, the certificate and every gap. -/
def profileJson (pt : Run.Point) : Json :=
  let (c, gs) := gaps pt
  Json.mkObj [("block_bits", natJ pt.params.w), ("blocks", natJ pt.params.blocks),
    ("certificate", match c with | some c => c.toJson | none => .null), ("experimental", .str Certificate.EXPERIMENTAL),
    ("gaps", .arr (gs.map Json.str).toArray), ("k", natJ pt.k), ("limit_ns", natJ pt.limitNs)]

/-- `certify` and `profileJson` at the point of the difftest's line `{"point": …}`, as `{"profile", "public"}`:
`pous-run serve`'s `certify`, which `tests/test_run_protocol.py` drives against `period.certify` and `period.profile`
at points no setup could reach, Demo B's among them. -/
def certifyFn (j : Json) : V Json := do
  let pt ← Run.Point.ofJson (← J.get j "point" "certify")
  pure (Json.mkObj [("profile", profileJson pt), ("public", (certify pt).toJson)])

deriving instance DecidableEq for SetupProtocol.Public

/-- The setup tie: setup accepts `C*` against the server's commitment and salt `c` at the point's parameters
(`SetupProtocol.check`, the setup protocol's `check`), and what it publishes is exactly `pub`, the run's commitment. -/
def tie (pt : Run.Point) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) (pub : SetupProtocol.Public) :
    Except String Unit :=
  match SetupProtocol.check pt.params c C with
  | .accept published () =>
    if published = pub then pure () else throw "the run's commitment is not what its setup published"
  | .reject reason => throw s!"setup: {reason}"
  | .undecided reason => throw s!"setup: {reason}"

/-- The verifier's whole decision on a run at `pt` under `schemes`: the setup tie, then the composed check. -/
def accepts (schemes : Schemes) (pt : Run.Point) (trusted : List ByteArray) (c : SetupProtocol.Commitment)
    (C : Array (Array ByteArray)) (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.ChallengerRecord)
    (sid : ByteArray) : Except String Unit := do
  tie pt c C pub
  Run.checkAt schemes pt trusted pub digests r sid

def check (schemes : Schemes) (p : Params) (pub : SetupProtocol.Public) (e : Evidence) :
    Core.Outcome Public Run.Point :=
  match accepts schemes p.point p.trusted.toList e.setup.commitment e.setup.store pub e.digests e.session p.sessionId with
  | .ok () => .accept (certify p.point) p.point
  | .error reason => .reject reason

theorem accepts_iff (schemes : Schemes) (pt : Run.Point) (trusted : List ByteArray) (c : SetupProtocol.Commitment)
    (C : Array (Array ByteArray)) (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.ChallengerRecord)
    (sid : ByteArray) :
    accepts schemes pt trusted c C pub digests r sid = .ok () ↔
      SetupProtocol.protocol.check pt.params c C = .accept pub () ∧
        Run.checkAt schemes pt trusted pub digests r sid = .ok () := by
  show (do tie pt c C pub; Run.checkAt schemes pt trusted pub digests r sid) = .ok () ↔
    SetupProtocol.check pt.params c C = .accept pub () ∧ _
  unfold tie
  cases SetupProtocol.check pt.params c C with
  | accept published u =>
    cases u
    by_cases e : published = pub
    · subst e
      simp only [↓reduceIte, true_and]
      rfl
    · simp only [e, ↓reduceIte, Core.Outcome.accept.injEq, false_and, iff_false]
      intro hc
      cases hc
  | reject reason =>
    simp only [reduceCtorEq, false_and, iff_false]
    intro hc
    cases hc
  | undecided reason =>
    simp only [reduceCtorEq, false_and, iff_false]
    intro hc
    cases hc

/-- **At P2's deployed point under S1's LMS the verdict is `Pous.EndToEnd.Accepted`'s**: setup accepts `C*` against `c`
and publishes `pub`, and `Run.check` accepts the run against `pub`, word for word that definition's body
(`q = P2.Params.deployed`). -/
theorem accepts_deployed (trusted : List ByteArray) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray))
    (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.ChallengerRecord) (sid : ByteArray) :
    accepts SCHEMES Run.DEPLOYED trusted c C pub digests r sid = .ok () ↔
      SetupProtocol.protocol.check P2.Params.deployed c C = .accept pub () ∧
        Run.check trusted pub digests r sid = .ok () :=
  accepts_iff SCHEMES Run.DEPLOYED trusted c C pub digests r sid

def mkProtocol (id : String) (schemes : Schemes) : Core.Protocol where
  id := id
  Params := Params
  Commitment := SetupProtocol.Public
  Evidence := Evidence
  Public := Public
  Profile := Run.Point
  params := decodeParams
  commitment := decodeCommitment
  evidence := decodeEvidence
  digest := digest
  check := check schemes
  publicJson := Public.toJson
  profileJson := profileJson

/-- `verity/pous/p2/v4/run` under S1's LMS. -/
def protocol : Core.Protocol := mkProtocol ID SCHEMES

/-- The same check under the test signers' schemes alone, its id `test/verity/pous/p2/v4/run`. -/
def testProtocol : Core.Protocol := mkProtocol (TEST_PREFIX ++ ID) TEST_SCHEMES

/-- **The protocol accepts exactly when `accepts` does**, on the case's point, keys, setup, table, session and id. -/
theorem protocol_accepts (p : Params) (pub : SetupProtocol.Public) (e : Evidence) :
    protocol.Accepts p pub e ↔
      accepts SCHEMES p.point p.trusted.toList e.setup.commitment e.setup.store pub e.digests e.session p.sessionId
        = .ok () := by
  show (∃ out prof, check SCHEMES p pub e = .accept out prof) ↔ _
  unfold check
  cases accepts SCHEMES p.point p.trusted.toList e.setup.commitment e.setup.store pub e.digests e.session p.sessionId with
  | ok u => cases u; exact ⟨fun _ => rfl, fun _ => ⟨_, _, rfl⟩⟩
  | error reason =>
    constructor
    · rintro ⟨_, _, h⟩
      cases h
    · intro h
      cases h

/-! The tie on `SetupProtocol`'s honest setup: `w = 64`, `c0 = 189`, one segment of two blocks. -/

private def TIE_POINT : Run.Point := ⟨⟨64, 189, 2, 1⟩, 1, 1000⟩
private def TIE_C : SetupProtocol.Commitment :=
  match SetupProtocol.commitment ((unhex SetupProtocol.COMMITMENT).toOption.getD .empty) with
  | .ok c => c
  | .error _ => ⟨.empty, .empty⟩
private def TIE_STORE : Array (Array ByteArray) := #[(SetupProtocol.HONEST.mapM unhex).toOption.getD #[]]
private def TIE_PUB : SetupProtocol.Public :=
  ⟨2, TIE_C.w, (unhex SetupProtocol.ROOT).toOption.getD .empty, TIE_C.salt⟩

#guard tie TIE_POINT TIE_C TIE_STORE TIE_PUB matches .ok ()
#guard tie TIE_POINT TIE_C TIE_STORE { TIE_PUB with root := Sha512.hash TIE_PUB.root } matches
  .error "the run's commitment is not what its setup published"
#guard tie TIE_POINT TIE_C TIE_STORE { TIE_PUB with blocks := 3 } matches
  .error "the run's commitment is not what its setup published"
#guard tie TIE_POINT { TIE_C with w := Sha512.hash TIE_C.w } TIE_STORE TIE_PUB matches
  .error "setup: Dec(C*) does not match the commitment to W"
#guard tie { TIE_POINT with params := ⟨64, 189, 1, 2⟩ } TIE_C TIE_STORE TIE_PUB matches
  .error "setup: segment 0 of C* is not 1 blocks of 8 bytes"

#guard (certify Run.DEPLOYED).sBits == 2 ^ 19 * 16448 * 18 / 19
#guard (certify Run.DEPLOYED).eps.toString == s!"{2 ^ 126 + 25}/{25 * 2 ^ 128}"
#guard (certify ⟨⟨64, 189, 4, 2⟩, 3, 1000⟩).sBits == 0
#guard (gaps ⟨⟨64, 189, 4, 2⟩, 3, 1000⟩).2 == ["P2 has no certificate at (w, c0, blocks) = (64, 189, 8)"]
#guard (gaps { Run.DEPLOYED with k := 104 }).2 == ["k = 104 is below the certificate's 105"]
#guard (certify { Run.DEPLOYED with params := ⟨16448, 21065, 256, 1⟩ }).sBits == 256 * 16448 * 18 / 19
#guard (certify { Run.DEPLOYED with params := ⟨16448, 21065, 256, 1⟩ }).eps.toString == s!"{2 ^ 126 + 25}/{25 * 2 ^ 128}"
#guard (certify { Run.DEPLOYED with params := ⟨16448, 21065, 2 ^ 16, 2⟩ }).certified
#guard !(certify { Run.DEPLOYED with params := ⟨16448, 21065, 255, 1⟩ }).certified
#guard ((certificate Run.DEPLOYED).map (·.segments)) == some (some 8)
#guard ((certificate { Run.DEPLOYED with params := ⟨16448, 21065, 2 ^ 16, 128⟩ }).map (·.theoremName)) ==
  some "Pous.SecurityProofs.P2SlackFamilyUncond"
#guard ((certificate { Run.DEPLOYED with params := ⟨16448, 21065, 2 ^ 16, 129⟩ }).map (·.theoremName)) ==
  some "Pous.Period.SecurityProofs.periodHeldAt"
#guard (certify { Run.DEPLOYED with params := ⟨16448, 21065, 2 ^ 16, 1038⟩ }).sBits == 1060008348186
#guard (certify { Run.DEPLOYED with params := ⟨16448, 21065, 2 ^ 16, 1038⟩ }).eps.toString ==
  s!"{2 ^ 126 + 25}/{25 * 2 ^ 128}"
#guard (gaps { Run.DEPLOYED with params := ⟨16448, 21065, 2 ^ 16, 1038⟩, k := 104 }).2 ==
  ["k = 104 is below the certificate's 105"]
#guard (certify { Run.DEPLOYED with params := ⟨16448, 21065, 2 ^ 16, 2 ^ 11⟩ }).certified
#guard !(certify { Run.DEPLOYED with params := ⟨16448, 21065, 2 ^ 16 - 1, 2 ^ 11 + 1⟩ }).certified
#guard (gaps { Run.DEPLOYED with params := ⟨16448, 21065, 2 ^ 16, 2 ^ 11 + 1⟩ }).2 ==
  [s!"P2 has no certificate at (w, c0, blocks) = (16448, 21065, {2 ^ 27 + 2 ^ 16})"]
#guard (protocol.answer "{}".toUTF8 ByteArray.empty ByteArray.empty).reason.map (·.startsWith "params: ") == some true

end PousCheck.RunProtocol
