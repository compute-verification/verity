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
* the evidence: `{"digests": hex, "session": Run.Session.toJson, "setup": S}`, the memory challenger's table of `C*`,
  the session it signed, and the setup in one of two forms (`SetupCase`):
  - inline, `S = {"commitment": hex, "evidence": hex}`: the setup's own commitment and evidence as its case carries
    them (`SetupProtocol`'s `w_commitment ‖ salt` and `C*` framed), which the check decodes again;
  - cited, `S = {"answer": Core.Answer.toJson, "params": manifest, "record": id}` (`SetupRef`): the setup check's
    recorded answer, the parameters it checked at (`SetupProtocol.params`' manifest) and where the record is (a run or
    `art:` id), so a store too large to decode every period (Demo B's 130 GiB) is checked once, at setup;
* the check: `decision`, the verifier's whole decision, its first refusal the reason. Inline it is `accepts`: setup
  accepts `C*` against the server's commitment and salt at the point's parameters and publishes exactly the commitment
  (`tie`), and `checkAt` accepts the run against it. Cited it is `acceptsCited`: the cited answer is the setup
  protocol's acceptance, at the point's parameters, publishing exactly the commitment (`cite`; a setup that refused is
  refused with its reason), and `checkAt` accepts the run against it, reading no block of `C*`. A run committed to any
  other setup, or to outputs no accepted setup published, is refused. At P2's deployed point under S1's LMS, `accepts`
  is `SetupProtocol.protocol.check q c C = .accept pub ()` and `Run.check trusted pub digests r sid = .ok ()`
  (`accepts_deployed`), the body of `Pous.EndToEnd.Accepted`, the decision `Pous.Period.PeriodHeld` is about, and
  `acceptsCited` is `cite` and `Run.check` (`acceptsCited_deployed`): with the acceptance the citation records it is
  `Accepted` again (`Pous.Cited.CitedRunAccepted`), and every store at the published root decodes to the committed `W`
  or the trees hold a collision (`Pous.Cited.RootDecodes`). `protocol_accepts` reads the protocol's acceptance as
  `decision`.

On acceptance it publishes the storage it certifies and the error that holds at, `{"certified", "eps", "s_bits"}`
(`certify`): when a certificate covers the point (`certificate`: P2's registered one, `Certificate.p2`, at its
`(w, c0, blocks per segment)` and segment count, else P2's proved family, `familyCertificate`, at the registered width
and prime for `2^8` to `2^23` blocks; `k` at least its own and `Δ + RTT` within its rounds), `s_bits` is
`⌊ρ·blocks·w⌋` and `eps` is `δ + 2^eps_log2`: a node holding fewer bits when the challenges arrived passes with
probability at most `eps`, under the certificate's claims. Off it, `s_bits` is 0 and `eps` 0, the vacuous statement,
and the profile names each gap.

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

/-- A citation of the setup check's recorded acceptance, in place of its inputs: the answer `verity/pous/p2/v4/setup`
returned (`Core.Answer`), the parameters it checked at, and where its record is (a run or `art:` id). -/
structure SetupRef where
  answer : Core.Answer
  params : P2.Params
  record : String

def SetupRef.toJson (s : SetupRef) : Json :=
  Json.mkObj [("answer", s.answer.toJson), ("params", Commit.manifestJson s.params), ("record", .str s.record)]

/-- The answer read by `Core.Answer.ofJson`, the parameters by `SetupProtocol.params` (valid, and exactly their
manifest). -/
def SetupRef.ofJson (j : Json) : V SetupRef := do
  let ctx := "evidence.setup"
  let params ← match SetupProtocol.params (← J.get j "params" ctx) with
    | .ok q => pure q
    | .error e => throw s!"{ctx}.params: {e}"
  pure ⟨← Core.Answer.ofJson (← J.get j "answer" ctx) s!"{ctx}.answer", params, ← J.str j "record" ctx⟩

/-- How a run's case gives its setup: the setup's own inputs (`inline`), or a citation of the setup check's recorded
acceptance (`cited`). -/
inductive SetupCase where
  | inline (s : SetupEvidence)
  | cited (s : SetupRef)

def SetupCase.toJson : SetupCase → Json
  | .inline s => s.toJson
  | .cited s => s.toJson

/-- A citation when the setup has an `answer`, else the setup's own inputs. -/
def SetupCase.ofJson (j : Json) : V SetupCase :=
  match j.getObjVal? "answer" with
  | .ok _ => return .cited (← SetupRef.ofJson j)
  | .error _ => return .inline (← SetupEvidence.ofJson j)

structure Evidence where
  digests : ByteArray
  session : Run.Session
  setup : SetupCase

def Evidence.toJson (e : Evidence) : Json :=
  Json.mkObj [("digests", hexJ e.digests), ("session", e.session.toJson), ("setup", e.setup.toJson)]

def decodeEvidence (b : ByteArray) : V Evidence := do
  exactly (fun j => do pure ⟨← hexField j "digests" "evidence", ← Run.Session.ofJson (← J.get j "session" "evidence"),
      ← SetupCase.ofJson (← J.get j "setup" "evidence")⟩)
    Evidence.toJson (← Core.canonicalJson b)

/-- SHA-512 of the setup's public outputs as canonical JSON, the bytes they decoded from. -/
def digest (pub : SetupProtocol.Public) : ByteArray :=
  match canon pub.toJson with
  | .ok s => Sha512.hash s.toUTF8
  | .error _ => ByteArray.empty

/-- `p2.family_certificate`: `P2SlackFamilyUncond`'s `λ = 128` row for a store at the registered width and prime of any
block count from `2^8` to `2^23` (`p2.CERT_FAMILY_BLOCKS`), in any number of segments: the theorem is about
`m1p B 16448 (2^16448 − 21065) b` for every such `B`. -/
def familyCertificate (w c0 blocks : Nat) : Option Certificate.Certificate :=
  if w == Certificate.POINT.1 && c0 == Certificate.POINT.2.1 && 2 ^ 8 ≤ blocks && blocks ≤ 2 ^ 23 then
    some { k := Certificate.CERT_K, rounds := 0, work := 2 ^ 20, callNs := Certificate.ROOT_CALL_NS,
           theoremName := "Pous.SecurityProofs.P2SlackFamilyUncond", segments := none, claims := Certificate.P2_CLAIMS }
  else none

/-- `period.certificate`: P2's registered certificate (`Certificate.p2`) when the store is its point, segment count and
all, else P2's proved family's (`familyCertificate`), else none. -/
def certificate (q : Run.Point) : Option Certificate.Certificate :=
  let p := q.params
  match Certificate.p2 p.w p.c0 p.blocksPerSegment with
  | some c => if c.segments == some p.segments then some c else familyCertificate p.w p.c0 p.blocks
  | none => familyCertificate p.w p.c0 p.blocks

/-- What an accepted run certifies at `pt`: the certificate when it covers the point, and why it doesn't otherwise. -/
def gaps (pt : Run.Point) : Option Certificate.Certificate × List String :=
  let q := pt.params
  match certificate pt with
  | none => (none, [s!"P2 has no certificate at (w, c0, blocks) = ({q.w}, {q.c0}, {q.blocks})"])
  | some c => (some c, c.refusals pt.k pt.limitNs)

/-- `δ + 2^eps_log2`, a certificate's error, exactly. -/
def Certificate.eps (c : Certificate.Certificate) : Core.Ratio :=
  let e := 2 ^ (-c.epsLog2).toNat
  (Core.Ratio.mk' (c.delta.1 * e + c.delta.2) (c.delta.2 * e)).getD ⟨1, 1⟩

/-- The public outputs: whether the certificate covers the point, the error, and the storage it certifies, in bits. -/
structure Public where
  certified : Bool
  eps : Core.Ratio
  sBits : Nat

def certify (pt : Run.Point) : Public :=
  match gaps pt with
  | (some c, []) => ⟨true, Certificate.eps c, pt.params.blocks * pt.params.w * c.rho.1 / c.rho.2⟩
  | _ => ⟨false, ⟨0, 1⟩, 0⟩

def Public.toJson (p : Public) : Json :=
  Json.mkObj [("certified", .bool p.certified), ("eps", .str p.eps.toString), ("s_bits", natJ p.sBits)]

/-- The profile: the store and the audit the run was checked at, the certificate and every gap. -/
def profileJson (pt : Run.Point) : Json :=
  let (c, gs) := gaps pt
  Json.mkObj [("block_bits", natJ pt.params.w), ("blocks", natJ pt.params.blocks),
    ("certificate", match c with | some c => c.toJson | none => .null), ("experimental", .str Certificate.EXPERIMENTAL),
    ("gaps", .arr (gs.map Json.str).toArray), ("k", natJ pt.k), ("limit_ns", natJ pt.limitNs)]

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
    (C : Array (Array ByteArray)) (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session)
    (sid : ByteArray) : Except String Unit := do
  tie pt c C pub
  Run.checkAt schemes pt trusted pub digests r sid

deriving instance DecidableEq for P2.Params

/-- The public outputs an answer accepted with, read as the setup's: by `SetupProtocol.Public.ofJson`, and exactly
what `toJson` writes. -/
def publishedOf (a : Core.Answer) : Option SetupProtocol.Public :=
  a.published.bind fun j => (exactly SetupProtocol.Public.ofJson SetupProtocol.Public.toJson j).toOption

/-- The citation: it names its record, and its answer is the setup protocol's acceptance at the point's parameters,
its digest `w ‖ salt` and its public outputs exactly `pub`, the run's commitment (the server's commitment to `W`, the
salt, the root and the block count). A setup that did not accept is refused with its reason. -/
def cite (pt : Run.Point) (pub : SetupProtocol.Public) (s : SetupRef) : Except String Unit := do
  if s.record.isEmpty then throw "the citation names no record of the setup check"
  if s.answer.protocol ≠ SetupProtocol.ID then
    throw s!"the cited answer is {s.answer.protocol}'s, not the setup check's"
  if s.answer.verdict ≠ .accept then throw s!"the cited setup did not accept: {s.answer.reason.getD ""}"
  if s.params ≠ pt.params then throw "the cited setup was checked at other parameters than the run's"
  if s.answer.digest ≠ some (pub.commitment ++ pub.salt) ∨ publishedOf s.answer ≠ some pub then
    throw "the run's commitment is not what its setup published"

/-- The verifier's decision on a run whose case cites the setup's acceptance: the citation, then the composed check. -/
def acceptsCited (schemes : Schemes) (pt : Run.Point) (trusted : List ByteArray) (s : SetupRef)
    (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session) (sid : ByteArray) : Except String Unit := do
  cite pt pub s
  Run.checkAt schemes pt trusted pub digests r sid

/-- The verifier's whole decision on a case, in the setup's form: `accepts` on its own inputs, `acceptsCited` on a
citation. -/
def decision (schemes : Schemes) (p : Params) (pub : SetupProtocol.Public) (e : Evidence) : Except String Unit :=
  match e.setup with
  | .inline s => accepts schemes p.point p.trusted.toList s.commitment s.store pub e.digests e.session p.sessionId
  | .cited s => acceptsCited schemes p.point p.trusted.toList s pub e.digests e.session p.sessionId

def check (schemes : Schemes) (p : Params) (pub : SetupProtocol.Public) (e : Evidence) :
    Core.Outcome Public Run.Point :=
  match decision schemes p pub e with
  | .ok () => .accept (certify p.point) p.point
  | .error reason => .reject reason

theorem accepts_iff (schemes : Schemes) (pt : Run.Point) (trusted : List ByteArray) (c : SetupProtocol.Commitment)
    (C : Array (Array ByteArray)) (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session)
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
    (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session) (sid : ByteArray) :
    accepts SCHEMES Run.DEPLOYED trusted c C pub digests r sid = .ok () ↔
      SetupProtocol.protocol.check P2.Params.deployed c C = .accept pub () ∧
        Run.check trusted pub digests r sid = .ok () :=
  accepts_iff SCHEMES Run.DEPLOYED trusted c C pub digests r sid

theorem acceptsCited_iff (schemes : Schemes) (pt : Run.Point) (trusted : List ByteArray) (s : SetupRef)
    (pub : SetupProtocol.Public) (digests : ByteArray) (r : Run.Session) (sid : ByteArray) :
    acceptsCited schemes pt trusted s pub digests r sid = .ok () ↔
      cite pt pub s = .ok () ∧ Run.checkAt schemes pt trusted pub digests r sid = .ok () := by
  unfold acceptsCited
  cases cite pt pub s with
  | ok u =>
    cases u
    exact ⟨fun h => ⟨rfl, h⟩, fun h => h.2⟩
  | error e =>
    constructor
    · intro h
      change Except.error e = _ at h
      cases h
    · intro h
      cases h.1

/-- **At P2's deployed point under S1's LMS the cited form's verdict is the citation and `Run.check`**, the second
conjunct of `Pous.EndToEnd.Accepted`; the first is the acceptance the citation records. -/
theorem acceptsCited_deployed (trusted : List ByteArray) (s : SetupRef) (pub : SetupProtocol.Public)
    (digests : ByteArray) (r : Run.Session) (sid : ByteArray) :
    acceptsCited SCHEMES Run.DEPLOYED trusted s pub digests r sid = .ok () ↔
      cite Run.DEPLOYED pub s = .ok () ∧ Run.check trusted pub digests r sid = .ok () :=
  acceptsCited_iff SCHEMES Run.DEPLOYED trusted s pub digests r sid

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

/-- **The protocol accepts exactly when `decision` does**, on the case's point, keys, setup, table, session and id. -/
theorem protocol_accepts (p : Params) (pub : SetupProtocol.Public) (e : Evidence) :
    protocol.Accepts p pub e ↔ decision SCHEMES p pub e = .ok () := by
  show (∃ out prof, check SCHEMES p pub e = .accept out prof) ↔ _
  unfold check
  cases decision SCHEMES p pub e with
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

/-! The citation of that setup's acceptance: the setup protocol's answer on its case, and on the case with the commitment
to `W` one bit off, which it refuses. -/

private def setupAnswer (commitment : String) : Core.Answer :=
  SetupProtocol.protocol.answer ((Commit.manifest TIE_POINT.params).toOption.getD .empty)
    ((unhex commitment).toOption.getD .empty) (SetupProtocol.encodeEvidence TIE_STORE)
private def TIE_REF : SetupRef := ⟨setupAnswer SetupProtocol.COMMITMENT, TIE_POINT.params, "art:setup-record"⟩

#guard (setupAnswer SetupProtocol.COMMITMENT).verdict == .accept
#guard cite TIE_POINT TIE_PUB TIE_REF matches .ok ()
#guard cite TIE_POINT { TIE_PUB with root := Sha512.hash TIE_PUB.root } TIE_REF matches
  .error "the run's commitment is not what its setup published"
#guard cite TIE_POINT { TIE_PUB with blocks := 3 } TIE_REF matches
  .error "the run's commitment is not what its setup published"
#guard cite TIE_POINT TIE_PUB { TIE_REF with answer := { TIE_REF.answer with digest := some TIE_C.w } } matches
  .error "the run's commitment is not what its setup published"
#guard cite TIE_POINT TIE_PUB { TIE_REF with answer := setupAnswer ("6c" ++ SetupProtocol.COMMITMENT.drop 2) } matches
  .error "the cited setup did not accept: Dec(C*) does not match the commitment to W"
#guard cite TIE_POINT TIE_PUB { TIE_REF with answer := { TIE_REF.answer with protocol := ID } } matches
  .error "the cited answer is verity/pous/p2/v4/run's, not the setup check's"
#guard cite { TIE_POINT with params := ⟨64, 189, 1, 2⟩ } TIE_PUB TIE_REF matches
  .error "the cited setup was checked at other parameters than the run's"
#guard cite TIE_POINT TIE_PUB { TIE_REF with record := "" } matches
  .error "the citation names no record of the setup check"
#guard (SetupCase.ofJson (SetupRef.toJson TIE_REF)) matches .ok (.cited _)
#guard (SetupCase.ofJson (SetupEvidence.toJson ⟨TIE_C, TIE_STORE⟩)) matches .ok (.inline _)

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
#guard (protocol.answer "{}".toUTF8 ByteArray.empty ByteArray.empty).reason.map (·.startsWith "params: ") == some true

end PousCheck.RunProtocol
