module

public import PousCheck.RunProtocol
public import MemoryChallenger.WipeProtocol
public import Properties.Pous.EndToEndAt
import Mathlib.Basic.ENNReal.Real

/-! What the compiled checks of PoUS and the wipe, onsite's components, give the guarantees when they accept. -/

@[expose] public section

open MemoryChallenger MemoryChallenger.Properties
open PousCheck PousCheck.Properties
open Pous.EndToEnd

namespace Pous.Component

/-- **An accepted run is `AcceptedAt`.** Deterministic and assumption free: if PoUS's compiled check
(`RunProtocol.protocol.check`, the check onsite runs on PoUS's case) accepts the setup outputs `pub` and the evidence
`e` under the auditor's parameters `p`, then setup accepts the case's `C*` against its commitment at `p`'s point and
publishes `pub`, and the composed check at that point accepts the case's table and session against `pub` under `p`'s
keys and session id (`AcceptedAt`, the decision `Pous.Period.PeriodHeldAt` is about); and it publishes `certify` of
the point and profiles the point, neither of which reads the case. -/
def RunAccepts : Prop :=
  ∀ (p : RunProtocol.Params) (pub : SetupProtocol.Public) (e : RunProtocol.Evidence) (out : RunProtocol.Public)
    (prof : Run.Point),
    RunProtocol.protocol.check p pub e = .accept out prof →
      AcceptedAt p.point p.trusted.toList e.setup.commitment e.setup.store pub e.digests e.session p.sessionId ∧
        out = RunProtocol.certify p.point ∧ prof = p.point

/-- **The digest of PoUS's commitment is SHA-512 of its bytes.** If `RunProtocol.decodeCommitment` decodes the bytes
`b` to the setup outputs `pub`, the digest the composition binds for PoUS's case (`RunProtocol.digest`) is SHA-512 of
`b`: the decoder takes only canonical JSON, in the form `pub`'s encoder writes. It has no nontrivial lemma, since no
check takes part in it (it is an equation between two functions of `b`, which no verifier's decision changes), and no
honest lemma beyond its hypothesis that `b` decodes, which the composition establishes before it binds a digest. -/
def RunDigest : Prop :=
  ∀ (b : ByteArray) (pub : SetupProtocol.Public),
    RunProtocol.decodeCommitment b = .ok pub → RunProtocol.digest pub = Flock.Sha512.hash b

/-- What two runs on the setup outputs `pub` at the parameters `q` hash for their trees and their blocks: each run's
`tableStringsAt` (its table's and its `C*`'s trees, inner nodes and leaves) and `blockInput` of every block of each
run's `C*`. -/
def forkStrings (q : P2.Params) (pub : SetupProtocol.Public) (e e' : RunProtocol.Evidence) : List ByteArray :=
  tableStringsAt q pub e.digests e.setup.store ++ tableStringsAt q pub e'.digests e'.setup.store ++
    ((flat e.setup.store).toList ++ (flat e'.setup.store).toList).map blockInput

/-- **Two accepted runs on one setup output carry one setup.** Deterministic and assumption free: if PoUS's compiled
check accepts two evidences `e` and `e'` against the same setup outputs `pub` under the same parameters `p`, they carry
the same setup commitment, the same `C*` and the same table, or the two runs' hashes hold a SHA-512 collision
(`forkStrings`, both runs' data). It is the fork form of the commit-then-coin step: a prover whose `pub` is fixed
before the memory challenger's coins presents, at every coin it is accepted on, one setup and one table, outside a
collision between two of its runs. -/
def RunBinds : Prop :=
  ∀ (p : RunProtocol.Params) (pub : SetupProtocol.Public) (e e' : RunProtocol.Evidence),
    RunProtocol.protocol.Accepts p pub e → RunProtocol.protocol.Accepts p pub e' →
      (e.setup.commitment = e'.setup.commitment ∧ e.setup.store = e'.setup.store ∧ e.digests = e'.digests) ∨
        (let xs := forkStrings p.point.params pub e e'
         ∃ x ∈ xs, ∃ y ∈ xs, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y)

/-- **The signed session `s` passes every check of the wipe judge against the commitment `c`**
(`StreamedWipe.verdict` under `schemes` and the verifier's keys `trusted`), each as a proposition: its scheme is
known, its key is trusted and its signature verifies on its message; it is of `c`; `c` reveals one index a batch; it
ended complete with all of `c`'s batches; and batch `b` is the session's batch `b`, with no wrong answer, its reply
within `c`'s deadline, and the first batch sent after the fill ended. -/
def WipeChecked (schemes : Schemes) (trusted : List ByteArray) (s : StreamedWipe.Signed)
    (c : StreamedWipe.Commitment) : Prop :=
  (∃ p, schemes.lookup s.scheme = some p ∧ s.publicKey ∈ trusted ∧
      ∃ m, s.message = .ok m ∧ lmsVerify p s.publicKey s.commitment.session m s.signature = .ok true) ∧
    s.commitment = c ∧ c.params.batch = 1 ∧ s.ended = StreamedWipe.COMPLETE ∧ s.batches.size = c.params.batches ∧
    ∀ (b : Nat) (hb : b < s.batches.size),
      s.batches[b].b = (b : Int) ∧ s.batches[b].first = ((b * c.params.batch : Nat) : Int) ∧
        s.batches[b].count = min (c.params.batch : Int) ((c.params.k : Int) - s.batches[b].first) ∧
        s.batches[b].wrong = 0 ∧ 0 ≤ s.batches[b].elapsedNs ∧ s.batches[b].elapsedNs ≤ (c.params.limitNs : Int) ∧
        (b = 0 → s.fillEndNs ≤ s.batches[b].sendNs)

/-- **An accepted wipe is a checked signed session.** Deterministic and assumption free: if the wipe's compiled check
(`WipeProtocol.protocol.check`) accepts the commitment `c` and the evidence `e` under the parameters `p`, then `c` is
of `p`'s server and session, and `e` carries a signed session that passes every check of the wipe judge against `c`
under `p`'s keys (`WipeChecked`); and it publishes `certify` of `c`'s parameters at `EGRESS_MAX_UNITS`, which
certifies nothing, whatever count the evidence gives, and profiles `c`'s parameters with the evidence's egress in MiB
(`e.egress`: the count the evidence gives beside the signed session, `EGRESS_MAX_UNITS` when it gives none). Neither
the signed session nor a network attester signs that count. -/
def WipeAccepts : Prop :=
  ∀ (p : WipeProtocol.Params) (c : StreamedWipe.Commitment) (e : WipeProtocol.Evidence) (out : WipeProtocol.Public)
    (prof : StreamedWipe.Params × Int),
    WipeProtocol.protocol.check p c e = .accept out prof →
      c.server = p.server ∧ c.session = p.session ∧
        ∃ s, e.signed = some s ∧ WipeChecked SCHEMES p.trusted.toList s c ∧
          out = WipeProtocol.certify c.params StreamedWipe.EGRESS_MAX_UNITS ∧ prof = (c.params, e.egress)

/-- **The digest of the wipe's commitment is SHA-512 of its bytes**: `RunDigest` for `WipeProtocol`, and like it with
no honest or nontrivial lemma: no check takes part in it. -/
def WipeDigest : Prop :=
  ∀ (b : ByteArray) (c : StreamedWipe.Commitment),
    WipeProtocol.decodeCommitment b = .ok c → WipeProtocol.digest c = Flock.Sha512.hash b

/-- `check` accepts every input, whatever it publishes. -/
def AcceptsAll {P C E O R : Type} (check : P → C → E → Core.Outcome O R) : Prop :=
  ∀ p c e, ∃ out prof, check p c e = .accept out prof

/-- `RunAccepts` with `check` in place of PoUS's compiled check. -/
def RunAcceptsBy
    (check : RunProtocol.Params → SetupProtocol.Public → RunProtocol.Evidence →
      Core.Outcome RunProtocol.Public Run.Point) : Prop :=
  ∀ (p : RunProtocol.Params) (pub : SetupProtocol.Public) (e : RunProtocol.Evidence) (out : RunProtocol.Public)
    (prof : Run.Point),
    check p pub e = .accept out prof →
      AcceptedAt p.point p.trusted.toList e.setup.commitment e.setup.store pub e.digests e.session p.sessionId ∧
        out = RunProtocol.certify p.point ∧ prof = p.point

/-- `RunBinds` with `check`'s acceptance in place of PoUS's compiled check's. -/
def RunBindsBy
    (check : RunProtocol.Params → SetupProtocol.Public → RunProtocol.Evidence →
      Core.Outcome RunProtocol.Public Run.Point) : Prop :=
  ∀ (p : RunProtocol.Params) (pub : SetupProtocol.Public) (e e' : RunProtocol.Evidence),
    (∃ out prof, check p pub e = .accept out prof) → (∃ out prof, check p pub e' = .accept out prof) →
      (e.setup.commitment = e'.setup.commitment ∧ e.setup.store = e'.setup.store ∧ e.digests = e'.digests) ∨
        (let xs := forkStrings p.point.params pub e e'
         ∃ x ∈ xs, ∃ y ∈ xs, x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y)

/-- `WipeAccepts` with `check` in place of the wipe's compiled check. -/
def WipeAcceptsBy
    (check : WipeProtocol.Params → StreamedWipe.Commitment → WipeProtocol.Evidence →
      Core.Outcome WipeProtocol.Public (StreamedWipe.Params × Int)) : Prop :=
  ∀ (p : WipeProtocol.Params) (c : StreamedWipe.Commitment) (e : WipeProtocol.Evidence) (out : WipeProtocol.Public)
    (prof : StreamedWipe.Params × Int),
    check p c e = .accept out prof →
      c.server = p.server ∧ c.session = p.session ∧
        ∃ s, e.signed = some s ∧ WipeChecked SCHEMES p.trusted.toList s c ∧
          out = WipeProtocol.certify c.params StreamedWipe.EGRESS_MAX_UNITS ∧ prof = (c.params, e.egress)

end Pous.Component
