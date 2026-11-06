import PousCheck.RunProtocol
import MemoryChallenger.WipeProtocol
import Properties.Pous.EndToEndAt

/-!
# PoUS and the wipe as components: what their compiled checks' acceptance gives the guarantees

`Pous.Period.PeriodHeldAt` bounds the probability that the verifier's whole decision at a point `pt`
(`Pous.EndToEnd.AcceptedAt`, read by `AuditsOf.Accepts`) accepts a period with no break while the node held too few
bit-ticks, over an audit whose verifier's keys and session id are fixed and whose setup commitment, `C*`, setup outputs
and table are functions of the coins drawn before the memory challenger's (`AuditsOf`). Onsite's verifier of record
runs `PousCheck.RunProtocol.protocol` on bytes instead: the one theorem's leg for PoUS turns that protocol's acceptance
into `AcceptedAt` on fields of that kind.
Each statement below is one step of it, about one function of the code, in the shape of the code's properties.

* `RunAccepts`: what the compiled check accepts is `AcceptedAt` at the auditor's point, keys and session id, on the
  case's setup, table and session, and what it publishes is `certify` of the point alone. With
  `Properties.Composition.DecodedAccepted` at `Verity.component .pous`, which yields `RunProtocol.protocol.Accepts`, it
  gives the event `PeriodHeldAt` bounds.
* `RunDigest`: the digest the composition binds for PoUS's case is SHA-512 of the commitment's bytes, so the
  composition's commitment, fixed before the coins, fixes those bytes outside a SHA-512 collision, and they decode to
  one setup output `pub`.
* `RunBinds`: two runs the check accepts on one `pub` carry one setup commitment, one `C*` and one table, or the two
  runs' trees and blocks hold a SHA-512 collision (`forkStrings`): the fork form of the commit-then-coin step, which
  makes `AuditsOf`'s `commitment`, `store` and `digests` functions of the coins before PoUS's.

The wipe (`MemoryChallenger.WipeProtocol.protocol`, `verity/wipe/v0`) is no component of onsite yet; its statements are
the same steps for the composition that takes it.

* `WipeChecked`: the signed session passes every check of the wipe judge of record (`StreamedWipe.verdict`), as
  propositions: a signature under a trusted key on the session's message, the session's commitment, batch size 1,
  every batch present, in order, with no wrong answer and within the deadline, the first sent after the fill.
* `WipeAccepts`: what the compiled check accepts is of the period's server and session and carries a signed session
  that is `WipeChecked`, and it publishes `certify` of the commitment's parameters and that session's egress. It is
  the accepted session that `MemoryChallenger.Properties.WipeSurvivorBound` reads.
* `WipeDigest`: the wipe's `RunDigest`.

Each is proved in `Proofs.Properties.Pous.Component` (`Pous.Component.SecurityProofs`) and listed in the lock.

`RunAcceptsBy`, `RunBindsBy` and `WipeAcceptsBy` are `RunAccepts`, `RunBinds` and `WipeAccepts` about any `check` of
the compiled check's type: at the compiled check each is its guarantee (`Iff.rfl`), and at a check that accepts
everything (`AcceptsAll`) each is false, which is each guarantee's nontrivial lemma. The digests have none: they are
equations about the code's digest, and no check takes part in them.
-/

open MemoryChallenger MemoryChallenger.Properties
open PousCheck PousCheck.Properties
open Pous.EndToEnd

namespace Pous.Component

/-! ### PoUS: `verity/pous/p2/v4/run`, onsite's component -/

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

/-! ### The wipe: `verity/wipe/v0` -/

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
under `p`'s keys (`WipeChecked`); and it publishes `certify` of `c`'s parameters and that session's egress in MiB, and
profiles them. -/
def WipeAccepts : Prop :=
  ∀ (p : WipeProtocol.Params) (c : StreamedWipe.Commitment) (e : WipeProtocol.Evidence) (out : WipeProtocol.Public)
    (prof : StreamedWipe.Params × Int),
    WipeProtocol.protocol.check p c e = .accept out prof →
      c.server = p.server ∧ c.session = p.session ∧
        ∃ s, e.signed = some s ∧ WipeChecked SCHEMES p.trusted.toList s c ∧
          out = WipeProtocol.certify c.params (StreamedWipe.egressUnits s.egressBits) ∧
          prof = (c.params, StreamedWipe.egressUnits s.egressBits)

/-- **The digest of the wipe's commitment is SHA-512 of its bytes**: `RunDigest` for `WipeProtocol`, and like it with
no honest or nontrivial lemma: no check takes part in it. -/
def WipeDigest : Prop :=
  ∀ (b : ByteArray) (c : StreamedWipe.Commitment),
    WipeProtocol.decodeCommitment b = .ok c → WipeProtocol.digest c = Flock.Sha512.hash b

/-! ### The statements about any check -/

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
          out = WipeProtocol.certify c.params (StreamedWipe.egressUnits s.egressBits) ∧
          prof = (c.params, StreamedWipe.egressUnits s.egressBits)

end Pous.Component
