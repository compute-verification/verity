import PousCheck.RunProtocol
import Properties.Pous.EndToEnd

/-!
# A run that cites the setup check's recorded acceptance

A run's params may cite the verifier's setup check's recorded acceptance (`PousCheck.RunProtocol.SetupRef`: the setup
protocol's answer, the parameters it checked at, and where its record is) instead of the case carrying `C*` for the run's
check to decode again: at Demo B's 130 GiB `C*` is decoded once, at setup. The citation is the verifier's, as its list of
memory-challenger keys is, and the premise both statements take, the setup check's acceptance, is what its record
holds; a prover never supplies it. The run's check then reads no block of `C*`: the cited answer
must be the setup protocol's acceptance, at the run's parameters, publishing exactly the run's commitment (the server's
commitment to `W`, the salt, the root and the block count: `RunProtocol.cite`), and the composed check (`Run.checkAt`)
must accept the run against that commitment. Two statements say what that keeps:

* `RootDecodes`, at any parameters `q`: if the setup check accepts `C*` under the server's commitment and salt `c` and
  publishes `pub` (the acceptance a citation records), then every `C*'` of `q`'s shape (`ShapedAt`) whose leaves under
  the salt have the published root decodes canonically to a payload with the server's commitment
  (`DecodesToCommitment`), or the two stores' trees hold two distinct strings with one SHA-512 digest (`StoreBreak`,
  `cr/sha-512`, every witness one of the two stores' strings). So whatever store the server holds when the run's session
  audits the published root, it is the store setup decoded, outside a collision.
* `CitedRunAccepted`, at P2's deployed point under S1's LMS: with the setup check's acceptance publishing `pub`, a run
  whose citation and composed check accept (`RunProtocol.acceptsCited`) is `Pous.EndToEnd.Accepted`, word for word, so
  `Pous.EndToEnd.RunCertifies` applies to it unchanged.
* `CitedAnswerAccepted`, the same with the record's answer in place of the acceptance: when the cited answer is what the
  setup protocol answered (`Core.Protocol.answer`) on the bytes its record holds, at the citation's parameters, what
  `RunProtocol.cite` checks of that answer (the setup protocol's acceptance, at the run's parameters, publishing exactly
  `pub`) is the acceptance `CitedRunAccepted` takes, of the commitment and `C*` those bytes decode to.

All three are proved in `Proofs.Properties.Pous.Cited`, from the setup check's `AcceptsDecodes` and `PublishesRoot` and
the trees' binding (`mth_binds`), under no hash assumption: a collision is a disjunct of the conclusion, as in
`RunCheckAccepts`.
-/

open PousCheck PousCheck.Properties
open Pous.EndToEnd (leafInput blockInput)

namespace Pous.Cited

/-- What `C*`'s tree under `salt` hashes: each inner node, then for block `v` of segment `s` its leaf's input
(`leafInput` of the block's row digest) and its row's (`blockInput`). -/
def storeStrings (salt : ByteArray) (C : Array (Array ByteArray)) : List ByteArray :=
  let L := leaves salt C
  nodeInputs L 0 L.size ++
    (C.toList.mapIdx fun s seg =>
      (seg.toList.mapIdx fun v block => [leafInput salt s v (Merkle.rowDigest block), blockInput block]).flatten).flatten

/-- Two stores' trees under one salt hold a break: a string of one and a distinct string of the other with one SHA-512
digest (`cr/sha-512`). -/
def StoreBreak (salt : ByteArray) (C C' : Array (Array ByteArray)) : Prop :=
  ∃ x ∈ storeStrings salt C, ∃ y ∈ storeStrings salt C', x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y

/-- **The setup check's acceptance binds the published root to the decoded store.** For all parameters `q`, if the setup
check accepts `C*` under the server's commitment and salt `c` and publishes `pub`, then every `C*'` of `q`'s shape whose
leaves under `c`'s salt have the published root decodes canonically to a payload whose commitment is the server's, or
the two stores' trees hold a SHA-512 collision. A run that cites the acceptance instead of carrying `C*` checks that its
commitment is `pub` and that its session audits `pub.root`; this is what the store it audits then is. -/
def RootDecodes : Prop :=
  ∀ (q : P2.Params) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)) (pub : SetupProtocol.Public)
    (prof : Unit), SetupProtocol.protocol.check q c C = .accept pub prof →
      ∀ C' : Array (Array ByteArray), ShapedAt q C' → Merkle.root (leaves c.salt C') = some pub.root →
        DecodesToCommitment q c C' ∨ StoreBreak c.salt C C'

/-- **A run accepted on a citation is `Accepted`.** At P2's deployed point under S1's LMS, for every verifier's list
`trusted` of memory-challenger keys: if the setup check accepts `C*` under `c` and publishes `pub` (the acceptance the
citation records) and the run's check accepts the citation `s` and the run against `pub` (`RunProtocol.acceptsCited`),
then setup and the composed check accept the run (`Pous.EndToEnd.Accepted`), which is all `RunCertifies` asks. -/
def CitedRunAccepted : Prop :=
  ∀ (trusted : List ByteArray) (c : SetupProtocol.Commitment) (C : Array (Array ByteArray))
    (pub : SetupProtocol.Public) (prof : Unit) (s : RunProtocol.SetupRef) (digests : ByteArray) (r : Run.Session)
    (sid : ByteArray),
    SetupProtocol.protocol.check Pous.EndToEnd.q c C = .accept pub prof →
    RunProtocol.acceptsCited MemoryChallenger.SCHEMES Run.DEPLOYED trusted s pub digests r sid = .ok () →
      Pous.EndToEnd.Accepted trusted c C pub digests r sid

/-- **A run accepted on the setup check's own answer is `Accepted`.** At P2's deployed point under S1's LMS, for every
verifier's list `trusted` of memory-challenger keys: if the citation's answer is what the setup protocol answered on the
bytes `pb`, `cb` and `eb` of its parameters, commitment and `C*` (the inputs the citation's record holds), those
parameters decode to the citation's, and the run's check accepts the citation `s` and the run against `pub`
(`RunProtocol.acceptsCited`), then `cb` and `eb` decode to a commitment and a `C*` that setup and the composed check
accept the run on (`Pous.EndToEnd.Accepted`). -/
def CitedAnswerAccepted : Prop :=
  ∀ (trusted : List ByteArray) (s : RunProtocol.SetupRef) (pub : SetupProtocol.Public) (digests : ByteArray)
    (r : Run.Session) (sid pb cb eb : ByteArray),
    s.answer = SetupProtocol.protocol.answer pb cb eb →
    (Core.canonicalJson pb >>= SetupProtocol.protocol.params) = .ok s.params →
    RunProtocol.acceptsCited MemoryChallenger.SCHEMES Run.DEPLOYED trusted s pub digests r sid = .ok () →
      ∃ (c : SetupProtocol.Commitment) (C : Array (Array ByteArray)),
        SetupProtocol.protocol.commitment cb = .ok c ∧ SetupProtocol.protocol.evidence eb = .ok C ∧
          Pous.EndToEnd.Accepted trusted c C pub digests r sid

end Pous.Cited
