import Attest.Period
import Properties.Pous.MemoryChallenger
import Definitions.Claims
import Definitions.Accounting.Assumptions.Parties

/-!
# A party's signed period roots and liveness records: what a period's proof establishes, and what the auditor sees

An auditor party (the memory challenger; a network certifier) holds a lower HSS key its device's identity certified for
it, commits each certification period's records as the leaves of one frame-v3-sha512 tree and signs the root once, and
signs one liveness record per sampling block (`Attest.Period`, `verity.core.attest`). The records, the root
and its signature go to the prover only, as witness; a period's proof runs `Attest.check` on them under the lower key.
The lower key and its certificate (`Attest.PartyKey`) are public, as every liveness record shows them, and the verifier
checks the certificate in the clear (`Attest.checkKey`); `Attest.accepts` is both. The liveness records are public,
checked in the clear (`Attest.checkLiveness`).

**Soundness.** `PeriodSigned`: when `accepts` accepts, the records it reads are the ones the device committed, at their
positions, in a period its party's key closed, with the period's attachment, link, index and count, under a key its
identity certified for that party in that deployment (`OfDevice`); or the run's own data carries a break (`Break`): two
distinct strings of `runStrings` with one SHA-512 digest (`cr/sha-512`), or a signature that verifies over a message the
device never signed under that key (`euf-cma/lms-sha512-n64`): the certificate under the identity, or the root's under
the lower key. No assumption: what the device signed is the model `Device`. `PeriodSound` is the same at every prover
strategy as probabilities, with the collision finder `collisionFinder` (`B'_τ`) and the forger `forger` (`B_τ`) built
from the run: `Pr[accepts ∧ ¬OfDevice] ≤ Pr[B'_τ outputs a collision] + Pr[B_τ outputs a forgery]`. The run is `τ ω`, at
coins `ω`: what the devices signed, the public key the verifier checked, and the values the extractor `E` reads off the
accepted proof, so that with the period circuit's knowledge soundness (`Claims.KS`: `Pr[proof accepted ∧ E's values
fail check] ≤ κ`) this bounds `Pr[proof accepted ∧ a record E reads is not the device's]` by `κ` plus the two. Collision resistance is applied to
`B'_τ` outside the theorem (`Claims.StrictBirthday`), and EUF-CMA to `B_τ`: under `Faithful`, `Secret` and `Confined`
of the party, all `B_τ` gets of the party is its code's messages on its true inputs (`ProverSeesMessages`), and its key
signs each LMS leaf once, so `B_τ` is an EUF-CMA adversary against the party's keys with their signing oracle.

**Zero knowledge.** `LivenessContentFree`: what the party's code sends in the clear (its liveness records) is what it
sends on its inputs' skeleton, the records' bytes dropped. `AuditorSeesSkeleton`: under `Faithful`, `Confined` and
`WitnessPathPrivate`, what the auditor sees of the party is that, on the party's coins and the skeleton of its true
inputs, so it learns nothing of the records from the party.
What it learns from the prover is the period's proofs, whose statement `x` is public (the identity, the deployment, the
party, the lower key and its certificate, which the party's liveness records already show, the attachment, the link,
the period's index, the positions read, and outputs of at most 16 bits) and whose witness is `(signature, count, root,
records read, their paths)`: their zero knowledge is C-Flock's (`Claims.ZKmalR` at the period's circuit,
`Claims.Flock.ZK`), which these statements don't restate.
-/

namespace MemoryChallenger.Attest.Properties

open Flock
open Accounting.Assumptions

/-! ## What the devices sign -/

/-- A party key an identity certified: the deployment's hash, the party, and the lower public key. -/
structure Issued where
  deployment : ByteArray
  party : String
  lower : ByteArray

/-- A certification period a lower key closed: its deployment, party, attachment, link and index, and the records it
committed, in order. -/
structure Closed where
  deployment : ByteArray
  party : String
  attachment : ByteArray
  link : String
  period : Nat
  records : Array ByteArray

/-- A sampling block a lower key signed a liveness record for. -/
structure Live where
  deployment : ByteArray
  party : String
  attachment : ByteArray
  link : String
  period : Nat
  block : Nat

/-- What the auditor devices sign, by key: the party keys each identity certifies, the periods each lower key closes
and the sampling blocks it marks live. A key signs nothing else: one role per key (`lms.ROLES`), so a key also used for
another role is outside the model. -/
structure Device where
  issued : ByteArray → List Issued
  closed : ByteArray → List Closed
  live : ByteArray → List Live

/-- The message a closed period's root signature signs under the key with identifier `I`. -/
def Closed.message (I : ByteArray) (c : Closed) : Lms.E ByteArray :=
  Lms.periodRootMessage I c.deployment c.party c.attachment c.link c.period c.records.size (Attest.root c.records)

/-- The messages the devices sign under the key `pk`. -/
def Signed (dev : Device) (pk m : ByteArray) : Prop :=
  (∃ i ∈ dev.issued pk, Lms.partyKeyMessage (pk.extract 0 16) i.deployment i.party i.lower = .ok m) ∨
    (∃ c ∈ dev.closed pk, c.message (pk.extract 0 16) = .ok m) ∨
    ∃ l ∈ dev.live pk,
      Lms.livenessMessage (pk.extract 0 16) l.deployment l.party l.attachment l.link l.period l.block = .ok m

/-! ## The strings a run hashes -/

/-- What `Lms.roleDigest tag I context record` hashes. -/
def roleInput (tag : String) (I context record : ByteArray) : ByteArray :=
  Lms.lenField tag.toUTF8 ++ I ++ Lms.lenField context ++ Lms.lenField record

/-- What a role digest over the context `context` hashes, where it hashes anything. -/
def roleStrings (tag : String) (I : ByteArray) (context : Lms.E ByteArray) (record : ByteArray) : List ByteArray :=
  match context with
  | .ok ctx => match Lms.roleDigest tag I ctx record with
    | .ok _ => [roleInput tag I ctx record]
    | .error _ => []
  | .error _ => []

/-- What the period tree over `records` hashes: each leaf, and each node of each level below the root. -/
def treeStrings (records : Array ByteArray) : List ByteArray :=
  let d := Attest.periodDomain records.size
  let levels := FrameV3Sha512.levels d (Attest.leaves records)
  (List.range records.size).map (fun r =>
      MemoryChallenger.Properties.frameInput "leaf" [d.id, FrameV3Sha512.uint r, FrameV3Sha512.uint r, Attest.PERIOD_SCHEMA.toUTF8, records[r]!]) ++
    (List.range (levels.size - 1)).flatMap fun l => (List.range (levels[l]!.size / 2)).map fun i =>
      MemoryChallenger.Properties.frameInput "node" [d.id, FrameV3Sha512.uint l, FrameV3Sha512.uint i, (levels[l]!)[2 * i]!, (levels[l]!)[2 * i + 1]!]

/-- What SHA-512 hashes as a climb goes up from `digest` at `level`, at `cursor`, along the siblings `path`. -/
def climbInputs (d : FrameV3Sha512.Domain) : Nat → Nat → ByteArray → List ByteArray → List ByteArray
  | _, _, _, [] => []
  | level, cursor, digest, sib :: rest =>
    let input := if cursor % 2 = 0 then
        MemoryChallenger.Properties.frameInput "node" [d.id, FrameV3Sha512.uint level, FrameV3Sha512.uint (cursor >>> 1), digest, sib]
      else MemoryChallenger.Properties.frameInput "node" [d.id, FrameV3Sha512.uint level, FrameV3Sha512.uint (cursor >>> 1), sib, digest]
    input :: climbInputs d (level + 1) (cursor >>> 1) (Sha512.hash input) rest

/-- What `FrameV3Sha512.climb d position schema value path` hashes: the leaf, then each node up to the root. -/
def climbStrings (d : FrameV3Sha512.Domain) (position : Nat) (schema : String) (value : ByteArray)
    (path : Array ByteArray) : List ByteArray :=
  let leaf := MemoryChallenger.Properties.frameInput "leaf" [d.id, FrameV3Sha512.uint position, FrameV3Sha512.uint position, schema.toUTF8, value]
  leaf :: climbInputs d 0 position (Sha512.hash leaf) path.toList

/-- What the devices hash for what they sign under the key `pk`: each signed message's input, and each closed period's
tree. -/
def keyStrings (dev : Device) (pk : ByteArray) : List ByteArray :=
  let I := pk.extract 0 16
  (dev.issued pk).flatMap (fun i => roleStrings Lms.PARTY_KEY_TAG I (Lms.partyContext i.deployment i.party) i.lower) ++
    (dev.closed pk).flatMap (fun c => roleStrings Lms.PERIOD_ROOT_TAG I
      (Lms.periodRootContext c.deployment c.party c.attachment c.link c.period c.records.size) (Attest.root c.records) ++
      treeStrings c.records) ++
    (dev.live pk).flatMap fun l => roleStrings Lms.LIVENESS_TAG I
      (Lms.livenessContext l.deployment l.party l.attachment l.link l.period l.block) ByteArray.empty

/-- The strings a period's acceptance and the devices hash: its two messages' inputs (the lower key's certificate,
the root's) and each read's climb, and what the devices hash under the identity and under the lower key. -/
def runStrings (dev : Device) (identity deployment : ByteArray) (party : String) (k : Attest.PartyKey)
    (s : Attest.SignedRoot) (reads : Array Attest.Read) : List ByteArray :=
  let lower := k.lower
  roleStrings Lms.PARTY_KEY_TAG (identity.extract 0 16) (Lms.partyContext deployment party) lower ++
    roleStrings Lms.PERIOD_ROOT_TAG (lower.extract 0 16)
      (Lms.periodRootContext deployment party s.attachment s.link s.period s.count) s.root ++
    reads.toList.flatMap (fun r => climbStrings (Attest.periodDomain s.count) r.position Attest.PERIOD_SCHEMA r.record r.path) ++
    keyStrings dev identity ++ keyStrings dev lower

/-! ## A period's proof is the device's -/

/-- The period `s` and the records `reads` read are the device's: its identity certified `k`'s lower key for the party
in the deployment, and that key closed a period with `s`'s attachment, link, index and count whose record at each read's
position is the read's record. -/
def OfDevice (dev : Device) (identity deployment : ByteArray) (party : String) (k : Attest.PartyKey)
    (s : Attest.SignedRoot) (reads : Array Attest.Read) : Prop :=
  (∃ i ∈ dev.issued identity, i.deployment = deployment ∧ i.party = party ∧ i.lower = k.lower) ∧
    ∃ c ∈ dev.closed k.lower, c.deployment = deployment ∧ c.party = party ∧ c.attachment = s.attachment ∧
      c.link = s.link ∧ c.period = s.period ∧ c.records.size = s.count ∧
      ∀ r ∈ reads, c.records[r.position]? = some r.record

/-- A signature `σ` that verifies under the key `pk` over the message `m`, which the devices never signed under `pk`. -/
def Forges (p : Lms.Params) (dev : Device) (pk m : ByteArray) (σ : Lms.Signature) : Prop :=
  Lms.verify p pk m σ = .ok true ∧ ¬ Signed dev pk m

/-- The run's break: two distinct strings of the run with one SHA-512 digest; or the certificate verifies under the
identity over the lower key's party-key message, which the identity never signed; or the root's signature verifies
under the lower key over its message, which the lower key never signed. -/
def Break (p : Lms.Params) (dev : Device) (identity deployment : ByteArray) (party : String) (k : Attest.PartyKey)
    (s : Attest.SignedRoot) (reads : Array Attest.Read) : Prop :=
  (∃ x ∈ runStrings dev identity deployment party k s reads, ∃ y ∈ runStrings dev identity deployment party k s reads,
      x ≠ y ∧ Sha512.hash x = Sha512.hash y) ∨
    (∃ m, Lms.partyKeyMessage (identity.extract 0 16) deployment party k.lower = .ok m ∧
      Forges p dev identity m k.certificate) ∨
    ∃ m, s.message k.lower deployment party = .ok m ∧ Forges p dev k.lower m s.signature

/-- **A period's proof reads the device's records.** For every device, when the verifier accepts a period (the party's
public key's certificate in the clear, the period check under its lower key), the period and the records it reads are
the device's (`OfDevice`), or the run carries a break (`Break`). -/
def PeriodSigned : Prop :=
  ∀ (p : Lms.Params) (identity deployment : ByteArray) (party : String) (k : Attest.PartyKey) (s : Attest.SignedRoot)
    (reads : Array Attest.Read) (dev : Device),
    Attest.accepts p identity deployment party k s reads = .ok true →
      OfDevice dev identity deployment party k s reads ∨ Break p dev identity deployment party k s reads

/-! ## At every prover strategy -/

/-- One run of a period's audit: what the devices signed, the party's public key the verifier checks, and the values
the period's proof is checked on as the extractor reads them off the accepted proof (the identity and deployment it
names, the party, the signed root, the reads). -/
structure PeriodRun where
  dev : Device
  identity : ByteArray
  deployment : ByteArray
  party : String
  key : Attest.PartyKey
  root : Attest.SignedRoot
  reads : Array Attest.Read

/-- SHA-512 on the claims' messages. -/
def sha512 (m : Claims.Msg) : Claims.Msg := (Sha512.hash ⟨m.toArray⟩).data.toList

/-- **`B'_τ`**: on coins `ω`, the run `τ ω`'s strings with their digests, and the first collision among them; its cost
is the number of strings. -/
def collisionFinder (Ω : Type) [Fintype Ω] [Nonempty Ω] (τ : Ω → PeriodRun) : Claims.Finder where
  S := Ω
  out ω := let r := τ ω
    Claims.findColl ((runStrings r.dev r.identity r.deployment r.party r.key r.root r.reads).map fun x =>
      (x.data.toList, (Sha512.hash x).data.toList))
  cost ω := let r := τ ω
    (runStrings r.dev r.identity r.deployment r.party r.key r.root r.reads).length

open Classical in
/-- **`B_τ`**: on coins `ω`, the signature the run `τ ω` holds that the devices never signed, with its key and message:
the certificate where the identity never signed the lower key's party-key message, else the root's signature where the
lower key never signed its message, else none. -/
noncomputable def forger {Ω : Type} (τ : Ω → PeriodRun) (ω : Ω) : Option (ByteArray × ByteArray × Lms.Signature) :=
  let r := τ ω
  let lower := r.key.lower
  let certificate := match Lms.partyKeyMessage (r.identity.extract 0 16) r.deployment r.party lower with
    | .ok m => if Signed r.dev r.identity m then none else some (r.identity, m, r.key.certificate)
    | .error _ => none
  let rootSignature := match r.root.message lower r.deployment r.party with
    | .ok m => if Signed r.dev lower m then none else some (lower, m, r.root.signature)
    | .error _ => none
  match certificate with
  | some f => some f
  | none => rootSignature

/-- **`Pr[B_τ outputs a forgery]`**: over coins `ω` uniform over `Ω`, the probability that the forger's output on the
run `τ ω` verifies over a message the devices never signed under its key. -/
noncomputable def forgeryPr (p : Lms.Params) {Ω : Type} [Fintype Ω] (τ : Ω → PeriodRun) : ℝ :=
  Claims.Pr fun ω => ∃ pk m σ, forger τ ω = some (pk, m, σ) ∧ Forges p (τ ω).dev pk m σ

/-- **A period's proof reads the device's records, at every prover strategy.** For coins `ω` uniform over any finite
`Ω` (the prover's tape, the devices' coins, the verifier's and the extractor's) and the run `τ ω` of any prover
strategy against the devices, read through the extractor: the probability that the verifier accepts the run's values
and they are not the device's is at most the probability that `B'_τ` outputs a SHA-512 collision plus the probability that
`B_τ` outputs a forgery against the devices' keys (`forgeryPr`). -/
def PeriodSound : Prop :=
  ∀ (p : Lms.Params) (Ω : Type) [Fintype Ω] [Inhabited Ω] (τ : Ω → PeriodRun),
    Claims.Pr (fun ω =>
      Attest.accepts p (τ ω).identity (τ ω).deployment (τ ω).party (τ ω).key (τ ω).root (τ ω).reads = .ok true ∧
        ¬ OfDevice (τ ω).dev (τ ω).identity (τ ω).deployment (τ ω).party (τ ω).key (τ ω).root (τ ω).reads) ≤
      (collisionFinder Ω τ).Pr sha512 + forgeryPr p τ

/-! ## What the auditor and the prover see of the party -/

/-- One step of the party (`attest.Period`): a sampling block's records, which it adds to the period's and marks live
in the clear (`Period.block`), or the period's close, which signs its root (`Period.close`). -/
inductive Step
  | block (attachment : ByteArray) (link : String) (period index : Nat) (records : List ByteArray)
  | close (attachment : ByteArray) (link : String) (period : Nat)

/-- A step with each record's bytes dropped (their number kept). -/
def Step.skeleton : Step → Step
  | .block a l p b records => .block a l p b (records.map fun _ => ByteArray.empty)
  | .close a l p => .close a l p

/-- The party's channels: its liveness records, in the clear; its signed roots and its records, to the prover. -/
inductive Ch
  | liveness
  | root
  | record
  deriving DecidableEq

/-- Where each channel goes. -/
def dest : Ch → Parties.Dest
  | .liveness => .clear
  | .root | .record => .witness

/-- The party's code, as `attest.Period` runs under one lower key (`lms.PartyKey`): its state is its coins (its keys),
the number of signatures its lower key has made (the next LMS leaf), and the open period's records. A block's step adds
its records and sends the HSS signature (`hss ω q m`, the lower key's signature at leaf `q` with the identity's
certificate) of its liveness message in the clear; a close sends the signature of the root's message and the records
to the prover and empties the period. `I ω` is the lower key's identifier. A message the code refuses sends nothing. -/
def code {Ω : Type} (deployment : ByteArray) (party : String) (I : Ω → ByteArray)
    (hss : Ω → Nat → ByteArray → ByteArray) : Parties.Code Ω (Ω × Nat × List ByteArray) Step Ch ByteArray where
  init ω := (ω, 0, [])
  step st x := match x with
    | .block a l p b records => match Lms.livenessMessage (I st.1) deployment party a l p b with
      | .ok m => ((st.1, st.2.1 + 1, st.2.2 ++ records), [(Ch.liveness, hss st.1 st.2.1 m)])
      | .error _ => ((st.1, st.2.1, st.2.2 ++ records), [])
    | .close a l p =>
      match Lms.periodRootMessage (I st.1) deployment party a l p st.2.2.length (Attest.root st.2.2.toArray) with
      | .ok m => ((st.1, st.2.1 + 1, []), (Ch.root, hss st.1 st.2.1 m) :: st.2.2.map fun v => (Ch.record, v))
      | .error _ => (st, [])

/-- What a list of messages puts in the clear. -/
def inClear (ms : List (Ch × ByteArray)) : List (Ch × ByteArray) := ms.filter fun m => dest m.1 = .clear

/-- **The party's public records carry no content.** For any keys, coins and signing function, what the party's code
sends in the clear on any inputs is what it sends on their skeleton (`Step.skeleton`, each record's bytes dropped): a
function of the skeleton, so the same on any two inputs that differ only in the records' bytes. -/
def LivenessContentFree : Prop :=
  ∀ {Ω : Type} (deployment : ByteArray) (party : String) (I : Ω → ByteArray) (hss : Ω → Nat → ByteArray → ByteArray)
    (ω : Ω) (xs : List Step),
      inClear ((code deployment party I hss).messages ω xs) =
        inClear ((code deployment party I hss).messages ω (xs.map Step.skeleton))

/-- **The auditor sees no record of the party's.** A run of the party `Faithful` to its code, `Confined` and with its
witness path private (`WitnessPathPrivate`) shows the auditor what its code sends in the clear on its coins and the
skeleton of its true inputs, as it goes on the wire: a function of its coins and the skeleton, so two runs from the
same coins on inputs that differ only in the records' bytes show the auditor the same. -/
def AuditorSeesSkeleton : Prop :=
  ∀ {Ω Wi A : Type} (deployment : ByteArray) (party : String) (I : Ω → ByteArray)
    (hss : Ω → Nat → ByteArray → ByteArray) (wire : Ch × ByteArray → Wi)
    (r : Parties.Run Ω (Ω × Nat × List ByteArray) Step Ch ByteArray Wi A),
    Parties.Faithful (code deployment party I hss) r → Parties.Confined dest wire r → Parties.WitnessPathPrivate r →
      Parties.auditorView wire r =
        (inClear ((code deployment party I hss).messages r.coins (r.truth.map Step.skeleton))).map wire

/-- **The prover gets of the party only its code's messages.** A run of the party `Faithful` to its code, `Secret` and
`Confined`: all the prover gets of it is its code's messages on its true inputs from its coins, as they go on the wire,
and no access to its secret state. -/
def ProverSeesMessages : Prop :=
  ∀ {Ω Wi A : Type} (deployment : ByteArray) (party : String) (I : Ω → ByteArray)
    (hss : Ω → Nat → ByteArray → ByteArray) (wire : Ch × ByteArray → Wi)
    (r : Parties.Run Ω (Ω × Nat × List ByteArray) Step Ch ByteArray Wi A),
    Parties.Faithful (code deployment party I hss) r → Parties.Secret r → Parties.Confined dest wire r →
      Parties.proverView r = (((code deployment party I hss).messages r.coins r.truth).map wire, [])

end MemoryChallenger.Attest.Properties
