import NetworkCheck.Check
import Definitions.NetworkCertifier
import Mathlib.Data.Set.Card

/-!
# The network certificate check: what its Lean code (`NetworkCheck.Check`) does

The properties of `NetworkCheck.protocol` (`verity/network-certificate/v2`, the port of `audit.py`'s `Audit`), stated
against its functions as they run: what `answer` and the check refuse, what an acceptance establishes about each
link-window and what an undecided outcome means, that the honest transcript is accepted, and what an accepted run is:
a run of the certifier program under the declared advice, or a run from whose own data a function finds a SHA-512
collision or an LMS forgery.
Each is a `Prop`; `Properties.Proofs.NetworkCertifier` proves the ones about the code, and
`Proofs.NetworkCertifier.Accepted` (in `security_proofs`, through the reference's `Proofs.NetworkCertifier.Code`) the
last two, `AcceptedRunsAreProgramRuns` and the count it gives, `EncardAcceptedObservationsLe`.

The predicates below say what a link-window's records, certificate and digests are, in the check's own terms; none
restates the check. A link-window is a direction, a link of the unit in that direction and a window `w < W`.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check
open NetworkCertifier (ClockSync LogicalFrame expectedRow)

namespace Properties.NetworkCertifier

/-! ## A link-window's records and certificate -/

/-- The unit's links in direction `d`. -/
def linkNames (p : Params) : Direction → List String
  | .egress => p.egress.toList
  | .ingress => p.ingress.toList

/-- The records of the commitment whose head names link-window `(d, l, w)`, in the commitment's order. -/
def recordsOf (c : Commitment) (d : Direction) (l : String) (w : Nat) : List Certified :=
  c.records.toList.filter fun r => decide (r.head.direction = d ∧ r.head.link = l ∧ r.head.window = w)

/-- The LMS verifier accepts `sig` (once decoded) under the parameters' key over the message `m`. -/
def LmsVerifies (p : Params) (m sig : ByteArray) : Prop :=
  ∃ s, Lms.Signature.decode p.lms sig = .ok s ∧ Lms.verify p.lms p.key m s = .ok true

/-- Link-window `(d, l, w)` has exactly one record `r`, and its certificate is the key's (the key's identifier `I`),
names `(d, l, w)`, signs the head's digest `dg`, and verifies under the key over `certificateMessage` of them. -/
def Signs (p : Params) (c : Commitment) (d : Direction) (l : String) (w : Nat) (r : Certified) (dg : ByteArray) :
    Prop :=
  recordsOf c d l w = [r] ∧ r.head.digest = .ok dg ∧ r.certificate.digest = dg ∧
    r.certificate.keyId = p.key.extract 0 16 ∧ r.certificate.link = l ∧ r.certificate.direction = d ∧
    r.certificate.window = w ∧
    ∃ m, certificateMessage (p.key.extract 0 16) l d w dg = .ok m ∧ LmsVerifies p m r.certificate.signature

/-- `Signs`, and the signed head is complete and settled by the deadline. -/
def Settled (p : Params) (c : Commitment) (d : Direction) (l : String) (w : Nat) (r : Certified) (dg : ByteArray) :
    Prop :=
  Signs p c d l w r dg ∧ r.head.status = none ∧ r.head.deadline = true

/-- Link-window `(d, l, w)` awaits its record: it has none, or one whose certificate `Signs` but whose head is not
settled by the deadline. -/
def Pending (p : Params) (c : Commitment) (d : Direction) (l : String) (w : Nat) : Prop :=
  recordsOf c d l w = [] ∨ ∃ r dg, Signs p c d l w r dg ∧ r.head.deadline = false

/-! ## The evidence, the advice and the Program's grid -/

/-- The evidence fits the parameters: one entry per egress link, `W` declared syncs on each, every payload `8·B` bits. -/
def EvidenceFits (p : Params) (e : Evidence) : Prop :=
  e.links.size = p.egress.size ∧ (∀ link ∈ e.links.toList, link.syncs.size = p.windows) ∧
    ∀ link ∈ e.links.toList, ∀ f ∈ link.frames, f.payload.length = 8 * p.grid.frameBytes

/-- `eg` and `ig` are `parameters_digest` of the egress grid with Σ_sync and of the ingress grid. -/
def ParamsDigests (p : Params) (eg ig : ByteArray) : Prop :=
  parametersDigest p.grid (some (manifestSet p.syncSet)) = .ok eg ∧ parametersDigest p.ingressGrid none = .ok ig

/-- `syncs` are the clock syncs `link` declares, one per window. -/
def Declares {W : Nat} (link : Link) (syncs : Fin W → ClockSync) : Prop := ∀ v : Fin W, link.syncs[v.1]? = some (syncs v)

/-- The sync `link` declares for window `w` is in Σ_sync and is the unit's, the one the first egress link declares. -/
def SyncAdmitted (p : Params) (e : Evidence) (link : Link) (w : Nat) : Prop :=
  ∃ s, link.syncs[w]? = some s ∧ s.delta ∈ p.syncSet.deltas ∧ s.rho ∈ p.syncSet.rhos ∧
    ∃ first s₀, e.links[0]? = some first ∧ first.syncs[w]? = some s₀ ∧ s₀.delta = s.delta ∧ s₀.rho = s.rho

/-- A grid of window `w` from rows of bits at global buckets: row `i` is `row (w·T + i)`, each slot's bits as bytes. -/
def bytesGrid {r : Nat} (T : Nat) (row : Nat → Fin r → Option (List Bool)) (w : Nat) :
    Array (Array (Option ByteArray)) :=
  (List.range T).toArray.map fun i => (List.ofFn (row (w * T + i))).toArray.map (·.map bytesOfBits)

/-- The Program's grid of an egress link in window `w`: its rows are `expectedRow` of the link's frames under the
declared syncs. -/
def programGrid (p : Params) (syncs : Fin p.windows → ClockSync) (frames : List LogicalFrame) (w : Nat) :
    Array (Array (Option ByteArray)) :=
  bytesGrid p.grid.buckets (expectedRow p.grid.allocation p.grid.buckets p.windows syncs frames) w

/-- Egress link-window `(l, w)`, opened by `link`, is accepted: it is `Settled` with digest `dg`, its declared sync is
admitted, and `dg` is the digest of `record_head` of the Program's grid under the declared syncs (complete, settled,
under `eg`). -/
def EgressAccepted (p : Params) (c : Commitment) (e : Evidence) (eg : ByteArray) (l : String) (link : Link)
    (w : Nat) : Prop :=
  ∃ r dg, Settled p c .egress l w r dg ∧ SyncAdmitted p e link w ∧
    ∀ syncs : Fin p.windows → ClockSync, Declares link syncs →
      ∃ h, recordHead l .egress w (programGrid p syncs link.frames w) none p.grid eg true = .ok h ∧ h.digest = .ok dg

/-- Ingress link-window `(l, w)` is accepted: it is `Settled`, and its head is of the ingress grid (T, r, B, τ_b) under
its parameters digest `ig`. -/
def IngressAccepted (p : Params) (c : Commitment) (ig : ByteArray) (l : String) (w : Nat) : Prop :=
  ∃ r dg, Settled p c .ingress l w r dg ∧ r.head.buckets = (p.ingressGrid.buckets : Int) ∧
    r.head.allocation = (p.ingressGrid.allocation : Int) ∧ r.head.frameBytes = (p.ingressGrid.frameBytes : Int) ∧
    r.head.bucketMs = (p.ingressGrid.bucketMs : Int) ∧ r.head.parameters = ig

/-! ## (1) Refusals -/

/-- `answer` rejects bytes that a decoder refuses: parameters that are not canonical JSON or not parameters, a
commitment that does not decode, evidence that is not canonical JSON or not evidence. -/
def DecodeRefusals : Prop :=
  ∀ a b c : ByteArray,
    (Core.canonicalJson a >>= protocol.params).toOption = none ∨ (protocol.commitment b).toOption = none ∨
      (protocol.evidence c).toOption = none →
    (protocol.answer a b c).verdict = .reject

/-- `answer` accepts exactly when the three decode and the check accepts what they decode to. -/
def AnswerAccepts : Prop :=
  ∀ a b c : ByteArray, (protocol.answer a b c).verdict = .accept ↔
    ∃ p cm e, (Core.canonicalJson a >>= protocol.params) = .ok p ∧ protocol.commitment b = .ok cm ∧
      protocol.evidence c = .ok e ∧ protocol.Accepts p cm e

/-- Evidence that doesn't fit the parameters is a rejection. -/
def MalformedEvidenceRejects : Prop :=
  ∀ (p : Params) c e, ¬ EvidenceFits p e → ∃ why, protocol.check p c e = .reject why

/-- Parameters without a parameters digest (no canonical JSON) are a rejection. -/
def UndigestedParamsReject : Prop :=
  ∀ (p : Params) c e, (∀ eg ig, ¬ ParamsDigests p eg ig) → ∃ why, protocol.check p c e = .reject why

/-- An accepted unit has exactly one record of every link-window. -/
def AcceptsOneRecordEach : Prop :=
  ∀ (p : Params) c e, protocol.Accepts p c e →
    ∀ d, ∀ l ∈ linkNames p d, ∀ w < p.windows, (recordsOf c d l w).length = 1

/-! ## (2) What an acceptance establishes -/

/-- **The check accepts exactly the units every link-window of which is accepted**: the evidence fits, the
parameters have their digests, every egress link-window is `EgressAccepted` and every ingress link-window
`IngressAccepted`. -/
def AcceptsIff : Prop :=
  ∀ (p : Params) c e, protocol.Accepts p c e ↔
    EvidenceFits p e ∧ ∃ eg ig, ParamsDigests p eg ig ∧
      (∀ l link, (l, link) ∈ p.egress.toList.zip e.links.toList → ∀ w < p.windows, EgressAccepted p c e eg l link w) ∧
      ∀ l ∈ p.ingress.toList, ∀ w < p.windows, IngressAccepted p c ig l w

/-! ## (3) Completeness -/

/-- The record an honest certifier issues for link-window `(d, l, w)`: head `h`, and a certificate of the key's
identifier over `(d, l, w)` and `h`'s digest `dg` with signature `σ`. -/
def honestRecord (p : Params) (d : Direction) (l : String) (w : Nat) (h : RecordHead) (dg σ : ByteArray) : Certified :=
  ⟨h, ⟨p.key.extract 0 16, l, d, w, dg, σ⟩⟩

/-- **The honest transcript is accepted.** Evidence that fits, with every declared sync admitted; parameters with their
digests; and a commitment holding exactly one record of every link-window, the honest one: on egress the head
`record_head` makes of the Program's grid under the declared syncs, on ingress a complete head settled by the deadline
of the ingress grid and its digest, each signed by the key over its `certificateMessage`. -/
def HonestAccepted : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (eg ig : ByteArray),
    EvidenceFits p e → ParamsDigests p eg ig →
    (∀ link ∈ e.links.toList, ∀ w < p.windows, SyncAdmitted p e link w) →
    (∀ l link, (l, link) ∈ p.egress.toList.zip e.links.toList → ∀ w < p.windows,
      ∃ syncs : Fin p.windows → ClockSync, Declares link syncs ∧ ∃ (h : RecordHead) (dg σ : ByteArray),
        recordHead l .egress w (programGrid p syncs link.frames w) none p.grid eg true = .ok h ∧ h.digest = .ok dg ∧
        recordsOf c .egress l w = [honestRecord p .egress l w h dg σ] ∧
        ∃ m, certificateMessage (p.key.extract 0 16) l .egress w dg = .ok m ∧ LmsVerifies p m σ) →
    (∀ l ∈ p.ingress.toList, ∀ w < p.windows, ∃ (h : RecordHead) (dg σ : ByteArray),
      h.buckets = (p.ingressGrid.buckets : Int) ∧
      h.allocation = (p.ingressGrid.allocation : Int) ∧ h.frameBytes = (p.ingressGrid.frameBytes : Int) ∧
      h.bucketMs = (p.ingressGrid.bucketMs : Int) ∧ h.parameters = ig ∧ h.status = none ∧ h.deadline = true ∧
      h.digest = .ok dg ∧ recordsOf c .ingress l w = [honestRecord p .ingress l w h dg σ] ∧
      ∃ m, certificateMessage (p.key.extract 0 16) l .ingress w dg = .ok m ∧ LmsVerifies p m σ) →
    protocol.Accepts p c e

/-! ## (4) Undecided -/

/-- **The check is undecided exactly when no link-window is refused and some awaits its record**: the evidence fits,
the parameters have their digests, every link-window is accepted or `Pending`, and some is `Pending`. -/
def UndecidedIff : Prop :=
  ∀ (p : Params) c e, (∃ why, protocol.check p c e = .undecided why) ↔
    EvidenceFits p e ∧ ∃ eg ig, ParamsDigests p eg ig ∧
      (∀ l link, (l, link) ∈ p.egress.toList.zip e.links.toList → ∀ w < p.windows,
        EgressAccepted p c e eg l link w ∨ Pending p c .egress l w) ∧
      (∀ l ∈ p.ingress.toList, ∀ w < p.windows, IngressAccepted p c ig l w ∨ Pending p c .ingress l w) ∧
      ∃ d, ∃ l ∈ linkNames p d, ∃ w < p.windows, Pending p c d l w

/-! ## (5) What an accepted run is

The certifier device of the unit, as the check meets it: on each egress link the row it records at every global
bucket, the status it gives each link-window, and every head it issues in the run, each for a link-window. It is
honest (`Honest`, which the deployment's `trusted-nebius-host` is about): an egress head is `record_head` of the grid of
its recorded rows under its status, an ingress head carries its status, every head names its own link-window, and every
frame it records is `8·B` bits (W3). It signs, with the key, the `certificateMessage` of each head it issues (`Signed`).

Nothing is assumed of SHA-512 or LMS. Each claim about an accepted run holds, or a function of the run finds a break in
the run's own data: `findCollision` searches `runStrings`, the strings the check and the device hash, for two distinct
strings with one SHA-512 digest (`cr/sha-512`), and `findForgery` searches the commitment's certificates for a message
the device never signed whose signature verifies under the key (`euf-cma/lms-sha512-n64`). -/

/-- One head the device issues, for link-window `(direction, link, window)`. -/
structure Issue where
  direction : Direction
  link : String
  window : Nat
  head : RecordHead

/-- The unit's certifier device in one run. -/
structure Device (p : Params) where
  rows : Fin p.egress.size → Nat → Fin p.grid.allocation → Option (List Bool)
  status : Direction → String → Nat → Option String
  issued : List Issue

/-- The device's grid of egress link `ℓ` in window `w`. -/
def deviceGrid {p : Params} (dev : Device p) (ℓ : Fin p.egress.size) (w : Nat) : Array (Array (Option ByteArray)) :=
  bytesGrid p.grid.buckets (dev.rows ℓ) w

/-- The device is honest: each head is its link-window's (on egress, `record_head` of the recorded grid under the
link-window's status, with the egress parameters digest `eg`), and its frames are `8·B` bits. -/
def Honest (p : Params) (dev : Device p) (eg : ByteArray) : Prop :=
  (∀ i ∈ dev.issued, i.head.link = i.link ∧ i.head.direction = i.direction ∧ i.head.window = i.window ∧
    i.head.status = dev.status i.direction i.link i.window) ∧
  (∀ i ∈ dev.issued, i.direction = .egress → ∀ ℓ : Fin p.egress.size, p.egress[ℓ] = i.link →
    ∃ dl, recordHead i.link .egress i.window (deviceGrid dev ℓ i.window) (dev.status .egress i.link i.window) p.grid eg
      dl = .ok i.head) ∧
  ∀ ℓ t s f, dev.rows ℓ t s = some f → f.length = 8 * p.grid.frameBytes

/-- The messages the device signs: `certificateMessage` of each head it issues, under the key's identifier. -/
def Signed {p : Params} (dev : Device p) (m : ByteArray) : Prop :=
  ∃ i ∈ dev.issued, ∃ dg, i.head.digest = .ok dg ∧
    certificateMessage (p.key.extract 0 16) i.link i.direction i.window dg = .ok m

/-- The LMS verifier, as a test of a message and a signature's bytes under the parameters' key. -/
def lmsAccepts (p : Params) (m sig : ByteArray) : Bool :=
  match Lms.Signature.decode p.lms sig with
  | .ok s => (Lms.verify p.lms p.key m s).toOption == some true
  | .error _ => false

/-- The messages the device signs (`Signed`), as a list. -/
def signedMessages {p : Params} (dev : Device p) : List ByteArray :=
  dev.issued.filterMap fun i =>
    i.head.digest.toOption.bind fun dg =>
      (certificateMessage (p.key.extract 0 16) i.link i.direction i.window dg).toOption

/-- The message a record's certificate signs, `certificateMessage` of its own fields. -/
def certMessage (r : Certified) : Option ByteArray :=
  (certificateMessage r.certificate.keyId r.certificate.link r.certificate.direction r.certificate.window
    r.certificate.digest).toOption

/-- The first certificate of the commitment that is a forgery: its message and signature, where the signature verifies
under the key over a message the device never signed. -/
def findForgery {p : Params} (c : Commitment) (dev : Device p) : Option (ByteArray × ByteArray) :=
  c.records.toList.findSome? fun r => (certMessage r).bind fun m =>
    if lmsAccepts p m r.certificate.signature && !(signedMessages dev).any (· == m) then
      some (m, r.certificate.signature)
    else none

/-- A message and a signature are an LMS forgery under the key: the signature verifies over a message the device never
signed. -/
def IsForgery {p : Params} (dev : Device p) : Option (ByteArray × ByteArray) → Prop
  | some (m, σ) => LmsVerifies p m σ ∧ ¬ Signed dev m
  | none => False

/-- The first pair of distinct strings of `xs` with one SHA-512 digest. -/
def findCollision : List ByteArray → Option (ByteArray × ByteArray)
  | [] => none
  | x :: xs =>
    match xs.find? fun y => !(x == y) && Flock.Sha512.hash x == Flock.Sha512.hash y with
    | some y => some (x, y)
    | none => findCollision xs

/-- `xy` is a SHA-512 collision: two distinct strings with one digest. -/
def IsCollision : Option (ByteArray × ByteArray) → Prop
  | some (x, y) => x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y
  | none => False

/-- What SHA-512 hashes for a record head: `tagged_sha512(RECORD_TAG, encode(h))`'s input. -/
def headStrings (h : RecordHead) : List ByteArray := (h.encode.toOption.map (taggedInput recordTag)).toList

/-- The node inputs `pairs lv` hashes on one level. -/
def pairInputs (lv : Nat) : List ByteArray → List ByteArray
  | a :: b :: rest => nodeInput a b :: pairInputs lv rest
  | [a] => [nodeInput a (empty lv)]
  | [] => []

/-- Level `k` of the tree over `level`, from level 0. -/
def levelAt : Nat → List ByteArray → List ByteArray
  | 0, level => level
  | k + 1, level => pairs k (levelAt k level)

/-- What SHA-512 hashes for the tree of `frames` under grid parameters `g`: each frame's leaf, every empty subtree up
to the depth, and each node of each level below the root. -/
def framesStrings (g : Grid) (frames : Array ByteArray) : List ByteArray :=
  frames.toList.filterMap (fun f => (leafInput f g.frameBytes).toOption) ++ [emptyInput] ++
    (List.range (depth g)).map (fun l => nodeInput (empty l) (empty l)) ++
    match leaves frames g with
    | .ok ls => (List.range (depth g)).flatMap fun lv => pairInputs lv (levelAt lv ls.toList)
    | .error _ => []

/-- What SHA-512 hashes for the frame tree of a grid: `framesStrings` of its real frames. -/
def treeStrings (g : Grid) (grid : Array (Array (Option ByteArray))) : List ByteArray :=
  match realFrames grid g with
  | .ok frames => framesStrings g frames
  | .error _ => []

/-- The syncs `link` declares, one per window, as the check reads them. -/
def declared (W : Nat) (link : Link) : Fin W → ClockSync := fun v => link.syncs.getD v.1 noSync

/-- The strings the run hashes: the check's (each record's head and certificate message; the frame tree and head it
recomputes of each egress link-window) and the device's (each issued head and its message; the frame tree of each
egress link-window). -/
def runStrings (p : Params) (c : Commitment) (e : Evidence) (dev : Device p) (eg : ByteArray) : List ByteArray :=
  c.records.toList.flatMap (fun r => headStrings r.head ++
    (messageInput r.certificate.keyId r.certificate.link r.certificate.direction r.certificate.window
      r.certificate.digest).toOption.toList) ++
  (p.egress.toList.zip e.links.toList).flatMap (fun q => (List.range p.windows).flatMap fun w =>
    treeStrings p.grid (programGrid p (declared p.windows q.2) q.2.frames w) ++
      (recordHead q.1 .egress w (programGrid p (declared p.windows q.2) q.2.frames w) none p.grid eg
        true).toOption.toList.flatMap headStrings) ++
  dev.issued.flatMap (fun i => headStrings i.head ++
    (i.head.digest.toOption.bind fun dg =>
      (messageInput (p.key.extract 0 16) i.link i.direction i.window dg).toOption).toList) ++
  (List.finRange p.egress.size).flatMap fun ℓ =>
    (List.range p.windows).flatMap fun w => treeStrings p.grid (deviceGrid dev ℓ w)

/-- **An accepted run's rows are the Program's (5a).** If the check accepts and the device is honest, then on every
egress link the device recorded, at every bucket of the `W` windows, exactly the row `expectedRow` gives of the link's
frames under its declared syncs, and it gave every link-window of the unit a complete status; or the run's strings hold
a SHA-512 collision, which `findCollision` finds, or one of its certificates is an LMS forgery, which `findForgery`
finds. -/
def AcceptedRowsAreProgramRows : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (dev : Device p) (eg ig : ByteArray),
    protocol.Accepts p c e → ParamsDigests p eg ig → Honest p dev eg →
    ((∀ (ℓ : Fin p.egress.size) link, e.links[ℓ.1]? = some link → ∀ syncs : Fin p.windows → ClockSync,
      Declares link syncs → ∀ t < p.windows * p.grid.buckets,
        dev.rows ℓ t = expectedRow p.grid.allocation p.grid.buckets p.windows syncs link.frames t) ∧
      ∀ d, ∀ l ∈ linkNames p d, ∀ w < p.windows, dev.status d l w = none) ∨
    IsCollision (findCollision (runStrings p c e dev eg)) ∨ IsForgery dev (findForgery c dev)

/-- **An accepted run is a run of the certifier program under the declared advice (5b).** Take an accepted unit, an
honest device, a reference sync set `ss` with Σ_sync's values, and a reference observation sequence `o` of the `W·T`
buckets whose rows are the device's and whose statuses are complete exactly where the device's link-window statuses are
(each boundary link `linkOf κ` one of the unit's). If the opened frames are the reference Program's (`frames` of the
operator's ingress and content advice `a`, on each egress link), then the reference `Accepted` holds of `o` for the
constant-rate Program with advice `(a, σ)`, `σ` the declared syncs, so Theorems 1(b) and 3
(`EncardAccSeqsLeConstantRate` and the rest) count `o`; or, as in `AcceptedRowsAreProgramRows`, `findCollision` finds a
SHA-512 collision among the run's strings or `findForgery` an LMS forgery among its certificates. -/
def AcceptedRunsAreProgramRuns : Prop :=
  ∀ (p : Params) (c : Commitment) (e : Evidence) (dev : Device p) (eg ig : ByteArray) {K : Nat} {In Ac : Type}
    (ss : NetTiming.SyncSet) (frames : List In → Ac → Fin p.egress.size → List NetTiming.LogicalFrame) (a : Ac)
    (op : Nat → List (NetTiming.Obs p.egress.size K p.grid.allocation) → In)
    (o : List (NetTiming.Obs p.egress.size K p.grid.allocation)) (linkOf : Fin K → Direction × String),
    protocol.Accepts p c e → ParamsDigests p eg ig → Honest p dev eg →
    (∀ δ, δ ∈ p.syncSet.deltas ↔ δ ∈ ss.deltas) → (∀ ρ, ρ ∈ p.syncSet.rhos ↔ ρ ∈ ss.rhos) →
    o.length = p.windows * p.grid.buckets →
    (∀ t (ht : t < o.length), o[t].rows = fun ℓ => dev.rows ℓ t) →
    (∀ t (ht : t < o.length) κ,
      o[t].status κ = .complete ↔ dev.status (linkOf κ).1 (linkOf κ).2 (t / p.grid.buckets) = none) →
    (∀ κ, (linkOf κ).2 ∈ linkNames p (linkOf κ).1) →
    (∀ t < p.windows * p.grid.buckets, ∀ (ℓ : Fin p.egress.size) link, e.links[ℓ.1]? = some link →
      (frames (NetTiming.ingressUpTo op o t) a ℓ).map (fun f => (f.payload, f.anchor, f.index)) =
        link.frames.map (fun f => (f.payload, f.anchor, f.index))) →
    (∃ σ : Fin p.windows → ss.Elem,
      (∀ (w : Fin p.windows) (ℓ : Fin p.egress.size) link, e.links[ℓ.1]? = some link →
        ∃ s, link.syncs[w.1]? = some s ∧ (σ w).1.1 = s.delta ∧ (σ w).2.1 = s.rho) ∧
      NetTiming.Accepted (NetTiming.constantRate (S := p.grid.allocation) p.grid.buckets p.windows ss frames) op
        (p.windows * p.grid.buckets) o (a, σ)) ∨
    IsCollision (findCollision (runStrings p c e dev eg)) ∨ IsForgery dev (findForgery c dev)

/-- `o` is the observation sequence of an accepted run `(c, e, dev, eg, ig)` with advice `a`: the check accepts the
unit, the device is honest, and `o`'s rows and statuses are the device's, read through `linkOf` as in
`AcceptedRunsAreProgramRuns`, with the opened frames `frames` of the operator's ingress and `a`. -/
def AcceptedRun (p : Params) {K : Nat} {In Ac : Type}
    (frames : List In → Ac → Fin p.egress.size → List NetTiming.LogicalFrame)
    (op : Nat → List (NetTiming.Obs p.egress.size K p.grid.allocation) → In) (linkOf : Fin K → Direction × String)
    (o : List (NetTiming.Obs p.egress.size K p.grid.allocation)) (c : Commitment) (e : Evidence) (dev : Device p)
    (eg ig : ByteArray) (a : Ac) : Prop :=
  protocol.Accepts p c e ∧ ParamsDigests p eg ig ∧ Honest p dev eg ∧
    (∀ t (ht : t < o.length), o[t].rows = fun ℓ => dev.rows ℓ t) ∧
    (∀ t (ht : t < o.length) κ,
      o[t].status κ = .complete ↔ dev.status (linkOf κ).1 (linkOf κ).2 (t / p.grid.buckets) = none) ∧
    ∀ t < p.windows * p.grid.buckets, ∀ (ℓ : Fin p.egress.size) link, e.links[ℓ.1]? = some link →
      (frames (NetTiming.ingressUpTo op o t) a ℓ).map (fun f => (f.payload, f.anchor, f.index)) =
        link.frames.map (fun f => (f.payload, f.anchor, f.index))

/-- The observation sequences of accepted runs, of the `W·T` buckets with jitter classes below `D`. -/
def AcceptedObservations (p : Params) {K : Nat} {In Ac : Type}
    (frames : List In → Ac → Fin p.egress.size → List NetTiming.LogicalFrame)
    (op : Nat → List (NetTiming.Obs p.egress.size K p.grid.allocation) → In) (linkOf : Fin K → Direction × String)
    (D : Nat) : Set (List (NetTiming.Obs p.egress.size K p.grid.allocation)) :=
  {o | o.length = p.windows * p.grid.buckets ∧ (∀ x ∈ o, x.jitter < D) ∧
    ∃ (c : Commitment) (e : Evidence) (dev : Device p) (eg ig : ByteArray) (a : Ac),
      AcceptedRun p frames op linkOf o c e dev eg ig a}

/-- **What accepted runs can carry (Theorem 3 for the check).** Over every accepted run under the hypotheses of
`AcceptedRunsAreProgramRuns`, the device's content is a function of the frames and the per-window sync advice
`Fin W → Σ_sync`, so the accepted observation sequences number at most `#Ac · |Σ_sync|^W · D^(W·T)`
(`ss.size = |Σ_sync|`): `NetTiming.Guarantees.EncardAccSeqsLeConstantRate` with `n = W·T`. Or some accepted run yields
a break: `findCollision` finds a SHA-512 collision among its strings, or `findForgery` an LMS forgery among its
certificates. -/
def EncardAcceptedObservationsLe : Prop :=
  ∀ (p : Params) {K : Nat} {In Ac : Type} [Fintype Ac] (ss : NetTiming.SyncSet)
    (frames : List In → Ac → Fin p.egress.size → List NetTiming.LogicalFrame)
    (op : Nat → List (NetTiming.Obs p.egress.size K p.grid.allocation) → In) (linkOf : Fin K → Direction × String)
    (D : Nat),
    (∀ δ, δ ∈ p.syncSet.deltas ↔ δ ∈ ss.deltas) → (∀ ρ, ρ ∈ p.syncSet.rhos ↔ ρ ∈ ss.rhos) →
    (∀ κ, (linkOf κ).2 ∈ linkNames p (linkOf κ).1) →
    (AcceptedObservations p frames op linkOf D).encard ≤
      ((Fintype.card Ac * ss.size ^ p.windows * D ^ (p.windows * p.grid.buckets) : ℕ) : ℕ∞) ∨
    ∃ o c e dev eg ig a, AcceptedRun p frames op linkOf o c e dev eg ig a ∧
      (IsCollision (findCollision (runStrings p c e dev eg)) ∨ IsForgery dev (findForgery c dev))

end Properties.NetworkCertifier
