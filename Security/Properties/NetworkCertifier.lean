import NetworkCheck.Check
import Properties.Identity

/-!
# The network certificate check: what its Lean code (`NetworkCheck.Check`) does

The properties of `NetworkCheck.protocol` (`verity/network-certificate/v3`, the port of `audit.py`'s `Audit`), stated
against its functions as they run, each in one shape: for every input, if the function accepts (or, for a refusal,
doesn't reject; for the undecided outcome, is undecided), its named conclusion holds. They say what `answer` and the
check refuse, what an acceptance establishes about each link-window and what an undecided outcome means.
`Properties.Proofs.NetworkCertifier` proves them; the converses (that the check accepts, or is undecided on, what
these conclusions describe) and the honest transcript's acceptance are completeness, at the end of
`Properties.Proofs.NetworkCertifier.Outcomes`, each with one instance in `Properties.Proofs.NetworkCertifier.Instance`. What an accepted run of the certifier device is, is
`Properties.CertifierNetwork.Runs`.

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
names `(d, l, w)`, signs the head's digest `dg`, and verifies under the key over `certificateMessage` of them in the
parameters' certifier session. -/
def Signs (p : Params) (c : Commitment) (d : Direction) (l : String) (w : Nat) (r : Certified) (dg : ByteArray) :
    Prop :=
  recordsOf c d l w = [r] ∧ r.head.digest = .ok dg ∧ r.certificate.digest = dg ∧
    r.certificate.keyId = p.key.extract 0 16 ∧ r.certificate.link = l ∧ r.certificate.direction = d ∧
    r.certificate.window = w ∧
    ∃ m, certificateMessage (p.key.extract 0 16) p.session l d w dg = .ok m ∧ LmsVerifies p m r.certificate.signature

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

/-- The three byte strings decode: the parameters as canonical JSON and then as parameters, the commitment and the
evidence. -/
def Decodes (a b c : ByteArray) : Prop :=
  (∃ p, (Core.canonicalJson a >>= protocol.params) = .ok p) ∧ (∃ cm, protocol.commitment b = .ok cm) ∧
    ∃ e, protocol.evidence c = .ok e

/-- **`answer` rejects bytes that a decoder refuses**, stated as its contrapositive: whatever `answer` doesn't reject
decodes (parameters that are canonical JSON and parameters, a commitment, evidence that is canonical JSON and
evidence). -/
def DecodeRefusals : Prop :=
  ∀ a b c : ByteArray, (protocol.answer a b c).verdict ≠ .reject → Decodes a b c

/-- The three byte strings decode, and the check accepts what they decode to. -/
def DecodedAccepted (a b c : ByteArray) : Prop :=
  ∃ p cm e, (Core.canonicalJson a >>= protocol.params) = .ok p ∧ protocol.commitment b = .ok cm ∧
    protocol.evidence c = .ok e ∧ protocol.Accepts p cm e

/-- **What `answer` accepts decodes, and the check accepts it.** (Its converse, that `answer` accepts such bytes, is
completeness: `answer_of_decodedAccepted`.) -/
def AnswerAccepts : Prop :=
  ∀ a b c : ByteArray, (protocol.answer a b c).verdict = .accept → DecodedAccepted a b c

/-- **Evidence that doesn't fit the parameters is a rejection**, stated as its contrapositive: evidence the check
doesn't reject fits. -/
def MalformedEvidenceRejects : Prop :=
  ∀ (p : Params) c e, (∀ why, protocol.check p c e ≠ .reject why) → EvidenceFits p e

/-- The parameters have their two parameters digests. -/
def ParamsDigested (p : Params) : Prop := ∃ eg ig, ParamsDigests p eg ig

/-- **Parameters without a parameters digest (no canonical JSON) are a rejection**, stated as its contrapositive:
parameters the check doesn't reject have their digests. -/
def UndigestedParamsReject : Prop :=
  ∀ (p : Params) c e, (∀ why, protocol.check p c e ≠ .reject why) → ParamsDigested p

/-- Every link-window of the unit has exactly one record. -/
def OneRecordEach (p : Params) (c : Commitment) : Prop :=
  ∀ d, ∀ l ∈ linkNames p d, ∀ w < p.windows, (recordsOf c d l w).length = 1

/-- **An accepted unit has exactly one record of every link-window.** -/
def AcceptsOneRecordEach : Prop :=
  ∀ (p : Params) c e, protocol.Accepts p c e → OneRecordEach p c

/-! ## (2) What an acceptance establishes -/

/-- Every link-window of the unit is accepted: the evidence fits, the parameters have their digests, every egress
link-window is `EgressAccepted` and every ingress link-window `IngressAccepted`. -/
def LinkWindowsAccepted (p : Params) (c : Commitment) (e : Evidence) : Prop :=
  EvidenceFits p e ∧ ∃ eg ig, ParamsDigests p eg ig ∧
    (∀ l link, (l, link) ∈ p.egress.toList.zip e.links.toList → ∀ w < p.windows, EgressAccepted p c e eg l link w) ∧
    ∀ l ∈ p.ingress.toList, ∀ w < p.windows, IngressAccepted p c ig l w

/-- **An accepted unit has every link-window accepted** (`LinkWindowsAccepted`). (Its converse, that the check accepts
such a unit, is completeness: `accepts_of_linkWindowsAccepted`.) -/
def AcceptsLinkWindows : Prop :=
  ∀ (p : Params) c e, protocol.Accepts p c e → LinkWindowsAccepted p c e

/-! ## (3) Undecided -/

/-- No link-window is refused and some awaits its record: the evidence fits, the parameters have their digests, every
link-window is accepted or `Pending`, and some is `Pending`. -/
def SomePending (p : Params) (c : Commitment) (e : Evidence) : Prop :=
  EvidenceFits p e ∧ ∃ eg ig, ParamsDigests p eg ig ∧
    (∀ l link, (l, link) ∈ p.egress.toList.zip e.links.toList → ∀ w < p.windows,
      EgressAccepted p c e eg l link w ∨ Pending p c .egress l w) ∧
    (∀ l ∈ p.ingress.toList, ∀ w < p.windows, IngressAccepted p c ig l w ∨ Pending p c .ingress l w) ∧
    ∃ d, ∃ l ∈ linkNames p d, ∃ w < p.windows, Pending p c d l w

/-- **An undecided unit has no link-window refused and some awaiting its record** (`SomePending`). (Its converse, that
such a unit is undecided, is completeness: `undecided_of_somePending`.) -/
def UndecidedPending : Prop :=
  ∀ (p : Params) c e, (∃ why, protocol.check p c e = .undecided why) → SomePending p c e

/-! ## The strings the check hashes

What an accepted run's claims search for a SHA-512 collision (`Properties.CertifierNetwork.Runs`): the check's part. -/

/-- The message a record's certificate signs in certifier session `session`, `certificateMessage` of its own fields. -/
def certMessage (session : ByteArray) (r : Certified) : Option ByteArray :=
  (certificateMessage r.certificate.keyId session r.certificate.link r.certificate.direction r.certificate.window
    r.certificate.digest).toOption

/-- What SHA-512 hashes for a record head: `tagged_sha512(RECORD_TAG, encode(h))`'s input. -/
def headStrings (h : RecordHead) : List ByteArray :=
  (h.encode.toOption.bind fun enc => (Properties.Identity.taggedInput recordTag enc).toOption).toList

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

/-- The strings the check hashes under egress parameters digest `eg`: each record's head and certificate message, and
the frame tree and head it recomputes of each egress link-window. -/
def checkStrings (p : Params) (c : Commitment) (e : Evidence) (eg : ByteArray) : List ByteArray :=
  c.records.toList.flatMap (fun r => headStrings r.head ++
    (messageInput r.certificate.keyId p.session r.certificate.link r.certificate.direction r.certificate.window
      r.certificate.digest).toOption.toList) ++
  (p.egress.toList.zip e.links.toList).flatMap fun q => (List.range p.windows).flatMap fun w =>
    treeStrings p.grid (programGrid p (declared p.windows q.2) q.2.frames w) ++
      (recordHead q.1 .egress w (programGrid p (declared p.windows q.2) q.2.frames w) none p.grid eg
        true).toOption.toList.flatMap headStrings

end Properties.NetworkCertifier
