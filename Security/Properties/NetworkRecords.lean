import NetworkCheck.RecordsGrid
import Properties.NetworkCertifier

/-!
# The records check: what acceptance by its Lean code (`NetworkCheck.Records.protocol`) means

`verity/network-records/v3` is the network audit's verifier of record: `audit.py`'s `Audit` over raw, committed and
certified records. `AcceptMeans` is what its acceptance establishes about the unit's records, stated over the check's
inputs (the parameters, the published records, the evidence) and never over its internals: every sampling block of the
unit has exactly one record, complete, and it is the Program's (`ProgramRecords`). `AnswerAcceptMeans` is the same for
the bytes `Core.Protocol.answer` decodes. The check keeps each payload as its bytes (`Records.ByteEvidence`), and the
conclusion reads that evidence as `verity/network-certificate/v4` reads the same JSON (`ByteEvidence.toEvidence`, each
payload its bits, which `RecordsBytes.evidenceOf_toEvidence` proves is `Check.evidenceOf`). `Proofs.Properties.NetworkRecords` proves both, and that the check accepts
every unit whose records are the Program's (`programRecords_accepted`, completeness).

Without the grid (`Records.evaluateNoGrid`, `check_link(grid=False)`, N2's, whose head check holds each certified
head's grid), the check's verdicts are `evaluate`'s but for the egress grid comparison (`GridOnly`): `NoGridMeans` says
that where it answers, `evaluate` answers too, and each sampling block has the same verdict or an egress `mismatch` that
it accepts; `NoGridAcceptsMore`, that where `evaluate` answers it answers so too, unless a raw or committed record names
an egress link, which it refuses. So it accepts every sampling block `evaluate` accepts.

The Program's grid of an egress sampling block is `programGrid`, rows of `NetworkCertifier.expectedRow` under the syncs
the link declares, which the check reads through `Records.expectedGridsOfBytes` (`RecordsBytes.expectedGridsOfBytes_eq`). Nothing is
assumed of SHA-256, SHA-512 or LMS. A committed record's digest is the one `commitment.py` gives the Program's grid,
and any grid of `B`-byte frames with that digest holds the Program's frames, or two distinct strings the two digests
hash have one SHA-256 digest. A certified record's certificate verifies under the parameters' key over its message in
their certifier session (LMS's verifier accepts it; that only the key's holder could sign is not claimed), and its head
is the record head of the Program's grid, or two distinct strings the two heads' digests hash have one SHA-512 digest.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check (Params Certified Link Evidence Grid Profile LinkWindow)
open NetworkCertifier (ClockSync)
open Properties.NetworkCertifier (EvidenceFits ParamsDigests Declares SyncAdmitted programGrid LmsVerifies headStrings)

namespace Properties.NetworkRecords

/-! ## A sampling block's records -/

/-- The published records naming sampling block `(d, l, w)`, in the commitment's order: a raw or committed record of
link `l` and window `w`, which carries no direction, and a certified record whose head names `(d, l, w)`. -/
def recordsAt (c : Records.Commitment) (d : Direction) (l : String) (w : Nat) : List Records.Published :=
  c.records.toList.filter fun
    | .raw l' w' _ _ | .committed l' w' _ _ => decide (l' = l ∧ w' = w)
    | .certified r => decide (r.head.direction = d ∧ r.head.link = l ∧ r.head.window = w)

/-- The record's status is complete: in the clear for a raw or a committed record; signed, and settled by the
deadline, for a certified one. -/
def Complete : Records.Published → Prop
  | .raw _ _ s _ | .committed _ _ s _ => s = none
  | .certified r => r.head.status = none ∧ r.head.deadline = true

/-- `r`'s certificate is the key's (its identifier `I`), names `(d, l, w)`, signs the head's digest `dg` under the
record's salt (`certificateDigest`), and its LMS signature verifies under the key over `certificateMessage` of them in
the parameters' certifier session. -/
def CertificateVerifies (p : Params) (d : Direction) (l : String) (w : Nat) (r : Certified) (dg : ByteArray) : Prop :=
  r.head.digest = .ok dg ∧ r.certificate.digest = certificateDigest r.salt dg ∧
    r.certificate.keyId = p.key.extract 0 16 ∧ r.certificate.link = l ∧ r.certificate.direction = d ∧
    r.certificate.window = w ∧
    ∃ m, certificateMessage (p.key.extract 0 16) p.session l d w (certificateDigest r.salt dg) = .ok m ∧
      LmsVerifies p m r.certificate.signature

/-! ## What a committed record's digest hashes (`commitment.py`) -/

/-- What `tagged_sha256(tag, payload)` hashes: `"veritor/tagged-sha256/v1\0" ‖ u32be len(tag) ‖ tag ‖
u64be len(payload) ‖ payload`. -/
def taggedInput (tag : String) (payload : ByteArray) : ByteArray :=
  "veritor/tagged-sha256/v1".toUTF8.push 0 ++ Flock.be32 tag.toUTF8.size.toUInt32 ++ tag.toUTF8 ++
    Flock.be64 payload.size.toUInt64 ++ payload

/-- What `row_digest` hashes for a row: its count and its real frames, padding elided. -/
def rowInput (row : Array (Option ByteArray)) : ByteArray :=
  taggedInput "verity-network-warden/row/v1"
    ((row.filterMap id).foldl (· ++ ·) (Flock.be32 (row.filterMap id).size.toUInt32))

/-- The canonical JSON header `record_digest` hashes for sampling block `(l, w)` of a complete record under `g`. -/
def header (l : String) (w : Nat) (g : Grid) : Flock.V String :=
  Flock.canon (Lean.Json.mkObj [
    ("link", .str l), ("window", Lean.toJson w), ("buckets", Lean.toJson g.buckets),
    ("allocation", Lean.toJson g.allocation), ("frame_bytes", Lean.toJson g.frameBytes),
    ("bucket_ms", Lean.toJson g.bucketMs), ("status", .str "complete")])

/-- What `record_digest` hashes for a grid under header `h`: the header and every row's digest. -/
def recordInput (h : String) (grid : Records.RawGrid) : ByteArray :=
  taggedInput "verity-network-warden/record/v1"
    (grid.foldl (fun acc row => acc ++ Records.rowDigest row) (Flock.be32 h.toUTF8.size.toUInt32 ++ h.toUTF8))

/-- The strings SHA-256 hashes for the record digest of `grid` at sampling block `(l, w)` under `g`: each row's input
and the record's. -/
def committedInputs (l : String) (w : Nat) (grid : Records.RawGrid) (g : Grid) : List ByteArray :=
  grid.toList.map rowInput ++ (header l w g).toOption.toList.map (recordInput · grid)

/-- A grid's frames, row by row, padding elided: what its record digest commits to. -/
def frames (grid : Records.RawGrid) : List (List ByteArray) := grid.toList.map fun row => (row.filterMap id).toList

/-- `grid`'s frames are `B ≥ 1` bytes each. -/
def FramesOf (B : Nat) (grid : Records.RawGrid) : Prop :=
  0 < B ∧ ∀ row ∈ grid.toList, ∀ f ∈ (row.filterMap id).toList, f.size = B

/-! ## A record is the Program's -/

/-- A committed record of `(l, w)` with digest `dg` commits to `P`: `dg` is the record digest of `P`, and every grid of
`B`-byte frames with that digest holds `P`'s frames row by row, or two distinct strings the two digests hash have one
SHA-256 digest. -/
def CommitsTo (p : Params) (l : String) (w : Nat) (dg : String) (P : Records.RawGrid) : Prop :=
  Records.committedDigest l w P p.grid = .ok dg ∧
    ∀ G : Records.RawGrid, FramesOf p.grid.frameBytes G → Records.committedDigest l w G p.grid = .ok dg →
      frames G = frames P ∨
        ∃ x ∈ committedInputs l w G p.grid ++ committedInputs l w P p.grid,
          ∃ y ∈ committedInputs l w G p.grid ++ committedInputs l w P p.grid,
            x ≠ y ∧ Flock.Sha256.hash x = Flock.Sha256.hash y

/-- A certified egress record of `(l, w)` certifies `P`: its certificate verifies under the key over the digest `dg` of
its head under its salt, `dg` is the digest of `record_head` of `P` (complete, settled, under the egress parameters digest `eg`), and
its head is that head, or two distinct strings the two heads' digests hash have one SHA-512 digest. -/
def CertifiesProgram (p : Params) (eg : ByteArray) (l : String) (w : Nat) (r : Certified) (P : Records.RawGrid) :
    Prop :=
  ∃ dg H, CertificateVerifies p .egress l w r dg ∧ recordHead l .egress w P none p.grid eg true = .ok H ∧
    H.digest = .ok dg ∧
    (r.head = H ∨ ∃ x ∈ headStrings r.head ++ headStrings H, ∃ y ∈ headStrings r.head ++ headStrings H,
      x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y)

/-- The record of egress sampling block `(l, w)` is the Program's, whose grid there is `P`: a raw record's grid is `P`
slot for slot, a committed record commits to `P`, a certified record certifies `P`. -/
def ProgramRecord (p : Params) (eg : ByteArray) (l : String) (w : Nat) (P : Records.RawGrid) :
    Records.Published → Prop
  | .raw _ _ _ grid => grid = P
  | .committed _ _ _ dg => CommitsTo p l w dg P
  | .certified r => CertifiesProgram p eg l w r P

/-- Egress sampling block `(l, w)`, opened by `link`: exactly one record names it, its status is complete, the sync
`link` declares for `w` is in Σ_sync and is the unit's (the first egress link's), and the record is the Program's
under the declared syncs. -/
def EgressRecord (p : Params) (c : Records.Commitment) (e : Evidence) (eg : ByteArray) (l : String) (link : Link)
    (w : Nat) : Prop :=
  ∃ r, recordsAt c .egress l w = [r] ∧ Complete r ∧ SyncAdmitted p e link w ∧
    ∀ syncs : Fin p.windows → ClockSync, Declares link syncs → ProgramRecord p eg l w (programGrid p syncs link.frames w) r

/-- A certified ingress record of `(l, w)` verifies under the key, and its head is of the ingress grid (T, r, B, τ_b)
under its parameters digest `ig`; a raw or committed record carries nothing more to check. -/
def IngressHead (p : Params) (ig : ByteArray) (l : String) (w : Nat) : Records.Published → Prop
  | .raw .. | .committed .. => True
  | .certified r => ∃ dg, CertificateVerifies p .ingress l w r dg ∧ r.head.buckets = (p.ingressGrid.buckets : Int) ∧
      r.head.allocation = (p.ingressGrid.allocation : Int) ∧ r.head.frameBytes = (p.ingressGrid.frameBytes : Int) ∧
      r.head.bucketMs = (p.ingressGrid.bucketMs : Int) ∧ r.head.parameters = ig

/-- Ingress sampling block `(l, w)`: exactly one record names it, its status is complete, and a certified one's head is
of the ingress grid and its parameters digest. -/
def IngressRecord (p : Params) (c : Records.Commitment) (ig : ByteArray) (l : String) (w : Nat) : Prop :=
  ∃ r, recordsAt c .ingress l w = [r] ∧ Complete r ∧ IngressHead p ig l w r

/-- **The unit's records are the Program's.** The evidence fits the parameters (one opening per egress link, `W`
declared syncs on each, every payload `8·B` bits); `eg` and `ig` are the parameters digests of the egress grid with
Σ_sync and of the ingress grid; every egress sampling block `w < W` of every egress link has its one complete record,
the Program's (`EgressRecord`), and every ingress sampling block its one complete record (`IngressRecord`). -/
def ProgramRecords (p : Params) (c : Records.Commitment) (e : Evidence) : Prop :=
  EvidenceFits p e ∧ ∃ eg ig, ParamsDigests p eg ig ∧
    (∀ l link, (l, link) ∈ p.egress.toList.zip e.links.toList → ∀ w < p.windows, EgressRecord p c e eg l link w) ∧
    ∀ l ∈ p.ingress.toList, ∀ w < p.windows, IngressRecord p c ig l w

/-! ## Acceptance -/

/-- **What the records check's acceptance means.** If `NetworkCheck.Records.protocol` accepts the parameters, the
published records and the evidence `e`, the unit's records are the Program's under `e`, its payloads read as bits. -/
def AcceptMeans : Prop :=
  ∀ (p : Params) (c : Records.Commitment) (e : Records.ByteEvidence) (pub : Unit) (prof : Profile),
    Records.protocol.check p c e = .accept pub prof → ProgramRecords p c e.toEvidence

/-- **The same, from the bytes.** If `answer` accepts parameters `a`, commitment `b` and evidence `v`, they decode, and
the records they decode to are the Program's. -/
def AnswerAcceptMeans : Prop :=
  ∀ a b v : ByteArray, (Records.protocol.answer a b v).verdict = .accept →
    ∃ (p : Params) (c : Records.Commitment) (e : Records.ByteEvidence),
      (Core.canonicalJson a >>= Records.protocol.params) = .ok p ∧
      Records.protocol.commitment b = .ok c ∧ Records.protocol.evidence v = .ok e ∧ ProgramRecords p c e.toEvidence

/-! ## Without the grid (`check_link(grid=False)`, N2's) -/

/-- Sampling block `v'` is `v` but for the egress grid comparison: the same sampling block, with the same verdict, or
`v` an egress `mismatch` that `v'` accepts. So `v'` accepts every sampling block `v` accepts. -/
def GridOnly (v v' : LinkWindow) : Prop :=
  v'.direction = v.direction ∧ v'.link = v.link ∧ v'.window = v.window ∧
    (v'.violation = v.violation ∨ v.direction = .egress ∧ v.violation = some "mismatch" ∧ v'.violation = none)

/-- `vs'` is `vs` but for the egress grid comparison: as many sampling blocks, each `GridOnly` the one at its place in
`vs`. -/
def AllGridOnly (vs vs' : Array LinkWindow) : Prop :=
  vs.size = vs'.size ∧ ∀ x ∈ vs.toList.zip vs'.toList, GridOnly x.1 x.2

/-- A raw or committed record names an egress link of the unit: a record whose grid nothing but a comparison with the
Program's checks. -/
def UncertifiedEgress (p : Params) (c : Records.Commitment) : Prop :=
  ∃ r ∈ c.records.toList, match r with
    | .raw l .. | .committed l .. => l ∈ p.egress.toList
    | .certified _ => False

/-- **What the check without the grid answers.** If `evaluateNoGrid` gives verdicts `vs'`, no raw or committed record
names an egress link, and `evaluate`, the check with the grid compared, gives verdicts that `vs'` is but for the egress
grid comparison: the same sampling blocks in order, each with the same verdict, or an egress `mismatch` accepted. -/
def NoGridMeans : Prop :=
  ∀ (p : Params) (c : Records.Commitment) (e : Records.ByteEvidence) (vs' : Array LinkWindow),
    Records.evaluateNoGrid p c e = .ok vs' →
      ¬ UncertifiedEgress p c ∧ ∃ vs, Records.evaluate p c e = .ok vs ∧ AllGridOnly vs vs'

/-- **The check without the grid accepts more.** If `evaluate` gives verdicts `vs`, `evaluateNoGrid` refuses the unit
when a raw or committed record names an egress link, and otherwise gives verdicts that are `vs` but for the egress grid
comparison, so it accepts every sampling block `evaluate` accepts. -/
def NoGridAcceptsMore : Prop :=
  ∀ (p : Params) (c : Records.Commitment) (e : Records.ByteEvidence) (vs : Array LinkWindow),
    Records.evaluate p c e = .ok vs →
      UncertifiedEgress p c ∧ (∃ why, Records.evaluateNoGrid p c e = .error why) ∨
        ∃ vs', Records.evaluateNoGrid p c e = .ok vs' ∧ AllGridOnly vs vs'

end Properties.NetworkRecords
