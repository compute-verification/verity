import NetworkCheck.RecordsGrid
import Properties.NetworkCertifier

/-!
# The records check: what acceptance by its Lean code (`NetworkCheck.Records.protocol`) means

`verity/network-records/v1` is the network audit's verifier of record: `audit.py`'s `Audit` over raw, committed and
certified records. `AcceptMeans` is what its acceptance establishes about the unit's records, stated over the check's
inputs (the parameters, the published records, the evidence) and never over its internals: every link-window of the
unit has exactly one record, complete, and it is the Program's (`ProgramRecords`). `AnswerAcceptMeans` is the same for
the bytes `Core.Protocol.answer` decodes. The check keeps each payload as its bytes (`Records.ByteEvidence`), and the
conclusion reads that evidence as v2 reads the same JSON (`ByteEvidence.toEvidence`, each payload its bits, which
`RecordsBytes.evidenceOf_toEvidence` proves is `Check.evidenceOf`). `Properties.Proofs.NetworkRecords` proves both, and that the check accepts
every unit whose records are the Program's (`programRecords_accepted`, completeness).

The Program's grid of an egress link-window is `programGrid`, rows of `NetworkCertifier.expectedRow` under the syncs
the link declares, which the check reads through `Records.expectedGridsOfBytes` (`RecordsBytes.expectedGridsOfBytes_eq`). Nothing is
assumed of SHA-256, SHA-512 or LMS. A committed record's digest is the one `commitment.py` gives the Program's grid,
and any grid of `B`-byte frames with that digest holds the Program's frames, or two distinct strings the two digests
hash have one SHA-256 digest. A certified record's certificate verifies under the parameters' key (LMS's verifier
accepts it; that only the key's holder could sign is not claimed), and its head is the record head of the Program's
grid, or two distinct strings the two heads' digests hash have one SHA-512 digest.
-/

open NetworkCheck hiding Params
open NetworkCheck.Check (Params Certified Link Evidence Grid Profile)
open NetworkCertifier (ClockSync)
open Properties.NetworkCertifier (EvidenceFits ParamsDigests Declares SyncAdmitted programGrid LmsVerifies headStrings)

namespace Properties.NetworkRecords

/-! ## A link-window's records -/

/-- The published records naming link-window `(d, l, w)`, in the commitment's order: a raw or committed record of link
`l` and window `w`, which carries no direction, and a certified record whose head names `(d, l, w)`. -/
def recordsAt (c : Records.Commitment) (d : Direction) (l : String) (w : Nat) : List Records.Published :=
  c.records.toList.filter fun
    | .raw l' w' _ _ | .committed l' w' _ _ => decide (l' = l ∧ w' = w)
    | .certified r => decide (r.head.direction = d ∧ r.head.link = l ∧ r.head.window = w)

/-- The record's status is complete: in the clear for a raw or a committed record; signed, and settled by the
deadline, for a certified one. -/
def Complete : Records.Published → Prop
  | .raw _ _ s _ | .committed _ _ s _ => s = none
  | .certified r => r.head.status = none ∧ r.head.deadline = true

/-- `r`'s certificate is the key's (its identifier `I`), names `(d, l, w)`, signs the head's digest `dg`, and its LMS
signature verifies under the key over `certificateMessage` of them. -/
def CertificateVerifies (p : Params) (d : Direction) (l : String) (w : Nat) (r : Certified) (dg : ByteArray) : Prop :=
  r.head.digest = .ok dg ∧ r.certificate.digest = dg ∧ r.certificate.keyId = p.key.extract 0 16 ∧
    r.certificate.link = l ∧ r.certificate.direction = d ∧ r.certificate.window = w ∧
    ∃ m, certificateMessage (p.key.extract 0 16) l d w dg = .ok m ∧ LmsVerifies p m r.certificate.signature

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

/-- The canonical JSON header `record_digest` hashes for link-window `(l, w)` of a complete record under `g`. -/
def header (l : String) (w : Nat) (g : Grid) : Flock.V String :=
  Flock.canon (Lean.Json.mkObj [
    ("link", .str l), ("window", Lean.toJson w), ("buckets", Lean.toJson g.buckets),
    ("allocation", Lean.toJson g.allocation), ("frame_bytes", Lean.toJson g.frameBytes),
    ("bucket_ms", Lean.toJson g.bucketMs), ("status", .str "complete")])

/-- What `record_digest` hashes for a grid under header `h`: the header and every row's digest. -/
def recordInput (h : String) (grid : Records.RawGrid) : ByteArray :=
  taggedInput "verity-network-warden/record/v1"
    (grid.foldl (fun acc row => acc ++ Records.rowDigest row) (Flock.be32 h.toUTF8.size.toUInt32 ++ h.toUTF8))

/-- The strings SHA-256 hashes for the record digest of `grid` at link-window `(l, w)` under `g`: each row's input and
the record's. -/
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
its head, `dg` is the digest of `record_head` of `P` (complete, settled, under the egress parameters digest `eg`), and
its head is that head, or two distinct strings the two heads' digests hash have one SHA-512 digest. -/
def CertifiesProgram (p : Params) (eg : ByteArray) (l : String) (w : Nat) (r : Certified) (P : Records.RawGrid) :
    Prop :=
  ∃ dg H, CertificateVerifies p .egress l w r dg ∧ recordHead l .egress w P none p.grid eg true = .ok H ∧
    H.digest = .ok dg ∧
    (r.head = H ∨ ∃ x ∈ headStrings r.head ++ headStrings H, ∃ y ∈ headStrings r.head ++ headStrings H,
      x ≠ y ∧ Flock.Sha512.hash x = Flock.Sha512.hash y)

/-- The record of egress link-window `(l, w)` is the Program's, whose grid there is `P`: a raw record's grid is `P`
slot for slot, a committed record commits to `P`, a certified record certifies `P`. -/
def ProgramRecord (p : Params) (eg : ByteArray) (l : String) (w : Nat) (P : Records.RawGrid) :
    Records.Published → Prop
  | .raw _ _ _ grid => grid = P
  | .committed _ _ _ dg => CommitsTo p l w dg P
  | .certified r => CertifiesProgram p eg l w r P

/-- Egress link-window `(l, w)`, opened by `link`: exactly one record names it, its status is complete, the sync
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

/-- Ingress link-window `(l, w)`: exactly one record names it, its status is complete, and a certified one's head is
of the ingress grid and its parameters digest. -/
def IngressRecord (p : Params) (c : Records.Commitment) (ig : ByteArray) (l : String) (w : Nat) : Prop :=
  ∃ r, recordsAt c .ingress l w = [r] ∧ Complete r ∧ IngressHead p ig l w r

/-- **The unit's records are the Program's.** The evidence fits the parameters (one opening per egress link, `W`
declared syncs on each, every payload `8·B` bits); `eg` and `ig` are the parameters digests of the egress grid with
Σ_sync and of the ingress grid; every egress link-window `w < W` of every egress link has its one complete record, the
Program's (`EgressRecord`), and every ingress link-window its one complete record (`IngressRecord`). -/
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

end Properties.NetworkRecords
