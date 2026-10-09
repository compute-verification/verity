import NetworkCheck.Check
import Difftest

/-!
# The audit over every published record, a core protocol: `verity/network-records/v1`

`audit.py`'s `Audit` over its whole domain, `Published = Record | CommittedRecord | CertifiedRecord`, mixed lists
included: `check_link` on every egress link in order, then `check_ingress_link` on every ingress link. A certified record
is checked exactly as `verity/network-certificate/v2` checks it (`NetworkCheck.Check`'s `egressVerdict` and
`ingressVerdict` on that one record); a raw or committed record carries no direction, so it is matched by link and window
alone (`_by_window`), as one record of the link's egress windows and of its ingress windows alike.

**Params** and **evidence**: `verity/network-certificate/v2`'s, read by its decoders.

**Commitment**: the unit's published records, each `u8 kind ‖ u64be len(body) ‖ body`, concatenated, decoded exactly:

    kind 0, raw (`grid.Record`):       u16be len(link) ‖ link ‖ u64be window ‖ u8 status ‖ u32be T' ‖
                                       T' × (u32be r' ‖ r' × (u8 0 | u8 1 ‖ u32be len(frame) ‖ frame))
    kind 1, committed (`CommittedRecord`): u16be len(link) ‖ link ‖ u64be window ‖ u8 status ‖ u16be len(digest) ‖ digest
    kind 2, certified (`CertifiedRecord`): `service.encode`'s bytes, one record

UTF-8 link and digest, `STATUS_CODES` for the status, a raw grid of any shape (`u8 0` a padding slot, `u8 1` a frame of
any length), nothing left over in a body or after the last record. Its `digest` is
`tagged_sha512("verity-network-certifier/records/v1", the commitment's bytes)`.

**The verdict** of each link-window, in `Audit`'s order and with its names: `no-verdict` (no record), `duplicate` (more
than one); one certified record's verdict is v2's; one raw or committed record has `status:<reason>` when its status in
the clear is not complete, then on egress `advice` as v2, then `mismatch` when a raw record's grid is not the Program's
(`expected_grids`, slot for slot) or a committed record's digest is not `record_digest` of the Program's grid with a
complete status (`commitment.py`: SHA-256, padding elided). On ingress a raw or committed record is checked on its status
alone.

**Public** and **profile**: v2's. The `network-records` executable serves the protocol (`Core.Protocol.main`) and, as
`network-records verdicts`, every link-window's verdict, `Difftest.serve`'s `verdicts` on a `serve` case (`verdictsFn`).
-/

namespace NetworkCheck.Records

open Lean (Json toJson)
open NetworkCheck.Check (Grid Certified Link Evidence LinkWindow Profile)
open NetworkCertifier (ClockSync)

def protocolId : String := "verity/network-records/v1"
def commitmentTag : String := "verity-network-certifier/records/v1"

abbrev RawGrid := Array (Array (Option ByteArray))

/-- `Published`: a raw record (`grid.Record`), a committed one (`commitment.CommittedRecord`) or a certified one. -/
inductive Published where
  | raw (link : String) (window : Nat) (status : Option String) (grid : RawGrid)
  | committed (link : String) (window : Nat) (status : Option String) (digest : String)
  | certified (record : Certified)

structure Commitment where
  records : Array Published
  bytes : ByteArray

/-! ## Decoding -/

section Decoding

open Flock.Reader

private def be (n : Nat) : M Nat := do
  pure ((← bytes n).foldl (fun a x => a * 256 + x.toNat) 0)

private def utf8 (what : String) : M String := do
  match String.fromUTF8? (← bytes (← be 2)) with
  | some s => pure s
  | none => throw s!"{what} is not UTF-8"

private def status : M (Option String) := do
  let c ← be 1
  if c = 0 then return none
  match ["monitor-unavailable", "overflow", "malformed", "dropped", "delivery-refused",
      "coverage-never-established"][c - 1]? with
  | some r => pure (some r)
  | none => throw s!"not a status code: {c}"

private def slot : M (Option ByteArray) := do
  match ← be 1 with
  | 0 => pure none
  | 1 => pure (some (← bytes (← be 4)))
  | t => throw s!"a slot is 0 (padding) or 1 (a frame), not {t}"

/-- `n` items of `item`, each of which takes at least one byte, so a count past the bytes left fails on truncation. -/
private def many {α} (n : Nat) (item : M α) : M (Array α) := do
  let mut out := #[]
  for _ in [0:n] do out := out.push (← item)
  pure out

private def raw : M Published := do
  let link ← utf8 "a raw record's link"
  let window ← be 8
  let status ← status
  let grid ← many (← be 4) do many (← be 4) slot
  pure (.raw link window status grid)

private def committed : M Published := do
  let link ← utf8 "a committed record's link"
  let window ← be 8
  let status ← status
  pure (.committed link window status (← utf8 "a committed record's digest"))

private def whole {α} (body : ByteArray) (m : M α) : Flock.V α := do
  let (a, r) ← m.run ⟨body, 0⟩
  if r.pos ≠ body.size then throw "bytes left after a record"
  pure a

private def published : M Published := do
  let kind ← be 1
  let body ← bytes (← be 8)
  match kind with
  | 0 => whole body raw
  | 1 => whole body committed
  | 2 =>
    match (← Check.commitmentOf body).records with
    | #[r] => pure (.certified r)
    | rs => throw s!"a certified record's body holds one record, not {rs.size}"
  | k => throw s!"a record's kind is 0 (raw), 1 (committed) or 2 (certified), not {k}"

/-- The records, one after another until the bytes are spent; each takes at least nine bytes. -/
def commitmentOf (b : ByteArray) : Flock.V Commitment := do
  let records : M (Array Published) := do
    let mut out := #[]
    for _ in [0:b.size] do
      if (← get).pos ≥ b.size then break
      out := out.push (← published)
    pure out
  let (rs, r) ← records.run ⟨b, 0⟩
  if r.pos ≠ b.size then throw "bytes left after the records"
  pure ⟨rs, b⟩

end Decoding

/-! ## The committed record's digest (`commitment.py`) -/

/-- `tagged_sha256(tag, payload)`'s bytes: SHA-256 of `"veritor/tagged-sha256/v1\0" ‖ u32be len(tag) ‖ tag ‖
u64be len(payload) ‖ payload`. -/
def taggedSha256 (tag : String) (payload : ByteArray) : ByteArray :=
  Flock.Sha256.hash ("veritor/tagged-sha256/v1".toUTF8.push 0 ++ Flock.be32 tag.toUTF8.size.toUInt32 ++ tag.toUTF8 ++
    Flock.be64 payload.size.toUInt64 ++ payload)

/-- `row_digest` of a left-packed row: its count and its real frames, padding elided. -/
def rowDigest (row : Array (Option ByteArray)) : ByteArray :=
  let frames := row.filterMap id
  taggedSha256 "verity-network-warden/row/v1" (frames.foldl (· ++ ·) (Flock.be32 frames.size.toUInt32))

/-- `record_digest(link, window, grid, COMPLETE, params)` as `tagged_sha256` writes it, lowercase hex. -/
def committedDigest (link : String) (window : Nat) (grid : RawGrid) (g : Grid) : Flock.V String := do
  let header ← Flock.canon (Json.mkObj [
    ("link", .str link), ("window", toJson window), ("buckets", toJson g.buckets), ("allocation", toJson g.allocation),
    ("frame_bytes", toJson g.frameBytes), ("bucket_ms", toJson g.bucketMs), ("status", .str "complete")])
  let h := header.toUTF8
  pure (Flock.hexOf (taggedSha256 "verity-network-warden/record/v1"
    (grid.foldl (fun acc row => acc ++ rowDigest row) (Flock.be32 h.size.toUInt32 ++ h))))

/-! ## The check -/

def Published.link : Published → String
  | .raw l .. | .committed l .. => l
  | .certified r => r.head.link

/-- `_by_window`: the records of a link-window; a certified record of the other direction is another link-window's. -/
def found (c : Commitment) (d : Direction) (l : String) (w : Nat) : Array Published :=
  c.records.filter fun
    | .raw l' w' .. | .committed l' w' .. => l' == l && w' == w
    | .certified r => r.head.direction == d && r.head.link == l && r.head.window == (w : Int)

/-- One certified record as a v2 commitment, so that v2's verdict reads it. -/
def single (r : Certified) : Check.Commitment := ⟨#[r], .empty⟩

def statusViolation (s : Option String) : Option String := s.map (s!"status:{·}")

/-- An egress link-window's violation (`check_link`), none when it is accepted. -/
def egressVerdict (p : Check.Params) (c : Commitment) (e : Evidence) (egressDigest : ByteArray) (l : String) (link : Link)
    (w : Nat) : Option String :=
  match (found c .egress l w).toList with
  | [] => some "no-verdict"
  | [.certified r] => Check.egressVerdict p (single r) e egressDigest l link w
  | [r] =>
    let (st, grids) := match r with
      | .raw _ _ st grid => (st, fun expected => grid == expected)
      | .committed _ _ st digest => (st, fun expected => (committedDigest l w expected p.grid).toOption == some digest)
      | .certified _ => (none, fun _ => false)
    match statusViolation st with
    | some v => some v
    | none =>
      let s := link.syncs.getD w Check.noSync
      let u := Check.unitSync e w
      if !(Check.inSyncSet p.syncSet s && s.delta == u.delta && s.rho == u.rho) then some "advice"
      else if grids (Check.expectedGrid p.grid p.windows link w) then none
      else some "mismatch"
  | _ => some "duplicate"

/-- An ingress link-window's violation (`check_ingress_link`), none when it is accepted. -/
def ingressVerdict (p : Check.Params) (c : Commitment) (ingressDigest : ByteArray) (l : String) (w : Nat) : Option String :=
  match (found c .ingress l w).toList with
  | [] => some "no-verdict"
  | [.certified r] => Check.ingressVerdict p (single r) ingressDigest l w
  | [.raw _ _ st _] | [.committed _ _ st _] => statusViolation st
  | _ => some "duplicate"

/-- Every link-window's verdict, in `Audit`'s order: each egress link's windows, then each ingress link's. -/
def verdicts (p : Check.Params) (c : Commitment) (e : Evidence) (egressDigest ingressDigest : ByteArray) : Array LinkWindow :=
  (((p.egress.toList.zip e.links.toList).flatMap fun (l, link) => (List.range p.windows).map fun w =>
      ({ direction := .egress, link := l, window := w, violation := egressVerdict p c e egressDigest l link w } :
        LinkWindow)) ++
    p.ingress.toList.flatMap fun l => (List.range p.windows).map fun w =>
      ({ direction := .ingress, link := l, window := w, violation := ingressVerdict p c ingressDigest l w } :
        LinkWindow)).toArray

/-- Every link-window's verdict, or why the evidence or the parameters can't give them (v2's refusals). -/
def evaluate (p : Check.Params) (c : Commitment) (e : Evidence) : Except String (Array LinkWindow) :=
  if e.links.size ≠ p.egress.size then
    .error s!"evidence: {e.links.size} links' frames and syncs for {p.egress.size} egress links"
  else if e.links.any (·.syncs.size ≠ p.windows) then
    .error s!"evidence: a link declares other than {p.windows} syncs"
  else if e.links.any (·.frames.any (·.payload.length ≠ 8 * p.grid.frameBytes)) then
    .error s!"evidence: a frame of other than {p.grid.frameBytes} bytes"
  else
    match parametersDigest p.grid (some (Check.manifestSet p.syncSet)), parametersDigest p.ingressGrid none with
    | .ok eg, .ok ig => .ok (verdicts p c e eg ig)
    | _, _ => .error "the parameters digest has no canonical JSON"

def check (p : Check.Params) (c : Commitment) (e : Evidence) : Core.Outcome Unit Profile :=
  match evaluate p c e with
  | .error why => .reject why
  | .ok vs =>
    let named (v : LinkWindow) (why : String) := s!"{v.direction.value} link {v.link} window {v.window}: {why}"
    match vs.find? (·.violation.any (· != "no-verdict")) with
    | some v => .reject (named v (v.violation.getD ""))
    | none =>
      match vs.find? (·.violation.isSome) with
      | some v => .undecided (named v "no-verdict")
      | none =>
        let syncs := match e.links[0]? with
          | some l => l.syncs
          | none => #[]
        .accept () { verdicts := vs, windows := p.windows, syncs }

def protocol : Core.Protocol where
  id := protocolId
  Params := Check.Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Unit
  Profile := Profile
  params := Check.paramsOf
  commitment := commitmentOf
  evidence b := do Check.evidenceOf (← Core.canonicalJson b)
  digest c := (Identity.taggedSha512 commitmentTag c.bytes).toOption.getD .empty
  check := check
  publicJson _ := Json.mkObj []
  profileJson := Check.profileJson

def linkWindowJson (v : LinkWindow) : Json :=
  Json.mkObj [("direction", .str v.direction.value), ("link", .str v.link), ("verdict", .str (v.violation.getD "accept")),
    ("window", toJson v.window)]

/-- `verdicts` on a `serve` case: `{"link_windows": [...]}`, every link-window's verdict in the audit's order, or a refusal
named as `Core.Protocol.answer` names it (`case: `, `params: `, `commitment: `, `evidence: `). -/
def verdictsFn (j : Json) : Except String Json := do
  let parsed : Flock.V (Json × ByteArray × ByteArray) := do
    pure (← Flock.J.get j "params" "the case", ← Flock.unhex (← Flock.J.str j "commitment" "the case"),
      ← Flock.unhex (← Flock.J.str j "evidence" "the case"))
  let (pj, cb, eb) ← parsed.mapError (s!"case: {·}")
  let p ← (Check.paramsOf pj).mapError (s!"params: {·}")
  let c ← (commitmentOf cb).mapError (s!"commitment: {·}")
  let e ← (Core.canonicalJson eb >>= Check.evidenceOf).mapError (s!"evidence: {·}")
  let vs ← evaluate p c e
  pure (Json.mkObj [("link_windows", .arr (vs.map linkWindowJson))])

end NetworkCheck.Records
