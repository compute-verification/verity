import NetworkCheck.Check
import Difftest

/-!
# The audit over every published record, a core protocol: `verity/network-records/v3`

`audit.py`'s `Audit` over its whole domain, `Published = Record | CommittedRecord | CertifiedRecord`, mixed lists
included: `check_link` on every egress link in order, then `check_ingress_link` on every ingress link. A certified record
is checked exactly as `verity/network-certificate/v4` checks it (`NetworkCheck.Check`'s `egressVerdict`, with the
Program's grid from `expectedGrids`, and `ingressVerdict` on that one record); a raw or committed record carries no direction, so it is matched by link and window
alone (`_by_window`), as one record of the link's egress windows and of its ingress windows alike.

**Params** and **evidence**: `verity/network-certificate/v4`'s, read by its decoders; the evidence keeps each payload as
its bytes (`ByteEvidence`, which `NetworkCheck.RecordsBytes` proves is v4's reading of the same JSON), since v4's bits
take a cons cell per bit.

**Commitment**: the unit's published records, each `u8 kind ‖ u64be len(body) ‖ body`, concatenated, decoded exactly:

    kind 0, raw (`grid.Record`):       u16be len(link) ‖ link ‖ u64be window ‖ u8 status ‖ u32be T' ‖
                                       T' × (u32be r' ‖ r' × (u8 0 | u8 1 ‖ u32be len(frame) ‖ frame))
    kind 1, committed (`CommittedRecord`): u16be len(link) ‖ link ‖ u64be window ‖ u8 status ‖ u16be len(digest) ‖ digest
    kind 2, certified (`CertifiedRecord`): `service.encode`'s bytes, one record

UTF-8 link and digest, `STATUS_CODES` for the status, a raw grid of any shape (`u8 0` a padding slot, `u8 1` a frame of
any length), nothing left over in a body or after the last record. Its `digest` is
`tagged_sha512("verity-network-certifier/records/v2", the commitment's bytes)`.

**The verdict** of each link-window, in `Audit`'s order and with its names: `no-verdict` (no record), `duplicate` (more
than one); one certified record's verdict is v4's; one raw or committed record has `status:<reason>` when its status in
the clear is not complete, then on egress `advice` as v4, then `mismatch` when a raw record's grid is not the Program's
(`expected_grids`, slot for slot) or a committed record's digest is not `record_digest` of the Program's grid with a
complete status (`commitment.py`: SHA-256, padding elided). On ingress a raw or committed record is checked on its status
alone.

**Without the grid** (`check_link(grid=False)`, N2's, whose head check holds each certified head's grid):
`evaluateNoGrid` gives each egress link-window `egressVerdictNoGrid`'s verdict, every test above but the grid comparison,
in the same order and with the same names, and it refuses a raw or committed egress record, whose grid nothing else
holds. Its refusals are otherwise `evaluate`'s, and its ingress verdicts the same.

**Public** and **profile**: v4's. The `network-records` executable serves the protocol (`Core.Protocol.main`) and, as
`network-records verdicts`, every link-window's verdict on a `serve` case through `Difftest.serve`: `verdicts`
(`verdictsFn`) and `verdicts-no-grid` (`verdictsNoGridFn`).
-/

namespace NetworkCheck.Records

open Lean (Json toJson)
open NetworkCheck.Check (Grid Certified Link Evidence LinkWindow Profile)
open NetworkCertifier (ClockSync)

def protocolId : String := "verity/network-records/v3"
def commitmentTag : String := "verity-network-certifier/records/v2"

abbrev RawGrid := Array (Array (Option ByteArray))

/-- `Published`: a raw record (`grid.Record`), a committed one (`commitment.CommittedRecord`) or a certified one. -/
inductive Published where
  | raw (link : String) (window : Nat) (status : Option String) (grid : RawGrid)
  | committed (link : String) (window : Nat) (status : Option String) (digest : String)
  | certified (record : Certified)

structure Commitment where
  records : Array Published
  bytes : ByteArray

/-- A logical frame of the evidence with its payload kept as bytes. `toLogical` is the frame as `Check.evidenceOf` reads
it, `NetworkCertifier.LogicalFrame` with the payload's bits, most significant first. -/
structure ByteFrame where
  payload : ByteArray
  anchor : Nat
  index : Nat

def ByteFrame.toLogical (f : ByteFrame) : NetworkCertifier.LogicalFrame := ⟨Check.bitsOfBytes f.payload, f.anchor, f.index⟩

/-- The frame without its content: when the shaper hands it (`eligible`) reads only its anchor and its index. -/
def ByteFrame.timing (f : ByteFrame) : NetworkCertifier.LogicalFrame := ⟨[], f.anchor, f.index⟩

/-- `Check.Link` with each payload kept as bytes. -/
structure ByteLink where
  frames : List ByteFrame
  syncs : Array ClockSync

def ByteLink.toLink (l : ByteLink) : Link := ⟨l.frames.map ByteFrame.toLogical, l.syncs⟩

/-- `Check.Evidence` with each payload kept as bytes; `toEvidence` is `Check.evidenceOf`'s reading of the same JSON
(`RecordsBytes.evidenceOf_toEvidence`). -/
structure ByteEvidence where
  links : Array ByteLink

def ByteEvidence.toEvidence (e : ByteEvidence) : Evidence := ⟨e.links.map ByteLink.toLink⟩

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

/-- `Flock.unhex` of the characters `cs`, there `s.toList.toArray`. -/
def unhexChars (cs : Array Char) : Flock.V ByteArray := do
  if cs.size % 2 != 0 then throw s!"odd-length hex"
  let v := fun (c : Char) =>
    if c.isDigit then some (c.toNat - 48) else if 'a' ≤ c && c ≤ 'f' then some (c.toNat - 87) else none
  let mut b := ByteArray.emptyWithCapacity (cs.size / 2)
  for i in [0:cs.size / 2] do
    match v (cs.getD (2 * i) ' '), v (cs.getD (2 * i + 1) ' ') with
    | some h, some l => b := b.push (h * 16 + l).toUInt8
    | _, _ => throw "not lowercase hex"
  return b

/-- `Flock.unhex` (`RecordsBytes.unhex_eq`), its characters read into an array with no list between, 8 bytes a character
rather than 32: the evidence a case carries is hex of hex. -/
def unhex (s : String) : Flock.V ByteArray := unhexChars (s.foldl Array.push (Array.emptyWithCapacity s.length))

/-- `Check.evidenceOf`'s frame, its payload's bytes as they are. -/
def byteFrameOf (j : Json) : Flock.V ByteFrame := do
  let #[p, a, i] ← Flock.J.asArr j "a frame" | throw "a frame is [payload, anchor, index]"
  pure ⟨← unhex (← Flock.J.asStr p "a frame's payload"), ← Flock.J.asNat a "an anchor",
    ← Flock.J.asNat i "an index"⟩

/-- `Check.evidenceOf`, each payload's bytes as they are. -/
def evidenceOf (j : Json) : Flock.V ByteEvidence := do
  let links ← (← Flock.J.arr j "links" "the object").mapM fun l => do
    pure { frames := (← (← Flock.J.arr l "frames" "a link").mapM byteFrameOf).toList,
           syncs := ← (← Flock.J.arr l "syncs" "a link").mapM Check.syncOf : ByteLink }
  pure ⟨links⟩

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

/-- `record_digest(link, window, grid, COMPLETE, params)` as `tagged_sha256` writes it, lowercase hex. Its header is in
the identity form (`Flock.canonUtf8`, `canonical_json_bytes`), a link's characters past ASCII in UTF-8. -/
def committedDigest (link : String) (window : Nat) (grid : RawGrid) (g : Grid) : Flock.V String := do
  let header ← Flock.canonUtf8 (Json.mkObj [
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

/-- One certified record as a v4 commitment, so that v4's verdict reads it. -/
def single (r : Certified) : Check.Commitment := ⟨#[r], .empty⟩

def statusViolation (s : Option String) : Option String := s.map (s!"status:{·}")

/-- `handingFrom` with the hand times so far reversed in `acc`, so a run of any length takes no stack. -/
def handingRev (T W : Nat) (syncs : Fin W → ClockSync) : Nat → List NetworkCertifier.LogicalFrame → List Nat → List Nat
  | _, [], acc => acc
  | last, f :: fs, acc =>
    let h := if hw : f.anchor / T < W then max (NetworkCertifier.eligible f (syncs ⟨f.anchor / T, hw⟩)) last
      else max (max last (W * T)) f.anchor
    handingRev T W syncs h fs (h :: acc)

/-- The enqueue counts of buckets `0 … H − 1`: bucket `t` counts the hand times `t` of `hs`. -/
def enqueueCounts (H : Nat) (hs : List Nat) : Array Nat :=
  hs.foldl (fun a h => a.modify h (· + 1)) (Array.replicate H 0)

/-- The FIFO's release counts of `c` buckets from bucket `t` with `q` frames queued: `released` and `queue`, carried. -/
def releases (S : Nat) (enq : Nat → Nat) : Nat → Nat → Nat → List Nat
  | _, _, 0 => []
  | t, q, c + 1 => min (q + enq t) S :: releases S enq (t + 1) (q + enq t - min (q + enq t) S) c

/-- `fill` of the buckets releasing `ns`, the first from the `k`-th frame on: a bucket releasing `n` holds frames in slots
`0 … n − 1`. -/
def rows (S : Nat) (pay : Nat → ByteArray) : Nat → List Nat → List (Array (Option ByteArray))
  | _, [] => []
  | k, n :: ns => Array.ofFn (fun s : Fin S => if s.1 < n then some (pay (k + s.1)) else none) :: rows S pay (k + n) ns

/-- `expected_grids(frames, syncs, params)`: the link's grid in each of the `W` windows, as `schedule.py` computes it, in
time linear in the frames and the buckets: the hand times, the enqueue counts, the release counts, and the frames filled
in order. Window `w`'s is `Check.expectedGrid`'s (`RecordsGrid.expectedGrids_eq`). -/
def expectedGrids (g : Grid) (W : Nat) (l : Link) : Array RawGrid :=
  let T := g.buckets
  let syncs : Fin W → ClockSync := fun v => l.syncs.getD v.1 Check.noSync
  let enq := enqueueCounts (W * T) (handingRev T W syncs 0 l.frames []).reverse
  let payloads := l.frames.toArray.map fun f => Check.bytesOfBits f.payload
  let all := (rows g.allocation (fun k => payloads.getD k (Check.bytesOfBits [])) 0
    (releases g.allocation (fun t => enq.getD t 0) 0 0 (W * T))).toArray
  (Array.range W).map fun w => all.extract (w * T) (w * T + T)

/-- `expectedGrids` of the link as `Check.evidenceOf` reads it, from its payloads' bytes: the hand times from the frames'
anchors and indices, the payloads as they are (`RecordsBytes.expectedGridsOfBytes_eq`). -/
def expectedGridsOfBytes (g : Grid) (W : Nat) (l : ByteLink) : Array RawGrid :=
  let T := g.buckets
  let syncs : Fin W → ClockSync := fun v => l.syncs.getD v.1 Check.noSync
  let enq := enqueueCounts (W * T) (handingRev T W syncs 0 (l.frames.map ByteFrame.timing) []).reverse
  let payloads := l.frames.toArray.map ByteFrame.payload
  let all := (rows g.allocation (fun k => payloads.getD k (Check.bytesOfBits [])) 0
    (releases g.allocation (fun t => enq.getD t 0) 0 0 (W * T))).toArray
  (Array.range W).map fun w => all.extract (w * T) (w * T + T)

/-- `Check.unitSync`: the spatial unit's clock in window `w`, the sync the first egress link declares. -/
def unitSync (e : ByteEvidence) (w : Nat) : ClockSync :=
  ((e.links[0]?).map (·.syncs.getD w Check.noSync)).getD Check.noSync

/-- An egress link-window's violation (`check_link`), none when it is accepted; `expected` is the Program's grid of the
link-window. A certified record's is v4's `Check.egressVerdict`. -/
def egressVerdict (p : Check.Params) (c : Commitment) (e : ByteEvidence) (egressDigest : ByteArray) (l : String)
    (link : ByteLink) (expected : RawGrid) (w : Nat) : Option String :=
  let screened : Except String (RawGrid → Bool) := match (found c .egress l w).toList with
    | [] => .error "no-verdict"
    | [.certified r] => (Check.screen p #[r]).map fun r grid =>
      match recordHead l .egress w grid none p.grid egressDigest true >>= RecordHead.digest, r.head.digest with
      | .ok d, .ok d' => d' == d
      | _, _ => false
    | [.raw _ _ st grid] => match statusViolation st with
      | some v => .error v
      | none => .ok (· == grid)
    | [.committed _ _ st digest] => match statusViolation st with
      | some v => .error v
      | none => .ok fun grid => (committedDigest l w grid p.grid).toOption == some digest
    | _ => .error "duplicate"
  match screened with
  | .error v => some v
  | .ok agrees =>
    let s := link.syncs.getD w Check.noSync
    let u := unitSync e w
    if !(Check.inSyncSet p.syncSet s && s.delta == u.delta && s.rho == u.rho) then some "advice"
    else if agrees expected then none
    else some "mismatch"

/-- An egress link-window's violation without its grid comparison (`check_link(grid=False)`, N2's), none when it is
accepted: `egressVerdict`'s tests before that comparison, in its order and with its names (`no-verdict`, `duplicate`,
`signature`, `status:<reason>`, `advice`), and nothing after them, so it reads no Program grid and no record's grid
or digest. -/
def egressVerdictNoGrid (p : Check.Params) (c : Commitment) (e : ByteEvidence) (l : String) (link : ByteLink) (w : Nat) :
    Option String :=
  let screened : Except String Unit := match (found c .egress l w).toList with
    | [] => .error "no-verdict"
    | [.certified r] => (Check.screen p #[r]).map fun _ => ()
    | [.raw _ _ st _] => match statusViolation st with
      | some v => .error v
      | none => .ok ()
    | [.committed _ _ st _] => match statusViolation st with
      | some v => .error v
      | none => .ok ()
    | _ => .error "duplicate"
  match screened with
  | .error v => some v
  | .ok _ =>
    let s := link.syncs.getD w Check.noSync
    let u := unitSync e w
    if !(Check.inSyncSet p.syncSet s && s.delta == u.delta && s.rho == u.rho) then some "advice"
    else none

/-- An ingress link-window's violation (`check_ingress_link`), none when it is accepted. -/
def ingressVerdict (p : Check.Params) (c : Commitment) (ingressDigest : ByteArray) (l : String) (w : Nat) : Option String :=
  match (found c .ingress l w).toList with
  | [] => some "no-verdict"
  | [.certified r] => Check.ingressVerdict p (single r) ingressDigest l w
  | [.raw _ _ st _] | [.committed _ _ st _] => statusViolation st
  | _ => some "duplicate"

/-- Every link-window's verdict, in `Audit`'s order: each egress link's windows, then each ingress link's. -/
def verdicts (p : Check.Params) (c : Commitment) (e : ByteEvidence) (egressDigest ingressDigest : ByteArray) :
    Array LinkWindow :=
  (((p.egress.toList.zip e.links.toList).flatMap fun (l, link) =>
      let grids := expectedGridsOfBytes p.grid p.windows link
      (List.range p.windows).map fun w =>
        ({ direction := .egress, link := l, window := w,
           violation := egressVerdict p c e egressDigest l link (grids.getD w #[]) w } : LinkWindow)) ++
    p.ingress.toList.flatMap fun l => (List.range p.windows).map fun w =>
      ({ direction := .ingress, link := l, window := w, violation := ingressVerdict p c ingressDigest l w } :
        LinkWindow)).toArray

/-- Every link-window's verdict, or why the evidence or the parameters can't give them (v4's refusals). -/
def evaluate (p : Check.Params) (c : Commitment) (e : ByteEvidence) : Except String (Array LinkWindow) :=
  if e.links.size ≠ p.egress.size then
    .error s!"evidence: {e.links.size} links' frames and syncs for {p.egress.size} egress links"
  else if e.links.any (·.syncs.size ≠ p.windows) then
    .error s!"evidence: a link declares other than {p.windows} syncs"
  else if e.links.any (·.frames.any (·.payload.size ≠ p.grid.frameBytes)) then
    .error s!"evidence: a frame of other than {p.grid.frameBytes} bytes"
  else
    match parametersDigest p.grid (some (Check.manifestSet p.syncSet)), parametersDigest p.ingressGrid none with
    | .ok eg, .ok ig => .ok (verdicts p c e eg ig)
    | _, _ => .error "the parameters digest has no canonical JSON"

/-- `verdicts` with each egress link-window's verdict `egressVerdictNoGrid`'s, so no Program grid is computed. -/
def verdictsNoGrid (p : Check.Params) (c : Commitment) (e : ByteEvidence) (ingressDigest : ByteArray) :
    Array LinkWindow :=
  (((p.egress.toList.zip e.links.toList).flatMap fun (l, link) =>
      (List.range p.windows).map fun w =>
        ({ direction := .egress, link := l, window := w, violation := egressVerdictNoGrid p c e l link w } :
          LinkWindow)) ++
    p.ingress.toList.flatMap fun l => (List.range p.windows).map fun w =>
      ({ direction := .ingress, link := l, window := w, violation := ingressVerdict p c ingressDigest l w } :
        LinkWindow)).toArray

/-- A raw or committed record names an egress link of the unit: nothing holds its grid but a comparison with the
Program's. -/
def uncertifiedEgress (p : Check.Params) (c : Commitment) : Bool :=
  c.records.any fun
    | .raw l .. | .committed l .. => p.egress.contains l
    | .certified _ => false

/-- Every link-window's verdict without the egress grid comparison (`check_link(grid=False)`, N2's, whose head check
holds each certified head's grid), or why not: `evaluate`'s refusals in its order, with a refusal of a raw or committed
egress record (`uncertifiedEgress`) after the evidence's, and then `verdictsNoGrid`. -/
def evaluateNoGrid (p : Check.Params) (c : Commitment) (e : ByteEvidence) : Except String (Array LinkWindow) :=
  if e.links.size ≠ p.egress.size then
    .error s!"evidence: {e.links.size} links' frames and syncs for {p.egress.size} egress links"
  else if e.links.any (·.syncs.size ≠ p.windows) then
    .error s!"evidence: a link declares other than {p.windows} syncs"
  else if e.links.any (·.frames.any (·.payload.size ≠ p.grid.frameBytes)) then
    .error s!"evidence: a frame of other than {p.grid.frameBytes} bytes"
  else if uncertifiedEgress p c then
    .error "only a certified record's grid is held by its head check: compare a raw or committed egress record's (verdicts)"
  else
    match parametersDigest p.grid (some (Check.manifestSet p.syncSet)), parametersDigest p.ingressGrid none with
    | .ok _, .ok ig => .ok (verdictsNoGrid p c e ig)
    | _, _ => .error "the parameters digest has no canonical JSON"

def check (p : Check.Params) (c : Commitment) (e : ByteEvidence) : Core.Outcome Unit Profile :=
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
  Evidence := ByteEvidence
  Public := Unit
  Profile := Profile
  params := Check.paramsOf
  commitment := commitmentOf
  evidence b := do evidenceOf (← Core.canonicalJson b)
  digest c := (Identity.taggedSha512 commitmentTag c.bytes).toOption.getD .empty
  check := check
  publicJson _ := Json.mkObj []
  profileJson := Check.profileJson

def linkWindowJson (v : LinkWindow) : Json :=
  Json.mkObj [("direction", .str v.direction.value), ("link", .str v.link), ("verdict", .str (v.violation.getD "accept")),
    ("window", toJson v.window)]

/-- A `serve` case's parameters, commitment and evidence, or a refusal named as `Core.Protocol.answer` names it
(`case: `, `params: `, `commitment: `, `evidence: `). -/
def caseOf (j : Json) : Except String (Check.Params × Commitment × ByteEvidence) := do
  let parsed : Flock.V (Json × ByteArray × ByteArray) := do
    pure (← Flock.J.get j "params" "the case", ← unhex (← Flock.J.str j "commitment" "the case"),
      ← unhex (← Flock.J.str j "evidence" "the case"))
  let (pj, cb, eb) ← parsed.mapError (s!"case: {·}")
  let p ← (Check.paramsOf pj).mapError (s!"params: {·}")
  let c ← (commitmentOf cb).mapError (s!"commitment: {·}")
  let e ← (Core.canonicalJson eb >>= evidenceOf).mapError (s!"evidence: {·}")
  pure (p, c, e)

/-- `verdicts` on a `serve` case: `{"link_windows": [...]}`, every link-window's verdict in the audit's order, or a refusal
(`caseOf`'s, or `evaluate`'s). -/
def verdictsFn (j : Json) : Except String Json := do
  let (p, c, e) ← caseOf j
  let vs ← evaluate p c e
  pure (Json.mkObj [("link_windows", .arr (vs.map linkWindowJson))])

/-- `verdictsFn` without the egress grid comparison: `evaluateNoGrid`'s verdicts on a `serve` case, or its refusal. -/
def verdictsNoGridFn (j : Json) : Except String Json := do
  let (p, c, e) ← caseOf j
  let vs ← evaluateNoGrid p c e
  pure (Json.mkObj [("link_windows", .arr (vs.map linkWindowJson))])

end NetworkCheck.Records
