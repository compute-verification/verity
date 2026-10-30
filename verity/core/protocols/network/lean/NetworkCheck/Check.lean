import NetworkCheck.Certificate
import NetworkCertifier.Schedule
import Lms.Verify
import CoreProtocol.Cli

/-!
# The network certificate check, a core protocol: `verity/network-certificate/v4`

The auditor's check of one spatial unit's certified link-windows (record v2, `certificate.py`), as `audit.py`'s `Audit`
runs it: `check_link` on every egress link in order, then `check_ingress_link` on every ingress link, each against the
network certifier's public key (`LmsKey`) in the certifier session the parameters name. The id is the check's; the message
a certificate signs has its own domain-separation tag, `verity/network-certificate-message/v2` (`messageTag`).

**Params**, canonical JSON:

    {"egress": {"grid": G, "links": [link, …]}, "ingress": {"grid": G, "links": [link, …]},
     "key": hex(I ‖ T[1]), "lms": {"H": h, "N": n, "W": w, "toy": bool}, "session": hex(session),
     "sync_set": {"deltas": [δ, …], "rhos": [[num, den], …]}, "windows": W}
    G = {"allocation": r, "bucket_ms": τ_b, "buckets": T, "frame_bytes": B, "queue_capacity": Q}

The spatial unit's boundary links by direction, each list duplicate-free, a name at most 65,535 UTF-8 bytes, at least one
link in all; the grid parameters of each direction (`GridParams`, with T, r, B and τ_b u32s, since a record encodes them so,
and every value ≥ 1); Σ_sync (`SyncSet`: non-empty, every δ and every ρ = num/den listed once, num and den ≥ 1); the W ≥ 1
windows checked; the LMS parameters (`lms.Params`) and the public key, 16 + N bytes; and the certifier session's id, the 32
bytes the challenger drew for the certifier's session, which every certificate's message binds.

**Commitment**: what the auditor holds before the check, the unit's certified records (`CertifiedRecord`: the record
head, the 64 bytes the certifier drew for it, `salt_c`, and the certificate that signs the head's digest salted by them),
each in `service.encode`'s bytes, concatenated:

    RecordHead.encode() ‖ salt_c (64) ‖ u16be len(I) ‖ I ‖ u16be len(link) ‖ link ‖ u8 direction ‖ u64be window ‖
    digest (64) ‖ u32be len(signature) ‖ signature

decoded exactly: UTF-8 links, direction 0 or 1, a status code of `STATUS_CODES`, deadline 0 or 1, T counts each at most r,
nothing left over. Its `digest` is `tagged_sha512("verity-network-certifier/certificates/v3", the commitment's bytes)`
(`Identity.taggedSha512`), which every commitment under `2 ^ 64` bytes has; a longer one, which Python refuses to hash,
gets the empty digest, since `Core.Protocol.digest` is total.

**Evidence**: what the prover opens, canonical JSON, one entry per egress link in the order of `egress.links`:

    {"links": [{"frames": [[payload hex, anchor, index], …], "syncs": [[δ, num, den], …]}, …]}

the Program's logical frames on the link in canonical order (`LogicalFrame`) and the clock sync it declares for each of the W
windows (`ClockSync`, num and den ≥ 1). The audit reads no frames or Merkle openings from a certified record: it recomputes
the Program's grid from these frames and syncs (`NetworkCertifier.expectedRow`, the rows of the constant-rate Program
`constantRate`) and compares its record digest, root and counts included, with the signed head's. Evidence that doesn't fit
the parameters (another number of links or of syncs, a payload of other than B bytes) is a rejection.

**The verdict** of each link-window, in `Audit`'s order and with its names (`_status_violation`, then the link's own checks):
* `no-verdict`: no record for it, so no certificate (a refused signer issues none), or a head not settled by the deadline;
* `duplicate`: more than one record;
* `signature`: the certificate is not the key's (its I), names another link, direction or window than the head, signs
  another digest than the head's salted by the record's `salt_c` (`certificateDigest`), or its LMS signature doesn't
  verify over `certificate_message` in the parameters' session (a certificate of another session of the certifier
  included);
* `status:<reason>`: the signed status is not complete;
* `advice` (egress): the declared sync is outside Σ_sync or not the spatial unit's, the first egress link's;
* `mismatch`: on egress, the head's record digest is not that of `record_head` of the Program's grid under the declared syncs
  (complete, settled, under `parameters_digest` of the grid and Σ_sync); on ingress, the head's T, r, B or τ_b, or its
  parameters digest, is not the ingress grid's.

A record of a link or window the unit doesn't check is ignored, as `Audit` ignores it. The check rejects if any
link-window has a violation other than `no-verdict`, is undecided if some are `no-verdict` and none is another, and accepts
only when every link-window is accepted.

**Public**: nothing beyond the verdict, `{}`. `service.py` publishes a link-window's `verdict` and `deadline`; on an accepted
unit every head is settled by the deadline.

**Profile**, the check's own: `link_windows`, every link-window's verdict in the audit's order
(`{"direction", "link", "verdict", "window"}`, all `accept` on an accepted unit), `windows`, the accepted windows
`[0, …, W − 1]`, and `syncs`, the unit's clock, the sync declared for each window as `[δ, num, den]` (the advice the
accepted-runs bound charges, `log2 #Σ_sync` bits per window); integers only.
-/

namespace NetworkCheck.Check

open Lean (Json toJson)
open Flock (V)
open NetworkCertifier (ClockSync LogicalFrame expectedRow)

def protocolId : String := "verity/network-certificate/v4"
def commitmentTag : String := "verity-network-certifier/certificates/v3"

/-- The grid parameters of one direction (`grid.GridParams`). -/
abbrev Grid := NetworkCheck.Params

structure Params where
  egress : Array String
  ingress : Array String
  windows : Nat
  grid : Grid
  ingressGrid : Grid
  syncSet : NetworkCertifier.SyncSet
  lms : Lms.Params
  key : ByteArray
  session : ByteArray

/-- `certificate.Certificate`. -/
structure Certificate where
  keyId : ByteArray
  link : String
  direction : Direction
  window : Nat
  digest : ByteArray
  signature : ByteArray

/-- `certificate.CertifiedRecord`: the head, the certifier's salt for it (`salt_c`) and the certificate. -/
structure Certified where
  head : RecordHead
  salt : ByteArray
  certificate : Certificate

structure Commitment where
  records : Array Certified
  bytes : ByteArray

/-- One egress link's opening: its logical frames, payloads as bits (most significant first), and its declared syncs. -/
structure Link where
  frames : List LogicalFrame
  syncs : Array ClockSync

structure Evidence where
  links : Array Link

structure LinkWindow where
  direction : Direction
  link : String
  window : Nat
  violation : Option String

structure Profile where
  verdicts : Array LinkWindow
  windows : Nat
  syncs : Array ClockSync

/-! ## The bits of a payload -/

def bitsOfBytes (b : ByteArray) : List Bool :=
  b.data.toList.flatMap fun x => (List.range 8).map fun i => (x.toNat >>> (7 - i)) % 2 == 1

/-- Eight bits as a byte, most significant first. -/
def byteOfBits (bits : List Bool) : UInt8 := (bits.foldl (fun (a : Nat) (b : Bool) => 2 * a + b.toNat) 0).toUInt8

/-- The bytes of whole groups of eight bits, most significant first; a last group of fewer is dropped. -/
def bytesOfBitsList : List Bool → List UInt8
  | b0 :: b1 :: b2 :: b3 :: b4 :: b5 :: b6 :: b7 :: rest =>
    byteOfBits [b0, b1, b2, b3, b4, b5, b6, b7] :: bytesOfBitsList rest
  | _ => []

def bytesOfBits (bits : List Bool) : ByteArray := ⟨(bytesOfBitsList bits).toArray⟩

/-! ## Decoding -/

private def gridOf (j : Json) (ctx : String) : V Grid := do
  let field (k : String) : V Nat := do
    let v ← Flock.J.nat j k ctx
    unless 1 ≤ v ∧ v < 2 ^ 32 do throw s!"{ctx}.{k} is a u32 of at least 1, not {v}"
    pure v
  let q ← Flock.J.nat j "queue_capacity" ctx
  unless 1 ≤ q do throw s!"{ctx}.queue_capacity is at least 1"
  pure ⟨← field "buckets", ← field "allocation", ← field "frame_bytes", ← field "bucket_ms", q⟩

private def linksOf (j : Json) (ctx : String) : V (Array String) := do
  let ls ← (← Flock.J.arr j "links" ctx).mapM (Flock.J.asStr · s!"{ctx}.links")
  if ls.any (·.toUTF8.size ≥ 2 ^ 16) then throw s!"{ctx}.links: a name is at most 65,535 UTF-8 bytes"
  unless ls.toList.Nodup do throw s!"{ctx}.links lists each link once"
  pure ls

/-- A positive fraction `[num, den]`, both at least 1. -/
def ratOf (j : Json) (ctx : String) : V Rat := do
  let #[n, d] ← Flock.J.asArr j ctx | throw s!"{ctx}: not [numerator, denominator]"
  let n ← Flock.J.asNat n ctx
  let d ← Flock.J.asNat d ctx
  unless 1 ≤ n ∧ 1 ≤ d do throw s!"{ctx}: a numerator and a denominator of at least 1"
  pure (mkRat n d)

private def syncSetOf (j : Json) : V NetworkCertifier.SyncSet := do
  let deltas := (← (← Flock.J.arr j "deltas" "sync_set").mapM (Flock.J.asNat · "sync_set.deltas")).toList
  let rhos := (← (← Flock.J.arr j "rhos" "sync_set").mapM (ratOf · "sync_set.rhos")).toList
  if h : deltas ≠ [] ∧ rhos ≠ [] ∧ (∀ ρ ∈ rhos, 0 < ρ) ∧ deltas.Nodup ∧ rhos.Nodup then
    pure ⟨deltas, rhos, h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2⟩
  else throw "sync_set: non-empty, each delta and each rho listed once"

def paramsOf (j : Json) : V Params := do
  let e ← Flock.J.get j "egress" "the object"
  let i ← Flock.J.get j "ingress" "the object"
  let egress ← linksOf e "egress"
  let ingress ← linksOf i "ingress"
  unless egress.size + ingress.size ≥ 1 do throw "a spatial unit has at least one link"
  let windows ← Flock.J.nat j "windows" "the object"
  unless 1 ≤ windows do throw "at least one window"
  let l ← Flock.J.get j "lms" "the object"
  let lms ← Lms.Params.make (← Flock.J.nat l "N" "lms") (← Flock.J.nat l "W" "lms") (← Flock.J.nat l "H" "lms")
    (← Flock.J.bool l "toy" "lms")
  let key ← Flock.unhex (← Flock.J.str j "key" "the object")
  unless key.size = 16 + lms.N do throw s!"an LMS public key is {16 + lms.N} bytes, not {key.size}"
  let session ← Flock.unhex (← Flock.J.str j "session" "the object")
  unless session.size = sessionBytes do throw s!"a certifier session id is {sessionBytes} bytes, not {session.size}"
  pure { egress, ingress, windows, grid := ← gridOf (← Flock.J.get e "grid" "egress") "egress.grid",
         ingressGrid := ← gridOf (← Flock.J.get i "grid" "ingress") "ingress.grid",
         syncSet := ← syncSetOf (← Flock.J.get j "sync_set" "the object"), lms, key, session }

section Commitment

open Flock.Reader

private def be (n : Nat) : M Nat := do
  pure ((← bytes n).foldl (fun a x => a * 256 + x.toNat) 0)

private def utf8 (what : String) : M String := do
  match String.fromUTF8? (← bytes (← be 2)) with
  | some s => pure s
  | none => throw s!"{what} is not UTF-8"

private def direction : M Direction := do
  match ← be 1 with
  | 0 => pure .egress
  | 1 => pure .ingress
  | c => throw s!"a direction code is 0 or 1, not {c}"

/-- `STATUS_CODES` read back: 0 is complete, then each `Reason` by its code. -/
private def status : M (Option String) := do
  let c ← be 1
  if c = 0 then return none
  match ["monitor-unavailable", "overflow", "malformed", "dropped", "delivery-refused",
      "coverage-never-established"][c - 1]? with
  | some r => pure (some r)
  | none => throw s!"not a status code: {c}"

private def head : M RecordHead := do
  let link ← utf8 "a record's link"
  let direction ← direction
  let window ← be 8
  let buckets ← be 4
  let allocation ← be 4
  let frameBytes ← be 4
  let bucketMs ← be 4
  let status ← status
  let deadline ← match ← be 1 with
    | 0 => pure false
    | 1 => pure true
    | d => throw s!"a deadline byte is 0 or 1, not {d}"
  let parameters ← bytes 64
  need (4 * buckets + 64)
  let mut counts := #[]
  for _ in [0:buckets] do
    let n ← be 4
    if n > allocation then throw s!"a record has counts of 0 … {allocation}, not {n}"
    counts := counts.push (n : Int)
  pure { link, direction, window, buckets, allocation, frameBytes, bucketMs, status, deadline, parameters, counts,
         root := ← bytes 64 }

private def certificate : M Certificate := do
  let keyId ← bytes (← be 2)
  let link ← utf8 "a certificate's link"
  let direction ← direction
  let window ← be 8
  let digest ← bytes 64
  pure { keyId, link, direction, window, digest, signature := ← bytes (← be 4) }

/-- The records, one after another until the bytes are spent; each takes at least one byte. -/
def commitmentOf (b : ByteArray) : V Commitment := do
  let records : M (Array Certified) := do
    let mut out := #[]
    for _ in [0:b.size] do
      if (← get).pos ≥ b.size then break
      out := out.push ⟨← head, ← bytes saltBytes, ← certificate⟩
    pure out
  let (rs, r) ← records.run ⟨b, 0⟩
  if r.pos ≠ b.size then throw "bytes left after the records"
  pure ⟨rs, b⟩

end Commitment

def frameOf (j : Json) : V LogicalFrame := do
  let #[p, a, i] ← Flock.J.asArr j "a frame" | throw "a frame is [payload, anchor, index]"
  pure ⟨bitsOfBytes (← Flock.unhex (← Flock.J.asStr p "a frame's payload")), ← Flock.J.asNat a "an anchor",
    ← Flock.J.asNat i "an index"⟩

def syncOf (j : Json) : V ClockSync := do
  let #[d, n, m] ← Flock.J.asArr j "a sync" | throw "a sync is [delta, num, den]"
  let δ ← Flock.J.asNat d "a sync's delta"
  let ρ ← ratOf (.arr #[n, m]) "a sync's rho"
  if h : 0 < ρ then pure ⟨δ, ρ, h⟩ else throw "a sync's rho is positive"

def evidenceOf (j : Json) : V Evidence := do
  let links ← (← Flock.J.arr j "links" "the object").mapM fun l => do
    pure { frames := (← (← Flock.J.arr l "frames" "a link").mapM frameOf).toList,
           syncs := ← (← Flock.J.arr l "syncs" "a link").mapM syncOf : Link }
  pure ⟨links⟩

/-! ## The check -/

private def sameSync (a b : ClockSync) : Bool := a.delta == b.delta && a.rho == b.rho

/-- `sync in sync_set`: its δ is one of Σ_sync's and its ρ one of Σ_sync's. -/
def inSyncSet (ss : NetworkCertifier.SyncSet) (s : ClockSync) : Bool := decide (s.delta ∈ ss.deltas ∧ s.rho ∈ ss.rhos)

/-- Σ_sync as `parameters_digest` reads it, each ρ in lowest terms. -/
def manifestSet (ss : NetworkCertifier.SyncSet) : NetworkCheck.SyncSet :=
  ⟨ss.deltas.toArray.map Int.ofNat, ss.rhos.toArray.map fun ρ => (ρ.num, ρ.den)⟩

/-- `CertifiedRecord.verify` under `LmsKey` in the parameters' session: the certificate is the key's, names the head's
link-window, signs the head's digest salted by the record's salt (`certificateDigest`), and its signature verifies over
the message in that session; a signature that doesn't decode is false. -/
def verifies (p : Params) (r : Certified) : Bool :=
  let c := r.certificate
  let h := r.head
  c.keyId == p.key.extract 0 16 && c.link == h.link && c.direction == h.direction && (c.window : Int) == h.window &&
    (match h.digest with
     | .ok d => c.digest == certificateDigest r.salt d
     | .error _ => false) &&
    match certificateMessage c.keyId p.session c.link c.direction c.window c.digest,
        Lms.Signature.decode p.lms c.signature with
    | .ok m, .ok sig => (Lms.verify p.lms p.key m sig).toOption == some true
    | _, _ => false

/-- `_status_violation`: the one record of the link-window, or its violation. -/
def screen (p : Params) (found : Array Certified) : Except String Certified :=
  match found.toList with
  | [] => .error "no-verdict"
  | [r] =>
    if !verifies p r then .error "signature"
    else if !r.head.deadline then .error "no-verdict"
    else match r.head.status with
      | some why => .error s!"status:{why}"
      | none => .ok r
  | _ => .error "duplicate"

/-- The sync read where none is declared; the evidence's shape is checked first, so no verdict reads it. -/
def noSync : ClockSync := ⟨0, 1, by decide⟩

/-- `expected_grids(frames, syncs, params)[w]`: the Program's grid of the link in window `w`, each bucket's row
`expectedRow` at its global bucket. -/
def expectedGrid (g : Grid) (W : Nat) (l : Link) (w : Nat) : Array (Array (Option ByteArray)) :=
  let syncs : Fin W → ClockSync := fun v => l.syncs.getD v.1 noSync
  (List.range g.buckets).toArray.map fun i =>
    (List.ofFn (expectedRow g.allocation g.buckets W syncs l.frames (w * g.buckets + i))).toArray.map
      (·.map bytesOfBits)

/-- The unit's records of a link-window. -/
def found (c : Commitment) (d : Direction) (l : String) (w : Nat) : Array Certified :=
  c.records.filter fun r => r.head.direction == d && r.head.link == l && r.head.window == (w : Int)

/-- The spatial unit's clock in window `w`: the sync the first egress link declares. -/
def unitSync (e : Evidence) (w : Nat) : ClockSync := ((e.links[0]?).map (·.syncs.getD w noSync)).getD noSync

/-- An egress link-window's violation (`check_link`), none when it is accepted. -/
def egressVerdict (p : Params) (c : Commitment) (e : Evidence) (egressDigest : ByteArray) (l : String) (link : Link)
    (w : Nat) : Option String :=
  match screen p (found c .egress l w) with
  | .error v => some v
  | .ok r =>
    let s := link.syncs.getD w noSync
    if !(inSyncSet p.syncSet s && sameSync s (unitSync e w)) then some "advice"
    else match recordHead l .egress w (expectedGrid p.grid p.windows link w) none p.grid egressDigest true >>=
        RecordHead.digest, r.head.digest with
      | .ok d, .ok d' => if d' == d then none else some "mismatch"
      | _, _ => some "mismatch"

/-- An ingress link-window's violation (`check_ingress_link`), none when it is accepted. -/
def ingressVerdict (p : Params) (c : Commitment) (ingressDigest : ByteArray) (l : String) (w : Nat) : Option String :=
  match screen p (found c .ingress l w) with
  | .error v => some v
  | .ok r =>
    let h := r.head
    let g := p.ingressGrid
    if h.buckets == g.buckets && h.allocation == g.allocation && h.frameBytes == g.frameBytes &&
        h.bucketMs == g.bucketMs && h.parameters == ingressDigest then none
    else some "mismatch"

/-- Every link-window's verdict, in `Audit`'s order: each egress link's windows, then each ingress link's. `egressDigest`
and `ingressDigest` are `parameters_digest` of the egress grid with Σ_sync and of the ingress grid. -/
def verdicts (p : Params) (c : Commitment) (e : Evidence) (egressDigest ingressDigest : ByteArray) : Array LinkWindow :=
  (((p.egress.toList.zip e.links.toList).flatMap fun (l, link) => (List.range p.windows).map fun w =>
      ({ direction := .egress, link := l, window := w, violation := egressVerdict p c e egressDigest l link w } :
        LinkWindow)) ++
    p.ingress.toList.flatMap fun l => (List.range p.windows).map fun w =>
      ({ direction := .ingress, link := l, window := w, violation := ingressVerdict p c ingressDigest l w } :
        LinkWindow)).toArray

def check (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Unit Profile :=
  if e.links.size ≠ p.egress.size then
    .reject s!"evidence: {e.links.size} links' frames and syncs for {p.egress.size} egress links"
  else if e.links.any (·.syncs.size ≠ p.windows) then
    .reject s!"evidence: a link declares other than {p.windows} syncs"
  else if e.links.any (·.frames.any (·.payload.length ≠ 8 * p.grid.frameBytes)) then
    .reject s!"evidence: a frame of other than {p.grid.frameBytes} bytes"
  else
    match parametersDigest p.grid (some (manifestSet p.syncSet)), parametersDigest p.ingressGrid none with
    | .ok eg, .ok ig =>
      let vs := verdicts p c e eg ig
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
    | _, _ => .reject "the parameters digest has no canonical JSON"

def syncJson (s : ClockSync) : Json := .arr #[toJson s.delta, toJson s.rho.num, toJson s.rho.den]

def profileJson (pr : Profile) : Json :=
  Json.mkObj [
    ("link_windows", .arr <| pr.verdicts.map fun v => Json.mkObj [("direction", .str v.direction.value),
      ("link", .str v.link), ("verdict", .str (v.violation.getD "accept")), ("window", toJson v.window)]),
    ("syncs", .arr (pr.syncs.map syncJson)),
    ("windows", .arr ((List.range pr.windows).toArray.map toJson))]

def protocol : Core.Protocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Unit
  Profile := Profile
  params := paramsOf
  commitment := commitmentOf
  evidence b := do evidenceOf (← Core.canonicalJson b)
  digest c := (Identity.taggedSha512 commitmentTag c.bytes).toOption.getD .empty
  check := check
  publicJson _ := Json.mkObj []
  profileJson := profileJson

end NetworkCheck.Check
