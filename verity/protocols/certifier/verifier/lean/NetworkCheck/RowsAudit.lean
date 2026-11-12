module

public import NetworkCheck.Period
public import NetworkCheck.Records
public import NetworkCertifier.Rows

/-!
# The network certifier's run audited as rows over its certified records, a core protocol: `verity/network-rows/v1`

The audit of a run of the active network certifier (`warden_serve`) over the records it signed: each of the run's `W`
windows' requests (the link's ingress), responses (its egress) and fails (`<link>-fails`, egress) records, each held to its
certified record, then the certifier program's `NetworkCertifier.Rows.audit` over their grids, the executable form of
`security/Definitions/NetworkCertifier/RecordRows.lean`. That audit accepts only when the egress record is the schedule
it shows rendered over the lanes, so a run whose cells the network certifier cut, clocked or ordered otherwise than the
schedule is refused here, not taken on trust.

**Params**, canonical JSON with exactly these keys:

    {"key": hex(I ‖ T[1]), "link": name, "lms": {"H": h, "N": n, "W": w, "toy": bool},
     "schedule": {"bucket_ms", "cell_bytes", "delay", "egress_cells", "every", "ingress_buckets", "ingress_cells",
                  "lanes", "queue", "slots", "window"},
     "session": hex(session), "windows": W}

`NetworkCheck.Period`'s rows-form terms, read by its decoders: the network certifier's LMS public key (16 + N bytes) and
parameters, the link (`warden_serve.LINK`, at most 65,535 UTF-8 bytes), the schedule (`warden.WardenParams`,
`Period.scheduleOf`), whose digest every record head binds (`Period.Schedule.digest`), the certifier session (32 bytes)
every certificate binds, and the run's `W ≥ 1` windows, `0 … W − 1`.

**Commitment**: `verity/network-records/v3`'s encoding (`NetworkCheck.Records.commitmentOf`) of `6 · W` records: for each
window in order, the requests', the responses' and the fails' records, each raw (`grid.BlockGrid`, under the record's
own name, `requests`, `responses` or `fails`) and then certified (`CertifiedRecord`). Its digest is
`tagged_sha512("verity-network-certifier/rows-records/v1", the commitment's bytes)`.

**Evidence**: none, `{}`.

**The check**, the first finding a rejection: the commitment holds the `6 · W` records in that order; each raw record
names its record and window, and its certified record (`bound`) is the link's, its direction's and the window's, its head
`record_head` of the raw record's grid and status under the schedule's grid (`gridOf`: `T` rows of `lanes · slots` cells
of `cell_bytes`, the fails record's of `lanes`) and digest, settled by the deadline, and its certificate signs the head's
digest under the record's salt (`certificateDigest`) and is the key's and verifies in the session
(`Check.Certificate.signs`); then `NetworkCertifier.Rows.audit` over the grids (`auditOf`), with its refusal's reason.

What it reads is witness, the traffic's records and the heads and salts the certifier signed, so it publishes nothing:
**Public** `{}`, and the **profile** the link and `W`. Its work is linear in the records: each grid is hashed once into
its head and read once by the audit.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace NetworkCheck.RowsAudit

open Lean (Json toJson)
open Flock (V)
open NetworkCheck.Records (Published Commitment)

def protocolId : String := "verity/network-rows/v1"
def commitmentTag : String := "verity-network-certifier/rows-records/v1"

/-- The run's records by their own names (`warden_serve.INGRESS`, `EGRESS` and `FAILS`), each with the link and the
direction it is certified under (`warden_serve.CERTIFIED`, `Period.records`). -/
def roles (link : String) : Array (String × String × Direction) :=
  #["requests", "responses", "fails"].zip (Period.records link)

structure Params where
  key : ByteArray
  link : String
  lms : Lms.Params
  schedule : Period.Schedule
  digest : ByteArray
  session : ByteArray
  windows : Nat

structure Profile where
  link : String
  windows : Nat

/-! ## Decoding -/

private def keysOf : Json → List String
  | .obj kvs => kvs.keys
  | _ => []

private def exactly (j : Json) (want : List String) (ctx : String) : V Unit :=
  match j with
  | .obj _ => if keysOf j == want then pure () else throw s!"{ctx}: the keys must be {want}, not {keysOf j}"
  | _ => throw s!"{ctx}: not an object"

def paramsOf (j : Json) : V Params := do
  exactly j ["key", "link", "lms", "schedule", "session", "windows"] "the object"
  let link ← Flock.J.str j "link" "the object"
  unless link.toUTF8.size < 2 ^ 16 do throw "link: a name is at most 65,535 UTF-8 bytes"
  let l ← Flock.J.get j "lms" "the object"
  exactly l ["H", "N", "W", "toy"] "lms"
  let lms ← Lms.Params.make (← Flock.J.nat l "N" "lms") (← Flock.J.nat l "W" "lms") (← Flock.J.nat l "H" "lms")
    (← Flock.J.bool l "toy" "lms")
  let key ← Flock.unhex (← Flock.J.str j "key" "the object")
  unless key.size = 16 + lms.N do throw s!"an LMS public key is {16 + lms.N} bytes, not {key.size}"
  let session ← Flock.unhex (← Flock.J.str j "session" "the object")
  unless session.size = sessionBytes do throw s!"a certifier session id is {sessionBytes} bytes, not {session.size}"
  let schedule ← Period.scheduleOf (← Flock.J.get j "schedule" "the object")
  let windows ← Flock.J.nat j "windows" "the object"
  unless 1 ≤ windows do throw "windows: a run has at least one window"
  pure { key, link, lms, schedule, digest := ← schedule.digest, session, windows }

def evidenceOf (b : ByteArray) : V Unit := do exactly (← Core.canonicalJson b) [] "the object"

/-! ## The check -/

/-- A record's grid parameters under the schedule (`WardenParams.grid`, `fails_grid`): `T` rows of `lanes · slots` cells
of `B` bytes, the fails record's of `lanes`, at τ_b; the queue's capacity is `GridParams`' default, which no head reads. -/
def gridOf (s : Period.Schedule) (fails : Bool) : Check.Grid :=
  { buckets := s.window, allocation := if fails then s.lanes else s.allocation, frameBytes := s.cellBytes,
    bucketMs := s.bucketMs, queueCapacity := 65536 }

/-- **Window `w`'s record `role`, held to its certified record** (`warden_serve.check`'s): `raw` is the record under
its name at `w`, and `cert` the record certified for the role's link, direction and `w`, whose head is `record_head` of
the raw record's grid and status under the schedule's grid and digest, settled by the deadline, and whose certificate
names the same, signs the head's digest under the record's salt and is the key's in the session. -/
def bound (p : Params) (w : Nat) (role : String × String × Direction) (raw cert : Option Published) :
    Except String NetworkCertifier.Rows.BlockGrid := do
  let (name, link, d) := role
  let what := s!"{name} window {w}"
  let some (.raw l w' status grid) := raw | throw s!"{what}: not a raw record"
  unless l == name && w' == w do throw s!"{what}: the raw record is {l}'s at window {w'}"
  let some (.certified r) := cert | throw s!"{what}: not a certified record"
  let c := r.certificate
  unless r.head.link == link && r.head.direction == d && r.head.window == (w : Int) && c.link == link &&
      c.direction == d && c.window == w do
    throw s!"{what}: the certified record isn't {link}'s {d.value} record at window {w}"
  let head ← (recordHead link d w grid status (gridOf p.schedule (name == "fails")) p.digest true >>=
    RecordHead.digest).mapError (s!"{what}: " ++ ·)
  unless r.head.digest.toOption == some head do throw s!"{what}: the certified head isn't the record's"
  unless c.digest == certificateDigest r.salt head do throw s!"{what}: the certificate doesn't sign the head"
  unless c.signs p.lms p.key p.session do throw s!"{what}: the certificate isn't the key's in the session"
  pure { status, grid }

/-- The run's records as `NetworkCertifier.Rows.audit` reads them, each held to its certified record (`bound`): the
requests', the responses' and the fails' records of windows `0 … W − 1`, in the commitment's order. -/
def recordsOf (p : Params) (c : Commitment) : Except String (Array (Array NetworkCertifier.Rows.BlockGrid)) := do
  unless c.records.size == 6 * p.windows do
    throw s!"the commitment holds {c.records.size} records, not 6 · W = {6 * p.windows}: each window's requests', \
      responses' and fails' records, each raw and then certified"
  (roles p.link).zipIdx.mapM fun (role, i) => (Array.range p.windows).mapM fun w =>
    bound p w role c.records[6 * w + 2 * i]? c.records[6 * w + 2 * i + 1]?

/-- The schedule's terms the rows audit reads. -/
def rowsParams (s : Period.Schedule) : NetworkCertifier.Rows.Params :=
  { cellBytes := s.cellBytes, slots := s.slots, delay := s.delay, lanes := s.lanes, window := s.window,
    egressCells := s.egressCells }

/-- **The audit of the run**: its records held to their certified records (`recordsOf`), then
`NetworkCertifier.Rows.audit` over the requests', responses' and fails' grids. -/
def auditOf (p : Params) (c : Commitment) : Except String NetworkCertifier.Rows.Summary := do
  let rs ← recordsOf p c
  NetworkCertifier.Rows.audit (rowsParams p.schedule) (rs.getD 0 #[]) (rs.getD 1 #[]) (rs.getD 2 #[])

def check (p : Params) (c : Commitment) (_ : Unit) : Core.Outcome Unit Profile :=
  match auditOf p c with
  | .error why => .reject why
  | .ok _ => .accept () { link := p.link, windows := p.windows }

def profileJson (pr : Profile) : Json := Json.mkObj [("link", .str pr.link), ("windows", toJson pr.windows)]

def protocol : Core.Protocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Unit
  Public := Unit
  Profile := Profile
  params := paramsOf
  commitment := Records.commitmentOf
  evidence := evidenceOf
  digest c := (Identity.taggedSha512 commitmentTag c.bytes).toOption.getD .empty
  check := check
  publicJson _ := Json.mkObj []
  profileJson := profileJson

end NetworkCheck.RowsAudit
