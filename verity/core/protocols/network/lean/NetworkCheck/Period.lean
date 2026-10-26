import NetworkCheck.Check
import NetworkCheck.LinkWindow
import CoreProtocol.Blobs
import CoreProtocol.Profile
import SampledProofs.OneStageZk
import SampledProofs.Partition
import Flock.Public

/-!
# The network certifier over a period, a core protocol: `verity/certifier/v0`

Onsite's communication component (`Core.Onsite.components`), a `Core.BlobProtocol`. A period is one node's link over `m`
link-windows beside the period's compute audits (sampled proofs' one-stage, one answer per registered window of the
workload). Its link-windows come in one of two forms, which the parameters name:

* **the clock form**: N2's link-windows on a planned link (`service.LinkWindows`, Selection `All`, no served or advised
  profile), each proved, its proof checked here by one-stage v2 (`SampledProofs.OneStageZk`) and opened against the
  verifier's own public row, both its ends certified and its clock certified;
* **the rows form**: the network certifier's own run, where it functions as an active warden (`warden_serve`): each
  link-window's cells committed as rows (`warden_rows`), one registration a direction under one fixed schedule, its
  three records certified, and the scan's one-stage answer for the window (`scan_zk --warden-run`), which the check
  refuses (below).

It accepts when every link-window passes its form's check and every compute audit accepted; it then publishes `K`, the
bits a covert sender in the server can move through the network certifier in the period, as
`NetTiming.Guarantees.PeriodLeakAccepted` counts them.

**Params**, JSON with exactly these keys (and each nested object exactly its own, but `one_stage`), in the clock form

    {"audits": [{"free": f, "steps": k}, …], "cells": [cap, …], "egress": end, "first": w₀, "ingress": end,
     "jitter": Dj, "key": hex(I ‖ T[1]), "link": name, "lms": {"H": h, "N": n, "W": w, "toy": bool},
     "one_stage": params, "run": run, "session": hex(session), "sync_set": {"deltas": [δ, …], "rhos": [[num, den], …]},
     "windows": m}
    end = {"grid": {"allocation", "bucket_ms", "buckets", "frame_bytes", "queue_capacity"}, "profile": [row, …] | null}

each compute audit's layout terms (`free`, the free inputs it doesn't register, `#(freeInputs \ reg)`; `steps`, its
monitoring period `k`), each network certifier connection's cell cap over the period (`cap_c`), each end's declared grid
and occupancy profile (`LinkWindows.params`, `ingress_params`, `occupancy`: a profile's row is the end's T counts at a
session window, `LinkWindow.End`), the session's window
the period starts at (`w₀`: its link-windows are the session's `w₀ … w₀ + m − 1`, so one certifier session spans many
periods),
the observations' jitter bound `Dj ≥ 1`, the network certifier's LMS public key and parameters and the certifier session
(32 bytes) every certificate binds, the link (at most 65,535 UTF-8 bytes), the auditor's one-stage v2 parameters for the
link-windows (`OneStageZk.paramsOfJson`'s, whose partition object names N2's registered Program for the link,
`LinkWindows.partition`) without `ports`, which are each link-window's own, and with `log` the stream's records before
the period, N2's run (`LinkWindows.run`), Σ_sync, and the period's `m ≥ 1` link-windows.
The rows form has, in place of `egress`, `ingress`, `one_stage`, `run` and `sync_set`, the SHA-512 of the scan's
registered Program, its layout terms and the schedule,

    "program": hex,
    "scan": {"free": f, "steps": k},
    "schedule": {"bucket_ms", "cell_bytes", "delay", "egress_cells", "ingress_buckets", "ingress_cells", "lanes",
                 "queue", "slots", "window"}

the scan's layout terms, as an audit's, and the network certifier's one fixed schedule (`warden.WardenParams`: each a
positive integer, B = `cell_bytes` past a cell's 9-byte header, and B, S = `lanes` · `slots` and T = `window` u32s),
whose digest `WardenParams.digest` every registration binds; its `link` is the run's (`warden_serve.LINK`), whose fails
record is certified under the link `<link>-fails`.

**Commitment**, canonical JSON `{"audits": [registration, …], "windows": [registration, …]}`: one-stage's registration
record of each compute audit and of each link-window, in order, each of format `verity/registration/v1` or `v2`. Its
digest is `tagged_sha512("verity-network-certifier/period/v0", the commitment's bytes)`, and each registration's is
one-stage's, SHA-512(format ‖ 0 ‖ canonical record) (`registration.digest`).

**Evidence**, canonical JSON:

    {"audits": [answer, …],
     "windows": [{"certificates": [certificate, certificate], "clock": certificate, "one_stage": evidence,
                  "salts": {"o0": hex, "pub": hex}}, …]}
    certificate = {"digest": hex, "direction": "egress" | "ingress", "key_id": hex, "link": s, "signature": hex,
                   "window": w}

in the clock form, with each link-window's `one_stage` one-stage v2's evidence for its registration
(`OneStageZk.evidenceOf`: v1's without verdicts, with the proofs' files by their SHA-512) and `salts` the salts N2's
prover delivered for the drawn unit's ``pub`` and ``o0`` row commitments (`Delivered.salts`, 192 bytes each); the
**blobs** (`Core.Blobs`) hold the files every link-window's proofs name: the circuit file, the public file, each
session's record and rep files. A link-window that carries an answer is refused: the check runs one-stage on its proof
itself and takes no answer for it. In the rows form each link-window
`{"answer": answer, "certificates": [certificate, certificate, certificate], "registrations": [rows, rows]}`, with

    rows = {"cell_bytes", "certificate": {"digest", "key", "signature"}, "direction", "domain", "format",
            "layout": {"buckets", "connection_slots", "first_bucket", "slots"}, "link", "parameters", "root",
            "row_schema", "rows", "value", "window", "wiring", "word_bits", "words"}

the requests', responses' and fails' records' certificates (`certified.jsonl`'s), and the ingress and the egress rows'
registrations (`warden_rows.registration`, `cells.jsonl`'s); a link-window of the other form than the parameters' is
refused.

In the clock form: each audit's one-stage answer (`Core.Answer.ofJson`), and per link-window, beside its proof, the two
certificates (`certificate.Certificate`, egress first) and the clock certificate the certifier, the auditor's device,
issued for it:
the public part of `certificate.CertifiedClock`, which signs the window's sync salted, `clock_commitment` =
SHA-512("verity/net-cert/v1" ‖ salt_c ‖ `clock_digest(sync)`). What the certifier signs is witness: the sync and its
salt are in neither the evidence nor the profile, and the check takes neither. The certifier is the auditor's trusted
device (`trusted-nebius-host`), and its own code picks each window's sync from its schedule set Σ_sync, so no
clear-text check that the sync is in Σ_sync is needed. N2's proof reading each window's sync through its opening
against that salted digest is a later change to its circuit (the certifier PROTOCOL.md's gap "The clock's opening").

**The check**, in order; any rejection rejects, then any undecided answer leaves the period undecided:

* the commitment registers the parameters' audits and `m` link-windows, and the evidence answers each;
* link-window `w` (the `i`-th, `w = w₀ + i`): its registration names the link's Program and the provenance
  `{"kind": "served", "window": w}`;
  both certificates are the key's, name the link, their end and `w`, and verify under LMS over `certificate_message` in
  the session; the clock certificate is the key's egress certificate of the link at `w` and verifies so, and the
  digest it signs has the salted form: no sync of Σ_sync's `clock_digest` itself (the unsalted form, which shows the
  sync), and no earlier link-window's clock digest in the period (a reused salt, under which two windows of one sync
  sign one digest); then one-stage v2 on the link-window (`OneStageZk.checkBlobs`), under the parameters `one_stage`
  with the link-window's own port (`ports`: `certificates`, four leaves under its served domain, `LinkWindow.domainId`,
  as `LinkWindows.ports` gives the challenger) and `log` the auditor's followed by the period's earlier link-windows'
  registrations, on its registration, its `one_stage` evidence and the blobs: v2's rejection rejects and its undecided
  leaves the period undecided (each named `one-stage v2: …`), and its acceptance must have the profile of one unit drawn
  `subset:1`; then the opening (`LinkWindow.pubRow`, `opened`, as N2's verdict of record makes it beside the proof,
  `LinkWindows.verify`): the public file's ``pub`` row commitment for the drawn unit opens under the delivered salt to
  the verifier's own public row on a plain link (both certificates' signed digests, then each end's registered root,
  served domain, link, window, grid and profile row), and its ``o0`` row commitment to the word 3, verdict and deadline
  bits 1. A proof of another link-window or of other certificates, or one whose verdict or deadline bit is 0, is refused
  there; so is a served or advised link's, whose unit reads the window's sync or advice in its public row, which the
  clock form holds as witness;
* in the rows form, link-window `w`: its registration (the scan's, per window) names the scan's Program; its three
  certificates are the requests' record's (the link's ingress), the responses' (its egress) and the fails'
  (`<link>-fails`, egress) at `w`, each the key's and verifying in the session, so a link-window missing any record's
  certificate is refused; its two registrations, ingress then egress, are what `warden_rows.check_registration` reads:
  the link-window's value, link, end and window, the schedule's digest (a registration under another schedule is
  refused), the schedule's rows, words, word bits, row format and layout, the link-window's domain in the session
  (`warden_rows.domain`), and a certificate that is the key's, signs the head (`CellsHead.digest`) and verifies in the
  session (a registration not under the deployment key is refused); the scan's registration holds both, an entry of its
  `  roots` or `registered` naming each one's root and domain, which bind the session, link, end and window (an answer
  for another window's rows is refused); and then the scan's answer is refused: the check takes no answer for a
  link-window, and one-stage v2 can't run the scan's proof itself (the scan's Program is a population of several
  templates drawn stratified, a statement each, which v2's one circuit, one public file and one instance a session
  don't cover), so no period of the rows form accepts until one-stage runs the scan;
* audit `i`: the answer is one-stage's for its registration's digest, accepted with one level drawn `subset:k` or
  stratified (a work law's draw is stratified), and its `bound`.

**Public**, on acceptance: `K` and its terms, each a ceiling of `log2` of `PeriodLeakAccepted`'s factor:

    {"K": K, "audits": [{"delta", "drawn": k, "faulty_bits", "faulty_sets", "free", "jitter", "units": N,
                         "wrong_units": b}, …],
     "cells": Σ_c ⌈log2 (cap_c + 1)⌉, "eps": Σ_i δ_i, "schedules": ⌈log2 #Σ_sync^m⌉, "windows": m}

with `b` the audit's bound on wrong units at its δ (one-stage's `bound`, `Bound.boundOf`), `faulty_sets` =
`⌈log2 Σ_{j ≤ b} C(N, j)⌉` (every set of at most `b` units, which holds the sets a `subset:k` or a stratified draw leaves
possible; `drawn` is the draw's `k`, or its `Σ k` stratified),
`faulty_bits` = `16 · b` (a proof unit's outputs are at most 16 bits), `jitter` = `⌈log2 Dj^steps⌉`, and `K` the sum of
the terms. The schedules term counts every assignment of a sync of Σ_sync to the `m` link-windows, so it is the same
whichever syncs the certifier ran them under, and needs none of them. `eps` is the probability that `K` fails to bound
the period's leak, the union of its audits' `δ`: their exact sum in lowest terms (`Core.Ratio.toString`, `"0"` with no
audits), as PoUS's and remote bandwidth's publics carry theirs.

In the rows form the public has `"scans": [terms, …]` beside the audits, each link-window's scan's terms as an
audit's: its `free` and `steps` the parameters' `scan`, `b` its answer's `bound` (over its draw's strata, `countBound`),
`drawn` its draw's `Σ k`, and `faulty_sets` every set of at most `b` of its `N` units, which holds the sets a stratified
draw leaves possible too. `schedules` is 0: the schedule is fixed, `#Σ = 1`. `K` adds the scans' terms and `eps` their
`δ`.

**Profile**: the link and `m`.

`opening` (`PeriodMain`'s `opening` command) is each link-window's opening alone, as the check makes it once v2 has
accepted, for the tests: no test makes a proof v2 accepts.

What it doesn't check: in the rows form, the records' certificates' digests against the rows (each signs its salted
record, witness); and that the audits' answers are the sampled-proofs component's own, which the composition's theorem
is to bind.
-/

namespace NetworkCheck.Period

open Lean (Json toJson)
open Flock (V)
open NetworkCertifier (ClockSync)
open NetworkCheck.Check (Certificate ratOf)

def protocolId : String := "verity/certifier/v0"
def commitmentTag : String := "verity-network-certifier/period/v0"
def clockTag : String := "verity-network-certifier/clock/v1"
def oneStage : String := "verity/sampled-proofs/one-stage/v1"
def formats : List String := ["verity/registration/v1", "verity/registration/v2"]
/-- A proof unit's outputs are at most 16 bits. -/
def outBits : Nat := 16
def unitLevel : String := "unit"
def scheduleTag : String := "verity-network-warden/parameters/v2"
def domainTag : String := "verity-network-warden/cells-domain/v2"
def cellsTag : String := "verity-network-warden/cells/v2"
def rowsFormat : String := "verity-network-warden/registration/v2"
def rowSchema : String := "hm96-sha512/row/v2"
/-- The cell kinds (`PAD`, `DATA`, `END`, `FAIL`, `REFUSED`) and the fail reasons the schedule's digest binds. -/
def cellKinds : List Nat := [0, 1, 32, 33, 34]
def failReasons : List String := ["late", "limit", "upstream", "ingress", "malformed"]
/-- A cell's header: its kind u8, its connection's ordinal u32 and its payload's length u32. -/
def cellHeader : Nat := 9

/-- A compute audit's layout terms: its free inputs it doesn't register, and its monitoring period. -/
structure Audit where
  free : Nat
  steps : Nat

/-- The network certifier's schedule when it functions as an active warden (`warden.WardenParams`). -/
structure Schedule where
  bucketMs : Nat
  cellBytes : Nat
  delay : Nat
  egressCells : Nat
  ingressBuckets : Nat
  ingressCells : Nat
  lanes : Nat
  queue : Nat
  slots : Nat
  window : Nat

/-- S = lanes · r, a row's slots. -/
def Schedule.allocation (s : Schedule) : Nat := s.lanes * s.slots

/-- The clock form's link as N2 declares it: Σ_sync, its run, each end's grid and occupancy profile, and the auditor's
one-stage v2 parameters for its link-windows (`oneStage`, without `ports`) with their partition object's digest. -/
structure Link where
  syncSet : NetworkCertifier.SyncSet
  run : String
  egress : LinkWindow.End
  ingress : LinkWindow.End
  oneStage : Json
  partition : String

/-- Which run the link-windows are: the clock form's, N2's on the link, or the rows form's, under one schedule (and its
digest), read by the scan whose layout terms are the last. -/
inductive Form where
  | clock (link : Link)
  | rows (schedule : Schedule) (digest : ByteArray) (scan : Audit)

structure Params where
  audits : Array Audit
  cells : Array Nat
  first : Nat
  jitter : Nat
  link : String
  lms : Lms.Params
  key : ByteArray
  session : ByteArray
  program : String
  form : Form
  windows : Nat

/-- A registration record and its one-stage digest. -/
structure Registration where
  record : Json
  digest : ByteArray

structure Commitment where
  audits : Array Registration
  windows : Array Registration
  bytes : ByteArray

structure Layout where
  buckets : Nat
  slots : Nat
  connectionSlots : Nat
  firstBucket : Nat
  deriving BEq

/-- A link-window's rows in one direction, as `warden_rows.registration` writes them. -/
structure Rows where
  format : String
  value : String
  link : String
  direction : String
  window : Nat
  parameters : String
  cellBytes : Nat
  rows : Nat
  words : Nat
  wordBits : Nat
  rowSchema : String
  domain : String
  wiring : ByteArray
  root : ByteArray
  layout : Layout
  key : ByteArray
  digest : ByteArray
  signature : ByteArray

/-- The salts N2's prover delivered for the drawn unit's ``pub`` and ``o0`` row commitments (`Delivered.salts`). -/
structure Salts where
  pub : ByteArray
  out : ByteArray

/-- What a link-window shows beside its certificates: the clock form's clock certificate, the delivered salts and
one-stage v2's evidence for its proof, or the rows form's registrations. -/
inductive Shown where
  | clock (c : Certificate) (salts : Salts) (oneStage : SampledProofs.OneStageZk.Evidence)
  | rows (rs : Array Rows)

structure Window where
  certificates : Array Certificate
  shown : Shown

def Window.clockDigest? (x : Window) : Option ByteArray :=
  match x.shown with
  | .clock c _ _ => some c.digest
  | .rows _ => none

structure Evidence where
  audits : Array Core.Answer
  windows : Array Window

/-- What an accepted audit establishes: its units, its draw, its δ, and its bound on wrong units. -/
structure Facts where
  units : Nat
  drawn : Nat
  delta : Core.Ratio
  wrong : Nat

/-! ## Decoding -/

private def keysOf : Json → List String
  | .obj kvs => kvs.keys
  | _ => []

private def exactly (j : Json) (want : List String) (ctx : String) : V Unit :=
  match j with
  | .obj _ => if keysOf j == want then pure () else throw s!"{ctx}: the keys must be {want}, not {keysOf j}"
  | _ => throw s!"{ctx}: not an object"

def isDigest (s : String) : Bool := s.length = 128 && s.all fun c => c.isDigit || ('a' ≤ c && c ≤ 'f')

def syncSetOf (j : Json) : V NetworkCertifier.SyncSet := do
  exactly j ["deltas", "rhos"] "sync_set"
  let deltas := (← (← Flock.J.arr j "deltas" "sync_set").mapM (Flock.J.asNat · "sync_set.deltas")).toList
  let rhos := (← (← Flock.J.arr j "rhos" "sync_set").mapM (ratOf · "sync_set.rhos")).toList
  if h : deltas ≠ [] ∧ rhos ≠ [] ∧ (∀ ρ ∈ rhos, 0 < ρ) ∧ deltas.Nodup ∧ rhos.Nodup then
    pure ⟨deltas, rhos, h.1, h.2.1, h.2.2.1, h.2.2.2.1, h.2.2.2.2⟩
  else throw "sync_set: non-empty, each delta and each rho listed once"

def paramsKeys : List String :=
  ["audits", "cells", "egress", "first", "ingress", "jitter", "key", "link", "lms", "one_stage", "run", "session",
    "sync_set", "windows"]

def rowsParamsKeys : List String :=
  ["audits", "cells", "first", "jitter", "key", "link", "lms", "program", "scan", "schedule", "session", "windows"]

def auditOf (a : Json) (ctx : String) : V Audit := do
  exactly a ["free", "steps"] ctx
  pure { free := ← Flock.J.nat a "free" ctx, steps := ← Flock.J.nat a "steps" ctx }

def scheduleOf (j : Json) : V Schedule := do
  exactly j ["bucket_ms", "cell_bytes", "delay", "egress_cells", "ingress_buckets", "ingress_cells", "lanes", "queue",
    "slots", "window"] "schedule"
  let pos : String → V Nat := fun k => do
    let x ← Flock.J.nat j k "schedule"
    unless 1 ≤ x do throw s!"schedule: {k} is a positive integer"
    pure x
  let s : Schedule :=
    { bucketMs := ← pos "bucket_ms", cellBytes := ← pos "cell_bytes", delay := ← pos "delay",
      egressCells := ← pos "egress_cells", ingressBuckets := ← pos "ingress_buckets",
      ingressCells := ← pos "ingress_cells", lanes := ← pos "lanes", queue := ← pos "queue", slots := ← pos "slots",
      window := ← pos "window" }
  unless cellHeader + 1 < s.cellBytes do throw s!"schedule: a cell holds its {cellHeader}-byte header and a byte"
  unless s.cellBytes < 2 ^ 32 ∧ s.allocation < 2 ^ 32 ∧ s.window < 2 ^ 32 do
    throw "schedule: B, S = lanes · slots and T = window are u32s"
  pure s

/-- An end of the clock form's link: its declared grid, exactly `LinkWindow.gridOf`'s keys, and its occupancy profile or
null (`LinkWindow.endOf`). -/
def endOf (j : Json) (ctx : String) : V LinkWindow.End := do
  exactly j ["grid", "profile"] ctx
  exactly (← Flock.J.get j "grid" ctx) ["allocation", "bucket_ms", "buckets", "frame_bytes", "queue_capacity"]
    s!"{ctx}.grid"
  LinkWindow.endOf j ctx

/-- `j` with `k` set to `v`, `j` an object. -/
def withKey (j : Json) (k : String) (v : Json) : Json :=
  match j with
  | .obj kvs => .obj (kvs.insert k v)
  | x => x

/-- The auditor's one-stage v2 parameters for the link-windows (`OneStageZk.paramsOfJson`'s with no `ports`, which are
each link-window's), and their partition object's `program` and digest. -/
def oneStageOf (j : Json) : V (String × String) := do
  if (keysOf j).contains "ports" then
    throw "one_stage: ports are each link-window's own (its served domain), which the check derives"
  let ps ← (SampledProofs.OneStageZk.paramsOfJson (withKey j "ports" (Json.mkObj []))).mapError (s!"one_stage: {·}")
  let .str program := ps.own.partition.getObjValD "program" | throw "one_stage: the partition names no program"
  let some partition := SampledProofs.Partition.digest ps.own.partition
    | throw "one_stage: the partition object has no digest"
  pure (program, partition)

/-- `WardenParams.digest`: the schedule's canonical JSON, with the cell kinds and the fail reasons, tagged. -/
def Schedule.digest (s : Schedule) : V ByteArray :=
  Identity.identityDigestSha512 scheduleTag (Json.mkObj [
    ("bucket_ms", toJson s.bucketMs), ("cell_bytes", toJson s.cellBytes), ("delay", toJson s.delay),
    ("egress_cells", toJson s.egressCells), ("ingress_buckets", toJson s.ingressBuckets),
    ("ingress_cells", toJson s.ingressCells), ("lanes", toJson s.lanes), ("queue", toJson s.queue),
    ("slots", toJson s.slots), ("window", toJson s.window), ("kinds", toJson cellKinds),
    ("reasons", toJson failReasons)])

def paramsOf (j : Json) : V Params := do
  match j with
  | .obj _ =>
    unless keysOf j == paramsKeys || keysOf j == rowsParamsKeys do
      throw s!"the object: the keys must be {paramsKeys} or {rowsParamsKeys}, not {keysOf j}"
  | _ => throw "the object: not an object"
  let audits ← (← Flock.J.arr j "audits" "the object").mapM (auditOf · "an audit")
  let cells ← (← Flock.J.arr j "cells" "the object").mapM (Flock.J.asNat · "cells")
  let first ← Flock.J.nat j "first" "the object"
  let jitter ← Flock.J.nat j "jitter" "the object"
  unless 1 ≤ jitter do throw "jitter: the observations' jitter bound is at least 1"
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
  let (program, form) ← if keysOf j == paramsKeys then do
      let syncSet ← syncSetOf (← Flock.J.get j "sync_set" "the object")
      let egress ← endOf (← Flock.J.get j "egress" "the object") "egress"
      let ingress ← endOf (← Flock.J.get j "ingress" "the object") "ingress"
      let oneStage ← Flock.J.get j "one_stage" "the object"
      let (program, partition) ← oneStageOf oneStage
      pure (program, Form.clock { syncSet, run := ← Flock.J.str j "run" "the object", egress, ingress, oneStage,
                                  partition })
    else do
      let s ← scheduleOf (← Flock.J.get j "schedule" "the object")
      pure (← Flock.J.str j "program" "the object",
        Form.rows s (← s.digest) (← auditOf (← Flock.J.get j "scan" "the object") "scan"))
  unless isDigest program do throw "program: not a SHA-512 digest (128 lowercase hex digits)"
  let windows ← Flock.J.nat j "windows" "the object"
  unless 1 ≤ windows do throw "windows: a period has at least one link-window"
  pure { audits, cells, first, jitter, link, lms, key, session, program, form, windows }

/-- `registration.digest`: SHA-512(format ‖ 0 ‖ the canonical record), the record's own format the tag. -/
def registrationOf (j : Json) (ctx : String) : V Registration := do
  let format ← Flock.J.str j "format" ctx
  unless formats.contains format do throw s!"{ctx}: format is not {" or ".intercalate formats}"
  let body ← Flock.canon j
  pure ⟨j, Flock.Sha512.hash (format.toUTF8 ++ ByteArray.mk #[0] ++ body.toUTF8)⟩

def commitmentOf (b : ByteArray) : V Commitment := do
  let j ← Core.canonicalJson b
  exactly j ["audits", "windows"] "the object"
  let audits ← (← Flock.J.arr j "audits" "the object").mapM (registrationOf · "an audit's registration")
  let windows ← (← Flock.J.arr j "windows" "the object").mapM (registrationOf · "a link-window's registration")
  pure { audits, windows, bytes := b }

def certificateOf (j : Json) (ctx : String) : V Certificate := do
  exactly j ["digest", "direction", "key_id", "link", "signature", "window"] ctx
  let direction ← Direction.ofValue (← Flock.J.str j "direction" ctx)
  pure { keyId := ← Flock.unhex (← Flock.J.str j "key_id" ctx), link := ← Flock.J.str j "link" ctx, direction,
         window := ← Flock.J.nat j "window" ctx, digest := ← Flock.unhex (← Flock.J.str j "digest" ctx),
         signature := ← Flock.unhex (← Flock.J.str j "signature" ctx) }

private def hexAt (j : Json) (k ctx : String) : V ByteArray := do Flock.unhex (← Flock.J.str j k ctx)

def rowsOf (j : Json) (ctx : String) : V Rows := do
  exactly j ["cell_bytes", "certificate", "direction", "domain", "format", "layout", "link", "parameters", "root",
    "row_schema", "rows", "value", "window", "wiring", "word_bits", "words"] ctx
  let l ← Flock.J.get j "layout" ctx
  exactly l ["buckets", "connection_slots", "first_bucket", "slots"] s!"{ctx}.layout"
  let c ← Flock.J.get j "certificate" ctx
  exactly c ["digest", "key", "signature"] s!"{ctx}.certificate"
  pure { format := ← Flock.J.str j "format" ctx, value := ← Flock.J.str j "value" ctx,
         link := ← Flock.J.str j "link" ctx, direction := ← Flock.J.str j "direction" ctx,
         window := ← Flock.J.nat j "window" ctx,
         parameters := ← Flock.J.str j "parameters" ctx, cellBytes := ← Flock.J.nat j "cell_bytes" ctx,
         rows := ← Flock.J.nat j "rows" ctx, words := ← Flock.J.nat j "words" ctx,
         wordBits := ← Flock.J.nat j "word_bits" ctx, rowSchema := ← Flock.J.str j "row_schema" ctx,
         domain := ← Flock.J.str j "domain" ctx, wiring := ← hexAt j "wiring" ctx, root := ← hexAt j "root" ctx,
         layout := { buckets := ← Flock.J.nat l "buckets" ctx, slots := ← Flock.J.nat l "slots" ctx,
                     connectionSlots := ← Flock.J.nat l "connection_slots" ctx,
                     firstBucket := ← Flock.J.nat l "first_bucket" ctx },
         key := ← hexAt c "key" ctx, digest := ← hexAt c "digest" ctx, signature := ← hexAt c "signature" ctx }

def windowKeys : List String := ["certificates", "clock", "one_stage", "salts"]
def rowsWindowKeys : List String := ["answer", "certificates", "registrations"]

def saltsOf (j : Json) (ctx : String) : V Salts := do
  exactly j ["o0", "pub"] ctx
  let s : Salts := { pub := ← hexAt j "pub" ctx, out := ← hexAt j "o0" ctx }
  unless s.pub.size == LinkWindow.saltBytes && s.out.size == LinkWindow.saltBytes do
    throw s!"{ctx}: a salt is {LinkWindow.saltBytes} bytes"
  pure s

/-- A link-window: in the clock form its certificates, clock certificate, delivered salts and one-stage v2's evidence
(`OneStageZk.evidenceOf`), and one that carries an answer is refused; in the rows form its certificates and its rows'
registrations, beside the scan's answer, which the check refuses. -/
def windowOf (j : Json) (ctx : String) : V Window := do
  let rows := keysOf j == rowsWindowKeys
  unless rows do
    if (keysOf j).contains "answer" then
      throw s!"{ctx}: answer: the check runs one-stage v2 on the link-window's proof itself and takes no answer for it"
    exactly j windowKeys ctx
  let certificates ← (← Flock.J.arr j "certificates" ctx).mapM (certificateOf · s!"{ctx}.certificates")
  let shown ← if rows then do
      pure (Shown.rows (← (← Flock.J.arr j "registrations" ctx).mapM (rowsOf · s!"{ctx}.registrations")))
    else do
      let clock ← certificateOf (← Flock.J.get j "clock" ctx) s!"{ctx}.clock"
      let salts ← saltsOf (← Flock.J.get j "salts" ctx) s!"{ctx}.salts"
      let b ← Flock.canon (← Flock.J.get j "one_stage" ctx)
      let e ← (SampledProofs.OneStageZk.evidenceOf b.toUTF8).mapError (s!"{ctx}.one_stage: {·}")
      pure (Shown.clock clock salts e)
  pure { certificates, shown }

def evidenceOf (b : ByteArray) : V Evidence := do
  let j ← Core.canonicalJson b
  exactly j ["audits", "windows"] "the object"
  let audits ← (← Flock.J.arr j "audits" "the object").mapM (Core.Answer.ofJson · "an audit's answer")
  let windows ← (← Flock.J.arr j "windows" "the object").mapM (windowOf · "a link-window")
  pure { audits, windows }

/-! ## The check -/

/-- `Certificate.verify` under the parameters' key in their session: the certificate is the key's (its I) and its LMS
signature verifies over `certificate_message` of its fields; a message that can't be formed or a signature that doesn't
decode is false. -/
def signs (p : Params) (c : Certificate) : Bool :=
  c.keyId == p.key.extract 0 16 &&
    match certificateMessage c.keyId p.session c.link c.direction c.window c.digest,
        Lms.Signature.decode p.lms c.signature with
    | .ok m, .ok sig => (Lms.verify p.lms p.key m sig).toOption == some true
    | _, _ => false

/-- `clock_digest`: `tagged_sha512("verity-network-certifier/clock/v1", u64be δ ‖ u64be p ‖ u64be q)` of the sync, ρ = p/q
in lowest terms; `none` past a u64. -/
def clockDigest (s : ClockSync) : Option ByteArray :=
  (do
    let d ← uBE 8 s.delta "δ"
    let n ← uBE 8 s.rho.num "p"
    let q ← uBE 8 s.rho.den "q"
    Identity.taggedSha512 clockTag (d ++ n ++ q)).toOption

/-- Each sync of Σ_sync's `clock_digest` itself: the unsalted form, whose digest shows the sync to anyone holding
Σ_sync. -/
def clearClocks (ss : NetworkCertifier.SyncSet) : List ByteArray :=
  ss.deltas.flatMap fun d => ss.rhos.attach.filterMap fun ρ => clockDigest ⟨d, ρ.1, ss.rhos_pos ρ.1 ρ.2⟩

/-- A certificate of `link`'s end `d` at window `w`, the key's and verifying. -/
def certifiesOn (p : Params) (link : String) (d : Direction) (w : Nat) (c : Certificate) : Bool :=
  c.direction == d && c.link == link && c.window == w && signs p c

/-- A certificate of the link's end `d` at window `w`, the key's and verifying. -/
def certifies (p : Params) (d : Direction) (w : Nat) (c : Certificate) : Bool := certifiesOn p p.link d w c

/-- A finding against the period: a rejection, or what leaves it undecided. -/
abbrev Found := Except (Core.Verdict × String)

def refuse {α} (r : String) : Found α := .error (.reject, r)

/-- One-stage's answer for the registration: its reject is a rejection, its undecided leaves the period undecided, and its
acceptance gives the profile. -/
def answered (what : String) (reg : Registration) (a : Core.Answer) : Found (Core.Integrity × Json) := do
  if a.protocol != oneStage then refuse s!"{what}: protocol {a.protocol}, not one-stage's {oneStage}"
  else if a.digest != some reg.digest then refuse s!"{what}: the answer is for another registration"
  else match a.verdict, a.published, a.profile with
    | .accept, some pub, some prof =>
      match Core.Integrity.ofJson prof with
      | .ok i => pure (i, pub)
      | .error e => refuse s!"{what}: {e}"
    | .undecided, _, _ => .error (.undecided, s!"{what}: service: {a.reason.getD ""}")
    | _, _, _ => refuse s!"{what}: service: {a.reason.getD ""}"

def served (w : Nat) : Json := Json.mkObj [("kind", "served"), ("window", toJson w)]

/-- The canonical text of a value read from canonical bytes, which `Flock.canon` always writes. -/
def canonText (j : Json) : Option String := (Flock.canon j).toOption

/-- The link-window's registered root: its registration's `roots` entry for the `certificates` port
(`LinkWindows.registration`), empty for none. -/
def rootOf (reg : Registration) : ByteArray :=
  let roots : Array Json := match reg.record.getObjValD "roots" with | .arr rs => rs | _ => #[]
  let root : Option Json :=
    (roots.find? fun r => r.getObjValD "port" == .str LinkWindow.value).map fun r => r.getObjValD "root"
  match root with
  | some (.str h) => (Flock.unhex h).toOption.getD .empty
  | _ => .empty

/-- Link-window `w` of the clock form's link as N2's verdict of record reads it (`LinkWindow.Case`, a plain link): its
registered root, its certificates, the units its proof proved, its public file and the delivered salts, the output the
word 3; the fields neither its served domain nor its opening reads are empty. -/
def linkCase (p : Params) (l : Link) (w : Nat) (root : ByteArray) (certs : Array Certificate) (units : Array Nat)
    (statement : Option ByteArray) (s : Salts) : LinkWindow.Case :=
  { link := p.link, run := l.run, window := w, root, registration := "", units, program := p.program,
    partition := l.partition, gates := #[], findings := none, statement, saltPub := some s.pub, saltOut := some s.out,
    output := some 3, sessions := #[],
    certificates := certs.map fun c => some { keyId := c.keyId, link := c.link, direction := c.direction,
                                              window := c.window, digest := c.digest, signature := c.signature },
    digests := certs.map (·.digest), syncs := #[], deltas := l.syncSet.deltas.toArray,
    rhos := l.syncSet.rhos.toArray.map fun ρ => (ρ.num.toNat, ρ.den), lms := p.lms, key := p.key,
    session := p.session, egress := l.egress, ingress := l.ingress, kind := .plain }

/-- One-stage v2's parameters for a link-window whose served domain is `dom`: the auditor's, with the link-window's own
port (`LinkWindows.ports`: `certificates`, four leaves under `dom`) and the period's `earlier` link-windows'
registrations after the auditor's `log`. -/
def windowParams (l : Link) (dom : ByteArray) (earlier : Array Json) : V SampledProofs.OneStageZk.Params :=
  let log := match l.oneStage.getObjValD "log" with | .arr xs => xs | _ => #[]
  let port := Json.mkObj [("domain", .str (Flock.hexOf dom)), ("leaves", toJson LinkWindow.leaves)]
  SampledProofs.OneStageZk.paramsOfJson
    (withKey (withKey l.oneStage "ports" (Json.mkObj [(LinkWindow.value, port)])) "log" (.arr (log ++ earlier)))

/-- Link-window `w`'s opening, once one-stage v2 accepted its proof (`LinkWindows.verify`'s `_opened`): the public
file's ``pub`` row commitment for the drawn unit opens under the delivered salt to the verifier's own public row
(`LinkWindow.pubRow` of `linkCase`), and its ``o0`` row commitment to the word 3 (`LinkWindow.opened`). -/
def opening (p : Params) (l : Link) (w : Nat) (reg : Registration) (certs : Array Certificate) (s : Salts)
    (e : SampledProofs.OneStageZk.Evidence) (bs : Core.Blobs) : Found Unit := do
  let what := s!"window {w}: opening"
  let .ok file := bs.get e.pub "public file" | refuse s!"{what}: the public file is in no blob"
  let c := linkCase p l w (rootOf reg) certs e.base.provedUnits (some file) s
  match LinkWindow.pubRow c with
  | .error err => refuse s!"{what}: the verifier's public row: {err}"
  | .ok none =>
    refuse s!"{what}: the verifier's public row has no instance at window {w} (an end's profile ends before)"
  | .ok (some row) =>
    unless LinkWindow.opened c (some row) == some 3 do
      refuse (s!"{what}: the proof's public row doesn't open to the verifier's own under the delivered salt, or its " ++
        "output row to the word 3 (verdict and deadline bits 1)")

/-- Link-window `w`'s findings, in the order the module docstring gives, `clear` being `clearClocks` of Σ_sync,
`earlier` the period's earlier link-windows' clock digests and `log` their registrations. -/
def windowCheck (p : Params) (l : Link) (clear : List ByteArray) (earlier : Array ByteArray) (log : Array Json)
    (w : Nat) (reg : Registration) (x : Window) (bs : Core.Blobs) : Found Unit := do
  let what := s!"window {w}"
  let .clock clock salts e := x.shown | refuse s!"{what}: registrations, not a clock certificate: the rows form's"
  unless reg.record.getObjValD "program" == .str p.program &&
      canonText (reg.record.getObjValD "window") == canonText (served w) do
    refuse s!"{what}: registration: not the link's Program at window {w}"
  unless x.certificates.size == 2 && (x.certificates.zip #[Direction.egress, .ingress]).all
      (fun (c, d) => certifies p d w c) do
    refuse s!"{what}: certificates"
  unless certifies p .egress w clock do refuse s!"{what}: clock"
  if clear.contains clock.digest then refuse s!"{what}: clock: a sync's clock digest unsalted"
  if earlier.contains clock.digest then refuse s!"{what}: clock: an earlier link-window's digest, its salt reused"
  let .ok dom := LinkWindow.domainId (linkCase p l w .empty #[] #[] none salts)
    | refuse s!"{what}: one-stage v2: the link-window's served domain doesn't encode"
  let ps ← match windowParams l dom log with
    | .ok ps => pure ps
    | .error err => refuse s!"{what}: one-stage v2: params: {err}"
  let prof ← match SampledProofs.OneStageZk.checkBlobs ps ⟨reg.record, reg.digest⟩ e bs with
    | .accept _ prof => pure prof
    | .reject r => refuse s!"{what}: one-stage v2: {r}"
    | .undecided r => .error (.undecided, s!"{what}: one-stage v2: {r}")
  unless prof.levels.map (fun lv => (lv.name, lv.size, lv.draw)) == #[(unitLevel, 1, some (Core.Draw.subset 1))] &&
      prof.drawn == #[(unitLevel, 1)] do
    refuse s!"{what}: profile: not the link-window's one unit proved"
  opening p l w reg x.certificates salts e bs

/-- `warden_rows.domain`'s id: the frame-v3-sha512 domain of link-window `w`'s rows in direction `d`, in the session,
under the schedule whose digest is `sd`. -/
def rowsDomain (s : Schedule) (sd session : ByteArray) (link : String) (d : Direction) (w : Nat) : V ByteArray := do
  let binding ← Identity.identityDigestSha512 domainTag (Json.mkObj [
    ("parameters", .str (Flock.hexOf sd)), ("session", .str (Flock.hexOf session)), ("link", .str link),
    ("direction", .str d.value), ("window", toJson w), ("cell_bytes", toJson s.cellBytes),
    ("slots", toJson s.allocation), ("rows", toJson s.window), ("row_schema", .str rowSchema)])
  let positions ← Identity.identityDigestSha512 "veritor/indexed-domain/range/v1"
    (Json.mkObj [("start", toJson (0 : Nat)), ("step", toJson (1 : Nat)), ("stop", toJson s.window)])
  pure (Flock.FV3.hash "domain" [binding, Flock.FV3.uint 1, positions, Flock.FV3.uint s.window] Flock.Sha512.hash)

/-- `CellsHead.digest`: `tagged_sha512(CELLS_TAG, H)` of the head the network certifier signs for the rows. -/
def headDigest (link : String) (d : Direction) (w : Nat) (sd : ByteArray) (s : Schedule) (wiring dom root : ByteArray) :
    V ByteArray := do
  unless wiring.size == 64 && root.size == 64 do throw "the wiring digest and the root are 64 bytes"
  Identity.taggedSha512 cellsTag ((← text link "link") ++ (← uBE 1 d.code "direction") ++ (← uBE 8 w "window") ++ sd ++
    (← uBE 4 s.cellBytes "B") ++ (← uBE 4 s.allocation "S") ++ (← uBE 4 s.window "T") ++ wiring ++ dom ++ root)

/-- `warden_rows.check_registration` of the link's rows in direction `d` at window `w`. -/
def rowsCheck (p : Params) (s : Schedule) (sd : ByteArray) (w : Nat) (d : Direction) (r : Rows) : Found Unit := do
  let what := s!"window {w}: the {d.value} rows"
  let (B, S, T) := (s.cellBytes, s.allocation, s.window)
  unless r.format == rowsFormat && r.value == s!"{p.link}/{d.value}/{w}" && r.link == p.link &&
      r.direction == d.value && r.window == w do
    refuse s!"{what}: not the link's {d.value} rows at window {w}"
  unless r.parameters == Flock.hexOf sd do refuse s!"{what}: another schedule's digest than the bound one"
  unless r.cellBytes == B && r.rows == T && r.words == S * B / 2 && r.wordBits == outBits && r.rowSchema == rowSchema &&
      r.layout == ({ buckets := T, slots := S, connectionSlots := s.slots, firstBucket := w * T } : Layout) do
    refuse s!"{what}: not the schedule's rows"
  let .ok dom := rowsDomain s sd p.session p.link d w | refuse s!"{what}: no domain"
  unless r.domain == Flock.hexOf dom do refuse s!"{what}: not the link-window's domain"
  let .ok head := headDigest p.link d w sd s r.wiring dom r.root | refuse s!"{what}: the head doesn't encode"
  unless r.digest == head do refuse s!"{what}: the certificate doesn't sign the head"
  unless certifiesOn p p.link d w
      { keyId := r.key, link := p.link, direction := d, window := w, digest := r.digest, signature := r.signature } do
    refuse s!"{what}: not under the deployment key in the session"

/-- The scan's registration holds the rows: an entry of its `roots` or `registered` names their root and domain. -/
def holds (reg : Registration) (r : Rows) : Bool :=
  let entries := fun (k : String) => match reg.record.getObjValD k with | .arr xs => xs | _ => #[]
  (entries "roots" ++ entries "registered").any fun e =>
    e.getObjValD "root" == .str (Flock.hexOf r.root) && e.getObjValD "domain" == .str r.domain

/-- The certified records of the rows form's link-window: the requests' (ingress), responses' (egress) and fails'. -/
def records (link : String) : Array (String × Direction) :=
  #[(link, .ingress), (link, .egress), (link ++ "-fails", .egress)]

/-- An accepted scan's or audit's facts: one level drawn `subset:k` or stratified (a work draw's profile is stratified,
its `Σ k_s` drawn), and its bound. -/
def factsOf (what : String) (prof : Core.Integrity) (pub : Json) : Found Facts := do
  let #[l] := prof.levels | refuse s!"{what}: profile: not one level"
  let drawn ← match l.draw with
    | some (.subset k) => pure k
    | some (.stratified ss) => pure (ss.foldl (fun n st => n + st.k) 0)
    | _ => refuse s!"{what}: profile: not a subset or stratified draw"
  match Flock.J.nat pub "bound" "public" with
  | .ok b => pure { units := l.size, drawn, delta := prof.delta, wrong := b }
  | .error e => refuse s!"{what}: {e}"

/-- Link-window `w`'s findings in the rows form, in the order the module docstring gives, ending in the refusal of the
scan's answer: no scan's facts until one-stage runs the scan. -/
def rowsWindowCheck (p : Params) (s : Schedule) (sd : ByteArray) (w : Nat) (reg : Registration) (x : Window) :
    Found Facts := do
  let what := s!"window {w}"
  let .rows rs := x.shown | refuse s!"{what}: a clock certificate, not registrations: the clock form's"
  unless reg.record.getObjValD "program" == .str p.program do refuse s!"{what}: registration: not the scan's Program"
  unless x.certificates.size == 3 && (x.certificates.zip (records p.link)).all
      (fun (c, l, d) => certifiesOn p l d w c) do
    refuse s!"{what}: certificates: the requests', responses' and fails' records', each the key's"
  unless rs.size == 2 do refuse s!"{what}: registrations: the ingress rows, then the egress rows"
  for (r, d) in rs.zip #[Direction.ingress, .egress] do rowsCheck p s sd w d r
  unless rs.all (holds reg) do refuse s!"{what}: the scan's registration doesn't hold the rows: another window's answer"
  refuse (s!"{what}: the scan's answer: the check takes no answer for a link-window, and one-stage v2 can't run the " ++
    "scan's proof (several templates drawn stratified, a statement each)")

/-- Audit `i`'s findings, and on acceptance its facts. -/
def auditCheck (i : Nat) (reg : Registration) (a : Core.Answer) : Found Facts := do
  let what := s!"audit {i}"
  let (prof, pub) ← answered what reg a
  factsOf what prof pub

def findingOf {α : Type} : Found α → Option (Core.Verdict × String)
  | .error x => some x
  | .ok _ => none

/-- The first rejection among the findings, else the first undecided one. -/
def firstFinding (fs : List (Option (Core.Verdict × String))) : Option (Core.Verdict × String) :=
  match fs.findSome? fun f => f.filter (·.1 == .reject) with
  | some r => some r
  | none => fs.findSome? id

/-! ## The bound -/

/-- `⌈log2 x⌉`, 0 for x ≤ 1. -/
def clog2 (x : Nat) : Nat := if x ≤ 1 then 0 else Nat.log2 (x - 1) + 1

/-- `Σ_{j ≤ b} C(n, j)`: the sets of at most `b` of `n` units. -/
def atMost (n b : Nat) : Nat := Id.run do
  let mut c := 1
  let mut s := 1
  for j in [0:min b n] do
    c := c * (n - j) / (j + 1)
    s := s + c
  return s

structure AuditTerms where
  facts : Facts
  free : Nat
  faultySets : Nat
  faultyBits : Nat
  jitter : Nat

def AuditTerms.bits (t : AuditTerms) : Nat := t.free + t.faultySets + t.faultyBits + t.jitter

/-- An audit's or a scan's terms from its layout terms, its facts and the jitter bound. -/
def AuditTerms.of (a : Audit) (f : Facts) (jitter : Nat) : AuditTerms :=
  { facts := f, free := a.free, faultySets := clog2 (atMost f.units f.wrong), faultyBits := outBits * f.wrong,
    jitter := clog2 (jitter ^ a.steps) }

structure Public where
  audits : Array AuditTerms
  cells : Nat
  /-- The rows form's scans' terms, one a link-window; `none` in the clock form. -/
  scans : Option (Array AuditTerms) := none
  schedules : Nat
  windows : Nat

/-- The audits' terms, then the scans'. -/
def Public.units (x : Public) : Array AuditTerms := x.audits ++ x.scans.getD #[]

def Public.K (x : Public) : Nat := x.schedules + x.cells + (x.units.map (·.bits)).foldl (· + ·) 0

/-- `Σ_i δ_i` over the audits and the scans, exactly, in lowest terms: the union bound on `K` failing. -/
def Public.eps (x : Public) : Core.Ratio :=
  let (n, d) := x.units.foldl (fun (n, d) t => (n * t.facts.delta.den + t.facts.delta.num * d, d * t.facts.delta.den))
    (0, 1)
  ⟨n / Nat.gcd n d, d / Nat.gcd n d⟩

/-- The bound's terms from the audits' layout terms and facts, the cells' caps, the jitter bound, `#Σ_sync` and the
link-windows. -/
def terms (audits : Array Audit) (facts : Array Facts) (cells : Array Nat) (jitter syncs windows : Nat) : Public :=
  { audits := (audits.zip facts).map fun (a, f) => AuditTerms.of a f jitter
    cells := (cells.map fun cap => clog2 (cap + 1)).foldl (· + ·) 0
    schedules := clog2 (syncs ^ windows)
    windows }

/-- The rows form's terms: the audits' and the cells' as the clock form's under one schedule, and each link-window's
scan's from the scan's layout terms. -/
def rowsTerms (audits : Array Audit) (facts : Array Facts) (scan : Audit) (scans : Array Facts) (cells : Array Nat)
    (jitter windows : Nat) : Public :=
  { terms audits facts cells jitter 1 windows with scans := some (scans.map (AuditTerms.of scan · jitter)) }

structure Profile where
  link : String
  windows : Nat

def check (p : Params) (c : Commitment) (e : Evidence) (bs : Core.Blobs) : Core.Outcome Public Profile :=
  if c.windows.size ≠ p.windows then
    .reject s!"the commitment registers {c.windows.size} link-windows, not the period's {p.windows}"
  else if c.audits.size ≠ p.audits.size then
    .reject s!"the commitment registers {c.audits.size} audits, not the parameters' {p.audits.size}"
  else if e.windows.size ≠ c.windows.size then
    .reject s!"evidence: {e.windows.size} link-windows for {c.windows.size} registered"
  else if e.audits.size ≠ c.audits.size then
    .reject s!"evidence: {e.audits.size} audits' answers for {c.audits.size} registered"
  else
    let pairs := (c.windows.zip e.windows).zipIdx
    let ws : Array (Found (Option Facts)) := match p.form with
      | .clock l =>
        let clear := clearClocks l.syncSet
        pairs.map fun ((reg, x), i) => (fun () => none) <$>
          windowCheck p l clear ((e.windows.extract 0 i).filterMap (·.clockDigest?))
            ((c.windows.extract 0 i).map (·.record)) (p.first + i) reg x bs
      | .rows s sd _ => pairs.map fun ((reg, x), i) => some <$> rowsWindowCheck p s sd (p.first + i) reg x
    let as := (c.audits.zip e.audits).zipIdx.map fun ((reg, a), i) => auditCheck i reg a
    match firstFinding (ws.toList.map findingOf ++ as.toList.map findingOf) with
    | some (.reject, r) => .reject r
    | some (_, r) => .undecided r
    | none =>
      let audits := as.filterMap (·.toOption)
      .accept (match p.form with
          | .clock l =>
            terms p.audits audits p.cells p.jitter (l.syncSet.deltas.length * l.syncSet.rhos.length) p.windows
          | .rows _ _ scan =>
            rowsTerms p.audits audits scan (ws.filterMap (·.toOption.join)) p.cells p.jitter p.windows)
        { link := p.link, windows := p.windows }

def termsJson (ts : Array AuditTerms) : Json :=
  .arr <| ts.map fun t => Json.mkObj [
    ("delta", .str t.facts.delta.toString), ("drawn", toJson t.facts.drawn), ("faulty_bits", toJson t.faultyBits),
    ("faulty_sets", toJson t.faultySets), ("free", toJson t.free), ("jitter", toJson t.jitter),
    ("units", toJson t.facts.units), ("wrong_units", toJson t.facts.wrong)]

def publicJson (x : Public) : Json :=
  Json.mkObj ([
    ("K", toJson x.K), ("audits", termsJson x.audits), ("cells", toJson x.cells), ("eps", .str x.eps.toString),
    ("schedules", toJson x.schedules), ("windows", toJson x.windows)] ++
    (x.scans.map fun ss => [("scans", termsJson ss)]).getD [])

def profileJson (pr : Profile) : Json :=
  Json.mkObj [("link", .str pr.link), ("windows", toJson pr.windows)]

def protocol : Core.BlobProtocol where
  id := protocolId
  Params := Params
  Commitment := Commitment
  Evidence := Evidence
  Public := Public
  Profile := Profile
  params := paramsOf
  commitment := commitmentOf
  evidence := evidenceOf
  digest c := (Identity.taggedSha512 commitmentTag c.bytes).toOption.getD .empty
  check p c e := check p c e Core.Blobs.empty
  publicJson := publicJson
  profileJson := profileJson
  checkBlobs := check

/-- Each clock-form link-window's `opening` alone, as `check` makes it once one-stage v2 accepted, on the case's three
byte strings and the blobs: none where it opens to the word 3, else the refusal's reason. For the tests, which make no
proof v2 accepts. -/
def openings (params commitment evidence : ByteArray) (bs : Core.Blobs) : V (Array (Option String)) := do
  let p ← paramsOf (← Core.canonicalJson params)
  let .clock l := p.form | throw "the rows form opens no proof"
  let c ← commitmentOf commitment
  let e ← evidenceOf evidence
  unless c.windows.size == e.windows.size do throw "the commitment and the evidence hold other link-windows"
  pure <| (c.windows.zip e.windows).zipIdx.map fun ((reg, x), i) =>
    match x.shown with
    | .clock _ s ev =>
      match opening p l (p.first + i) reg x.certificates s ev bs with
      | .ok () => none
      | .error (_, r) => some r
    | .rows _ => some "a rows-form link-window"

#guard clog2 0 == 0 && clog2 1 == 0 && clog2 2 == 1 && clog2 3 == 2 && clog2 4 == 2 && clog2 5 == 3
#guard atMost 10 0 == 1 && atMost 10 1 == 11 && atMost 10 2 == 56 && atMost 4 9 == 16 && atMost 0 3 == 1
-- an audit of 10 units drawn subset:3 at δ = 1/10 (at most 4 wrong), 7 cells on one connection and none on another,
-- 2 syncs over 3 link-windows, and jitter below 2 over 3 steps
#guard (terms #[⟨0, 3⟩] #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] #[7, 0] 2 2 3).K ==
  3 + 3 + (0 + 9 + 64 + 3)
-- two audits at δ = 1/100 and 1/10^6: ε = 10001/1000000
#guard (terms #[⟨0, 1⟩, ⟨0, 1⟩] #[{ units := 4, drawn := 1, delta := ⟨1, 100⟩, wrong := 0 },
  { units := 4, drawn := 1, delta := ⟨1, 1000000⟩, wrong := 0 }] #[] 1 1 1).eps.toString == "10001/1000000"
-- eps over audits at δ = 1/10, 1/15 and 1/6 is 1/3; over none it is 0
#guard (terms #[⟨0, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩] #[⟨1, 1, ⟨1, 10⟩, 0⟩, ⟨1, 1, ⟨1, 15⟩, 0⟩, ⟨1, 1, ⟨1, 6⟩, 0⟩] #[] 1 1 1
  ).eps.toString == "1/3"
#guard (terms #[] #[] #[] 1 1 1).eps.toString == "0"
-- the rows form over 3 link-windows: no schedules term, 7 cells on one connection, an audit of 10 units drawn
-- subset:3 with at most 4 wrong as above, and one scan of 20 units drawn at δ = 1/15 with at most 3 wrong, 1 free input
-- and jitter below 2 over 2 steps: 1351 sets of at most 3 of 20 units
#guard atMost 20 3 == 1351 && clog2 1351 == 11
#guard (rowsTerms #[⟨0, 3⟩] #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] ⟨1, 2⟩
  #[{ units := 20, drawn := 5, delta := ⟨1, 15⟩, wrong := 3 }] #[7] 2 3).K ==
  0 + 3 + (0 + 9 + 64 + 3) + (1 + 11 + 48 + 2)
-- eps over the audit's δ = 1/10 and the scans' 1/15 and 1/6 is 1/3
#guard (rowsTerms #[⟨0, 0⟩] #[⟨1, 1, ⟨1, 10⟩, 0⟩] ⟨0, 0⟩ #[⟨1, 1, ⟨1, 15⟩, 0⟩, ⟨1, 1, ⟨1, 6⟩, 0⟩] #[] 1 2
  ).eps.toString == "1/3"
#guard (rowsTerms #[] #[] ⟨0, 0⟩ #[] #[] 1 1).schedules == 0
-- a clock-form link-window that carries an answer is refused by name, whatever else it carries
#guard match windowOf (Json.mkObj [("answer", Json.mkObj [("verdict", Json.str "accept")]),
    ("certificates", Json.arr #[])]) "a link-window" with
  | .error r =>
    r == "a link-window: answer: the check runs one-stage v2 on the link-window's proof itself and takes no " ++
      "answer for it"
  | .ok _ => false
-- the auditor's one-stage v2 parameters carry no ports: each link-window's is its served domain
#guard match oneStageOf (Json.mkObj [("ports", Json.mkObj [])]) with
  | .error r => r == "one_stage: ports are each link-window's own (its served domain), which the check derives"
  | .ok _ => false
#guard match saltsOf (Json.mkObj [("o0", Json.str "00"), ("pub", Json.str "00")]) "salts" with
  | .error r => r == "salts: a salt is 192 bytes"
  | .ok _ => false
#guard (withKey (Json.mkObj [("a", toJson 1), ("b", toJson 2)]) "a" (toJson 3)).getObjValD "a" == toJson 3 &&
  (withKey (Json.mkObj [("a", toJson 1), ("b", toJson 2)]) "a" (toJson 3)).getObjValD "b" == toJson 2 &&
  withKey (toJson 1) "a" (toJson 3) == toJson 1

end NetworkCheck.Period
