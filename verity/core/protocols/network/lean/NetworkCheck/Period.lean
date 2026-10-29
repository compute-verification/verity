import NetworkCheck.Check
import CoreProtocol.Profile
import Flock.Program
import Flock.Public
import Flock.WindowRecord
import NciCheck.Desc

/-!
# The network certifier over a period, a core protocol: `verity/certifier/v0`

Onsite's communication component (`Core.Onsite.components`). A period is one node's link over `m` link-windows beside
the period's compute audits (sampled proofs' one-stage, one answer per registered window of the workload). Its
link-windows come in one of two forms, which the parameters name:

* **the clock form**: N2's link-windows (`service.LinkWindows`, Selection `All`), each proved, both its ends certified
  and its clock certified;
* **the rows form**: the network certifier's own run, where it functions as an active warden (`warden_serve`): each
  link-window's cells committed as rows (`warden_rows`), one registration a direction under one fixed schedule, its
  three records certified, and the scan's one-stage answer for the window (`scan_zk --warden-run`).

It accepts when every link-window passes its form's check and every compute audit accepted; it then publishes `K`, the
bits a covert sender in the server can move through the network certifier in the period, as
`NetTiming.Guarantees.PeriodLeakAccepted` counts them. In the rows form each link-window's scan's Programs' row must be a
compute audit's slot entries (`wired`, below), and the state that audit's Programs read (its `SCAN_STATE`) the scan's
state, by root and domain (`stateTied`), so a slot's entry is the one its Program computed on the requests the scan
read.

**Params**, JSON with exactly these keys (and each nested object exactly its own), in the clock form

    {"audits": [{"free": f, "program": descriptor, "steps": k}, …], "cells": [cap, …], "first": w₀, "jitter": Dj,
     "key": hex(I ‖ T[1]), "link": name, "lms": {"H": h, "N": n, "W": w, "toy": bool}, "program": hex,
     "session": hex(session), "sync_set": {"deltas": [δ, …], "rhos": [[num, den], …]}, "windows": m}

each compute audit's layout terms (`free`, the free inputs it doesn't register, `#(freeInputs \ reg)`; `steps`, its
monitoring period `k`) and its Program, the descriptor (`verity-ir/descriptor/v1`) of the window's population program,
which must decode (`Flock.Program.decode`) with a template in its root: the audit's units are the root's template
instances, and each commits its template's return, the outputs it computes that other units or the client read, which
can be wider than a proof unit's 16-bit public output; each network certifier connection's cell cap over the period
(`cap_c`), the session's window
the period starts at (`w₀`: its link-windows are the session's `w₀ … w₀ + m − 1`, so one certifier session spans many
periods),
the observations' jitter bound `Dj ≥ 1`, the network certifier's LMS public key and parameters and the certifier session
(32 bytes) every certificate binds, the link (at most 65,535 UTF-8 bytes), the SHA-512 of N2's registered Program for
the link (the partition object's `program`, `LinkWindows.partition`), Σ_sync, and the period's `m ≥ 1` link-windows.
The rows form has, in place of `sync_set`,

    "scan": descriptor,
    "schedule": {"bucket_ms", "cell_bytes", "delay", "egress_cells", "every", "ingress_buckets", "ingress_cells",
                 "lanes", "queue", "slots", "window"}

the scan's Program, its descriptor (`verity-ir/descriptor/v1`, `scan_zk`'s `program.json`), and the network
certifier's one fixed schedule (`warden.WardenParams`: each a positive integer, B = `cell_bytes` past a cell's 9-byte
header, B, S = `lanes` · `slots` and T = `window` u32s, and `every`, a timestep's buckets, dividing T, so each
link-window is T / `every` timesteps of `lanes` Program slots, `Schedule.programSlots` and `Schedule.windowPrograms`),
whose digest `WardenParams.digest` every registration binds; its `program` is the scan's registered Program, the
descriptor's SHA-512 (`partition_object.program_sha512`), and its `link` the run's (`warden_serve.LINK`), whose fails record is certified under the link `<link>-fails`. The
descriptor must decode (`Flock.Program.decode`) and be the scan of the schedule's link-window in the generator form:
its root batches one instance of a template whose parameters are `row` and `prog` (`IngressScanFirst`, bucket 0), T − 1
of one whose are `row`, `sin` and `prog` (`IngressScanBucket`; none when T = 1), each `row` the schedule's (S · B
bytes) and both returning the same, then T of one whose parameter is `state` and which returns a u16 and the
schedule's row (`EgressBucket`), in the Boolean profile's types (a byte 8 bits, a u16 16), and its every input gate is
a template's parameter. An egress template that takes its row as an input, the checker form, is refused by name: no
egress gate is an input (`NetTiming.Layout.Fits`), so `K` counts only what a Program that computes its egress rows
leaves free. The rest of what `NetTiming.Layout.Fits` asks is read off the description (`scanOf`): the Program laid out
as the decoder lays one out and within its resolution's walk (`NciCheck.Desc`), every egress leaf returned by the root
and computed by the egress template, and every template's argument one input node's leaves, no egress template's an
ingress row's. In the rows form a compute audit's free inputs are in the term of each link-window whose scan's `prog`
is its output (`tiedAudits`), and its other layout terms and its units' bits are in no term; their Programs are checked
as in the clock form.

**Commitment**, canonical JSON `{"audits": [registration, …], "windows": [registration, …]}`: one-stage's registration
record of each compute audit and of each link-window, in order, each of format `verity/registration/v1` or `v2`. Its
digest is `tagged_sha512("verity-network-certifier/period/v0", the commitment's bytes)`, and each registration's is
one-stage's, SHA-512(format ‖ 0 ‖ canonical record) (`registration.digest`).

**Evidence**, canonical JSON:

    {"audits": [answer, …],
     "windows": [{"answer": answer, "certificates": [certificate, certificate], "clock": certificate}, …]}
    certificate = {"digest": hex, "direction": "egress" | "ingress", "key_id": hex, "link": s, "signature": hex,
                   "window": w}

in the clock form, and in the rows form each link-window
`{"answer": answer, "certificates": [certificate, certificate, certificate], "registrations": [rows, rows]}`, with

    rows = {"cell_bytes", "certificate": {"digest", "key", "signature"}, "direction", "domain", "format",
            "layout": {"buckets", "connection_slots", "first_bucket", "slots"}, "link", "parameters", "root",
            "row_schema", "rows", "value", "window", "wiring", "word_bits", "words"}

the requests', responses' and fails' records' certificates (`certified.jsonl`'s), and the ingress and the egress rows'
registrations (`warden_rows.registration`, `cells.jsonl`'s); a link-window of the other form than the parameters' is
refused.

In the clock form: each audit's and each link-window's one-stage answer (`Core.Answer.ofJson`), and per link-window
the two certificates (`certificate.Certificate`, egress first) and the clock certificate the certifier, the auditor's
device, issued for it:
the public part of `certificate.CertifiedClock`, which signs the window's sync salted, `clock_commitment` =
SHA-512("verity/net-cert/v1" ‖ salt_c ‖ `clock_digest(sync)`). What the certifier signs is witness: the sync and its
salt are in neither the evidence nor the profile, and the check takes neither. The certifier is the auditor's trusted
device (`trusted-host`), and its own code picks each window's sync from its schedule set Σ_sync, so no
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
  sign one digest); and the answer is one-stage's for that registration's digest,
  accepted with the profile of one unit drawn `subset:1`: the link-window was proved (N2's verifier of record,
  `LinkWindows.verify`, is the answer's verifier of record);
* in the rows form, link-window `w`: its registration (the scan's, per window) names the scan's Program; its three
  certificates are the requests' record's (the link's ingress), the responses' (its egress) and the fails'
  (`<link>-fails`, egress) at `w`, each the key's and verifying in the session, so a link-window missing any record's
  certificate is refused, and the network certifier certifies no record of a link-window in which it refused a
  connection (traffic past the window's Program slots, `warden`'s "Admission"), so such a link-window is refused; its two registrations, ingress then egress, are what `warden_rows.check_registration` reads:
  the link-window's value, link, end and window, the schedule's digest (a registration under another schedule is
  refused), the schedule's rows, words, word bits, row format and layout, the link-window's domain in the session
  (`warden_rows.domain`), and a certificate that is the key's, signs the head (`CellsHead.digest`) and verifies in the
  session (a registration not under the deployment key is refused); the scan's registration holds both, an entry of its
  `roots` or `registered` naming each one's root and domain, which bind the session, link, end and window (an answer
  for another window's rows is refused); its Programs' row is a compute audit's output (`progTied`): its `scan/prog`
  entry names the root and domain of an output port's entry (`<unit>/o<j>`, j ≥ 1) or a `SlotEntry_v1` template's
  entries port's (`<descriptor id>/c0/o0`, `entriesPortName`) in the `roots` of an audit's registration, so a scan
  whose `prog` the prover committed alone is refused; and the answer is one-stage's for that
  registration's digest, accepted with one level of the scan's 2T units drawn `subset:k` or stratified, and its
  `bound`;
* in the rows form, every accepted link-window's scan is at one δ, `PeriodLeakAccepted`'s;
* audit `i`: its registration names its Program (the descriptor's SHA-512), and the answer is one-stage's for its
  registration's digest, accepted with one level of its Program's units drawn `subset:k` or stratified (a work law's
  draw is stratified), and its `bound`; in the rows form, an audit tied to a link-window's scan (`tiedIn`: some
  link-window's `prog` is its output) is refused unless every unit is drawn (`Σ k = N`), as `K` charges no faulty term
  for it;
* in the rows form, link-window `w`'s scan's Programs' row is a compute audit's slot entries (`wired`): its `scan/prog`
  entry names the root and domain of the entries port `<descriptor id>/c0/o0` in the `roots` of an audit's
  registration, where the descriptor id is a template node's of that audit's own Program calling `SlotEntry_v1`, a
  Definition the check pins by closure digest (`slotEntryPins`: a template under that name with another body is
  refused), with one member a slot of the window (T / `every` · `lanes`), so each due, occupied slot's entry, read bit
  included, is
  the one its Program computed, not the prover's word, which with read 1 and no outputs would pass for a request no
  Program reads (`Properties.NetworkPeriod.RowsProgWired`);
* in the rows form, link-window `w`'s scan's state is the `SCAN_STATE` of an audit its `prog` is wired to
  (`stateTied`): the scan's registration has an entry of its `roots` whose port is `scan/state` or of its `registered`
  whose value is `verity/certifier/scan-state/v1`, every such entry at one root and domain, and so has a `wired`
  audit's registration, every such entry at that root and domain. The scan's ingress unit t reads `sin` as that root's
  row t and registers its `o1` as row t + 1, so each row past row 0 is the state the scan's transition computed from
  the row before it, the prover's word for none; row 0 is the empty state, which no unit reads, and only the Programs
  anchored at bucket 0 read it, whose entries the scan reads for no slot, since its state before bucket 0 has none
  occupied. A slot's Program anchored at bucket a reads row a, the scan's state before a, so it computes its entry on
  the requests the scan read, not on a state the prover chose (`Properties.NetworkPeriod.RowsStateTied`).

**Public**, on acceptance: `K` and its terms, each a ceiling of `log2` of `PeriodLeakAccepted`'s factor:

    {"K": K, "audits": [{"delta", "drawn": k, "faulty_bits", "faulty_sets", "free", "jitter", "units": N,
                         "wrong_units": b}, …],
     "cells": Σ_c ⌈log2 (cap_c + 1)⌉, "eps": Σ_i δ_i, "schedules": ⌈log2 #Σ_sync^m⌉, "windows": m}

with `b` the audit's bound on wrong units at its δ (one-stage's `bound`, `Bound.boundOf`), `faulty_sets` =
`⌈log2 Σ_{j ≤ b} C(N, j)⌉` (every set of at most `b` units, which holds the sets a `subset:k` or a stratified draw leaves
possible; `drawn` is the draw's `k`, or its `Σ k` stratified),
`faulty_bits` = `b · bits`, with `bits` the most one of its Program's units commits (its template's return, `bitsOf`:
the committed values `faultyBits` counts, the outputs a wrong unit computes and other units or the client read),
`jitter` = `⌈log2 Dj^steps⌉`, and `K` the sum of the terms. The schedules term counts every assignment of a sync of
Σ_sync to the `m` link-windows, so it is the same whichever syncs the certifier ran them under, and needs none of them.
`eps` is the probability that `K` fails to bound the period's leak, the union of its audits' `δ`: their exact sum in
lowest terms (`Core.Ratio.toString`, `"0"` with no audits), as PoUS's and remote bandwidth's publics carry theirs.

In the rows form the terms are `PeriodLeakAccepted`'s over its windows, the link-windows' scans, whose statements read
the network certifier's rows. A compute audit reads none, so its own terms are in none (it is an acceptance condition),
but the free inputs of each audit a scan's `prog` is an output of are in that scan's link-window's term: the egress
renders the audit's outputs, so the bits the prover chose of its inputs are the egress's to leak. A wrong unit of such
an audit would leak through the egress too, and no term charges one: the check refuses a tied audit with any unit
undrawn (`Properties.NetworkPeriod.RowsTiedDrawn`), and a tied audit's `faulty_bits` and the audit's δ in `eps` come
with the change that ties a compute audit drawn in part and lifts that refusal:

    {"K": K, "audits": [{"delta", "drawn", "units", "wrong_units"}, …], "cells": c, "eps": m · δ,
     "scans": [terms, …], "schedules": c, "windows": m}

* `schedules` is `cells`, c = Σ_c ⌈log2 (cap_c + 1)⌉, as the statement multiplies `#Ss` and `∏_c (cap_c + 1)`. A
  schedule (`WardenSchedule`) is r and D, fixed, and each connection's admission bucket and lane length. The server
  sets a lane's length, when its stream ends, at most `cap_c` cells, and nothing else of it: the network certifier
  admits a queued connection at the first lane that frees, in the clients' arrival order, so the admissions are a
  function of the lengths, and `#Ss ≤ ∏_c (cap_c + 1)`. The second factor is the fail cell's position (`failCell`). A
  failed lane ends at its fail cell, so in a run the two coincide; `K` charges the product as the statement states it.
* each scan's terms, from the scan's Program and its tied audits: `free` the parameters' `free` of the audits its
  `prog` is an output of, summed (`tiedAudits`, each once; the scan's own is 0, as it registers the rows, its `prog` is
  those audits' output, and bucket 0 starts from the empty state, constants of `IngressScanFirst`: the egress tie
  moves the free term from the scan's `prog` to the tied audit's inputs), `steps` T, `faulty_bits`
  `b · max(b_ing, b_eg)` with `b_ing` an ingress instance's return bits (its verdict u16, state and view) and `b_eg` an
  egress instance's (its u16 and row): each instance's committed values (`faultyBits`), the outputs it computes and
  other units or the client read. `b` is its answer's `bound` (over its draw's strata, `countBound`), 0 when every
  unit is drawn, `drawn` its draw's `Σ k`, and `faulty_sets` every set of at most `b` of its 2T units, which holds the
  sets a stratified draw leaves possible too.
* `eps` is the scans' δ summed, `m · δ`: the δ part of the union over the monitoring period's windows that
  `CommOfPeriodLeakAccepted` states, each window at δ plus its knowledge error and its collision errors.

**Profile**: the link and `m`.

What it doesn't check: the proof's public row against the certificates' digests and the output word, which N2's
verifier of record checks inside the answer (`LinkWindows.verify`, with C-Flock's Lean verifier on the `--zk` proof);
the clock form's link-window registration's domain; in the rows form, the records' certificates' digests against the
rows (each signs its salted record, witness) and the scan's statements against its registration, which one-stage's
verifier of record checks inside the answer, and each connection's cap against its lane (the parameters' `cells`; the
network certifier holds a lane to `egress_cells`); and that the audits' answers are the sampled-proofs component's own,
which the composition's theorem is to bind.
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
/-- A proof unit's public outputs are at most 16 bits (the values it commits can be wider). -/
def outBits : Nat := 16
def unitLevel : String := "unit"
def scheduleTag : String := "verity-network-warden/parameters/v3"
def domainTag : String := "verity-network-warden/cells-domain/v2"
def cellsTag : String := "verity-network-warden/cells/v2"
def rowsFormat : String := "verity-network-warden/registration/v2"
def rowSchema : String := "hm96-sha512/row/v2"
/-- The cell kinds (`PAD`, `DATA`, `END`, `FAIL`, `REFUSED`) and the fail reasons the schedule's digest binds. -/
def cellKinds : List Nat := [0, 1, 32, 33, 34]
def failReasons : List String := ["late", "limit", "upstream", "ingress", "malformed"]
/-- A cell's header: its kind u8, its connection's ordinal u32 and its payload's length u32. -/
def cellHeader : Nat := 9

/-- A compute audit's layout terms, its free inputs it doesn't register and its monitoring period, and what its Program
gives: its SHA-512, its units (its root's template instances), the most bits one of them commits, and its slot entries'
templates (`entries`: each root template node calling a `SlotEntry_v1` Definition, by its descriptor id, members and
closure digest, `slotEntryDigest`). -/
structure Audit where
  free : Nat
  steps : Nat
  program : String
  units : Nat
  bits : Nat
  entries : Array (String × Nat × String)

/-- The network certifier's schedule when it functions as an active warden (`warden.WardenParams`). -/
structure Schedule where
  bucketMs : Nat
  cellBytes : Nat
  delay : Nat
  egressCells : Nat
  every : Nat
  ingressBuckets : Nat
  ingressCells : Nat
  lanes : Nat
  queue : Nat
  slots : Nat
  window : Nat

/-- S = lanes · r, a row's slots. -/
def Schedule.allocation (s : Schedule) : Nat := s.lanes * s.slots

/-- A timestep's Program slots, one a lane (`WardenParams.program_slots`). -/
def Schedule.programSlots (s : Schedule) : Nat := s.lanes

/-- A window's Program slots: T / `every` timesteps of `programSlots` (`WardenParams.window_programs`). -/
def Schedule.windowPrograms (s : Schedule) : Nat := s.window / s.every * s.programSlots

/-- The scan's terms, read off its Program: its monitoring period (`T`, the window's buckets), its own free inputs
(none: it registers the rows, and its `prog` is a compute audit's output, `progTied`, whose free inputs a link-window's
term counts, `tiedAudits`), and the bits one ingress and one egress instance commit (its return: the outputs other
units read or the client receives, and its verdict). -/
structure Scan where
  steps : Nat
  free : Nat
  ingress : Nat
  egress : Nat

/-- Which run the link-windows are: the clock form's, under Σ_sync, or the rows form's, under one schedule (and its
digest), read by the scan whose terms are the last. -/
inductive Form where
  | clock (syncSet : NetworkCertifier.SyncSet)
  | rows (schedule : Schedule) (digest : ByteArray) (scan : Scan)

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

/-- What a link-window shows beside its answer and certificates: the clock form's clock certificate, or the rows form's
registrations. -/
inductive Shown where
  | clock (c : Certificate)
  | rows (rs : Array Rows)

structure Window where
  answer : Core.Answer
  certificates : Array Certificate
  shown : Shown

def Window.clockDigest? (x : Window) : Option ByteArray :=
  match x.shown with
  | .clock c => some c.digest
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
  ["audits", "cells", "first", "jitter", "key", "link", "lms", "program", "session", "sync_set", "windows"]

def rowsParamsKeys : List String :=
  ["audits", "cells", "first", "jitter", "key", "link", "lms", "program", "scan", "schedule", "session", "windows"]

def scheduleOf (j : Json) : V Schedule := do
  exactly j ["bucket_ms", "cell_bytes", "delay", "egress_cells", "every", "ingress_buckets", "ingress_cells", "lanes",
    "queue", "slots", "window"] "schedule"
  let pos : String → V Nat := fun k => do
    let x ← Flock.J.nat j k "schedule"
    unless 1 ≤ x do throw s!"schedule: {k} is a positive integer"
    pure x
  let s : Schedule :=
    { bucketMs := ← pos "bucket_ms", cellBytes := ← pos "cell_bytes", delay := ← pos "delay",
      egressCells := ← pos "egress_cells", every := ← pos "every", ingressBuckets := ← pos "ingress_buckets",
      ingressCells := ← pos "ingress_cells", lanes := ← pos "lanes", queue := ← pos "queue", slots := ← pos "slots",
      window := ← pos "window" }
  unless cellHeader + 1 < s.cellBytes do throw s!"schedule: a cell holds its {cellHeader}-byte header and a byte"
  unless s.cellBytes < 2 ^ 32 ∧ s.allocation < 2 ^ 32 ∧ s.window < 2 ^ 32 do
    throw "schedule: B, S = lanes · slots and T = window are u32s"
  unless s.window % s.every == 0 do throw "schedule: every divides the window: a window is whole timesteps"
  pure s

/-- `WardenParams.digest`: the schedule's canonical JSON, with the cell kinds and the fail reasons, tagged. -/
def Schedule.digest (s : Schedule) : V ByteArray :=
  Identity.identityDigestSha512 scheduleTag (Json.mkObj [
    ("bucket_ms", toJson s.bucketMs), ("cell_bytes", toJson s.cellBytes), ("delay", toJson s.delay),
    ("egress_cells", toJson s.egressCells), ("every", toJson s.every), ("ingress_buckets", toJson s.ingressBuckets),
    ("ingress_cells", toJson s.ingressCells), ("lanes", toJson s.lanes), ("queue", toJson s.queue),
    ("slots", toJson s.slots), ("window", toJson s.window), ("kinds", toJson cellKinds),
    ("reasons", toJson failReasons)])

/-- The scan's templates' parameters in the generator form: bucket 0's (`ingress_scan.IngressScanFirst`, from the empty
state), each later ingress bucket's (`ingress_scan.IngressScanBucket`) and the egress bucket's (`egress.EgressBucket`),
which computes its row and returns it. An ingress bucket's `prog<j>` is lane `j`'s slot entry, one a lane. -/
def progParams (lanes : Nat) : Array String := (Array.range lanes).map (s!"prog{·}")
def firstParams (lanes : Nat) : Array String := #["row"] ++ progParams lanes
def ingressParams (lanes : Nat) : Array String := #["row", "sin"] ++ progParams lanes
def egressParams : Array String := #["state"]

/-- A type's bits: its leaves' widths summed. -/
def bitsOf (t : Flock.Program.Ty) : Nat := t.widths.foldl (· + ·) 0

/-- Parameters' bits: their types' summed. -/
def paramBits (ps : Array (String × Flock.Program.Ty)) : Nat := ps.foldl (fun n p => n + bitsOf p.2) 0

/-- The collections a part of references names: its space and index, or each item's of an explicit part. -/
def partNames : Flock.Program.Part → Array (Bool × Nat)
  | .affine p i _ _ => #[(p, i)]
  | .strided p i _ _ _ _ _ => #[(p, i)]
  | .explicit it => it.map fun r => (r.p, r.i)

/-- Per Definition, whether every leaf it returns is a gate it computes, never a parameter's leaf passed through: each
part of its return names nodes of its body, none a scan, each calling a primitive or a Definition decoded before it of
which the same holds. Once per Definition, at the cost of its return's parts. -/
def ownReturns (P : Flock.Program.Prog) : Array Bool :=
  (List.range P.defs.size).foldl (fun a d =>
    let defn := P.defs[d]!
    a.push (!defn.prim && defn.body.ret.parts.all fun q => (partNames q).all fun (p, k) =>
      !p && match defn.body.nodes[k]? with
        | some n => n.form != .scan && ((P.defs[n.fn]!).prim || (n.fn < d && a[n.fn]!))
        | none => false)) #[]

/-- The root's template nodes, in order: those calling a composite. -/
def templateNodes (P : Flock.Program.Prog) : List Nat :=
  let nodes := (P.defs[P.root]!).body.nodes
  (List.range nodes.size).filter fun k => match P.defs[nodes[k]!.fn]? with
    | some d => !d.prim
    | none => false

/-- The root node a template's argument reads when the argument is that input node's leaves in order from its first
(one affine run), as the generator form binds each parameter to an input array of its own. -/
def inputArg (P : Flock.Program.Prog) (a : Flock.Program.Arg) : Option Nat :=
  match a.seq.parts.toList with
  | [.affine false k 0 _] => match (P.defs[P.root]!).body.nodes[k]? with
    | some n => if NciCheck.Desc.isInputNode P n then some k else none
    | none => none
  | _ => none

/-- Refuses with `why` unless `b`: one step of a `do` block, so what its acceptance gives is read off it step by step
(`Properties.NetworkScan`). -/
def need (b : Bool) (why : String) : V Unit := if b then pure () else throw why

/-- `o`'s value, refusing with `why` where there is none, as one step. -/
def needSome {α : Type} (o : Option α) (why : String) : V α := match o with
  | some a => pure a
  | none => throw why

/-- The generator form's ingress templates, from the templates before the egress template's: bucket 0's, and the later
ingress buckets' when there are later buckets. -/
def ingressTemplates : List (Flock.Program.Node × Flock.Program.Defn) →
    V ((Flock.Program.Node × Flock.Program.Defn) × Option (Flock.Program.Node × Flock.Program.Defn))
  | [f] => pure (f, none)
  | [f, b] => pure (f, some b)
  | _ => throw "scan: not the generator form's templates: bucket 0's, the later ingress buckets' and the egress \
    bucket's"

/-- The slot entry's Definition (`io_shell.SlotEntry_v1`): one unit a due, occupied slot's Program, committing the
entry the scan's ingress reads at that slot (its `o0`). -/
def slotEntryId : String := "SlotEntry_v1"

/-- A Definition's descriptor entry `e` with the descriptor's interning resolved (`callees`, `types`): each node's
callee id (`fn`), output type (`out`) and argument types inline, as `annotations.definition_digest` hashes it, so one
Definition has one entry in any Program. -/
def resolvedEntry (callees types : Array Json) (e : Json) : Json :=
  let set := fun (j : Json) (k : String) (v : Json) => match j with
    | .obj kvs => Json.obj (kvs.insert k v)
    | j => j
  let at_ := fun (xs : Array Json) (i : Json) => match i.getNat? with
    | .ok i => xs.getD i .null
    | .error _ => .null
  match e.getObjValD "form", e.getObjValD "body" with
  | .str "composite", body => match body.getObjValD "nodes" with
    | .arr nodes => set e "body" (set body "nodes" (.arr (nodes.map fun n =>
        let args := match n.getObjValD "args" with
          | .arr xs => Json.arr (xs.map fun a => set a "type" (at_ types (a.getObjValD "type")))
          | x => x
        set (set (set n "fn" (at_ callees (n.getObjValD "fn"))) "out" (at_ types (n.getObjValD "out"))) "args" args)))
    | _ => e
  | _, _ => e

/-- `annotations.definition_digest`: the SHA-512, in hex, of a Definition's canonical `resolvedEntry`. -/
def definitionDigest (callees types : Array Json) (e : Json) : V String := do
  pure (Flock.hexOf (Flock.Sha512.hash (← Flock.canon (resolvedEntry callees types e)).toUTF8))

/-- The closure digest of Definition `key` of the Program whose descriptor is `j`: the SHA-512, in hex, of the
canonical JSON object that maps each Definition `key` reaches (itself, its nodes' callees, theirs, …), by descriptor
id, to its `definitionDigest`. It digests the template's whole body, so another Definition under its id has another
digest. -/
def slotEntryDigest (j : Json) (key : String) : V String := do
  let defs := j.getObjValD "definitions"
  let tab := fun (k : String) => match j.getObjValD k with | .arr xs => xs | _ => #[]
  let (callees, types) := (tab "callees", tab "types")
  let nodesOf := fun (e : Json) => match (e.getObjValD "body").getObjValD "nodes" with | .arr xs => xs | _ => #[]
  let fuel := match defs with
    | .obj kvs => kvs.foldl (fun n _ e => n + (nodesOf e).size) 1
    | _ => 1
  let mut seen : Array (String × Json) := #[]
  let mut todo : List String := [key]
  for _ in [0:fuel] do
    match todo with
    | [] => break
    | k :: rest =>
      todo := rest
      if seen.any (·.1 == k) then continue
      let e := defs.getObjValD k
      if let .null := e then throw s!"program: {key} reaches {k}, no Definition of its descriptor"
      seen := seen.push (k, .str (← definitionDigest callees types e))
      if e.getObjValD "form" == .str "composite" then
        for n in nodesOf e do
          match (n.getObjValD "fn").getNat?.toOption.bind (callees[·]?) with
          | some (.str c) => todo := c :: todo
          | _ => throw s!"program: {k}: a node's callee is outside callees"
  unless todo.isEmpty do throw s!"program: {key}'s callees outrun its descriptor's nodes"
  pure (Flock.hexOf (Flock.Sha512.hash (← Flock.canon (Json.mkObj seen.toList)).toUTF8))

/-- The slot entry's Definitions the check takes, by closure digest (`slotEntryDigest`): `io_shell.SlotEntry_v1` at
its circuit-check rows, `SlotEntry_v1{KO=2,IDB=17,NP=4,STOP=[]}` and `SlotEntry_v1{KO=64,IDB=17,NP=1024,STOP=[128001]}`
(a Llama-3 tokenizer's 17-bit ids and end-of-text id), in that order. A template under `SlotEntry_v1`'s name with
another body has neither digest, so its entries are no slot entries (`wired`); a new binding or body of `SlotEntry_v1`
is pinned here. -/
def slotEntryPins : List String :=
  ["0763102b5fd38583279471486ff8ce51b9caae1bc5dece3d7712ddcec456f1bb\
    4e543343ed1399c0665043402d7100173a270cb6a8cc541b60bcdd29b37d7d0a",
   "662360bfd2b31949f09447a32dec775c3e2e26e2d9132af8898609475e75a753\
    4cb522ce4f794bc5fcce71a7f09ec5a2170e0b2406339cc4d9268ea8487e3361"]

#guard slotEntryPins.all (·.length == 128)

/-- A compute audit's layout terms and its Program, from the Program's descriptor (`Flock.Program.decode` of its
canonical bytes): its units are its root's template instances, each committing its template's return; a root with no
template is refused. Its slot entries' templates are its root's template nodes whose Definition is `SlotEntry_v1`, each
with its closure digest (`slotEntryDigest`). -/
def auditOf (a : Json) (ctx : String) : V Audit := do
  exactly a ["free", "program", "steps"] ctx
  let pj ← Flock.J.get a "program" ctx
  let prog ← match Flock.Program.decode (← Flock.canon pj).toUTF8 with
    | .ok prog => pure prog
    | .error e => throw s!"{ctx}: program: {e}"
  let some root := prog.defs[prog.root]? | throw s!"{ctx}: program: no root"
  let templates := root.body.nodes.filterMap fun x => match prog.defs[x.fn]? with
    | some d => if d.prim then none else some (x.members, bitsOf d.ret)
    | none => none
  if templates.isEmpty then throw s!"{ctx}: program: no template"
  let slots := root.body.nodes.filterMap fun x => match prog.defs[x.fn]? with
    | some d => if !d.prim && d.id == slotEntryId then some (d.key, x.members) else none
    | none => none
  let entries ← slots.mapM fun (k, n) => do
    pure (k, n, ← (slotEntryDigest pj k).mapError (s!"{ctx}: " ++ ·))
  pure { free := ← Flock.J.nat a "free" ctx, steps := ← Flock.J.nat a "steps" ctx, program := Flock.hexOf prog.sha512,
         units := templates.foldl (· + ·.1) 0, bits := templates.foldl (fun b t => max b t.2) 0, entries }

/-- Whether a batch node of ingress buckets `t0, t0 + 1, …` reads the slot entries at their slots: for each lane `j`
and member `m` (bucket `t0 + m`), its argument `o + j` (`prog<j>`) is row `j · nb + ⌊(t0 + m) / every⌋` of input node
`k`, `E` leaves a row, leaf by leaf (`Flock.Program.Arg.memberRef`). That row is lane `j`'s slot of the boundary whose
Programs bucket `t0 + m` reads (`io_shell.SlotEntry_v1`'s row k is slot k, lane-major). -/
def readsEntries (x : Flock.Program.Node) (o t0 lanes nb every E k : Nat) : Bool :=
  (List.range lanes).all fun j => match x.args[o + j]? with
    | some a => (List.range x.n).all fun m => (List.range E).all fun l =>
        a.memberRef m l == { p := false, i := k, j := (j * nb + (t0 + m) / every) * E + l }
    | none => false

/-- A batch of 3 members whose argument 1 reads leaf `rows[m]` of node 7 at member `m`, one leaf a member. -/
def entriesProbe (rows : List Nat) : Flock.Program.Node :=
  let arg := fun (rs : List Nat) => Flock.Program.Arg.mk (.value 1)
    (.ofParts (rs.toArray.map fun r => Flock.Program.Part.affine false 7 r 1)) (some (rs.length, 1))
  Flock.Program.Node.mk .batch 0 #[arg [0, 0, 0], arg rows] (.value 1) #[] 3 0

#guard readsEntries (entriesProbe [0, 1, 1]) 1 1 1 2 2 1 7 && !readsEntries (entriesProbe [0, 1, 0]) 1 1 1 2 2 1 7 &&
  !readsEntries (entriesProbe [0, 1, 1]) 1 1 2 2 2 1 7 && !readsEntries (entriesProbe [0, 1, 1]) 1 1 1 2 2 1 6

/-- The scan's terms from its Program's descriptor (`Flock.Program.decode` of its canonical bytes), which must be the
registered Program (`program`, its SHA-512) and the generator form of the schedule's link-window: its root batches one
instance of bucket 0's template, `T - 1` of the later ingress buckets' (no such template when `T = 1`), both returning
the same and each one's `row` the schedule's row (`S · B` bytes), then `T` of the egress template, which returns a u16
and the schedule's row, and every input gate is a template's parameter. The types are the Boolean profile's, as the
scan's Definitions have them: a byte 8 bits, a u16 16. An egress template with a row among its parameters, the checker
form, is refused first, by name: there the egress row is an input gate, which `NetTiming.Layout.Fits` refuses, and `K`
would count the free inputs of a Program that only compares its egress rows.

Then what `NetTiming.Layout.Fits` asks of the layout the templates give (`Properties.NetworkScan.scanLayout`), read off
the description, at the cost of its nodes and returns: the Program is laid out as C-Flock's decoder lays one out and
within what its resolution walks (`NciCheck.Desc`'s `rootLaidOut`, `inputsAtRoot` and `walkable`); the root returns
every egress leaf (one affine run from the egress node's first leaf over all `T` instances'); the egress template
returns only gates it computes (`ownReturns`: no parameter's leaf passed through, so no egress gate is an input); each
ingress template's row, and every template's row, state and egress argument, is one input node's leaves (`inputArg`),
so each ingress gate is an input and a template reads only input gates outside its own; no egress template's argument
is an ingress row's input node, so no egress gate depends on an ingress gate; and the Program is topological
(`Prog.orderViolation`, as `NciCheck.Desc.evaluate` checks it), so a gate reads only gates before it.

The ingress templates' entry ports `prog<j>` all have one type, a slot entry's, and read one input node of their own,
the slot entries (`lanes · T / every` rows, counted once among the input gates): bucket `t`'s `prog<j>` is its row
`j · T / every + ⌊t / every⌋` (`readsEntries`), the row the period check's `wired` holds to an audit's
`SlotEntry_v1` entries port. -/
def scanOf (j : Json) (program : String) (s : Schedule) : V Scan := do
  let prog ← (Flock.Program.decode (← Flock.canon j).toUTF8).mapError fun e => s!"scan: {e}"
  need (Flock.hexOf prog.sha512 == program) "scan: its SHA-512 is not the registered Program's (program)"
  let root ← needSome prog.defs[prog.root]? "scan: no root"
  let templates := (templateNodes prog).toArray.map fun k => (root.body.nodes[k]!, prog.defs[root.body.nodes[k]!.fn]!)
  let T := s.window
  let row : Flock.Program.Ty := .array (s.allocation * s.cellBytes) (.array 8 (.value 1))
  let (xe, de) ← needSome templates.back? "scan: no template"
  need (!de.params.any (fun p => p.1 == "row" || p.2 == row)) "scan: the egress template takes its row as an input \
    (the checker form): no egress gate is an input (NetTiming.Layout.Fits)"
  let ((xf, df), later) ← ingressTemplates templates.pop.toList
  need (df.params.map (·.1) == firstParams s.lanes && later.all (·.2.params.map (·.1) == ingressParams s.lanes) &&
      de.params.map (·.1) == egressParams)
    s!"scan: the templates' parameters are not {firstParams s.lanes}, {ingressParams s.lanes} and {egressParams}"
  let laterN := match later with
    | some (x, _) => if x.form == .batch && 1 ≤ x.n then some x.n else none
    | none => some 0
  need (xf.form == .batch && xf.n == 1 && laterN == some (T - 1) && xe.form == .batch && xe.n == T)
    s!"scan: not bucket 0, {T - 1} later ingress buckets and {T} egress buckets, the link-window's"
  need (df.params[0]!.2 == row && later.all (·.2.params[0]!.2 == row))
    "scan: an ingress template's row is not the schedule's"
  let ety := df.params[1]!.2
  need ((df.params.toList.drop 1).all (·.2 == ety) &&
      later.all fun (_, d) => (d.params.toList.drop 2).all (·.2 == ety) && d.ret == df.ret)
    "scan: the ingress templates' entry ports' types or returns differ"
  need (de.ret == .tuple #[.array outBits (.value 1), row])
    "scan: the egress template doesn't return a u16 and the schedule's row"
  let inputs := root.body.nodes.foldl (fun n x => n + match prog.defs[x.fn]? with
    | some d => ((Flock.Program.inputWidth? d.key).map (· * x.members)).getD 0
    | none => 0) 0
  let nb := T / s.every
  let laterBits := (later.map fun (_, d) => paramBits (d.params.extract 0 2)).getD 0
  need (inputs == paramBits (df.params.extract 0 1) + (T - 1) * laterBits + T * paramBits de.params +
      s.lanes * nb * bitsOf ety)
    "scan: an input gate that is no template's row, state or egress parameter, nor one of the slot entries"
  need (NciCheck.Desc.rootLaidOut prog && NciCheck.Desc.inputsAtRoot prog && NciCheck.Desc.walkable prog)
    "scan: not laid out as C-Flock's decoder lays a Program out, or past what its resolution walks \
      (NciCheck.Desc.rootLaidOut, inputsAtRoot and walkable)"
  let ke := (templateNodes prog).getLast?.getD 0
  need (root.body.ret.parts.any fun q => match q with
      | .affine false k 0 c => k == ke && T * de.outLeaves ≤ c
      | _ => false)
    "scan: the root doesn't return every egress leaf, so an egress gate would be one the Program doesn't return \
      (NetTiming.Layout.Fits)"
  need ((ownReturns prog)[xe.fn]!) "scan: the egress template returns a leaf it doesn't compute (a parameter's, passed \
    through), so an egress gate would be an input (NetTiming.Layout.Fits)"
  let ingress := #[xf] ++ (later.map (#[·.1])).getD #[]
  let rows := ingress.map fun x => x.args[0]?.bind (inputArg prog)
  need (rows.all (·.isSome)) "scan: an ingress template's row is not one input node's leaves, so an ingress gate \
    would be no input (NetTiming.Layout.Fits)"
  need (xf.args.size == 1 + s.lanes && (later.all fun (x, _) => x.args.size == 2 + s.lanes) &&
      (xf.args.extract 0 1).all (fun a => (inputArg prog a).isSome) &&
      (later.all fun (x, _) => (x.args.extract 0 2).all fun a => (inputArg prog a).isSome) &&
      xe.args.all fun a => (inputArg prog a).isSome)
    "scan: a template's row, state or egress argument is not one input node's leaves"
  need (xe.args.all fun a => !rows.contains (inputArg prog a)) "scan: an egress template's argument is an ingress \
    row's input node, so an egress gate would depend on the ingress (NetTiming.Layout.Fits)"
  let a0 ← needSome xf.args[1]? "scan: bucket 0 has no entry port"
  let pk := (a0.memberRef 0 0).i
  need (!(a0.memberRef 0 0).p && !rows.contains (some pk) && match root.body.nodes[pk]? with
      | some n => NciCheck.Desc.isInputNode prog n
      | none => false)
    "scan: bucket 0's prog0 reads no input node of its own, the slot entries'"
  need (readsEntries xf 1 0 s.lanes nb s.every ety.leaves pk &&
      later.all fun (x, _) => readsEntries x 2 1 s.lanes nb s.every ety.leaves pk)
    s!"scan: bucket t's prog<j> is not row j · {nb} + ⌊t / {s.every}⌋ of the slot entries' input node, lane j's \
      slot of the boundary bucket t reads (NetworkCheck.Period.readsEntries)"
  need prog.orderViolation.isNone "scan: a body names a later node (Flock.Program.Prog.orderViolation): the Program \
    is not topological, so a gate's operands could be gates after it"
  pure { steps := T, free := 0, ingress := bitsOf df.ret, egress := bitsOf de.ret }

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
  let program ← Flock.J.str j "program" "the object"
  unless isDigest program do throw "program: not a SHA-512 digest (128 lowercase hex digits)"
  let form ← if keysOf j == paramsKeys then do pure (Form.clock (← syncSetOf (← Flock.J.get j "sync_set" "the object")))
    else do
      let s ← scheduleOf (← Flock.J.get j "schedule" "the object")
      pure (Form.rows s (← s.digest) (← scanOf (← Flock.J.get j "scan" "the object") program s))
  let windows ← Flock.J.nat j "windows" "the object"
  unless 1 ≤ windows do throw "windows: a period has at least one link-window"
  pure { audits, cells, first, jitter, link, lms, key, session, program, form, windows }

/-- A record of one of `formats` and its digest, the window record's (`Flock.WindowRecord.digestOf` of its format and
canonical JSON, `registration.digest`). -/
def registrationOf (j : Json) (ctx : String) : V Registration := do
  let format ← Flock.J.str j "format" ctx
  unless formats.contains format do throw s!"{ctx}: format is not {" or ".intercalate formats}"
  let body ← Flock.canon j
  pure ⟨j, Flock.WindowRecord.digestOf format body.toUTF8⟩

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

def windowOf (j : Json) (ctx : String) : V Window := do
  let clock := keysOf j == ["answer", "certificates", "clock"]
  unless clock do exactly j ["answer", "certificates", "registrations"] ctx
  let answer ← Core.Answer.ofJson (← Flock.J.get j "answer" ctx) s!"{ctx}.answer"
  let certificates ← (← Flock.J.arr j "certificates" ctx).mapM (certificateOf · s!"{ctx}.certificates")
  let shown ← if clock then do pure (Shown.clock (← certificateOf (← Flock.J.get j "clock" ctx) s!"{ctx}.clock"))
    else do pure (Shown.rows (← (← Flock.J.arr j "registrations" ctx).mapM (rowsOf · s!"{ctx}.registrations")))
  pure { answer, certificates, shown }

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

/-- Link-window `w`'s findings, in the order the module docstring gives, `clear` being `clearClocks` of Σ_sync and
`earlier` the period's earlier link-windows' clock digests. -/
def windowCheck (p : Params) (clear : List ByteArray) (earlier : Array ByteArray) (w : Nat) (reg : Registration)
    (x : Window) : Found Unit := do
  let what := s!"window {w}"
  let .clock clock := x.shown | refuse s!"{what}: registrations, not a clock certificate: the rows form's"
  unless reg.record.getObjValD "program" == .str p.program &&
      canonText (reg.record.getObjValD "window") == canonText (served w) do
    refuse s!"{what}: registration: not the link's Program at window {w}"
  unless x.certificates.size == 2 && (x.certificates.zip #[Direction.egress, .ingress]).all
      (fun (c, d) => certifies p d w c) do
    refuse s!"{what}: certificates"
  unless certifies p .egress w clock do refuse s!"{what}: clock"
  if clear.contains clock.digest then refuse s!"{what}: clock: a sync's clock digest unsalted"
  if earlier.contains clock.digest then refuse s!"{what}: clock: an earlier link-window's digest, its salt reused"
  let (prof, _) ← answered what reg x.answer
  unless prof.levels.map (fun l => (l.name, l.size, l.draw)) == #[(unitLevel, 1, some (Core.Draw.subset 1))] &&
      prof.drawn == #[(unitLevel, 1)] do
    refuse s!"{what}: profile: not the link-window's one unit proved"

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

/-- The scan's Programs' row in its registration: the entry of its `roots` whose port, or of its `registered` whose
value, is `scan/prog`. -/
def progName : String := "scan/prog"

/-- A unit's committed output port, `<unit>/o<j>` with j ≥ 1: an output a unit computes past its public word `o0`. Read
over the port's characters, which the kernel evaluates (`String.splitOn` it doesn't). -/
def outputPort (port : String) : Bool :=
  let cs := port.toList
  cs.contains '/' && match (cs.reverse.takeWhile (· != '/')).reverse with
    | 'o' :: d :: ds => (d :: ds).all Char.isDigit && d != '0'
    | _ => false

/-- A `SlotEntry_v1` template's entry rows, the public word of its first chunk's members (one-stage names a template's
members `<descriptor id>/c<j>`): `<descriptor id>/c0/o0`. -/
def entriesPort (key : String) : String := key ++ "/c0/o0"

/-- Whether a port is some `SlotEntry_v1` template's entries port (`entriesPort`), read over its characters: its `o0`
is the entry the slot's Program computes, so it ties its audit as an output port does. -/
def entriesPortName (port : String) : Bool :=
  let cs := port.toList
  (slotEntryId ++ "{").toList.isPrefixOf cs && "}/c0/o0".toList.isSuffixOf cs

/-- The audits the scan's Programs' row is an output of, by index: those whose registration's `roots` has an output
port's or an entries port's entry (`outputPort`, `entriesPortName`) at the root and domain of the scan's registration's
`scan/prog` entry (`progName`), so the rows its ingress units read as `prog` are the ones that audit's units
computed. -/
def tiedAudits (audits : Array Registration) (reg : Registration) : Array Nat :=
  let entries := fun (j : Json) (k : String) => match j.getObjValD k with | .arr xs => xs | _ => #[]
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str s => s == n | _ => false
  let prog := (entries reg.record "roots").filter (named "port" progName) ++
    (entries reg.record "registered").filter (named "value" progName)
  (Array.range audits.size).filter fun i => prog.any fun e => match e.getObjValD "root", e.getObjValD "domain" with
    | .str root, .str dom => ((audits[i]?.map (entries ·.record "roots")).getD #[]).any fun o =>
        (match o.getObjValD "port" with | .str q => outputPort q || entriesPortName q | _ => false) &&
          named "root" root o &&
          named "domain" dom o
    | _, _ => false

/-- The scan's Programs' row is a compute audit's output: some audit is tied to it (`tiedAudits`). -/
def progTied (audits : Array Registration) (reg : Registration) : Bool := !(tiedAudits audits reg).isEmpty

/-- The scan's Programs' row is a compute audit's slot entries: the root and domain of the scan's registration's
`scan/prog` entry (`progName`) are those of an audit's entries port (`entriesPort`) of a `SlotEntry_v1` template of
that audit's own Program (`Audit.entries`) that is a pinned Definition (its closure digest one of `slotEntryPins`, so
its whole body is `io_shell.SlotEntry_v1`'s), with one member a slot of the window (`Schedule.windowPrograms`), so
bucket `t`'s port `prog<j>`, which reads slot `j · (T / every) + ⌊t / every⌋`'s entry (`scanOf`), reads the entry that
slot's Program computed, read bit included. One-stage binds the rest: a reused value is its port's leaves, one chunk's,
and an audit's roots are the ports its Program's layout derives. -/
def wired (audits : Array Audit) (s : Schedule) (regs : Array Registration) (reg : Registration) : Bool :=
  let entries := fun (j : Json) (k : String) => match j.getObjValD k with | .arr xs => xs | _ => #[]
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  let prog := (entries reg.record "roots").filter (named "port" progName) ++
    (entries reg.record "registered").filter (named "value" progName)
  (audits.zip regs).any fun (a, r) => a.entries.any fun (k, n, d) => n == s.windowPrograms &&
    slotEntryPins.contains d &&
    prog.any fun e => match e.getObjValD "root", e.getObjValD "domain" with
      | .str root, .str dom => (entries r.record "roots").any fun o =>
          named "port" (entriesPort k) o && named "root" root o && named "domain" dom o
      | _, _ => false

/-- The scan's state in a registration: the entry of its `roots` whose port is `scan/state`. -/
def stateName : String := "scan/state"

/-- The value a window circuit registers the scan's state rows as (`ingress_scan`'s `SCAN_STATE`), row a the state
before bucket a: the entry of an audit's `registered`, or of the scan's, whose value it is. -/
def stateValue : String := "verity/certifier/scan-state/v1"

/-- A registration's state entries: those of its `roots` whose port is `stateName` and of its `registered` whose value
is `stateValue`. -/
def stateEntries (reg : Registration) : Array Json :=
  let entries := fun (k : String) => match reg.record.getObjValD k with | .arr xs => xs | _ => #[]
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  (entries "roots").filter (named "port" stateName) ++ (entries "registered").filter (named "value" stateValue)

/-- The scan's state is the state its slots' Programs read: the scan's registration has a state entry (`stateEntries`),
every one at one root and domain, and an audit `wired` to the scan's `prog` has one, every one at that root and domain.
The scan's units read and register their states at that root (`sin` row t, `o1` row t + 1), so the `SCAN_STATE` that
audit's Programs read is the chain of the scan's transitions. -/
def stateTied (audits : Array Audit) (s : Schedule) (regs : Array Registration) (reg : Registration) : Bool :=
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  match (stateEntries reg)[0]? with
  | some e => match e.getObjValD "root", e.getObjValD "domain" with
    | .str root, .str dom =>
      let atRoot := fun (o : Json) => named "root" root o && named "domain" dom o
      (stateEntries reg).all atRoot && (audits.zip regs).any fun (a, r) =>
        wired #[a] s #[r] reg && !(stateEntries r).isEmpty && (stateEntries r).all atRoot
    | _, _ => false
  | none => false

/-- Audit `i` is tied to a link-window's scan: some link-window's registration's `prog` is its output (`tiedAudits`). -/
def tiedIn (audits windows : Array Registration) (i : Nat) : Bool :=
  windows.any fun reg => (tiedAudits audits reg).contains i

/-- The audits the form ties to a link-window's scan: the rows form's `tiedIn`, and none in the clock form. -/
def tiedOf : Form → Commitment → Nat → Bool
  | .clock _, _ => fun _ => false
  | .rows .., c => tiedIn c.audits c.windows

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

/-- Link-window `w`'s findings in the rows form, in the order the module docstring gives, and on acceptance its scan's
facts and its tied audits' free inputs (`tiedAudits`: the parameters' `free` of each audit whose output the scan's
`prog` is, each once). -/
def rowsWindowCheck (p : Params) (s : Schedule) (sd : ByteArray) (audits : Array Registration) (w : Nat)
    (reg : Registration) (x : Window) : Found (Facts × Nat) := do
  let what := s!"window {w}"
  let .rows rs := x.shown | refuse s!"{what}: a clock certificate, not registrations: the clock form's"
  unless reg.record.getObjValD "program" == .str p.program do refuse s!"{what}: registration: not the scan's Program"
  unless x.certificates.size == 3 && (x.certificates.zip (records p.link)).all
      (fun (c, l, d) => certifiesOn p l d w c) do
    refuse s!"{what}: certificates: the requests', responses' and fails' records', each the key's (the network \
      certifier certifies none of a window in which it refused a connection: traffic past the Program slots)"
  unless rs.size == 2 do refuse s!"{what}: registrations: the ingress rows, then the egress rows"
  for (r, d) in rs.zip #[Direction.ingress, .egress] do rowsCheck p s sd w d r
  unless rs.all (holds reg) do refuse s!"{what}: the scan's registration doesn't hold the rows: another window's answer"
  unless progTied audits reg do
    refuse s!"{what}: the scan's prog is no compute audit's output: its {progName} entry's root and domain are no \
      output port's of an audit's registration"
  let (prof, pub) ← answered what reg x.answer
  let f ← factsOf what prof pub
  unless f.units == 2 * s.window do refuse s!"{what}: profile: {f.units} units, not the scan's 2T = {2 * s.window}"
  pure (f, (tiedAudits audits reg).foldl (fun n i => n + ((p.audits[i]?.map (·.free)).getD 0)) 0)

/-- Audit `i`'s findings, and on acceptance its facts: its registration names its Program, the answer's units are the
Program's, and one tied to a link-window's scan (`tied`: `tiedIn`, in the rows form) has every unit drawn, since its
link-window's term charges its free inputs and no faulty term for it. -/
def auditCheck (i : Nat) (a : Audit) (reg : Registration) (x : Core.Answer) (tied : Bool := false) : Found Facts := do
  let what := s!"audit {i}"
  unless reg.record.getObjValD "program" == .str a.program do refuse s!"{what}: registration: not its Program"
  let (prof, pub) ← answered what reg x
  let f ← factsOf what prof pub
  unless f.units == a.units do refuse s!"{what}: profile: {f.units} units, not its Program's {a.units}"
  if tied && f.drawn != f.units then
    refuse s!"{what}: tied to a link-window's scan with {f.drawn} of its {f.units} units drawn: K charges a tied audit \
      no faulty term (its wrong units' outputs are the egress's), so draw all {f.units}"
  pure f

/-- The audits' findings, in order, each tied as the period's form ties it (`tiedOf`). -/
def auditChecks (p : Params) (c : Commitment) (e : Evidence) : Array (Found Facts) :=
  ((p.audits.zip c.audits).zip e.audits).zipIdx.map fun (((a, reg), x), i) => auditCheck i a reg x (tiedOf p.form c i)

/-- Link-window `w`'s wiring finding in the rows form: a refusal unless its scan's Programs' row is a compute audit's
slot entries (`wired`). -/
def wiredCheck (p : Params) (s : Schedule) (audits : Array Registration) (w : Nat) (reg : Registration) :
    Option (Core.Verdict × String) :=
  if wired p.audits s audits reg then none else
    some (.reject, s!"window {w}: the scan's prog is no compute audit's slot entries: its {progName} entry's root and \
      domain are no audit's {slotEntryId} entries port (<descriptor id>/c0/o0) of {s.windowPrograms} members, one a \
      slot, whose template is a pinned {slotEntryId} (its closure digest one of slotEntryPins), so a due, occupied \
      slot's read bit is the prover's word: reuse that port of the pinned template as the scan's prog")

/-- The rows form's wiring findings, one a link-window, in order (`wiredCheck`); none in the clock form. -/
def wiredChecks (p : Params) (c : Commitment) : Array (Option (Core.Verdict × String)) := match p.form with
  | .rows s .. => c.windows.zipIdx.map fun (reg, i) => wiredCheck p s c.audits (p.first + i) reg
  | .clock _ => #[]

/-- Link-window `w`'s state finding in the rows form: a refusal unless its scan's state is the `SCAN_STATE` of an audit
its `prog` is wired to (`stateTied`). With the `prog` wired, each slot's entry is the one its Program computed, and a
Program reads the requests from its own registered `SCAN_STATE`; unless that is the scan's state, it computes its entry
on a state the prover chose. -/
def stateCheck (p : Params) (s : Schedule) (audits : Array Registration) (w : Nat) (reg : Registration) :
    Option (Core.Verdict × String) :=
  if stateTied p.audits s audits reg then none else
    some (.reject, s!"window {w}: the scan's state is no wired audit's SCAN_STATE: its {stateName} entries' root and \
      domain are not every {stateValue} entry's of an audit whose slot entries are its prog, so a slot's Program can \
      compute its entry on a state the prover chose: register the scan's state tree as that audit's SCAN_STATE")

/-- The rows form's state findings, one a link-window, in order (`stateCheck`); none in the clock form. -/
def stateChecks (p : Params) (c : Commitment) : Array (Option (Core.Verdict × String)) := match p.form with
  | .rows s .. => c.windows.zipIdx.map fun (reg, i) => stateCheck p s c.audits (p.first + i) reg
  | .clock _ => #[]

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

/-- A compute audit's terms in the clock form, from its layout terms, its facts and the jitter bound: each of its at
most `b` wrong units commits at most its Program's widest unit's bits. -/
def AuditTerms.of (a : Audit) (f : Facts) (jitter : Nat) : AuditTerms :=
  { facts := f, free := a.free, faultySets := clog2 (atMost f.units f.wrong), faultyBits := f.wrong * a.bits,
    jitter := clog2 (jitter ^ a.steps) }

/-- A link-window's scan's terms in the rows form, from the scan's terms, its facts, its tied audits' free inputs
(`tied`: the egress renders their outputs, so the bits the prover chose of them are the link-window's free inputs) and
the jitter bound: each of its at most `b` wrong units commits at most the wider instance's bits. -/
def Scan.terms (s : Scan) (f : Facts) (tied jitter : Nat) : AuditTerms :=
  { facts := f, free := s.free + tied, faultySets := clog2 (atMost f.units f.wrong),
    faultyBits := f.wrong * max s.ingress s.egress, jitter := clog2 (jitter ^ s.steps) }

structure Public where
  /-- The compute audits' facts. -/
  audits : Array Facts
  cells : Nat
  /-- The terms of `PeriodLeakAccepted`'s windows: the clock form's compute audits', the rows form's scans', one a
  link-window. -/
  units : Array AuditTerms
  /-- The rows form, whose units are its scans. -/
  rows : Bool := false
  schedules : Nat
  windows : Nat

def Public.K (x : Public) : Nat := x.schedules + x.cells + (x.units.map (·.bits)).foldl (· + ·) 0

/-- `Σ_i δ_i` over the units, exactly, in lowest terms: the union bound on `K` failing. -/
def Public.eps (x : Public) : Core.Ratio :=
  let (n, d) := x.units.foldl (fun (n, d) t => (n * t.facts.delta.den + t.facts.delta.num * d, d * t.facts.delta.den))
    (0, 1)
  ⟨n / Nat.gcd n d, d / Nat.gcd n d⟩

/-- `Σ_c ⌈log2 (cap_c + 1)⌉`. -/
def capsTerm (cells : Array Nat) : Nat := (cells.map fun cap => clog2 (cap + 1)).foldl (· + ·) 0

/-- The bound's terms from the audits' layout terms, Programs' bits and facts, the cells' caps, the jitter bound,
`#Σ_sync` and the link-windows. -/
def terms (audits : Array Audit) (facts : Array Facts) (cells : Array Nat) (jitter syncs windows : Nat) : Public :=
  { audits := facts
    cells := capsTerm cells
    units := (audits.zip facts).map fun (a, f) => AuditTerms.of a f jitter
    schedules := clog2 (syncs ^ windows)
    windows }

/-- The rows form's terms: the cells' as the clock form's, the schedules' the same count (each connection's lane
length, one of `cap_c + 1`, fixes its admission and the lanes it frees), and each link-window's scan's from the scan's
terms, its facts and its tied audits' free inputs (`scans`, a pair a link-window). The compute audits' own terms are in
none: a tied audit's free inputs are in its link-window's. -/
def rowsTerms (facts : Array Facts) (scan : Scan) (scans : Array (Facts × Nat)) (cells : Array Nat)
    (jitter windows : Nat) : Public :=
  { audits := facts, cells := capsTerm cells, units := scans.map fun (f, t) => scan.terms f t jitter, rows := true,
    schedules := capsTerm cells, windows }

structure Profile where
  link : String
  windows : Nat

def check (p : Params) (c : Commitment) (e : Evidence) : Core.Outcome Public Profile :=
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
    let ws : Array (Found (Option (Facts × Nat))) := match p.form with
      | .clock ss =>
        let clear := clearClocks ss
        pairs.map fun ((reg, x), i) => (fun () => none) <$>
          windowCheck p clear ((e.windows.extract 0 i).filterMap (·.clockDigest?)) (p.first + i) reg x
      | .rows s sd _ => pairs.map fun ((reg, x), i) => some <$> rowsWindowCheck p s sd c.audits (p.first + i) reg x
    let scans := ws.filterMap (·.toOption.join)
    let oneDelta : Option (Core.Verdict × String) := scans[0]?.bind fun (f0, _) =>
      (scans.find? (·.1.delta != f0.delta)).map fun (f, _) =>
        (.reject, s!"a scan's δ {f.delta.toString} is not the period's, {f0.delta.toString}")
    let as := auditChecks p c e
    match firstFinding (ws.toList.map findingOf ++ [oneDelta] ++ as.toList.map findingOf ++ (wiredChecks p c).toList ++
        (stateChecks p c).toList) with
    | some (.reject, r) => .reject r
    | some (_, r) => .undecided r
    | none =>
      let audits := as.filterMap (·.toOption)
      .accept (match p.form with
          | .clock ss => terms p.audits audits p.cells p.jitter (ss.deltas.length * ss.rhos.length) p.windows
          | .rows _ _ scan => rowsTerms audits scan scans p.cells p.jitter p.windows)
        { link := p.link, windows := p.windows }

def termsJson (ts : Array AuditTerms) : Json :=
  .arr <| ts.map fun t => Json.mkObj [
    ("delta", .str t.facts.delta.toString), ("drawn", toJson t.facts.drawn), ("faulty_bits", toJson t.faultyBits),
    ("faulty_sets", toJson t.faultySets), ("free", toJson t.free), ("jitter", toJson t.jitter),
    ("units", toJson t.facts.units), ("wrong_units", toJson t.facts.wrong)]

def factsJson (fs : Array Facts) : Json :=
  .arr <| fs.map fun f => Json.mkObj [("delta", .str f.delta.toString), ("drawn", toJson f.drawn),
    ("units", toJson f.units), ("wrong_units", toJson f.wrong)]

def publicJson (x : Public) : Json :=
  Json.mkObj ([
    ("K", toJson x.K), ("audits", if x.rows then factsJson x.audits else termsJson x.units), ("cells", toJson x.cells),
    ("eps", .str x.eps.toString), ("schedules", toJson x.schedules), ("windows", toJson x.windows)] ++
    (if x.rows then [("scans", termsJson x.units)] else []))

def profileJson (pr : Profile) : Json :=
  Json.mkObj [("link", .str pr.link), ("windows", toJson pr.windows)]

def protocol : Core.Protocol where
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
  check := check
  publicJson := publicJson
  profileJson := profileJson

#guard clog2 0 == 0 && clog2 1 == 0 && clog2 2 == 1 && clog2 3 == 2 && clog2 4 == 2 && clog2 5 == 3
#guard atMost 10 0 == 1 && atMost 10 1 == 11 && atMost 10 2 == 56 && atMost 4 9 == 16 && atMost 0 3 == 1
-- an audit of 10 units, each committing 16 bits, drawn subset:3 at δ = 1/10 (at most 4 wrong), 7 cells on one
-- connection and none on another, 2 syncs over 3 link-windows, and jitter below 2 over 3 steps
#guard (terms #[⟨0, 3, "", 10, 16, #[]⟩] #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] #[7, 0] 2 2 3).K ==
  3 + 3 + (0 + 9 + 64 + 3)
-- the same audit with a unit committing 40 bits: its 4 wrong units carry 160 bits, not 16 · 4
#guard (terms #[⟨0, 3, "", 10, 40, #[]⟩] #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] #[7, 0] 2 2 3).K ==
  3 + 3 + (0 + 9 + 160 + 3)
-- two audits at δ = 1/100 and 1/10^6: ε = 10001/1000000
#guard (terms #[⟨0, 1, "", 4, 16, #[]⟩, ⟨0, 1, "", 4, 16, #[]⟩] #[{ units := 4, drawn := 1, delta := ⟨1, 100⟩, wrong := 0 },
  { units := 4, drawn := 1, delta := ⟨1, 1000000⟩, wrong := 0 }] #[] 1 1 1).eps.toString == "10001/1000000"
-- eps over audits at δ = 1/10, 1/15 and 1/6 is 1/3; over none it is 0
#guard (terms #[⟨0, 0, "", 1, 16, #[]⟩, ⟨0, 0, "", 1, 16, #[]⟩, ⟨0, 0, "", 1, 16, #[]⟩]
  #[⟨1, 1, ⟨1, 10⟩, 0⟩, ⟨1, 1, ⟨1, 15⟩, 0⟩, ⟨1, 1, ⟨1, 6⟩, 0⟩] #[] 1 1 1).eps.toString == "1/3"
#guard (terms #[] #[] #[] 1 1 1).eps.toString == "0"
-- the rows form over 1 link-window: 7 cells on one connection and 2 on another (3 + 2 bits, for the schedules and
-- again for the fail cells), an audit whose own terms are in none, and one scan of 20 units (T = 10) drawn at
-- δ = 1/15 with at most 3 wrong, 1,120 free inputs, ingress and egress instances committing 300 and 512 bits, and
-- jitter below 2 over its 10 steps: 1351 sets of at most 3 of 20 units
#guard atMost 20 3 == 1351 && clog2 1351 == 11
#guard (rowsTerms #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] ⟨10, 1120, 300, 512⟩
  #[({ units := 20, drawn := 5, delta := ⟨1, 15⟩, wrong := 3 }, 0)] #[7, 2] 2 1).K ==
  (3 + 2) + (3 + 2) + (1120 + 11 + 3 * 512 + 10)
-- every unit drawn: no wrong unit, so no faulty set and no faulty bits
#guard (rowsTerms #[] ⟨4, 0, 300, 512⟩ #[({ units := 8, drawn := 8, delta := ⟨1, 100⟩, wrong := 0 }, 0)] #[] 1 1).K ==
  0
-- w63's class with the egress tie: T = 32, the scan's own free inputs none, every unit drawn, and its tied audit (the
-- slot entries' `SlotEntry_v1` template) of 8 units, a slot at each of the 2 boundaries, whose free inputs are their
-- entries, 1,136 bits each: K is the tied audit's 9,088
#guard (rowsTerms #[{ units := 8, drawn := 8, delta := ⟨1, 100⟩, wrong := 0 }] ⟨32, 0, 300, 512⟩
  #[({ units := 64, drawn := 64, delta := ⟨1, 100⟩, wrong := 0 }, 8 * 1136)] #[] 1 1).K == 9088
-- eps over the scans' δ = 1/15 and 1/15 is 2/15; the audit's 1/10 is in it no more
#guard (rowsTerms #[⟨1, 1, ⟨1, 10⟩, 0⟩] ⟨0, 0, 0, 0⟩ #[(⟨1, 1, ⟨1, 15⟩, 0⟩, 0), (⟨1, 1, ⟨1, 15⟩, 0⟩, 0)] #[] 1 2
  ).eps.toString == "2/15"
-- a unit's output port past o0, its index in ASCII digits without a leading zero; never o0, an input, or no unit's
#guard outputPort "block-3/o1" && outputPort "a/b/o10" && outputPort "/o2"
#guard !(outputPort "o1" || outputPort "u/o0" || outputPort "u/o01" || outputPort "u/o" || outputPort "u/O1" ||
  outputPort "u/o1x" || outputPort "u/o٣" || outputPort "u/prog" || outputPort "")
-- a SlotEntry_v1 template's entries port, its first chunk's o0; never another Definition's, chunk's or port's
#guard entriesPortName (entriesPort "SlotEntry_v1{IDB=8,KO=2,NP=4,STOP=1}") && entriesPortName "SlotEntry_v1{}/c0/o0"
#guard !(entriesPortName "SlotEntry_v2{KO=2}/c0/o0" || entriesPortName "SlotEntry_v1{KO=2}/c1/o0" ||
  entriesPortName "SlotEntry_v1{KO=2}/c0/o1" || entriesPortName "SlotEntry_v1/c0/o0" || entriesPortName "")

end NetworkCheck.Period
