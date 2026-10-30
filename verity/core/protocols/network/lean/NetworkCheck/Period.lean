import NetworkCheck.Check
import CoreProtocol.Profile
import Flock.Program
import Flock.Public
import NciCheck.Desc
import Flock.WindowRecord
import SampledProofs.Wires

/-!
# The network certifier over a period, a core protocol: `verity/certifier/v0`

Onsite's communication component (`Core.Onsite.components`). A period is one node's link over `m` link-windows beside
the period's compute audits (sampled proofs' one-stage, one answer per registered window of the workload). Its
link-windows come in one of two forms, which the parameters name:

* **the clock form**: N2's link-windows (`service.LinkWindows`, Selection `All`), each proved, both its ends certified
  and its clock certified;
* **the rows form**: the network certifier's own run, where it functions as an active warden (`warden_serve`): each
  execution phase's cells committed as rows (`warden_rows`), one registration a direction under one fixed schedule, its
  three records certified, and the scan's one-stage answer for the window (`scan_zk --warden-run`).

It accepts when every execution phase passes its form's check and every compute audit accepted; it then publishes `K`, the
bits a covert sender in the server can move through the network certifier in the period, as
`NetTiming.Guarantees.PeriodLeakAccepted` counts them. In the rows form each link-window's scan's Programs' row must be a
compute audit's slot entries (`wired`, below), and the state that audit's Programs read (its `SCAN_STATE`) the scan's
state, by root and domain (`stateTied`), so a slot's entry is the one its Program computed on the requests the scan
read.

**Params**, JSON with exactly these keys (and each nested object exactly its own), in the clock form

    {"audits": [{"circuit"?: record, "free"?: f, "pins"?: {param: root, …}, "program": descriptor, "steps": k}, …],
     "cells": [cap, …], "first": w₀, "jitter": Dj, "key": hex(I ‖ T[1]), "link": name,
     "lms": {"H": h, "N": n, "W": w, "toy": bool}, "program": hex, "session": hex(session),
     "sync_set": {"deltas": [δ, …], "rhos": [[num, den], …]}, "windows": m}

each compute audit's monitoring period (`steps`, `k`) and its Program, the descriptor (`verity-ir/descriptor/v1`) of the
window's population program, which must decode (`Flock.Program.decode`) with a template in its root: the audit's proof
blocks are the root's template instances, and each commits its template's return, the outputs it computes that other
proof blocks or the client read, which can be wider than a proof unit's 16-bit public output; optionally its window's
circuit record (`circuit.json`, `verity/window-circuit/v1`, naming the Program), whose SHA-512 the audit's registration
binds (its window's `circuit`), and the verifier's pins of its registered parameters (`pins`, each root a SHA-512
digest, fixed before the run). The record's `entries` (`interface.Entries`, `entriesOf`) are what the integration
registers for the scan: the templates whose `o0` is a slot's entry, each by descriptor id and closure digest
(`closureDigest`), and the vocabulary size and stop ids each binds (its `V` and its `STOP`, where it binds them), so the
check names no model, template or binding of its own. The audit's free inputs, the ones the prover chose,
`#(freeInputs \ reg)`, are not a parameter: the check counts them from its registered circuit (`Audit.free`,
`interface.input_count`), its Program's input bits less those the record's wires and ingress fill and those its
registered reads, table reads and openings read of a parameter whose root in the registration (`registered_roots`) is
the pin (`inputsOf`), where the registration binds the record, and all of them where it binds none or the record states
no wires; an audit's `free`, a natural, is read as nothing, so a caller's count, right or wrong, changes no verdict and
no bound. Then each network certifier connection's cell cap over the period
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
header, B, S = `lanes` · `slots` and T = `window` u32s, and `every`, a slot stride's buckets, dividing T, so each
link-window is T / `every` timesteps of `lanes` Program slots, `Schedule.programSlots` and `Schedule.windowPrograms`),
whose digest `WardenParams.digest` every registration binds; its `program` is the scan's registered Program, the
descriptor's SHA-512 (`partition_object.program_sha512`), and its `link` the run's (`warden_serve.LINK`), whose fails record is certified under the link `<link>-fails`. The
descriptor must decode (`Flock.Program.decode`) and be the scan of the schedule's link-window in the generator form:
its root batches one instance of a template whose parameters are `row` and a `prog<j>` a lane (`IngressScanFirst`,
bucket 0), T − 1 of one whose are `row`, `sin` and the `prog<j>` (`IngressScanBucket`; none when T = 1), lane j's
`prog<j>` rows of an input node of its own (`scanOf`), each `row` the schedule's (S · B
bytes) and both returning the same, then T of one whose parameter is `state` and which returns a u16 and the
schedule's row (`EgressBucket`), in the Boolean profile's types (a byte 8 bits, a u16 16), and its every input gate is
a template's parameter. An egress template that takes its row as an input, the checker form, is refused by name: no
egress gate is an input (`NetTiming.BoundaryGates.Fits`), so `K` counts only what a Program that computes its egress rows
leaves free. The rest of what `NetTiming.BoundaryGates.Fits` asks is read off the description (`scanOf`): the Program laid out
as the decoder lays one out and within its resolution's walk (`NciCheck.Desc`), every egress leaf returned by the root
and computed by the egress template, and every template's argument one input node's leaves, no egress template's an
ingress row's. In the rows form a compute audit's free inputs are in the term of each link-window whose scan's `prog`
is its output (`tiedAudits`), and its other layout terms and its units' bits are in no term; their Programs are checked
as in the clock form.

**Commitment**, canonical JSON `{"audits": [registration, …], "windows": [registration, …]}`: one-stage's registration
record of each compute audit and of each execution phase, in order, each of format `verity/registration/v1` or `v2`. Its
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
registrations (`warden_rows.registration`, `cells.jsonl`'s); an execution phase of the other form than the parameters'
is refused.

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
  refused), the schedule's rows, words, word bits, row format and layout, the execution phase's domain in the session
  (`warden_rows.domain`), and a certificate that is the key's, signs the head (`CellsHead.digest`) and verifies in the
  session (a registration not under the deployment key is refused); the scan's registration holds both, an entry of its
  `roots` or `registered` naming each one's root and domain, which bind the session, link, end and window (an answer
  for another window's rows is refused); its Programs' rows are a compute audit's output (`progTied`): a lane's
  `scan/prog<j>` entry names the root and domain of an output port's entry (`<unit>/o<j>`, j ≥ 1) or a template's
  entries port's (`<descriptor id>/c<j>/o0`, `entriesPortName`) in the `roots` of an audit's registration, so a scan
  whose `prog` the prover committed alone is refused; and the answer is one-stage's for that registration's digest,
  accepted with one level of the scan's 2T units drawn `subset:k` or stratified, and its `bound`;
* in the rows form, every accepted link-window's scan is at one δ, `PeriodLeakAccepted`'s;
* audit `i`: its registration names its Program (the descriptor's SHA-512), and the answer is one-stage's for its
  registration's digest, accepted with one level of its Program's units drawn `subset:k` or stratified (a work law's
  draw is stratified), and its `bound`, at any sample: this is the consistency check (each audit's answer is its
  registration's, and in the rows form its outputs are the scan's `prog` and its `SCAN_STATE` the scan's state, below),
  and what a sample in part leaves unproved is in `K` (in the rows form, a tied audit's `faulty_entries` and δ, below);
* in the rows form, link-window `w`'s scan's Programs' rows are a compute audit's slot entries, a lane a chunk
  (`wired`): for each lane `j`, its `scan/prog<j>` entry names the root and domain of the entries port
  `<descriptor id>/c<j>/o0`, of T / `every` leaves, in the `roots` of one audit's registration, where the descriptor id
  is a template node's of that audit's own Program that the audit's circuit record registers as an entry template, at
  the closure digest the record registers for it (a template under that id with another body, or calling other
  Definitions, is refused), a record the audit's registration binds (`Audit.boundIn`: what it registers was fixed
  before any coin), with one member a slot of the window (T / `every` · `lanes`). One-stage cuts a template into
  members of consecutive units, in order (`<descriptor id>/c<j>`), and an entry template whose units read their lane's
  occupancy a wire a lane into a member a lane; so chunk `j` is lane `j`'s slots, its leaf `b` slot
  `j · T / every + b`'s entry, the row the scan's lane `j` reads at boundary `b` (`scanOf`), and each due, occupied
  slot's entry, read bit included, is the one its Program computed, not the prover's word, which with read 1 and no
  outputs would pass for a request no Program reads (`Properties.NetworkPeriod.RowsProgWired`);
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

with `free` the audit's free inputs under its registration (`Audit.free`), `b` the audit's bound on faulty proof
blocks at its δ (one-stage's `bound`, `Bound.boundOf`), `faulty_sets` =
`⌈log2 Σ_{j ≤ b} C(N, j)⌉` (every set of at most `b` proof blocks, which holds the sets a `subset:k` or a stratified
sample leaves possible; `drawn` is the sample's `k`, or its `Σ k` stratified),
`faulty_bits` = `b · bits`, with `bits` the most one of its Program's proof blocks commits (its template's return,
`bitsOf`: the committed values `faultyBits` counts, the outputs a faulty proof block computes and other proof blocks or
the client read), `jitter` = `⌈log2 Dj^steps⌉`, and `K` the sum of the terms. The schedules term counts every assignment
of a sync of Σ_sync to the `m` execution phases, so it is the same whichever syncs the certifier ran them under, and
needs none of them. `eps` is the probability that `K` fails to bound the period's leak, the union of its audits' `δ`:
their exact sum in lowest terms (`Core.Ratio.toString`, `"0"` with no audits), as PoUS's and remote bandwidth's publics
carry theirs.

In the rows form the terms are `PeriodLeakAccepted`'s over its windows, the link-windows' scans, whose statements read
the network certifier's rows. A compute audit reads none, so its own terms are in none (it is an acceptance condition),
but the free inputs of each audit a scan's `prog` is an output of are in that scan's link-window's term: the egress
renders the audit's outputs, so the bits the prover chose of its inputs are the egress's to leak. An incorrect proof
block of such an audit reaches the scan only as the slot entries the scan reads: the scan reads the audit only through
its `prog`, one input node of `E_w` entry bits a link-window (`wired`, `scanOf`), and its state is the scan's own
(`stateTied`). So an incorrect proof block of any template, an entry template's setting its slot's entry, or one
setting the occupancy and prompts its slots' Programs read, or any other its entries read, changes at most those `E_w`
bits; the scan reads an entry only where its own state has the slot due and occupied,
and refuses one there whose `read` is 0, so no `occ` the audit computes drops a request. The check accepts a tied audit
at any sample; sampled in part, its at most `b` incorrect proof blocks can set any of its link-window's entries, so that
link-window's term charges every entry bit (`faulty_entries`: `β_w ≤ 1[b ≥ 1] · E_w`) when `b ≥ 1`, and its δ is in
`eps`:

    {"K": K, "audits": [{"delta", "drawn", "units", "wrong_units"}, …], "cells": c, "eps": m · δ + Σ δ_i,
     "scans": [terms, …], "schedules": c, "windows": m}

* `schedules` is `cells`, c = Σ_c ⌈log2 (cap_c + 1)⌉, as the statement multiplies `#Ss` and `∏_c (cap_c + 1)`. A
  schedule (`WardenSchedule`) is r and D, fixed, and each connection's admission bucket and lane length. The server
  sets a lane's length, when its stream ends, at most `cap_c` cells, and nothing else of it: the network certifier
  admits a queued connection at the first lane that frees, in the clients' arrival order, so the admissions are a
  function of the lengths, and `#Ss ≤ ∏_c (cap_c + 1)`. The second factor is the fail cell's position (`failCell`). A
  failed lane ends at its fail cell, so in a run the two coincide; `K` charges the product as the statement states it.
* each scan's terms, from the scan's Program and its tied audits: `free` the free inputs of the audits its `prog` is an
  output of, each under its registration (`Audit.free`), summed (`tiedFree`, each once; the scan's own is 0, as it
  registers the rows, its `prog` is
  those audits' output, and bucket 0 starts from the empty state, constants of `IngressScanFirst`: the egress tie
  moves the free term from the scan's `prog` to the tied audit's inputs), `steps` T, `faulty_bits`
  `b · max(b_ing, b_eg)` with `b_ing` an ingress instance's return bits (its verdict u16, state and view) and `b_eg` an
  egress instance's (its u16 and row): each instance's committed values (`faultyBits`), the outputs it computes and
  other proof blocks or the client read. `b` is its answer's `bound` (over its sample's strata, `countBound`), 0 when every
  unit is drawn, `drawn` its draw's `Σ k`, and `faulty_sets` every set of at most `b` of its 2T units, which holds the
  sets a stratified draw leaves possible too. `faulty_entries` is `E_w`, its slot entries' bits (`Scan.entries`:
  `lanes · T / every` entries of its `prog` ports' type, 8 · 112 = 896 at w63k4), when an audit its `prog` is tied to
  is drawn in part with a `bound` of at least one wrong unit (`faultyEntries`), else 0.
* `eps` is the scans' δ summed, `m · δ`, and each tied audit's drawn in part (`chargedOf`), whose bound
  `faulty_entries` reads: the δ part of the union over the monitoring period's windows that
  `CommOfPeriodLeakAccepted` states, each window at δ plus its knowledge error and its collision errors.

**Profile**: the link and `m`.

What it doesn't check: the proof's public row against the certificates' digests and the output word, which N2's
verifier of record checks inside the answer (`LinkWindows.verify`, with C-Flock's Lean verifier on the `--zk` proof);
the clock form's link-window registration's domain; in the rows form, the records' certificates' digests against the
rows (each signs its salted record, witness) and the scan's statements against its registration (lane `j`'s slot
entries' input node the value it registers as `scan/prog<j>`), which one-stage's verifier of record checks inside the
answer, and each connection's cap against its lane (the parameters' `cells`; the network certifier holds a lane to
`egress_cells`); that a tied audit's entry templates' units read their lanes' occupancy from its `SCAN_STATE` (the check
reads an audit's root templates' ids and closure digests, and its circuit record's wires and declarations only as counts
of the rows they fill, `inputsOf`), and that an entry template computes the entry as the scan reads it, which is the
registered circuit's, as every guarantee is about the registered circuit; and that the audits' answers are the
sampled-proofs component's own, which the composition's theorem is to bind.

**The clock openings**, apart from the period's check: `clocksFn`, `network-records verdicts`' `clocks`, reads the
syncs the certifier signed for a link through its salted clock certificates' openings, each sync and salt against the
digest its certificate signs (`certificate.SignedClocks`, `signedSyncs`). The certifier chain's record takes each
window's sync from it, with `certificate.SignedClocks` beside it as the reference (`lean_audit.signed_clocks`).
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

/-- What a compute audit's circuit record registers of its slot entries (its `entries`, `interface.Entries`), fixed
before any coin with the record, whose SHA-512 the audit's registration binds: its entry templates, each by descriptor
id with its closure digest (`closureDigest`), whose `o0` is a slot's entry; and the vocabulary size and the stop ids
each of them binds (its `V` and its `STOP`, where it binds them). An audit without a circuit record registers none. -/
structure Entries where
  templates : Array (String × String) := #[]
  vocabulary : Nat := 0
  stop : Array Nat := #[]

/-- What a compute audit's free inputs are counted from (`Audit.free`, `interface.input_count`): its Program's input
bits (`total`, every unit's parameters'); from its circuit record's statements, the bits its wires and its ingress fill
(`declared`) and the bits each registered read, table read and opening of an input row reads, by registered parameter
(`reads`); and the verifier's pins of the registered parameters (`pins`, `(param, root)`), fixed before the run. A
record that states no wires (no record, or one that lists its members) declares and reads none. -/
structure Inputs where
  total : Nat := 0
  declared : Nat := 0
  reads : Array (String × Nat) := #[]
  pins : Array (String × String) := #[]

/-- A compute audit's inputs (`inputs`, whose free bits `Audit.free` counts) and monitoring period, and what its
Program gives: its SHA-512, its units (its root's template instances), the most bits one of them commits, and its slot
entries' templates (`entries`: each root template node whose descriptor id its circuit record registers as an entry
template, by its descriptor id, members and closure digest, `closureDigest`); then its circuit record's SHA-512
(`circuitDigest`, empty with no record) and what the record registers (`registered`). -/
structure Audit where
  inputs : Inputs
  steps : Nat
  program : String
  units : Nat
  bits : Nat
  entries : Array (String × Nat × String)
  circuit : String
  registered : Entries

/-- The network certifier's schedule when it functions as an active network certifier (`warden.WardenParams`). -/
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

/-- A slot stride's Program slots, one a lane (`WardenParams.program_slots`). -/
def Schedule.programSlots (s : Schedule) : Nat := s.lanes

/-- A window's Program slots: T / `every` slot strides of `programSlots` (`WardenParams.window_programs`). -/
def Schedule.windowPrograms (s : Schedule) : Nat := s.window / s.every * s.programSlots

/-- The scan's terms, read off its Program: its monitoring period (`T`, the window's buckets), its own free inputs
(none: it registers the rows, and its `prog` is a compute audit's output, `progTied`, whose free inputs a link-window's
term counts, `tiedAudits`), the bits one ingress and one egress instance commit (its return: the outputs other
units read or the client receives, and its verdict), and its slot entries' bits (`E_w`: `lanes · T / every` entries of
its `prog` ports' type), all a tied audit's wrong units can set. -/
structure Scan where
  steps : Nat
  free : Nat
  ingress : Nat
  egress : Nat
  entries : Nat

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

/-- The one-stage window records of the compute audits and of the link-windows, in order. -/
structure Commitment where
  audits : Array Json
  windows : Array Json
  bytes : ByteArray

structure Layout where
  buckets : Nat
  slots : Nat
  connectionSlots : Nat
  firstBucket : Nat
  deriving BEq

/-- An execution phase's rows in one direction, as `warden_rows.registration` writes them. -/
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

/-- What an accepted audit establishes: its proof blocks, its sample, its δ, and its bound on faulty proof blocks. -/
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

/-- A Definition's descriptor entry `e` with the descriptor's interning resolved (`callees`, `types`): each node's
callee id (`fn`), output type (`out`) and argument types inline, as `metadata.definition_digest` hashes it, so one
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

/-- `metadata.definition_digest`: the SHA-512, in hex, of a Definition's canonical `resolvedEntry`. -/
def definitionDigest (callees types : Array Json) (e : Json) : V String := do
  pure (Flock.hexOf (Flock.Sha512.hash (← Flock.canon (resolvedEntry callees types e)).toUTF8))

/-- The closure digest of Definition `key` of the Program whose descriptor is `j` (`metadata.closure_digest`): the
SHA-512, in hex, of the canonical JSON object that maps each Definition `key` reaches (itself, its nodes' callees,
theirs, …), by descriptor id, to its `definitionDigest`. It digests the template's whole body, so another Definition
under its id has another digest. -/
def closureDigest (j : Json) (key : String) : V String := do
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

/-- The format of a window's circuit record, `circuit.json` (`interface.CIRCUIT`). -/
def circuitFormat : String := "verity/window-circuit/v1"

/-- A circuit record's SHA-512, in hex, as its registration binds it (the window's `circuit`,
`interface.circuit_bytes`): of its canonical JSON and a newline. -/
def circuitDigest (r : Json) : V String := do
  pure (Flock.hexOf (Flock.Sha512.hash ((← Flock.canon r) ++ "\n").toUTF8))

/-- A list of natural numbers, or none. -/
def natsOf (j : Json) : Option (Array Nat) := match j with
  | .arr xs => xs.mapM fun x => x.getNat?.toOption
  | _ => none

/-- What the circuit record `r` registers of its slot entries (its `entries`, `interface.Entries`), checked against the
Program whose descriptor is `pj`, whose SHA-512 is `program` and whose root's templates are `keys`: the record is
`circuitFormat`'s and names that Program; without `entries` it registers none; else its `templates`, each a descriptor
id and a closure digest, are at least one and root templates of the Program, each once; its vocabulary size is at
least 1; and each entry template binds `STOP`, where it binds it, to the registered stop ids and `V`, where it binds
it, to the registered vocabulary size. -/
def entriesOf (r pj : Json) (program : String) (keys : Array String) (ctx : String) : V Entries := do
  let ctx := s!"{ctx}: circuit"
  unless r.getObjValD "format" == .str circuitFormat do throw s!"{ctx}: not a {circuitFormat} record"
  unless r.getObjValD "program" == .str program do throw s!"{ctx}: names another Program than the audit's"
  let e := r.getObjValD "entries"
  if e == .null then return {}
  exactly e ["stop", "templates", "vocabulary"] s!"{ctx}.entries"
  let templates ← (← Flock.J.arr e "templates" s!"{ctx}.entries").mapM fun t => match t with
    | .arr #[.str k, .str d] =>
      if isDigest d then pure (k, d) else throw s!"{ctx}.entries: {k}'s closure digest is not a SHA-512 digest"
    | _ => throw s!"{ctx}.entries: a template is [descriptor id, closure digest]"
  let vocabulary ← Flock.J.nat e "vocabulary" s!"{ctx}.entries"
  let some stop := natsOf (e.getObjValD "stop") | throw s!"{ctx}.entries: stop is not a list of ids"
  let ids := templates.map (·.1)
  unless 0 < ids.size && 1 ≤ vocabulary && decide ids.toList.Nodup && ids.all keys.contains do
    throw s!"{ctx}.entries: its templates, at least one, are root templates of the Program, each once, and its \
      vocabulary is at least 1"
  for k in ids do
    let b := ((pj.getObjValD "definitions").getObjValD k).getObjValD "bindings"
    unless (b.getObjValD "STOP" == .null || natsOf (b.getObjValD "STOP") == some stop) &&
        (b.getObjValD "V" == .null || (b.getObjValD "V").getNat?.toOption == some vocabulary) do
      throw s!"{ctx}.entries: {k} binds other stop ids (STOP) than the registered {stop} or another vocabulary size \
        (V) than the registered {vocabulary}"
  pure { templates, vocabulary, stop }

/-! ## A compute audit's free inputs

`#(freeInputs \ reg)`, the input bits only the prover chose, counted from the audit's registered circuit as
`interface.input_count` counts them, at the size of the description (its templates, wires and declarations, never its
units): its Program's input bits, less those its circuit record's wires fill (one-stage's wire check,
`SampledProofs.Wires.check`, and its `wiredBits`), its ingress fills, and its registered reads, table reads and openings
of an input row read of a held parameter (`Audit.free`), each declaration refused unless it names its template's ports,
units and its parameter's or table's leaves, and no input row is filled twice. -/

open SampledProofs

/-- A return's fields, each an output row `o<j>` of its template (`interface.layout`): a tuple's or a record's, else the
return itself, as is a tuple or a record of none. -/
def retFields (t : Flock.Program.Ty) : Array Flock.Program.Ty := match t with
  | .tuple fs => if fs.isEmpty then #[t] else fs
  | .record fs => if fs.isEmpty then #[t] else fs.map (·.2)
  | _ => #[t]

/-- A parameter's outer length, which a cut divides: its outermost array's, 1 when it is not an array. -/
def outerOf : Flock.Program.Ty → Nat
  | .array n _ => n
  | _ => 1

/-- The root's templates in order of their first instance, each with its units, the members of its nodes summed (a
wire counts a template's units together). -/
def rootTemplates (P : Flock.Program.Prog) : Array (Flock.Program.Defn × Nat) :=
  ((P.defs[P.root]?.map (·.body.nodes)).getD #[]).foldl (fun acc x => match P.defs[x.fn]? with
    | some d => if d.prim then acc else match acc.findIdx? (·.1.key == d.key) with
      | some i => acc.modify i fun (e, n) => (e, n + x.members)
      | none => acc.push (d, x.members)
    | none => acc) #[]

/-- The templates' input bits: every unit's parameters'. -/
def inputBits (ts : Array (Flock.Program.Defn × Nat)) : Nat := ts.foldl (fun n (d, k) => n + k * paramBits d.params) 0

/-- Template `d`'s rows (`interface.layout`), of `units` units: each parameter a row of its u16 words, at least one, with
its outer length, and each return field (`retFields`) an output row `o<j>` of its bits in u16 words, rounded up; refused
where a parameter is not whole u16 words or is named like an output row. -/
def layoutOf (d : Flock.Program.Defn) (units : Nat) : V Wires.Template := do
  let outs := (retFields d.ret).toList.zipIdx.map fun (t, j) => (s!"o{j}", (bitsOf t + 15) / 16)
  let rows ← d.params.toList.mapM fun (p, t) =>
    if bitsOf t % 16 == 0 && 16 ≤ bitsOf t then
      pure (p, ({ words := bitsOf t / 16, outer := outerOf t } : Wires.Port))
    else throw s!"{d.key}: its parameter {p} is {bitsOf t} bits, not whole u16 words"
  unless decide (rows.map (·.1) ++ outs.map (·.1)).Nodup do
    throw s!"{d.key}: a parameter is named like an output row"
  pure { units, rows, outs }

/-- What one declaration fills of template `tid`'s port `port`: units `[lo, hi)`, rows `[r0, r1)` of the port's cut. -/
structure Rect where
  tid : String
  port : String
  lo : Nat
  hi : Nat
  r0 : Nat
  r1 : Nat

/-- Two rectangles of one port that share a row of a unit. -/
def Rect.meets (a b : Rect) : Bool :=
  a.tid == b.tid && a.port == b.port && a.lo < b.hi && b.lo < a.hi && a.r0 < b.r1 && b.r0 < a.r1

/-- No two of the rectangles meet: no input row is filled twice. Pair by pair, per declaration. -/
def apart : List Rect → Bool
  | [] => true
  | r :: rs => !rs.any r.meets && apart rs

/-- A wire's or a read's reader rows. -/
def wireRect (w : Wires.Wire) : Rect := ⟨w.reader, w.port, w.start, w.start + w.readerUnits, w.cut0, w.cut0 + w.cuts⟩

/-- The keys of a registered read (`interface.Read`), beside its optional cut. -/
def readKeys : List String := ["reader", "port", "param", "at", "radix", "base", "step"]

/-- A registered read (`interface.Read`): a wire's pattern from the registered parameter `param` (its `writer`, its `out`
none), exactly the read's fields and optionally its cut. -/
def readOf (j : Json) : V Wires.Wire := do
  let ks := (Wires.keysOf j).getD []
  if ks.contains "cut" then Wires.sameKeys j ("cut" :: readKeys) "a read" else Wires.sameKeys j readKeys "a read"
  let kvs ← Wires.fields j
  Wires.Wire.ofJson (Json.mkObj (("out", .str "") :: kvs.map fun (k, v) => (if k == "param" then "writer" else k, v)))

/-- A natural field of a declaration. -/
private def natIn (j : Json) (k what : String) : V Nat := (Wires.natAt j k).mapError (s!"{what}: " ++ ·)

/-- A string field of a declaration. -/
private def strIn (j : Json) (k what : String) : V String := do Wires.strOf s!"{what}'s {k}" (← Wires.field j k)

/-- A record's list `k`, none when it has no `k`. -/
private def itemsOf (j : Json) (k : String) : V (List Json) :=
  match j.getObjVal? k with
  | .ok v => Wires.arrOf k v
  | .error _ => pure []

/-- The inputs of the audit whose root's templates are `ts` (`rootTemplates`), from its circuit record `r` and the
verifier's pins `pins` (`interface.input_count`). A record that states no wires (one that lists its members) declares
none, so all its inputs count free. Else its templates are the root's, each with its units, in order; its cuts and wires
pass one-stage's wire check (`SampledProofs.Wires.check`); each registered parameter has at least one leaf; each table
(`interface.Table`) is a new name, rows of a registered parameter's leaves; each read (`interface.Read`) names a
parameter or an output row of its template, its rows among the port's cut, units among its template's and leaves among
its parameter's; each table read (`interface.TableRead`) a 32-word input row, units among its template's, and a
declared table's root or domain id; each opening (`interface.Opening`) units among its template's, its key and its
ports input rows of it, a declared table and its verdict an output row; each ingress (`interface.Ingress`) an input row
and units among its template's; and no input row is filled by two (`apart`). Its declared bits are the wires'
(`Summary.wiredBits`) and the ingress's, and its reads' those of input rows, by the parameter each reads. -/
def inputsOf (r : Json) (ts : Array (Flock.Program.Defn × Nat)) (pins : Array (String × String)) : V Inputs := do
  let total := inputBits ts
  if (r.getObjVal? "wires").toOption.isNone then return { total, pins }
  let tpl ← ts.toList.mapM fun (d, n) => do pure (d.key, ← layoutOf d n)
  let named ← (← Wires.arrOf "templates" (← Wires.field r "templates")).mapM fun t => match t with
    | .arr #[.str k, n] => match Wires.nat? n with
      | some n => pure (k, n)
      | none => throw "a template is [descriptor id, units]"
    | _ => throw "a template is [descriptor id, units]"
  unless named == tpl.map fun (k, t) => (k, t.units) do
    throw s!"its templates are not its Program's root templates with their units, in order: {named}"
  let cuts ← match r.getObjVal? "cuts" with
    | .error _ => pure []
    | .ok cj => do
      let cs ← (← Wires.fields cj).mapM fun (tid, kv) => do
        pure (tid, ← (← Wires.fields kv).mapM fun (p, k) => do pure (p, ← Wires.natOf "a cut" k))
      pure (cs.filter fun (_, kv) => !kv.isEmpty)
  let wires ← (← Wires.arrOf "wires" (← Wires.field r "wires")).mapM Wires.Wire.ofJson
  let c : Wires.Circuit := { templates := tpl, cuts, wires }
  let sum ← match Wires.check c with
    | .ok s => pure s
    | .error e => throw s!"wires: refused ({e.code.name}{(e.wire.map (s!", wire {·}")).getD ""})"
  let lay := fun (tid : String) => tpl.lookup tid
  let row := fun (tid p : String) => (lay tid).bind (·.rows.lookup p)
  let unitsOk := fun (tid : String) (lo n : Nat) => match lay tid with
    | some t => 1 ≤ n && lo + n ≤ t.units
    | none => false
  let (params, tableJs, readJs, rootJs, openJs) ← match r.getObjVal? "registered" with
    | .error _ => pure ([], [], [], [], [])
    | .ok g => do
      exactly g ["openings", "params", "reads", "roots", "tables"] "registered"
      pure (← Wires.fields (← Wires.field g "params"), ← itemsOf g "tables", ← itemsOf g "reads",
        ← itemsOf g "roots", ← itemsOf g "openings")
  let params ← params.mapM fun (p, v) => do
    let n ← Wires.natOf s!"registered parameter {p}'s leaves" v
    unless 1 ≤ n do throw s!"registered parameter {p}: at least one leaf"
    pure (p, n)
  let mut tables : List (String × String) := []
  for t in tableJs do
    Wires.sameKeys t ["name", "param", "offset", "words", "rows"] "a table"
    let (name, param) := (← strIn t "name" "a table", ← strIn t "param" "a table")
    let (offset, words, rows) := (← natIn t "offset" "a table", ← natIn t "words" "a table", ← natIn t "rows" "a table")
    unless (tables.lookup name).isNone && 1 ≤ words && 1 ≤ rows &&
        (params.lookup param).any fun n => decide (offset + words * rows ≤ n) do
      throw s!"table {name}: a new name, rows of a registered parameter's leaves"
    tables := tables ++ [(name, param)]
  let mut rects := sum.wires.map wireRect
  let mut reads : Array (String × Nat) := #[]
  for j in readJs do
    let w ← readOf j
    let k := c.cutsOf w.reader w.port
    let some words := (row w.reader w.port).map (·.words) <|> ((lay w.reader).bind (·.outs.lookup w.port))
      | throw s!"read of {w.reader} {w.port}: names no parameter of a template of the circuit nor its output"
    let (lo, hi) := w.span
    unless w.cut0 + w.cuts ≤ k && (w.cuts == 1 || w.radix.getLast? == some w.cuts) && w.units % w.cuts == 0 &&
        unitsOk w.reader w.start w.readerUnits && decide (0 ≤ lo) &&
        (params.lookup w.writer).any fun n => decide (hi + ((words / k : Nat) : Int) ≤ (n : Int)) do
      throw s!"read of {w.reader} {w.port}: rows of its port's cut, units among its template's and leaves of a \
        registered parameter"
    rects := rects ++ [wireRect w]
    if (row w.reader w.port).isSome then reads := reads.push (w.writer, w.units * (16 * (words / k)))
  for j in rootJs do
    Wires.sameKeys j ["reader", "port", "table", "of", "at", "units"] "a table read"
    let (tid, port, table, of) := (← strIn j "reader" "a table read", ← strIn j "port" "a table read",
      ← strIn j "table" "a table read", ← strIn j "of" "a table read")
    let (lo, n) := (← natIn j "at" "a table read", ← natIn j "units" "a table read")
    let some p := row tid port | throw s!"table read of {tid} {port}: names no parameter of a template of the circuit"
    let some param := tables.lookup table | throw s!"table read of {tid} {port}: of no declared table"
    unless unitsOk tid lo n && (of == "root" || of == "domain") && p.words == 32 do
      throw s!"table read of {tid} {port}: a declared table's root or domain id, a 64-byte port, units among its \
        template's"
    rects := rects ++ [⟨tid, port, lo, lo + n, 0, c.cutsOf tid port⟩]
    reads := reads.push (param, n * (16 * p.words))
  for j in openJs do
    Wires.sameKeys j ["reader", "table", "key", "ports", "verdict", "at", "units"] "an opening"
    let (tid, table, key, verdict) := (← strIn j "reader" "an opening", ← strIn j "table" "an opening",
      ← strIn j "key" "an opening", ← strIn j "verdict" "an opening")
    let (lo, n) := (← natIn j "at" "an opening", ← natIn j "units" "an opening")
    let ports ← (← Wires.arrOf "ports" (← Wires.field j "ports")).mapM (Wires.strOf "a port")
    let some param := tables.lookup table | throw s!"opening of {tid}: of no declared table"
    unless unitsOk tid lo n && (row tid key).isSome && ((lay tid).bind (·.outs.lookup verdict)).isSome &&
        !ports.isEmpty && ports.all fun q => (row tid q).isSome do
      throw s!"opening of {tid}: a declared table's row at its key, with a verdict output, its ports and units its \
        template's"
    for q in ports do
      rects := rects ++ [⟨tid, q, lo, lo + n, 0, c.cutsOf tid q⟩]
      reads := reads.push (param, n * (16 * ((row tid q).map (·.words)).getD 0))
  let mut ingress := 0
  for j in ← itemsOf r "ingress" do
    Wires.sameKeys j ["reader", "port", "at", "units", "step"] "an ingress"
    let (tid, port) := (← strIn j "reader" "an ingress", ← strIn j "port" "an ingress")
    let (lo, n, _) := (← natIn j "at" "an ingress", ← natIn j "units" "an ingress", ← natIn j "step" "an ingress")
    let some p := row tid port | throw s!"ingress of {tid} {port}: names no parameter of a template of the circuit"
    unless unitsOk tid lo n do throw s!"ingress of {tid} {port}: units among its template's"
    rects := rects ++ [⟨tid, port, lo, lo + n, 0, c.cutsOf tid port⟩]
    ingress := ingress + n * (16 * p.words)
  unless apart rects do throw "an input row is filled by two of its wires and declarations"
  pure { total, declared := sum.wiredBits + ingress, reads, pins }

/-- The verifier's pins of audit `a`'s registered parameters (its `pins`), fixed before the run: `{param: root}`, each
root a SHA-512 digest; none without `pins`. -/
def pinsOf (a : Json) (ctx : String) : V (Array (String × String)) := do
  let .ok j := a.getObjVal? "pins" | pure #[]
  let kvs ← (Wires.fields j).mapError fun _ => s!"{ctx}: pins: not an object of roots by parameter"
  (kvs.toArray.mapM fun (p, v) => match v with
    | .str r => if isDigest r then pure (p, r) else throw s!"{ctx}: pins: {p}'s root is not a SHA-512 digest"
    | _ => throw s!"{ctx}: pins: {p}'s root is not a SHA-512 digest")

/-- A compute audit `a`'s inputs, monitoring period and Program, at the Program `P` its descriptor (`a`'s `program`)
decodes to: its units are its root's template instances, each committing its template's return; a root with no
template is refused. With its window's circuit record (`circuit`), its `circuit` is the record's SHA-512
(`circuitDigest`), `registered` what the record registers of its slot entries (`entriesOf`), and its inputs counted
from the record (`inputsOf`) under the verifier's pins (`pins`, `pinsOf`); its slot entries' templates are its root's
template nodes whose descriptor id the record registers as an entry template, each with its closure digest
(`closureDigest`). Without a record it registers none, and all its inputs count free. -/
def auditOfProgram (a : Json) (P : Flock.Program.Prog) (ctx : String) : V Audit := do
  let root ← needSome P.defs[P.root]? s!"{ctx}: program: no root"
  let templates := root.body.nodes.filterMap fun x => match P.defs[x.fn]? with
    | some d => if d.prim then none else some (d.key, x.members, bitsOf d.ret)
    | none => none
  need (!templates.isEmpty) s!"{ctx}: program: no template"
  let program := Flock.hexOf P.sha512
  let pins ← pinsOf a ctx
  let (circuit, registered, inputs) ← (match a.getObjVal? "circuit" with
    | .ok r => do
      let circuit ← (circuitDigest r).mapError (s!"{ctx}: circuit: " ++ ·)
      let registered ← entriesOf r (a.getObjValD "program") program (templates.map (·.1)) ctx
      pure (circuit, registered, ← (inputsOf r (rootTemplates P) pins).mapError (s!"{ctx}: circuit: " ++ ·))
    | .error _ => pure ("", ({} : Entries), ({ total := inputBits (rootTemplates P) } : Inputs)))
  let entries ← (templates.filter fun (k, _, _) => registered.templates.any (·.1 == k)).mapM fun (k, n, _) => do
    pure (k, n, ← (closureDigest (a.getObjValD "program") k).mapError (s!"{ctx}: " ++ ·))
  pure { inputs, steps := ← Flock.J.nat a "steps" ctx, program, units := templates.foldl (· + ·.2.1) 0,
         bits := templates.foldl (fun b t => max b t.2.2) 0, entries, circuit, registered }

/-- A compute audit's inputs, monitoring period and Program (`auditOfProgram`), from the Program's descriptor
(`Flock.Program.decode` of its canonical bytes), with its circuit record and the verifier's pins where it carries them.
A `free` count, a natural, is read as nothing: the check derives it (`Audit.free`), so no number the caller supplies
reaches the verdict or the bound. -/
def auditOf (a : Json) (ctx : String) : V Audit := do
  let ks := (keysOf a).filter (· != "free")
  need (ks == ["program", "steps"] || ks == ["circuit", "program", "steps"] ||
      ks == ["circuit", "pins", "program", "steps"]) s!"{ctx}: the keys must be [program, steps], [circuit, program, \
    steps] or [circuit, pins, program, steps], and optionally free, not {keysOf a}"
  discard <| if (keysOf a).contains "free" then Flock.J.nat a "free" ctx else pure 0
  let pj ← Flock.J.get a "program" ctx
  auditOfProgram a (← (Flock.Program.decode (← Flock.canon pj).toUTF8).mapError (s!"{ctx}: program: " ++ ·)) ctx

/-- Whether a batch node of ingress buckets `t0, t0 + 1, …` reads the slot entries at their slots: for each lane `j`
and member `m` (bucket `t0 + m`), its argument `o + j` (`prog<j>`) is row `⌊(t0 + m) / every⌋` of input node `ks[j]`,
lane `j`'s slot entries, `E` leaves a row, leaf by leaf (`Flock.Program.Arg.memberRef`). That row is lane `j`'s slot of
the boundary whose Programs bucket `t0 + m` reads, leaf `⌊(t0 + m) / every⌋` of the chunk `wired` holds lane `j`'s
tree to. -/
def readsEntries (x : Flock.Program.Node) (o t0 every E : Nat) (ks : List Nat) : Bool :=
  ks.zipIdx.all fun (k, j) => match x.args[o + j]? with
    | some a => (List.range x.n).all fun m => (List.range E).all fun l =>
        a.memberRef m l == { p := false, i := k, j := (t0 + m) / every * E + l }
    | none => false

/-- A batch of 3 members whose argument `1 + j` reads leaf `rows[m]` of node `k` at member `m`, one leaf a member, for
lane `j`'s `(k, rows)`. -/
def entriesProbe (lanes : List (Nat × List Nat)) : Flock.Program.Node :=
  let arg := fun (k : Nat) (rs : List Nat) => Flock.Program.Arg.mk (.value 1)
    (.ofParts (rs.toArray.map fun r => Flock.Program.Part.affine false k r 1)) (some (rs.length, 1))
  Flock.Program.Node.mk .batch 0 (#[arg 0 [0, 0, 0]] ++ (lanes.map fun (k, rs) => arg k rs).toArray) (.value 1) #[] 3 0

#guard readsEntries (entriesProbe [(7, [0, 1, 1])]) 1 1 2 1 [7] &&
  readsEntries (entriesProbe [(7, [0, 1, 1]), (8, [0, 1, 1])]) 1 1 2 1 [7, 8] &&
  !readsEntries (entriesProbe [(7, [0, 1, 0])]) 1 1 2 1 [7] &&
  !readsEntries (entriesProbe [(7, [0, 1, 1])]) 1 1 2 1 [7, 8] &&
  !readsEntries (entriesProbe [(7, [0, 1, 1])]) 1 1 2 1 [6] &&
  !readsEntries (entriesProbe [(7, [0, 1, 1]), (8, [0, 1, 1])]) 1 1 2 1 [8, 7] &&
  !readsEntries (entriesProbe [(7, [0, 1, 1]), (7, [2, 3, 3])]) 1 1 2 1 [7, 7]

/-- The scan's terms from its Program's descriptor (`Flock.Program.decode` of its canonical bytes), which must be the
registered Program (`program`, its SHA-512) and the generator form of the schedule's execution phase: its root batches
one instance of bucket 0's template, `T - 1` of the later ingress buckets' (no such template when `T = 1`), both
returning the same and each one's `row` the schedule's row (`S · B` bytes), then `T` of the egress template, which
returns a u16 and the schedule's row, and every input gate is a template's parameter. The types are the Boolean profile's, as the
scan's Definitions have them: a byte 8 bits, a u16 16. An egress template with a row among its parameters, the checker
form, is refused first, by name: there the egress row is an input gate, which `NetTiming.BoundaryGates.Fits` refuses, and `K`
would count the free inputs of a Program that only compares its egress rows.

Then what `NetTiming.BoundaryGates.Fits` asks of the layout the templates give (`Properties.NetworkScan.scanLayout`), read off
the description, at the cost of its nodes and returns: the Program is laid out as C-Flock's decoder lays one out and
within what its resolution walks (`NciCheck.Desc`'s `rootLaidOut`, `inputsAtRoot` and `walkable`); the root returns
every egress leaf (one affine run from the egress node's first leaf over all `T` instances'); the egress template
returns only gates it computes (`ownReturns`: no parameter's leaf passed through, so no egress gate is an input); each
ingress template's row, and every template's row, state and egress argument, is one input node's leaves (`inputArg`),
so each ingress gate is an input and a template reads only input gates outside its own; no egress template's argument
is an ingress row's input node, so no egress gate depends on an ingress gate; and the Program is topological
(`Prog.orderViolation`, as `NciCheck.Desc.evaluate` checks it), so a gate reads only gates before it.

The ingress templates' entry ports `prog<j>` all have one type, a slot entry's, and lane `j`'s read one input node of
its own, distinct from every other lane's and from the rows, lane `j`'s slot entries (`T / every` rows each, `lanes ·
T / every` counted once among the input gates): bucket `t`'s `prog<j>` is its row `⌊t / every⌋` (`readsEntries`), the
leaf the period check's `wired` holds to chunk `j` of an audit's registered entry template's entries port; their bits
are the scan's `entries`. -/
def scanOf (j : Json) (program : String) (s : Schedule) : V Scan := do
  let prog ← (Flock.Program.decode (← Flock.canon j).toUTF8).mapError fun e => s!"scan: {e}"
  need (Flock.hexOf prog.sha512 == program) "scan: its SHA-512 is not the registered Program's (program)"
  let root ← needSome prog.defs[prog.root]? "scan: no root"
  let templates := (templateNodes prog).toArray.map fun k => (root.body.nodes[k]!, prog.defs[root.body.nodes[k]!.fn]!)
  let T := s.window
  let row : Flock.Program.Ty := .array (s.allocation * s.cellBytes) (.array 8 (.value 1))
  let (xe, de) ← needSome templates.back? "scan: no template"
  need (!de.params.any (fun p => p.1 == "row" || p.2 == row)) "scan: the egress template takes its row as an input \
    (the checker form): no egress gate is an input (NetTiming.BoundaryGates.Fits)"
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
      (NetTiming.BoundaryGates.Fits)"
  need ((ownReturns prog)[xe.fn]!) "scan: the egress template returns a leaf it doesn't compute (a parameter's, passed \
    through), so an egress gate would be an input (NetTiming.BoundaryGates.Fits)"
  let ingress := #[xf] ++ (later.map (#[·.1])).getD #[]
  let rows := ingress.map fun x => x.args[0]?.bind (inputArg prog)
  need (rows.all (·.isSome)) "scan: an ingress template's row is not one input node's leaves, so an ingress gate \
    would be no input (NetTiming.BoundaryGates.Fits)"
  need (xf.args.size == 1 + s.lanes && (later.all fun (x, _) => x.args.size == 2 + s.lanes) &&
      (xf.args.extract 0 1).all (fun a => (inputArg prog a).isSome) &&
      (later.all fun (x, _) => (x.args.extract 0 2).all fun a => (inputArg prog a).isSome) &&
      xe.args.all fun a => (inputArg prog a).isSome)
    "scan: a template's row, state or egress argument is not one input node's leaves"
  need (xe.args.all fun a => !rows.contains (inputArg prog a)) "scan: an egress template's argument is an ingress \
    row's input node, so an egress gate would depend on the ingress (NetTiming.BoundaryGates.Fits)"
  let a0 ← needSome ((List.range s.lanes).mapM fun j => xf.args[1 + j]?) "scan: bucket 0 has no entry port a lane"
  let pks := a0.map fun a => (a.memberRef 0 0).i
  need (a0.all (fun a => !(a.memberRef 0 0).p) && decide pks.Nodup && pks.all fun pk =>
      !rows.contains (some pk) && match root.body.nodes[pk]? with
        | some n => NciCheck.Desc.isInputNode prog n
        | none => false)
    "scan: bucket 0's prog<j> reads no input node of lane j's own, its slot entries', distinct from every other lane's"
  need (readsEntries xf 1 0 s.every ety.leaves pks &&
      later.all fun (x, _) => readsEntries x 2 1 s.every ety.leaves pks)
    s!"scan: bucket t's prog<j> is not row ⌊t / {s.every}⌋ of lane j's slot entries' input node, its slot of the \
      boundary bucket t reads (NetworkCheck.Period.readsEntries)"
  need prog.orderViolation.isNone "scan: a body names a later node (Flock.Program.Prog.orderViolation): the Program \
    is not topological, so a gate's operands could be gates after it"
  pure { steps := T, free := 0, ingress := bitsOf df.ret, egress := bitsOf de.ret,
         entries := s.lanes * nb * bitsOf ety }

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

/-- A window record of one of `Flock.WindowRecord.FORMATS`. -/
def registrationOf (j : Json) (ctx : String) : V Json := do
  let format ← Flock.J.str j "format" ctx
  unless Flock.WindowRecord.FORMATS.contains format do
    throw s!"{ctx}: format is not {" or ".intercalate Flock.WindowRecord.FORMATS}"
  pure j

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

/-- A certificate of `link`'s end `d` at window `w`, the key's and verifying in the parameters' session
(`Certificate.signs`). -/
def certifiesOn (p : Params) (link : String) (d : Direction) (w : Nat) (c : Certificate) : Bool :=
  c.direction == d && c.link == link && c.window == w && c.signs p.lms p.key p.session

/-- A certificate of the link's end `d` at window `w`, the key's and verifying. -/
def certifies (p : Params) (d : Direction) (w : Nat) (c : Certificate) : Bool := certifiesOn p p.link d w c

/-- A finding against the period: a rejection, or what leaves it undecided. -/
abbrev Found := Except (Core.Verdict × String)

def refuse {α} (r : String) : Found α := .error (.reject, r)

/-- One-stage's answer for the window record `reg`, read from canonical bytes, so its digest
(`Flock.WindowRecord.digest`, `registration.digest`) is always defined: its reject is a rejection, its undecided leaves
the period undecided, and its acceptance gives the profile. -/
def answered (what : String) (reg : Json) (a : Core.Answer) : Found (Core.Integrity × Json) := do
  if a.protocol != oneStage then refuse s!"{what}: protocol {a.protocol}, not one-stage's {oneStage}"
  else if a.digest.isNone || a.digest != (Flock.WindowRecord.digest reg).toOption then
    refuse s!"{what}: the answer is for another registration"
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
def windowCheck (p : Params) (clear : List ByteArray) (earlier : Array ByteArray) (w : Nat) (reg : Json)
    (x : Window) : Found Unit := do
  let what := s!"window {w}"
  let .clock clock := x.shown | refuse s!"{what}: registrations, not a clock certificate: the rows form's"
  unless reg.getObjValD "program" == .str p.program &&
      canonText (reg.getObjValD "window") == canonText (served w) do
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

/-- `warden_rows.domain`'s id: the frame-v3-sha512 domain of execution phase `w`'s rows in direction `d`, in the
session, under the schedule whose digest is `sd`. -/
def rowsDomain (s : Schedule) (sd session : ByteArray) (link : String) (d : Direction) (w : Nat) : V ByteArray := do
  let binding ← Identity.identityDigestSha512 domainTag (Json.mkObj [
    ("parameters", .str (Flock.hexOf sd)), ("session", .str (Flock.hexOf session)), ("link", .str link),
    ("direction", .str d.value), ("window", toJson w), ("cell_bytes", toJson s.cellBytes),
    ("slots", toJson s.allocation), ("rows", toJson s.window), ("row_schema", .str rowSchema)])
  pure (Flock.FV3.Sha512.Domain.make binding (-1) 0 s.window).id

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
def holds (reg : Json) (r : Rows) : Bool :=
  let entries := fun (k : String) => match reg.getObjValD k with | .arr xs => xs | _ => #[]
  (entries "roots" ++ entries "registered").any fun e =>
    e.getObjValD "root" == .str (Flock.hexOf r.root) && e.getObjValD "domain" == .str r.domain

/-- The scan's Programs' rows in its registration, one tree a lane: lane `j`'s is the entry of its `roots` whose port,
or of its `registered` whose value, is `scan/prog<j>`. -/
def progName (j : Nat) : String := s!"scan/prog{j}"

/-- Whether `cs` is a lane's or a chunk's index: decimal digits, no leading zero. -/
def indexChars (cs : List Char) : Bool := match cs with
  | [] => false
  | d :: ds => (d :: ds).all Char.isDigit && (d != '0' || ds.isEmpty)

/-- Whether a port or a value is some lane's Programs' rows (`progName`), read over its characters. -/
def progPort (name : String) : Bool :=
  let p := "scan/prog".toList
  p.isPrefixOf name.toList && indexChars (name.toList.drop p.length)

/-- A unit's committed output port, `<unit>/o<j>` with j ≥ 1: an output a unit computes past its public word `o0`. Read
over the port's characters, which the kernel evaluates (`String.splitOn` it doesn't). -/
def outputPort (port : String) : Bool :=
  let cs := port.toList
  cs.contains '/' && match (cs.reverse.takeWhile (· != '/')).reverse with
    | 'o' :: d :: ds => (d :: ds).all Char.isDigit && d != '0'
    | _ => false

/-- An entry template's entry rows of chunk `j`, the public word of its `j`-th member's units:
`<descriptor id>/c<j>/o0`. One-stage cuts a template into members of consecutive units, in order, named
`<descriptor id>/c<j>`, at the ends of every wire its units read; an entry template whose units read their lane's
occupancy a wire a lane has members of a lane each. -/
def entriesPort (key : String) (j : Nat) : String := s!"{key}/c{j}/o0"

/-- Whether a port is some template's entries port of some chunk (`entriesPort`), `<descriptor id>/c<j>/o0` with a
descriptor id and the chunk's index, read over its characters: its `o0` is what the chunk's units compute, so it ties
its audit as an output port does, whichever template it is (`wired` holds the scan's lanes to the entry templates the
audit's circuit record registers). -/
def entriesPortName (port : String) : Bool :=
  let r := port.toList.reverse
  "0o/".toList.isPrefixOf r && match (r.drop 3).dropWhile Char.isDigit with
    | 'c' :: '/' :: _ :: _ => indexChars ((r.drop 3).takeWhile Char.isDigit).reverse
    | _ => false

/-- The audits the scan's Programs' rows are an output of, by index: those whose registration's `roots` has an output
port's or an entries port's entry (`outputPort`, `entriesPortName`) at the root and domain of a lane's `scan/prog<j>`
entry of the scan's registration (`progPort`), so the rows its ingress units read as `prog<j>` are the ones that audit's
units computed. -/
def tiedAudits (audits : Array Json) (reg : Json) : Array Nat :=
  let entries := fun (j : Json) (k : String) => match j.getObjValD k with | .arr xs => xs | _ => #[]
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str s => s == n | _ => false
  let lane := fun (k : String) (e : Json) => match e.getObjValD k with | .str s => progPort s | _ => false
  let prog := (entries reg "roots").filter (lane "port") ++
    (entries reg "registered").filter (lane "value")
  (Array.range audits.size).filter fun i => prog.any fun e => match e.getObjValD "root", e.getObjValD "domain" with
    | .str root, .str dom => ((audits[i]?.map (entries · "roots")).getD #[]).any fun o =>
        (match o.getObjValD "port" with | .str q => outputPort q || entriesPortName q | _ => false) &&
          named "root" root o &&
          named "domain" dom o
    | _, _ => false

/-- The scan's Programs' rows are a compute audit's output: some audit is tied to them (`tiedAudits`). -/
def progTied (audits : Array Json) (reg : Json) : Bool := !(tiedAudits audits reg).isEmpty

/-- Audit `a`'s registration `reg` binds its circuit record: the registration's window names the record's SHA-512
(`Audit.circuit`) as its `circuit`, so what the record registers was fixed with the registration, before any coin. It
compares strings, not `Json`, whose equality is partial, so the kernel reduces it. -/
def Audit.boundIn (a : Audit) (reg : Json) : Bool :=
  match (reg.getObjValD "window").getObjValD "circuit" with
  | .str c => c == a.circuit
  | _ => false

/-- A registered parameter is held in the registration `reg` when the root its window records for it
(`registered_roots`, `Window.write`) is the verifier's pin of it (`interface.held_params`), one fixed before the run, by
its own name: a root the prover commits after serving, or another root than the pinned one, constrains nothing. -/
def Inputs.held (i : Inputs) (reg : Json) (p : String) : Bool :=
  match ((reg.getObjValD "window").getObjValD "registered_roots").getObjValD p with
  | .str r => i.pins.contains (p, r)
  | _ => false

/-- The bits the registered reads, table reads and openings of input rows read of the parameters held in `reg`. -/
def Inputs.heldBits (i : Inputs) (reg : Json) : Nat :=
  i.reads.foldl (fun n (p, b) => if i.held reg p then n + b else n) 0

/-- **Audit `a`'s free inputs under its registration `reg`, `#(freeInputs \ reg)`**, from its registered circuit
(`interface.input_count`'s `free`): its Program's input bits less those its circuit record's wires and ingress fill and
those its registered reads read of a held parameter (`Inputs.held`), where the registration binds the record
(`Audit.boundIn`), so what it declares was fixed before any coin; else every input bit. -/
def Audit.free (a : Audit) (reg : Json) : Nat :=
  if a.boundIn reg then a.inputs.total - a.inputs.declared - a.inputs.heldBits reg else a.inputs.total

/-- The scan's Programs' rows are a compute audit's slot entries, a lane a chunk: for each lane `j`, the root and domain
of the scan's registration's `scan/prog<j>` entry (`progName`) are those of chunk `j`'s entries port
(`entriesPort`), of `T / every` leaves, of an entry template of that audit's own Program (`Audit.entries`) at the
closure digest its circuit record registers for it (`Entries.templates`, so its whole body, every Definition it calls
included, is the registered one), a record the audit's registration binds (`Audit.boundIn`), with one member a slot of
the window (`Schedule.windowPrograms`). Its members are cut in order, so chunk `j` is units
`[j · T / every, (j + 1) · T / every)` and its leaf `b` is slot `j · T / every + b`'s entry: bucket `t`'s port `prog<j>`,
which reads lane `j`'s row
`⌊t / every⌋` (`scanOf`), reads the entry that slot's Program computed, read bit included. One-stage binds the rest: a
reused value is its port's leaves, one chunk's, and an audit's roots are the ports its Program's layout derives, each
with its leaves. -/
def wired (audits : Array Audit) (s : Schedule) (regs : Array Json) (reg : Json) : Bool :=
  let entries := fun (j : Json) (k : String) => match j.getObjValD k with | .arr xs => xs | _ => #[]
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  let leaves := fun (n : Nat) (e : Json) => match (e.getObjValD "leaves").getNat? with
    | .ok l => l == n
    | .error _ => false
  let prog := fun (j : Nat) => (entries reg "roots").filter (named "port" (progName j)) ++
    (entries reg "registered").filter (named "value" (progName j))
  (audits.zip regs).any fun (a, r) => a.boundIn r && a.entries.any fun (k, n, d) => n == s.windowPrograms &&
    a.registered.templates.contains (k, d) && (List.range s.programSlots).all fun j =>
      (prog j).any fun e => match e.getObjValD "root", e.getObjValD "domain" with
        | .str root, .str dom => (entries r "roots").any fun o =>
            named "port" (entriesPort k j) o && leaves (s.window / s.every) o && named "root" root o &&
              named "domain" dom o
        | _, _ => false

/-- The scan's state in a registration: the entry of its `roots` whose port is `scan/state`. -/
def stateName : String := "scan/state"

/-- The value a window circuit registers the scan's state rows as (`ingress_scan`'s `SCAN_STATE`), row a the state
before bucket a: the entry of an audit's `registered`, or of the scan's, whose value it is. -/
def stateValue : String := "verity/certifier/scan-state/v1"

/-- A registration's state entries: those of its `roots` whose port is `stateName` and of its `registered` whose value
is `stateValue`. -/
def stateEntries (reg : Json) : Array Json :=
  let entries := fun (k : String) => match reg.getObjValD k with | .arr xs => xs | _ => #[]
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  (entries "roots").filter (named "port" stateName) ++ (entries "registered").filter (named "value" stateValue)

/-- The scan's state is the state its slots' Programs read: the scan's registration has a state entry (`stateEntries`),
every one at one root and domain, and an audit `wired` to the scan's `prog` has one, every one at that root and domain.
The scan's units read and register their states at that root (`sin` row t, `o1` row t + 1), so the `SCAN_STATE` that
audit's Programs read is the chain of the scan's transitions. -/
def stateTied (audits : Array Audit) (s : Schedule) (regs : Array Json) (reg : Json) : Bool :=
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  match (stateEntries reg)[0]? with
  | some e => match e.getObjValD "root", e.getObjValD "domain" with
    | .str root, .str dom =>
      let atRoot := fun (o : Json) => named "root" root o && named "domain" dom o
      (stateEntries reg).all atRoot && (audits.zip regs).any fun (a, r) =>
        wired #[a] s #[r] reg && !(stateEntries r).isEmpty && (stateEntries r).all atRoot
    | _, _ => false
  | none => false

/-- The free inputs of the audits a link-window's scan's `prog` is an output of (`tiedAudits`), each once, under its
own registration (`Audit.free`). -/
def tiedFree (audits : Array Audit) (regs : Array Json) (reg : Json) : Nat :=
  (tiedAudits regs reg).foldl (fun n i => n + match audits[i]?, regs[i]? with
    | some a, some r => a.free r
    | _, _ => 0) 0

/-- Audit `i` is tied to a link-window's scan: some link-window's registration's `prog` is its output (`tiedAudits`). -/
def tiedIn (audits windows : Array Json) (i : Nat) : Bool :=
  windows.any fun reg => (tiedAudits audits reg).contains i

/-- The audits the form ties to a link-window's scan: the rows form's `tiedIn`, and none in the clock form. -/
def tiedOf : Form → Commitment → Nat → Bool
  | .clock _, _ => fun _ => false
  | .rows .., c => tiedIn c.audits c.windows

/-- The certified records of the rows form's link-window: the requests' (ingress), responses' (egress) and fails'. -/
def records (link : String) : Array (String × Direction) :=
  #[(link, .ingress), (link, .egress), (link ++ "-fails", .egress)]

/-- An accepted scan's or audit's facts: one level sampled `subset:k` or stratified (a work sample's profile is
stratified, its `Σ k_s` sampled), and its bound. -/
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
facts. -/
def rowsWindowCheck (p : Params) (s : Schedule) (sd : ByteArray) (audits : Array Json) (w : Nat)
    (reg : Json) (x : Window) : Found Facts := do
  let what := s!"window {w}"
  let .rows rs := x.shown | refuse s!"{what}: a clock certificate, not registrations: the clock form's"
  unless reg.getObjValD "program" == .str p.program do refuse s!"{what}: registration: not the scan's Program"
  unless x.certificates.size == 3 && (x.certificates.zip (records p.link)).all
      (fun (c, l, d) => certifiesOn p l d w c) do
    refuse s!"{what}: certificates: the requests', responses' and fails' records', each the key's (the network \
      certifier certifies none of a window in which it refused a connection: traffic past the Program slots)"
  unless rs.size == 2 do refuse s!"{what}: registrations: the ingress rows, then the egress rows"
  for (r, d) in rs.zip #[Direction.ingress, .egress] do rowsCheck p s sd w d r
  unless rs.all (holds reg) do refuse s!"{what}: the scan's registration doesn't hold the rows: another window's answer"
  unless progTied audits reg do
    refuse s!"{what}: the scan's prog is no compute audit's output: no scan/prog<j> entry's root and domain are an \
      output port's of an audit's registration"
  let (prof, pub) ← answered what reg x.answer
  let f ← factsOf what prof pub
  unless f.units == 2 * s.window do refuse s!"{what}: profile: {f.units} units, not the scan's 2T = {2 * s.window}"
  pure f

/-- Audit `i`'s findings, and on acceptance its facts: its registration names its Program and the answer's proof
blocks are the Program's, at any sample; a tied audit sampled in part is charged in its link-windows' terms
(`chargedOf`, `faultyEntries`). -/
def auditCheck (i : Nat) (a : Audit) (reg : Json) (x : Core.Answer) : Found Facts := do
  let what := s!"audit {i}"
  unless reg.getObjValD "program" == .str a.program do refuse s!"{what}: registration: not its Program"
  let (prof, pub) ← answered what reg x
  let f ← factsOf what prof pub
  unless f.units == a.units do refuse s!"{what}: profile: {f.units} units, not its Program's {a.units}"
  pure f

/-- The audits' findings, in order. -/
def auditChecks (p : Params) (c : Commitment) (e : Evidence) : Array (Found Facts) :=
  ((p.audits.zip c.audits).zip e.audits).zipIdx.map fun (((a, reg), x), i) => auditCheck i a reg x

/-- Link-window `w`'s wiring finding in the rows form: a refusal unless its scan's Programs' row is a compute audit's
slot entries (`wired`). -/
def wiredCheck (p : Params) (s : Schedule) (audits : Array Json) (w : Nat) (reg : Json) :
    Option (Core.Verdict × String) :=
  if wired p.audits s audits reg then none else
    some (.reject, s!"window {w}: the scan's prog is no compute audit's slot entries: its scan/prog<j> entries' roots \
      and domains, j < {s.programSlots}, are not one audit's entries ports (<descriptor id>/c<j>/o0, \
      {s.window / s.every} leaves each) of {s.windowPrograms} members, one a slot, of an entry template at the closure \
      digest the audit's circuit record registers, a record its registration binds (window.circuit), so a due, \
      occupied slot's read bit is the prover's word: register the entry template in the circuit record and reuse \
      chunk j's port as the scan's prog<j>")

/-- The rows form's wiring findings, one a link-window, in order (`wiredCheck`); none in the clock form. -/
def wiredChecks (p : Params) (c : Commitment) : Array (Option (Core.Verdict × String)) := match p.form with
  | .rows s .. => c.windows.zipIdx.map fun (reg, i) => wiredCheck p s c.audits (p.first + i) reg
  | .clock _ => #[]

/-- Link-window `w`'s state finding in the rows form: a refusal unless its scan's state is the `SCAN_STATE` of an audit
its `prog` is wired to (`stateTied`). With the `prog` wired, each slot's entry is the one its Program computed, and a
Program reads the requests from its own registered `SCAN_STATE`; unless that is the scan's state, it computes its entry
on a state the prover chose. -/
def stateCheck (p : Params) (s : Schedule) (audits : Array Json) (w : Nat) (reg : Json) :
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

/-- `Σ_{j ≤ b} C(n, j)`: the sets of at most `b` of `n` proof blocks. -/
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
  /-- The slot entries' bits a tied audit's wrong units can set (`β_w`, the rows form's; 0 in the clock form). -/
  faultyEntries : Nat := 0

def AuditTerms.bits (t : AuditTerms) : Nat := t.free + t.faultySets + t.faultyBits + t.jitter + t.faultyEntries

/-- A compute audit's terms in the clock form, from its free inputs under its registration `reg` (`Audit.free`), its
monitoring period, its facts and the jitter bound: each of its at most `b` faulty proof blocks commits at most its
Program's widest proof block's bits. -/
def AuditTerms.of (a : Audit) (reg : Json) (f : Facts) (jitter : Nat) : AuditTerms :=
  { facts := f, free := a.free reg, faultySets := clog2 (atMost f.units f.wrong), faultyBits := f.wrong * a.bits,
    jitter := clog2 (jitter ^ a.steps) }

/-- A link-window's scan's terms in the rows form, from the scan's terms, its facts, its tied audits' free inputs
(`tied`: the egress renders their outputs, so the bits the prover chose of them are the link-window's free inputs), the
slot entries' bits its tied audits' wrong units can set (`faulty`, `faultyEntries`) and the jitter bound: each of its
at most `b` wrong units commits at most the wider instance's bits. -/
def Scan.terms (s : Scan) (f : Facts) (tied faulty jitter : Nat) : AuditTerms :=
  { facts := f, free := s.free + tied, faultySets := clog2 (atMost f.units f.wrong),
    faultyBits := f.wrong * max s.ingress s.egress, jitter := clog2 (jitter ^ s.steps), faultyEntries := faulty }

structure Public where
  /-- The compute audits' facts. -/
  audits : Array Facts
  cells : Nat
  /-- The terms of `PeriodLeakAccepted`'s windows: the clock form's compute audits', the rows form's scans', one a
  execution phase. -/
  units : Array AuditTerms
  /-- The rows form, whose proof blocks are its scans. -/
  rows : Bool := false
  schedules : Nat
  windows : Nat
  /-- The facts of the rows form's tied audits drawn in part (`chargedOf`), whose bounds `faulty_entries` reads, so
  whose δ is in `eps`. -/
  charged : Array Facts := #[]

def Public.K (x : Public) : Nat := x.schedules + x.cells + (x.units.map (·.bits)).foldl (· + ·) 0

/-- `Σ_i δ_i` over the units and the charged audits, exactly, in lowest terms: the union bound on `K` failing. -/
def Public.eps (x : Public) : Core.Ratio :=
  let (n, d) := (x.units.map (·.facts) ++ x.charged).foldl
    (fun (n, d) f => (n * f.delta.den + f.delta.num * d, d * f.delta.den)) (0, 1)
  ⟨n / Nat.gcd n d, d / Nat.gcd n d⟩

/-- `Σ_c ⌈log2 (cap_c + 1)⌉`. -/
def capsTerm (cells : Array Nat) : Nat := (cells.map fun cap => clog2 (cap + 1)).foldl (· + ·) 0

/-- The bound's terms from the audits (their free inputs under their registrations `regs`, monitoring periods and
Programs' bits) and facts, the cells' caps, the jitter bound, `#Σ_sync` and the execution phases. -/
def terms (audits : Array Audit) (regs : Array Json) (facts : Array Facts) (cells : Array Nat)
    (jitter syncs windows : Nat) : Public :=
  { audits := facts
    cells := capsTerm cells
    units := ((audits.zip regs).zip facts).map fun ((a, r), f) => AuditTerms.of a r f jitter
    schedules := clog2 (syncs ^ windows)
    windows }

/-- The rows form's terms: the cells' as the clock form's, the schedules' the same count (each connection's lane
length, one of `cap_c + 1`, fixes its admission and the lanes it frees), and each link-window's scan's from the scan's
terms, its facts, its tied audits' free inputs and the entry bits their wrong units can set (`scans`, a triple a
link-window). The compute audits' own terms are in none: a tied audit's free inputs and wrong units are in its
link-window's, and the δ of one drawn in part (`charged`) in `eps`. -/
def rowsTerms (facts : Array Facts) (scan : Scan) (scans : Array (Facts × Nat × Nat)) (cells : Array Nat)
    (jitter windows : Nat) (charged : Array Facts := #[]) : Public :=
  { audits := facts, cells := capsTerm cells, units := scans.map fun (f, t, b) => scan.terms f t b jitter,
    rows := true, schedules := capsTerm cells, windows, charged }

/-- Audit `i`'s facts when the rows form charges it: accepted, tied to a link-window's scan (`tiedOf`) and drawn in
part, so its at most `b` wrong units are in its link-windows' terms (`faultyEntries`) and its δ in `eps`. -/
def chargedOf (p : Params) (c : Commitment) (as : Array (Found Facts)) (i : Nat) : Option Facts :=
  match as[i]? with
  | some (.ok f) => if tiedOf p.form c i && f.drawn < f.units then some f else none
  | _ => none

/-- A link-window's `β_w`: every slot entry bit of its scan (`Scan.entries`, `E_w`) when an audit its `prog` is tied
to (`tiedAudits`) is charged (`chargedOf`) with a bound of at least one wrong unit, which can set any of the entries,
else 0. -/
def faultyEntries (p : Params) (c : Commitment) (as : Array (Found Facts)) (scan : Scan) (reg : Json) : Nat :=
  if (tiedAudits c.audits reg).any fun i => (chargedOf p c as i).any (1 ≤ ·.wrong) then scan.entries
  else 0

/-- The rows form's public on acceptance: `rowsTerms` over the accepted audits' facts, each link-window's scan's facts,
its tied audits' free inputs (`tiedFree`) and `β_w` (`faultyEntries`), and the charged audits (`chargedOf`). -/
def rowsPublic (p : Params) (c : Commitment) (as : Array (Found Facts)) (scan : Scan) (scans : Array Facts) : Public :=
  rowsTerms (as.filterMap (·.toOption)) scan
    ((scans.zip c.windows).map fun (f, reg) => (f, tiedFree p.audits c.audits reg, faultyEntries p c as scan reg))
    p.cells p.jitter p.windows ((Array.range as.size).filterMap (chargedOf p c as))

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
    let ws : Array (Found (Option Facts)) := match p.form with
      | .clock ss =>
        let clear := clearClocks ss
        pairs.map fun ((reg, x), i) => (fun () => none) <$>
          windowCheck p clear ((e.windows.extract 0 i).filterMap (·.clockDigest?)) (p.first + i) reg x
      | .rows s sd _ => pairs.map fun ((reg, x), i) => some <$> rowsWindowCheck p s sd c.audits (p.first + i) reg x
    let scans := ws.filterMap (·.toOption.join)
    let oneDelta : Option (Core.Verdict × String) := scans[0]?.bind fun f0 =>
      (scans.find? (·.delta != f0.delta)).map fun f =>
        (.reject, s!"a scan's δ {f.delta.toString} is not the period's, {f0.delta.toString}")
    let as := auditChecks p c e
    match firstFinding (ws.toList.map findingOf ++ [oneDelta] ++ as.toList.map findingOf ++ (wiredChecks p c).toList ++
        (stateChecks p c).toList) with
    | some (.reject, r) => .reject r
    | some (_, r) => .undecided r
    | none =>
      .accept (match p.form with
          | .clock ss =>
            terms p.audits c.audits (as.filterMap (·.toOption)) p.cells p.jitter (ss.deltas.length * ss.rhos.length)
              p.windows
          | .rows _ _ scan => rowsPublic p c as scan scans)
        { link := p.link, windows := p.windows }

/-- The terms' JSON, each with its `faulty_entries` when `entries` (the rows form's scans). -/
def termsJson (ts : Array AuditTerms) (entries : Bool := false) : Json :=
  .arr <| ts.map fun t => Json.mkObj ([
    ("delta", .str t.facts.delta.toString), ("drawn", toJson t.facts.drawn), ("faulty_bits", toJson t.faultyBits)] ++
    (if entries then [("faulty_entries", toJson t.faultyEntries)] else []) ++ [
    ("faulty_sets", toJson t.faultySets), ("free", toJson t.free), ("jitter", toJson t.jitter),
    ("units", toJson t.facts.units), ("wrong_units", toJson t.facts.wrong)])

def factsJson (fs : Array Facts) : Json :=
  .arr <| fs.map fun f => Json.mkObj [("delta", .str f.delta.toString), ("drawn", toJson f.drawn),
    ("units", toJson f.units), ("wrong_units", toJson f.wrong)]

def publicJson (x : Public) : Json :=
  Json.mkObj ([
    ("K", toJson x.K), ("audits", if x.rows then factsJson x.audits else termsJson x.units), ("cells", toJson x.cells),
    ("eps", .str x.eps.toString), ("schedules", toJson x.schedules), ("windows", toJson x.windows)] ++
    (if x.rows then [("scans", termsJson x.units true)] else []))

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

/-! ## The clock openings -/

/-- A salted clock certificate's opening (`certificate.CertifiedClock`): the sync the certifier ran the window under and
its salt `salt_c`, both witness, and the certificate, which signs the sync's clock digest salted by it. -/
structure ClockOpening where
  sync : ClockSync
  salt : ByteArray
  certificate : Certificate

/-- Two openings are one: the same sync, salt and certificate, field by field. -/
def ClockOpening.same (a b : ClockOpening) : Bool :=
  let x := a.certificate
  let y := b.certificate
  a.sync.delta == b.sync.delta && a.sync.rho == b.sync.rho && a.salt == b.salt && x.keyId == y.keyId &&
    x.link == y.link && x.direction == y.direction && x.window == y.window && x.digest == y.digest &&
    x.signature == y.signature

/-- One certifier session's clock openings (`certificate.SignedClocks`): its LMS key and session, the openings, and the
link whose syncs are read. -/
structure Openings where
  lms : Lms.Params
  key : ByteArray
  session : ByteArray
  link : String
  openings : Array ClockOpening

/-- `{"key", "link", "lms", "openings", "session"}`, each opening `{"certificate", "salt", "sync"}`, its certificate as
the evidence's and its sync `[δ, num, den]`. -/
def openingsOf (j : Json) : V Openings := do
  exactly j ["key", "link", "lms", "openings", "session"] "the object"
  let l ← Flock.J.get j "lms" "the object"
  exactly l ["H", "N", "W", "toy"] "lms"
  let lms ← Lms.Params.make (← Flock.J.nat l "N" "lms") (← Flock.J.nat l "W" "lms") (← Flock.J.nat l "H" "lms")
    (← Flock.J.bool l "toy" "lms")
  let key ← Flock.unhex (← Flock.J.str j "key" "the object")
  unless key.size = 16 + lms.N do throw s!"an LMS public key is {16 + lms.N} bytes, not {key.size}"
  let openings ← (← Flock.J.arr j "openings" "the object").mapM fun o => do
    exactly o ["certificate", "salt", "sync"] "an opening"
    pure { sync := ← Check.syncOf (← Flock.J.get o "sync" "an opening"), salt := ← hexAt o "salt" "an opening",
           certificate := ← certificateOf (← Flock.J.get o "certificate" "an opening") "an opening's certificate" }
  pure { lms, key, session := ← hexAt j "session" "the object", link := ← Flock.J.str j "link" "the object",
         openings }

/-- Why an opening gives no sync (`CertifiedClock.refusal`), or none when it holds: its certificate isn't an egress one
(`egress`), isn't the key's (`key`), signs another digest than its sync's clock digest salted by its 64-byte salt
(`clock_commitment`, `opening`), or its signature doesn't verify in the session (`Certificate.signs`: `signature`). -/
def openingRefusal (s : Openings) (o : ClockOpening) : Option String :=
  let c := o.certificate
  if c.direction != .egress then some "egress"
  else if c.keyId != s.key.extract 0 16 then some "key"
  else if !(o.salt.size == saltBytes && (clockDigest o.sync).any (c.digest == certificateDigest o.salt ·)) then
    some "opening"
  else if !c.signs s.lms s.key s.session then some "signature"
  else none

/-- The openings that hold by `first` (none), by salt: the first of each salt, and whether another of that salt isn't
it. -/
def bySalt (s : Openings) (first : Array (Option String)) : Std.TreeMap.Raw (Array UInt8) (ClockOpening × Bool) :=
  (s.openings.zip first).foldl (init := ∅) fun m (o, r) =>
    if r.isSome then m
    else m.alter o.salt.data fun
      | some (x, another) => some (x, another || !x.same o)
      | none => some (o, false)

/-- Each opening's refusal: `openingRefusal`'s, else `salt` where another opening that holds, not this one, has its
salt, under which two windows of one sync would sign one digest; none where it holds. The openings are keyed by salt
once (`bySalt`), so past their signatures this takes n log n steps in the openings. -/
def openingRefusals (s : Openings) : Array (Option String) :=
  let first := s.openings.map (openingRefusal s)
  let salts := bySalt s first
  (s.openings.zip first).map fun (o, r) =>
    if r.isSome then r else if (salts.get? o.salt.data).any (·.2) then some "salt" else none

/-- Link-windows in order: by link, then by window. -/
def linkWindowOrder (a b : String × Nat) : Ordering := (compare a.1 b.1).then (compare a.2 b.2)

/-- What the openings that hold sign, by link-window: the first one's sync, and whether another signs another sync. -/
abbrev Signed := Std.TreeMap.Raw (String × Nat) (ClockSync × Bool) linkWindowOrder

/-- The openings that hold (`codes` none), keyed by link-window once, so that `signedSyncs` and `split` take n log n
steps in the openings. -/
def signedBy (s : Openings) (codes : Array (Option String)) : Signed :=
  (s.openings.zip codes).foldl (init := ∅) fun m (o, r) =>
    if r.isSome then m
    else m.alter (o.certificate.link, o.certificate.window) fun
      | some (x, two) => some (x, two || !(x.delta == o.sync.delta && x.rho == o.sync.rho))
      | none => some (o.sync, false)

/-- `SignedClocks.syncs(link)`: window 0's sync on, each the one sync the holding openings sign for the window, up to
the first window without exactly one. -/
def signedSyncs (s : Openings) (signed : Signed) : Array ClockSync := Id.run do
  let mut out := #[]
  for w in [0:s.openings.size] do
    match signed.get? (s.link, w) with
    | some (x, false) => out := out.push x
    | _ => break
  return out

/-- The link-windows, of any link, that the holding openings sign two or more syncs for, which have none, in
`linkWindowOrder`. -/
def split (signed : Signed) : Array (String × Nat) :=
  signed.foldl (init := #[]) fun acc k (_, two) => if two then acc.push k else acc

/-- `network-records verdicts`' `clocks`: on `openingsOf`'s object, each opening's refusal (`openingRefusals`, null
where it holds), the link-windows with two or more signed syncs (`split`, each `[link, window]`), and the link's signed
syncs (`signedSyncs`, each `[δ, num, den]`), which `lean_audit.signed_clocks` reads with `certificate.SignedClocks`
beside it as the reference, each refusal and each such link-window by identity. -/
def clocksFn (j : Json) : Except String Json := do
  let s ← openingsOf j
  let codes := openingRefusals s
  let signed := signedBy s codes
  let sync (x : ClockSync) : Json := .arr #[toJson x.delta, toJson x.rho.num, toJson x.rho.den]
  return Json.mkObj [("refused", .arr (codes.map fun | some r => .str r | none => .null)),
    ("split", .arr ((split signed).map fun (l, w) => .arr #[.str l, toJson w])),
    ("syncs", .arr ((signedSyncs s signed).map sync))]

#guard clog2 0 == 0 && clog2 1 == 0 && clog2 2 == 1 && clog2 3 == 2 && clog2 4 == 2 && clog2 5 == 3
#guard atMost 10 0 == 1 && atMost 10 1 == 11 && atMost 10 2 == 56 && atMost 4 9 == 16 && atMost 0 3 == 1
-- an audit of 10 proof blocks, each committing 16 bits, sampled subset:3 at δ = 1/10 (at most 4 wrong), 7 cells on one
-- connection and none on another, 2 syncs over 3 link-windows, and jitter below 2 over 3 steps
#guard (terms #[⟨{}, 3, "", 10, 16, #[], "", {}⟩] #[.null] #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }]
  #[7, 0] 2 2 3
  ).K == 3 + 3 + (0 + 9 + 64 + 3)
-- the same audit with a unit committing 40 bits: its 4 wrong units carry 160 bits, not 16 · 4
#guard (terms #[⟨{}, 3, "", 10, 40, #[], "", {}⟩] #[.null] #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }]
  #[7, 0] 2 2 3
  ).K == 3 + 3 + (0 + 9 + 160 + 3)
-- two audits at δ = 1/100 and 1/10^6: ε = 10001/1000000
#guard (terms #[⟨{}, 1, "", 4, 16, #[], "", {}⟩, ⟨{}, 1, "", 4, 16, #[], "", {}⟩] #[.null, .null]
  #[{ units := 4, drawn := 1, delta := ⟨1, 100⟩, wrong := 0 }, { units := 4, drawn := 1, delta := ⟨1, 1000000⟩, wrong := 0 }]
  #[] 1 1 1).eps.toString == "10001/1000000"
-- eps over audits at δ = 1/10, 1/15 and 1/6 is 1/3; over none it is 0
#guard (terms #[⟨{}, 0, "", 1, 16, #[], "", {}⟩, ⟨{}, 0, "", 1, 16, #[], "", {}⟩,
    ⟨{}, 0, "", 1, 16, #[], "", {}⟩] #[.null, .null, .null]
  #[⟨1, 1, ⟨1, 10⟩, 0⟩, ⟨1, 1, ⟨1, 15⟩, 0⟩, ⟨1, 1, ⟨1, 6⟩, 0⟩] #[] 1 1 1).eps.toString == "1/3"
#guard (terms #[] #[] #[] #[] 1 1 1).eps.toString == "0"
-- the rows form over 1 link-window: 7 cells on one connection and 2 on another (3 + 2 bits, for the schedules and
-- again for the fail cells), an audit whose own terms are in none, and one scan of 20 units (T = 10) drawn at
-- δ = 1/15 with at most 3 wrong, 1,120 free inputs, ingress and egress instances committing 300 and 512 bits, and
-- jitter below 2 over its 10 steps: 1351 sets of at most 3 of 20 units
#guard atMost 20 3 == 1351 && clog2 1351 == 11
#guard (rowsTerms #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] ⟨10, 1120, 300, 512, 0⟩
  #[({ units := 20, drawn := 5, delta := ⟨1, 15⟩, wrong := 3 }, 0, 0)] #[7, 2] 2 1).K ==
  (3 + 2) + (3 + 2) + (1120 + 11 + 3 * 512 + 10)
-- every unit drawn: no wrong unit, so no faulty set and no faulty bits
#guard (rowsTerms #[] ⟨4, 0, 300, 512, 0⟩ #[({ units := 8, drawn := 8, delta := ⟨1, 100⟩, wrong := 0 }, 0, 0)] #[] 1 1
  ).K == 0
-- w63's class with the egress tie: T = 32, the scan's own free inputs none, every unit drawn, and its tied audit (the
-- slot entries' template) of 8 units, a slot at each of the 2 boundaries, whose free inputs are their
-- entries, 1,136 bits each: K is the tied audit's 9,088
#guard (rowsTerms #[{ units := 8, drawn := 8, delta := ⟨1, 100⟩, wrong := 0 }] ⟨32, 0, 300, 512, 0⟩
  #[({ units := 64, drawn := 64, delta := ⟨1, 100⟩, wrong := 0 }, 8 * 1136, 0)] #[] 1 1).K == 9088
-- eps over the scans' δ = 1/15 and 1/15 is 2/15; the audit's 1/10 is in it no more
#guard (rowsTerms #[⟨1, 1, ⟨1, 10⟩, 0⟩] ⟨0, 0, 0, 0, 0⟩ #[(⟨1, 1, ⟨1, 15⟩, 0⟩, 0, 0), (⟨1, 1, ⟨1, 15⟩, 0⟩, 0, 0)] #[]
  1 2).eps.toString == "2/15"
-- w63k4: T = 32, 4 lanes, every 16, so 8 slot entries of 112 bits a link-window; its tied audit drawn in part, with at
-- most 2 wrong units, which can set any entry: its link-window's term charges E_w = 896 bits beside its free inputs
#guard (rowsTerms #[{ units := 16, drawn := 9, delta := ⟨1, 100⟩, wrong := 2 }] ⟨32, 0, 300, 512, 8 * 112⟩
  #[({ units := 64, drawn := 64, delta := ⟨1, 100⟩, wrong := 0 }, 9088, 8 * 112)] #[] 1 1).K == 9088 + 896
-- a charged audit's δ = 1/10 is in eps beside its scan's 1/15: 1/6
#guard (rowsTerms #[⟨2, 1, ⟨1, 10⟩, 0⟩] ⟨0, 0, 0, 0, 0⟩ #[(⟨1, 1, ⟨1, 15⟩, 0⟩, 0, 0)] #[] 1 1 #[⟨2, 1, ⟨1, 10⟩, 0⟩]
  ).eps.toString == "1/6"
-- w63k4's tied audit of 16 proof blocks sampled at one (`subset:1`), with a bound of 2 incorrect ones: charged, its
-- link-window's β_w every one of the scan's E_w = 896 entry bits; sampled whole, uncharged and β_w = 0
#guard
  let scan : Scan := ⟨32, 0, 300, 512, 8 * 112⟩
  let p : Params := ⟨#[⟨{}, 0, "", 16, 0, #[], "", {}⟩], #[], 0, 1, "", Lms.deployed, .empty, .empty, "",
    .rows ⟨0, 0, 0, 0, 16, 0, 0, 4, 0, 0, 32⟩ .empty scan, 1⟩
  let at_ := fun (q : String) => Json.mkObj [("roots", Json.arr #[
    Json.mkObj [("port", .str q), ("root", .str "r"), ("domain", .str "d")]])]
  let c : Commitment := ⟨#[at_ (entriesPort "E{KO=4}" 0)], #[at_ (progName 0)], .empty⟩
  let one : Array (Found Facts) := #[.ok ⟨16, 1, ⟨1, 100⟩, 2⟩]
  let whole : Array (Found Facts) := #[.ok ⟨16, 16, ⟨1, 100⟩, 0⟩]
  tiedIn c.audits c.windows 0 && (chargedOf p c one 0).isSome && faultyEntries p c one scan (at_ (progName 0)) == 896 &&
    (chargedOf p c whole 0).isNone && faultyEntries p c whole scan (at_ (progName 0)) == 0
-- two templates of 2 units: A's x one u16 word, B's y two, cut in two rows, each wired from an A's o0, and A's x read
-- from the registered parameter w: of 96 input bits, 64 wired and 32 read, so none free where the registration binds
-- the record and holds w at its pin, 32 where its root for w is another, and all 96 where it binds another record
#guard
  let u16 : Flock.Program.Ty := .value 16
  let d := fun (k p : String) (t : Flock.Program.Ty) => ({
    key := k, id := k, prim := false, params := #[(p, t)], ret := u16, gates := 0, paramLeaves := 0, outLeaves := 0
    } : Flock.Program.Defn)
  let ts := #[(d "A" "x" u16, 2), (d "B" "y" (.array 2 u16), 2)]
  let reg := fun (c root : String) => Json.mkObj [("window", Json.mkObj [("circuit", .str c),
    ("registered_roots", Json.mkObj [("w", .str root)])])]
  let wire := r#"{"reader": "B", "port": "y", "writer": "A", "out": "o0", "at": 0, "radix": [2, 2], "base": 0,
    "step": [1, 0], "cut": [0, 2]}"#
  let read := r#"{"reader": "A", "port": "x", "param": "w", "at": 0, "radix": [2], "base": 0, "step": [1]}"#
  let record := fun (ingress : String) => Json.parse s!"\{\"templates\": [[\"A\", 2], [\"B\", 2]], \
    \"cuts\": \{\"B\": \{\"y\": 2}}, \"wires\": [{wire}], \"registered\": \{\"params\": \{\"w\": 2}, \"tables\": [], \
    \"reads\": [{read}], \"roots\": [], \"openings\": []}, \"ingress\": [{ingress}]}"
  let counted := fun (ingress : String) => (record ingress).toOption.bind fun r => (inputsOf r ts #[("w", "p")]).toOption
  match counted "" with
  | some i =>
    let a : Audit := ⟨i, 0, "", 4, 16, #[], "c", {}⟩
    i.total == 96 && i.declared == 64 && a.free (reg "c" "p") == 0 && a.free (reg "c" "q") == 32 &&
      a.free (reg "x" "p") == 96 &&
      (counted r#"{"reader": "B", "port": "y", "at": 1, "units": 1, "step": 0}"#).isNone &&
      (counted r#"{"reader": "A", "port": "x", "at": 1, "units": 1, "step": 0}"#).isNone &&
      ((record "").toOption.bind fun r => (inputsOf r ts.reverse #[]).toOption).isNone &&
      ((record "").toOption.bind fun r => (inputsOf r (ts.map fun (d, _) => (d, 3)) #[]).toOption).isNone
  | none => false
-- three units of a template E whose key (one word), root (32) and row (2) are its inputs, its root a table read of
-- table t over the registered parameter w and its row an opening of t at its key, verdict o1: of 1680 input bits,
-- 1632 read of w, so 48 free (the keys) where w is held at its pin, all 1680 where it isn't; and a table read of a
-- port that isn't 64 bytes, an opening of an undeclared table and a table past w's leaves are refused
#guard
  let e : Flock.Program.Defn :=
    { key := "E", id := "E", prim := false, params := #[("key", .value 16), ("root", .array 32 (.value 16)),
      ("row", .array 2 (.value 16))], ret := .tuple #[.value 16, .value 16], gates := 0, paramLeaves := 0,
      outLeaves := 0 }
  let root := r#"{"reader": "E", "port": "root", "table": "t", "of": "root", "at": 0, "units": 3}"#
  let opening := r#"{"reader": "E", "table": "t", "key": "key", "ports": ["row"], "verdict": "o1", "at": 0, "units": 3}"#
  let table := r#"{"name": "t", "param": "w", "offset": 0, "words": 2, "rows": 4}"#
  let record := fun (t r o : String) => Json.parse s!"\{\"templates\": [[\"E\", 3]], \"wires\": [], \"registered\": \
    \{\"params\": \{\"w\": 8}, \"tables\": [{t}], \"reads\": [], \"roots\": [{r}], \"openings\": [{o}]}}"
  let counted := fun (t r o : String) => (record t r o).toOption.bind fun j =>
    (inputsOf j #[(e, 3)] #[("w", "p")]).toOption
  let reg := fun (root : String) => Json.mkObj [("window", Json.mkObj [("circuit", .str "c"),
    ("registered_roots", Json.mkObj [("w", .str root)])])]
  match counted table root opening with
  | some i =>
    let a : Audit := ⟨i, 0, "", 3, 16, #[], "c", {}⟩
    i.total == 1680 && i.declared == 0 && a.free (reg "p") == 48 && a.free (reg "q") == 1680 &&
      (counted table (root.replace "\"root\", \"table\"" "\"row\", \"table\"") opening).isNone &&
      (counted table root (opening.replace "\"t\"" "\"u\"")).isNone &&
      (counted (table.replace "\"rows\": 4" "\"rows\": 5") root opening).isNone
  | none => false
-- a record that states no wires counts every input bit free
#guard
  let a : Flock.Program.Defn :=
    { key := "A", id := "A", prim := false, params := #[("x", .value 16)], ret := .value 16, gates := 0,
      paramLeaves := 0, outLeaves := 0 }
  match inputsOf (Json.mkObj [("templates", Json.arr #[])]) #[(a, 3)] #[] with
  | .ok i => i.total == 48 && i.declared == 0 && i.reads.isEmpty
  | .error _ => false
-- a unit's output port past o0, its index in ASCII digits without a leading zero; never o0, an input, or no unit's
#guard outputPort "block-3/o1" && outputPort "a/b/o10" && outputPort "/o2"
#guard !(outputPort "o1" || outputPort "u/o0" || outputPort "u/o01" || outputPort "u/o" || outputPort "u/O1" ||
  outputPort "u/o1x" || outputPort "u/o٣" || outputPort "u/prog" || outputPort "")
-- a template's entries port, a chunk's o0, whatever the template, the chunk's index in ASCII digits without a leading
-- zero; never another port, or one with no descriptor id
#guard entriesPortName (entriesPort "E{IDB=8,KO=2,NP=4,STOP=1}" 0) && entriesPortName "E{}/c0/o0" &&
  entriesPortName "E{KO=2}/c1/o0" && entriesPortName (entriesPort "F{V=256,C=16,KO=2,FIRST=false,STOP=[]}" 3) &&
  entriesPortName "F{KO=2}/c10/o0" && entriesPortName "G{V=512,STOP=[3]}/c2/o0" && entriesPortName "E/c0/o0" &&
  entriesPort "F{}" 3 == "F{}/c3/o0"
#guard !(entriesPortName "E{KO=2}/c01/o0" || entriesPortName "E{KO=2}/c0/o1" || entriesPortName "/c0/o0" ||
  entriesPortName "c0/o0" || entriesPortName "" || entriesPortName "E{KO=2}/c/o0" || entriesPortName "E{KO=2}c0/o0" ||
  entriesPortName "E{KO=2}/cx/o0" || entriesPortName "E{KO=2}/c٣/o0" || entriesPortName "F{KO=2}/c1/o1" ||
  entriesPortName "F{KO=2}/C1/o0" || entriesPortName "u/o0")
-- two lanes, T = 4, every 2: lane j's scan/prog<j> is chunk j's entries port, of 2 leaves, of one template of 4
-- members at the closure digest "d0" its circuit record "c" registers, a record its registration binds; not with the
-- lanes swapped, a chunk of other leaves, lane 1's tree missing, another template's chunk 1, the template at another
-- digest than the registered one, or the record not its registration's; and the scan's lane-1 tree alone ties the audit
#guard
  let s : Schedule := ⟨0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 4⟩
  let a : Audit := ⟨{}, 0, "", 4, 0, #[("E{KO=1}", 4, "d0")], "c", { templates := #[("E{KO=1}", "d0")] }⟩
  let port := fun (q root : String) (l : Nat) =>
    Json.mkObj [("port", .str q), ("leaves", toJson l), ("root", .str root), ("domain", .str "d")]
  let reg := fun (os : Array Json) => Json.mkObj [("roots", Json.arr os)]
  let chunksIn := fun (c : String) (l0 l1 : Nat) (k1 : String) =>
    Json.mkObj [("window", Json.mkObj [("circuit", .str c)]),
      ("roots", Json.arr #[port "E{KO=1}/c0/o0" "r0" l0, port (k1 ++ "/c1/o0") "r1" l1])]
  let chunks := chunksIn "c"
  let scan := fun (r0 r1 : String) => reg #[port "scan/prog0" r0 2, port "scan/prog1" r1 2]
  wired #[a] s #[chunks 2 2 "E{KO=1}"] (scan "r0" "r1") &&
    !wired #[a] s #[chunks 2 2 "E{KO=1}"] (scan "r1" "r0") &&
    !wired #[a] s #[chunks 3 1 "E{KO=1}"] (scan "r0" "r1") &&
    !wired #[a] s #[chunks 2 2 "E{KO=1}"] (reg #[port "scan/prog0" "r0" 2]) &&
    !wired #[a] s #[chunks 2 2 "E{KO=2}"] (scan "r0" "r1") &&
    !wired #[{ a with entries := #[("E{KO=1}", 4, "d1")] }] s #[chunks 2 2 "E{KO=1}"] (scan "r0" "r1") &&
    !wired #[a] s #[chunksIn "x" 2 2 "E{KO=1}"] (scan "r0" "r1") &&
    tiedAudits #[chunks 2 2 "E{KO=1}"] (reg #[port "scan/prog1" "r1" 2]) == #[0]
-- a lane's Programs' rows, the lane's index in ASCII digits without a leading zero
#guard progPort (progName 0) && progPort "scan/prog3" && progPort "scan/prog12" && progName 3 == "scan/prog3"
#guard !(progPort "scan/prog" || progPort "scan/prog01" || progPort "scan/prog/" || progPort "Scan/prog0" ||
  progPort "scan/prog0x" || progPort "prog0" || progPort "scan/state" || progPort "")

end NetworkCheck.Period
