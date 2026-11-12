import NetworkCheck.Check
import NetworkCheck.Circuit
import CoreProtocol.Profile
import SampledProofs.Bound
import Flock.Program
import Flock.Public
import Flock.RegKnown
import Flock.WindowRecord
import NciCheck.Desc

/-!
# The network certifier over a period, a core protocol: `verity/certifier/v0`

The network certifier, one of onsite's subprotocols (`Core.Onsite.components`). A period is one node's link over `m`
link-windows beside the period's compute audits (sampled proofs' one-stage, one answer per registered window of the
workload). Its link-windows come in one of two forms, which the parameters name:

* **the clock form**: N2's link-windows (`service.LinkWindows`, Selection `All`), each proved, both its ends certified
  and its clock certified;
* **the rows form**: the network certifier's own run, where it functions as an active warden (`warden_serve`): each
  execution phase's cells committed as rows (`warden_rows`), one registration a direction under one fixed schedule, its
  three records certified, and the scan's one-stage answer for the window (`scan_zk --warden-run`).

It accepts when every execution phase passes its form's check and every compute audit accepted; it then publishes `K`, the
bits a covert sender in the server can move through the network certifier in the period, as
`NetTiming.Guarantees.PeriodLeakAccepted` counts them. In the rows form each link-window's scan's Programs' row must be a
compute audit's slot entries (`wired`, below), and the state that audit's Programs read (its `SCAN_STATE`) the scan's
state, by root and domain, which the scan's units read and register as its transitions chain it from the empty state
(`stateTied`, `stateEmpty`), so a slot's entry is the one its Program computed on the requests the scan read.

**Params**, JSON with exactly these keys (and each nested object exactly its own), in the clock form

    {"audits": [{"program": descriptor, "steps": k}, …], "cells": [cap, …], "first": w₀, "jitter": Dj,
     "key": hex(I ‖ T[1]), "link": name, "lms": {"H": h, "N": n, "W": w, "toy": bool}, "program": hex,
     "session": hex(session), "sync_set": {"deltas": [δ, …], "rhos": [[num, den], …]}, "windows": m}

each compute audit's monitoring period (`steps`, `k`) and its Program, the descriptor (`verity-ir/descriptor/v1`) of
the window's population program, which must decode (`Flock.Program.decode`) with a template in its root: the audit's
proof blocks are the root's template instances, and each commits its template's parameters and return, its input gates
and the outputs it computes that other proof blocks or the client read, which can be wider than a proof unit's 16-bit
public output; each network
certifier connection's cell cap over the period (`cap_c`), the session's window
the period starts at (`w₀`: its link-windows are the session's `w₀ … w₀ + m − 1`, so one certifier session spans many
periods),
the observations' jitter bound `Dj ≥ 1`, the network certifier's LMS public key and parameters and the certifier session
(32 bytes) every certificate binds, which are the auditor's: its own record of its device (`Registration.Auditor.key`,
the key of each of its own devices), never a field of the evidence or the prover's copy of the run, so a certificate
under another key or in another session is refused,
the link (at most 65,535 UTF-8 bytes), the SHA-512 of N2's registered Program for
the link (the partition object's `program`, `LinkWindows.partition`), Σ_sync, and the period's `m ≥ 1` link-windows.
An audit's free inputs (`free`) are no parameter: the check counts them off its descriptor, the Program's input gates
(`inputGates`: its root's input nodes' members times their widths, of either binding time). That is
`#(freeInputs \ reg)` at `reg = ∅` with no ingress taken out, and at least it at every `reg` and horizon, so `K` takes
no registered value as pinned and `WindowGood` holds no registration values. An audit may instead carry its window's
circuit record, its wires' order and the verifier's pins,

    {"circuit": record, "order": order, "pins": {param: root, …}, "program": descriptor, "steps": k}

the record (`verity/window-circuit/v1`, `interface.circuit_record`) its registration's window binds by digest
(`window.circuit`), a certificate that its wires are acyclic (`verity/wire-order/v1`), and the registered parameters
whose roots the verifier fixed before the run, each its registration's window's registered root for that parameter
(`window.registered_roots`, `auditCheck`). Its free inputs are then `NetworkCheck.Circuit.check`'s count: the record
checked against the Program (its population, wires, declarations and order), and the Program's input gates less the
rows its wires fill and those its registered reads of a pinned parameter fill.
The rows form has, in place of `sync_set`,

    "circuit": hex,
    "scan": descriptor,
    "schedule": {"bucket_ms", "cell_bytes", "delay", "egress_cells", "every", "ingress_buckets", "ingress_cells",
                 "lanes", "queue", "slots", "window"}

the class circuit, the SHA-512 of the Program of the one window circuit the workload's class registers (the verifier's
own window-circuit record's `program`), which every tied compute audit runs (`held`);
the scan's Program, its descriptor (`verity-ir/descriptor/v1`, `scan_zk`'s `program.json`), and the network
certifier's one fixed schedule (`warden.WardenParams`: each a positive integer, B = `cell_bytes` past a cell's 9-byte
header, B, S = `lanes` · `slots` and T = `window` u32s, and `every`, a slot stride's buckets, dividing T, so each
link-window is T / `every` timesteps of `lanes` Program slots, `Schedule.programSlots` and `Schedule.windowPrograms`),
whose digest `WardenParams.digest` every registration binds; its `program` is the scan's registered Program, the
descriptor's SHA-512 (`partition_object.program_sha512`), and its `link` the run's (`warden_serve.LINK`), whose fails record is certified under the link `<link>-fails`. Its
`cells` must be the schedule's caps over the period (`Schedule.caps`): `egress_cells` for each of the
`lanes · m · window` connections the network certifier can carry in `m` link-windows, since it admits a connection at a
bucket's start only while fewer than `lanes` are live and serves it in that bucket. Any other list, an empty one
included, is refused, so `K`'s schedules and cells terms are never read off caps that miss a connection. Before
that, those `lanes · m · window` connections, the k-th at ordinal k + 1, must have distinct u32 ordinals, and a cell's
`B` bytes must be under 2^32 (`rowsWf`): a frame names its connection and its payload's length by u32s, so past 2^32
two frames would encode alike and the record would not read back (`NetTiming.RowsFormat.Wf` at `NetTiming.cellRows`).
The check refuses such a period too (`rowsWfCheck`), so whatever parameters it accepts, its frames read back. The
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
  `scan/prog<j>` entry names the root and domain of an output port's entry (`<unit>/o<j>`, j ≥ 1) or a `SlotEntry_v1`
  or `SelectEntry_v1` template's entries port's (`<descriptor id>/c<j>/o0`, `entriesPortName`) in the `roots` of an
  audit's registration, so a scan whose `prog` the prover committed alone is refused; and the answer is one-stage's
  for that registration's digest, accepted with one level of the scan's 2T units drawn `subset:k` or stratified, and
  its `bound`;
* in the rows form, every accepted link-window's scan is at one δ, `PeriodLeakAccepted`'s;
* audit `i`: its registration names its Program (the descriptor's SHA-512), and the answer is one-stage's for its
  registration's digest, accepted with one level of its Program's units drawn `subset:k` or stratified (a work law's
  draw is stratified), and its `bound`, at any sample: this is the consistency check (each audit's answer is its
  registration's, and in the rows form its outputs are the scan's `prog` and its `SCAN_STATE` the scan's state, below),
  and what a sample in part leaves unproved is in `K` (in the rows form, a tied audit's `faulty_entries` and δ, below);
* in the rows form, link-window `w`'s scan's Programs' rows are a compute audit's slot entries, a lane a chunk
  (`wired`): for each lane `j`, its `scan/prog<j>` entry names the root and domain of the entries port
  `<descriptor id>/c<j>/o0`, of T / `every` leaves, in the `roots` of one audit's registration, where the descriptor id
  is a template node's of that audit's own Program calling `SlotEntry_v1`, or `SelectEntry_v1` (the slot's last
  position, whose select and collector end in `SlotEntry_v1`, so its `o0` is the entry), a Definition the check pins by
  closure digest (`entryPins`: `slotEntryPins` and `selectEntryPins`; a template under either name with another body is
  refused), with one member a slot of the window (T / `every` · `lanes`). One-stage cuts a template into members of
  consecutive units, in order (`<descriptor id>/c<j>`), and a `SelectEntry_v1` template a member a lane, as its units
  read their lane's occupancy from `ScanSlots_v2` a wire a lane; so chunk `j` is lane `j`'s slots, its leaf `b` slot
  `j · T / every + b`'s entry, the row the scan's lane `j` reads at boundary `b` (`scanOf`), and each due, occupied
  slot's entry, read bit included, is the one its Program computed, not the prover's word, which with read 1 and no
  outputs would pass for a request no Program reads (`Properties.NetworkPeriod.RowsProgWired`);
* in the rows form, link-window `w`'s scan's state is the `SCAN_STATE` of an audit its `prog` is wired to
  (`stateTied`): the scan's registration has an entry of its `roots` whose port is `scan/state` or of its `registered`
  whose value is `verity/certifier/scan-state/v1`, every such entry at one root and domain, and so has a `wired`
  audit's registration, every such entry at that root and domain, with the entry of that value in its `window`'s
  `activations.values`, where a window circuit's registration commits its per-window values (`auditStateEntries`,
  which its driver, not one-stage's registration checks, holds its reads to); and the scan's registration states its
  units' reads of that state and opens its row 0, its `window`'s

      "reads": {"scan/state": {"o1": {"leaf": 1, "units": [0, T − 1]}, "sin": {"leaf": 1, "units": [1, T]}}, …},
      "empty": {"scan/state": {"words": W, "salt": hex, "path": hex}, …}

  (`stateRead`, a run a port: unit `lo + i` at leaf `l + i`), so the scan's ingress unit t (bucket t) reads `sin` as
  that root's row t and registers its `o1` as row t + 1, but for its last unit, whose `o1` no row holds (the egress
  tie's is the prover's one-leaf `scan/last`, registered before any coin), and each row past row 0 is the state the
  scan's transition computed from the row before it, the prover's word for none; and row 0, which no unit computes and only the
  Programs anchored at bucket 0 read, is the empty state, `W` zero u16 words, opened under its salt through its path
  to the state entry's root and domain (`stateEmpty`: the row and its salt are public, the other leaves hidden); and
  each audit tied to the scan (`tiedAudits`) whose circuit record reads `verity/certifier/scan-state/v1` reads it at
  its slots' anchors (`anchoredAt`): each such registered read is whole rows of its reader's uncut input port of `W`
  words, unit `l · nb + b` (lane `l`'s slot at boundary `b`, `nb` = T / `every`, `wired`'s chunk `l` leaf `b`)
  reading from leaf `b · every · W`, row `b · every`, of a parameter of the T rows, base 0, and no table is over it
  (`anchorRead`, read off the read's digits, a digit of radix 1 aside; base 0, as a scan is of the link-window's T
  buckets alone, never a group's, `scanOf`). So a slot's Program anchored at bucket a reads row a, the
  scan's state before a, and computes its entry on the requests the scan read, not on a state the prover chose or
  one the scan reached after a (`Properties.NetworkPeriod.RowsStateTied`, `Properties.NetworkPeriod.RowsStateAnchored`).
* in the rows form, phase `w`'s tied compute audits run the class circuit (`held`): each audit its scan's `prog`
  is an output of (`tiedAudits`) has a registration naming the parameters' `circuit` as its Program. A window circuit's
  own digest binds its window's run, so it is another each window; its Program is the class's, one for the period, and
  is what the audit's free inputs are read off. So no phase's tied audit is registered under a Program the prover
  chose for that phase (`Properties.NetworkPeriod.RowsCircuitHeld`), and, since the audit check holds the registered
  Program to the descriptor whose input gates its term counts (`auditCheck`), and an audit's circuit record to its
  registration's window circuit, those are the class circuit's.

**Public**, on acceptance: `K` and its terms, each a ceiling of `log2` of `PeriodLeakAccepted`'s factor:

    {"K": K, "audits": [{"delta", "drawn": k, "faulty_bits", "faulty_sets", "free", "jitter", "units": N,
                         "wrong_units": b}, …],
     "cells": Σ_c ⌈log2 (cap_c + 1)⌉, "eps": Σ_i δ_i, "schedules": ⌈log2 #Σ_sync^m⌉, "windows": m}

with `b` the audit's bound on faulty proof blocks at its δ (one-stage's `bound`, `Bound.boundOf`), `faulty_sets` =
`⌈log2 Σ_{j ≤ b} C(N, j)⌉` (every set of at most `b` proof blocks, which holds the sets a `subset:k` or a stratified
sample leaves possible; `drawn` is the sample's `k`, or its `Σ k` stratified),
`faulty_bits` = `b · bits`, with `bits` the most one of its Program's proof blocks commits (its template's parameters
and return, `paramBits` and `bitsOf`: the committed values `faultyBits` counts, a faulty proof block's input gates and
the outputs it computes that other proof blocks or the client read), `jitter` = `⌈log2 Dj^steps⌉`, and `K` the sum of the terms. The schedules term counts every assignment
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
(`stateTied`). So an incorrect proof block of any template, a `SlotEntry_v1` or `SelectEntry_v1` one setting its
slot's entry, a `ScanSlots_v2` one setting the `occ` and prompts its slots' Programs read, or any other its entries
read, changes at most those `E_w` bits; the scan reads an entry only where its own state has the slot due and occupied,
and refuses one there whose `read` is 0, so no `occ` the audit computes drops a request.

A link-window's term charges what the prover chose of its scan's inputs by the scan's view tie and its own draw.

The view tie: egress unit `t`'s `state`, the one parameter an egress instance reads (`scanOf`), is the view ingress
unit `t` returns. The egress rows are computed from those `state` rows alone, and each is an input gate of the scan's
Program, so until a scan's registration states the tie before the coins and the check reads it, as it reads the
state's chain (`stateRead`), nothing holds a `state` row to what the ingress computed. No registration states it and
the check reads none (`viewTieHeld`), so every term charges its scan's view rows (`view`: `Scan.view`, its `T` egress
instances' `state`, 32 · 800 = 25,600 bits at w63k4), and beside them:

* every unit of the scan is proved (`Facts.whole`: its draw takes each stratum whole and the strata hold all its
  units): nothing more of the scan's inputs. Neither its tied audits' free inputs nor its slot entries are charged:
  they reach the egress rows only through the view, every bit of which the term charges, whatever the tied audits'
  draws, so no `δ_i` enters `eps` for it.
* the scan is sampled in part. The term charges its tied audits' free inputs (`free`), and the check accepts a tied
  audit at any sample: sampled in part, the period reads the audit's draw at its own `δ_i` = 2^-64 (`tiedDelta`, fixed
  before any coin), where its at most `b` incorrect proof blocks can set any of its phase's entries, so that phase's
  term charges every entry bit (`faulty_entries`: `β_w ≤ 1[b ≥ 1] · E_w`) when `b ≥ 1`, and `δ_i` is in `eps`.

With the view tie held (`Scan.terms` at `viewTie`, which no accepted period's term is at), no term charges the view
rows, and a scan with every unit proved is charged its slot entries instead: its only input gates beside the certified
ingress rows, its own state and its view are its slot entries (`scanOf` refuses any other), so the term charges those
`E_w` bits once (`entries`), in place of its tied audits' free inputs and `faulty_entries` together. It takes no
minimum with the tied audits' own count: where that count is the smaller, the smaller figure would rest on the audits'
tie to the entries, which this term does not read.

The public:

    {"K": K, "audits": [{"delta", "drawn", "units", "wrong_units"}, …], "cells": c,
     "charged": [{"delta": δ_i, "drawn", "units", "wrong_units"}, …], "eps": m · δ + Σ δ_i,
     "held": ["circuit", "free"], "scan_bits": {"entries": E_w, "view": V_w}, "scans": [terms, …],
     "schedules": c, "windows": m}

* `audits` are the compute audits' facts as their answers give them: each at the δ sampled proofs reads its draw at, and
  its answer's `bound` there.
* `charged` are the audits drawn in part that are tied to a scan sampled in part (`partOf`), in the order of `audits`,
  each read at `δ_i` off the same draw
  (`Facts.readAt`: one-stage's `IntegrityProfile.bound` at `δ_i`, no second draw): `wrong_units` is `b` at `δ_i`, which
  `faulty_entries` reads. Sampled proofs' reading stays its own, at its own δ, so the composite's error
  (`Verity.SoundnessOf`, the union over the subprotocols) has each reading as its own term: `δ_i` in this protocol's
  `eps`, the answer's δ in sampled proofs'.

* `schedules` is `cells`, c = Σ_c ⌈log2 (cap_c + 1)⌉, as the statement multiplies `#Ss` and `∏_c (cap_c + 1)`. A
  schedule (`WardenSchedule`) is r and D, fixed, and each connection's admission bucket and lane length. The server
  sets a lane's length, when its stream ends, at most `cap_c` cells, and nothing else of it: the network certifier
  admits a queued connection at the first lane that frees, in the clients' arrival order, so the admissions are a
  function of the lengths, and `#Ss ≤ ∏_c (cap_c + 1)`. The second factor is the fail cell's position (`failCell`). A
  failed lane ends at its fail cell, so in a run the two coincide; `K` charges the product as the statement states it.
* `scan_bits` are the scan's Program's sizes a link-window, whichever a term charges: `entries`, `E_w`, its slot
  entries' bits (`Scan.entries`: `lanes · T / every` entries of its `prog` ports' type, 8 · 112 = 896 at w63k4), and
  `view`, `V_w`, its view rows' bits (`Scan.view`: its `T` egress instances' `state`).
* each scan's terms, from the scan's Program and its tied audits: `view`, `V_w`, since no view tie is held; `whole`,
  whether every unit of the scan is proved (`Facts.whole`); `entries`, `E_w` where every unit of the scan is proved
  and its view tie is held, so 0 in every accepted period's; `free`, where the scan is sampled in part, the free
  inputs the check counts off the
  descriptor (`inputGates`, or with its circuit record `NetworkCheck.Circuit.check`'s count, its reads of the scan's
  state held where its `SCAN_STATE` is the scan's state in every phase tied to it, `chargeOf`) of each audit its `prog`
  is an output of, summed (`tiedAudits`, each once), and with every unit proved the scan's own alone (the scan's own is
  0, as it registers the rows, its `prog` is
  those audits' output, bucket 0 starts from the empty state, constants of `IngressScanFirst`, and its view rows are
  their own term, `view`: the egress tie
  moves the free term from the scan's `prog` to the tied audit's inputs), `steps` T, `faulty_bits`
  `b · max(b_ing, b_eg)` with `b_ing` an ingress instance's parameter and return bits (its row, state and slot
  entries, the more of bucket 0's and a later bucket's, and its verdict u16, state and view) and `b_eg` an egress
  instance's (its parameters, and its u16 and row): each instance's committed values (`faultyBits`), its input gates
  and the outputs it computes that other proof blocks or the client read. `b` is its answer's `bound` (over its sample's strata, `countBound`), 0 when every
  unit is drawn, `drawn` its draw's `Σ k`, and `faulty_sets` every set of at most `b` of its 2T units, which holds the
  sets a stratified draw leaves possible too. `faulty_entries` is `E_w` when the scan is sampled in part and an audit
  its `prog` is tied to is drawn in part with a bound at `δ_i` of at least one wrong unit (`faultyEntries`), else 0, so
  a term charges the entries at most once. `tied` names the audits its `prog` is an output of, whose free inputs its
  `free` sums where it is sampled in part, by their index in `audits` (`tiedAudits`, ascending).
* `eps` is the scans' δ summed, `m · δ`, and `δ_i` for each charged audit (`chargedOf`), whose bound `faulty_entries`
  reads, so none for an audit tied only to scans proved whole: the δ part of the union over the monitoring period's
  windows that `CommOfPeriodLeakAccepted` states, each
  window at δ plus its knowledge error and its collision errors. Each δ is a ratio, summed exactly in lowest terms.
* `held` names, sorted, the checks of a tied audit the rows form holds (`heldNames`): `circuit`, its Program is the
  class circuit (`heldChecks`), and `free`, the free inputs its phase's term counts are that Program's input
  gates, counted off its registered descriptor (`inputGates`), or, with the circuit record its registration's window
  binds, those less the rows its wires and its pinned parameters' registered reads fill
  (`NetworkCheck.Circuit.check`), and, where its `SCAN_STATE` is the scan's state in every phase tied to it, the rows
  its reads of the scan's state fill (`chargeOf`), never a parameter. The check refuses a period where
  either fails, so every accepted rows-form public names both, and a reader of `K` charges a tied audit its counted
  free inputs only for a public that does.

**Profile**: the link and `m`.

What it doesn't check: the proof's public row against the certificates' digests and the output word, which N2's
verifier of record checks inside the answer (`LinkWindows.verify`, with C-Flock's Lean verifier on the `--zk` proof);
the clock form's link-window registration's domain; in the rows form, the records' certificates' digests against the
rows (each signs its salted record, witness) and the scan's statements against its registration (lane `j`'s slot
entries' input node the value it registers as `scan/prog<j>`, and each unit's `sin` and `o1` read at the leaf its
`reads` state), which one-stage's verifier of record checks inside the answer, and that the network certifier holds
each lane to `egress_cells` cells with at most `lanes` connections live,
which the caps read (`Schedule.caps`; `trusted-host`); that a tied audit's `SelectEntry_v1` units read their `occ` from
its `ScanSlots_v2` units, and that the units its circuit record reads its `SCAN_STATE` at are those (the check reads
none of an audit's wires, only its root templates' ids and closure digests and its record's registered reads of the
scan's state, `anchored`);
and that the audits' answers are the sampled-proofs component's own,
which the composition's theorem is to bind.

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

/-- A compute audit's layout terms, its free inputs (its Program's input gates, `inputGates`, or with its circuit
record `NetworkCheck.Circuit.check`'s count) and its monitoring period, and what its Program gives: its SHA-512, its
units (its root's template instances), the most bits one of them commits, and its slot entries' templates (`entries`:
each root template node calling a `SlotEntry_v1` or `SelectEntry_v1` Definition, `entryIds`, by its descriptor id,
members and closure digest, `slotEntryDigest`); with a circuit record, its digest (`NetworkCheck.Circuit.digest`) and the
pins the count held (`record`), which its registration must bind (`auditCheck`), and the record as read against its
Program (`circuit`, `NetworkCheck.Circuit.recordOf`), whose reads of the scan's state the rows form holds to its slots'
anchors (`anchored`); and its free inputs with those reads held too (`stateFree`, `free` without a record), which the
rows form charges where its `SCAN_STATE` is the scan's state (`chargeOf`). -/
structure Audit where
  free : Nat
  steps : Nat
  program : String
  units : Nat
  bits : Nat
  entries : Array (String × Nat × String)
  record : Option (String × List (String × String))
  circuit : Option NetworkCheck.Circuit.Stated := none
  stateFree : Nat := free

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

/-- The rows form's caps over `windows` link-windows (`WardenParams.caps`): `egressCells` for each of the
`lanes · windows · window` connections the network certifier can carry, as at most `lanes` are live in a bucket and
each is live in its admission bucket. -/
def Schedule.caps (s : Schedule) (windows : Nat) : Array Nat :=
  Array.replicate (s.lanes * (windows * s.window)) s.egressCells

/-- **The rows form's frames read back** (`NetTiming.RowsFormat.Wf` at the network certifier's frames,
`NetTiming.cellRows`, `warden.py`'s `encode_frame`: kind u8, ordinal u32, payload length u32, payload, zero fill): a
cell's `B` bytes are under 2^32, so its payload's length is a u32, and the `lanes · windows · window` connections of a
period of `windows` link-windows, the k-th at ordinal k + 1 (0 is a refusal's), have distinct u32 ordinals, so no two
frames encode alike. -/
def rowsWf (s : Schedule) (windows : Nat) : Bool :=
  s.cellBytes < 2 ^ 32 && s.lanes * (windows * s.window) < 2 ^ 32

def rowsWfRefusal (s : Schedule) (windows : Nat) : String :=
  s!"rows format: a frame's ordinal and payload length are u32s, so the lanes · windows · window = \
    {s.lanes * (windows * s.window)} connections and a cell's {s.cellBytes} bytes must each be under 2^32, or two \
    frames could encode alike (rowsWf)"

/-- The scan's terms, read off its Program: its monitoring period (`T`, the window's buckets), its own free inputs
(none: it registers the rows, its `prog` is a compute audit's output, `progTied`, whose free inputs a link-window's
term counts, `tiedAudits`, and its view rows are their own term, `view`), the bits one ingress and one egress instance
commit (its parameters, its input gates, and its return: the outputs other units read or the client receives, and its
verdict), its slot entries' bits (`E_w`: `lanes · T / every` entries of its `prog` ports' type), all a tied audit's
wrong units can set, and its view rows' bits (its `T` egress instances' `state` parameters, each an input node's
leaves, `scanOf`: every input gate an egress instance reads). -/
structure Scan where
  steps : Nat
  free : Nat
  ingress : Nat
  egress : Nat
  entries : Nat
  view : Nat

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
  /-- The rows form's class circuit, its window circuit's Program's SHA-512, which every tied compute audit runs
  (`held`); none in the clock form, which ties no audit. -/
  circuit : Option String
  form : Form
  windows : Nat

/-- The one-stage window records of the compute audits and of the link-windows, in order. -/
structure Commitment where
  audits : Array Json
  windows : Array Json
  bytes : ByteArray

structure BucketSchedule where
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
  layout : BucketSchedule
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

/-- What an accepted audit establishes: its proof blocks, its sample, its δ, its bound on faulty proof blocks at that δ,
and the draw that bound is read off (`draw`), which bounds them at any other δ (`Facts.readAt`). -/
structure Facts where
  units : Nat
  drawn : Nat
  delta : Core.Ratio
  wrong : Nat
  /-- The sample's design; `subset 0`, none drawn, where none is read, at which every unit may be faulty at any δ. -/
  draw : Core.SamplingDesign := .subset 0

/-- The same draw's bound at `δ`, one-stage's `IntegrityProfile.bound` (`SampledProofs.Bound.boundOf`): its faulty proof
blocks at `δ`, read off the draw the facts were established by, with no second draw. -/
def Facts.readAt (f : Facts) (δ : Core.Ratio) : Facts :=
  { f with delta := δ, wrong := SampledProofs.Bound.boundOf f.units f.draw δ }

/-- Every unit is drawn, none sampled: the draw, read as its strata (`SampledProofs.Bound.strataOf`), draws each
stratum whole, and the strata hold every unit. Read off the draw itself, never off `drawn`, a sum a stratum drawn past
its size could bring to `units` beside a stratum drawn in part. No draw (`subset 0` over at least one unit), a Bernoulli
draw and a draw in part are not whole. -/
def Facts.whole (f : Facts) : Bool :=
  match SampledProofs.Bound.strataOf f.units f.draw with
  | some ss => ss.all (fun s => s.size ≤ s.k) && f.units ≤ ss.foldl (fun n s => n + s.size) 0
  | none => false

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
  ["audits", "cells", "circuit", "first", "jitter", "key", "link", "lms", "program", "scan", "schedule", "session",
    "windows"]

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

/-- The slot's last position (`SelectEntry_v1`, pinned by `selectEntryPins`): one unit a due, occupied slot's Program,
whose select and collector end in `SlotEntry_v1`, so its `o0` is the slot's entry. -/
def selectEntryId : String := "SelectEntry_v1"

/-- The Definitions whose `o0` is a slot's entry: `SlotEntry_v1` and `SelectEntry_v1`. -/
def entryIds : List String := [slotEntryId, selectEntryId]

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
another body has neither digest, so its entries are no slot entries (`wired`). These are the rows a window circuit's
slot entries take as a template of their own; its `SlotEntry_v1:4` row (KO 4, 32 prompt ids) is no pin, since at KO 4
a slot's entry is a `SelectEntry_v1` unit's (`selectEntryPins`). A new body of `SlotEntry_v1`, or a binding a window
circuit takes as its slot entries' template, is pinned here. -/
def slotEntryPins : List String :=
  ["8c8a033047ec49110114023204a9c265720ae9d1b63963c51a98e14f0314ebd3\
    7b741b90d6a374aeb681a1a5fe364580b68ff60cf5514fdf57fdaff88b36de30",
   "2559f1c8e260aac6e8755c317cde07f990fcd3619dd394010d367a3b78156c2f\
    24b44567891a9ea2efb2f8a6830c9f9c957c14dc65041f8dd77bfd8d50c5b7ff"]

#guard slotEntryPins.all (·.length == 128)

/-- The slot's last position's Definitions the check takes, by closure digest (`slotEntryDigest`, which covers the
`SlotEntry_v1` it calls): `SelectEntry_v1` at `SelectEntry_v1{V=256,C=16,KO=2,FIRST=false,IDB=17,NP=4,STOP=[]}`,
`SelectEntry_v1{V=128256,C=16,KO=4,FIRST=false,IDB=17,NP=32,STOP=[128001]}` and
`SelectEntry_v1{V=128256,C=16,KO=64,FIRST=false,IDB=17,NP=1024,STOP=[128001]}`, in that order, each calling the
`SlotEntry_v1` that writes the scan's read bit: `io_shell.SelectEntry_v1` at its circuit-check rows. A new binding or
body is pinned here. -/
def selectEntryPins : List String :=
  ["f817250e0ac1d5028ea992bbe0c1ab12925bc1b709f020fdc54a57a0ebc0c9dc\
    bd6513d2bf72aa243c499833d1173fb14236e85bb27b4d03de88df93344007d2",
   "236d569741e3fe2026aaa65be2077b4da82d495a5b35322428051e4e07ca556b\
    b44d1c5f59f2aef22d7c4c5b563c3819569add04ed3d2d4d57563ad471cf71c4",
   "5460098589051bb756c5d7781e1727495952c296826b29315ec269b492ca3cea\
    bc394ee273a2dae92ffb8eef31d8d22e83208a9e280dd80e39be9073dfc7a68a"]

#guard selectEntryPins.all (·.length == 128)

/-- The entries' Definitions the check takes, by closure digest: `slotEntryPins` and `selectEntryPins`. -/
def entryPins : List String := slotEntryPins ++ selectEntryPins

/-- The value a window circuit registers the scan's state rows as (`ingress_scan`'s `SCAN_STATE`), row a the state
before bucket a: the entry of an audit's `registered`, or of the scan's, whose value it is, or an audit's entry of it in
its `window`'s `activations.values` (`auditStateEntries`). -/
def stateValue : String := "verity/certifier/scan-state/v1"

/-- A compute audit's layout terms and its Program, from the Program's descriptor (`Flock.Program.decode` of its
canonical bytes): its units are its root's template instances, each committing its template's parameters and return
(`paramBits`, `bitsOf`); a root with no
template is refused. Its free inputs are counted here and never read from the parameters: the Program's input gates
(`inputGates`), or, where the audit carries its window's circuit record, the order of its wires and the verifier's
pins, what `NetworkCheck.Circuit.check` counts free after checking the record against the Program (its input gates
less the rows its wires and its pinned parameters' registered reads fill), and the record as read against the Program
(`NetworkCheck.Circuit.checked`, `Audit.circuit`). A pin of `stateValue` is refused: the scan's state is no value the
verifier fixes before the run, so an audit's reads of it count free (`free`). Its count with them held too
(`stateFree`, `NetworkCheck.Circuit.countsOf` at `stateValue` beside the pins) the rows form charges a tied audit only
where its `SCAN_STATE` is the scan's state in every phase tied to it, the pin the scan's registration's own
(`chargeOf`), and which it reads at its slots' anchors (`anchored`). Its slot entries' templates are its root's
template nodes whose Definition is `SlotEntry_v1` or `SelectEntry_v1` (`entryIds`), each with its closure digest
(`slotEntryDigest`). -/
def auditOf (a : Json) (ctx : String) : V Audit := do
  let forms := [["program", "steps"], ["circuit", "order", "pins", "program", "steps"]]
  unless (match a with | .obj _ => true | _ => false) && forms.contains (keysOf a) do
    throw s!"{ctx}: an object whose keys are {forms[0]!} or {forms[1]!}, not {keysOf a}"
  let pj ← Flock.J.get a "program" ctx
  let prog ← match Flock.Program.decode (← Flock.canon pj).toUTF8 with
    | .ok prog => pure prog
    | .error e => throw s!"{ctx}: program: {e}"
  let some root := prog.defs[prog.root]? | throw s!"{ctx}: program: no root"
  let templates := root.body.nodes.filterMap fun x => match prog.defs[x.fn]? with
    | some d => if d.prim then none else some (x.members, paramBits d.params + bitsOf d.ret)
    | none => none
  if templates.isEmpty then throw s!"{ctx}: program: no template"
  let slots := root.body.nodes.filterMap fun x => match prog.defs[x.fn]? with
    | some d => if !d.prim && entryIds.contains d.id then some (d.key, x.members) else none
    | none => none
  let entries ← slots.mapM fun (k, n) => do
    pure (k, n, ← (slotEntryDigest pj k).mapError (s!"{ctx}: " ++ ·))
  let (free, stateFree, record, circuit) ← match a.getObjVal? "circuit" with
    | .ok c => do
      let why := (s!"{ctx}: circuit: " ++ ·)
      let pins ← Flock.J.get a "pins" ctx
      let held ← (NetworkCheck.Circuit.pinsOf pins).mapError why
      if held.any (·.1 == stateValue) then
        throw (why s!"a pin of {stateValue}: the scan's state is no value the verifier fixes before the run, so its \
          reads count free")
      let (r, s) ← (NetworkCheck.Circuit.checked prog c (← Flock.J.get a "order" ctx)).mapError why
      let count := fun (held : List String) => (NetworkCheck.Circuit.countsOf r s held).free
      pure (count (held.map (·.1)), count (stateValue :: held.map (·.1)),
        some (← (NetworkCheck.Circuit.digest c).mapError why, held), some r)
    | .error _ => pure (inputGates prog, inputGates prog, none, none)
  pure { free, steps := ← Flock.J.nat a "steps" ctx, program := Flock.hexOf prog.sha512,
         units := templates.foldl (· + ·.1) 0, bits := templates.foldl (fun b t => max b t.2) 0, entries, record,
         circuit, stateFree }

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
leaf the period check's `wired` holds to chunk `j` of an audit's `SlotEntry_v1` or `SelectEntry_v1` entries port;
their bits are the scan's `entries`. -/
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
  pure { steps := T, free := 0,
         ingress := max (paramBits df.params) ((later.map fun (_, d) => paramBits d.params).getD 0) + bitsOf df.ret,
         egress := paramBits de.params + bitsOf de.ret, entries := s.lanes * nb * bitsOf ety,
         view := T * paramBits de.params }

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
  let circuit ← if keysOf j == paramsKeys then pure none else do
    let d ← Flock.J.str j "circuit" "the object"
    unless isDigest d do throw "circuit: not a SHA-512 digest (128 lowercase hex digits)"
    pure (some d)
  let form ← if keysOf j == paramsKeys then do pure (Form.clock (← syncSetOf (← Flock.J.get j "sync_set" "the object")))
    else do
      let s ← scheduleOf (← Flock.J.get j "schedule" "the object")
      pure (Form.rows s (← s.digest) (← scanOf (← Flock.J.get j "scan" "the object") program s))
  let windows ← Flock.J.nat j "windows" "the object"
  unless 1 ≤ windows do throw "windows: a period has at least one link-window"
  if let .rows s _ _ := form then
    unless rowsWf s windows do throw (rowsWfRefusal s windows)
    unless cells == s.caps windows do
      throw s!"cells: the rows form's caps are egress_cells = {s.egressCells} for each of the lanes · windows · \
        window = {s.lanes * (windows * s.window)} connections the schedule can carry (Schedule.caps), not these \
        {cells.size}"
  pure { audits, cells, first, jitter, link, lms, key, session, program, circuit, form, windows }

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
  unless prof.levels.map (fun l => (l.name, l.size, l.draw)) == #[(unitLevel, 1, some (Core.SamplingDesign.subset 1))] &&
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
      r.layout == ({ buckets := T, slots := S, connectionSlots := s.slots, firstBucket := w * T } : BucketSchedule) do
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

/-- A `SlotEntry_v1` or `SelectEntry_v1` template's entry rows of chunk `j`, the public word of its `j`-th member's
units: `<descriptor id>/c<j>/o0`. One-stage cuts a template into members of consecutive units, in order, named
`<descriptor id>/c<j>`, at the ends of every wire its units read; `SelectEntry_v1`'s read their lane's occupancy from
`ScanSlots_v2` a wire a lane, so its members are a lane each. -/
def entriesPort (key : String) (j : Nat) : String := s!"{key}/c{j}/o0"

/-- Whether a port is some `SlotEntry_v1` or `SelectEntry_v1` template's entries port of some chunk (`entriesPort`,
`entryIds`), read over its characters: its `o0` is the entries the slots' Programs compute, so it ties its audit as an
output port does. -/
def entriesPortName (port : String) : Bool :=
  let cs := port.toList
  let r := cs.reverse.drop 3
  entryIds.any (fun id => (id ++ "{").toList.isPrefixOf cs) && "/o0".toList.isSuffixOf cs &&
    "c/}".toList.isPrefixOf (r.dropWhile Char.isDigit) && indexChars (r.takeWhile Char.isDigit).reverse

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

/-- The scan's Programs' rows are a compute audit's slot entries, a lane a chunk: for each lane `j`, the root and domain
of the scan's registration's `scan/prog<j>` entry (`progName`) are those of chunk `j`'s entries port
(`entriesPort`), of `T / every` leaves, of a `SlotEntry_v1` or `SelectEntry_v1` template of that audit's own Program
(`Audit.entries`) that is a pinned Definition (its closure digest one of `entryPins`, so its whole body, the
`SlotEntry_v1` a `SelectEntry_v1` ends in included, is a pinned one), with one member a slot of the window
(`Schedule.windowPrograms`). Its members are cut in order, so chunk `j` is units `[j · T / every, (j + 1) · T / every)`
and its leaf `b` is slot `j · T / every + b`'s entry: bucket `t`'s port `prog<j>`, which reads lane `j`'s row
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
  (audits.zip regs).any fun (a, r) => a.entries.any fun (k, n, d) => n == s.windowPrograms &&
    entryPins.contains d && (List.range s.programSlots).all fun j =>
      (prog j).any fun e => match e.getObjValD "root", e.getObjValD "domain" with
        | .str root, .str dom => (entries r "roots").any fun o =>
            named "port" (entriesPort k j) o && leaves (s.window / s.every) o && named "root" root o &&
              named "domain" dom o
        | _, _ => false

/-- The scan's state in a registration: the entry of its `roots` whose port is `scan/state`. -/
def stateName : String := "scan/state"

/-- A registration's state entries: those of its `roots` whose port is `stateName` and of its `registered` whose value
is `stateValue`. -/
def stateEntries (reg : Json) : Array Json :=
  let entries := fun (k : String) => match reg.getObjValD k with | .arr xs => xs | _ => #[]
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  (entries "roots").filter (named "port" stateName) ++ (entries "registered").filter (named "value" stateValue)

/-- A compute audit's state entries: its `stateEntries` and, where its `window` commits its per-window values
(`window.activations.values`, fixed with the record before the sample), its entry of `stateValue` when that entry's
`value` is `stateValue`. That entry is the root the audit's reads of `stateValue` open at only where its verifier is given
those reads at its window's entries: one-stage's registration checks hold `roots` and `registered` but read nothing of
`window`, so it is the audit's driver, which pins each per-window value from `activations.values`, that holds it. A
scan's state entries are its `stateEntries` alone. -/
def auditStateEntries (reg : Json) : Array Json :=
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  stateEntries reg ++ #[(((reg.getObjValD "window").getObjValD "activations").getObjValD "values").getObjValD
    stateValue].filter (named "value" stateValue)

-- a registration whose window's activations hold the state: an audit's state entry, no scan's; one whose activations
-- entry names another value: none
#guard
  let act := fun (v : String) => Json.mkObj [("window", Json.mkObj [("activations", Json.mkObj [("program", "c"),
    ("values", Json.mkObj [(stateValue, Json.mkObj [("value", v), ("root", "s"), ("domain", "d")])])])])]
  (auditStateEntries (act stateValue)).size == 1 && (stateEntries (act stateValue)).isEmpty &&
    (auditStateEntries (act "w")).isEmpty

/-- Whether `j` is the run `{"leaf": l, "units": [lo, hi]}`: units `lo` to `hi − 1`, unit `lo + i` at leaf `l + i`. Read
field by field, which the kernel evaluates (`Json`'s `==` it doesn't). -/
def isRun (j : Json) (lo hi l : Nat) : Bool :=
  let nat := fun (x : Json) (n : Nat) => match x.getNat? with
    | .ok m => m == n
    | .error _ => false
  nat (j.getObjValD "leaf") l && match j.getObjValD "units" with
    | .arr xs => match xs.toList with
      | [a, b] => nat a lo && nat b hi
      | _ => false
    | _ => false

/-- The scan's registration states its units' reads of its state as the scan's transitions chain them: its `window`'s
`reads` of `stateName` has `sin` the run of units 1 to T − 1 from leaf 1 and `o1` the run of units 0 to T − 2 from leaf 1
(`isRun`), so ingress unit `t` reads `sin` at leaf `t`, S_{t−1}, and registers its `o1`, S_t, at leaf `t + 1`. Unit `t` is
bucket `t`: one-stage numbers the scan's units in its root's order, bucket 0's first and then the later ingress
buckets' (`scanOf`). Leaf 0, the empty state, no unit reads, and the last ingress unit's `o1` no leaf holds. One run a
port, so the check reads the description, never a unit. -/
def stateRead (s : Schedule) (reg : Json) : Bool :=
  let r := ((reg.getObjValD "window").getObjValD "reads").getObjValD stateName
  isRun (r.getObjValD "sin") 1 s.window 1 && isRun (r.getObjValD "o1") 0 (s.window - 1) 1

#guard stateRead ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 3⟩ (Json.mkObj [("window", Json.mkObj [("reads", Json.mkObj
  [(stateName, Json.mkObj [("o1", Json.mkObj [("leaf", toJson 1), ("units", toJson [0, 2])]),
                           ("sin", Json.mkObj [("leaf", toJson 1), ("units", toJson [1, 3])])])])])]) &&
  !stateRead ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 3⟩ (Json.mkObj [("window", Json.mkObj [("reads", Json.mkObj
  [(stateName, Json.mkObj [("o1", Json.mkObj [("leaf", toJson 0), ("units", toJson [0, 2])]),
                           ("sin", Json.mkObj [("leaf", toJson 1), ("units", toJson [1, 3])])])])])])

/-- The scan's state is the state its slots' Programs read, read where the scan's transitions chain it: the scan's
registration states its state reads as the chain (`stateRead`: unit `t`'s `sin` at leaf `t`, its `o1` at leaf `t + 1`, but
the last unit's `o1`, which no leaf holds), has a state entry (`stateEntries`), every one at one root and domain, and an
audit `wired` to the scan's `prog` has one (`auditStateEntries`), every one at that root and domain. So the `SCAN_STATE`
that audit's Programs read is the chain of the scan's transitions, each row past row 0 the one the unit before
registered. -/
def stateTied (audits : Array Audit) (s : Schedule) (regs : Array Json) (reg : Json) : Bool :=
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  stateRead s reg && match (stateEntries reg)[0]? with
  | some e => match e.getObjValD "root", e.getObjValD "domain" with
    | .str root, .str dom =>
      let atRoot := fun (o : Json) => named "root" root o && named "domain" dom o
      (stateEntries reg).all atRoot && (audits.zip regs).any fun (a, r) =>
        wired #[a] s #[r] reg && !(auditStateEntries r).isEmpty && (auditStateEntries r).all atRoot
    | _, _ => false
  | none => false

/-- The scan's registration opens its state's row 0 to the empty state: its `window`'s
`"empty": {"scan/state": {"words": W, "salt": hex, "path": hex}}` is leaf 0's opening, `W` zero u16 words (`W` at least
1) under a salt of `HmRow.SALT_BYTES` bytes, whose hm96 leaf (`Registered.knownPayload`, as the verifier registers a
value at known content), framed at 0 under the state entry's domain, climbs `path`, one sibling a level of the T
leaves', to that entry's root. Row 0 is the state before bucket 0, which no unit computes and only the Programs
anchored at bucket 0 read; a row's digest binds its length, so a reader of another width doesn't open it. The row is
public, the empty state, and so is its salt; every other leaf stays hidden. -/
def stateEmpty (s : Schedule) (reg : Json) : Bool :=
  let o := ((reg.getObjValD "window").getObjValD "empty").getObjValD stateName
  let hex := fun (k : String) (j : Json) => match j.getObjValD k with
    | .str x => (Flock.unhex x).toOption
    | _ => none
  let depth := Flock.log2ceil s.window
  match (stateEntries reg)[0]?, (o.getObjValD "words").getNat? with
  | some e, .ok w => match hex "salt" o, hex "path" o, hex "root" e, hex "domain" e with
    | some salt, some path, some root, some dom =>
      decide (0 < w) && decide (salt.size = Flock.HmRow.SALT_BYTES) && decide (path.size = 64 * depth) &&
        decide (Flock.FV3.Sha512.climb dom (Flock.FV3.Sha512.leafAs Flock.HmRow.ROW_SCHEMA_V2 dom 0
          (Flock.Registered.knownPayload (Array.replicate w 0) salt)) 0 depth path = root)
    | _, _, _, _ => false
  | _, _ => false

/-- The digits, slowest first, of the read a tied audit's slots make of the scan's state at their anchors, `W` the
reader port's words and `nb` = T / `every` the boundaries: unit `l · nb + b`, lane `l`'s slot at boundary `b`
(lane-major, as `wired`'s chunk `l` leaf `b`), reads from leaf `b · every · W`, row `b · every`, the state before its
anchor's bucket; a digit of radix 1, which moves no unit, left out. -/
def anchorDigits (s : Schedule) (W : Nat) : List (Nat × Int) :=
  [(s.lanes, 0), (s.window / s.every, ((s.every * W : Nat) : Int))].filter (·.1 != 1)

/-- Whether registered read `x` of the scan's state, in a record of wiring `c` and parameters `params`, is its slots'
anchors' read: whole rows of its reader's uncut input port of `W` words, units `[0, lanes · nb)` from base 0, its
digits less those of radix 1 the anchors' (`anchorDigits`), of a parameter of `T · W` leaves, the scan's `T` state
rows. Read off the read's description, never a unit. -/
def anchorRead (s : Schedule) (c : NetworkCheck.Circuit.Wiring) (params : List (String × Nat))
    (x : NetworkCheck.Circuit.Read) : Bool :=
  match NetworkCheck.Circuit.portOf c x.pat.reader x.pat.port false with
  | some (W, k) => k == 1 && x.pat.start == 0 && x.pat.cut0 == 0 && x.pat.cuts == 1 && x.pat.base == 0 &&
      (x.pat.radix.zip x.pat.step).filter (·.1 != 1) == anchorDigits s W &&
      params.lookup stateValue == some (s.window * W)
  | none => false

/-- A compute audit reads the scan's state at its slots' anchors: with its circuit record, each of the record's
registered reads of `stateValue` is the anchors' (`anchorRead`) and no table is over it; with none, it states no read,
and its Program's inputs count free. -/
def anchored (s : Schedule) (a : Audit) : Bool := match a.circuit with
  | some r => (r.registered.reads.filter (·.param == stateValue)).all (anchorRead s r.wiring r.registered.params) &&
      !r.registered.tables.any (·.param == stateValue)
  | none => true

/-- The audits tied to a link-window's scan (`tiedAudits`) read its state at their slots' anchors (`anchored`). -/
def anchoredAt (audits : Array Audit) (s : Schedule) (regs : Array Json) (reg : Json) : Bool :=
  (tiedAudits regs reg).all fun i => (audits[i]?.map (anchored s)).getD true

-- two lanes, T = 2, every 1: a record whose template S of 4 units reads the scan's state at its port of 3 words, unit
-- l · 2 + b from row b, is anchored, with a digit of radix 1 between, a read of another value, or no record; not from
-- base 3, at rows 2b, with the lanes and boundaries swapped, from unit 1, over a parameter of 8 leaves, a read cut in
-- two, or with a table over the state
#guard
  let s : Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 2⟩
  let c : NetworkCheck.Circuit.Wiring :=
    { templates := [("S", { units := 4, rows := [("state", ⟨3, 1⟩)], outs := [("o0", 1)] })], cuts := [], wires := [] }
  let rd := fun (v : String) (start : Nat) (base : Int) (radix : List Nat) (step : List Int) (cut0 cuts : Nat) =>
    ({ param := v, pat := { reader := "S", port := "state", writer := v, out := "", start, radix, base, step, cut0,
                            cuts } } : NetworkCheck.Circuit.Read)
  let audit := fun (x : NetworkCheck.Circuit.Read) (n : Nat) (tables : List NetworkCheck.Circuit.Table) =>
    ({ free := 0, steps := 0, program := "", units := 4, bits := 0, entries := #[], record := none,
       circuit := some { templates := [("S", 4)], cap := 0, wiring := c,
                         registered := { params := [(x.param, n)], tables, reads := [x] } } } : Audit)
  let ok := rd stateValue 0 0 [2, 2] [0, 3] 0 1
  anchored s (audit ok 6 []) && anchored s (audit (rd stateValue 0 0 [2, 1, 2] [0, 5, 3] 0 1) 6 []) &&
    anchored s (audit (rd "w" 0 3 [2, 2] [0, 6] 0 1) 6 []) &&
    anchored s { (audit ok 6 []) with circuit := none } &&
    !anchored s (audit (rd stateValue 0 3 [2, 2] [0, 3] 0 1) 6 []) &&
    !anchored s (audit (rd stateValue 0 0 [2, 2] [0, 6] 0 1) 6 []) &&
    !anchored s (audit (rd stateValue 0 0 [2, 2] [3, 0] 0 1) 6 []) &&
    !anchored s (audit (rd stateValue 1 0 [2, 2] [0, 3] 0 1) 6 []) &&
    !anchored s (audit ok 8 []) &&
    !anchored s (audit (rd stateValue 0 0 [2, 2, 2] [0, 3, 0] 0 2) 6 []) &&
    !anchored s (audit ok 6 [{ name := "t", param := stateValue, offset := 0, words := 3, rows := 2 }])

/-- Audit `i` is tied to a link-window's scan: some link-window's registration's `prog` is its output (`tiedAudits`). -/
def tiedIn (audits windows : Array Json) (i : Nat) : Bool :=
  windows.any fun reg => (tiedAudits audits reg).contains i

/-- The audits the form ties to a link-window's scan: the rows form's `tiedIn`, and none in the clock form. -/
def tiedOf : Form → Commitment → Nat → Bool
  | .clock _, _ => fun _ => false
  | .rows .., c => tiedIn c.audits c.windows

/-- Audit `i`'s `SCAN_STATE` is the scan's state in every phase tied to it: its registration has a state entry
(`auditStateEntries`), and in each link-window whose scan's `prog` is its output (`tiedAudits`) the scan's first state entry
has a root and a domain, each a string, at which every one of the audit's is. So the root its reads of the scan's state
are held to is the scan's registration's own in every term that charges it, never a parameter's. -/
def stateHeldIn (audits windows : Array Json) (i : Nat) : Bool :=
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  match audits[i]? with
  | some r => !(auditStateEntries r).isEmpty && windows.all fun reg => !(tiedAudits audits reg).contains i ||
      match (stateEntries reg)[0]? with
      | some e => match e.getObjValD "root", e.getObjValD "domain" with
        | .str root, .str dom => (auditStateEntries r).all fun o => named "root" root o && named "domain" dom o
        | _, _ => false
      | none => false
  | none => false

/-- What the rows form charges tied audit `i` in a link-window's term: its free inputs with its reads of the scan's
state held (`Audit.stateFree`) where its `SCAN_STATE` is the scan's state in every phase tied to it (`stateHeldIn`),
else its free inputs (`Audit.free`); none for no audit. -/
def chargeOf (p : Params) (audits windows : Array Json) (i : Nat) : Nat :=
  (p.audits[i]?.map fun a => if stateHeldIn audits windows i then a.stateFree else a.free).getD 0

-- an audit tied by its port `b/o1` to a phase whose scan's state is at root `s`: held with its `SCAN_STATE` there, in
-- one phase or two, beside a phase not tied to it, and charged its `stateFree`; not with no state entry, at root `t`,
-- beside one at `t`, or tied to a second phase whose state is at `t`, and charged its `free`; held too with its
-- `SCAN_STATE` in its window's activations alone at `s`, not there at `t` or beside a `registered` one at `t`
#guard
  let st := fun (root : String) => Json.mkObj [("value", stateValue), ("root", root), ("domain", "d")]
  let audit := fun (states : List Json) => Json.mkObj [("registered", Json.arr states.toArray),
    ("roots", Json.arr #[Json.mkObj [("port", "b/o1"), ("root", "p"), ("domain", "q")]])]
  let acted := fun (root : String) (states : List Json) => Json.mkObj [("registered", Json.arr states.toArray),
    ("roots", Json.arr #[Json.mkObj [("port", "b/o1"), ("root", "p"), ("domain", "q")]]),
    ("window", Json.mkObj [("activations", Json.mkObj [("values", Json.mkObj [(stateValue, st root)])])])]
  let window := fun (prog root : String) => Json.mkObj [("roots", Json.arr #[
    Json.mkObj [("port", progName 0), ("root", prog), ("domain", "q")],
    Json.mkObj [("port", stateName), ("root", root), ("domain", "d")]])]
  let a : Audit :=
    { free := 9, steps := 0, program := "", units := 1, bits := 0, entries := #[], record := none, stateFree := 4 }
  let p : Params := ⟨#[a], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", none,
    .rows ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1⟩ .empty ⟨0, 0, 0, 0, 0, 0⟩, 1⟩
  stateHeldIn #[audit [st "s"]] #[window "p" "s"] 0 &&
    stateHeldIn #[audit [st "s"]] #[window "p" "s", window "p" "s"] 0 &&
    stateHeldIn #[audit [st "s"]] #[window "p" "s", window "x" "t"] 0 &&
    chargeOf p #[audit [st "s"]] #[window "p" "s"] 0 == 4 &&
    !stateHeldIn #[audit []] #[window "p" "s"] 0 && !stateHeldIn #[audit [st "t"]] #[window "p" "s"] 0 &&
    !stateHeldIn #[audit [st "s", st "t"]] #[window "p" "s"] 0 &&
    !stateHeldIn #[audit [st "s"]] #[window "p" "s", window "p" "t"] 0 &&
    chargeOf p #[audit [st "s"]] #[window "p" "s", window "p" "t"] 0 == 9 && chargeOf p #[] #[] 1 == 0 &&
    stateHeldIn #[acted "s" []] #[window "p" "s"] 0 && chargeOf p #[acted "s" []] #[window "p" "s"] 0 == 4 &&
    !stateHeldIn #[acted "t" []] #[window "p" "s"] 0 && !stateHeldIn #[acted "s" [st "t"]] #[window "p" "s"] 0

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
  | .ok b => pure { units := l.size, drawn, delta := prof.delta, wrong := b, draw := l.draw.getD (.subset 0) }
  | .error e => refuse s!"{what}: {e}"

/-- A link-window's tied audits' free inputs, its scan's registration `reg` among the commitment's `windows`: what the
form charges each audit its scan's `prog` is an output of (`tiedAudits`, each once), `chargeOf`, summed. -/
def tiedFree (p : Params) (audits windows : Array Json) (reg : Json) : Nat :=
  (tiedAudits audits reg).foldl (fun n i => n + chargeOf p audits windows i) 0

/-- Link-window `w`'s findings in the rows form, in the order the module docstring gives, and on acceptance its scan's
facts, which its term charges beside its tied audits' free inputs (`tiedFree`, `rowsPhase`). -/
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

/-- Phase `q` of the rows form, its registration and window and its index: its findings (`rowsWindowCheck`), and on
acceptance its scan's facts and its tied audits' charges (`tiedFree`), which its term sums. -/
def rowsPhase (p : Params) (s : Schedule) (sd : ByteArray) (c : Commitment) (q : (Json × Window) × Nat) :
    Found (Option (Facts × Nat)) :=
  (fun f => some (f, tiedFree p c.audits c.windows q.1.1)) <$>
    rowsWindowCheck p s sd c.audits (p.first + q.2) q.1.1 q.1.2

/-- Whether pin `(p, root)` is registration `reg`'s window's registered root for parameter `p` (its
`registered_roots`, `interface.held_params`: the hex compared without case). -/
def pinHeld (reg : Json) (pin : String × String) : Bool :=
  match ((reg.getObjValD "window").getObjValD "registered_roots").getObjValD pin.1 with
  | .str r => r.toLower == pin.2.toLower
  | _ => false

/-- Audit `i`'s findings, and on acceptance its facts: its registration names its Program and the answer's proof
blocks are the Program's, at any sample; with a circuit record (`Audit.record`), its registration's window binds that
record (`window.circuit`, its digest) and records each pin the count held as its parameter's registered root
(`pinHeld`); a tied audit sampled in part is charged in the terms of its link-windows whose scans are sampled in part
(`chargedOf`, `faultyEntries`). -/
def auditCheck (i : Nat) (a : Audit) (reg : Json) (x : Core.Answer) : Found Facts := do
  let what := s!"audit {i}"
  unless isStr (reg.getObjValD "program") a.program do refuse s!"{what}: registration: not its Program"
  if let some (d, pins) := a.record then
    unless isStr ((reg.getObjValD "window").getObjValD "circuit") d do
      refuse s!"{what}: registration: its window binds another circuit record than the one its free inputs are \
        counted on"
    if let some (p, _) := pins.find? (!pinHeld reg ·) then
      refuse s!"{what}: registration: the pin of {p} is not its window's registered root for {p}, so the count would \
        hold the reads of a value the prover chose"
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
      and domains, j < {s.programSlots}, are not one audit's {slotEntryId} or {selectEntryId} entries ports \
      (<descriptor id>/c<j>/o0, {s.window / s.every} leaves each) of {s.windowPrograms} members, one a slot, whose \
      template is a pinned {slotEntryId} or {selectEntryId} (its closure digest one of entryPins), so a due, \
      occupied slot's read bit is the prover's word: reuse chunk j's port of the pinned template as the scan's \
      prog<j>")

/-- The rows form's wiring findings, one a link-window, in order (`wiredCheck`); none in the clock form. -/
def wiredChecks (p : Params) (c : Commitment) : Array (Option (Core.Verdict × String)) := match p.form with
  | .rows s .. => c.windows.zipIdx.map fun (reg, i) => wiredCheck p s c.audits (p.first + i) reg
  | .clock _ => #[]

/-- Link-window `w`'s state finding in the rows form: a refusal unless its scan's state is the `SCAN_STATE` of an audit
its `prog` is wired to, read where the scan's transitions chain it (`stateTied`), its row 0 the empty state
(`stateEmpty`), and each tied audit's circuit record reading it at its slots' anchors (`anchoredAt`). With the `prog`
wired, each slot's entry is the one its Program computed, and a Program reads the requests from its own registered
`SCAN_STATE`; unless that is the scan's state, each of its rows past row 0 the one the scan's unit before registered
(`stateRead`) and row 0 the empty state, it computes its entry on a state the prover chose; and unless its slot reads
the row before its anchor's bucket, it computes its entry on a state the scan reached after it, past the requests it
admitted there. -/
def stateCheck (p : Params) (s : Schedule) (audits : Array Json) (w : Nat) (reg : Json) :
    Option (Core.Verdict × String) :=
  if stateTied p.audits s audits reg then
    if stateEmpty s reg then
      if anchoredAt p.audits s audits reg then none else
        let nb := s.window / s.every
        let i := ((tiedAudits audits reg).find? fun i => !(p.audits[i]?.map (anchored s)).getD true).getD 0
        some (.reject, s!"window {w}: tied audit {i} reads the scan's state off its slots' anchors: a read of \
          {stateValue} in its circuit record is not whole rows of its reader's uncut input port at units \
          [0, {s.lanes * nb}), unit l·{nb} + b (lane l's slot at boundary b) reading row b·{s.every} of a \
          {s.window}-row parameter (at 0, base 0, radix [{s.lanes}, {nb}], step [0, {s.every}·W], W the port's \
          words, a digit of radix 1 aside), or a table is over it, so a slot's Program can read a state the scan \
          reached after its anchor (anchored): read each slot's state at its anchor's row")
    else
      some (.reject, s!"window {w}: the scan's state's row 0 is not the empty state: its registration's window.empty \
        of {stateName} is no opening of leaf 0 to zero u16 words under its salt at its state entry's root and domain \
        (stateEmpty), so the Programs anchored at bucket 0 can read a state the prover chose: open leaf 0, the empty \
        state, with its salt and path")
  else
    some (.reject, if stateRead s reg then s!"window {w}: the scan's state is no wired audit's SCAN_STATE: its \
      {stateName} entries' root and domain are not every {stateValue} entry's of an audit whose slot entries are its \
      prog, so a slot's Program can compute its entry on a state the prover chose: register the scan's state tree as \
      that audit's SCAN_STATE"
    else s!"window {w}: the scan's state reads are not its transitions' chain: its registration's window.reads of \
      {stateName} are not sin \{\"leaf\": 1, \"units\": [1, {s.window}]} and o1 \{\"leaf\": 1, \"units\": [0, \
      {s.window - 1}]} (stateRead), so a unit could read a state no unit registered: register unit t's sin at leaf t \
      and its o1 at leaf t + 1")

/-- The rows form's state findings, one a link-window, in order (`stateCheck`); none in the clock form. -/
def stateChecks (p : Params) (c : Commitment) : Array (Option (Core.Verdict × String)) := match p.form with
  | .rows s .. => c.windows.zipIdx.map fun (reg, i) => stateCheck p s c.audits (p.first + i) reg
  | .clock _ => #[]

/-- Whether a compute audit's registration names `circuit`, a class circuit's Program's SHA-512, as its Program: its
window circuit is that class's. None with no class circuit. -/
def runsCircuit (circuit : Option String) (reg : Json) : Bool := match circuit, reg.getObjValD "program" with
  | some d, .str x => x == d
  | _, _ => false

/-- The class circuit is held at a phase: each audit its scan's `prog` is an output of (`tiedAudits`) runs it
(`runsCircuit`), so the phase's tied audit runs the one window circuit the class registers, and, with the audit
check holding its registration to its descriptor (`auditCheck`), the free inputs its term counts are that Program's. -/
def held (circuit : Option String) (audits : Array Json) (reg : Json) : Bool :=
  (tiedAudits audits reg).all fun i => (audits[i]?.map (runsCircuit circuit)).getD false

/-- Phase `w`'s class circuit finding in the rows form: a refusal unless the audits its scan's `prog` is an output
of run the class circuit (`held`). A window circuit's own digest binds its window's run, so only its Program can be
held; without the hold a phase's audit could run a Program the prover chose for that phase, and its term would
count that Program's free inputs, not the class circuit's. -/
def heldCheck (p : Params) (audits : Array Json) (w : Nat) (reg : Json) : Option (Core.Verdict × String) :=
  if held p.circuit audits reg then none else
    some (.reject, s!"window {w}: the tied audit's window circuit is not the registered class circuit: an audit the \
      scan's prog is an output of (tiedAudits) is registered under another Program than the parameters' circuit, so \
      its phase's term would count the free inputs of a Program the prover chose for that phase: register every \
      phase's audit under the class's window circuit")

/-- The rows form's class circuit findings, one a phase, in order (`heldCheck`); none in the clock form. -/
def heldChecks (p : Params) (c : Commitment) : Array (Option (Core.Verdict × String)) := match p.form with
  | .rows .. => c.windows.zipIdx.map fun (reg, i) => heldCheck p c.audits (p.first + i) reg
  | .clock _ => #[]

/-- The rows form's frames finding (`rowsWf`): `paramsOf` refuses such a period before its caps, and the check refuses
it too, so whatever parameters it accepts, its rows form's frames read back. None in the clock form. -/
def rowsWfCheck (p : Params) : Option (Core.Verdict × String) := match p.form with
  | .rows s _ _ => if rowsWf s p.windows then none else some (.reject, rowsWfRefusal s p.windows)
  | .clock _ => none

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
  /-- The slot entries' bits a tied audit's wrong units can set (`β_w`, the rows form's, where its scan is sampled in
  part; 0 in the clock form and where `entries` charges them). -/
  faultyEntries : Nat := 0
  /-- The scan's slot entries' bits (`Scan.entries`, `E_w`), charged once where every unit of the scan is proved
  (`Facts.whole`) and its view tie is held, in place of its tied audits' free inputs and `faultyEntries` together; 0
  otherwise and in the clock form. -/
  entries : Nat := 0
  /-- The scan's view rows' bits (`Scan.view`), charged where its view tie is not held (`viewTieHeld`); 0 where it is
  and in the clock form. -/
  view : Nat := 0

def AuditTerms.bits (t : AuditTerms) : Nat :=
  t.free + t.faultySets + t.faultyBits + t.jitter + t.faultyEntries + t.entries + t.view

/-- A compute audit's terms in the clock form, from its layout terms, its facts and the jitter bound: each of its at
most `b` faulty proof blocks commits at most its Program's widest proof block's bits. -/
def AuditTerms.of (a : Audit) (f : Facts) (jitter : Nat) : AuditTerms :=
  { facts := f, free := a.free, faultySets := clog2 (atMost f.units f.wrong), faultyBits := f.wrong * a.bits,
    jitter := clog2 (jitter ^ a.steps) }

/-- Whether the check holds a scan's view tie: egress unit `t`'s `state` is the view ingress unit `t` returns, stated
in the scan's registration, so fixed before any coin, and read by the check as `stateRead` reads the state's chain. No
registration states it and no step of the check reads it, so it is held for no scan, and every link-window's term
charges its scan's view rows (`Scan.terms`). -/
def viewTieHeld : Bool := false

/-- A link-window's scan's terms in the rows form, from the scan's terms, its facts, its tied audits' free inputs
(`tied`: the egress renders their outputs, so the bits the prover chose of them are the link-window's free inputs), the
slot entries' bits its tied audits' wrong units can set (`faulty`, `faultyEntries`), the jitter bound and whether the
scan's view tie is held (`viewTie`, the check's `viewTieHeld`): each of its at most `b` wrong units commits at most the
wider instance's bits.

Where the view tie is not held, an egress instance's `state` is an input gate nothing ties to what the ingress
computed, so the term charges the scan's view rows (`view` = `Scan.view`), every input gate its egress instances read.
Where every unit of the scan is proved (`Facts.whole`) it charges nothing more of the scan's inputs: neither `tied`
and `faulty` nor the slot entries, which reach the egress rows only through the view.

Where the view tie is held and every unit of the scan is proved, the term charges the scan's slot entries once
(`entries` = `Scan.entries`) in place of `tied` and `faulty` together, and reads neither: the entries are every input
gate of the scan the prover chooses. It takes no minimum of the two charges, so it can exceed `tied + faulty` where
the tied audits count fewer free inputs than the entries' bits.

A scan sampled in part is charged `tied` and `faulty`, beside its view rows where the view tie is not held. -/
def Scan.terms (s : Scan) (f : Facts) (tied faulty jitter : Nat) (viewTie : Bool := viewTieHeld) : AuditTerms :=
  { facts := f, free := s.free + (if f.whole then 0 else tied), faultySets := clog2 (atMost f.units f.wrong),
    faultyBits := f.wrong * max s.ingress s.egress, jitter := clog2 (jitter ^ s.steps),
    faultyEntries := if f.whole then 0 else faulty, entries := if f.whole && viewTie then s.entries else 0,
    view := if viewTie then 0 else s.view }

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
  /-- The facts of the rows form's audits drawn in part and tied to a scan sampled in part, each read at `tiedDelta`
  (`chargedOf`), whose bounds `faulty_entries` reads, so whose `δ_i` is in `eps`. -/
  charged : Array Facts := #[]
  /-- Each of the rows form's link-windows' tied audits, by index (`tiedAudits`), the audits its scan's `free` sums
  where the scan is sampled in part; none in the clock form. -/
  tied : Array (Array Nat) := #[]
  /-- The rows form's scan's slot entries' bits and view rows' bits a link-window (`Scan.entries`, `Scan.view`), as
  its Program gives them, whichever a term charges. -/
  scanBits : Nat × Nat := (0, 0)

def Public.K (x : Public) : Nat := x.schedules + x.cells + (x.units.map (·.bits)).foldl (· + ·) 0

/-- `Σ_i δ_i` over the units and the charged audits, exactly, in lowest terms: the union bound on `K` failing. -/
def Public.eps (x : Public) : Core.Ratio :=
  let (n, d) := (x.units.map (·.facts) ++ x.charged).foldl
    (fun (n, d) f => (n * f.delta.den + f.delta.num * d, d * f.delta.den)) (0, 1)
  ⟨n / Nat.gcd n d, d / Nat.gcd n d⟩

/-- `Σ_c ⌈log2 (cap_c + 1)⌉`. -/
def capsTerm (cells : Array Nat) : Nat := (cells.map fun cap => clog2 (cap + 1)).foldl (· + ·) 0

/-- The bound's terms from the audits' layout terms, Programs' bits and facts, the cells' caps, the jitter bound,
`#Σ_sync` and the execution phases. -/
def terms (audits : Array Audit) (facts : Array Facts) (cells : Array Nat) (jitter syncs windows : Nat) : Public :=
  { audits := facts
    cells := capsTerm cells
    units := (audits.zip facts).map fun (a, f) => AuditTerms.of a f jitter
    schedules := clog2 (syncs ^ windows)
    windows }

/-- The rows form's terms: the cells' as the clock form's, the schedules' the same count (each connection's lane
length, one of `cap_c + 1`, fixes its admission and the lanes it frees), and each link-window's scan's from the scan's
terms, its facts, its tied audits' free inputs and the entry bits their wrong units can set (`scans`, a triple a
link-window; `Scan.terms` charges a scan whose every unit is proved its view rows, or with its view tie held
(`viewTie`) its entries, in place of the last two). The compute audits' own terms are in none: a tied audit's free
inputs and wrong units are in its link-window's, and the δ of one drawn in part and tied to a scan sampled in part
(`charged`) in `eps`. -/
def rowsTerms (facts : Array Facts) (scan : Scan) (scans : Array (Facts × Nat × Nat)) (cells : Array Nat)
    (jitter windows : Nat) (charged : Array Facts := #[]) (viewTie : Bool := viewTieHeld) : Public :=
  { audits := facts, cells := capsTerm cells, units := scans.map fun (f, t, b) => scan.terms f t b jitter viewTie,
    rows := true, schedules := capsTerm cells, windows, charged, scanBits := (scan.entries, scan.view) }

/-- `δ_i` = 2^-64, the δ at which the rows form reads a tied audit's draw (`chargedOf`): a constant of the check, so
fixed before any coin, and its own reading of the draw, beside the δ sampled proofs reads the same draw at (the audit's
answer's, in `audits`). -/
def tiedDelta : Core.Ratio := ⟨1, 2 ^ 64⟩

/-- Audit `i`'s facts when the rows form charges it: accepted, tied to a link-window's scan among `c`'s (`tiedOf`)
and drawn in part, read at `tiedDelta` (`Facts.readAt`), so its at most `b` wrong units at `δ_i` are in its phases'
terms (`faultyEntries`) and `δ_i` in `eps`. `rowsPublic` reads it over the link-windows whose scans are sampled in part
(`partOf`). -/
def chargedOf (p : Params) (c : Commitment) (as : Array (Found Facts)) (i : Nat) : Option Facts :=
  match as[i]? with
  | some (.ok f) => if tiedOf p.form c i && f.drawn < f.units then some (f.readAt tiedDelta) else none
  | _ => none

/-- A link-window's `β_w`: every slot entry bit of its scan (`Scan.entries`, `E_w`) when an audit its `prog` is tied
to (`tiedAudits`) is charged (`chargedOf`) with a bound of at least one wrong unit, which can set any of the entries,
else 0. -/
def faultyEntries (p : Params) (c : Commitment) (as : Array (Found Facts)) (scan : Scan) (reg : Json) : Nat :=
  if (tiedAudits c.audits reg).any fun i => (chargedOf p c as i).any (1 ≤ ·.wrong) then scan.entries
  else 0

/-- The commitment with only the link-windows whose scans are sampled in part (`Facts.whole` false), `scans` the
link-windows' scans' facts in order: the windows whose terms read a tied audit's bound (`faultyEntries`). A scan whose
every unit is proved is charged its view rows or its entries whatever its tied audits' draws (`Scan.terms`), so reads
none. -/
def partOf (c : Commitment) (scans : Array (Facts × Nat)) : Commitment :=
  { c with windows := (scans.zip c.windows).filterMap fun ((f, _), reg) => if f.whole then none else some reg }

/-- The rows form's public on acceptance: `rowsTerms` over the accepted audits' facts and each link-window's scan's
facts, tied audits' free inputs and `β_w` (`faultyEntries`), of which `Scan.terms` charges a scan whose every unit is
proved its view rows instead, or with its view tie held (`viewTie`) its entries; the audits charged are those tied to
a scan sampled in part (`chargedOf` over `partOf`), so no `δ_i` enters `eps` for an audit tied only to scans proved
whole; with each link-window's tied audits (`tiedAudits`). -/
def rowsPublic (p : Params) (c : Commitment) (as : Array (Found Facts)) (scan : Scan) (scans : Array (Facts × Nat))
    (viewTie : Bool := viewTieHeld) : Public :=
  { rowsTerms (as.filterMap (·.toOption)) scan
      ((scans.zip c.windows).map fun ((f, t), reg) => (f, t, faultyEntries p (partOf c scans) as scan reg)) p.cells
      p.jitter p.windows ((Array.range as.size).filterMap (chargedOf p (partOf c scans) as)) viewTie with
    tied := (scans.zip c.windows).map fun (_, reg) => tiedAudits c.audits reg }

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
      | .rows s sd _ => pairs.map (rowsPhase p s sd c)
    let scans := ws.filterMap (·.toOption.join)
    let oneDelta : Option (Core.Verdict × String) := scans[0]?.bind fun (f0, _) =>
      (scans.find? (·.1.delta != f0.delta)).map fun (f, _) =>
        (.reject, s!"a scan's δ {f.delta.toString} is not the period's, {f0.delta.toString}")
    let as := auditChecks p c e
    match firstFinding (ws.toList.map findingOf ++ [oneDelta, rowsWfCheck p] ++ as.toList.map findingOf ++
        (wiredChecks p c).toList ++ (stateChecks p c).toList ++ (heldChecks p c).toList) with
    | some (.reject, r) => .reject r
    | some (_, r) => .undecided r
    | none =>
      .accept (match p.form with
          | .clock ss =>
            terms p.audits (as.filterMap (·.toOption)) p.cells p.jitter (ss.deltas.length * ss.rhos.length) p.windows
          | .rows _ _ scan => rowsPublic p c as scan scans)
        { link := p.link, windows := p.windows }

/-- The terms' JSON, each with its `entries`, its `faulty_entries`, its tied audits (`tied`, the `i`th's at `i`), its
`view` and whether every unit of it is proved (`whole`, `Facts.whole`) when `entries` (the rows form's scans). -/
def termsJson (ts : Array AuditTerms) (entries : Bool := false) (tied : Array (Array Nat) := #[]) : Json :=
  .arr <| ts.zipIdx.map fun (t, i) => Json.mkObj ([
    ("delta", .str t.facts.delta.toString), ("drawn", toJson t.facts.drawn)] ++
    (if entries then [("entries", toJson t.entries)] else []) ++ [("faulty_bits", toJson t.faultyBits)] ++
    (if entries then [("faulty_entries", toJson t.faultyEntries)] else []) ++ [
    ("faulty_sets", toJson t.faultySets), ("free", toJson t.free), ("jitter", toJson t.jitter)] ++
    (if entries then [("tied", toJson (tied[i]?.getD #[]))] else []) ++ [("units", toJson t.facts.units)] ++
    (if entries then [("view", toJson t.view), ("whole", toJson t.facts.whole)] else []) ++ [
    ("wrong_units", toJson t.facts.wrong)])

def factsJson (fs : Array Facts) : Json :=
  .arr <| fs.map fun f => Json.mkObj [("delta", .str f.delta.toString), ("drawn", toJson f.drawn),
    ("units", toJson f.units), ("wrong_units", toJson f.wrong)]

/-- The checks of a tied audit the rows form holds, by name, sorted: its Program is the class circuit (`heldChecks`),
and its free inputs are its Program's input gates (`inputGates`), less, with its circuit record, the rows its wires and
pinned registered reads fill (`NetworkCheck.Circuit.check`) and, at the scan's state, its reads of it (`chargeOf`).
The rows form's public names them. -/
def heldNames : Array String := #["circuit", "free"]

def publicJson (x : Public) : Json :=
  Json.mkObj ([
    ("K", toJson x.K), ("audits", if x.rows then factsJson x.audits else termsJson x.units), ("cells", toJson x.cells),
    ("eps", .str x.eps.toString), ("schedules", toJson x.schedules), ("windows", toJson x.windows)] ++
    (if x.rows then
      [("charged", factsJson x.charged), ("held", toJson heldNames),
       ("scan_bits", Json.mkObj [("entries", toJson x.scanBits.1), ("view", toJson x.scanBits.2)]),
       ("scans", termsJson x.units true x.tied)]
    else []))

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
#guard (terms #[⟨0, 3, "", 10, 16, #[], none, none, 0⟩]
    #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] #[7, 0] 2 2 3).K == 3 + 3 + (0 + 9 + 64 + 3)
-- the same audit with a unit committing 40 bits: its 4 wrong units carry 160 bits, not 16 · 4
#guard (terms #[⟨0, 3, "", 10, 40, #[], none, none, 0⟩]
    #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] #[7, 0] 2 2 3).K == 3 + 3 + (0 + 9 + 160 + 3)
-- two audits at δ = 1/100 and 1/10^6: ε = 10001/1000000
#guard (terms #[⟨0, 1, "", 4, 16, #[], none, none, 0⟩, ⟨0, 1, "", 4, 16, #[], none, none, 0⟩]
  #[{ units := 4, drawn := 1, delta := ⟨1, 100⟩, wrong := 0 }, { units := 4, drawn := 1, delta := ⟨1, 1000000⟩, wrong := 0 }]
  #[] 1 1 1).eps.toString == "10001/1000000"
-- eps over audits at δ = 1/10, 1/15 and 1/6 is 1/3; over none it is 0
#guard (terms #[⟨0, 0, "", 1, 16, #[], none, none, 0⟩, ⟨0, 0, "", 1, 16, #[], none, none, 0⟩,
  ⟨0, 0, "", 1, 16, #[], none, none, 0⟩]
  #[⟨1, 1, ⟨1, 10⟩, 0, .subset 1⟩, ⟨1, 1, ⟨1, 15⟩, 0, .subset 1⟩, ⟨1, 1, ⟨1, 6⟩, 0, .subset 1⟩] #[] 1 1 1
  ).eps.toString == "1/3"
#guard (terms #[] #[] #[] 1 1 1).eps.toString == "0"
-- the rows form over 1 link-window: 7 cells on one connection and 2 on another (3 + 2 bits, for the schedules and
-- again for the fail cells), an audit whose own terms are in none, and one scan of 20 units (T = 10) drawn at
-- δ = 1/15 with at most 3 wrong, 1,120 free inputs, ingress and egress instances committing 300 and 512 bits, and
-- jitter below 2 over its 10 steps: 1351 sets of at most 3 of 20 units
#guard atMost 20 3 == 1351 && clog2 1351 == 11
#guard (rowsTerms #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] ⟨10, 1120, 300, 512, 0, 0⟩
  #[({ units := 20, drawn := 5, delta := ⟨1, 15⟩, wrong := 3 }, 0, 0)] #[7, 2] 2 1).K ==
  (3 + 2) + (3 + 2) + (1120 + 11 + 3 * 512 + 10)
-- every unit proved: a `subset` draw of all its units, or strata each drawn whole that hold them all; not a draw in
-- part, a stratum drawn past its size beside one drawn in part (its Σ k the units'), strata short of the units, a
-- Bernoulli draw, or facts with no draw
#guard (⟨64, 64, tiedDelta, 0, .subset 64⟩ : Facts).whole &&
  (⟨64, 64, tiedDelta, 0, .stratified #[⟨"a", 32, 32⟩, ⟨"b", 32, 32⟩]⟩ : Facts).whole &&
  !(⟨64, 63, tiedDelta, 1, .subset 63⟩ : Facts).whole &&
  !(⟨64, 64, tiedDelta, 0, .stratified #[⟨"a", 32, 33⟩, ⟨"b", 32, 31⟩]⟩ : Facts).whole &&
  !(⟨64, 32, tiedDelta, 0, .stratified #[⟨"a", 32, 32⟩]⟩ : Facts).whole &&
  !(⟨64, 64, tiedDelta, 0, .bernoulli ⟨1, 1⟩⟩ : Facts).whole &&
  !({ units := 64, drawn := 64, delta := tiedDelta, wrong := 0 } : Facts).whole
-- no view tie is held, so a term charges its scan's view rows
#guard !viewTieHeld
-- every unit proved: no wrong unit, so no faulty set and no faulty bits, and the term charges the scan's 40 view bits
-- alone: neither its tied audits' 500 free inputs, nor the 96 entry bits their wrong units can set, nor its 96 entry
-- bits. With the view tie held it charges those 96 entry bits once, in their place, and no view row
#guard (rowsTerms #[] ⟨4, 0, 300, 512, 96, 40⟩ #[(⟨8, 8, ⟨1, 100⟩, 0, .subset 8⟩, 500, 96)] #[] 1 1).K == 40 &&
  (rowsTerms #[] ⟨4, 0, 300, 512, 96, 40⟩ #[(⟨8, 8, ⟨1, 100⟩, 0, .subset 8⟩, 500, 96)] #[] 1 1 #[] true).K == 96
-- w63's class with the egress tie: T = 32, the scan's own free inputs none, its view rows 32 of 4,832 bits, and its
-- tied audit (the slot entries' `SlotEntry_v1` template) of 8 units, a slot at each of the 2 boundaries, whose free
-- inputs are their entries, 1,136 bits each. The scan sampled at all but one of its 64 units, with at most 1 wrong
-- (one of 65 sets, its 512 bits): K is the tied audit's 9,088 beside them and the view rows' 154,624. Every unit
-- proved: K is the view rows' 154,624. With the view tie held: the tied audit's 9,088 beside the wrong unit's, and
-- every unit proved the scan's 8 slot entries of 112 bits, 896
#guard (rowsTerms #[{ units := 8, drawn := 8, delta := ⟨1, 100⟩, wrong := 0 }] ⟨32, 0, 300, 512, 8 * 112, 32 * 4832⟩
  #[(⟨64, 63, ⟨1, 100⟩, 1, .subset 63⟩, 8 * 1136, 0)] #[] 1 1).K == 9088 + 7 + 512 + 154624
#guard (rowsTerms #[{ units := 8, drawn := 8, delta := ⟨1, 100⟩, wrong := 0 }] ⟨32, 0, 300, 512, 8 * 112, 32 * 4832⟩
  #[(⟨64, 64, ⟨1, 100⟩, 0, .subset 64⟩, 8 * 1136, 0)] #[] 1 1).K == 154624
#guard (rowsTerms #[{ units := 8, drawn := 8, delta := ⟨1, 100⟩, wrong := 0 }] ⟨32, 0, 300, 512, 8 * 112, 32 * 4832⟩
  #[(⟨64, 63, ⟨1, 100⟩, 1, .subset 63⟩, 8 * 1136, 0)] #[] 1 1 #[] true).K == 9088 + 7 + 512
#guard (rowsTerms #[{ units := 8, drawn := 8, delta := ⟨1, 100⟩, wrong := 0 }] ⟨32, 0, 300, 512, 8 * 112, 32 * 4832⟩
  #[(⟨64, 64, ⟨1, 100⟩, 0, .subset 64⟩, 8 * 1136, 0)] #[] 1 1 #[] true).K == 896
-- eps over the scans' δ = 1/15 and 1/15 is 2/15; the audit's 1/10 is in it no more
#guard (rowsTerms #[⟨1, 1, ⟨1, 10⟩, 0, .subset 1⟩] ⟨0, 0, 0, 0, 0, 0⟩
  #[(⟨1, 1, ⟨1, 15⟩, 0, .subset 1⟩, 0, 0), (⟨1, 1, ⟨1, 15⟩, 0, .subset 1⟩, 0, 0)] #[] 1 2).eps.toString == "2/15"
-- w63k4: T = 32, 4 lanes, every 16, so 8 slot entries of 112 bits a link-window, and 32 view rows of 800 bits; its
-- tied audit drawn in part, with at most 2 wrong units, which can set any entry. Its scan sampled in part (at most 1
-- wrong of 64): its link-window's term charges E_w = 896 bits beside its free inputs, and its view rows' 25,600. Every
-- unit of its scan proved: the term charges the view rows alone, and neither the tied audit's free inputs, nor E_w for
-- its wrong units, nor E_w once. With the view tie held, the scan sampled in part is charged no view row, and with
-- every unit proved E_w once, in place of the tied audit's free inputs and E_w for its wrong units
#guard (rowsTerms #[{ units := 16, drawn := 9, delta := ⟨1, 100⟩, wrong := 2 }] ⟨32, 0, 300, 512, 8 * 112, 32 * 800⟩
  #[(⟨64, 63, ⟨1, 100⟩, 1, .subset 63⟩, 9088, 8 * 112)] #[] 1 1).K == 9088 + 7 + 512 + 896 + 25600
#guard (rowsTerms #[{ units := 16, drawn := 9, delta := ⟨1, 100⟩, wrong := 2 }] ⟨32, 0, 300, 512, 8 * 112, 32 * 800⟩
  #[(⟨64, 64, ⟨1, 100⟩, 0, .subset 64⟩, 9088, 8 * 112)] #[] 1 1).K == 25600
#guard (rowsTerms #[{ units := 16, drawn := 9, delta := ⟨1, 100⟩, wrong := 2 }] ⟨32, 0, 300, 512, 8 * 112, 32 * 800⟩
  #[(⟨64, 63, ⟨1, 100⟩, 1, .subset 63⟩, 9088, 8 * 112)] #[] 1 1 #[] true).K == 9088 + 7 + 512 + 896
#guard (rowsTerms #[{ units := 16, drawn := 9, delta := ⟨1, 100⟩, wrong := 2 }] ⟨32, 0, 300, 512, 8 * 112, 32 * 800⟩
  #[(⟨64, 64, ⟨1, 100⟩, 0, .subset 64⟩, 9088, 8 * 112)] #[] 1 1 #[] true).K == 896
-- w63k4's period of one link-window, every unit of its scan proved, over its schedule's 128 connections of 5 egress
-- cells and jitter below 2 over its 32 steps: the schedules' and the fail cells' 384 bits each, the view rows' 25,600
-- and the jitter's 32; with the view tie held, the slot entries' 896 in the view rows' place
#guard (rowsTerms #[] ⟨32, 0, 300, 512, 8 * 112, 32 * 800⟩ #[(⟨64, 64, ⟨1, 100⟩, 0, .subset 64⟩, 9088, 0)]
    (Array.replicate 128 5) 2 1).K == 384 + 384 + 25600 + 32 &&
  (rowsTerms #[] ⟨32, 0, 300, 512, 8 * 112, 32 * 800⟩ #[(⟨64, 64, ⟨1, 100⟩, 0, .subset 64⟩, 9088, 0)]
    (Array.replicate 128 5) 2 1 #[] true).K == 384 + 384 + 896 + 32
-- neither rule takes a minimum: a scan proved whole whose tied audits count 100 free inputs, below its 25,600 view
-- bits and its 896 entry bits, and have no wrong unit is charged the 25,600, and with the view tie held the 896, where
-- its tied audits' count is 100
#guard (rowsTerms #[] ⟨32, 0, 300, 512, 8 * 112, 32 * 800⟩ #[(⟨64, 64, ⟨1, 100⟩, 0, .subset 64⟩, 100, 0)] #[] 1
    1).K == 25600 &&
  (rowsTerms #[] ⟨32, 0, 300, 512, 8 * 112, 32 * 800⟩ #[(⟨64, 64, ⟨1, 100⟩, 0, .subset 64⟩, 100, 0)] #[] 1 1 #[]
    true).K == 896 &&
  (rowsTerms #[] ⟨32, 0, 300, 512, 8 * 112, 32 * 800⟩ #[(⟨64, 63, ⟨1, 100⟩, 0, .subset 63⟩, 100, 0)] #[] 1 1 #[]
    true).K == 100
-- w63k4's schedule: 4 lanes over a 32-bucket link-window carry at most 128 connections, each capped at its 5 egress
-- cells, so the schedules and the fail cells each charge 128 · ⌈log2 6⌉ = 384 bits a link-window
#guard
  let s : Schedule := ⟨100, 512, 27, 5, 16, 12, 47, 4, 4, 4, 32⟩
  s.caps 1 == Array.replicate 128 5 && capsTerm (s.caps 1) == 384 && (s.caps 2).size == 256
-- w63's frames, B = 512 and 128 connections a link-window, read back over 2^25 − 1 link-windows and not over 2^25,
-- whose last connection's ordinal is 2^32; the check refuses that period
#guard
  let s : Schedule := ⟨100, 512, 27, 65, 16, 12, 47, 4, 4, 4, 32⟩
  let p := fun (w : Nat) =>
    (⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", none, .rows s .empty ⟨0, 0, 0, 0, 0, 0⟩, w⟩ : Params)
  rowsWf s (2 ^ 25 - 1) && !rowsWf s (2 ^ 25) && (rowsWfCheck (p (2 ^ 25 - 1))).isNone &&
    (rowsWfCheck (p (2 ^ 25))).isSome
-- a charged audit's δ = 1/10 is in eps beside its scan's 1/15: 1/6
#guard (rowsTerms #[⟨2, 1, ⟨1, 10⟩, 0, .subset 1⟩] ⟨0, 0, 0, 0, 0, 0⟩ #[(⟨1, 1, ⟨1, 15⟩, 0, .subset 1⟩, 0, 0)] #[] 1 1
  #[⟨2, 1, ⟨1, 10⟩, 0, .subset 1⟩]).eps.toString == "1/6"
-- the same draw read at δ_i = 2^-64: one of 16 drawn (`subset:1`, 8 at δ = 1/2) leaves 15 possibly faulty, all 16 drawn
-- none, and facts with no draw every unit; a stratified draw's bound at δ = 1/2 is `countBound`'s
#guard (Facts.readAt ⟨16, 1, ⟨1, 2⟩, 8, .subset 1⟩ tiedDelta).wrong == 15 &&
  (Facts.readAt ⟨16, 1, ⟨1, 2⟩, 8, .subset 1⟩ ⟨1, 2⟩).wrong == 8 &&
  (Facts.readAt ⟨16, 16, ⟨1, 2⟩, 0, .subset 16⟩ tiedDelta).wrong == 0 &&
  (Facts.readAt { units := 16, drawn := 1, delta := ⟨1, 2⟩, wrong := 8 } tiedDelta).wrong == 16 &&
  (Facts.readAt
    ⟨9, 5, ⟨1, 10⟩, 0, .stratified #[{ name := "a", size := 4, k := 4 }, { name := "b", size := 5, k := 1 }]⟩
    ⟨1, 2⟩).wrong == 2
-- w63k4's tied audit of 16 proof blocks sampled at one (`subset:1`), its answer's bound 8 at δ = 1/2: among a
-- commitment's link-windows it is charged, read at δ_i = 2^-64 with a bound of 15, its phase's β_w every one of the
-- scan's E_w = 896 entry bits; sampled whole, uncharged and β_w = 0. The public keeps the answer's δ and bound in
-- `audits`, and the scan's E_w and its view rows' 25,600 bits in `scanBits`. Its scan sampled in part (63 of 64 units,
-- at most 1 wrong): the period's reading is in `charged`, the term charges the audit's 9,088 free inputs, β_w and the
-- view rows, and eps is the scan's δ = 2^-64 and δ_i, 2^-63. Every unit of its scan proved: no audit is charged, the
-- term is the scan's view rows alone, and eps is the scan's 2^-64 alone; the same with the audit sampled whole and no
-- free input. Over two link-windows, the first's scan proved whole and the second's sampled in part, the audit is
-- charged for the second alone: the first's term is the view rows. With the view tie held no term charges a view row:
-- the scan proved whole is charged E_w once, and sampled in part the audit's free inputs and β_w
#guard
  let scan : Scan := ⟨32, 0, 300, 512, 8 * 112, 32 * 800⟩
  let p : Params := ⟨#[⟨0, 0, "", 16, 0, #[], none, none, 0⟩], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", none,
    .rows ⟨0, 0, 0, 0, 16, 0, 0, 4, 0, 0, 32⟩ .empty scan, 1⟩
  let at_ := fun (q : String) => Json.mkObj [("roots", Json.arr #[
    Json.mkObj [("port", .str q), ("root", .str "r"), ("domain", .str "d")]])]
  let c : Commitment := ⟨#[at_ (entriesPort "SelectEntry_v1{KO=4}" 0)], #[at_ (progName 0)], .empty⟩
  let c2 : Commitment := ⟨c.audits, #[at_ (progName 0), at_ (progName 0)], .empty⟩
  let one : Array (Found Facts) := #[.ok ⟨16, 1, ⟨1, 2⟩, 8, .subset 1⟩]
  let whole : Array (Found Facts) := #[.ok ⟨16, 16, ⟨1, 2⟩, 0, .subset 16⟩]
  let all : Facts := ⟨64, 64, tiedDelta, 0, .subset 64⟩
  let part : Facts := ⟨64, 63, tiedDelta, 1, .subset 63⟩
  let read := fun (x : Public) => x.units.map fun t => (t.free, t.faultyEntries, t.entries, t.view)
  let pub := rowsPublic p c one scan #[(all, 9088)]
  let pub' := rowsPublic p c one scan #[(part, 9088)]
  let pub2 := rowsPublic p c2 one scan #[(all, 9088), (part, 9088)]
  let tied := rowsPublic p c one scan #[(all, 9088)] true
  let tied' := rowsPublic p c one scan #[(part, 9088)] true
  tiedIn c.audits c.windows 0 && (chargedOf p c one 0).map (fun f => (f.delta, f.wrong)) == some (tiedDelta, 15) &&
    faultyEntries p c one scan (at_ (progName 0)) == 896 &&
    (chargedOf p c whole 0).isNone && faultyEntries p c whole scan (at_ (progName 0)) == 0 &&
    pub.audits.map (fun f => (f.delta, f.wrong)) == #[((⟨1, 2⟩ : Core.Ratio), 8)] &&
    pub'.charged.map (fun f => (f.delta, f.wrong)) == #[(tiedDelta, 15)] && pub'.eps == (⟨1, 2 ^ 63⟩ : Core.Ratio) &&
    read pub' == #[(9088, 896, 0, 25600)] && pub'.K == 9088 + 7 + 512 + 896 + 25600 &&
    pub.charged.isEmpty && pub.eps == tiedDelta && read pub == #[(0, 0, 0, 25600)] && pub.K == 25600 &&
    pub.scanBits == (896, 25600) &&
    (rowsPublic p c whole scan #[(all, 0)]).K == 25600 &&
    (rowsPublic p c whole scan #[(part, 0)]).K == 7 + 512 + 25600 &&
    pub2.charged.map (fun f => (f.delta, f.wrong)) == #[(tiedDelta, 15)] &&
    read pub2 == #[(0, 0, 0, 25600), (9088, 896, 0, 25600)] && pub2.eps == (⟨3, 2 ^ 64⟩ : Core.Ratio) &&
    tied.charged.isEmpty && tied.eps == tiedDelta && read tied == #[(0, 0, 896, 0)] && tied.K == 896 &&
    read tied' == #[(9088, 896, 0, 0)] && tied'.K == 9088 + 7 + 512 + 896 && tied'.eps == (⟨1, 2 ^ 63⟩ : Core.Ratio)
-- a unit's output port past o0, its index in ASCII digits without a leading zero; never o0, an input, or no unit's
#guard outputPort "block-3/o1" && outputPort "a/b/o10" && outputPort "/o2"
#guard !(outputPort "o1" || outputPort "u/o0" || outputPort "u/o01" || outputPort "u/o" || outputPort "u/O1" ||
  outputPort "u/o1x" || outputPort "u/o٣" || outputPort "u/prog" || outputPort "")
-- a SlotEntry_v1 or SelectEntry_v1 template's entries port, a chunk's o0, the chunk's index in ASCII digits without a
-- leading zero; never another Definition's or port's
#guard entriesPortName (entriesPort "SlotEntry_v1{IDB=8,KO=2,NP=4,STOP=1}" 0) &&
  entriesPortName "SlotEntry_v1{}/c0/o0" && entriesPortName "SlotEntry_v1{KO=2}/c1/o0" &&
  entriesPortName (entriesPort "SelectEntry_v1{V=256,C=16,KO=2,FIRST=false,IDB=17,NP=4,STOP=[]}" 3) &&
  entriesPortName "SelectEntry_v1{}/c0/o0" && entriesPortName "SelectEntry_v1{KO=2}/c10/o0" &&
  entriesPort "SelectEntry_v1{}" 3 == "SelectEntry_v1{}/c3/o0"
#guard !(entriesPortName "SlotEntry_v2{KO=2}/c0/o0" || entriesPortName "SlotEntry_v1{KO=2}/c01/o0" ||
  entriesPortName "SlotEntry_v1{KO=2}/c0/o1" || entriesPortName "SlotEntry_v1/c0/o0" || entriesPortName "" ||
  entriesPortName "SlotEntry_v1{KO=2}/c/o0" || entriesPortName "SlotEntry_v1{KO=2}c0/o0" ||
  entriesPortName "SlotEntry_v1{KO=2}/cx/o0" || entriesPortName "SlotEntry_v1{KO=2}/c٣/o0" ||
  entriesPortName "SelectEntry_v2{KO=2}/c0/o0" || entriesPortName "SelectEntry_v1{KO=2}/c1/o1" ||
  entriesPortName "SelectEntry_v1/c0/o0" || entriesPortName "ScanSlots_v2{KO=2}/c0/o0")
-- two lanes, T = 4, every 2: lane j's scan/prog<j> is chunk j's entries port, of 2 leaves, of one pinned template of 4
-- members; not with the lanes swapped, a chunk of other leaves, lane 1's tree missing, or another template's chunk 1;
-- and the scan's lane-1 tree alone ties the audit
#guard
  let s : Schedule := ⟨0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 4⟩
  let a : Audit := ⟨0, 0, "", 4, 0, #[("SlotEntry_v1{KO=1}", 4, slotEntryPins[0]!)], none, none, 0⟩
  let port := fun (q root : String) (l : Nat) =>
    Json.mkObj [("port", .str q), ("leaves", toJson l), ("root", .str root), ("domain", .str "d")]
  let reg := fun (os : Array Json) => Json.mkObj [("roots", Json.arr os)]
  let chunks := fun (l0 l1 : Nat) (k1 : String) =>
    reg #[port "SlotEntry_v1{KO=1}/c0/o0" "r0" l0, port (k1 ++ "/c1/o0") "r1" l1]
  let scan := fun (r0 r1 : String) => reg #[port "scan/prog0" r0 2, port "scan/prog1" r1 2]
  wired #[a] s #[chunks 2 2 "SlotEntry_v1{KO=1}"] (scan "r0" "r1") &&
    !wired #[a] s #[chunks 2 2 "SlotEntry_v1{KO=1}"] (scan "r1" "r0") &&
    !wired #[a] s #[chunks 3 1 "SlotEntry_v1{KO=1}"] (scan "r0" "r1") &&
    !wired #[a] s #[chunks 2 2 "SlotEntry_v1{KO=1}"] (reg #[port "scan/prog0" "r0" 2]) &&
    !wired #[a] s #[chunks 2 2 "SlotEntry_v1{KO=2}"] (scan "r0" "r1") &&
    tiedAudits #[chunks 2 2 "SlotEntry_v1{KO=1}"] (reg #[port "scan/prog1" "r1" 2]) == #[0]
-- the class circuit c is held when every tied audit is registered under it, an untied audit's Program unread; not when
-- one tied audit's is another, or with no class circuit; with no tied audit it holds (`wired` refuses that)
#guard
  let o := fun (q root : String) => Json.mkObj [("port", .str q), ("root", .str root), ("domain", .str "d")]
  let audit := fun (prog root : String) =>
    Json.mkObj [("program", .str prog), ("roots", Json.arr #[o "SlotEntry_v1{KO=1}/c0/o0" root])]
  let scan := Json.mkObj [("roots", Json.arr #[o "scan/prog0" "r0"])]
  held (some "c") #[audit "c" "r0"] scan && held (some "c") #[audit "c" "r0", audit "x" "r1"] scan &&
    !held (some "c") #[audit "d" "r0"] scan && !held none #[audit "c" "r0"] scan &&
    !held (some "c") #[audit "c" "r0", audit "d" "r0"] scan && held (some "c") #[] scan
-- a pin is held where its registration's window records its root as the parameter's registered root, the hex without
-- case; not at another root, another parameter, or a registration with no window
#guard
  let reg := Json.mkObj [("window", Json.mkObj [("registered_roots", Json.mkObj [("w", .str "AB")])])]
  pinHeld reg ("w", "ab") && !pinHeld reg ("w", "ac") && !pinHeld reg ("v", "ab") && !pinHeld (Json.mkObj []) ("w", "ab")
#guard isStr (.str "ab") "ab" && !isStr (.str "ab") "AB" && !isStr (toJson 1) "1" && !isStr .null ""
-- a lane's Programs' rows, the lane's index in ASCII digits without a leading zero
#guard progPort (progName 0) && progPort "scan/prog3" && progPort "scan/prog12" && progName 3 == "scan/prog3"
#guard !(progPort "scan/prog" || progPort "scan/prog01" || progPort "scan/prog/" || progPort "Scan/prog0" ||
  progPort "scan/prog0x" || progPort "prog0" || progPort "scan/state" || progPort "")

end NetworkCheck.Period
