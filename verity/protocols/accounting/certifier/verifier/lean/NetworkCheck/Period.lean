module

public import NetworkCheck.Check
public import NetworkCheck.Clocks
public import NetworkCheck.ScanOk
public import CoreProtocol.Profile
public import Flock.Program
public import Flock.Public
public import Flock.WindowRecord
public import NciCheck.Desc
public import NciCheck.DefinitionDigest
public import NetworkCheck.Period.ConnectionRows
meta import NetworkCheck.Check
meta import NetworkCheck.Clocks
meta import NetworkCheck.ScanOk
meta import CoreProtocol.Profile
meta import Flock.Program
meta import Flock.Public
meta import Flock.WindowRecord
meta import NciCheck.Desc
meta import NciCheck.DefinitionDigest
meta import NetworkCheck.Period.ConnectionRows

/-!
# The network certifier over a period, a core protocol: `verity/certifier/v0`

The network certifier, one of onsite's subprotocols (`Core.Onsite.components`). A period is one node's link over `m`
execution phases beside the period's compute audits (sampled proofs' one-stage, one answer per registered window of the
workload). Its execution phases come in one of two forms, which the parameters name:

* **the clock form**: N2's execution phases (`service.LinkWindows`, Selection `All`), each proved, both its ends
  certified and its clock certified;
* **the rows form**: the network certifier's own run, where it functions as an active network certifier
  (`warden_serve`): each execution phase's cells committed as rows (`warden_rows`), one registration a direction under
  one fixed schedule, its three records certified, and the scan's one-stage answer for the window (`scan_zk
  --warden-run`).

It accepts when every execution phase passes its form's check and every compute audit accepted; it then publishes `K`, the
bits a covert sender in the server can move through the network certifier in the period, as
`NetTiming.Guarantees.MonitoringPeriodLeakAccepted` counts them. In the rows form each execution phase's scan's
Programs' row must be a compute audit's slot entries (`wired`, below), and the state that audit's Programs read (its
`SCAN_STATE`) the scan's state, by root and domain, which the scan's units read and register as its transitions chain it
from the state its start unit copies, the empty state (`stateTied`, `startCheck`), so a slot's entry is the one its
Program computed on the requests the scan read; and each scan's ok bits must be at the roots the check computes from the
schedule's T (`okChecks`).

**Params**, JSON with exactly these keys (and each nested object exactly its own), in the clock form

    {"audits": [{"free": f, "program": descriptor, "steps": k}, …], "cells": [cap, …], "first": w₀, "jitter": Dj,
     "key": hex(I ‖ T[1]), "link": name, "lms": {"H": h, "N": n, "W": w, "toy": bool}, "program": hex,
     "session": hex(session), "sync_set": {"deltas": [δ, …], "rhos": [[num, den], …]}, "windows": m}

each compute audit's layout terms (`free`, the free inputs it doesn't register, `#(freeInputs \ reg)`; `steps`, its
monitoring period `k`) and its Program, the descriptor (`verity-ir/descriptor/v1`) of the window's population program,
which must decode (`Flock.Program.decode`) with a template in its root: the audit's proof blocks are the root's template
instances, and each commits its template's return, the outputs it computes that other proof blocks or the client read, which
can be wider than a proof block's 16-bit public output; each network certifier connection's cell cap over the period
(`cap_c`), the session's window
the period starts at (`w₀`: its execution phases are the session's `w₀ … w₀ + m − 1`, so one certifier session spans
many periods),
the observations' jitter bound `Dj ≥ 1`, the network certifier's LMS public key and parameters and the certifier session
(32 bytes) every certificate binds, the link (at most 65,535 UTF-8 bytes), the SHA-512 of N2's registered Program for
the link (the partition object's `program`, `LinkWindows.partition`), Σ_sync, and the period's `m ≥ 1` execution phases.
The rows form has, in place of `sync_set`,

    "circuit": hex,
    "scan": descriptor,
    "schedule": {"bucket_ms", "cell_bytes", ["content",] "delay", "egress_cells", "every", "ingress_buckets",
                 "ingress_cells", "lanes", ["lengths",] ["max_tokens_cap",] "queue", ["request_words",] "slots",
                 "window"}

the class circuit, the SHA-512 of the Program of the one window circuit the workload's class registers, which every
tied compute audit runs (`held`);
the scan's Program, its descriptor (`verity-ir/descriptor/v1`, `scan_zk`'s `program.json`), and the network
certifier's one fixed schedule (`warden.WardenParams`: each a positive integer, B = `cell_bytes` past a cell's 9-byte
header, B, S = `lanes` · `slots` and T = `window` u32s, and `every`, a slot stride's buckets, dividing T, so each
execution phase is T / `every` timesteps of `lanes` Program slots, `Schedule.programSlots` and
`Schedule.executionPhasePrograms`; its `max_tokens_cap` when it has one, a u32 below `egress_cells`, under which a
cell's payload holds an id request's `max_tokens` field, `maxTokensEnd`; its lengths rows' form `lengths` when it has
them, 1 or 2, under which ingress ends by the delay, a connection ends within a window of its admission
(`Schedule.span`), its egress cells count in a u16, a cell's payload holds an id request's head and first prompt id
(`firstPromptEnd`) and a slot stride's releases are at most a lengths row's slots (`Schedule.strideReleases`,
`mostSlots`); and its content rows' classes `content` when it has them, which only a schedule with lengths rows has,
each (P, O, rows) under form 1 and (P, O) under form 2, `classesRefusal`; and `request_words`, true, when it has request
words, which only a schedule with content rows under form 2 and `max_tokens_cap` has, and which lists at most
`mostClassesWords` classes), whose digest `WardenParams.digest` every registration binds. A schedule with lengths rows
registers its connection rows, which this check does not read: it is parsed as the network certifier builds it, its
digest is the network certifier's, and it is refused by name, in the parameters and in the check
(`connectionRowsRefusal`,
`connectionRowsCheck`), so no period is accepted under one. The rows form's `program` is the scan's registered
Program, the descriptor's SHA-512 (`partition_object.program_sha512`), and its `link` the run's
(`warden_serve.LINK`), whose fails record is certified under the link `<link>-fails`. Its
`cells` must be the schedule's cells over the period (`Schedule.rowsCells`): `egress_cells` for each of the `lanes · m ·
window` connections the network certifier can carry in `m` of a link's execution phases, since it admits a connection at
a bucket's start only while fewer than `lanes` are live and serves it in that bucket. Any other list, an empty one
included, is refused, so `K`'s schedules and cells terms are never read off cells that miss a connection. Before
that, those `lanes · m · window` connections, the k-th at ordinal k + 1, must have distinct u32 ordinals, and a cell's
`B` bytes must be under 2^32 (`rowsWf`): a frame names its connection and its payload's length by u32s, so past 2^32
two frames would encode alike and the record would not read back (`NetTiming.RowsFormat.Wf`).
The check refuses such a period too (`rowsWfCheck`), so whatever parameters it accepts, its frames read back. The
descriptor must decode (`Flock.Program.decode`) and be the scan of the schedule's execution phase in the generator form:
its root batches one instance of a template whose one parameter is `sin` and which returns the state alone
(`IngressScanStart`, the start unit, which copies the state the execution phase starts from), T of one whose parameters
are `row`, `sin` and a `prog<j>` a lane (`IngressScanBucket`), lane j's `prog<j>` rows of an input node of its own
(`scanOf`), each `row` the schedule's (S · B bytes), then T of one whose parameter is `state` and which returns a u16
and the schedule's row (`EgressBucket`), in the Boolean profile's types (a byte 8 bits, a u16 16), and its every input
gate is a template's parameter. An egress template that takes its row as an input, the checker form, is refused by name: no
egress gate is an input (`NetTiming.BoundaryGates.Fits`), so `K` counts only what a Program that computes its egress rows
leaves free. The rest of what `NetTiming.BoundaryGates.Fits` asks is read off the description (`scanOf`): the Program laid out
as the decoder lays one out and within its resolution's walk (`NciCheck.Desc`), every egress leaf returned by the root
and computed by the egress template, and every template's argument one input node's leaves, no egress template's the
ingress rows'. In the rows form a compute audit's free inputs are in the term of each execution phase whose scan's
`prog` is its output (`tiedAudits`), and its other layout terms and its units' bits are in no term; their Programs are
checked as in the clock form.

**Commitment**, canonical JSON `{"audits": [registration, …], "windows": [registration, …]}`: one-stage's registration
record of each compute audit and of each execution phase, in order, each of format `verity/registration/v1` or `v2`. Its
digest is `tagged_sha512("verity-network-certifier/period/v0", the commitment's bytes)`, and each registration's is
one-stage's, SHA-512(format ‖ 0 ‖ canonical record) (`registration.digest`).

**Evidence**, canonical JSON:

    {"audits": [answer, …],
     "windows": [{"answer": answer, "certificates": [certificate, certificate], "clock": certificate}, …]}
    certificate = {"digest": hex, "direction": "egress" | "ingress", "key_id": hex, "link": s, "signature": hex,
                   "window": w}

in the clock form, and in the rows form each execution phase
`{"answer": answer, "certificates": [certificate, certificate, certificate], "registrations": [rows, rows]}`, with

    rows = {"cell_bytes", "certificate": {"digest", "key", "signature"}, "direction", "domain", "format",
            "layout": {"buckets", "connection_slots", "first_bucket", "slots"}, "link", "parameters", "root",
            "row_schema", "rows", "value", "window", "wiring", "word_bits", "words"}

the requests', responses' and fails' records' certificates (`certified.jsonl`'s), and the ingress and the egress rows'
registrations (`warden_rows.registration`, `cells.jsonl`'s); an execution phase of the other form than the parameters'
is refused.

In the clock form: each audit's and each execution phase's one-stage answer (`Core.Answer.ofJson`), and per execution
phase the two certificates (`certificate.Certificate`, egress first) and the clock certificate the certifier, the
auditor's device, issued for it:
the public part of `certificate.CertifiedClock`, which signs the window's sync salted, `clock_commitment` =
SHA-512("verity/net-cert/v1" ‖ salt_c ‖ `clock_digest(sync)`). What the certifier signs is witness: the sync and its
salt are in neither the evidence nor the profile, and the check takes neither. The certifier is the auditor's trusted
device (`trusted-host`), and its own code picks each window's sync from its schedule set Σ_sync, so no
clear-text check that the sync is in Σ_sync is needed. N2's proof reading each window's sync through its opening
against that salted digest is a later change to its circuit (the certifier PROTOCOL.md's gap "The clock's opening").

**The check**, in order; any rejection rejects, then any undecided answer leaves the period undecided:

* the commitment registers the parameters' audits and `m` execution phases, and the evidence answers each;
* execution phase `w` (the `i`-th, `w = w₀ + i`): its registration names the link's Program and the provenance
  `{"kind": "served", "window": w}`;
  both certificates are the key's, name the link, their end and `w`, and verify under LMS over `certificate_message` in
  the session; the clock certificate is the key's egress certificate of the link at `w` and verifies so, and the
  digest it signs has the salted form: no sync of Σ_sync's `clock_digest` itself (the unsalted form, which shows the
  sync), and no earlier execution phase's clock digest in the period (a reused salt, under which two windows of one sync
  sign one digest); and the answer is one-stage's for that registration's digest,
  accepted with the profile of one unit sampled `subset:1`: the execution phase was proved (N2's verifier,
  `LinkWindows.verify`, is the answer's verifier);
* in the rows form, execution phase `w`: its registration (the scan's, per window) names the scan's Program; its three
  certificates are the requests' record's (the link's ingress), the responses' (its egress) and the fails'
  (`<link>-fails`, egress) at `w`, each the key's and verifying in the session, so an execution phase missing any
  record's certificate is refused, and the network certifier certifies no record of an execution phase in which it
  refused a connection (traffic past the window's Program slots, `warden`'s "Admission"), so such an execution phase is
  refused; its two registrations, ingress then egress, are what `warden_rows.check_registration` reads: the execution
  phase's value, link, end and window, the schedule's digest (a registration under another schedule is refused), the
  schedule's rows, words, word bits, row format and layout, the execution phase's domain in the session
  (`warden_rows.domain`), and a certificate that is the key's, signs the head (`CellsHead.digest`) and verifies in the
  session (a registration not under the deployment key is refused); the scan's registration holds both, an entry of its
  `roots` or `registered` naming each one's root and domain, which bind the session, link, end and window (an answer
  for another window's rows is refused); its Programs' rows are a compute audit's output (`progTied`): a lane's
  `scan/prog<j>` entry names the root and domain of an output port's entry (`<unit>/o<j>`, j ≥ 1) or a `SlotEntry_v1`
  or `SelectEntry_v1` template's entries port's (`<descriptor id>/c<j>/o0`, `entriesPortName`) in the `roots` of an
  audit's registration, so a scan whose `prog` the prover committed alone is refused; and the answer is one-stage's
  for that registration's digest, accepted with one level of the scan's 2T + 1 units sampled `subset:k` or stratified,
  sampling no more than the level holds and no more of a stratum than the stratum holds, its strata's sizes summing
  to the level's (`factsOf` refuses a profile that fails any of the three), and its `bound`;
* in the rows form, every accepted execution phase's scan is at one δ, `MonitoringPeriodLeakAccepted`'s;
* audit `i`: its registration names its Program (the descriptor's SHA-512), and the answer is one-stage's for its
  registration's digest, accepted with one level of its Program's units sampled `subset:k` or stratified (a work sampling design's
  sample is stratified), held to the same three (`factsOf`), and its `bound`, at any sample: this is the consistency
  check (each audit's answer is its registration's, and in the rows form its outputs are the scan's `prog` and its
  `SCAN_STATE` the scan's state, below), and what a sample in part leaves unproved is in `K` (in the rows form, a tied
  audit's `faulty_entries` and δ, below);
* in the rows form, execution phase `w`'s scan's Programs' rows are a compute audit's slot entries, a lane a chunk
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
* in the rows form, execution phase `w`'s scan's state is the `SCAN_STATE` of an audit its `prog` is wired to
  (`stateTied`): the scan's registration has an entry of its `roots` whose port is `scan/state` or of its `registered`
  whose value is `verity/certifier/scan-state/v1`, every such entry at one root and domain, and so has a `wired`
  audit's registration, every such entry at that root and domain; and the scan's registration states its units' reads
  of that state, its `window`'s

      "reads": {"scan/state": {"o0": {"leaf": 0, "units": [0, 1]}, "o1": {"leaf": 1, "units": [1, T]},
                               "sin": {"leaf": 0, "units": [1, T + 1]}}, …}

  (`stateRead`, a run a port: unit `lo + i` at leaf `l + i`; unit 0 is the start unit and unit t + 1 bucket t), so the
  start unit registers its `o0` as that root's row 0, and the scan's ingress bucket t reads `sin` as row t and
  registers its `o1` as row t + 1, but for the last bucket, whose `o1` no row holds; each row past row 0 is the state
  the scan's transition computed from the row before it, and row 0 the state the start unit copied, the prover's word
  for none. A slot's Program anchored at bucket a reads row a, the scan's state before a, so it computes its entry on
  the requests the scan read, not on a state the prover chose (`Properties.NetworkPeriod.RowsStateTied`);
* in the rows form, execution phase `w` is the run's first, execution phase 0, and its scan starts from the empty state
  (`startCheck`): its registration has an entry of its `registered` whose value is `0.sin`, the start unit's `sin`,
  and its `window`'s

      "empty": {"0.sin": {"words": W, "salt": hex}, …}

  opens every such entry to `W` zero u16 words under its salt: that row's leaf under the entry's domain is the
  entry's root, a tree of the one leaf (`startEmpty`: the row and its salt are public). The start unit reads its
  `sin` there and copies it to row 0 of the scan's state, so the Programs anchored at bucket 0 read the empty state
  (`Properties.NetworkPeriod.RowsStartEmpty`). A later execution phase starts from the state the one before it ended in,
  which the check holds no start to: it refuses every execution phase but the run's first (`startCarriedRefusal`), so
  the periods it accepts are of the run's first execution phase alone;
* in the rows form, execution phase `w`'s tied compute audits run the class circuit (`held`): each audit its scan's
  `prog` is an output of (`tiedAudits`) has a registration naming the parameters' `circuit` as its Program. A window
  circuit's own digest binds its window's run, so it is another each window; its Program is the class's, one for the
  period. So no execution phase's tied audit is registered under a Program the prover chose for that phase
  (`Properties.NetworkPeriod.RowsCircuitHeld`), and, since the audit check holds the registered Program to the
  descriptor its term is read off (`auditCheck`), that descriptor is the class circuit's.
* in the rows form, each execution phase's scan's ok bits are the schedule's constants (`okChecks`,
  `NetworkCheck.ScanOk`): a scan proof block's public word `o0` is its ok bit, and a member's `o0` rows are committed
  under public salts, so the root of a member whose every proof block is ok is a function of the schedule's T alone.
  The check computes those roots once a period (`ScanOk.consts`, `ScanOk.calls` SHA-512 calls whatever `m`) and holds
  every registration to them by comparison (`ScanOk.held`, no hash an execution phase): its `roots` name each member's
  `<member>/o0` at its constant's root, leaves and domain, and no other port ending `/o0`. A member's statement opens
  its proof blocks' ok words at the root its registration names, so each is the row `[1]` the verifier computed, not a
  row the prover's registered file states. The start unit has no ok bit: its `o0` is the state it copies, a row of
  the scan's state (`stateRead`), and `ScanOk.held` leaves that port, `start/o0`, alone. A schedule of fewer than two
  ticks has no constants (`ScanOk.consts` refuses it: `ingress-0` would be a member of no proof block), so every
  execution phase under one is refused.

**Public**, on acceptance: `K` and its terms, each a ceiling of `log2` of `MonitoringPeriodLeakAccepted`'s factor:

    {"K": K, "audits": [{"delta", "drawn": k, "faulty_bits", "faulty_sets", "free", "jitter", "units": N,
                         "wrong_units": b}, …],
     "cells": Σ_c ⌈log2 (cap_c + 1)⌉, "eps": Σ_i δ_i, "schedules": ⌈log2 #Σ_sync^m⌉, "windows": m}

with `b` the audit's bound on incorrect proof blocks at its δ (one-stage's `bound`, `Bound.boundOf`), `faulty_sets` =
`⌈log2 Σ_{j ≤ b} C(N, j)⌉` (every set of at most `b` proof blocks, which holds the sets a `subset:k` or a stratified
sample leaves possible; `drawn` is the sample's `k`, or its `Σ k` stratified),
`faulty_bits` = `b · bits`, with `bits` the most one of its Program's proof blocks commits (its template's return,
`bitsOf`: the committed values `faultyBits` counts, the outputs an incorrect proof block computes and other proof blocks or
the client read), `jitter` = `⌈log2 Dj^steps⌉`, and `K` the sum of the terms. The schedules term counts every assignment
of a sync of Σ_sync to the `m` execution phases, so it is the same whichever syncs the certifier ran them under, and
needs none of them. `eps` is the `δ` part of the probability that `K` fails to bound the period's leak, the union of
its audits' `δ`; each audit's knowledge error and collision errors add to it, and the check evaluates neither: their
exact sum in lowest terms (`Core.Ratio.toString`, `"0"` with no audits), as PoUS's and remote bandwidth's publics carry
theirs.

In the rows form the terms are `MonitoringPeriodLeakAccepted`'s over its windows, the execution phases' scans, whose
statements read the network certifier's rows. A compute audit reads none, so its own terms are in none (it is an
acceptance condition), but the free inputs of each audit a scan's `prog` is an output of are in that scan's execution
phase's term: the egress renders the audit's outputs, so the bits the prover chose of its inputs are the egress's to
leak. An incorrect proof block of such an audit reaches the scan only as the slot entries the scan reads: the scan reads
the audit only through its `prog`, one input node of `E_w` entry bits an execution phase (`wired`, `scanOf`), and its
state is the scan's own (`stateTied`). So an incorrect proof block of any template, a `SlotEntry_v1` or `SelectEntry_v1`
one setting its slot's entry, a `ScanSlots_v2` one setting the `occ` and prompts its slots' Programs read, or any other
its entries read, changes at most those `E_w` bits; the scan reads an entry only where its own state has the slot due
and occupied, and refuses one there whose `read` is 0, so no `occ` the audit computes drops a request. The check accepts
a tied audit at any sample; sampled in part, its at most `b` incorrect proof blocks can set any of its execution phase's
entries, so that execution phase's term charges every entry bit (`faulty_entries`: `β_w ≤ 1[b ≥ 1] · E_w`) when `b ≥ 1`,
and its δ is in
`eps`:

    {"K": K, "audits": [{"delta", "drawn", "units", "wrong_units"}, …], "cells": c, "eps": m · δ + Σ δ_i,
     ["max_tokens_cap": cap,] "scans": [terms, …], "schedules": c, "windows": m}

* `max_tokens_cap`, present when the schedule has a response cap and absent when it has none, is that cap: each
  response the network certifier released in the period has at most its request's `max_tokens` tokens, and every
  request it released has `max_tokens` at most the cap (`warden`'s "The response cap",
  `NetTiming.Guarantees.ResponseCapped`). The schedule's digest, which every registration binds, binds it
  (`Schedule.digest`), and the check publishes the schedule's (`Properties.NetworkPeriod.RowsCapPublished`).
* `schedules` is `cells`, c = Σ_c ⌈log2 (cap_c + 1)⌉, as the statement multiplies `#Ss` and `∏_c (cap_c + 1)`. A
  schedule (`CertifierSchedule`) is r and D, fixed, and each connection's admission bucket and lane length. The server
  sets a lane's length, when its stream ends, at most `cap_c` cells, and nothing else of it: the network certifier
  admits a queued connection at the first lane that frees, in the clients' arrival order, so the admissions are a
  function of the lengths, and `#Ss ≤ ∏_c (cap_c + 1)`. The second factor is the fail cell's position (`failCell`). A
  failed lane ends at its fail cell, so in a run the two coincide; `K` charges the product as the statement states it.
* each scan's terms, from the scan's Program and its tied audits: `free` the parameters' `free` of the audits its
  `prog` is an output of, summed (`tiedAudits`, each once; the scan's own is 0, as it registers the rows, its `prog` is
  those audits' output, and the start unit's `sin` is the registered start the check holds, `startCheck`: the egress
  tie moves the free term from the scan's `prog` to the tied audit's inputs), `steps` T, `faulty_bits`
  `b · max(b_ing, b_eg)` with `b_ing` an ingress instance's return bits (its verdict u16, state and view) and `b_eg` an
  egress instance's (its u16 and row): each instance's committed values (`faultyBits`), the outputs it computes and
  other proof blocks or the client read; the start unit's are a state's, no more than an ingress instance's
  (`startReturn`). `b` is its answer's `bound` (over its sample's strata, `countBound`), 0 when every
  unit is sampled, `drawn` its sample's `Σ k`, and `faulty_sets` every set of at most `b` of its 2T + 1 units, which holds the
  sets a stratified sample leaves possible too. `faulty_entries` is `E_w`, its slot entries' bits (`Scan.entries`:
  `lanes · T / every` entries of its `prog` ports' type, 8 · 112 = 896 at w63k4), when an audit its `prog` is tied to
  is sampled in part with a `bound` of at least one incorrect proof block (`faultyEntries`), else 0.
* `eps` is the scans' δ summed, `m · δ`, and each tied audit's sampled in part (`chargedOf`), whose bound
  `faulty_entries` reads: the δ part of the union over the monitoring period's windows that
  `CommOfMonitoringPeriodLeakAccepted` states, each window at δ plus its knowledge error and its collision errors.

**Profile**: the link and `m`.

What it doesn't check: the proof's public row against the certificates' digests and the output word, which N2's
verifier checks inside the answer (`LinkWindows.verify`, with Flock's Lean verifier on the `--zk` proof);
the clock form's execution phase registration's domain; in the rows form, the records' certificates' digests against the
rows (each signs its salted record, witness) and the scan's statements against its registration (lane `j`'s slot
entries' input node the value it registers as `scan/prog<j>`, and a member's public word its `<member>/o0` entry's
root, which `okChecks` holds to the constant), which one-stage's verifier checks inside the
answer, and each connection's cap against its lane (the parameters' `cells`; the network certifier holds a lane to
`egress_cells`); that a tied audit's `SelectEntry_v1` units read their `occ` from its `ScanSlots_v2` units, which read
its `SCAN_STATE` (the check reads no wires of an audit's Program, only its root templates' ids and closure digests);
and that the audits' answers are the sampled-proofs component's own,
which the composition's theorem is to bind.

**The clock openings** are apart from the period's check, in `NetworkCheck.Clocks` and this namespace: `clocksFn`,
`network-records verdicts`' `clocks`, with the decoders and the clock digest the period's check reads from there
(`exactly`, `hexAt`, `certificateOf`, `clockDigest`).
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace NetworkCheck.Period

open Lean (Json toJson)
open Flock (V)
open NetworkCertifier (ClockSync)
open NetworkCheck.Check (Certificate ratOf)
open NciCheck.DefinitionDigest (closureDigest)

def protocolId : String := "verity/certifier/v0"
def commitmentTag : String := "verity-network-certifier/period/v0"
def oneStage : String := "verity/sampled-proofs/one-stage/v1"
/-- A proof block's public outputs are at most 16 bits (the values it commits can be wider). -/
def outBits : Nat := 16
def unitLevel : String := "unit"
def scheduleTag : String := "verity-network-warden/parameters/v3"
def domainTag : String := "verity-network-warden/cells-domain/v2"
def cellsTag : String := "verity-network-warden/cells/v2"
def rowsFormat : String := "verity-network-warden/registration/v2"
def rowSchema : String := "hm96-sha512/row/v2"
/-- The cell kinds (`PAD`, `DATA`, `END`, `FAIL`, `REFUSED`) and the fail reasons the schedule's digest binds. -/
def cellKinds : List Nat := [0, 1, 32, 33, 34]
def failReasons : List String := ["late", "limit", "upstream", "ingress", "malformed", "format"]
/-- The fail reason a schedule with a response cap adds after `failReasons` (`warden.REASONS`' last): a request past
the cap, or whose `max_tokens` doesn't parse. -/
def capReason : String := "cap"
/-- The egress form (`WardenParams.egress`) of every schedule the check reads: the egress record's id form. A schedule
of another form has a digest no schedule here has. -/
def egressForm : String := "id"
/-- A cell's header: its kind u8, its connection's ordinal u32 and its payload's length u32. -/
def cellHeader : Nat := 9
/-- The byte of an id request past its `max_tokens` field (`active.MAX_TOKENS_AT + ID_DIGITS`): under a response cap a
cell's payload holds at least this much, so the network certifier reads the field off the request's first cell. -/
def maxTokensEnd : Nat := 130

/-- A compute audit's layout terms, its free inputs it doesn't register and its monitoring period, and what its Program
gives: its SHA-512, its units (its root's template instances), the most bits one of them commits, and its slot entries'
templates (`entries`: each root template node calling a `SlotEntry_v1` or `SelectEntry_v1` Definition, `entryIds`, by
its descriptor id, members and closure digest, `closureDigest`). -/
structure Audit where
  free : Nat
  steps : Nat
  program : String
  units : Nat
  bits : Nat
  entries : Array (String × Nat × String)

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
  /-- The response cap (`WardenParams.max_tokens_cap`), or none. -/
  maxTokensCap : Option Nat := none
  /-- The lengths rows' form (`WardenParams.lengths`), 1 or 2, or none. -/
  lengths : Option Nat := none
  /-- The content rows' classes (`WardenParams.content`), each (P, O, rows) under the lengths rows' form 1 and (P, O)
  under form 2, or none. -/
  content : Option (Array (Array Nat)) := none
  /-- Whether the schedule has request words (`WardenParams.request_words`, true or unset): a words row a release,
  in its class. -/
  requestWords : Bool := false

/-- The `fails` record's reasons under the schedule: `failReasons`, and `capReason` after them under a response cap
(`WardenParams.reasons`). -/
def Schedule.reasons (s : Schedule) : List String :=
  if s.maxTokensCap.isSome then failReasons ++ [capReason] else failReasons

/-- S = lanes · r, a row's slots. -/
def Schedule.allocation (s : Schedule) : Nat := s.lanes * s.slots

/-- A slot stride's Program slots, one a lane (`WardenParams.program_slots`). -/
def Schedule.programSlots (s : Schedule) : Nat := s.lanes

/-- A window's Program slots: T / `every` slot strides of `programSlots` (`WardenParams.window_programs`). -/
def Schedule.executionPhasePrograms (s : Schedule) : Nat := s.window / s.every * s.programSlots

/-- The most buckets from a connection's admission to its last egress cell's: D + ⌊(`egress_cells` − 1) / r⌋
(`WardenParams.span`). With lengths rows a connection ends by then, and it is at most T. -/
def Schedule.span (s : Schedule) : Nat := s.delay + (s.egressCells - 1) / s.slots

/-- The most requests the network certifier releases in one slot stride with lengths rows, a lengths row's slots
(`WardenParams.stride_releases`): a released connection holds its lane at least D + 2 − `ingress_buckets` buckets from
its release, so that many buckets hold at most `lanes` releases, and it holds it D + 1 buckets around its release, so
a stride's releases are live at one of 1 + ⌈(`every` − 1) / (D + 1)⌉ buckets. It is read where ingress ends by the
delay (`ingress_buckets` ≤ D), as a schedule with lengths rows has it. -/
def Schedule.strideReleases (s : Schedule) : Nat :=
  s.lanes * min (divCeil s.every (s.delay + 2 - s.ingressBuckets)) (1 + divCeil (s.every - 1) (s.delay + 1))

/-- A bound on the requests the network certifier releases in one execution phase with lengths rows
(`WardenParams.phase_releases`): `lanes` · ⌈(T + `ingress_buckets` − 1) / (D + 1)⌉, as a release is at most
`ingress_buckets` − 1 buckets after its connection's admission and the connections admitted in any D + 1 buckets in
a row are live together, at most `lanes` of them. -/
def Schedule.executionPhaseReleases (s : Schedule) : Nat :=
  s.lanes * divCeil (s.window + s.ingressBuckets - 1) (s.delay + 1)

/-- Whether the lengths rows end every execution phase of a run, its last included: their form 2
(`WardenParams.closes`). -/
def Schedule.closes (s : Schedule) : Bool := s.lengths == some 2

/-- Whether the schedule registers connection rows (`connection_rows`): its lengths rows, which a schedule with
content rows has too, as one with request words has content rows. Each is asked, since a `Schedule` the parse did not
build may have one without the other. -/
def Schedule.connectionRows (s : Schedule) : Bool := s.lengths.isSome || s.content.isSome || s.requestWords

/-- The rows form's cells over `windows` of a link's execution phases (`WardenParams.rows_cells`): `egressCells` for
each of the `lanes · windows · window` connections the network certifier can carry, as at most `lanes` are live in a
bucket and each is live in its admission bucket. -/
def Schedule.rowsCells (s : Schedule) (windows : Nat) : Array Nat :=
  Array.replicate (s.lanes * (windows * s.window)) s.egressCells

/-- **The rows form's frames read back** (`NetTiming.RowsFormat.Wf` at the network certifier's frames,
`active.encode_frame`: kind u8, ordinal u32, payload length u32, payload, zero fill): a cell's `B` bytes are under 2^32,
so its payload's length is a u32, and the `lanes · windows · window` connections of a period of `windows` of a link's
execution phases, the k-th at ordinal k + 1 (0 is a refusal's), have distinct u32 ordinals, so no two frames encode
alike. -/
def rowsWf (s : Schedule) (windows : Nat) : Bool :=
  s.cellBytes < 2 ^ 32 && s.lanes * (windows * s.window) < 2 ^ 32

def rowsWfRefusal (s : Schedule) (windows : Nat) : String :=
  s!"rows format: a frame's ordinal and payload length are u32s, so the lanes · windows · window = \
    {s.lanes * (windows * s.window)} connections and a cell's {s.cellBytes} bytes must each be under 2^32, or two \
    frames could encode alike (rowsWf)"

/-- The scan's terms, read off its Program: its monitoring period (`T`, the window's buckets), its own free inputs
(none: it registers the rows, and its `prog` is a compute audit's output, `progTied`, whose free inputs an execution
phase's term counts, `tiedAudits`), the bits one ingress and one egress instance commit (its return: the outputs other
units read or the client receives, and its verdict), and its slot entries' bits (`E_w`: `lanes · T / every` entries of
its `prog` ports' type), all a tied audit's incorrect proof blocks can set. -/
structure Scan where
  steps : Nat
  free : Nat
  ingress : Nat
  egress : Nat
  entries : Nat

/-- Which run the execution phases are: the clock form's, under Σ_sync, or the rows form's, under one schedule (and its
digest), read by the scan whose terms are the last. -/
inductive Form where
  | clock (syncSet : NetworkCertifier.SyncSet)
  | rows (schedule : Schedule) (digest : ByteArray) (scan : Scan)

/-- The response cap of the form's schedule: the rows form's `maxTokensCap`, none in the clock form. -/
def Form.maxTokensCap : Form → Option Nat
  | .clock _ => none
  | .rows s .. => s.maxTokensCap

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
  /-- The rows form's class circuit, its window circuit's Program's SHA-512, which every tied compute audit runs
  (`held`); none in the clock form, which ties no audit. -/
  circuit : Option String := none

/-- The one-stage window records of the compute audits and of the execution phases, in order. -/
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

/-- What an execution phase shows beside its answer and certificates: the clock form's clock certificate, or the rows
form's registrations. -/
inductive Shown where
  | clock (c : Certificate)
  | rows (rs : Array Rows)

structure ExecutionPhase where
  answer : Core.Answer
  certificates : Array Certificate
  shown : Shown

def ExecutionPhase.clockDigest? (x : ExecutionPhase) : Option ByteArray :=
  match x.shown with
  | .clock c => some c.digest
  | .rows _ => none

structure Evidence where
  audits : Array Core.Answer
  windows : Array ExecutionPhase

/-- What an accepted audit establishes: its proof blocks, its sample, its δ, and its bound on incorrect proof blocks. -/
structure Facts where
  units : Nat
  drawn : Nat
  delta : Core.Ratio
  wrong : Nat

/-! ## Decoding -/

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

/-- The keys every schedule has (`warden.INTEGER_FIELDS`). -/
def scheduleKeys : List String :=
  ["bucket_ms", "cell_bytes", "delay", "egress_cells", "every", "ingress_buckets", "ingress_cells", "lanes", "queue",
    "slots", "window"]

/-- The keys of a schedule that sets the optional keys `set` holds, in the order an object lists its keys. -/
def scheduleKeysWith (set : String → Bool) : List String :=
  ["bucket_ms", "cell_bytes", "content", "delay", "egress_cells", "every", "ingress_buckets", "ingress_cells", "lanes",
    "lengths", "max_tokens_cap", "queue", "request_words", "slots", "window"].filter fun k =>
      !optionalScheduleKeys.contains k || set k

/-- What `content_rows.classes` refuses of a schedule's content rows' classes, or none: a schedule with content rows
has lengths rows; it lists 1 to `mostClasses` classes, or to `mostClassesWords` with request words, each (P, O, rows)
under the lengths rows' form 1 and (P, O) under form 2; P and O are each a class's size, O at most `mostIds` and P
at most the bytes of a request, a prompt id at least one (`ingress_cells` · a cell's payload); and under form 1 a
class's tables have 1 to T / `every` · `strideReleases` rows, the lengths table's slots, since an execution phase
releases no more. -/
def classesRefusal (s : Schedule) (classes : Array (Array Nat)) : Option String :=
  match s.lengths with
  | none => some "content: the content rows' tables are listed in the connection rows' head, so a schedule with \
      content has lengths rows (lengths is 1 or 2)"
  | some form =>
    let slots := s.window / s.every * s.strideReleases
    let most := if s.requestWords then mostClassesWords else mostClasses
    if classes.isEmpty || most < classes.size then
      some s!"content: 1 to {most} classes, not {classes.size}"
    else (classes.zipIdx.findSome? fun (c, i) =>
      let p := c.getD 0 0
      let o := c.getD 1 0
      if c.size != (if form == 2 then 2 else 3) then
        some s!"class {i} is (P, O) under the lengths rows' form 2 and (P, O, rows) under form 1, not {c.size} numbers"
      else if !(classSize p && classSize o) then
        some s!"class {i}: its sizes of prompt ids and of ids released are each a power of two of at least 2"
      else if mostIds < o then some s!"class {i}: a response row is at most {mostIds} ids, not O = {o}"
      else if s.ingressCells * (s.cellBytes - cellHeader) < p then
        some s!"class {i} has P = {p} prompt ids, and a request of the schedule is at most \
          {s.ingressCells * (s.cellBytes - cellHeader)} bytes (ingress_cells · a cell's payload), each id at least one"
      else if form != 2 && !(1 ≤ c.getD 2 0 && c.getD 2 0 ≤ slots) then
        some s!"class {i} has 1 to {slots} rows, the lengths table's slots, since an execution phase releases no more"
      else none).map ("content " ++ ·)

def scheduleOf (j : Json) : V Schedule := do
  exactly j (scheduleKeysWith (Flock.J.has j)) "schedule"
  let pos : String → V Nat := fun k => do
    let x ← Flock.J.nat j k "schedule"
    unless 1 ≤ x do throw s!"schedule: {k} is a positive integer"
    pure x
  let opt : String → V (Option Nat) := fun k => if Flock.J.has j k then some <$> pos k else pure none
  let content ← if Flock.J.has j "content" then do
      let classes ← (← Flock.J.arr j "content" "schedule").mapM fun c => do
        (← Flock.J.asArr c "schedule.content").mapM (Flock.J.asNat · "schedule.content")
      pure (some classes)
    else pure none
  let requestWords ← if Flock.J.has j "request_words" then do
      unless (← Flock.J.bool j "request_words" "schedule") do
        throw "schedule: request_words is true, for a schedule with request words, or unset"
      pure true
    else pure false
  let s : Schedule :=
    { bucketMs := ← pos "bucket_ms", cellBytes := ← pos "cell_bytes", delay := ← pos "delay",
      egressCells := ← pos "egress_cells", every := ← pos "every", ingressBuckets := ← pos "ingress_buckets",
      ingressCells := ← pos "ingress_cells", lanes := ← pos "lanes", queue := ← pos "queue", slots := ← pos "slots",
      window := ← pos "window", maxTokensCap := ← opt "max_tokens_cap", lengths := ← opt "lengths", content,
      requestWords }
  unless cellHeader + 1 < s.cellBytes do throw s!"schedule: a cell holds its {cellHeader}-byte header and a byte"
  unless s.cellBytes < 2 ^ 32 ∧ s.allocation < 2 ^ 32 ∧ s.window < 2 ^ 32 do
    throw "schedule: B, S = lanes · slots and T = window are u32s"
  unless s.window % s.every == 0 do throw "schedule: every divides the window: a window is whole timesteps"
  if let some cap := s.maxTokensCap then
    unless cap < 2 ^ 32 ∧ cap < s.egressCells do
      throw "schedule: max_tokens_cap is a u32 below egress_cells: a capped response's END is its egress cell \
        max_tokens"
    unless maxTokensEnd + cellHeader ≤ s.cellBytes do
      throw s!"schedule: under a response cap a cell is at least {maxTokensEnd + cellHeader} bytes, so it holds an id \
        request's max_tokens"
  if let some form := s.lengths then
    unless form == 1 ∨ form == 2 do throw s!"schedule: lengths is 1 or 2, the lengths rows' forms, not {form}"
    unless s.ingressBuckets ≤ s.delay do
      throw s!"schedule: with lengths rows ingress_buckets ({s.ingressBuckets}) is at most the delay ({s.delay}): a \
        released connection holds its lane from its release"
    unless s.span ≤ s.window ∧ s.egressCells ≤ 2 ^ 16 do
      throw s!"schedule: with lengths rows a connection ends within a window of its admission and its egress cells \
        count in a u16: delay + (egress_cells − 1) / slots ({s.span}) is at most the window ({s.window}) and \
        egress_cells at most {2 ^ 16}"
    unless firstPromptEnd + cellHeader ≤ s.cellBytes do
      throw s!"schedule: with lengths rows a cell is at least {firstPromptEnd + cellHeader} bytes, so its first holds \
        an id request's head and first prompt id"
    unless s.strideReleases ≤ mostSlots do
      throw s!"schedule: a lengths row is at most {mostSlots} slots, and this schedule's slot stride releases up to \
        {s.strideReleases} requests, a slot each"
  if s.requestWords then
    unless s.content.isSome ∧ s.closes ∧ s.maxTokensCap.isSome do
      throw "schedule: a words row is a release's in its class, one for each the class holds, and its max_tokens the \
        one its response is held to, so a schedule with request words has content rows under the lengths rows' form \
        2 and max_tokens_cap"
  if let some classes := s.content then
    if let some why := classesRefusal s classes then throw s!"schedule: {why}"
  pure s

/-- `WardenParams.digest`: the schedule's canonical JSON, with its egress form, the cell kinds and the fail reasons,
tagged. A schedule without a response cap has neither the cap nor `capReason` in it, so its digest is the one it had
before the cap was a parameter. So with the lengths rows' form, the content rows' classes and the request words: a
schedule without one has nothing of it in its digest. -/
def Schedule.digest (s : Schedule) : V ByteArray :=
  Identity.identityDigestSha512 scheduleTag (Json.mkObj ([
    ("bucket_ms", toJson s.bucketMs), ("cell_bytes", toJson s.cellBytes), ("delay", toJson s.delay),
    ("egress", toJson egressForm), ("egress_cells", toJson s.egressCells), ("every", toJson s.every),
    ("ingress_buckets", toJson s.ingressBuckets),
    ("ingress_cells", toJson s.ingressCells), ("lanes", toJson s.lanes), ("queue", toJson s.queue),
    ("slots", toJson s.slots), ("window", toJson s.window), ("kinds", toJson cellKinds),
    ("reasons", toJson s.reasons)] ++ (s.maxTokensCap.map fun cap => ("max_tokens_cap", toJson cap)).toList ++
    (s.lengths.map fun form => ("lengths", toJson form)).toList ++
    (s.content.map fun classes => ("content", toJson classes)).toList ++
    (if s.requestWords then [("request_words", toJson true)] else [])))

/-- The scan's templates' parameters in the generator form: the start unit's (`ingress_scan.IngressScanStart`, the state
the execution phase's scan starts from, which it copies), each ingress bucket's (`ingress_scan.IngressScanBucket`) and
the egress bucket's (`egress.EgressBucket`), which computes its row and returns it. An ingress bucket's `prog<j>` is
lane
`j`'s slot entry, one a lane. -/
def progParams (lanes : Nat) : Array String := (Array.range lanes).map (s!"prog{·}")
def startParams : Array String := #["sin"]
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

/-- Whether the arguments `own` (the templates' row, state and egress arguments) read input nodes of their own: their
input nodes (`inputArg`) are pairwise distinct, and none is one of `pks`, the slot entries' input nodes. At the cost of
the arguments. -/
def argsApart (P : Flock.Program.Prog) (own : List Flock.Program.Arg) (pks : List Nat) : Bool :=
  let ks := own.map (inputArg P)
  decide ks.Nodup && pks.all fun pk => !ks.contains (some pk)

/-- Whether the root's input nodes hold what the scan reads and no other leaf: each argument of `own` is one run over
all its node's leaves (as many as the node has members), each node of `pks` has `entries` leaves, and every input node
of the root is one of `pks` or an argument's of `own`. At the cost of the root's nodes, each against the arguments. -/
def inputsRead (P : Flock.Program.Prog) (own : List Flock.Program.Arg) (pks : List Nat) (entries : Nat) : Bool :=
  let nodes := (P.defs[P.root]!).body.nodes
  let ks := own.map (inputArg P)
  own.all (fun a => match a.seq.parts.toList with
    | [.affine false k 0 c] => nodes[k]?.map (·.members) == some c
    | _ => false) &&
  pks.all (fun pk => nodes[pk]?.map (·.members) == some entries) &&
  (List.range nodes.size).all fun k => !NciCheck.Desc.isInputNode P nodes[k]! || pks.contains k || ks.contains (some k)

/-- Whether two types have as many leaves, each of one width, the same in both (`Flock.Program.Widths.homo`): what a
tie of two rows of gates, leaf by leaf, needs of their types. From the types' structure, no walk over their leaves. -/
def sameLeaves (a b : Flock.Program.Ty) : Bool :=
  a.leaves == b.leaves && (Flock.Program.Widths.homo a).isSome &&
    Flock.Program.Widths.homo a == Flock.Program.Widths.homo b

/-- Whether `ret` is the generator form's ingress return, over the leaves the scan's two ties equate: a tuple of
exactly a u16, the state the next bucket reads and the view the bucket's egress unit reads, in that order. The state
has the leaves of `sin`, the ingress template's own parameter of that name, and the view the leaves of `state`, the
egress template's parameter (`sameLeaves`: a view of bits is read as u16 words, so the types themselves differ). -/
def ingressReturn (ret sin state : Flock.Program.Ty) : Bool :=
  match ret with
  | .tuple fs => fs.size == 3 && fs[0]! == .array outBits (.value 1) && sameLeaves fs[1]! sin &&
      sameLeaves fs[2]! state
  | _ => false

/-- Whether `ret` and `own`, the start template's return and its own parameter, are a state's, over the leaves its
two ties equate: `ret` a tuple of exactly one row, and that row and `own` each of the leaves of `sin`, an ingress
bucket's parameter of that name. The row it returns is the one bucket 0's `sin` reads, and its parameter is read
where an ingress bucket's returned state is committed (the last bucket's of the execution phase before), so each tie
equates rows of one length. -/
def startReturn (ret own sin : Flock.Program.Ty) : Bool :=
  match ret with
  | .tuple fs => fs.size == 1 && sameLeaves fs[0]! sin && sameLeaves own sin
  | _ => false

/-- Refuses with `why` unless `b`: one step of a `do` block, so what its acceptance gives is read off it step by step
(`Properties.NetworkScan`). -/
def need (b : Bool) (why : String) : V Unit := if b then pure () else throw why

/-- `o`'s value, refusing with `why` where there is none, as one step. -/
def needSome {α : Type} (o : Option α) (why : String) : V α := match o with
  | some a => pure a
  | none => throw why

/-- The generator form's templates before the egress template's: the start unit's, then the ingress buckets'. -/
def beforeEgress : List (Flock.Program.Node × Flock.Program.Defn) →
    V ((Flock.Program.Node × Flock.Program.Defn) × (Flock.Program.Node × Flock.Program.Defn))
  | [st, b] => pure (st, b)
  | _ => throw "scan: not the generator form's templates: the start unit's, the ingress buckets' and the egress \
    bucket's"

/-- The slot entry's Definition (`io_shell.SlotEntry_v1`): one unit a due, occupied slot's Program, committing the
entry the scan's ingress reads at that slot (its `o0`). -/
def slotEntryId : String := "SlotEntry_v1"

/-- The slot's last position (`SelectEntry_v1`, pinned by `selectEntryPins`): one unit a due, occupied slot's Program,
whose select and collector end in `SlotEntry_v1`, so its `o0` is the slot's entry. -/
def selectEntryId : String := "SelectEntry_v1"

/-- The Definitions whose `o0` is a slot's entry: `SlotEntry_v1` and `SelectEntry_v1`. -/
def entryIds : List String := [slotEntryId, selectEntryId]

/-- The slot entry's Definitions the check takes, by closure digest (`closureDigest`): `io_shell.SlotEntry_v1` at
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

/-- The slot's last position's Definitions the check takes, by closure digest (`closureDigest`, which covers the
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

/-- A compute audit's layout terms and its Program, from the Program's descriptor (`Flock.Program.decode` of its
canonical bytes): its units are its root's template instances, each committing its template's return; a root with no
template is refused. Its slot entries' templates are its root's template nodes whose Definition is `SlotEntry_v1` or
`SelectEntry_v1` (`entryIds`), each with its closure digest (`closureDigest`). -/
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
    | some d => if !d.prim && entryIds.contains d.id then some (d.key, x.members) else none
    | none => none
  let entries ← slots.mapM fun (k, n) => do
    pure (k, n, ← (closureDigest pj k).mapError (s!"{ctx}: " ++ ·))
  pure { free := ← Flock.J.nat a "free" ctx, steps := ← Flock.J.nat a "steps" ctx, program := Flock.hexOf prog.sha512,
         units := templates.foldl (· + ·.1) 0, bits := templates.foldl (fun b t => max b t.2) 0, entries }

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
one instance of the start template, whose one parameter is `sin` and which returns the state alone, `T` of the ingress
buckets' template, each one's `row` the schedule's row (`S · B` bytes), then `T` of the egress template, which returns
a u16 and the schedule's row, and every input gate is a template's parameter. The types are the Boolean profile's, as
the scan's Definitions have them: a byte 8 bits, a u16 16. An egress template with a row among its parameters, the
checker form, is refused first, by name: there the egress row is an input gate, which `NetTiming.BoundaryGates.Fits`
refuses, and `K` would count the free inputs of a Program that only compares its egress rows.

Then what `NetTiming.BoundaryGates.Fits` asks of the layout the templates give (`Properties.NetworkScan.scanLayout`), read off
the description, at the cost of its nodes and returns: the Program is laid out as Flock's decoder lays one out and
within what its resolution walks (`NciCheck.Desc`'s `rootLaidOut`, `inputsAtRoot` and `walkable`); the root returns
every egress leaf (one affine run from the egress node's first leaf over all `T` instances'); the egress template
returns only gates it computes (`ownReturns`: no parameter's leaf passed through, so no egress gate is an input); the
ingress template's row, and every template's row, state and egress argument, is one input node's leaves (`inputArg`),
so each ingress gate is an input and a template reads only input gates outside its own; no egress template's argument
is the ingress rows' input node, so no egress gate depends on an ingress gate; and the Program is topological
(`Prog.orderViolation`, as `NciCheck.Desc.evaluate` checks it), so a gate reads only gates before it.

The ingress template's entry ports `prog<j>` all have one type, a slot entry's, and lane `j`'s read one input node of
its own, distinct from every other lane's and from the rows, lane `j`'s slot entries (`T / every` rows each, `lanes ·
T / every` counted once among the input gates): bucket `t`'s `prog<j>` is its row `⌊t / every⌋` (`readsEntries`), the
leaf the period check's `wired` holds to chunk `j` of an audit's `SlotEntry_v1` or `SelectEntry_v1` entries port;
their bits are the scan's `entries`.

The scan's ties equate rows of one length: a bucket's `sin` with the state the bucket before returned, bucket 0's
with the state the start unit returned, an egress bucket's `state` with the view its ingress bucket returned, and the
start unit's `sin` with the state the execution phase before's last bucket returned. So the ingress template returns
exactly a u16, a state of `sin`'s leaves and a view of the egress template's `state`'s leaves (`ingressReturn`), and
the start template takes and returns exactly a state of those leaves (`startReturn`); a return of another shape is
refused. A start unit so commits no more bits than an ingress bucket does, so the wider of an ingress and an egress
instance's bits (`Scan.terms`) bounds what any incorrect proof block of the scan commits.

An input gate is read as one thing: the row, state and egress arguments read input nodes of their own, none a lane's
slot entries' (`argsApart`). The input gates are counted as a sum of bits against the parameters' and the slot
entries', so two arguments on one node would leave that sum room for an input node nothing reads, and a state argument
on a lane's slot entries' node would make a slot entry a state gate.

And every input gate is read: each of those arguments is all its node's leaves, each lane's slot entries' node holds
its `T / every` entries and no other leaf, and the root has no other input node (`inputsRead`). That sum counts bits
and this counts leaves, node by node, so the scan's input gates are read off the check itself, with no appeal to the
widths the decoder holds a reference to. A slot entry's leaves are at most its bits (a leaf is at least a bit wide,
which the decoder gives and this holds), so the entries' bits bound the entries' gates.

The scan's own free inputs are none: it registers the rows, its `prog` is its tied audits' output, each bucket's
`sin` is the state the unit before it returned, and the start unit's `sin` is the record's carried start, which the
check holds (`startCheck`). -/
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
  let ((xs, ds), (xi, di)) ← beforeEgress templates.pop.toList
  need (ds.params.map (·.1) == startParams && di.params.map (·.1) == ingressParams s.lanes &&
      de.params.map (·.1) == egressParams)
    s!"scan: the templates' parameters are not {startParams}, {ingressParams s.lanes} and {egressParams}"
  need (xs.form == .batch && xs.n == 1 && xi.form == .batch && xi.n == T && xe.form == .batch && xe.n == T)
    s!"scan: not the start unit, {T} ingress buckets and {T} egress buckets, the link-window's"
  need (di.params[0]!.2 == row) "scan: the ingress template's row is not the schedule's"
  let ety := di.params[2]!.2
  need ((di.params.toList.drop 2).all (·.2 == ety)) "scan: the ingress template's entry ports' types differ"
  need (de.ret == .tuple #[.array outBits (.value 1), row])
    "scan: the egress template doesn't return a u16 and the schedule's row"
  need (ingressReturn di.ret di.params[1]!.2 de.params[0]!.2 && startReturn ds.ret ds.params[0]!.2 di.params[1]!.2)
    "scan: the ingress template doesn't return a u16, a state of its sin's leaves and a view of an egress bucket's \
      state's leaves, or the start template doesn't take and return a state of those leaves, so a tie of the scan's \
      state or view would equate rows of two lengths (NetworkCheck.Period.ingressReturn, startReturn)"
  let inputs := root.body.nodes.foldl (fun n x => n + match prog.defs[x.fn]? with
    | some d => ((Flock.Program.inputWidth? d.key).map (· * x.members)).getD 0
    | none => 0) 0
  let nb := T / s.every
  need (inputs == paramBits ds.params + T * paramBits (di.params.extract 0 2) + T * paramBits de.params +
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
  let rows := xi.args[0]?.bind (inputArg prog)
  need rows.isSome "scan: the ingress template's row is not one input node's leaves, so an ingress gate would be no \
    input (NetTiming.BoundaryGates.Fits)"
  need (xs.args.size == 1 && xi.args.size == 2 + s.lanes && xs.args.all (fun a => (inputArg prog a).isSome) &&
      (xi.args.extract 0 2).all (fun a => (inputArg prog a).isSome) && xe.args.all fun a => (inputArg prog a).isSome)
    "scan: a template's row, state or egress argument is not one input node's leaves"
  need (xe.args.all fun a => inputArg prog a != rows) "scan: an egress template's argument is the ingress rows' \
    input node, so an egress gate would depend on the ingress (NetTiming.BoundaryGates.Fits)"
  let a0 ← needSome ((List.range s.lanes).mapM fun j => xi.args[2 + j]?) "scan: bucket 0 has no entry port a lane"
  let pks := a0.map fun a => (a.memberRef 0 0).i
  need (a0.all (fun a => !(a.memberRef 0 0).p) && decide pks.Nodup && pks.all fun pk =>
      rows != some pk && match root.body.nodes[pk]? with
        | some n => NciCheck.Desc.isInputNode prog n
        | none => false)
    "scan: bucket 0's prog<j> reads no input node of lane j's own, its slot entries', distinct from every other lane's"
  need (readsEntries xi 2 0 s.every ety.leaves pks)
    s!"scan: bucket t's prog<j> is not row ⌊t / {s.every}⌋ of lane j's slot entries' input node, its slot of the \
      boundary bucket t reads (NetworkCheck.Period.readsEntries)"
  let own := (xs.args.extract 0 1 ++ xi.args.extract 0 2 ++ xe.args).toList
  need (argsApart prog own pks) "scan: two of the templates' row, state and egress arguments read one input node, or \
    one reads a lane's slot entries': an input node would be left that nothing reads, or a slot entry would be a \
    state gate (NetworkCheck.Period.argsApart)"
  need (inputsRead prog own pks (nb * ety.leaves) && ety.leaves ≤ bitsOf ety)
    s!"scan: an input node with a leaf nothing reads: a row, state or egress argument that is not all its input \
      node's leaves, a lane's slot entries' node of other leaves than its {nb} entries' (each of at least a bit), \
      or an input node that is neither's (NetworkCheck.Period.inputsRead)"
  need prog.orderViolation.isNone "scan: a body names a later node (Flock.Program.Prog.orderViolation): the Program \
    is not topological, so a gate's operands could be gates after it"
  pure { steps := T, free := 0, ingress := bitsOf di.ret, egress := bitsOf de.ret,
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
  unless session.size = attachmentBytes do throw s!"a certifier session id is {attachmentBytes} bytes, not {session.size}"
  let program ← Flock.J.str j "program" "the object"
  unless isDigest program do throw "program: not a SHA-512 digest (128 lowercase hex digits)"
  let form ← if keysOf j == paramsKeys then do pure (Form.clock (← syncSetOf (← Flock.J.get j "sync_set" "the object")))
    else do
      let s ← scheduleOf (← Flock.J.get j "schedule" "the object")
      if s.connectionRows then throw connectionRowsRefusal
      pure (Form.rows s (← s.digest) (← scanOf (← Flock.J.get j "scan" "the object") program s))
  let windows ← Flock.J.nat j "windows" "the object"
  unless 1 ≤ windows do throw "windows: a period has at least one link-window"
  if let .rows s _ _ := form then
    unless rowsWf s windows do throw (rowsWfRefusal s windows)
    unless cells == s.rowsCells windows do
      throw s!"cells: the rows form's cells are egress_cells = {s.egressCells} for each of the lanes · windows · \
        window = {s.lanes * (windows * s.window)} connections the schedule can carry (Schedule.rowsCells), not these \
        {cells.size}"
  let circuit ← if keysOf j == paramsKeys then pure none else do
    let d ← Flock.J.str j "circuit" "the object"
    unless isDigest d do throw "circuit: not a SHA-512 digest (128 lowercase hex digits)"
    pure (some d)
  pure { audits, cells, first, jitter, link, lms, key, session, program, form, windows, circuit }

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

def executionPhaseOf (j : Json) (ctx : String) : V ExecutionPhase := do
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
  let windows ← (← Flock.J.arr j "windows" "the object").mapM (executionPhaseOf · "a link-window")
  pure { audits, windows }

/-! ## The check -/

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

/-- Execution phase `w`'s findings, in the order the module docstring gives, `clear` being `clearClocks` of Σ_sync and
`earlier` the period's earlier execution phases' clock digests. -/
def executionPhaseCheck (p : Params) (clear : List ByteArray) (earlier : Array ByteArray) (w : Nat) (reg : Json)
    (x : ExecutionPhase) : Found Unit := do
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

/-- `WardenParams()`'s schedule, the defaults. -/
def defaultSchedule : Schedule :=
  { bucketMs := 100, cellBytes := 256, delay := 10, egressCells := 4096, every := 1, ingressBuckets := 50,
    ingressCells := 64, lanes := 16, queue := 64, slots := 4, window := 600 }

-- `WardenParams().digest` and `WardenParams(max_tokens_cap=4).digest`: the cap and its reason are in the digest only
-- under a cap
#guard (defaultSchedule.digest.map Flock.hexOf).toOption == some
  "9a727d16f6c1f351d316465c101269c2a4666cbcdf1dbaaef01579fd7e72315a\
    ae2be1451412c6328d9f6688f534fb8bbe5416bbaeb5cfc0cae48597c5fd0d50"
#guard (({ defaultSchedule with maxTokensCap := some 4 } : Schedule).digest.map Flock.hexOf).toOption == some
  "8832fdae0616446e90e1b4178ab99c0e3d67bf790d5bb81512f13c651767adc3\
    0a29f9401b9eb0a2cdf2a8d1c02c50304353250f06af526a970074a207bf3d12"

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
(`Schedule.executionPhasePrograms`). Its members are cut in order, so chunk `j` is units `[j · T / every, (j + 1) · T / every)`
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
  (audits.zip regs).any fun (a, r) => a.entries.any fun (k, n, d) => n == s.executionPhasePrograms &&
    entryPins.contains d && (List.range s.programSlots).all fun j =>
      (prog j).any fun e => match e.getObjValD "root", e.getObjValD "domain" with
        | .str root, .str dom => (entries r "roots").any fun o =>
            named "port" (entriesPort k j) o && leaves (s.window / s.every) o && named "root" root o &&
              named "domain" dom o
        | _, _ => false

/-- The scan's state in a registration: the entry of its `roots` whose port is `scan/state`. -/
def stateName : String := "scan/state"

/-- The value a window circuit registers the scan's state rows as (the certifier's `blocks.SCAN_STATE`), row a the state
before bucket a: the entry of an audit's `registered`, or of the scan's, whose value it is. -/
def stateValue : String := "verity/certifier/scan-state/v1"

/-- A registration's state entries: those of its `roots` whose port is `stateName` and of its `registered` whose value
is `stateValue`. -/
def stateEntries (reg : Json) : Array Json :=
  let entries := fun (k : String) => match reg.getObjValD k with | .arr xs => xs | _ => #[]
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  (entries "roots").filter (named "port" stateName) ++ (entries "registered").filter (named "value" stateValue)

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
`reads` of `stateName` has `o0` the run of unit 0 at leaf 0, `sin` the run of units 1 to T from leaf 0 and `o1` the run
of units 1 to T − 1 from leaf 1 (`isRun`). Unit 0 is the start unit and unit `t + 1` bucket `t`: one-stage numbers the
scan's units in its root's order, the start unit first and then the ingress buckets' (`scanOf`). So the start unit
registers its `o0`, the state the execution phase starts from, at leaf 0, and ingress bucket `t` reads `sin` at leaf
`t`, S_{t−1}, and registers its `o1`, S_t, at leaf `t + 1`. The last ingress bucket's `o1` no leaf holds: it is a port
of its own, the one the next execution phase's start is read from. One run a port, so the check reads the description,
never a unit. -/
def stateRead (s : Schedule) (reg : Json) : Bool :=
  let r := ((reg.getObjValD "window").getObjValD "reads").getObjValD stateName
  isRun (r.getObjValD "o0") 0 1 0 && isRun (r.getObjValD "sin") 1 (s.window + 1) 0 &&
    isRun (r.getObjValD "o1") 1 s.window 1

#guard
  let s : Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 3, none, none, none, false⟩
  let run := fun (l lo hi : Nat) => Json.mkObj [("leaf", toJson l), ("units", toJson [lo, hi])]
  let reads := fun (o0 sin o1 : Json) => Json.mkObj [("window", Json.mkObj [("reads", Json.mkObj
    [(stateName, Json.mkObj [("o0", o0), ("o1", o1), ("sin", sin)])])])]
  stateRead s (reads (run 0 0 1) (run 0 1 4) (run 1 1 3)) &&
    !stateRead s (reads (run 0 0 1) (run 1 1 4) (run 1 1 3)) &&
    !stateRead s (reads (run 0 0 1) (run 0 1 4) (run 1 0 2)) &&
    !stateRead s (reads (run 1 0 1) (run 0 1 4) (run 1 1 3)) &&
    !stateRead s (reads Json.null (run 0 1 4) (run 1 1 3))

/-- The scan's state is the state its slots' Programs read, read where the scan's transitions chain it: the scan's
registration states its state reads as the chain (`stateRead`: the start unit's `o0` at leaf 0, bucket `t`'s `sin` at
leaf `t` and its `o1` at leaf `t + 1`, but the last bucket's `o1`, which no leaf holds), has a state entry
(`stateEntries`), every one at one root and domain, and an audit `wired` to the scan's `prog` has one, every one at
that root and domain. So the `SCAN_STATE` that audit's Programs read is the chain of the scan's transitions, each row
the one the unit before registered, row 0 the start unit's. -/
def stateTied (audits : Array Audit) (s : Schedule) (regs : Array Json) (reg : Json) : Bool :=
  let named := fun (k n : String) (e : Json) => match e.getObjValD k with | .str x => x == n | _ => false
  stateRead s reg && match (stateEntries reg)[0]? with
  | some e => match e.getObjValD "root", e.getObjValD "domain" with
    | .str root, .str dom =>
      let atRoot := fun (o : Json) => named "root" root o && named "domain" dom o
      (stateEntries reg).all atRoot && (audits.zip regs).any fun (a, r) =>
        wired #[a] s #[r] reg && !(stateEntries r).isEmpty && (stateEntries r).all atRoot
    | _, _ => false
  | none => false

/-- The value an execution phase's scan starts from, in its registration: the start unit's `sin`, the first parameter of
the scan's Program, as one-stage names a registered value (`<template's index>.<parameter>`, the scan's driver's
`CARRY`). -/
def startName : String := "0.sin"

/-- A registration's start entries: those of its `registered` whose value is `startName`. -/
def startEntries (reg : Json) : Array Json :=
  let named := fun (e : Json) => match e.getObjValD "value" with | .str x => x == startName | _ => false
  match reg.getObjValD "registered" with | .arr xs => xs.filter named | _ => #[]

/-- The scan's registration starts from the empty state: it has a start entry (`startEntries`), and its `window`'s
`"empty": {"0.sin": {"words": W, "salt": hex}}` opens every one to `W` zero u16 words (`W` at least 1) under a salt of
`HmRow.SALT_BYTES` bytes: the hm96 leaf of that row (`Registered.knownPayload`, as the verifier registers a value at
known content), framed at 0 under the entry's domain, is the entry's root, the root of a tree of that one leaf. The
start unit reads its `sin` there and copies it to row 0 of the scan's state, the state before bucket 0, which the
Programs anchored at bucket 0 read; a row's digest binds its length, so a reader of another width doesn't open it. The
row is public, the empty state, and so is its salt. -/
def startEmpty (reg : Json) : Bool :=
  let o := ((reg.getObjValD "window").getObjValD "empty").getObjValD startName
  let hex := fun (k : String) (j : Json) => match j.getObjValD k with
    | .str x => (Flock.unhex x).toOption
    | _ => none
  let es := startEntries reg
  !es.isEmpty && es.all fun e => match (o.getObjValD "words").getNat?, hex "salt" o, hex "root" e, hex "domain" e with
    | .ok w, some salt, some root, some dom =>
      decide (0 < w) && decide (salt.size = Flock.HmRow.SALT_BYTES) &&
        decide (Flock.FV3.Sha512.leafAs Flock.HmRow.ROW_SCHEMA_V2 dom 0
          (Flock.Registered.knownPayload (Array.replicate w 0) salt) = root)
    | _, _, _, _ => false

/-- A registration whose start entries are at `roots`, each under the domain of 64 bytes `0d`, opened as `words` zero
words under the salt of 192 bytes `07`. -/
def startProbe (roots : List String) (words : Nat) : Json :=
  let entry := fun (root : String) => Json.mkObj
    [("domain", .str (String.join (List.replicate 64 "0d"))), ("root", .str root), ("value", .str startName)]
  Json.mkObj [("registered", .arr (roots.map entry).toArray), ("window", Json.mkObj [("empty", Json.mkObj
    [(startName, Json.mkObj [("salt", .str (String.join (List.replicate 192 "07"))), ("words", toJson words)])])])]

-- the roots of one zero word and of two under that salt and domain (the mirror's `start_empty`), and of the word 1
#guard
  let one := "cc8a8f235c85967255dadcfea4db8f2534aa0a12d0c3e9c289462c14072a92cc\
    3c65bfd17a3df01f6dcb91eec5ff7a6cb18294dc0ad07ab56a1e2fcd341161e0"
  let two := "ea02849e4fcf22320c9818f4da9dcd116bd876c29c37e98672c7d5c65bd98c5c\
    efd29d7568977d2d3396a404ca2e6c7582a7280a71c9acdfd0dd30b8822fd27b"
  let chosen := "ef7087146312d7acc2a957d16a9375942589fb284a8b7d7f0b6f8b4544ef9a90\
    3312d70b3b50a232f4ace84ee521f0601cd99008a52c1b98315ba10d15c81a51"
  startEmpty (startProbe [one] 1) && startEmpty (startProbe [two] 2) && startEmpty (startProbe [one, one] 1) &&
    !startEmpty (startProbe [one] 2) && !startEmpty (startProbe [two] 1) && !startEmpty (startProbe [chosen] 1) &&
    !startEmpty (startProbe [] 1) && !startEmpty (startProbe [one, two] 1) && !startEmpty (startProbe [one] 0) &&
    !startEmpty Json.null

/-- Audit `i` is tied to an execution phase's scan: some execution phase's registration's `prog` is its output
(`tiedAudits`). -/
def tiedIn (audits windows : Array Json) (i : Nat) : Bool :=
  windows.any fun reg => (tiedAudits audits reg).contains i

/-- The audits the form ties to an execution phase's scan: the rows form's `tiedIn`, and none in the clock form. -/
def tiedOf : Form → Commitment → Nat → Bool
  | .clock _, _ => fun _ => false
  | .rows .., c => tiedIn c.audits c.windows

/-- The certified records of the rows form's execution phase: the requests' (ingress), responses' (egress) and fails'.
-/
def records (link : String) : Array (String × Direction) :=
  #[(link, .ingress), (link, .egress), (link ++ "-fails", .egress)]

/-- An accepted scan's or audit's facts: one level sampled `subset:k` or stratified (a work sample's profile is
stratified, its `Σ k_s` sampled), and its bound. A profile that names more sampled than its level holds is no sample
of the level and is refused. So is a stratified one that names more sampled of a stratum than the stratum holds, or
whose strata's sizes do not sum to the level's size: with both held, a level sampled whole is every stratum sampled
whole. -/
def factsOf (what : String) (prof : Core.Integrity) (pub : Json) : Found Facts := do
  let #[l] := prof.levels | refuse s!"{what}: profile: not one level"
  let drawn ← match l.draw with
    | some (.subset k) => pure k
    | some (.stratified ss) => pure (ss.foldl (fun n st => n + st.k) 0)
    | _ => refuse s!"{what}: profile: not a subset or stratified draw"
  unless drawn ≤ l.size do refuse s!"{what}: profile: {drawn} sampled of {l.size} units: more than the level holds"
  if let some (.stratified ss) := l.draw then
    if let some st := ss.find? (fun st => st.size < st.k) then
      refuse s!"{what}: profile: stratum '{st.name}': {st.k} sampled of {st.size} units: more than the stratum holds"
    let held := ss.foldl (fun n st => n + st.size) 0
    unless held = l.size do
      refuse s!"{what}: profile: strata of {held} units in a level of {l.size}: their sizes do not sum to the level's"
  match Flock.J.nat pub "bound" "public" with
  | .ok b => pure { units := l.size, drawn, delta := prof.delta, wrong := b }
  | .error e => refuse s!"{what}: {e}"

/-- Execution phase `w`'s findings in the rows form, in the order the module docstring gives, and on acceptance its
scan's facts and its tied audits' free inputs (`tiedAudits`: the parameters' `free` of each audit whose output the
scan's
`prog` is, each once). -/
def rowsExecutionPhaseCheck (p : Params) (s : Schedule) (sd : ByteArray) (audits : Array Json) (w : Nat)
    (reg : Json) (x : ExecutionPhase) : Found (Facts × Nat) := do
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
  unless f.units == 2 * s.window + 1 do
    refuse s!"{what}: profile: {f.units} units, not the scan's 2T + 1 = {2 * s.window + 1}"
  pure (f, (tiedAudits audits reg).foldl (fun n i => n + ((p.audits[i]?.map (·.free)).getD 0)) 0)

/-- Audit `i`'s findings, and on acceptance its facts: its registration names its Program and the answer's proof
blocks are the Program's, at any sample; a tied audit sampled in part is charged in its execution phases' terms
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

/-- Execution phase `w`'s wiring finding in the rows form: a refusal unless its scan's Programs' row is a compute
audit's slot entries (`wired`). -/
def wiredCheck (p : Params) (s : Schedule) (audits : Array Json) (w : Nat) (reg : Json) :
    Option (Core.Verdict × String) :=
  if wired p.audits s audits reg then none else
    some (.reject, s!"window {w}: the scan's prog is no compute audit's slot entries: its scan/prog<j> entries' roots \
      and domains, j < {s.programSlots}, are not one audit's {slotEntryId} or {selectEntryId} entries ports \
      (<descriptor id>/c<j>/o0, {s.window / s.every} leaves each) of {s.executionPhasePrograms} members, one a slot, whose \
      template is a pinned {slotEntryId} or {selectEntryId} (its closure digest one of entryPins), so a due, \
      occupied slot's read bit is the prover's word: reuse chunk j's port of the pinned template as the scan's \
      prog<j>")

/-- The rows form's wiring findings, one an execution phase, in order (`wiredCheck`); none in the clock form. -/
def wiredChecks (p : Params) (c : Commitment) : Array (Option (Core.Verdict × String)) := match p.form with
  | .rows s .. => c.windows.zipIdx.map fun (reg, i) => wiredCheck p s c.audits (p.first + i) reg
  | .clock _ => #[]

/-- Execution phase `w`'s state finding in the rows form: a refusal unless its scan's state is the `SCAN_STATE` of an
audit its `prog` is wired to, read where the scan's transitions chain it (`stateTied`). With the `prog` wired, each
slot's entry is the one its Program computed, and a Program reads the requests from its own registered `SCAN_STATE`;
unless that is the scan's state, each of its rows the one the scan's unit before registered (`stateRead`), it computes
its entry on a state the prover chose. -/
def stateCheck (p : Params) (s : Schedule) (audits : Array Json) (w : Nat) (reg : Json) :
    Option (Core.Verdict × String) :=
  if stateTied p.audits s audits reg then none else
    some (.reject, if stateRead s reg then s!"window {w}: the scan's state is no wired audit's SCAN_STATE: its \
      {stateName} entries' root and domain are not every {stateValue} entry's of an audit whose slot entries are its \
      prog, so a slot's Program can compute its entry on a state the prover chose: register the scan's state tree as \
      that audit's SCAN_STATE"
    else s!"window {w}: the scan's state reads are not its transitions' chain: its registration's window.reads of \
      {stateName} are not o0 \{\"leaf\": 0, \"units\": [0, 1]}, sin \{\"leaf\": 0, \"units\": [1, {s.window + 1}]} \
      and o1 \{\"leaf\": 1, \"units\": [1, {s.window}]} (stateRead), so a unit could read a state no unit registered: \
      register the start unit's o0 at leaf 0, bucket t's sin at leaf t and its o1 at leaf t + 1")

/-- What the check says of an execution phase that is not the run's first. Its scan starts from the state the execution
phase before it ended in, and the check holds no start to that state. -/
def startCarriedRefusal : String :=
  "the scan's start is the state the link-window before it ended in, and the check holds a start only to the empty \
    state, the run's first link-window's (startCheck): a period is the run's first link-window alone"

/-- Execution phase `w`'s start finding in the rows form: a refusal unless it is the run's first, execution phase 0, and
its scan starts from the empty state (`startEmpty`). The start unit copies the registered value `startName` to row 0 of
the scan's state, which the Programs anchored at bucket 0 read; unless the check holds that value, they read a state
the prover chose. A later execution phase's start is the state the one before it ended in, which the check does not hold
it to, so it refuses every execution phase but the run's first (`startCarriedRefusal`). -/
def startCheck (w : Nat) (reg : Json) : Option (Core.Verdict × String) :=
  if w != 0 then some (.reject, s!"window {w}: {startCarriedRefusal}")
  else if startEmpty reg then none
  else some (.reject, s!"window {w}: the scan's start is not the empty state: its registration has no {startName} \
    entry among its registered values, or its window.empty of {startName} is no opening of one to zero u16 words \
    under its salt at that entry's root and domain (startEmpty), so the Programs anchored at bucket 0 can read a \
    state the prover chose: register the start unit's sin as the empty state's row and give its salt")

#guard
  let one := "cc8a8f235c85967255dadcfea4db8f2534aa0a12d0c3e9c289462c14072a92cc\
    3c65bfd17a3df01f6dcb91eec5ff7a6cb18294dc0ad07ab56a1e2fcd341161e0"
  (startCheck 0 (startProbe [one] 1)).isNone && (startCheck 0 (startProbe [one] 2)).isSome &&
    (startCheck 1 (startProbe [one] 1)).map (·.2) == some s!"window 1: {startCarriedRefusal}"

/-- The rows form's state findings, one an execution phase, in order (`stateCheck`); none in the clock form. -/
def stateChecks (p : Params) (c : Commitment) : Array (Option (Core.Verdict × String)) := match p.form with
  | .rows s .. => c.windows.zipIdx.map fun (reg, i) => stateCheck p s c.audits (p.first + i) reg
  | .clock _ => #[]

/-- The rows form's start findings, one an execution phase, in order (`startCheck`); none in the clock form. -/
def startChecks (p : Params) (c : Commitment) : Array (Option (Core.Verdict × String)) := match p.form with
  | .rows .. => c.windows.zipIdx.map fun (reg, i) => startCheck (p.first + i) reg
  | .clock _ => #[]

/-- `okCheck`'s refusal, after its execution phase's number. -/
def okRefusal : String :=
  "the scan's ok bits are not the schedule's constants: its registration's roots do not name each member's public \
    word, <member>/o0, at the root, leaves and domain of that member's proof blocks' ok rows under the public salts \
    (ScanOk.consts at the schedule's T: ingress-0, last and egress-0), or name another port ending /o0 than the start \
    unit's, so a \
    proof block's ok bit is the prover's word: commit each member's o0 rows, one word 1 each, under the public salts \
    (scan_ok.salt)"

/-- Execution phase `w`'s ok finding in the rows form: a refusal unless its scan's registration names each member's ok
rows at the schedule's constants `cs` (`ScanOk.held`), and where the constants do not compute (a schedule of fewer
than two ticks, `ScanOk.consts`). It reads `cs`, which the period computes once, and hashes nothing. -/
def okCheck (cs : V (Array ScanOk.Const)) (w : Nat) (reg : Json) : Option (Core.Verdict × String) :=
  match cs with
  | .ok ks => if ScanOk.held ks reg then none else some (.reject, s!"execution phase {w}: {okRefusal}")
  | .error e => some (.reject, s!"execution phase {w}: the scan's ok bits' constants do not compute: {e}")

/-- The ok findings of the registrations `regs` at the constants `cs`, one an execution phase from `first`, in
order (`okCheck`): comparisons alone, no hash an execution phase. -/
def okChecksAt (cs : V (Array ScanOk.Const)) (first : Nat) (regs : Array Json) :
    Array (Option (Core.Verdict × String)) :=
  regs.zipIdx.map fun (reg, i) => okCheck cs (first + i) reg

/-- The rows form's ok findings, one an execution phase, in order: the schedule's constants (`ScanOk.consts` at its
`T`, the one place the period hashes for them, once) and each registration held to them (`okChecksAt`); none in the
clock form. -/
def okChecks (p : Params) (c : Commitment) : Array (Option (Core.Verdict × String)) := match p.form with
  | .rows s .. => okChecksAt (ScanOk.consts s.window) p.first c.windows
  | .clock _ => #[]

/-- Whether a compute audit's registration names `circuit`, a class circuit's Program's SHA-512, as its Program: its
window circuit is that class's. None with no class circuit. -/
def runsCircuit (circuit : Option String) (reg : Json) : Bool := match circuit, reg.getObjValD "program" with
  | some d, .str x => x == d
  | _, _ => false

/-- The class circuit is held at an execution phase: each audit its scan's `prog` is an output of (`tiedAudits`) runs
it (`runsCircuit`), so the phase's tied audit runs the one window circuit the class registers. -/
def held (circuit : Option String) (audits : Array Json) (reg : Json) : Bool :=
  (tiedAudits audits reg).all fun i => (audits[i]?.map (runsCircuit circuit)).getD false

/-- `heldCheck`'s refusal, after its execution phase's number. -/
def heldRefusal : String :=
  "the tied audit's window circuit is not the registered class circuit: an audit the scan's prog is an output of \
    (tiedAudits) is registered under another Program than the parameters' circuit, so its execution phase's term \
    would be read off a Program the prover chose for that phase: register every execution phase's audit under the \
    class's window circuit"

/-- Execution phase `w`'s class circuit finding in the rows form: a refusal unless the audits its scan's `prog` is an
output of run the class circuit (`held`). A window circuit's own digest binds its window's run, so only its Program
can be held; without the hold an execution phase's audit could run a Program the prover chose for that phase. -/
def heldCheck (p : Params) (audits : Array Json) (w : Nat) (reg : Json) : Option (Core.Verdict × String) :=
  if held p.circuit audits reg then none else some (.reject, s!"execution phase {w}: {heldRefusal}")

/-- The rows form's class circuit findings, one an execution phase, in order (`heldCheck`); none in the clock form. -/
def heldChecks (p : Params) (c : Commitment) : Array (Option (Core.Verdict × String)) := match p.form with
  | .rows .. => c.windows.zipIdx.map fun (reg, i) => heldCheck p c.audits (p.first + i) reg
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
  /-- The slot entries' bits a tied audit's incorrect proof blocks can set (`β_w`, the rows form's; 0 in the clock form). -/
  faultyEntries : Nat := 0

def AuditTerms.bits (t : AuditTerms) : Nat := t.free + t.faultySets + t.faultyBits + t.jitter + t.faultyEntries

/-- A compute audit's terms in the clock form, from its layout terms, its facts and the jitter bound: each of its at
most `b` incorrect proof blocks commits at most its Program's widest proof block's bits. -/
def AuditTerms.of (a : Audit) (f : Facts) (jitter : Nat) : AuditTerms :=
  { facts := f, free := a.free, faultySets := clog2 (atMost f.units f.wrong), faultyBits := f.wrong * a.bits,
    jitter := clog2 (jitter ^ a.steps) }

/-- An execution phase's scan's terms in the rows form, from the scan's terms, its facts, its tied audits' free inputs
(`tied`: the egress renders their outputs, so the bits the prover chose of them are the execution phase's free inputs),
the slot entries' bits its tied audits' incorrect proof blocks can set (`faulty`, `faultyEntries`) and the jitter bound:
each of its at most `b` incorrect proof blocks commits at most the wider instance's bits. -/
def Scan.terms (s : Scan) (f : Facts) (tied faulty jitter : Nat) : AuditTerms :=
  { facts := f, free := s.free + tied, faultySets := clog2 (atMost f.units f.wrong),
    faultyBits := f.wrong * max s.ingress s.egress, jitter := clog2 (jitter ^ s.steps), faultyEntries := faulty }

structure Public where
  /-- The compute audits' facts. -/
  audits : Array Facts
  cells : Nat
  /-- The terms of `MonitoringPeriodLeakAccepted`'s windows: the clock form's compute audits', the rows form's scans', one a
  execution phase. -/
  units : Array AuditTerms
  /-- The rows form, whose proof blocks are its scans. -/
  rows : Bool := false
  schedules : Nat
  windows : Nat
  /-- The facts of the rows form's tied audits sampled in part (`chargedOf`), whose bounds `faulty_entries` reads, so
  whose δ is in `eps`. -/
  charged : Array Facts := #[]
  /-- The schedule's response cap (`Form.maxTokensCap`), published when it has one. -/
  maxTokensCap : Option Nat := none

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
length, one of `cap_c + 1`, fixes its admission and the lanes it frees), and each execution phase's scan's from the
scan's terms, its facts, its tied audits' free inputs and the entry bits their incorrect proof blocks can set (`scans`,
a triple a execution phase). The compute audits' own terms are in none: a tied audit's free inputs and incorrect proof
blocks are in its execution phase's, and the δ of one sampled in part (`charged`) in `eps`. -/
def rowsTerms (facts : Array Facts) (scan : Scan) (scans : Array (Facts × Nat × Nat)) (cells : Array Nat)
    (jitter windows : Nat) (charged : Array Facts := #[]) : Public :=
  { audits := facts, cells := capsTerm cells, units := scans.map fun (f, t, b) => scan.terms f t b jitter,
    rows := true, schedules := capsTerm cells, windows, charged }

/-- Audit `i`'s facts when the rows form charges it: accepted, tied to an execution phase's scan (`tiedOf`) and sampled
in part, so its at most `b` incorrect proof blocks are in its execution phases' terms (`faultyEntries`) and its δ in
`eps`. -/
def chargedOf (p : Params) (c : Commitment) (as : Array (Found Facts)) (i : Nat) : Option Facts :=
  match as[i]? with
  | some (.ok f) => if tiedOf p.form c i && f.drawn < f.units then some f else none
  | _ => none

/-- An execution phase's `β_w`: every slot entry bit of its scan (`Scan.entries`, `E_w`) when an audit its `prog` is
tied to (`tiedAudits`) is charged (`chargedOf`) with a bound of at least one incorrect proof block, which can set any of
the entries, else 0. -/
def faultyEntries (p : Params) (c : Commitment) (as : Array (Found Facts)) (scan : Scan) (reg : Json) : Nat :=
  if (tiedAudits c.audits reg).any fun i => (chargedOf p c as i).any (1 ≤ ·.wrong) then scan.entries
  else 0

/-- The rows form's frames finding (`rowsWf`): `paramsOf` refuses such a period before its cells, and the check refuses
it too, so whatever parameters it accepts, its rows form's frames read back. None in the clock form. -/
def rowsWfCheck (p : Params) : Option (Core.Verdict × String) := match p.form with
  | .rows s _ _ => if rowsWf s p.windows then none else some (.reject, rowsWfRefusal s p.windows)
  | .clock _ => none

/-- The rows form's connection rows finding (`Schedule.connectionRows`): `paramsOf` refuses a schedule with
connection rows by name (`connectionRowsRefusal`), and the check refuses it too, so whatever parameters it accepts,
its schedule has no lengths rows and no content rows. None in the clock form. -/
def connectionRowsCheck (p : Params) : Option (Core.Verdict × String) := match p.form with
  | .rows s _ _ => if s.connectionRows then some (.reject, connectionRowsRefusal) else none
  | .clock _ => none

/-- The rows form's public on acceptance: `rowsTerms` over the accepted audits' facts, each execution phase's scan's
facts, tied audits' free inputs and `β_w` (`faultyEntries`), and the charged audits (`chargedOf`). -/
def rowsPublic (p : Params) (c : Commitment) (as : Array (Found Facts)) (scan : Scan) (scans : Array (Facts × Nat)) :
    Public :=
  rowsTerms (as.filterMap (·.toOption)) scan
    ((scans.zip c.windows).map fun ((f, t), reg) => (f, t, faultyEntries p c as scan reg)) p.cells p.jitter p.windows
    ((Array.range as.size).filterMap (chargedOf p c as))

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
          executionPhaseCheck p clear ((e.windows.extract 0 i).filterMap (·.clockDigest?)) (p.first + i) reg x
      | .rows s sd _ => pairs.map fun ((reg, x), i) => some <$> rowsExecutionPhaseCheck p s sd c.audits (p.first + i) reg x
    let scans := ws.filterMap (·.toOption.join)
    let oneDelta : Option (Core.Verdict × String) := scans[0]?.bind fun (f0, _) =>
      (scans.find? (·.1.delta != f0.delta)).map fun (f, _) =>
        (.reject, s!"a scan's δ {f.delta.toString} is not the period's, {f0.delta.toString}")
    let as := auditChecks p c e
    match firstFinding (ws.toList.map findingOf ++ [oneDelta, rowsWfCheck p, connectionRowsCheck p] ++
        as.toList.map findingOf ++ (wiredChecks p c).toList ++ (stateChecks p c).toList ++
        (startChecks p c).toList ++ (okChecks p c).toList ++ (heldChecks p c).toList) with
    | some (.reject, r) => .reject r
    | some (_, r) => .undecided r
    | none =>
      let audits := as.filterMap (·.toOption)
      .accept { (match p.form with
          | .clock ss => terms p.audits audits p.cells p.jitter (ss.deltas.length * ss.rhos.length) p.windows
          | .rows _ _ scan => rowsPublic p c as scan scans) with
          maxTokensCap := p.form.maxTokensCap }
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
    (if x.rows then [("scans", termsJson x.units true)] else []) ++
    (x.maxTokensCap.map fun cap => ("max_tokens_cap", toJson cap)).toList)

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
-- an audit of 10 proof blocks, each committing 16 bits, sampled subset:3 at δ = 1/10 (at most 4 wrong), 7 cells on one
-- connection and none on another, 2 syncs over 3 execution phases, and jitter below 2 over 3 steps
#guard (terms #[⟨0, 3, "", 10, 16, #[]⟩] #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] #[7, 0] 2 2 3
  ).K == 3 + 3 + (0 + 9 + 64 + 3)
-- the same audit with a unit committing 40 bits: its 4 incorrect proof blocks carry 160 bits, not 16 · 4
#guard (terms #[⟨0, 3, "", 10, 40, #[]⟩] #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] #[7, 0] 2 2 3
  ).K == 3 + 3 + (0 + 9 + 160 + 3)
-- two audits at δ = 1/100 and 1/10^6: ε = 10001/1000000
#guard (terms #[⟨0, 1, "", 4, 16, #[]⟩, ⟨0, 1, "", 4, 16, #[]⟩]
  #[{ units := 4, drawn := 1, delta := ⟨1, 100⟩, wrong := 0 }, { units := 4, drawn := 1, delta := ⟨1, 1000000⟩, wrong := 0 }]
  #[] 1 1 1).eps.toString == "10001/1000000"
-- eps over audits at δ = 1/10, 1/15 and 1/6 is 1/3; over none it is 0
#guard (terms #[⟨0, 0, "", 1, 16, #[]⟩, ⟨0, 0, "", 1, 16, #[]⟩, ⟨0, 0, "", 1, 16, #[]⟩]
  #[⟨1, 1, ⟨1, 10⟩, 0⟩, ⟨1, 1, ⟨1, 15⟩, 0⟩, ⟨1, 1, ⟨1, 6⟩, 0⟩] #[] 1 1 1).eps.toString == "1/3"
#guard (terms #[] #[] #[] 1 1 1).eps.toString == "0"
-- the rows form over 1 execution phase: 7 cells on one connection and 2 on another (3 + 2 bits, for the schedules and
-- again for the fail cells), an audit whose own terms are in none, and one scan of 21 units (T = 10) sampled at
-- δ = 1/15 with at most 3 wrong, 1,120 free inputs, ingress and egress instances committing 300 and 512 bits, and
-- jitter below 2 over its 10 steps: 1562 sets of at most 3 of 21 units
#guard atMost 21 3 == 1562 && clog2 1562 == 11
#guard (rowsTerms #[{ units := 10, drawn := 3, delta := ⟨1, 10⟩, wrong := 4 }] ⟨10, 1120, 300, 512, 0⟩
  #[({ units := 21, drawn := 5, delta := ⟨1, 15⟩, wrong := 3 }, 0, 0)] #[7, 2] 2 1).K ==
  (3 + 2) + (3 + 2) + (1120 + 11 + 3 * 512 + 10)
-- every unit sampled: no incorrect proof block, so no incorrect set and no faulty bits
#guard (rowsTerms #[] ⟨4, 0, 300, 512, 0⟩ #[({ units := 9, drawn := 9, delta := ⟨1, 100⟩, wrong := 0 }, 0, 0)] #[] 1 1
  ).K == 0
-- w63's class with the egress tie: T = 32, the scan's own free inputs none, every unit sampled, and its tied audit (the
-- slot entries' `SlotEntry_v1` template) of 8 units, a slot at each of the 2 boundaries, whose free inputs are their
-- entries, 1,136 bits each: K is the tied audit's 9,088
#guard (rowsTerms #[{ units := 8, drawn := 8, delta := ⟨1, 100⟩, wrong := 0 }] ⟨32, 0, 300, 512, 0⟩
  #[({ units := 65, drawn := 65, delta := ⟨1, 100⟩, wrong := 0 }, 8 * 1136, 0)] #[] 1 1).K == 9088
-- eps over the scans' δ = 1/15 and 1/15 is 2/15; the audit's 1/10 is in it no more
#guard (rowsTerms #[⟨1, 1, ⟨1, 10⟩, 0⟩] ⟨0, 0, 0, 0, 0⟩ #[(⟨1, 1, ⟨1, 15⟩, 0⟩, 0, 0), (⟨1, 1, ⟨1, 15⟩, 0⟩, 0, 0)] #[]
  1 2).eps.toString == "2/15"
-- w63k4: T = 32, 4 lanes, every 16, so 8 slot entries of 112 bits an execution phase; its tied audit sampled in part,
-- with at most 2 incorrect proof blocks, which can set any entry: its execution phase's term charges E_w = 896 bits
-- beside its free inputs
#guard (rowsTerms #[{ units := 16, drawn := 9, delta := ⟨1, 100⟩, wrong := 2 }] ⟨32, 0, 300, 512, 8 * 112⟩
  #[({ units := 65, drawn := 65, delta := ⟨1, 100⟩, wrong := 0 }, 9088, 8 * 112)] #[] 1 1).K == 9088 + 896
-- a charged audit's δ = 1/10 is in eps beside its scan's 1/15: 1/6
#guard (rowsTerms #[⟨2, 1, ⟨1, 10⟩, 0⟩] ⟨0, 0, 0, 0, 0⟩ #[(⟨1, 1, ⟨1, 15⟩, 0⟩, 0, 0)] #[] 1 1 #[⟨2, 1, ⟨1, 10⟩, 0⟩]
  ).eps.toString == "1/6"
-- w63k4's tied audit of 16 proof blocks sampled at one (`subset:1`), with a bound of 2 incorrect ones: charged, its
-- execution phase's β_w every one of the scan's E_w = 896 entry bits; sampled whole, uncharged and β_w = 0
#guard
  let scan : Scan := ⟨32, 0, 300, 512, 8 * 112⟩
  let p : Params := ⟨#[⟨0, 0, "", 16, 0, #[]⟩], #[], 0, 1, "", Lms.deployed, .empty, .empty, "",
    .rows ⟨0, 0, 0, 0, 16, 0, 0, 4, 0, 0, 32, none, none, none, false⟩ .empty scan, 1, none⟩
  let at_ := fun (q : String) => Json.mkObj [("roots", Json.arr #[
    Json.mkObj [("port", .str q), ("root", .str "r"), ("domain", .str "d")]])]
  let c : Commitment := ⟨#[at_ (entriesPort "SelectEntry_v1{KO=4}" 0)], #[at_ (progName 0)], .empty⟩
  let one : Array (Found Facts) := #[.ok ⟨16, 1, ⟨1, 100⟩, 2⟩]
  let whole : Array (Found Facts) := #[.ok ⟨16, 16, ⟨1, 100⟩, 0⟩]
  tiedIn c.audits c.windows 0 && (chargedOf p c one 0).isSome && faultyEntries p c one scan (at_ (progName 0)) == 896 &&
    (chargedOf p c whole 0).isNone && faultyEntries p c whole scan (at_ (progName 0)) == 0
-- `factsOf` reads a profile of one level sampled within it, `subset:k` or stratified, whole or in part, and refuses
-- one that names more sampled than the level has, 3 of 2 or strata of 2 of 1 and 1 of 1, then one that names more
-- sampled of a stratum than the stratum has, strata of 2 of 1 and 0 of 1 in a level of 2, then one whose strata's
-- sizes do not sum to the level's, strata of 1 of 2 and 1 of 2 or of 1 of 1 alone in a level of 2, each before the
-- bound is read
#guard
  let facts := fun (size : Nat) (d : Core.SamplingDesign) =>
    match (factsOf "x" ⟨#[], #[], false, #[], #[⟨"unit", none, size, some d⟩], ⟨1, 100⟩, #[]⟩
      (Json.mkObj [("bound", toJson (1 : Nat))])).toOption with
    | some ⟨units, k, _, b⟩ => some (units, k, b)
    | none => none
  facts 2 (.subset 2) == some (2, 2, 1) && facts 2 (.subset 1) == some (2, 1, 1) &&
    facts 3 (.stratified #[⟨"a", 2, 2⟩, ⟨"b", 1, 0⟩]) == some (3, 2, 1) &&
    facts 2 (.subset 3) == none && facts 2 (.stratified #[⟨"a", 1, 2⟩, ⟨"b", 1, 1⟩]) == none &&
    facts 2 (.stratified #[⟨"a", 1, 2⟩, ⟨"b", 1, 0⟩]) == none &&
    facts 2 (.stratified #[⟨"a", 2, 1⟩, ⟨"b", 2, 1⟩]) == none && facts 2 (.stratified #[⟨"a", 1, 1⟩]) == none
#guard
  let reason := fun (d : Core.SamplingDesign) =>
    match factsOf "x" ⟨#[], #[], false, #[], #[⟨"unit", none, 2, some d⟩], ⟨1, 100⟩, #[]⟩ (Json.mkObj []) with
    | .error (_, r) => r
    | .ok _ => ""
  reason (.subset 3) == "x: profile: 3 sampled of 2 units: more than the level holds" &&
    reason (.stratified #[⟨"a", 1, 2⟩, ⟨"b", 1, 1⟩]) == "x: profile: 3 sampled of 2 units: more than the level holds" &&
    reason (.stratified #[⟨"a", 1, 2⟩, ⟨"b", 1, 0⟩]) ==
      "x: profile: stratum 'a': 2 sampled of 1 units: more than the stratum holds" &&
    reason (.stratified #[⟨"a", 2, 1⟩, ⟨"b", 2, 1⟩]) ==
      "x: profile: strata of 4 units in a level of 2: their sizes do not sum to the level's"
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
  let s : Schedule := ⟨0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 4, none, none, none, false⟩
  let a : Audit := ⟨0, 0, "", 4, 0, #[("SlotEntry_v1{KO=1}", 4, slotEntryPins[0]!)]⟩
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
-- a lane's Programs' rows, the lane's index in ASCII digits without a leading zero
#guard progPort (progName 0) && progPort "scan/prog3" && progPort "scan/prog12" && progName 3 == "scan/prog3"
#guard !(progPort "scan/prog" || progPort "scan/prog01" || progPort "scan/prog/" || progPort "Scan/prog0" ||
  progPort "scan/prog0x" || progPort "prog0" || progPort "scan/state" || progPort "")
-- the ok findings, one an execution phase, numbered from the period's first: none where its registration is at the
-- schedule's constants, a refusal at another schedule's; every execution phase refused where the constants do not
-- compute; and none of no execution phase
#guard
  let entry := fun (c : ScanOk.Const) => Json.mkObj [
    ("port", .str c.port), ("leaves", toJson c.leaves), ("domain", .str c.domain), ("root", .str c.root)]
  let regOf := fun (T : Nat) =>
    (ScanOk.consts T).toOption.map fun cs => Json.mkObj [("roots", Json.arr (cs.map entry))]
  match regOf 3, regOf 4 with
  | some r3, some r4 =>
    (okChecksAt (ScanOk.consts 3) 5 #[r3, r4, r3]).map (·.map (·.2.startsWith "execution phase 6: ")) ==
        #[none, some true, none] &&
      (okChecksAt (.error "x") 0 #[r3, r3]).all (·.isSome) && (okChecksAt (ScanOk.consts 3) 0 #[]).isEmpty
  | _, _ => false
-- a schedule of 4 lanes over a link's 32-bucket execution phase carries at most 128 connections, each held to its 5
-- egress cells, so the schedules and the cells each charge 128 · ⌈log2 6⌉ = 384 bits a link's execution phase
#guard
  let s : Schedule := ⟨100, 512, 27, 5, 16, 12, 47, 4, 4, 4, 32, none, none, none, false⟩
  s.rowsCells 1 == Array.replicate 128 5 && capsTerm (s.rowsCells 1) == 384 && (s.rowsCells 2).size == 256
-- its frames, B = 512 and 128 connections a link's execution phase, read back over 2^25 − 1 such phases and not over
-- 2^25, whose last connection's ordinal is 2^32; the check refuses that period
#guard
  let s : Schedule := ⟨100, 512, 27, 65, 16, 12, 47, 4, 4, 4, 32, none, none, none, false⟩
  let p := fun (w : Nat) =>
    (⟨#[], #[], 0, 1, "", Lms.deployed, .empty, .empty, "", .rows s .empty ⟨0, 0, 0, 0, 0⟩, w, none⟩ : Params)
  rowsWf s (2 ^ 25 - 1) && !rowsWf s (2 ^ 25) && (rowsWfCheck (p (2 ^ 25 - 1))).isNone &&
    (rowsWfCheck (p (2 ^ 25))).isSome
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

end NetworkCheck.Period
