# Network certifier (`verity.core.protocols.certifier`)

A spec-level reference for network accounting's timing channel:
- the constant-rate, left-packed certifier;
- the Program's logical schedule, with its clock-sync advice (δ, ρ);
- the exact audit with record-and-reject, on raw records, records committed by hash (padding elided), or network
  certificates (`certificate`: the network certifier's signature over the record's SHA-512 digest hidden under a
  64-byte salt the certifier draws fresh for each record, with the window's frames in a fixed-depth SHA-512 tree), and
  the ingress links' status check;
- one link-window's certificates through the proof system (`service`): the network `Spec`, a salted root over the
  certified records, and the check, whose Boolean part is each head's check (`head_check`: the head's hiding opening,
  its fields, its counts against the public occupancy profile, and its record digest under the certifier's salt
  against the digest its certificate signs), with a host-side verifier of record (no proof of the
  check yet) that compares no grid in the clear. Its claim is counts only: on every link-window the head check holds
  the egress head's counts to the profile (and the ingress head's, where that end declares one), so the link's timing
  is at the profile. Nothing here holds the frames, the rest of the egress grid (N3, the content check prototyped in
  `benchmarks/network_sim`, is retired, superseded by the step-indexed scan, `ingress_scan` and `egress`, outside this
  protocol). The network verdict of record is
  `LinkWindows.network_verdict`, the AND over the link's link-windows with no ε or share credit: every window of the
  egress profile registered in order with an accepted outcome, and a rejected, late, missing or out-of-order window
  rejects the link;
- a served link's counts (`served`, `LinkWindows(served=ServedProfile(F, F_in, windows, own))`), whose egress end has no
  occupancy profile: N2's served unit, `NetworkServedLinkWindow`, adds a Boolean Definition,
  `NetworkServedCounts_v1` (678,795 ANDs at T = 600, F = 1024, F_in = 1, R = 6, R_in = 1), that holds the egress
  head's counts to the ones its ingress head's counts derive under the window's declared sync, the advice the audit
  already charges at log2 #Σ_sync bits a window. Each window starts empty and must end idle, so nothing carries
  between windows, and a request whose frames or response straddle a window's end refuses the window. The claim is
  counts only and rests on no premise. That the derived counts are the Program's grid rests on four: FIXLEN (every
  response F frames), a fixed request size (every request F_in frames, every ingress frame a request's but the
  certifier's salt frame heading each window, which `own` counts), a
  request-major canonical order, and one T, τ_b and declared sync for a window's anchors (`PROTOCOL.md`, "A served
  link"). The proxy's admission rule (`Proxy(served=…)`) admits a request only while its window still ends idle, so
  an honest served link is accepted. An egress end with no profile gets no verdict (`ValueError`, failing
  closed);
- one vLLM v1 scheduler step (`replay_step`): `ReplayStep_v1`, a Boolean Definition with `reference` its spec, takes the
  committed queue state, the certified arrivals and the committed decision bits to the step's rows (request, kind,
  start, n) and the next state, under sync or async scheduling and with `CacheStep_v1`'s walk entries, and refuses
  what its size class or the prefix layer's counts can't express. It is the compile unit's core, between the committed
  decisions and `compile_steps`' Programs, and is held step for step to the replay
  (`benchmarks/network_e2e/encodings/replay_vectors.py`); it is in no proof-system Spec yet. It takes the circuits
  and their evaluator from `verity`;
- the I/O tie (`io_tie`), a design (`PROTOCOL.md`, "The I/O tie") and its prototype on a fake upstream, in no Spec:
  served through the warden's egress, every cell of an egress link-window is opened, each a pad, the canonical event of
  one token id or the response's end on the stream the warden set, and each is the cell the advice's map names at its
  place (`window_map`, from the advice's outputs, release buckets and advice v1's order permutation, each request on its
  ingress ordinal's stream from 1, accepted on TRACE-CORPUS's heavy trace; the verifier computes it in the clear from
  the opened advice, the named gap, until the map is read in the proof under the advice's hiding root), with the id the
  Commit sampled for it (the port: one hiding row of ids a step, read under the Commit's root). The cells are the
  warden's (`warden`): a connection's r slots a bucket, a `PAD` on its stream before its first cell and after its `END`,
  one id event a `DATA` cell. The warden commits each bucket's cells as one hiding row. It has the reference and the
  unit's Boolean Definition, one a bucket row (`IoTieBucket`: the row and the port's rows read as registered rows,
  the warden's and the Commit's (`port_reads`), then `IoTieContent`: `IoTieCell` on every cell, the prover's Beneš
  routing of the port's entries `IoTieRoute`, `IoTieMatch`);
- the ingress tie (`io_request`, `PROTOCOL.md`, "The ingress tie") on the warden's own ingress (`serve`): one unit a
  registered bucket row (`warden_rows.bucket_port`), each live connection's lane of r slots read as the warden's cells
  (a `PAD` on its ordinal, a `DATA` of exactly B − 9 bytes, its `END`), each request cell a cell of CERT-VARLEN's id
  request, with its count of prompt fields carried across buckets from the lane's carry-in to its carry-out, which
  hold each request's prompt_len to the advice's and which N2-ADVICE's advised verdict (`advised.IoTieVerdict`) is
  built from. It has the reference, the lanes the verifier computes from the advice and the warden's release
  (`lanes`, `Release`), and the unit's Boolean Definition (`IoTieRequest`: one `IoTieRequestFrame` and
  `IoTieRequestStep` a cell, `IoTieRequestEnd` a lane); the ingress-ids tie (`io_ids`, `IoTieRequestIds`) adds the
  Commit's `c00.input/ids` rows to the same scan and holds every prompt id to them;
- the capacity counts;
- the egress side of the step-indexed scan (`egress`, N2) on a fake upstream, in no Spec: one unit a link and bucket
  t (`EgressBucket`) that reads the scan's slot state S_t, a registered read, and computes the bucket's row of cells
  from it with every switch computed and none witness (the generator form): its outputs are a u16, `ACCEPT` exactly
  when the state renders a row, the one public output, and the row, committed as a registered read of the warden's
  egress row t (`warden_rows.bucket_port`), which opens only if the computed row is the warden's. It has the reference
  (`bucket_map`, `computed`, `rendered`, `outputs`) and the state's egress-view layout (`Shape`'s offsets), which the
  scan's ingress step writes;
- the calibration of honest runs;
- advice v0 (`advice`): what vLLM's scheduler decided in a session (each request's lengths and finish, each step's
  rows with their chunks, prefix-cache hits and the KV they reuse, resumes, and each kept output's egress bucket) as one
  canonical bit string the compiler reads, refused unless it is a schedule vLLM could have run, with its bits split by
  the earliest egress window that can show each record (`bits`). v0 is the committed form; the charge is ENC-CODE's
  compressed length of the advice;
- advice v0's validity (`advice_check`): `AdviceCheck_v1{L}`, the compile unit's first part as a Boolean Definition, 1
  exactly when the committed bits (at most L) decode to a v0 advice, up to the three refusals of `decode` it defers
  (unique ids, a source's KV computed there, release order per request); its AND count at each L is `size(L)`. From
  `verity` it takes the circuits and their evaluator;
- the replay's state chain (`replay_chain`): ReplayStep_v1's state handed from one step unit to the next as the
  state rows of a produced value a window, with private heads into the window row every step reads: the certified
  arrivals not yet taken and the window's decisions, since the step's cursors are private. `ReplayAdvance` sits beside
  each step and checks its arrivals and decision bits are the window row's at the heads; `ReplayLoad` runs once a
  window and carries the decisions the last window left unread; `ReplayClose` ends the session with none unread. The
  windows join by equal openings and the session starts from the empty rows under public salts (`check_chain`). Its
  cost at a shape is `size`;
- the compile slice (`compile_slice`): `CompileSlice_v1`, a Boolean Definition with `check` its reference, composes
  `ReplayLoad`, `ReplayStep_v1` and `ReplayAdvance` on shared wires, `CacheStep_v1`'s calls on the certified tokens
  checking the hits `ReplayStep_v1` reads, and `ChargeLink_v1` over the decisions the window commits, from its
  carry-in to its carry-out, for one window. It holds the charge to the declared one and the step rows to its committed
  ``srv`` row, the served rows. The evictable and returned counts are a stand-in held at 0
  (`protocol/compile-slice.md`), and
  `benchmarks/network_traces/compile_slice_zk.py` proves it in zero knowledge. It takes the circuits and their
  evaluator from `verity`;
- the compile slice with the block pool (`compile_slice_v2`): `CompileSlice_v2`, `CompileSlice_v1` with its prefix cache
  replaced by `CacheStep_v2`'s pool, its calls in vLLM's order on the pool rows in ``wit``. Each free's ``held`` and
  each compute's ``new`` are `ReplayStep_v1`'s own, and the hits, evictable counts and returns `ReplayStep_v1` reads
  are the pool's, so eviction, preemption and a leaving request's returns are in the slice
  (`benchmarks/network_e2e/encodings/compile_slice_pool.py` holds its calls to the replay's `_KV` over whole traces);
- the compile slice computing its schedule (`compile_slice_v3`): `CompileSlice_v3`, `CompileSlice_v2` computing what v2
  checks. From the advice (DA decisions and the step count), the arrivals' ingress fields and their registered output
  lengths and tokens, it computes the served rows and the state, window and pool rows the window leaves, each hit,
  evictable count, return and stamp where `ReplayStep_v1` reads it, and holds them to the registered ``srv`` and
  ``cout`` it reads; its one output is the 16-bit verdict. So its only free input is the advice, ℓ_w = DA + PW bits
  (`free_width`), and the charge is ℓ_w;
- the served rows (`served_rows`): the step rows a served window ran, from the server's kept rows, committed before the
  draw under the run root as an hm96 row registration, a row each block of G steps (G = P unless given: a window's row,
  the slice's ``srv`` row; at G < P block j is the slice's ``srv{j}``, `compile_slice.CompileSliceBlocks_v1`). `leaf_ok`
  holds a unit's row b‖c to its registered leaf, as the advice's ``adv`` row is held,
  and `reads_port` is the same read as the Lean verifier's registered reads take it. It takes the advice registration's
  hm96 rows from `verity`;
- the served rows' tie (`served_tie`): `ServedRowsTie_v2`, a Boolean Definition with `check` its reference, holds a
  window's served rows to the positions its steps' Commit holds (the step-input port, `port_chunks`: each step's ids and
  then its positions at the head of its stream), so the rows registered at serve time are the batch the server ran. Every
  unit is proved, and its port, selection and cost are its docstring. It takes the circuits, their evaluator and the
  hm96 leaves from `verity`;
- the served rows held to their Programs (`served_statics`): `ServedStaticsTie_v1`, a Boolean Definition with `check`
  its reference, holds each block's registered served rows, read through the Lean verifier's registered reads, to the
  RoPE statics of the Programs its steps ran, with no row hashed but the block's
  ([protocol/served-statics.md](protocol/served-statics.md)). It takes the circuits and their evaluator from `verity`;
- the online prover (`online`): each window's clock sync declared at its first bucket from what it had seen `lag`
  buckets before;
- the prefix cache (`cache_step`): `CacheStep_v1`, a Boolean Definition of one scheduled row's pass over a table of
  whole cached units (an engine step's calls are `scan_step` and, as a circuit, `build_scan_step`), which checks the
  row's hit (the longest cached prefix while no block can have been evicted, a
  cached prefix past that) and owns or appends the units the row computes (`PROTOCOL.md`, "Advice v1"). It takes the
  circuits and their evaluator from `verity`;
- the block pool (`cache_step_v2`): `CacheStep_v2`, one call's pass over vLLM's prefix-cache block pool: which slots
  hold each cached block, the free queue's order by stamps, eviction named by a threshold stamp the call checks, and a
  leaving request's returns. So a lookup's hit and evictable count and a free's returns are derived past the
  cumulative pool bound too, in N = NB − 1 entries. `compile_slice_v2` wires it into the compile slice;
- advice v1's arithmetic encoder (`arith_encode`), Boolean Definitions each with its reference as spec. The counting
  variant, which the composition takes: `ArithCountStep_v1{K, BIN, CW}` runs ENC-CODE's 32-bit coder forward on one
  symbol and counts the bits it puts in the window's segment, `ArithCountFinish_v1{CW}` adds the termination and
  holds the window to its declared charge, and `BookKT_v1{C}` is `code_v1`'s KT estimate for a bit's context. `ChargeCount_v1{N}` composes them into one window's charge from its decisions, the compile
  unit's one call a window. The context and the frequencies are the circuit's, from the replay's position and the
  registered book's counts, never witness. The bit-checking variant, the alternative: `ArithEncodeStep_v1{K, BIN}` and
  `ArithEncodeFinish_v1` check each output bit against a committed segment in the decoder's order, with
  `KTContext_v1{C, NW}` ENC-REPLAY's scaled estimate. It is held symbol for symbol to `code.py` and `code_v1` on advice
  v1's windows and on ENC-DERIVED's still-sent fields (`benchmarks/network_e2e/encodings/arith_vectors.py`), and it is
  in no proof-system Spec yet (`PROTOCOL.md`, "Advice v1"). It takes the circuits and their evaluator from `verity`;
- the charge's carry chain (`charge_chain`): `ChargeCount_v1`'s carry committed from window to window as hm96 rows,
  window 0's the book's priors under a public salt, with the verifier's check of how the windows join (`check_chain`)
  and `ChargeLink_v1{N}`, the charge from the carry-in row held to the carry-out row (`protocol/charge-chain.md`). It
  takes the circuits, their evaluator and hm96 from `verity` and the produced rows from sampled proofs;
- the whole window's charge (`window_charge`): `ChargeCount_v2{N, H, O}` and `ChargeLink_v2{N, H, O}` charge advice
  v1's window (`Window`, the record `code_v1` codes) with its hits and order entries and their counts, coded as
  `code_v1` codes them, with the order's position and rank added to the carry and chained under the value
  `window-charge` (`protocol/charge-chain.md`, "The whole window"). It takes what `charge_chain` takes;
- the active warden (`warden`): a relay that knows bytes only, between any client and any server. It forwards each
  direction's bytes unchanged, in B-byte cells, r a bucket each way per connection, and egress cell i goes out in
  bucket s + D + ⌊i/r⌋ from its admission s. A miss fails the connection closed with a `FAIL` cell that carries no
  reason. It records and certifies exactly the cells it sends, and each failure's reason in a third record, `fails`.
  `PROTOCOL.md`'s "The active warden" has its leak bound, and `benchmarks/network_traces/warden_serve.py` runs it;
- the release order on the server's side (`release`): a certifier puts `x-verity-release: N` (N is the request's
  0-based place in its run's release order) and `x-verity-release-run: ID` on the server's copy of each request it
  releases, for `serve-openai --admit release-order`. The headers never appear in a cell or record, so nothing
  certified changes;
- the warden's cells as rows (`warden_rows`): each link-window's cells committed one `hm96-sha512/row/v2` row a
  bucket, padding and empty slots included, under a frame-v3-sha512 root the warden signs with the certifier's signer,
  beside a salted digest of the window's wiring. The registration is what a proof block's `--registered` port reads, so
  the sender's unit opens a cell's row as its output and the receiver's as its input, and the verifier places each
  read, the (connection, cell) the Program's units' wiring pins (`Units`, whose digest the port's value names), with
  the warden's wiring (`PROTOCOL.md`, "The warden's cells as rows"; `benchmarks/network_traces/warden_zk.py`);
- the output value the certifier relayed as a registered value (`out_commit`): `commit_out` commits its rows at the
  reading program's registered shape (`OutShape`: its program, the value, ``words`` a row and ``rows``, both sides
  taking them from the circuit's registered read) under the domain the verifier derives for that pinned value, with the
  certifier's fresh salts, and signs the head for the egress link-window; `check_out` reads the public record for the
  link-window it expects at the verifier's own shape and returns the root to pin, or refuses by name (`REFUSALS`:
  another width is ``out-width``, another row count ``out-rows``), so the output units' rows are registered reads of
  the certifier's root rather than a comparison in the clear;
- the run's boundary in a window's circuit (`boundary`): the circuit record's ``boundary``, which names the template
  whose unit ℓ·T + t holds bucket t of egress link ℓ (`EgressBucket`'s frame, the row it computes and returns,
  committed as a registered read of the warden's row, and its u16 verdict, whose bit 0 is the slots' presence) and of
  ingress link ℓ (its frame parameter row, the warden's registered row; with ``first``, `IngressScanFirst` holds bucket
  0), the same gates whatever the load, so arrival steps and prompt lengths stay private (`NetTiming.Layout`); its
  format is `verity/boundary/v3`. `refusal`, the verifier's check before it draws, checks that the layout fits the
  circuit (`NetTiming.Layout.Fits`: no egress gate is an input, and each is one the circuit returns) at the record's
  grain, which contains the gate grain's, held to `prog_fits`, `Layout.Fits` itself on a flattened program; the scan's
  state and view chains are wires of the record. A frame that is a parameter row, an input, is refused by name. `Fits`
  stays structural: a served window fits in the step-indexed scan's form, and a window in any other form is refused;
- the transfer rows (`transfer_rows`): a host's load, each value of the weights it loads once a session (the one kind,
  ``load``, ingress), committed as a registered value under the network certifier's own salts at the reading port's
  leaf width and row format, its root in a head the certifier signs for the session's first execution phase
  (`TransferHead`, `registration`); every other row into the host is the scan's state, with no head of its own.
  `check_registration` is the verifier's read of it for the transfer it expects, and `port` the verifier's own
  `--registered` port of the value at the signed root (the load relayed and committed as it crosses,
  `benchmarks/network_traces/load_relay.py`);
- the class circuit's check (`blocks`): a circuit record is one host's class window circuit, and which host runs which
  units, and which request occupies which slot and step, are private, placed by the scan inside the zero-knowledge
  proof; the class circuit and its padded counts (proof blocks per host, bits per link) are public. `check` refuses a
  record whose ``blocks`` state a server, server time or transfers (`PRIVATE`), an ``ingress`` declaration, a
  registered read or table of no registered parameter, a load (every registered parameter but the scan's state) read
  in other than whole rows of one width, by no unit, at an external port or by a table over part of it, and an external
  input port whose rows aren't each read exactly once from the scan's state (`ingress_scan.SCAN_STATE`), the host's one
  ingress. `pins` gives the verifier's pinned roots of the loads from their certificates alone (`certified`: each
  load's registration on the host's uplink, ingress, in the session's first execution phase, kind ``load``), so the
  interface's ``input_count`` under them and the scan's state's root has no free input;
- the scan's ingress side (`ingress_scan`): `IngressScanBucket`, one unit a certifier bucket, reads the bucket's
  warden ingress row, the slot state the bucket before carries and the Programs' row (registered reads), and computes
  the next state (each block to the slot holding its connection, a new connection to the lowest free slot, its prompt
  ids appended, a due slot's output count and ids taken from the Programs' row at its anchor, a slot held through its
  last egress bucket, `admitted + D + ⌊n/r⌋`) and its egress view, the state as `egress.EgressBucket` reads it (bucket,
  then per slot live, q, admitted, n and the output ids), its outputs beside a public ok bit; `IngressScanFirst`, the
  first bucket's, starts from the empty state, constants of its circuit. Programs start at fixed slots a network
  timestep (ASK-122): a timestep is `EVERY` buckets, the warden's `every`, which divides its window, with
  `SLOTS` = lanes Program slots each, and only `EVERY` and the slot count are public. The token-id row a slot's
  Program reads is the scan's state row b − 1 (`verity/certifier/scan-state/v1`, `token_row`): its occupancy word
  `occ` (1 for a slot whose Program starts at b, 0 for an empty slot), its prompt's length `fields` and its prompt ids,
  which the scan reads off the warden's signed ingress cells, so no other tie holds them to the certified bytes;
  `slot_row` gives each one's bits in the state row, for a reader that takes them as i32s (vllm's `ScanSlots_v2`), and
  `window_state` the rows a window circuit of the link-window from bucket t0 reads, the scan's rows t0 − 1 on (the
  empty state's for row −1). A Program gives all its outputs to the Programs' row at its anchor (n, u32, and 64 ids
  of 17 bits at w63's class); an empty slot's entry is all zero (its outputs masked by `occ`, never on the egress). A
  request the class can't hold is refused at the scan's unit, never truncated: a prompt of more than `KI` ids (1,024
  at w63's class) at its cell, a Program's n over `KO` at its anchor. A window whose warden refused a connection,
  traffic past its slots, has no certified record, and the period check refuses it. `Class.serves` names the warden
  parameters it takes every run of. `benchmarks/network_traces/scan_zk.py` proves both sides of a group of
  link-windows in one registration: ingress unit t's state and view outputs are committed as registered reads of the
  scan's state and view rows t, which ingress unit t + 1 and egress unit t read, and egress unit t's row as a
  registered read of the warden's egress row t, so the chains are wires through the registered roots;
- the active certifier (`active`): an asyncio TCP proxy that runs the certifiers on a live link and publishes their records.
  With its salt on (`Proxy(salt=active.fresh_salt)`, off by default), each PoUW window (`pouw_window` certifier
  windows) gets one salt of `salt_bits` bits (512 by default), carried into the block only. The ingress record holds a
  salt frame with a salt-free placeholder, so every record is a function of the traffic alone; the salt sits in the
  certifier's sealed record cell `_salts[p]` until `Proxy.reveal(p, registration)` opens it on the PoUW window's
  registration after its last ingress record is published, once and only within `reveal_deadline` of the window's
  end on the certifier's clock, and `active.check_salts` reads the salt frames. The weights' root PoUW's noise seed
  reads is recorded before the salt's draw, signed by the registered prover (`Proxy.commit_weights(p, root,
  signature)` under `prover_key`, the sealed cell `_weights[p]`), and given on the same conditions
  (`Proxy.committed(p, registration)`). With `Proxy(challenger_key=…, signer=…)`, both or neither, the challenger's key
  from the auditor's deployment record, the two calls take only a request the auditor's challenger signed under LMS
  for the certifier's key id and run (`Proxy.run_id`, 32 bytes drawn at construction) and answer an `active.Revealed`
  the certifier signed, which names both; `active.serve_reveals` serves them over the network, one JSON line each way,
  to the challenger's `active.RemoteCertifier`, given the run id by the deployment record, which returns the value
  once it verifies for that run. Without the keys they are the in-process calls, for the auditor's side only. A proxy
  back from maintenance (`Proxy(start=k, position=p)`, off by default) serves from window k on, its first egress frame
  the Program's at position p, and carries nothing from before: a window it was down for has no record. On a served
  link (`Proxy(served=ServedProfile(…))`, off by default; with the salt, `own` 1 at a PoUW window of one certifier
  window) its ingress admits whole requests by the admission rule (`served.Admission`). On a sealed pair link
  (`Proxy(pair=active.Pair.of(…))`, off by default), between two of the auditor's certifiers, every slot of the link is
  sealed after its record and opened before it (`verity.ml.crypto.aead`, SHA-512 counter mode then HMAC-SHA512, a
  nonce per slot), so the records are the unsealed run's and a unit that doesn't open fails the link closed;
- an advised link's counts (`advised`, `LinkWindows(advised=AdvisedProfile(windows, request_frames, own))`), a
  served link under arbitrary traffic: N2's advised unit, `NetworkAdvisedLinkWindow`, adds `NetworkAdvisedCounts_v1`
  (63,031 ANDs at T = 600, R_in = 1), which holds the egress head's counts to the releases of the prover's advice
  (`LinkAdvice`: I1's `advice.Advice` and each window's request frames), and the ingress head's counts to the advice:
  each release eligible under the sync the certifier ran its window under (the verifier's own clock source,
  `LinkWindows.clock_source`: the certifier's salted clock certificates, `certificate.CertifiedClock`, each sync read
  through its opening, witness, against the digest the public certificate signs, which each window's
  declared sync must match, one of Σ_sync, or that window is refused), so never before the bucket of its request's
  last frame (the request's end on the ingress link, which `advised.ends` derives from v0's prompt lengths and the
  link's public framing), and each window's ingress total the advice's. Carry-over between windows is the advice's.
  With the advice fixed the counts are determined, so the egress timing carries at most `advice.bits` (`PROTOCOL.md`,
  "An advised link"). The prompt lengths are tied to the certified ones only by IO-TIE's `IoTieRequest` verdict
  (`advised.IoTieVerdict`, naming the link and its ingress registrations), which the verifier takes from its own tie
  source (`LinkWindows.ties`), never beside the advice: the tie service's (below). The served size class isn't
  pinned, so there every advised link is refused (fail closed), as is a link without the certifier's clock
  certificates;
- the ingress tie through the proof system (`tie_service`): `IngressTie`, each ingress link-window's `IoTieRequest`
  units, one a bucket row of the warden's registered ingress cells, proved as `NetworkIngressTieBucket` (one gate
  naming its pinned circuit digest, `io_request.circuit-check.json`) with the row read off the warden's root
  (`warden_rows.bucket_port`), decided by the verifier of record on its own lanes from the advice and the warden's
  release, and `IngressTie.tie`, which builds `advised.IoTieVerdict` on the link and the stream's registrations from
  the stream's outcomes of record alone; `Ties`, the tie, its stream and the release as the `LinkWindows.ties` the
  advised link reads. A size class no sidecar pins has no accepted proof.

It is pure Python. It imports only the standard library and `verity`: for the record digests' `tagged_sha256` and
`tagged_sha512` and `canonical_json_bytes`, in `service` the proof system, the hm96 leaves and the check's
Definition, and in `active` the proof system's registration and receipt, which `reveal` reads, and the pair link's
AEAD (`verity.ml.crypto.aead`). The certificate's signature scheme is a parameter: the certifier's
`certificate.Issuer` takes a `Signer`, and the audit a `PublicKey`.
From `verity`, `head_check` takes the circuits, their evaluator and the commitment gadgets, and `served` and `advised`
the circuits and their evaluator.
`PROTOCOL.md` has the semantics and the correspondence to the Lean analysis (`Security/`, area `NetworkCertifier`).
`lean/` is the certifier program, the Lake package `certifier_program` (Lean core only): List forms of the analysis's
definitions, proved equal to them in `Security/Proofs/NetworkCertifier/Code.lean`, and the difftest that holds this
package to them (`lake exe certifier-difftest INSTANCES PYTHON N` on what `lean/scripts/difftest_vectors.py` writes; the audit runs it
from `lean/lean-audit.json`). Advice v1's charge has its own: `lake exe certifier-charge-difftest INSTANCES PYTHON N` on
what `lean/scripts/charge_vectors.py` writes from `arith_encode.reference_charge`, and its whole window's
`lake exe certifier-window-difftest INSTANCES PYTHON N` on what `lean/scripts/window_vectors.py` writes from
`window_charge.reference_window`.

The two terms of the charge the capacity bounds leave open are fixed in the trusted text, in the last section of
`Security/Definitions/NetworkCertifier/Guarantees.lean`'s documentation:
- the content advice `A_c`, which is the schedule. It gets a bound once the deployment declares a deterministic
  scheduler, admission on a step clock and a registered list of N profiles: `log2 N` once per audit, or
  `log2(N · #Σ_sync)` per window when the profile may switch at window boundaries;
- the challenger's coins (D7), an ingress link whose timing is charged on the ingress grid and whose content costs
  0 bits. The coin link (`coin_link`, run by `benchmarks/live_challenger/coin_link.py`) carries a window's session
  between the prover and the challenger: it re-cuts every frame into cells and re-clocks both directions on their own
  grids, exempts only the verified, expected, once-only coin openings, and counts everything else
  (`protocol/coin-link.md`).

~~~python
from verity.core.protocols.certifier.audit import Audit
from verity.core.protocols.certifier.grid import GridParams, Certifier
from verity.core.protocols.certifier.schedule import ClockSync, LogicalFrame, SyncSet, run_shaper

params = GridParams(buckets=600, allocation=1024, frame_bytes=4096)   # 1-min windows of 100 ms buckets
sync_set = SyncSet(tuple(range(16)), tuple(range(1, 9)))              # 7 bits per window
frames = [LogicalFrame(b"\0" * 4096, anchor=3, index=i) for i in range(5)]
syncs = [ClockSync(delta=2, rho=4)]
records = run_shaper(frames, ready=[4] * 5, syncs=syncs, certifier=Certifier("responses", params))
audit = Audit()
audit.check_link("responses", records, frames, syncs, sync_set, params)
assert audit.accepted
~~~

Calibration, from a JSON trace of honest runs:

~~~text
uv run network-certifier-calibrate trace.json --ingress-bucket-ms 100
~~~

## Zero knowledge: what N2's verdict reads, and what's open

Every protocol is to be zero-knowledge. The network audit's claim is counts only: N2 holds each window's per-bucket
counts to the occupancy profile on every link-window (on a served link, to the ones its ingress counts derive under
the declared sync), beside the proved timing bound (provisional). N3, the content check,
is retired, superseded by the step-indexed scan (`ingress_scan`, `egress`).

N2's verifier of record, `LinkWindows.verify`, decides each link-window on a `--zk` proof of N2's registered unit
(`NetworkLinkWindow` v5, served `NetworkServedLinkWindow` v4) and public inputs alone: its own run of C-Flock's Lean
verifier accepts the proof, the proof's public row opens to its own copy of the unit's public arguments, the output
row's verdict and deadline bits are 1, the certificates verify on their public fields (`LinkWindows.certified`), and
the window's sync is declared. It reads no head, salt or sibling. `benchmarks/network_traces/n2_zk.py` proves a
certifier-serve run's link-windows this way through the link's own stream, and scans what the verifier received. The
in-the-clear check, `LinkWindows.clear`, is a debug comparison, off by default. What's open:
- **The certificates aren't a Spec item.** The verifier receives them in the clear from the certifier, the auditor's
  device, beside the proof system, whose public kinds have none for a trusted device's signature over a hiding
  commitment. That is a provisional exception to the four public kinds (decided 24 Oct, option A), not a fifth kind.
- **A run that proves nothing** (a rehearsal, the simulator) is decided in the clear (`LinkWindows.unproved_outcome`,
  its verdict `"decided_by": "clear"`), which is never zero-knowledge.

`LinkWindows.verify`'s verdict of record on a plain, served or warden link is the Lean's, `NetworkCheck.LinkWindow`
(`verity/core/protocols/network/lean`, the `link-window` function of `network-records verdicts`), by the one switch line
`service.OF_RECORD`; an advised link's stays the Python's. `verify` runs both on every proof, and a link-window where
they differ in the accept bit, the public outputs or the digest, or which the Lean refuses, is rejected. The switch went
to `lean` under the 8 Oct ruling (the friction skill), on these grounds:
- the gap list is empty for plain and served links. Everything the Python checks, the Lean checks: the findings prove
  the registered Program and the drawn units with every gate pinned, the public row the verifier derives (root, served
  domain, link, window, each end's grid and profile row, both signed digests, a served link's sync words), both rows'
  openings and the word 0b11, both certificates under the LMS key in the certifier session, and the sync in Σ_sync. Both
  take as inputs what the proof system and the lowering give: the findings of the verifier's own C-Flock Lean run, the
  sessions' SHA-512s, the registration, the draw, and the Program's and partition's digests and the gate ids. On an
  advised link the window view (`advised.view`, from the tie and clock sources) is the Python's and an input to the Lean,
  which is why advised stays `python`. The Lean reads an LMS key only, and rejects any other.
- on a warden link (`service.WardenProfile`) the gap list is empty too: the Lean's warden form (`Kind.warden`) is the
  plain unit's public row with both ends' grids at the warden's parameters digest, which both take as an input (the
  warden's deployment publishes it), and no sync read, as in the Python; it refuses a warden case whose digest isn't 64
  bytes, whose ends have a profile, or whose two grids differ.
- every proving run of N2 recorded before the switch replayed through both to the verdict it recorded, negative
  controls included: 193 link-windows of seven runs, plain and served, 180 accepted and 13 refused, each with nine
  injected controls that both reject (r20261026-050730-1027, again r20261026-104647-916d). The
  warden runs give 30 link-windows from r20261026-080827-2bd2 and r20261026-100351-ca6a, the latter a draw of 8 on the
  H200 warden run c20261026T092338Z-22b5. Of those, 21 are accepted and 9 refused (r20261026-104647-c07b,
  r20261026-105654-6220). Each has ten controls: nine that both reject, among them another warden's digest, and
  `unread`, a sync off Σ_sync that changes neither verdict.
- the Python keeps running beside the Lean as the reference.
- the Lean's verdict has its property in `Security/Properties/NetworkCertifier/LinkWindow.lean`
  (`Properties.NetworkCertifier.LinkWindowAcceptMeans`, proved): an accepted link-window was proved, opened to the
  word 0b11, certified on both ends and synced within Σ_sync (a warden link's syncs are not asked of), and its public
  outputs are verdict 1 and deadline 1.

`tie_service.IngressTie.verify`, the ingress tie's verifier, is not ported and stays Python.

A proof of the unit is held to its circuit. Each gate names its Definition's circuit digest
(`Gate[NetworkHeadCheck_v2{…}@<digest>]_v2`, `served.gate` likewise), as `head_check.circuit-check.json` and
`served.circuit-check.json` pin it and circuit-check holds the pin to the Definition. So the unit's Program, which the
spec and each registration bind, binds both circuits, and `LinkWindows.verify` refuses a proof that names any other
Program, or a unit of a size class no sidecar pins (`LinkWindows.proved`).

The certificate no longer signs a hash of witness (NET-42). The certifier signs `D = SHA-512("verity/net-cert/v1" ‖
salt_c ‖ R)`, R the head's record digest and `salt_c` 64 bytes the certifier draws fresh for each record
(`certificate.Issuer`'s `coins`) and hands to the prover with the head, as witness only (`CertifiedRecord.salt`). D is
a hiding commitment to R, and the LMS check (`Certificate.verify`) reads only the certificate's public fields and D.
Each end's head check holds R to the head it opens and D to R under `salt_c`, so the certificates' check reads nothing
of the heads; the in-the-clear check also checks that link over the opening (`CertifiedRecord.verify`).

Tests: `uv run tools/verity/check/suites.py verity/core/protocols/certifier`.
