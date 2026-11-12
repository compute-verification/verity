# Network certifier (`verity.protocols.certifier.verifier`)

## The network bound

The network bound limits what a covert sender in the serving node can tell a client through the network certifier
beyond what the registered Program computes from its committed values: per execution phase, the probability that an
accepted audit leaves the certified traffic outside what that Program accepts, and over a monitoring period, the number
of bits of such a sender's secrets a decoder recovers from the client's view.

The guarantee is `NetTiming.Guarantees.PeriodLeakAccepted`
(`security/Definitions/NetworkCertifier/Guarantees/Period.lean`), proved as
`NetTiming.SecurityProofs.PeriodLeakAccepted` (`security/Proofs/NetworkCertifier/Period.lean`). Over a monitoring
period of n execution phases (the Lean's windows), execution phase i audited by the compiled sampled verifier at its
registered statement, with one network certifier on the server's uplink running its code on schedules in `Ss`:

- per execution phase, at a statement in scope, `Pr[accept ∧ ¬RLeakCertifier(δ)] ≤ δ + ε_ks + ε_bind + Pr[B_tap]`;
- where the network certifier registers its observations in every execution phase, the secrets whose every execution
  phase meets its good event (`WindowGood`) number at most
  `#Ss · ∏_c (cap_c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #faultySets_i(δ) · 2^β_i · Dj^k_i`, whose log2 bounds the bits.

`RLeakCertifier(δ)` says every observation sequence the network certifier registers is one the registered Program
accepts at the trace's committed values with some incorrect set the audit allows at δ. ε_ks and ε_bind are the sampled
verifier's errors (`Compiled.KS`'s) and `B_tap` a collision between the extracted trace's openings and the network
certifier's. `cap_c` bounds connection c's cells; `freeInputs_i \ reg_i` are execution phase i's free inputs (the
advice) outside its registered inputs; `faultySets_i(δ)` are its possible incorrect sets at δ and β_i bounds the bits
any of them commits; k_i is its steps and `Dj` bounds each step's jitter class.

It bounds each execution phase's failure on its own: the union over the monitoring period's live coins is not stated,
and `WindowGood` adds a condition no proof gives, the registration's values at the prover's registered inputs. It is a
statement about the network certifier's code (`recordingCode`), not about runs of the network check's Lean
(`lean`, `NetworkCheck`), which is the verifier's code and states no bound.
`NetTiming.Guarantees.PooledLeakAccepted` (`security/Definitions/NetworkCertifier/Guarantees/PooledLeak.lean`) is the
pooled count, with the incorrect proof blocks paid for once over the monitoring period,
`#Ss · ∏_c (cap_c + 1) · (N + 1)^Fmax · 2^(b · Fmax) · ∏_i 2^#(freeInputs_i \ reg_i) · Dj^k_i` at `Fmax ≥ ln(1/δ)/q`
(N the period's proof blocks, b each one's committed bits, q the per-execution-phase detection probability), failing
with probability at most `δ + Σ_i e_i + Σ_i e'_i`. It is conditional on two per-execution-phase hypotheses not yet
discharged (`hesc`, the conditional escape; `hleak`, the leak relation at the witness's incorrect set), so it is not
yet the deployed system's bound over a monitoring period.

What `PeriodLeakAccepted` takes, as its record in the lock (`security/lean-audit.json`, `guarantees`) lists it. Named
assumptions: none. A premise that is part of what it says (`domain`):

- `Claims.Sampling.CommitmentScheme.Binds`: the trace's commitment binds, a break giving an explicit collision.

Open premises (`open`):

- `Verity.Assumptions.TrustedHost` (`trusted-host`): the network certifier, the auditor's device on a host the developer
  doesn't control, runs its stated code.
- `Claims.SamplingGeneric.Compiled.KS`: the compiled sampled verifier is knowledge sound at its errors on the
  statements in scope.
- `NetTiming.ScopeAcc`: the statement is in the sampling scope and its boundary layout fits its Program over its steps.
- `NetTiming.CertifierRegisters`: the network certifier's openings pass under the statement's roots, cover every
  egress and ingress gate, and hold its record's rows there; the antecedent of the period's count.

`PooledLeakAccepted`'s record lists one open premise, `Verity.Assumptions.TrustedHost`.

## Interface

- **Python**: `verity.protocols.certifier.verifier`. Code outside this directory imports every module but `boundary`,
  `capacity` and `pacer`. Its importers: the prover side, `verity/protocols/certifier/prover`; `../device`'s vector
  scripts; `adapters/vllm` (`active`, `advice`, `blocks`, `charge_chain`, `compile_slice`, `io_tie`, `release`,
  `served_rows`, `warden`); `verity/composite/device/live_challenger`; the tests of `../../pouw/verifier` and
  `tools/verity`; and in `benchmarks/`, `network_traces`, `network_e2e`, `network_sim`, `prove_path`, `one_stage`,
  `pouw`, `remote_session`, `mempool` and `compile_cost`. In this directory, the network check's tests (`network/tests`)
  and vector script (`lean/scripts`) import it too. `../../spot`, `../../flock` and `../../recursion` import none.
- **Command**: `network-certifier-calibrate` (`calibration:main`), a console script of the root `pyproject.toml`.
- **Lean here**: `lean/` is the Lake package `network_check`, whose library `NetworkCheck` the
  packages at `../device/lean`, `../../onsite/verifier` and `security/Properties` require by path. Its executables:
  - `network-check`, `network-records` and `network-period`, each `id | check PARAMS COMMITMENT EVIDENCE | serve`: the
    core protocols `verity/network-certificate/v4`, `verity/network-records/v3` and `verity/certifier/v0`;
  - `network-records verdicts`, the process `lean_audit` starts and `service` asks (`VERITY_NETWORK_RECORDS_EXE` names
    a built one); `benchmarks/network_e2e` and `benchmarks/network_traces` build and run it;
  - `network-check-difftest INSTANCES PYTHON N`.
- **Lean beside it**: `../device/program/lean/`, the certifier program, is the Lake package `certifier_program`, whose
  library `NetworkCertifier` the packages at `lean`, `../device/lean` and `security/Proofs` require by path. Its
  executables (`lake exe`): `certifier-difftest`, `certifier-charge-difftest` and `certifier-window-difftest`, each
  `INSTANCES PYTHON N`. The Lean check runs each `../device/program/lean/scripts` script, then its difftest
  (`../device/program/lean/lean-audit.json`).
- **Documents others read**: the answer of `network-period check`, `verity/certifier/v0`. `benchmarks/network_traces`
  runs the check and writes it as the certifier's piece, which `benchmarks/one_stage` reads.
- **Files others load by path**: `tests/scan_traffic.py` (`benchmarks/network_traces`, `benchmarks/prove_path`, the
  tests of `adapters/vllm`); `tests/egress_traffic.py` (`benchmarks/network_traces`); in `tests/fixtures`,
  `compile_slice_smallkv.json` (`benchmarks/network_traces`, `benchmarks/prove_path`) and the vectors
  `benchmarks/network_e2e/encodings` writes; `advice.py` (`benchmarks/network_e2e/encodings`); and in `network/tests`,
  `period_case.py` (`benchmarks/network_traces`), and `test_network_period_protocol.py` and
  `test_inference_only_protocol.py` (the tests of `../../remote/verifier/composition`).
- **Loads from**: `lean/`, six Lake packages by path: `certifier_program`, `flock_verifier` (`../../flock/verifier`),
  `nci_check` (`../../inference_only/verifier/nci/lean`), `core_protocol` (`../../interface`), and `lms` and
  `core_identity` (`verity/commitments/lms/lean`, `verity/commitments/identity`); the certifier program, no package.
  The modules, `verity.circuits`, `verity.commitments`, `verity.commitments.crypto`, `verity.protocols.spot.verifier`
  and, `coin_link` alone, `verity.protocols.spot.device`, never the prover side, which
  `../device/program/lean/scripts/window_vectors.py` and the tests import.
- **Not here**: the certifier's host programs (`benchmarks/network_traces`).

## What this package holds

A spec-level reference for network accounting's timing channel:
- the constant-rate, left-packed certifier;
- the Program's logical schedule, with its clock-sync advice (δ, ρ);
- the exact audit with record-and-reject, on raw records, records committed by hash (padding elided), or network
  certificates (`certificate`: the network certifier's signature over the record's SHA-512 digest hidden under a
  fresh salt, with the execution phase's frames in a fixed-depth SHA-512 tree), and the ingress links' status check;
- one execution phase's certificates through the proof system (`service`): the network `Spec`, a salted root over the
  certified records, and the check, whose Boolean part is each head's check (`head_check`: the head's hiding opening,
  its fields, its counts against the public occupancy profile, and its record digest under the certifier's salt
  against the digest its certificate signs), with a host-side verifier (no proof of the
  check yet) that compares no grid in the clear. Its claim is counts only: on every execution phase the head check holds
  the egress head's counts to the profile (and the ingress head's, where that end declares one), so the link's timing
  is at the profile. Nothing here holds the frames, the rest of the egress grid. The network verdict of record is
  `LinkWindows.network_verdict`, the AND over the link's execution phases with no ε or share credit: every window of the
  egress profile registered in order with an accepted outcome, and a rejected, late, missing or out-of-order window
  rejects the link;
- a served link's counts (`served`, `LinkWindows(served=ServedProfile(F, F_in, windows, own))`), whose egress end has no
  occupancy profile: N2's served proof block, `NetworkServedLinkWindow`, adds a Boolean Definition,
  `NetworkServedCounts_v1`, that holds the egress
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
  what its size class or the prefix layer's counts can't express. It is the compile slice's proof block's core, between
  the committed decisions and `compile_steps`' Programs, and is held step for step to the replay
  (`benchmarks/network_e2e/encodings/replay_vectors.py`); it is in no proof-system Spec yet;
- the I/O tie (`io_tie`), a design (`PROTOCOL.md`, "The I/O tie") and its prototype on a fake upstream, in no Spec:
  served through the network certifier's egress, every cell of an egress execution phase is opened, each a pad, the
  canonical event of one token id or the response's end on the stream the network certifier set, and each is the cell
  the advice's map names at its place (`window_map`, from the advice's outputs, release buckets and advice v1's order
  permutation, each request on its ingress ordinal's stream from 1; the verifier
  computes it in the clear from the opened advice, the named gap, until the map is read in the proof under the advice's
  hiding root), with the id the Commit sampled for it (the port: one hiding row of ids a step, read under the Commit's
  root). The cells are the network certifier's (`warden`): a connection's r slots a bucket, a `PAD` on its stream before
  its first cell and after its `END`, one id event a `DATA` cell. The network certifier commits each bucket's cells as
  one hiding row. It has the reference and the proof block's Boolean Definition, one a bucket row (`IoTieBucket`: the
  row and the port's rows read as registered rows, the network certifier's and the Commit's (`port_reads`), then
  `IoTieContent`: `IoTieCell` on every cell, the prover's Beneš routing of the port's entries `IoTieRoute`,
  `IoTieMatch`);
- the ingress tie (`io_request`, `PROTOCOL.md`, "The ingress tie") on the network certifier's own ingress (`serve`): one
  proof block a registered bucket row (`warden_rows.bucket_port`), each live connection's lane of r slots read as the
  network certifier's cells (a `PAD` on its ordinal, a `DATA` of exactly B − 9 bytes, its `END`), each request cell a
  cell of CERT-VARLEN's id
  request, with its count of prompt fields carried across buckets from the lane's carry-in to its carry-out, which
  hold each request's prompt_len to the advice's and which the ingress tie's verdict (`tie_service.IoTieVerdict`) is
  built from. It has the reference, the lanes the verifier computes from the advice and the network certifier's release
  (`lanes`, `Release`), and the proof block's Boolean Definition (`IoTieRequest`: one `IoTieRequestFrame` and
  `IoTieRequestStep` a cell, `IoTieRequestEnd` a lane); the ingress-ids tie (`io_ids`, `IoTieRequestIds`) adds the
  Commit's `c00.input/ids` rows to the same scan and holds every prompt id to them;
- the capacity counts;
- the egress side of the step-indexed scan (`egress`, N2) on a fake upstream, in no Spec: one proof block a link and
  bucket t (`EgressBucket`) that reads the scan's slot state S_t, a registered read, and computes the bucket's row of
  cells from it with every switch computed and none witness (the generator form): its outputs are a u16, `ACCEPT`
  exactly when the state renders a row, the one public output, and the row, committed as a registered read of the
  network certifier's egress row t (`warden_rows.bucket_port`), which opens only if the computed row is the network
  certifier's. It has the reference (`bucket_map`, `computed`, `rendered`, `outputs`) and the state's egress-view layout
  (`Shape`'s offsets), which the scan's ingress step writes;
- the lengths row's proof block (`verity.protocols.certifier.prover.lengths_row`, `LengthsRow`), in no Spec: one a link,
  execution phase and slot stride (`PROTOCOL.md`, "The lengths row's proof block");
- the response tie's two proof blocks (`response_tie`, beside it), in no Spec;
- the calibration of honest runs;
- advice v0 (`advice`): what vLLM's scheduler decided in a session (each request's lengths and finish, each step's
  rows with their chunks, prefix-cache hits and the KV they reuse, resumes, and each kept output's egress bucket) as one
  canonical bit string the compiler reads, refused unless it is a schedule vLLM could have run, with its bits split by
  the earliest egress window that can show each record (`bits`). v0 is the committed form; the charge is ENC-CODE's
  compressed length of the advice;
- advice v0's validity (`advice_check`): `AdviceCheck_v1{L}`, the compile slice's proof block's first part as a Boolean
  Definition, 1 exactly when the committed bits (at most L) decode to a v0 advice, up to the three refusals of `decode`
  it defers (unique ids, a source's KV computed there, release order per request); its AND count at each L is `size(L)`;
- the replay's state chain (`replay_chain`): ReplayStep_v1's state handed from one step proof block to the next as the
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
  (`protocol/compile-slice.md`);
- the compile slice with the block pool (`compile_slice_v2`): `CompileSlice_v2`, `CompileSlice_v1` with its prefix cache
  replaced by `CacheStep_v2`'s pool, its calls in vLLM's order on the pool rows in ``wit``. Each free's ``held`` and
  each compute's ``new`` are `ReplayStep_v1`'s own, and the hits, evictable counts and returns `ReplayStep_v1` reads
  are the pool's, so eviction, preemption and a leaving request's returns are in the slice
  (`benchmarks/network_e2e/encodings/compile_slice_pool.py` holds its calls to the advice decoder's block pool,
  `verity_vllm.pipeline.advice_code.pool.BlockPool`, over whole traces);
- the compile slice computing its schedule (`compile_slice_v3`): `CompileSlice_v3`, `CompileSlice_v2` computing what v2
  checks. From the advice (DA decisions and the step count), the arrivals' ingress fields and their registered output
  lengths and tokens, it computes the served rows and the state, window and pool rows the window leaves, each hit,
  evictable count, return and stamp where `ReplayStep_v1` reads it, and holds them to the registered ``srv`` and
  ``cout`` it reads; its one output is the 16-bit verdict. So its only free input is the advice, ℓ_w = DA + PW bits
  (`free_width`), and the charge is ℓ_w;
- the compile slice over the memory-checked pool (`compile_slice_v4`): `CompileSlice_v4`, `CompileSlice_v3` with its
  pool calls records on a bus to `CacheStep_v3`'s stream, which runs in the unit from the empty pool with request keys
  the arrivals' `BlockKey_v1` chains printed under a live coin. The hits, evictable counts and returns the steps read,
  and the stream's slots and touched list, are checked witness, unique and bound by live coins. Until a Lean lemma
  shows that uniqueness, the witness counts as free input at its width, so the free
  input is the advice and the checked witness (`free_width`). A window costs its stream's slots, whatever the pool's size. Its served rows are CompileSlice_v3's, and
  `benchmarks/network_e2e/encodings/compile_slice_pool.py` (`check_trace_v4`) holds its calls to the replay's `_KV`;
- the compile slice at its window's activity (`compile_slice_v5`): `CompileSlice_v5`, `CompileSlice_v4` with the
  arrivals' tokens registered in ``req`` as one ragged row of the NRK blocks the window keys (a keyed block count a
  slot, and one key a block), RB's touched list the NRB addresses the stream touches, and each running slot's request
  number carried in the slot row, which ``cout`` registers, rather than looked up at every call. Its free input is the
  advice and the checked witness, as v4's (`free_width`), and its verdict adds bit 6 ``keys``. `parts` prices it at any
  class;
- the compile slice with the block pool carried between windows (`compile_slice_v6`): `CompileSlice_v5` cut in two units,
  both proved every window. `CompileSlice_v6` is its load and steps from the registered results, their pool calls'
  bus records held to the registered calls row. `PoolWindow_v1` opens the pool row under the input root, the previous
  window's output root registered before the window's coins, as the memory check's initial values; runs `CacheStep_v3`'s
  stream from it over the calls row with v5's ragged keys; and commits the row the stream leaves as the output root. A
  window from the empty pool's root (`genesis`) is v5's, and a chain restarts from it after a failed window. The compile
  slice's free input is the advice, the pool window's its checked rows (the left row, the blocks' sources and the
  stream's witness) at their widths until their lemma (`free_width`). A request still open at the window's edge is cut
  (its registered ``a_cut`` bit): `CompileEdge` frees every running slot in vLLM's preemption order and holds each open
  request to having emitted its ``a_O``, and the next window takes its continuation, (id, P + e) with its prompt then its
  released outputs, as an ordinary arrival that hits its blocks in the carried pool. The close gives each keyed block's
  page its source, the window's own write or the previous window's carried port, and a `PageCopy_v1` copy block holds
  the window's carried port to it. `PoolChain_v1` is two windows so chained, and `parts` prices both units at any
  class, which `benchmarks/mempool/cost.py` does at a served load's;
- the compile slice charging the length terms (`compile_slice_v7`): `CompileSlice_v7`, `CompileSlice_v1` with
  `ChargeLink_v2{D, 0, 0, K}` in place of `ChargeLink_v1{D}`, a length slot for each certified arrival read from
  ``cert`` (its output length and its certified max_tokens), so a window's charge covers `code_v1`'s segment and its
  length tail. vLLM's served windows name it (`serve_commit.SERVED_SLICE`), except under `--admit tick`, and
  `benchmarks/network_traces/compile_slice_zk.py` proves it in zero knowledge (and plain, ``--plain``);
- the served rows (`served_rows`): the step rows a served window ran, from the server's kept rows, committed before the
  sample under the run root as a hiding-commitment registration, a row each block of G steps (G = P unless given: a window's row,
  the slice's ``srv`` row; at G < P block j is the slice's ``srv{j}``, `compile_slice.CompileSliceRowGroups_v1`). `leaf_ok`
  holds a proof block's row b‖c to its registered leaf, as the advice's ``adv`` row is held,
  and `reads_port` is the same read as the Lean verifier's registered reads take it;
- the served rows' tie (`served_tie`): `ServedRowsTie_v2`, a Boolean Definition with `check` its reference, holds a
  window's served rows to the positions its steps' Commit holds (the step-input port, `port_chunks`: each step's ids and
  then its positions at the head of its stream), so the rows registered at serve time are the batch the server ran. Every
  proof block is proved, and its port, selection and cost are its docstring;
- the served rows held to their Programs (`served_statics`): `ServedStaticsTie_v1`, a Boolean Definition with `check`
  its reference, holds each block's registered served rows, read through the Lean verifier's registered reads, to the
  RoPE statics of the Programs its steps ran, with no row hashed but the block's
  ([protocol/served-statics.md](protocol/served-statics.md));
- the online prover (`online`): each window's clock sync declared at its first bucket from what it had seen `lag`
  buckets before;
- the prefix cache (`cache_step`): `CacheStep_v1`, a Boolean Definition of one scheduled row's pass over a table of
  whole cached units (an engine step's calls are `scan_step` and, as a circuit, `build_scan_step`), which checks the
  row's hit (the longest cached prefix while no block can have been evicted, a
  cached prefix past that) and owns or appends the units the row computes (`PROTOCOL.md`, "Advice v1");
- the block pool (`cache_step_v2`): `CacheStep_v2`, one call's pass over vLLM's prefix-cache block pool: which slots
  hold each cached block, the free queue's order by stamps, eviction named by a threshold stamp the call checks, and a
  leaving request's returns. So a lookup's hit and evictable count and a free's returns are derived past the
  cumulative pool bound too, in N = NB − 1 entries. `compile_slice_v2` wires it into the compile slice;
- the block pool as memory-checked RAM (`cache_step_v3`): `CacheStep_v3`, a window's pool calls as one stream of NS
  access slots over the five memories vLLM's `BlockPool` is made of (blocks, keys, key copies, request blocks, request
  keys), checked by `MemCheck_v1`'s fingerprint under live coins, with a bus product of the calls' records for
  `ReplayStep_v1`. A window's price is NS slots and NE list entries from a public ladder, whatever the pool's size;
  `Stream` is its specification, held to the replay's pool by `benchmarks/mempool/stream.py`;
- advice v1's arithmetic encoder (`arith_encode`), Boolean Definitions each with its reference as spec. The counting
  variant, which the composition takes: `ArithCountStep_v1{K, BIN, CW}` runs ENC-CODE's 32-bit coder forward on one
  symbol and counts the bits it puts in the window's segment, `ArithCountFinish_v1{CW}` adds the termination and
  holds the window to its declared charge, and `BookKT_v1{C}` is `code_v1`'s KT estimate for a bit's context. `ChargeCount_v1{N}` composes them into one window's charge from its decisions, the compile
  unit's one call a window. The context and the frequencies are the circuit's, from the replay's position and the
  registered book's counts, never witness. The bit-checking variant, the alternative: `ArithEncodeStep_v1{K, BIN}` and
  `ArithEncodeFinish_v1` check each output bit against a committed segment in the decoder's order, with
  `KTContext_v1{C, NW}` ENC-REPLAY's scaled estimate. It is held symbol for symbol to `code.py` and `code_v1` on advice
  v1's windows and on ENC-DERIVED's still-sent fields (`benchmarks/network_e2e/encodings/arith_vectors.py`), and it is
  in no proof-system Spec yet (`PROTOCOL.md`, "Advice v1");
- the charge's carry chain (`charge_chain`): `ChargeCount_v1`'s carry committed from window to window as hiding commitments,
  window 0's the book's priors under a public salt, with the verifier's check of how the windows join (`check_chain`)
  and `ChargeLink_v1{N}`, the charge from the carry-in row held to the carry-out row (`protocol/charge-chain.md`);
- the whole window's charge (`window_charge`): `ChargeCount_v2{N, H, O, L}` and `ChargeLink_v2{N, H, O, L}` charge
  advice v1's window (`Window`, the record `code_v1` codes) with its hits and order entries and their counts, coded as
  `code_v1` codes them, and its length terms' tail, with the order's position and rank added to the carry and chained
  under the value
  `window-charge` (`protocol/charge-chain.md`, "The whole window");
- the active network certifier (`warden`): a relay between any client and any server. It forwards each direction's
  bytes unchanged, in B-byte cells, r a bucket each way per connection, and egress cell i goes out in bucket
  s + D + ⌊i/r⌋ from its admission s. In the id egress form, its default, it releases a server's cell only if
  `io_tie.parse_cell` reads it ("The egress record"), and fails the connection (`format`) at any other, never
  rewriting it. It records and certifies exactly the cells it sends, and each failure's reason in a third record,
  `fails`.
  `PROTOCOL.md`'s "The active network certifier" has its leak bound, and `benchmarks/network_traces/warden_serve.py`
  runs it;
- the release order on the server's side (`release`): a certifier puts `x-verity-release: N` (N is the request's
  0-based place in its run's release order) and `x-verity-release-run: ID` on the server's copy of each request it
  releases, for `serve-openai --admit release-order`. The network certifier adds `x-verity-release-bucket: a`, the
  bucket of the request's `END`. With `Warden(watermark=…)` it also sends the server a watermark stream:
  one chunked `POST /verity/watermark` carrying one line `t n l…` a tick, meaning bucket t has closed, the releases of
  buckets ≤ t are exactly N < n, and each l is one of bucket t's that its server connection lost. Line t goes out once
  each release of bucket t is written or lost. An engine that admits at a logical time in certifier buckets waits on
  that line and on each N < n it doesn't name.
  The headers and the stream never appear in a cell or record, so nothing certified changes;
- the network certifier's cells as rows (`warden_rows`): each execution phase's cells committed one `hm96-sha512/row/v2`
  row a bucket, padding and empty slots included, under a frame-v3-sha512 root the network certifier signs with the
  certifier's signer, beside a salted digest of the window's wiring. The registration is what a proof block's
  `--registered` port reads, and the verifier places each read, the (connection, cell) the Program's proof blocks'
  wiring pins (`Units`, whose digest the port's value names), with the network certifier's wiring (`PROTOCOL.md`,
  "The network certifier's cells as rows"; `benchmarks/network_traces/warden_zk.py`);
- the network certifier's connection rows (`connection_rows`), in a schedule with lengths rows
  (`WardenParams.lengths`): each execution phase's releases as two tables a row a slot stride, `lengths` and `fails`,
  and, with content rows (`WardenParams.content`), three tables a class of what each filed release carried
  (`content_rows`: its request, response and end rows), all committed as the cells' rows are and listed under one
  head the network certifier signs once; the registration the verifier reads (`check_registration`) and its
  `--registered` port of the proof blocks that read a table, one a row (`port`); under form 2, the signed word
  that an execution phase has no head (`check_no_head`);
- the output value the certifier relayed as a registered value (`out_commit`): `commit_out` commits its rows at the
  reading program's registered shape (`OutShape`: its program, the value, ``words`` a row and ``rows``, both sides
  taking them from the circuit's registered read) under the domain the verifier derives for that pinned value, with the
  certifier's fresh salts, and signs the head for the egress execution phase; `check_out` reads the public record for
  the execution phase it expects at the verifier's own shape and returns the root to pin, or refuses by name
  (`REFUSALS`: another width is ``out-width``, another row count ``out-rows``), so the output proof blocks' rows are
  registered reads of the certifier's root rather than a comparison in the clear;
- the run's boundary in a window's circuit (`boundary`): the circuit record's ``boundary``, which names the template
  whose proof block ℓ·T + t holds bucket t of egress link ℓ (`EgressBucket`'s frame, the row it computes and returns,
  committed as a registered read of the network certifier's row, and its u16 verdict, whose bit 0 is the slots' presence) and of
  ingress link ℓ (its frame parameter row, the network certifier's registered row; with ``first``, `IngressScanFirst` holds bucket
  0), the same gates whatever the load, so arrival steps and prompt lengths stay private (`NetTiming.BoundaryGates`); its
  format is `verity/boundary/v3`. `refusal`, the verifier's check before it samples, checks that the layout fits the
  circuit (`NetTiming.BoundaryGates.Fits`: no egress gate is an input, and each is one the circuit returns) at the record's
  grain, which contains the gate grain's, held to `prog_fits`, `BoundaryGates.Fits` itself on a flattened program; the scan's
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
  proof; the class circuit and its padded counts (proof blocks per host, bits per link) are public, and its ``blocks``
  state no placement (the window interface's `Blocks`). `check` refuses an ``ingress`` declaration, a registered read
  or table of no registered parameter, a load (every registered parameter but the scan's state) read
  in other than whole rows of one width, by no unit, at an external port or by a table over part of it, and an external
  input port whose rows aren't each read exactly once from the scan's state (`blocks.SCAN_STATE`), the host's one
  ingress. `pins` gives the verifier's pinned roots of the loads from their certificates alone (`certified`: each
  load's registration on the host's uplink, ingress, in the session's first execution phase, kind ``load``), so the
  interface's ``input_count`` under them and the scan's state's root has no free input;
- the scan's ingress side (`ingress_scan`): `IngressScanBucket`, one unit a certifier bucket, reads the bucket's
  network certifier's ingress row, the slot state the bucket before carries and the Programs' entries, one port `prog<j>` a slot,
  each `SlotEntry_v1`'s word-aligned row of its slot (registered reads), and computes the next state (each block to the
  slot holding its connection, a new connection to the lowest free slot, its prompt ids appended, a due, occupied
  slot's output count and ids taken from its entry at its anchor, every other entry ignored, a slot held through its
  last egress bucket, `admitted + D + ⌊n/r⌋`) and its egress view, the state as `egress.EgressBucket` reads it (bucket,
  then per slot live, q, admitted, n and the output ids), its outputs beside a public ok bit, which is 0 at an
  occupied slot whose entry of the Programs' row isn't read (its read bit 0: a request no Program served). Unless the
  Programs' row is a compute audit's `SlotEntry_v1` template's entries, that template a pinned Definition (its closure
  digest one of `slotEntryPins`), the read bit is the prover's word (an entry of read 1 and no outputs passes), so the
  period check refuses such an execution phase (`NetworkCheck.Period`'s `wired`), and one whose scan's state isn't that
  audit's `SCAN_STATE`, by root and domain (`stateTied`): the scan's state tree is the window state (`window_state`),
  each row but the empty state's the one a unit computes (its o1) and the next reads (its sin), both registered reads,
  so a Program reads the states the scan's steps computed;
  `IngressScanFirst`, the first bucket's, starts from the empty state, constants of its circuit. The period check, the
  step from a period's records to the network bound, is not in the public tree, because it reads the inference-only
  policy's layout check, which is withheld. Programs start at
  fixed slots a network timestep: a timestep is `EVERY` buckets, the network certifier's `every`, which divides its
  execution phase, with `SLOTS` = lanes Program slots each, and only `EVERY` and the slot count are public. The token-id row a
  slot's Program reads is the scan's state row b − 1 (`verity/certifier/scan-state/v1`, `token_row`): its occupancy
  word `occ` (1 for a slot whose Program starts at b, 0 for an empty slot), its prompt's length `fields` and its prompt
  ids, which the scan reads off the network certifier's signed ingress cells, so no other tie holds them to the certified bytes;
  `slot_row` gives each one's bits in the state row, for a reader that takes them as i32s (vllm's `ScanSlots_v2`), and
  `window_state` the rows a window circuit of the execution phase from bucket t0 reads, the scan's rows t0 − 1 on (the
  empty state's for row −1). A Program gives all its outputs to its entry at its anchor (its read bit, n, u32, and 64
  ids of 17 bits at w63's class, 71 words, the row `SlotEntry_v1` writes); an execution phase's
  bucket t reads slot j's entry at row j·nb + ⌊t / EVERY⌋ of one slot-entries input, and an empty slot's entry is
  ignored (its outputs masked by `occ`, never on the egress). A request the class can't hold is refused at the scan's
  unit, never truncated: a prompt of more than `KI` ids (1,024 at w63's class) at its cell, a Program's n over `KO` at
  its anchor. A window whose network certifier refused a connection, traffic past its slots, has no certified record, and the
  period check refuses it. `Class.serves` names the network certifier parameters it takes every run of.
  `benchmarks/network_traces/scan_zk.py` proves both sides of a group of execution phases in one registration: ingress unit
  t's state and view outputs are committed as registered reads of the scan's state and view rows t, which ingress unit
  t + 1 and egress unit t read, and egress unit t's row as a registered read of the network certifier's egress row t, so the
  chains are wires through the registered roots;
- the active certifier (`active`): an asyncio TCP proxy that runs the certifiers on a live link and publishes their records.
  With its salt on (`Proxy(salt=active.fresh_salt)`, off by default), each PoUW window (`pouw_window` certifier
  windows) gets one salt of `salt_bits` bits (512 by default), carried into the block only. The ingress record holds a
  salt frame with a salt-free placeholder, so every record is a function of the traffic alone; the salt sits in the
  certifier's sealed record cell `_salts[p]` until `Proxy.reveal(p, registration)` opens it on the PoUW window's
  registration after its last ingress record is published, once and only within `reveal_deadline` of the window's
  end on the certifier's clock, and `active.check_salts` reads the salt frames. The weights' root PoUW's noise seed
  reads is recorded before the salt's sample, signed by the registered prover (`Proxy.commit_weights(p, root,
  signature)` under `prover_key`, the sealed cell `_weights[p]`), and given on the same conditions
  (`Proxy.committed(p, registration)`). With `Proxy(challenger_key=…, signer=…)`, both or neither, the challenger's key
  from the auditor's deployment record, the two calls take only a request the auditor's challenger signed under LMS
  for the certifier's key id and run (`Proxy.run_id`, 32 bytes sampled at construction) and answer an `active.Revealed`
  the certifier signed, which names both; `active.serve_reveals` serves them over the network, one JSON line each way,
  to the challenger's `active.CertifierClient`, given the run id by the deployment record, which returns the value
  once it verifies for that run. Without the keys they are the in-process calls, for the auditor's side only. A proxy
  back from maintenance (`Proxy(start=k, position=p)`, off by default) serves from window k on, its first egress frame
  the Program's at position p, and carries nothing from before: a window it was down for has no record. On a served
  link (`Proxy(served=ServedProfile(…))`, off by default; with the salt, `own` 1 at a PoUW window of one certifier
  window) its ingress admits whole requests by the admission rule (`served.Admission`). On a sealed pair link
  (`Proxy(pair=active.Pair.of(…))`, off by default), between two of the auditor's certifiers, every slot of the link is
  sealed after its record and opened before it (`verity.commitments.crypto.aead`, SHA-512 counter mode then HMAC-SHA512, a
  nonce per slot), so the records are the unsealed run's and a unit that doesn't open fails the link closed;
- the ingress tie through the proof system (`tie_service`): `IngressTie`, each ingress execution phase's `IoTieRequest`
  proof blocks, one a bucket row of the network certifier's registered ingress cells, proved as
  `NetworkIngressTieBucket` (one gate
  naming its pinned circuit digest, `io_request.circuit-check.json`) with the row read off the network certifier's root
  (`warden_rows.bucket_port`) and its lanes and output row off values the verifier registers at known content
  (`IngressTie.known`), decided by the verifier on its own lanes from the advice and the network certifier's
  release, and `IngressTie.tie`, which builds `IoTieVerdict` on the link and the stream's registrations from the
  stream's outcomes of record alone; `Ties`, the tie, its stream and the release as the verifier's own tie source,
  which gives the verdict only on its own link. A size class no sidecar pins has no accepted proof.

The certificate's signature scheme is a parameter: the certifier's
`certificate.Issuer` takes a `Signer`, and the audit a `PublicKey`.
The Boolean Definitions above that core doesn't import (`advice_check`, `cache_step_v2`, `cache_step_v3`, `compile_slice_v2`,
`compile_slice_v3`, `compile_slice_v4`, `compile_slice_v5`, `compile_slice_v6`, `compile_slice_v7`, `egress`, `ingress_scan`, `io_ids`, `lengths_row`, `response_tie`, `served_statics`, `served_tie`, `window_charge`) are in
`verity.protocols.certifier.prover`, stated over the ones here.
From `verity`, `advice_check`, `compile_slice` and `served_statics` take the circuits and their evaluator, `served_tie`
those and the hm96 leaves, and `charge_chain` and `window_charge` those and hm96, and from SPOT the produced rows.
`PROTOCOL.md` has the semantics and the correspondence to the Lean analysis (`security/`, area `NetworkCertifier`).

The coin link (`coin_link`, run by `verity/composite/device/live_challenger/coin_link.py`) carries a window's session
between the prover and the challenger: it re-cuts every frame into cells and re-clocks both directions on their own
grids, exempts only the verified, expected, once-only coin openings, and counts everything else
(`protocol/coin-link.md`).

~~~python
from verity.protocols.certifier.verifier.audit import Audit
from verity.protocols.certifier.verifier.grid import GridParams, Certifier
from verity.protocols.certifier.verifier.schedule import ClockSync, LogicalFrame, SyncSet, run_shaper

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
counts to the occupancy profile on every link's execution phase (on a served link, to the ones its ingress counts derive under
the declared sync), beside the proved timing bound (provisional).

N2's verifier, `LinkWindows.verify`, decides each execution phase on a `--zk` proof of N2's registered proof
block (`NetworkLinkWindow` v5, served `NetworkServedLinkWindow` v4) and public inputs alone: its own run of Flock's Lean
verifier accepts the proof, the proof's public row opens to its own copy of the proof block's public arguments, the
output row's verdict and deadline bits are 1, the certificates verify on their public fields (`LinkWindows.certified`),
and the window's sync is declared. It reads no head, salt or sibling. `benchmarks/network_traces/n2_zk.py` proves a
certifier-serve run's execution phases this way through the link's own stream, and scans what the verifier received. The
in-the-clear check, `LinkWindows.clear`, is a debug comparison, off by default. What's open:
- **The certificates aren't a Spec item.** The verifier receives them in the clear from the certifier, the auditor's
  device, beside the proof system, whose public kinds have none for a trusted device's signature over a hiding
  commitment. That is a provisional exception to the four public kinds, not a fifth kind.
- **A run that proves nothing** (a rehearsal, the simulator) is decided in the clear (`LinkWindows.unproved_outcome`,
  its verdict `"decided_by": "clear"`), which is never zero-knowledge.

`LinkWindows.verify`'s verdict of record on a plain, served or network certifier link is the Lean's, `NetworkCheck.LinkWindow`
(`verity/protocols/certifier/verifier/lean`, the `link-window` function of `network-records verdicts`), by the one switch line
`service.OF_RECORD`. `verify` runs both on every proof, and an execution phase where
they differ in the accept bit, the public outputs or the digest, or which the Lean refuses, is rejected. The switch is at
`lean` under the friction skill's switching ruling, on these grounds:
- the gap list is empty for plain and served links. Everything the Python checks, the Lean checks: the findings prove
  the registered Program and the sampled proof blocks with every gate pinned, the public row the verifier derives (root, served
  domain, link, window, each end's grid and profile row, both signed digests, a served link's sync words), both rows'
  openings and the word 0b11, both certificates under the LMS key in the certifier session, and the sync in Σ_sync. Both
  take as inputs what the proof system and the lowering give: the findings of the verifier's own Flock Lean run, the
  sessions' SHA-512s, the registration, the sample, and the Program's and partition's digests and the gate ids. The Lean
  reads an LMS key only, and rejects any other.
- on a network certifier's link (`service.WardenProfile`) the gap list is empty too: the Lean's form for it (`Kind.warden`) is the
  plain proof block's public row with both ends' grids at the network certifier's parameters digest, which both take as
  an input (the network certifier's deployment publishes it), and no sync read, as in the Python; it refuses a network
  certifier case whose digest isn't 64 bytes, whose ends have a profile, or whose two grids differ.
- every proving run of N2 recorded before the switch replayed through both to the verdict it recorded, negative
  controls included.
- the Python keeps running beside the Lean as the reference.
- the Lean's verdict has its property in `security/Properties/NetworkCertifier/Check/LinkWindow.lean`
  (`Properties.NetworkCertifier.LinkWindowAcceptMeans`, proved): an accepted execution phase was proved, opened to the
  word 0b11, certified on both ends and synced within Σ_sync (a network certifier link's syncs are not asked of), and
  its public outputs are verdict 1 and deadline 1.

`tie_service.IngressTie.verify`, the ingress tie's verifier, is not ported and stays Python.

A proof of the proof block is held to its circuit. Each gate names its Definition's circuit digest
(`Gate[NetworkHeadCheck_v2{…}@<digest>]_v2`, `served.gate` likewise), as `head_check.circuit-check.json` and
`served.circuit-check.json` pin it and circuit-check holds the pin to the Definition. So the proof block's Program,
which the spec and each registration bind, binds both circuits, and `LinkWindows.verify` refuses a proof that names any
other Program, or a proof block of a size class no sidecar pins (`LinkWindows.proved`).

The certificate signs no hash of witness. The certifier signs `D = SHA-512("verity/net-cert/v1" ‖
salt_c ‖ R)`, R the head's record digest and `salt_c` 64 bytes the certifier samples fresh for each record
(`certificate.Issuer`'s `coins`) and hands to the prover with the head, as witness only (`CertifiedRecord.salt`). D is
a hiding commitment to R, and the LMS check (`Certificate.verify`) reads only the certificate's public fields and D.
Each end's head check holds R to the head it opens and D to R under `salt_c`, so the certificates' check reads nothing
of the heads; the in-the-clear check also checks that link over the opening (`CertifiedRecord.verify`).

Tests: `uv run tools/verity/check/suites.py verity/protocols/certifier`.
