# Network attester (`verity.protocols.resource_accounting.network.verifier`)

## The network bound

The network bound limits what a covert sender in the serving node can tell a client through the network attester
beyond what the registered Program computes from its committed values: per execution phase, the probability that an
accepted audit leaves the certified traffic outside what that Program accepts, and over a monitoring period, the number
of bits of such a sender's secrets a decoder recovers from the client's view.

The guarantee is `NetTiming.Guarantees.MonitoringPeriodLeakAccepted`
(`security/Definitions/Certifier/Guarantees/MonitoringPeriod.lean`), proved as
`NetTiming.SecurityProofs.MonitoringPeriodLeakAccepted` (`security/Proofs/Certifier/MonitoringPeriod.lean`). Over a monitoring
period of n execution phases, execution phase i audited by the compiled sampled verifier at its
registered statement, with one network attester on the server's uplink running its code on schedules in `Ss`:

- per execution phase, at a statement in scope, `Pr[accept ∧ ¬RLeakCertifier(δ)] ≤ δ + ε_ks + ε_bind + Pr[B_tap]`;
- where the network attester registers its observations in every execution phase, the secrets whose every execution
  phase meets its good event (`ExecutionPhaseGood`) number at most
  `#Ss · ∏_c (cap_c + 1) · ∏_i 2^#(freeInputs_i \ reg_i) · #incorrectSets_i(δ) · 2^β_i · Dj^k_i`, whose log2 bounds the bits.

`RLeakCertifier(δ)` says every observation sequence the network attester registers is one the registered Program
accepts at the trace's committed values with some incorrect set the audit allows at δ. ε_ks and ε_bind are the sampled
verifier's errors (`Compiled.KS`'s) and `B_tap` a collision between the extracted trace's openings and the network
attester's. `cap_c` bounds connection c's cells; `freeInputs_i \ reg_i` are execution phase i's free inputs (the
advice) outside its registered inputs; `incorrectSets_i(δ)` are its possible incorrect sets at δ and β_i bounds the bits
any of them commits; k_i is its steps and `Dj` bounds each step's jitter class.

It bounds each execution phase's failure on its own: the union over the monitoring period's live coins is not stated,
and `ExecutionPhaseGood` adds a condition no proof gives, the registration's values at the prover's registered inputs. It is a
statement about the network attester's code (`recordingCode`), not about runs of the network check's Lean
(`lean`, `NetworkCheck`), which is the verifier's code and states no bound.
`NetTiming.Guarantees.PooledLeakAccepted` (`security/Definitions/Certifier/Guarantees/PooledLeak.lean`) is the
pooled count, with the incorrect proof blocks paid for once over the monitoring period,
`#Ss · ∏_c (cap_c + 1) · (N + 1)^Fmax · 2^(b · Fmax) · ∏_i 2^#(freeInputs_i \ reg_i) · Dj^k_i` at `Fmax ≥ ln(1/δ)/q`
(N the period's proof blocks, b each one's committed bits, q the per-execution-phase detection probability), failing
with probability at most `δ + Σ_i e_i + Σ_i e'_i`. It is conditional on two per-execution-phase hypotheses not yet
discharged (`hesc`, the conditional miss probability; `hleak`, the leak relation at the witness's incorrect set), so it is not
yet the deployed system's bound over a monitoring period.

What `MonitoringPeriodLeakAccepted` takes, as its record in the lock (`security/lean-audit.json`, `guarantees`) lists it. Named
assumptions: none. A premise that is part of what it says (`domain`):

- `Claims.Sampling.CommitmentScheme.Binds`: the trace's commitment binds, a break giving an explicit collision.

Open premises (`open`):

- `Verity.Assumptions.TrustedDevice` (`trusted-device`): the network attester, the auditor's device on a host the developer
  doesn't control, runs its stated code.
- `Claims.SamplingGeneric.Compiled.KS`: the compiled sampled verifier is knowledge sound at its errors on the
  statements in scope.
- `NetTiming.ScopeAcc`: the statement is in the sampling scope and its boundary layout fits its Program over its steps.
- `NetTiming.CertifierRegisters`: the network attester's openings pass under the statement's roots, cover every
  egress and ingress gate, and hold its record's rows there; the antecedent of the period's count.

`PooledLeakAccepted`'s record lists one open premise, `Verity.Assumptions.TrustedDevice`.

## Interface

- **Python**: `verity.protocols.resource_accounting.network.verifier`. Code outside this directory imports every module but `boundary`
  and `pacer`. Its importers: the prover side, `verity/protocols/resource_accounting/network/prover`; `../attester`'s vector
  scripts; `adapters/vllm` (`advice`, `arith_encode`, `blocks`, `charge_chain`, `head_check`, `id_form`, `io_tie`,
  `link_frame`, `network_attester`, `release`);
  `verity/protocols/compose/verifier/live_challenger`; `verity run`'s served table (`tools/verity/cli`); the tests of
  `../../compute/pouw/verifier` and `tools/verity`; and in `benchmarks/`, `network_traces`, `network_e2e`, `prove_path`,
  `one_stage`, `pouw`, `remote_session`, `mempool` and `compile_cost`. In this directory, the network check's tests
  (`network/tests`) and vector script (`lean/scripts`) import it too. `../../../proofs/spot`, `../../../proofs/flock`
  and `../../../proofs/recursion` import none.
- **Lean here**: `lean/` is the Lake package `network_check`, whose library `NetworkCheck` the
  packages at `../../../compose/verifier/onsite` and `security/Properties` require by path. Its executables:
  - `network-check`, `network-records` and `network-period`, each `id | check PARAMS COMMITMENT EVIDENCE | serve`: the
    core protocols `verity/network-certificate/v4`, `verity/network-records/v3` and `verity/certifier/v0`. The public
    tree has no `network-period`: it runs the period check, which reads the inference-only policy's layout check, which
    is withheld. The Python each is held to (`network/tests`) is `audit.Audit` for the first two and the mirror in
    `network/tests/period_case.py` for the third;
  - `network-rows`, with the same three verbs: the core protocol `verity/network-rows/v1` (`NetworkCheck.RowsAudit`),
    an active network attester's run, each record held to its certified head and its certificate to the key in the
    session, then the network attester's program's rows audit (below). The Python it is held to is `network_attester.py`'s `audit`.
    `benchmarks/network_traces/certifier_piece.py` runs it in step 1 of each of its steps, on the kept network
    attester run (`network_attester_chain.rows_audit_case`);
  - `network-records verdicts`, the process `lean_audit` starts and `service` asks (`VERITY_NETWORK_RECORDS_EXE` names
    a built one); `benchmarks/network_e2e` and `benchmarks/network_traces` build and run it;
  - `network-check-difftest INSTANCES PYTHON N`, which holds `certificate.py`'s encodings to
    `NetworkCheck.Certificate`'s.
- **Lean beside it**: `../attester/program/lean/`, the network attester's program, is the Lake package `certifier_program`, whose
  library `NetworkCertifier` the packages at `lean` and `security/Proofs` require by path. Its
  executables (`lake exe`): `certifier-difftest`, `certifier-charge-difftest` and `certifier-window-difftest`, each
  `INSTANCES PYTHON N`. The Lean check runs each `../attester/program/lean/scripts` script, then its difftest
  (`../attester/program/lean/lean-audit.json`). Its `NetworkCertifier.Rows` is the active network attester's records
  as rows and their audit, the code form of `RecordRows` over the network attester's lanes with `network_attester.py`'s `audit`
  its reference, which `network-rows` (`lean/`) runs on a run's certified records; no proof ties it to `RecordRows`.
  Against the reference it refuses three kinds of run the network attester never writes and holds two things less:
  no egress cell to the id egress form (the io tie's, `IoTieCell` on every cell) and no response to a `max_tokens_cap`
  (`network-rows` refuses such a schedule).
- **Documents others read**: the answer of `network-period check`, `verity/certifier/v0`. `benchmarks/network_traces`
  runs the check and writes it as network certification's piece (`certifier_piece.py`), which `benchmarks/one_stage` reads. It also writes the
  check's parameters, whose `"keys"` list the link's network attester's entries, each a key, its session and the
  start of its stretch of execution phases (`../prover/scan_period.py`: the deployment's one key and session, from execution
  phase 0), and its evidence, whose `"starts"` show each listed entry's session's signed start (the deployment's
  `start`).
- **Files others load by path**: `tests/scan_traffic.py` (`benchmarks/network_traces`, `benchmarks/prove_path`, the
  tests of `adapters/vllm`); `tests/egress_traffic.py` (`benchmarks/network_traces`); in `tests/fixtures`,
  `compile_slice_smallkv.json` (`benchmarks/network_traces`, `benchmarks/prove_path`) and the vectors
  `benchmarks/network_e2e/encodings` writes; `advice.py` (`benchmarks/network_e2e/encodings`); and in `network/tests`,
  `period_case.py` (`benchmarks/network_traces`), and `test_network_period_protocol.py` and
  `test_inference_only_protocol.py` (the tests of `../../../compose/verifier/lean`).
- **Loads from**: `lean/`, six Lake packages by path: `certifier_program`, `flock_verifier` (`../../../proofs/flock/verifier`),
  `inference_only_check` (`../../../workload_classification/training_detection/inference_only/verifier/lean`), `core_protocol` (`../../../interface`), and `lms` and
  `core_identity` (`verity/commitments/lms/lean`, `verity/commitments/identity`); the network attester's program, no package. The
  modules, `verity.circuits`, `verity.commitments`, `verity.commitments.crypto`, `verity.protocols.proofs.spot.verifier` and,
  `parties` alone, `verity.protocol` and `verity.channel`;
  never the prover side, which `../attester/program/lean/scripts/window_vectors.py` and the tests import.
- **Not here**: the network attester's host programs (`../attester`), and the programs that drive a run's proofs and
  chains (`benchmarks/network_traces`).

## What this package holds

A spec-level reference for network certification's timing channel:
- the constant-rate, left-packed network attester;
- the Program's logical schedule, with its clock-sync advice (δ, ρ);
- the exact audit with record-and-reject, on raw records, records committed by hash (padding elided), or network
  certificates (`certificate`: the network attester's signature over the record's SHA-512 digest hidden under a
  fresh salt, with the execution phase's frames in a fixed-depth SHA-512 tree), and the ingress links' status check;
- one execution phase's certificates through the proof system (`service`): the network `Spec`, a salted root over the
  certified records, and the check, whose Boolean part is each head's check (`head_check`: the head's hiding opening,
  its fields, and its record digest under the network attester's salt against the digest its certificate signs). A link is
  the network attester's (`service.NetworkCertifierProfile`), and its claim is the heads' only: on every execution phase both
  are the network attester's, on its grid under its parameters digest, so the link's timing is its schedule's.
  Nothing here holds the frames, the rest of the egress grid (the scan does). The network verdict of record is
  `TransferBlocks.network_verdict`, the AND over the link's execution phases with no ε or share credit: every execution phase of the
  link registered in order with an accepted outcome, and a rejected, late, missing or out-of-order one
  rejects the link;
- one vLLM v1 scheduler step, in vLLM's compiler (`verity_vllm.compiler.replay_step`): `ReplayStep_v1`, a Boolean
  Definition with `reference` its spec, takes the committed queue state, the certified arrivals and the committed
  decision bits to the step's rows (request, kind, start, n) and the next state, under sync or async scheduling and with
  `CacheStep_v1`'s walk entries, and refuses what its size class or the prefix layer's counts can't express. It is the
  compile slice's proof block's core, between the committed decisions and `compile_steps`' Programs, and is held step
  for step to the replay (`benchmarks/network_e2e/encodings/replay_vectors.py`); it is in no proof-system Spec yet;
- the I/O tie (`io_tie`), a design (`PROTOCOL.md`, "The I/O tie") and its prototype on a fake upstream, in no Spec:
  served through the network attester's egress, every cell of an egress execution phase is opened, each a pad, the
  canonical event of one token id or the response's end on the stream the network attester set, and each is the cell
  the advice's map names at its place (`window_map`, from the advice's outputs, release buckets and advice v1's order
  permutation, each request on its ingress ordinal's stream from 1; the verifier
  computes it in the clear from the opened advice, the named gap, until the map is read in the proof under the advice's
  hiding root), with the id the Commit sampled for it (the port: one hiding row of ids a step, read under the Commit's
  root). The cells are the network attester's (`network_attester`): a connection's r slots a bucket, a `PAD` on its stream before
  its first cell and after its `END`, one id event a `DATA` cell. The network attester commits each bucket's cells as
  one hiding row. It has the reference and the proof block's Boolean Definition, one a bucket row (`IoTieBucket`: the
  row and the port's rows read as registered rows, the network attester's and the Commit's (`port_reads`), then
  `IoTieContent`: `IoTieCell` on every cell, the prover's Beneš routing of the port's entries `IoTieRoute`,
  `IoTieMatch`);
- the ingress tie (`io_request`, `PROTOCOL.md`, "The ingress tie") on the network attester's own ingress (`serve`): one
  proof block a registered bucket row (`network_attester_rows.bucket_port`), each live connection's lane of r slots read as the
  network attester's cells (a `PAD` on its ordinal, a `DATA` of exactly B − 9 bytes, its `END`), each request cell a
  cell of CERT-VARLEN's id
  request, with its count of prompt fields carried across buckets from the lane's carry-in to its carry-out, which
  hold each request's prompt_len to the advice's and which the ingress tie's verdict (`tie_service.IoTieVerdict`) is
  built from. It has the reference, the lanes the verifier computes from the advice and the network attester's release
  (`lanes`, `Release`), and the proof block's Boolean Definition (`IoTieRequest`: one `IoTieRequestFrame` and
  `IoTieRequestStep` a cell, `IoTieRequestEnd` a lane); the ingress-ids tie (`io_ids`, `IoTieRequestIds`) adds the
  Commit's `c00.input/ids` rows to the same scan and holds every prompt id to them;
- the capacity counts;
- the egress side of the step-indexed scan (`egress`, N2) on a fake upstream, in no Spec: one proof block a link and
  bucket t (`EgressBucket`) that reads the scan's slot state S_t, a registered read, and computes the bucket's row of
  cells from it with every switch computed and none witness (the generator form): its outputs are a u16, `ACCEPT`
  exactly when the state renders a row, the one public output, and the row, committed as a registered read of the
  network attester's egress row t (`network_attester_rows.bucket_port`), which opens only if the computed row is the network
  attester's. It has the reference (`bucket_map`, `computed`, `rendered`, `outputs`) and the state's egress-view layout
  (`Shape`'s offsets), which the scan's ingress step writes;
- the lengths row's proof block (`verity.protocols.resource_accounting.network.prover.lengths_row`, `LengthsRow`), in no Spec: one a link,
  execution phase and slot stride (`PROTOCOL.md`, "The lengths row's proof block");
- the response tie's two proof blocks (`response_tie`, beside it), in no Spec;
- advice v0 (`advice`): what vLLM's scheduler decided in a session (each request's lengths and finish, each step's
  rows with their chunks, prefix-cache hits and the KV they reuse, resumes, and each kept output's egress bucket) as one
  canonical bit string the compiler reads, refused unless it is a schedule vLLM could have run, with its bits split by
  the earliest execution phase whose egress can show each record (`bits`). v0 is the committed form; the charge is ENC-CODE's
  compressed length of the advice;
- advice v0's validity, in vLLM's compiler (`verity_vllm.compiler.advice_check`): `AdviceCheck_v1{L}`, the compile
  slice's proof block's first part as a Boolean Definition, 1 exactly when the committed bits (at most L) decode to a v0
  advice, up to the three refusals of `decode` it defers (unique ids, a source's KV computed there, release order per
  request); its AND count at each L is `size(L)`;
- the replay's state chain, in vLLM's compiler (`verity_vllm.compiler.replay_chain`): ReplayStep_v1's state handed from
  one step proof block to the next as the state rows of a produced value an execution phase, with private heads into the
  execution-phase row (`window_fields`) every step reads: the certified arrivals not yet taken and the execution phase's decisions, since the step's cursors are
  private. `ReplayAdvance` sits beside each step and checks its arrivals and decision bits are the execution-phase row's at the
  heads; `ReplayLoad` runs once an execution phase and carries the decisions the last one left unread; `ReplayClose` ends the
  session with none unread. The execution phases join by equal openings and the session starts from the empty rows under public
  salts (`check_chain`). Its cost at a shape is `size`;
- the compile slice, in vLLM's compiler (`verity_vllm.compiler.compile_slice`): `CompileSlice_v1`, a Boolean Definition
  with `check` its reference, composes `ReplayLoad`, `ReplayStep_v1` and `ReplayAdvance` on shared wires,
  `CacheStep_v1`'s calls on the certified tokens checking the hits `ReplayStep_v1` reads, and `ChargeLink_v1` over the
  decisions the execution phase commits, from its carry-in to its carry-out, for one execution phase. It holds the charge to the declared
  one and the step rows to its committed ``srv`` row, the served rows. The evictable and returned counts are a stand-in
  held at 0 (`protocol/compile-slice.md`);
- the compile slice over the memory-checked pool, in vLLM's compiler (`verity_vllm.compiler.compile_slice_v4`):
  `CompileSlice_v4` computes its schedule rather than checking it. From the advice (DA decisions and the step count),
  the arrivals' ingress fields and their registered output lengths and tokens, it computes the served rows and the
  state, execution-phase and slot rows the execution phase leaves, each hit, evictable count and return where `ReplayStep_v1` reads it,
  and holds them to the registered ``srv`` and ``cout`` it reads; its one output is the 16-bit verdict. Its pool calls,
  in vLLM's order, are records on a bus to `CacheStep_v3`'s stream, which runs in the proof block from the empty pool with
  request keys the arrivals' `BlockKey_v1` chains printed under a live coin, so eviction, preemption and a leaving
  request's returns are in the slice. The hits, evictable counts and returns the steps read, and the stream's slots and
  touched list, are checked witness, unique and bound by live coins. Until a Lean lemma shows that uniqueness, the
  witness counts as free input at its width, so the free input is the advice and the checked witness (`free_width`). An
  execution phase costs its stream's slots, whatever the pool's size. `benchmarks/network_e2e/encodings/compile_slice_pool.py`
  (`check_trace_v4`) holds its calls to the advice decoder's block pool,
  `verity_vllm.pipeline.advice_code.pool.BlockPool`, over whole traces;
- the compile slice at its execution phase's activity, in vLLM's compiler (`verity_vllm.compiler.compile_slice_v5`):
  `CompileSlice_v5`, `CompileSlice_v4` with the arrivals' tokens registered in ``req`` as one ragged row of the NRK
  blocks the execution phase keys (a keyed block count a slot, and one key a block), RB's touched list the NRB addresses the
  stream touches, and each running slot's request number carried in the slot row, which ``cout`` registers, rather than
  looked up at every call. Its free input is the advice and the checked witness, as v4's (`free_width`), and its verdict
  adds bit 6 ``keys``. `parts` prices it at any class;
- the compile slice with the block pool carried between execution phases, in vLLM's compiler
  (`verity_vllm.compiler.compile_slice_v6`): `CompileSlice_v5` cut in two proof blocks, both proved every execution phase.
  `CompileSlice_v6` is its load and steps from the registered results, their pool calls' bus records held to the
  registered calls row. `PoolWindow_v1` opens the pool row under the input root, the previous execution phase's output root
  registered before its coins, as the memory check's initial values; runs `CacheStep_v3`'s stream from it over
  the calls row with v5's ragged keys; and commits the row the stream leaves as the output root. An execution phase from the empty
  pool's root (`genesis`) is v5's, and a chain restarts from it after a failed one. The compile slice's free input is
  the advice, `PoolWindow_v1`'s its checked rows (the left row, the blocks' sources and the stream's witness) at their
  widths until their lemma (`free_width`). A request still open at the execution phase's edge is cut (its registered ``a_cut``
  bit): `CompileEdge` frees every running slot in vLLM's preemption order and holds each open request to having emitted
  its ``a_O``, and the next execution phase takes its continuation, (id, P + e) with its prompt then its released outputs, as an
  ordinary arrival that hits its blocks in the carried pool. The close gives each keyed block's page its source, the
  execution phase's own write or the previous one's carried port, and a `PageCopy_v1` copy block holds the execution phase's carried
  port to it. `PoolChain_v1` is two execution phases so chained, and `parts` prices both proof blocks at any class, which
  `benchmarks/mempool/cost.py` does at a served load's;
- the compile slice charging the length terms, in vLLM's compiler (`verity_vllm.compiler.compile_slice_v7`):
  `CompileSlice_v7`, `CompileSlice_v1` with `ChargeLink_v2{D, 0, 0, K}` in place of `ChargeLink_v1{D}`, a length slot
  for each certified arrival read from ``cert`` (its output length and its certified max_tokens), so an execution phase's charge
  covers `code_v1`'s segment and its length tail. vLLM's served execution phases name it (`serve_commit.SERVED_SLICE`), except
  under `--admit tick`, and `benchmarks/network_traces/compile_slice_zk.py` proves it in zero knowledge (and plain,
  ``--plain``);
- the served rows, in vLLM's compiler (`verity_vllm.compiler.served_rows`): the step rows a served execution phase ran, from the
  server's kept rows, committed before the sample under the run root as a hiding-commitment registration, a row each
  block of G steps (G = P unless given: an execution phase's row, the slice's ``srv`` row; at G < P block j is the slice's
  ``srv{j}``, `verity_vllm.compiler.compile_slice.CompileSliceRowGroups_v1`). `leaf_ok` holds a proof block's row b‖c to
  its registered leaf, as the advice's ``adv`` row is held, and `reads_port` is the same read as the Lean verifier's
  registered reads take it;
- the served rows' tie, in vLLM's compiler (`verity_vllm.compiler.served_tie`): `ServedRowsTie_v2`, a Boolean Definition
  with `check` its reference, holds an execution phase's served rows to the positions its steps' Commit holds (the step-input
  port, `port_chunks`: each step's ids and then its positions at the head of its stream), so the rows registered at
  serve time are the batch the server ran. Every proof block is proved, and its port, selection and cost are its
  docstring;
- the served rows held to their Programs, in vLLM's compiler (`verity_vllm.compiler.served_statics`):
  `ServedStaticsTie_v1`, a Boolean Definition with `check` its reference, holds each block's registered served rows,
  read through the Lean verifier's registered reads, to the RoPE statics of the Programs its steps ran, with no row
  hashed but the block's ([protocol/served-statics.md](protocol/served-statics.md));
- the prefix cache, in vLLM's compiler (`verity_vllm.compiler.cache_step`): `CacheStep_v1`, a Boolean Definition of one
  scheduled row's pass over a table of whole cached units (an engine step's calls are `scan_step` and, as a circuit,
  `build_scan_step`), which checks the row's hit (the longest cached prefix while no block can have been evicted, a
  cached prefix past that) and owns or appends the units the row computes (`PROTOCOL.md`, "Advice v1");
- the block pool as memory-checked RAM, in vLLM's compiler (`verity_vllm.compiler.cache_step_v3`): `CacheStep_v3`, an
  execution phase's pool calls as one stream of NS access slots over the five memories vLLM's `BlockPool` is made of (blocks,
  keys, key copies, request blocks, request keys), checked by `MemCheck_v1`'s fingerprint under live coins, with a bus
  product of the calls' records for `ReplayStep_v1`. So a lookup's hit and evictable count and a free's returns are
  derived past the cumulative pool bound too. An execution phase's price is NS slots and NE list entries from a public ladder,
  whatever the pool's size; `Stream` is its specification, held to the replay's pool by `benchmarks/mempool/stream.py`;
- advice v1's arithmetic encoder (`arith_encode`), Boolean Definitions each with its reference as spec. The counting
  variant, which the composition takes: `ArithCountStep_v1{K, BIN, CW}` runs ENC-CODE's 32-bit coder forward on one
  symbol and counts the bits it puts in the execution phase's segment, `ArithCountFinish_v1{CW}` adds the termination and
  holds the execution phase to its declared charge, and `BookKT_v1{C}` is `code_v1`'s KT estimate for a bit's context. `ChargeCount_v1{N}` composes them into one execution phase's charge from its decisions, the compile
  slice's proof block's one call an execution phase. The context and the frequencies are the circuit's, from the replay's position and the
  registered book's counts, never witness. The bit-checking variant, the alternative: `ArithEncodeStep_v1{K, BIN}` and
  `ArithEncodeFinish_v1` check each output bit against a committed segment in the decoder's order, with
  `KTContext_v1{C, NW}` ENC-REPLAY's scaled estimate. It is held symbol for symbol to `code.py` and `code_v1` on advice
  v1's execution phases and on ENC-DERIVED's still-sent fields (`benchmarks/network_e2e/encodings/arith_vectors.py`), and it is
  in no proof-system Spec yet (`PROTOCOL.md`, "Advice v1");
- the charge's carry chain (`charge_chain`): `ChargeCount_v1`'s carry committed from execution phase to execution phase as hiding commitments,
  execution phase 0's the book's priors under a public salt, with the verifier's check of how they join (`check_chain`)
  and `ChargeLink_v1{N}`, the charge from the carry-in row held to the carry-out row (`protocol/charge-chain.md`);
- the whole execution phase's charge (`window_charge`): `ChargeCount_v2{N, H, O, L}` and `ChargeLink_v2{N, H, O, L}` charge
  advice v1's execution phase (`Window`, the record `code_v1` codes) with its hits and order entries and their counts, coded as
  `code_v1` codes them, and its length terms' tail, with the order's position and rank added to the carry and chained
  under the value
  `window-charge` (`protocol/charge-chain.md`, "The whole window");
- the active network attester (`network_attester`): a relay between any client and any server. It forwards each direction's
  bytes unchanged, in B-byte cells, r a bucket each way per connection, and egress cell i goes out in bucket
  s + D + ⌊i/r⌋ from its admission s. In the id egress form, its default, it releases a server's cell only if
  `io_tie.parse_cell` reads it ("The egress record"), and fails the connection (`format`) at any other, never
  rewriting it. It records and certifies exactly the cells it sends, and each failure's reason in a third record,
  `fails`.
  `PROTOCOL.md`'s "The active network attester" has its leak bound, and `verity/protocols/resource_accounting/network/attester/network_attester_serve.py`
  runs it;
- the release order on the server's side (`release`): a network attester puts `x-verity-release: N` (N is the request's
  0-based place in its run's release order) and `x-verity-release-run: ID` on the server's copy of each request it
  releases, for `serve-openai --admit release-order`. The network attester adds `x-verity-release-bucket: a`, the
  bucket of the request's `END`. With `NetworkCertifier(watermark=…)` it also sends the server a watermark stream:
  one chunked `POST /verity/watermark` carrying one line `t n l…` a tick, meaning bucket t has closed, the releases of
  buckets ≤ t are exactly N < n, and each l is one of bucket t's that its server connection lost. Line t goes out once
  each release of bucket t is written or lost. An engine that admits at a logical time in network attester buckets waits on
  that line and on each N < n it doesn't name. A client's request that opens as one to that path is refused
  (`release.to_watermark`, read on its first cell and, whole, at its `END`), unreleased; so is one a server would
  read a second request in, or read without the headers (`release.one_request`, read whole at its `END`).
  The headers and the stream never appear in a cell or record, so nothing certified changes;
- the network attester's cells as rows (`network_attester_rows`): each execution phase's cells committed one `hm96-sha512/row/v2`
  row a bucket, padding and empty slots included, under a frame-v3-sha512 root the network attester signs with its
  signer, beside a salted digest of the execution phase's wiring. The registration is what a proof block's
  `--registered` port reads, and the verifier places each read, the (connection, cell) the Program's proof blocks'
  wiring pins (`Units`, whose digest the port's value names), with the network attester's wiring (`PROTOCOL.md`,
  "The network attester's cells as rows"; `verity/protocols/resource_accounting/network/prover/network_attester_zk.py`);
- the network attester's connection rows (`connection_rows`), in a schedule with lengths rows
  (`NetworkCertifierParams.lengths`): each execution phase's releases as two tables a row a slot stride, `lengths` and `fails`,
  and, with content rows (`NetworkCertifierParams.content`), three tables a class of what each filed release carried
  (`content_rows`: its request, response and end rows; under form 2 a class's held rows alone, whose count the head
  makes public; with `request_words`, a fourth of its request's words), all committed as the cells' rows are and listed under one
  head the network attester signs once; the registration the verifier reads (`check_registration`) and its
  `--registered` port of the proof blocks that read a table, one a row (`port`); under form 2, the signed word
  that an execution phase has no head (`check_no_head`);
- how an execution phase's class is charged (`class_charge.charge`): below its schedule only with a count above 0
  in that head, its four verdicts, each accepted, complete and on the head, all of one serve and registration,
  and the head's schedule handed, one that can be tied; else its whole schedule;
- its deployment manifest (`parties`): no party of it runs under the runtime;
- the output value the network attester relayed as a registered value (`out_commit`): `commit_out` commits its rows at the
  reading program's registered shape (`OutShape`: its program, the value, ``words`` a row and ``rows``, both sides
  taking them from the circuit's registered read) under the domain the verifier derives for that pinned value, with the
  network attester's fresh salts, and signs the head for the egress execution phase; `check_out` reads the public record for
  the execution phase it expects at the verifier's own shape and returns the root to pin, or refuses by name
  (`REFUSALS`: another width is ``out-width``, another row count ``out-rows``), so the output proof blocks' rows are
  registered reads of the network attester's root rather than a comparison in the clear;
- the run's boundary in an execution phase's circuit (`boundary`): the circuit record's ``boundary``, which names the template
  whose proof block ℓ·T + t holds bucket t of egress link ℓ (`EgressBucket`'s frame, the row it computes and returns,
  committed as a registered read of the network attester's row, and its u16 verdict, whose bit 0 is the slots' presence) and of
  ingress link ℓ (its frame parameter row, the network attester's registered row), the same gates whatever the load, so arrival steps and prompt lengths stay private (`NetTiming.BoundaryGates`); its
  format is `verity/boundary/v3`. `refusal`, the verifier's check before it samples, checks that the layout fits the
  circuit (`NetTiming.BoundaryGates.Fits`: no egress gate is an input, and each is one the circuit returns) at the record's
  grain, which contains the gate grain's, held to `prog_fits`, `BoundaryGates.Fits` itself on a flattened program; the scan's
  state and view chains are wires of the record. A frame that is a parameter row, an input, is refused by name. `Fits`
  stays structural: a served execution phase fits in the step-indexed scan's form, and one in any other form is refused;
- the transfer rows (`transfer_rows`): a host's load, each value of the weights it loads once a session (the one kind,
  ``load``, ingress), committed as a registered value under the network attester's own salts at the reading port's
  leaf width and row format, its root in a head the network attester signs for the session's first execution phase
  (`TransferHead`, `registration`); every other row into the host is the scan's state, with no head of its own.
  `check_registration` is the verifier's read of it for the transfer it expects, and `port` the verifier's own
  `--registered` port of the value at the signed root (the load relayed and committed as it crosses,
  `verity/protocols/resource_accounting/network/attester/load_relay.py`);
- the class circuit's check (`blocks`): a circuit record is one host's class execution phase circuit, and which host runs which
  proof blocks, and which request occupies which slot and step, are private, placed by the scan inside the zero-knowledge
  proof; the class circuit and its padded counts (proof blocks per host, bits per link) are public, and its ``blocks``
  state no placement (`verity.commitments.periods.interface`'s `Blocks`). `check` refuses an ``ingress`` declaration, a registered read
  or table of no registered parameter, a load (every registered parameter but the scan's state) read
  in other than whole rows of one width, by no proof block, at an external port or by a table over part of it, and an external
  input port whose rows aren't each read exactly once from the scan's state (`blocks.SCAN_STATE`), the host's one
  ingress. `pins` gives the verifier's pinned roots of the loads from their certificates alone (`certified`: each
  load's registration on the host's uplink, ingress, in the session's first execution phase, kind ``load``), so the
  interface's ``input_count`` under them and the scan's state's root has no free input;
- the scan's ingress side (`ingress_scan`): `IngressScanBucket`, one proof block a network attester bucket, reads the bucket's
  network attester's ingress row, the slot state the bucket before carries and the Programs' entries, one port `prog<j>` a slot,
  each `SlotEntry_v1`'s word-aligned row of its slot (registered reads), and computes the next state (each block to the
  slot holding its connection, a new connection to the lowest free slot, its prompt ids appended, a due, occupied
  slot's output count and ids taken from its entry at its anchor, every other entry ignored, a slot held through its
  last egress bucket, `admitted + D + ⌊n/r⌋`) and its egress view, the state as `egress.EgressBucket` reads it (bucket,
  then per slot live, q, admitted, n and the output ids), its outputs beside a public ok bit, which is 0 at an
  occupied slot whose entry of the Programs' row isn't read (its read bit 0: a request no Program served). Unless each
  lane's entries are a compute audit's pinned `SlotEntry_v1` or `SelectEntry_v1` template's (`entryPins`), the read bit
  is the prover's word (an entry of read 1 and no outputs passes), so the period check refuses such an execution phase
  (`NetworkCheck.Period`'s `wired`), one whose scan's answer names a tied Input that is no lane's `prog<j>`
  (`tiedInputWired`), one whose scan names a member's ok bits (`<member>/o0`) at another root than its ok rows' under
  public salts, a constant of T ≥ 2 (`scan_ok`), and one whose scan's state isn't that audit's `SCAN_STATE`, by root
  and domain (`stateTied`): the scan's state tree is the execution phase's state (`window_state`), each row but leaf 0's the one a
  proof block computes (its o1) and the next reads (its sin), both registered reads, so a Program reads the states the scan's
  steps computed;
  `IngressScanStart`, the scan's start, commits leaf 0 by gates: the state its execution phase's record names (the
  empty state in the run's first execution phase, the previous one's last state row after it), so every bucket, the
  first included, is `IngressScanBucket`; the period check holds that state to the empty state in the run's first
  execution phase and, in a period's first that is not the run's first, charges it once at the state's full width
  (`scanStartCheck`, the public's `start`). The period check, the
  step from a period's records to the network bound, reads the inference-only policy's layout check. Programs start at
  fixed slots a network timestep: a timestep is `EVERY` buckets, the network attester's `every`, which divides its
  execution phase, with `SLOTS` = lanes Program slots each, and only `EVERY` and the slot count are public. The token-id row a
  slot's Program reads is the scan's state row b − 1 (`verity/certifier/scan-state/v1`, `token_row`): its occupancy
  word `occ` (1 for a slot whose Program starts at b, 0 for an empty slot), its prompt's length `fields` and its prompt
  ids, which the scan reads off the network attester's signed ingress cells, so no other tie holds them to the certified bytes;
  `slot_row` gives each one's bits in the state row, for a reader that takes them as i32s (vllm's `ScanSlots_v2`), and
  `window_state` the rows an execution phase's circuit from bucket t0 reads, the scan's rows t0 − 1 on (the
  empty state's for row −1). A Program gives all its outputs to its entry at its anchor (its read bit, n, u32, and 64
  ids of 17 bits at w63's class, 71 words, the row `SlotEntry_v1` writes); an execution phase's
  bucket t reads slot j's entry at row j·nb + ⌊t / EVERY⌋ of one slot-entries input, and an empty slot's entry is
  ignored (its outputs masked by `occ`, never on the egress). A request the class can't hold is refused at the scan's
  proof block, never truncated: a prompt of more than `KI` ids (1,024 at w63's class) at its cell, a Program's n over `KO` at
  its anchor. An execution phase whose network attester refused a connection, traffic past its slots, has no certified record, and the
  period check refuses it. `Class.serves` names the network attester parameters it takes every run of.
  `verity/protocols/resource_accounting/network/prover/scan_zk.py` proves both sides of a group of execution phases in one registration: ingress proof block
  t's state and view outputs are committed as registered reads of the scan's state and view rows t, which ingress proof block
  t + 1 and egress proof block t read, and egress proof block t's row as a registered read of the network attester's egress row t, so the
  chains are wires through the registered roots;
- the ingress tie through the proof system (`tie_service`): `IngressTie`, each ingress execution phase's `IoTieRequest`
  proof blocks, one a bucket row of the network attester's registered ingress cells, proved as
  `NetworkIngressTieBucket` (one gate
  naming its pinned circuit digest, `io_request.circuit-check.json`) with the row read off the network attester's root
  (`network_attester_rows.bucket_port`) and its lanes and output row off values the verifier registers at known content
  (`IngressTie.known`), decided by the verifier on its own lanes from the advice and the network attester's
  release, and `IngressTie.tie`, which builds `IoTieVerdict` on the link and the stream's registrations from the
  stream's outcomes of record alone; `Ties`, the tie, its stream and the release as the verifier's own tie source,
  which gives the verdict only on its own link. A size class no sidecar pins has no accepted proof.

The certificate's signature scheme is a parameter: the network attester's
`certificate.Issuer` takes a `Signer`, and the audit a `PublicKey`.
The Boolean Definitions above that core doesn't import and that aren't vLLM's compiler's (`egress`, `ingress_scan`,
`io_ids`, `lengths_row`, `response_tie`, `window_charge`) are in `verity.protocols.resource_accounting.network.prover`, stated
over the ones here. From `verity`, `charge_chain` and `window_charge` take the circuits, their evaluator and hm96, and
from SPOT the produced rows. `PROTOCOL.md` has the semantics and the correspondence to the Lean analysis (`security/`,
area `Certifier`).

The coin link (`../attester/coin_link.py`, run by `verity/protocols/compose/verifier/live_challenger/coin_link.py`) carries an execution phase's session
between the developer and the challenger: it re-cuts every frame into cells and re-clocks both directions on their own
grids, exempts only the verified, expected, once-only coin openings, and counts everything else
(`protocol/coin-link.md`).

~~~python
from verity.protocols.resource_accounting.network.verifier.audit import Audit
from verity.protocols.resource_accounting.network.verifier.grid import GridParams, Certifier
from verity.protocols.resource_accounting.network.verifier.schedule import ClockSync, LogicalFrame, SyncSet, run_shaper

params = GridParams(buckets=600, allocation=1024, frame_bytes=4096)   # 1-min windows of 100 ms buckets
sync_set = SyncSet(tuple(range(16)), tuple(range(1, 9)))              # 7 bits per window
frames = [LogicalFrame(b"\0" * 4096, anchor=3, index=i) for i in range(5)]
syncs = [ClockSync(delta=2, rho=4)]
records = run_shaper(frames, ready=[4] * 5, syncs=syncs, certifier=Certifier("responses", params))
audit = Audit()
audit.check_link("responses", records, frames, syncs, sync_set, params)
assert audit.accepted
~~~

## Zero knowledge: what N2's verdict reads, and what's open

Every protocol is to be zero-knowledge. The network audit's claim is the heads' only: N2 holds both heads of every
execution phase to the network attester's grid and parameters digest, so each execution phase's per-bucket counts are its
schedule's, beside the proved timing bound (provisional).

N2's verifier, `TransferBlocks.verify`, decides each execution phase on a `--zk` proof of N2's registered proof
block (`NetworkTransferBlocks` v5) and public inputs alone: its own run of Flock's Lean
verifier accepts the proof, the proof's public row opens to its own copy of the proof block's public arguments, the
output row's verdict and deadline bits are 1, and the certificates verify on their public fields (`TransferBlocks.certified`).
It reads no head, salt or sibling. `verity/protocols/resource_accounting/network/prover/n2_zk.py` proves a
`../attester/network_attester_serve.py` run's execution phases this way through the link's own stream, and scans what the verifier received. The
in-the-clear check, `TransferBlocks.clear`, is a debug comparison, off by default. What's open:
- **The certificates aren't a Spec item.** The verifier receives them in the clear from the network attester, the auditor's
  device, beside the proof system, whose public kinds have none for a trusted device's signature over a hiding
  commitment. That is a provisional exception to the four public kinds, not a fifth kind.
- **A run that proves nothing** (a rehearsal, the simulator) is decided in the clear (`TransferBlocks.unproved_outcome`,
  its verdict `"decided_by": "clear"`), which is never zero-knowledge.

`TransferBlocks.verify`'s verdict of record is the Lean's, `NetworkCheck.TransferBlocks`
(`verity/protocols/resource_accounting/network/verifier/lean`, the `link-window` function of `network-records verdicts`), by the one switch line
`service.OF_RECORD`. `verify` runs both on every proof, and an execution phase where
they differ in the accept bit, the public outputs or the digest, or which the Lean refuses, is rejected. The switch is at
`lean` under the friction skill's switching ruling, on these grounds:
- the gap list is empty for the Lean's plain and served forms, which no Python link takes now. Everything the Python checks, the Lean checks: the findings prove
  the registered Program and the sampled proof blocks with every gate pinned, the public row the verifier derives (root, served
  domain, link, execution phase, each end's grid and profile row, both signed digests, a served link's sync words), both rows'
  openings and the word 0b11, both certificates under the LMS key in the network attester's session, and the sync in Σ_sync. Both
  take as inputs what the proof system and the lowering give: the findings of the verifier's own Flock Lean run, the
  sessions' SHA-512s, the registration, the sample, and the Program's and partition's digests and the gate ids. The Lean
  reads an LMS key only, and rejects any other.
- on a network attester's link (`service.NetworkCertifierProfile`) the gap list is empty too: the Lean's form for it (`Kind.networkCertifier`) is the
  plain proof block's public row with both ends' grids at the network attester's parameters digest, which both take as
  an input (the network attester's deployment publishes it), and no sync read, as in the Python; it refuses a network
  attester case whose digest isn't 64 bytes, whose ends have a profile, or whose two grids differ.
- every proving run of N2 recorded before the switch replayed through both to the verdict it recorded, negative
  controls included.
- the Python keeps running beside the Lean as the reference.
- the Lean's verdict has its property in `security/Properties/Certifier/Check/TransferBlocks.lean`
  (`Properties.Certifier.TransferBlocksAcceptMeans`, proved): an accepted execution phase was proved, opened to the
  word 0b11, certified on both ends and synced within Σ_sync (a network attester link's syncs are not asked of), and
  its public outputs are verdict 1 and deadline 1.

`tie_service.IngressTie.verify`, the ingress tie's verifier, is not ported and stays Python.

A proof of the proof block is held to its circuit. Each gate names its Definition's circuit digest
(`Gate[NetworkHeadCheck_v2{…}@<digest>]_v2`), as `head_check.circuit-check.json` pins it and circuit-check holds the
pin to the Definition. So the proof block's Program, which the spec and each registration bind, binds its circuit, and `TransferBlocks.verify` refuses a proof that names any
other Program, or a proof block of a size class no sidecar pins (`TransferBlocks.proved`).

The certificate signs no hash of witness. The network attester signs `D = SHA-512("verity/net-cert/v1" ‖
salt_c ‖ R)`, R the head's record digest and `salt_c` 64 bytes the network attester samples fresh for each record
(`certificate.Issuer`'s `coins`) and hands to the developer with the head, as witness only (`CertifiedRecord.salt`). D is
a hiding commitment to R, and the LMS check (`Certificate.verify`) reads only the certificate's public fields and D.
Each end's head check holds R to the head it opens and D to R under `salt_c`, so the certificates' check reads nothing
of the heads; the in-the-clear check also checks that link over the opening (`CertifiedRecord.verify`).

Tests: `uv run tools/check/suites.py verity/protocols/resource_accounting/network`.
