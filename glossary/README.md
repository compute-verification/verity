# Glossary: notation, older names and narrower senses

The rest of the [Glossary](../ARCHITECTURE.md#glossary), and it changes with it: a term that replaces an older one gets its row here in the same change.

## Notation

The datacenter model's symbols, one meaning each (Daniel, 26 Oct). The canonical text is the Notion working draft
"Distinguishing inference from frontier training", §3, §8 and its notation table, with
`note:training/20261026T1840Z-draft-formal-properties`. A gate count that credits or reports is in AND gates, and the
(c, k)-partitions count every gate ("Size, credit and exclusion", below).

- D, A: the declared circuit, registered once as R = (θ, D, 𝓘), θ its registered input gates and values and 𝓘 its
  (c,k)-partition into inference blocks; the actual circuit, placed: gate g at device x(g) and tick tick(g).
- A[X, J], Φ_X: the gates of A placed at location X (a set of devices) during the interval J; X's capacity in gates per
  second, |A[X, J]| ≤ Φ_X·|J|. Φ_s is a server's, Φ_total the fleet's.
- W_cert(X, J), U(X, J): the certified declared gate occurrences at X during J; the uncertified gates,
  |A[X, J]| − |A[X, J] ∩ W_cert(X, J)|.
- A_phys: layer 1, the prior: the placed circuits the inventory can run under the hardware model.
- A_i(S), δ_i(S): layer 2, the envelope of subprotocol i's application S, and its failure bound,
  Pr[accept ∧ A ∉ A_i(S)] ≤ δ_i(S).
- A′, Break: A_phys ∩ ⋂ A_i(S), which every accepted run lies in except with probability Σ_i Σ_S δ_i(S) + Pr[Break];
  Break is a forgery of the devices' logged coins.
- (c, k): a partition's gates per block, exactly, and unregistered input bits per block, at most.
- c_S, k_S, Δ_wipe: a serving block's padded size ⌊Φ_s·Δ_wipe⌋ and its input allowance k_wipe + (r_in + r_side)·Δ_wipe
  (what the wipe leaves of M_s, at most 4×10^10 bits plus egress, and certified ingress and side channels over the
  interval); the wipe interval.
- ε_dark, ε_undecl, ε_gap, ε_skip, ε_break: unaccounted gates. ε_dark is the dark leaves' share of the fleet's capacity,
  Φ(L_dark) = ε_dark·Φ_total, ruled at 10^-3 or 10^-2; the rest are shares of the inventoried capacity: ε_undecl and
  ε_gap are U at locations with no certified declared gates or only partly certified ones, with
  ε_gap := 1 − |W_cert|/(Φ_X·|J|); ε_skip is PoUW's ε_W and the wrong certificates; ε_break is the gates of the faulty
  proof blocks F. Each use states its denominator and counts an overlap once.
- k_exfil(Y), k_infil(Y): block Y's exfiltration capacity log2|O_Y|, O_Y the observable output behaviors (values, and
  timing, lengths and routing where an observer can tell them apart) Y can produce and still be accepted; the information
  reaching Y that its input accounting has not already charged.
- HW, INV: named assumptions: the hardware model (credited rates, clocks, silicon semantics, what a wipe clears, link and
  side-channel rates); inventory coverage, the bound on dark capacity and the honesty of certifiers, challengers and
  their hosts.
- δ_coin, δ_crypto: probability bounds from the live-coin draws, under uniform coins; advantage bounds for SHA-512's
  binding, knowledge soundness and pseudorandomness, against a stated adversary.
- ρ, T_target: the circuit-size slowdown |C|/T_0 against the reference circuit C_0 of T_0 gates; the time to the target,
  a separate conclusion, T_target ≥ L·T_0/Φ_total when every target-reaching circuit has at least L·T_0 gates.

## Size, credit and exclusion

Daniel (26 Oct, 6:40 PM PDT, tentative): credit and all reporting are in circuit size, counted in AND gates.

- **Size** is a circuit's or a region's AND-gate count. AND gates are what C-Flock pays for, so size is the lingua franca
  every protocol credits in and every result reports computation in, in place of work, FLOPs, MAC-times and pricing. U uncertified
  gates complete at most U/s_× counted operations, s_× the fewest gates of a counted operation in the same count; this
  replaces κ/2.
- **Credit** is a statement about values. An accepted run credits a size S, and its guarantee is
  Pr[accept ∧ the AND gates whose committed values are right number fewer than (1 − ε_skip)·S] ≤ δ_coin + δ_crypto. It
  needs no algorithmic premise: SAK (`SAKResidual`) is a premise of exclusion only. For the served PoUW, the
  one computational premise left is `SharedNoiseLucks` (the fold receipts' detection) until the no-fold receipt. The
  remote audit (`Accounting.Remote`) makes no exclusion claim, so its PoUW guarantee is the size credit alone.
- **Exclusion** bounds how much hidden work could run beside the credited work (layer 3). It is the one place the
  accelerator model 𝓜's resources appear (tensor-pipe time, issue slots, memory bandwidth), because gates can't measure
  a device's capacity, and its result converts back to AND gates for reporting.
- **What counts every gate.** The (c, k)-partitions (inference blocks, a serving block's c_S) and the Lean's counting
  results (`Circuit.size`) count every gate, so a report that reads one converts it explicitly (lean's proposal:
  `Circuit.andSize` beside `Circuit.size`).
- **Not yet restated.** The Lean's device-model audits still price work per op (`Accounting.Run.ComputeSound` at a
  `Price`, and the covert residual F·T − W_cert, which is exclusion on one device), and ACCEL's `PROTOCOL.md` credits at
  pub(μ°) against W_reg until CMP-158 restates it in AND gates.

Daniel's standing instruction goes with it: anyone who finds a reason this is wrong floats it the moment they find it,
on its #agent-coordination thread or to @compute-research.

## Workload

Daniel (26 Oct, 5:49 and 5:56 PM PDT): a **workload** is what a datacenter runs over a monitoring period: its Programs,
their inputs (weights, and requests or data) and their placement. It is one thing at every scale, from one server to a
whole datacenter, inference and training alike, and workloads compose: workloads run side by side are one workload.
"Row" no longer names one; it stays only for a commitment's matrix rows. vLLM's workload files
(`integrations/vllm/workloads/*.json`, `verity-vllm/workload/v2`) each state one workload with the runtime that serves it
(engine, target, execution).

## Older names

Older names stay where they are baked into a digest, a frozen fixture, a recorded artifact field or a pinned vector. They mean:

| Old name, where it stays | Term |
|---|---|
| "Candidate A", "Candidate B"; the `candidate` feature and label keys, `CandidateA`, `CandidateB`, `account_candidate_a` | the archived proof systems A-GKR, B-Ligero |
| the `candidate` label values `SP1` and `C` (`research.store.vocab.CANDIDATES`) | D-SP1, and the VOLE candidate (not C-Flock) |
| `verity.ml.circuits.targets.target.Target`, `TARGETS`, `FIRST`; the result fingerprint's `profile` | subcircuit (a `GemmCoordinate<K>` at one semantics) |
| `Target.family`, `FAMILY_DIGEST`, a gate family such as `veritor.tc-ampere-bf16@1` | the semantics parameter's gate set |
| "backend", "backend family", "configured backend" for C-Flock; `backends/` (C-Flock's directory before the platform layout), `verity_flock.backend`, `BACKENDS` | the proof system |
| "IR", "Verity IR", "IR primitives", "IR evaluator"; `verity.ir` | circuit, Program; `verity.ml.circuits` |
| "instance", "instance set"; the `bench-instances/*` datasets and `fixtures/bench-instances/`, the fingerprint's `instances` (`contract.instances_ref`) and its tier (`vu-k1536`, …), the `instance-equiv/v1` and `vllm-vu-set/v1` kinds, the `instances_dataset` / `instances_tier` labels, `instances_per_second` | input, input set |
| "configuration" meaning a backend with a scheme; `drilldown.Variant`, `VARIANTS`; `views.Configuration`, `CONFIGURATIONS` | a proof system and a statement |
| `verity_sp1.proofs.statement.Statement` (statement formats 2 and 3 on the wire), its `KindProgram` and "kind" | the SP1-era wire encoding of a batch of proof-block obligations, not a statement in this sense; a kind is a subcircuit |
| "pin", "pinned theorem", "pinned statement"; `pins` in a `lean-audit.json` written before the rename (read as `guarantees`, rewritten by its next `--update`) and in `tools/verity/lean/Facts.lean`'s facts | guarantee |
| "law", "draw law"; `law` identifiers and record keys (`parse_law`, `same_law`, `law_object`, a registration's `law`), the Lean's `Law` | sampling design |
| "escape", "escape probability"; the Lean's `*_escape` theorems and `effEscape` | acceptance probability (P_a; `effEscape` is P̄_a) |
| "proof unit", and "unit" alone meaning one; `unitClasses`, `verity.ml.circuits.parts`, the audit record's `wrong_units` and `proved_units`, `LinkWindows.proof_unit`, the `"proof-unit"` kind of C-Flock's Boolean export, a `Spec`'s public-item reasons (its digest hashes them), and the `vu_*` names (below) | proof block |
| "wrong unit", "incorrect unit", "faulty unit"; the Lean's `wrong`, the audit record's `wrong_units` | faulty proof block |
| "transcript" for the values of a run's gates; `ExecutionTranscript` | trace |
| "δ_link", "the link term"; the Lean's `LinkSound` and `δlink` | ε_bind, the binding error |
| "verification unit" (VU); `vu_*` identifiers, the `verity/sampled-proofs/vu-sample/v1` domain, the `vllm-vu-set/v1` kind, the `vu-k1536` tier | proof block |
| "M0"; its first statement id, baked into the Lean verifier's `Flock/Tags.lean` | C-Flock's circuit prover: module `verity_flock.circuit`, CLI `flock-circuit`, statement `verity/flock-circuit` |
| "M1", "M2" (elsewhere in the code these also name unrelated things) | C-Flock ZK |
| `hm96-sha256/v1`, until one-hash deletes it | the hiding leaf `hm96-sha512/v1` |
| "coin server"; the `round_digest` strings of statement identities in `Flock/Tags.lean` | the challenger's live-coin part |
| "network transparency"; the `network-transparency/` paths in network accounting's project store that the `NetworkCertifier` Lean cites | network accounting |
| "warden", "network warden", "tap", "the auditor's tap", "uplink tap", "warden/certifier"; the domain tags `verity-network-warden/row/v1` and `verity-network-warden/record/v1`, the store label value `replay_set=warden/re-verify` | the network certifier; "the warden's records" are network certificates |
| "timed verifier"; the recorded config key `verifier_cpu` (`--verifier-cpu`) | the memory challenger |
| "prover gateway", "gateway"; one recorded statement string of recursive sessions | the firewall |
| "verifier gateway" | the challenger |
| "batch verifier" (outside `archive/`, where B-Ligero's is another thing) | the verifier |
| "GPU workers" | workers |
| "netlist"; C-Flock's `verity/flock-netlist/…` tags and domains and the `flock-netlist/v1` file format of its older statements (`netlist-vectors.patch`) | a private circuit, or the circuit's description |
| "trusted text", in the `layers` reasons of `lean-audit.json` and the module docs of the `Protocol.lean` files | spec |
| "horizon" and "accounting period" for the stretch of time a bound is about; the Lean's `*Horizon*` and `*Period*` names (`HorizonSampling`, `RateHorizon`, `ComposePeriod`, `CircuitWorkPeriod`), the remote bandwidth ledger's `horizon` field, until their owners rename them | monitoring period (`horizon` is its pooled reading) |
| "block" for a part of the device graph; `datacenter.physical.Block`, `block_of`, `network_sim`'s `--blocks` flag | zone |
| "block", "thread block" for CUDA's unit of a launch; CUDA's `blockIdx`, `blockDim` | CTA |
| "block" for a kernel's tile of a tensor; the Definitions `AttnBlock_v*`, `AttnBlockSoftcap_v*`, `AttnBlockFA3_v1`, `SparseMlaBlock_v1`, `TritonBlockRMSNorm_v1`, `MoeAlignBlocks_v1`, Triton's `BLOCK_*` | tile |
| "block", "message block" for a hash's compression input; the Lean's `Hm96.block`, `HmRow.blockBit`, `ROW_BLOCK`, `shaBlock`, `NetworkCheck.block`, until their owners rename them; BLAKE3's own terms | chunk (a compression's 1024-bit chunk) |
| "block" for a piece of a counter-mode stream (coins, noise); `Randomness.block`, `J10R.noiseBlock`, `verity.ml.randomness`'s `blocks()`, until their owners rename them | chunk (of the stream) |
| "block" for a unit of PoUS's encoded store; `blocks_per_segment`, `block_bytes`, until a format version | label (a segment's top label) |
| "block" for a unit of a store, registered or appended, and of a fill; the receipt kind `block-sha512`, `BLOCK_BYTES`, the Lean's `Requirement.liveBlocks`/`readBlocks`/`OneBlockEach`, `Tag.Block`, `appendCoinBlocks`, until a format version or their owners rename them | chunk (of the store) |
| "block", "block-scaled" for entries under one FP8 or NVFP4 scale; the checkpoint id `verity-fp8-block128/v1`, the Definitions `ScaledMmFp8Block_v1`, `ScaledMmFp8BlockSharedScale_v1`, `MoeExpertFp8BlockCoordinate_v2`, `BlockScaledAlignAdd`, `SiluMulFp8BlockQuant_v1`, vLLM's `weight_block_size` | scale group |
| "block" for a range of a GEMM's inner dimension; the Lean's `D1.nBlocks`, `ThreeBlock` | segment |
| "block" for a page of vLLM's KV cache, and its block pool; vLLM's `block_size`, `block_table`, `num_gpu_blocks` | KV page, page pool |
| "block" for a section of a record ("the workload record's sampling block") | section |
| "block" for a model's transformer layer; the model's `blocks.N` module paths, the tap family `moe_block_stream` | layer |
| "block" for a bucket of the network certifier's timing advice; the Lean's `OccupancyReadsOwnBlock`, `EncardAccSeqsLeBlocks`, until network accounting renames them | bucket |
| "block" for G of the certifier's served rows; the Definition `CompileSliceBlocks_v1` | row group |
| "physical block" (#4732) | server-time block |
| "isolation unit" (IU) | isolation block |
| "replay unit" (RU) | replay block |
| "leg"; "integrity", "work", "space" and "communication" naming subprotocols; `examples/toy_onsite/legs/` (`subprotocols/` after Demo B), the verdict keys `integrity` and `communication`, the Lean's `Core.Protocol` | subprotocol |
| "stage", outside "one-stage"; names and record keys that keep it, such as `research run --stage` | phase, or run |
| "row" for a server's window (s, w), "per-row bound" | serving block, serving-block bound |
| "work", "FLOPs", "MAC-times", "pricing" for how much a circuit computes or a protocol credits; pub(μ°), μ°, W_reg and W_b in ACCEL's docs until CMP-158; W_cert, the Lean's `WorkSound`, `CircuitWork*` names and the device model's `Price`; "κ/2" | size, in AND gates; U/s_× |
| "row" for a workload; vLLM's `verity-vllm row`, `row.py`, `config.RowConfig`, `row_file`, row ids and row dirs, and "row of record" (`manifests/records.json`), in `integrations/vllm/` and `examples/vllm/` until vLLM's workloads move | workload; a workload of record |
| "link-window", "link window"; code and records' `window` (`WardenParams.window`), the Lean's `LinkWindow` and `LinkWindowAcceptMeans`, until a format version | a sampling block of transfer blocks, over a sampling period |
| "window" for a span of time; `Accounting.Window`, `verity.ml.windows`, `window.kind`, until a format version | period: a sampling period, a certification period (`NetworkCheck.Period`), or a period of T ticks |
| "session"; code and records' `session` (`session_id`, `core-firewall session`, the challenger's gate session), until a format version | by sense: a run (one execution of a subprotocol); a release (the firewall's one outer proof per statement); a certifier's attachment to a link (its 32-byte id); a run's challenge phase (the live challenger's gate session) |
| "deploy", `verity deploy`, "deployment file", "deployment configuration" for running Verity's roles | `verity run CONFIG --party ROLE`; the configuration |
| "ACCEL" naming the protocol, and PoUW's `accel/` directory; the coin domain `verity/pouw-accel/<kind>/v1` in a Lean docstring (`Security/Definitions/Pouw/Assumptions/Dither.lean`), until DITHER-E2E renames it | Dither |
| "the accelerator model", "the machine model" for 𝓜 (`benchmarks/pouw/machine_model/` and its `machine-model` command keep the name) | the ACCEL model |
| `ShaSpeed` and `ServiceSound`, the Lean's names until DITHER-E2E's PR | `SAKSha` (SAK for SHA) and `ProofSystemSound` |

The `veritor.*` gate families are grandfathered and the prefix is frozen (circuits, 14 Oct 2026, CIR-12): each existing id
declared in `verity/ml/circuits/targets/target.py` (`veritor.tc-ampere-bf16@1`, the `veritor.tc-bf16.*@1` and
`veritor.tensor-core.*@2` families) keeps its exact spelling, because the `*_FAMILY_DIGEST` pins, fixtures and recorded runs
bake it in, and renaming would invalidate every one of them for no semantic gain. No new family is created under
`veritor.*`: a new family takes a fresh id, and `verity/ml/circuits/tests/test_targets_frozen_families.py` fails on any
`veritor.*` id outside the grandfathered set.

## The ACCEL model and Dither

Daniel named them on 26 Oct (7:51 PM PDT):

- **The ACCEL model** 𝓜 is the physical model: a period's devices as the Lean's `Accel` reads them
  (`Security/Definitions/Core/Accel/`), their resources (tensor cores, issue slots, shared-memory and register ports,
  device memory, links) each at its data-sheet rate, read at its floors on the H200 and the RTX PRO 6000. Its
  parameters, θ, are `benchmarks/pouw/machine_model/`'s data.
- **Dither** is the protocol, PoUW's no-fold design, at v3.2 (`verity/core/protocols/pouw/dither/PROTOCOL.md` while it is
  experimental; the Lean's `Pouw.Dither`), proved over the ACCEL model. Its receipt is no-fold (CMP-172): each
  segment's 16-bit noised prefix sums are committed as SHA-512 hiding leaves, hm96 rows of 16 KiB, before the check coin,
  and the sampled cells are opened after it. The served PoUW moves to it, and `SharedNoiseLucks`, which the served H1
  and J10-R folds rest on, has no place in it.

## Narrower senses

Some words keep a different, narrower sense:

- `verity.ml.circuits`'s `Circuit` is a Program's canonical gate layout and a `GateSelection` is a selected set of its gates; its query `Family` is an indexed sequence of gate sets; a primitive's `family` is the id stem of lazily materialised primitives such as `Const<w>`. The `subcircuits` of a `typed-obligation/v0` object (`verity_sp1.proofs.typed_obligation.Subcircuit`) are its shared gate bodies, and a sampled-proofs obligation's JSON `subcircuit` key holds its gate selection; both names are serialized and stay. `verity.ml.circuits`'s gates are word-level: each outputs one `Value<w>` of w bits, not one bit, and so are the gates of the Lean model in `Security/Definitions/Core/` (`Verity.Circuit`), whose Boolean profile is width 1.
- A **template instance** (the queries `Q_template_instance` and `Q_template_instances`) is one occurrence of a subcircuit template in a program, and in a claim id `property/instance`, such as `cr/sha-512`, the instance is the primitive an assumption is about; neither is an input.
- A **universal unit** (`verity.ml.private_circuits.universal.UniversalUnit`) is a public Boolean Definition, one per size class, that
  runs any **private circuit** of the class, whose description (its gates and wiring) is given as a value; in private circuits it
  is the circuit of one proof block, so the developer's circuit is hidden like any value.
- A **transition unit** is one tensor-core accumulator step; a GEMM-coordinate subcircuit chains `K/16` of them, and a "unit circuit" arithmetises one.
- "Candidate C" and "Candidate V" in older notes are the QuickSilver VOLE research backend in `archive/vole`, not C-Flock.
- In `integrations/vllm`, "family" and "kind" name circuit Definitions and query families, many of them serialized fields (see its README's vocabulary).

## Terms of one area

Vocabulary that one area uses, moved here from `ARCHITECTURE.md`'s Glossary so that it holds the shared terms:

- **Wire**: an edge from a gate to a gate that reads it; it carries its source gate's value and has none of its own.
- **Port** and **element**: a port is a named array of fixed-width elements on a circuit definition's boundary, one of its
  parameters or results, bound by each call to gates of the caller; under `Q_call`, a Call whose result ports fit in X bits
  is one proof block. An element is one member, such as 16 bits. A leaf serializes whole elements, but a partition counts
  bits, so a proof block may commit some of an element's bits and not others.
- **Format**: a label on a port's elements (BF16, FP32, E4M3, int32, token id) saying how their bits encode a number or a
  symbol; metadata, outside the content digest (`verity.ml.circuits.annotations`).
- **Boolean Definition** and **word view**: a Boolean Definition's gates are only gates (`verity.ml.circuits.boolean.is_boolean`),
  so its meaning is its gates; a *word Definition*'s gates output w-bit words whose meaning is registry code, and the word
  library migrates to Boolean Definitions under new versions of the same names. The word view reads each element's bits as
  one integer (`boolean.word_type`): under it, `GemmCoordinate_v3` takes and returns what `GemmCoordinate_v2` does.
- **Subcircuit template** and **subcircuit**: a template is named, with typed parameters for shape and numeric (silicon)
  semantics, such as `gemm_coordinate(K, semantics)`, a semantics value being an instruction id of
  `verity.ml.circuits.silicon.instructions.INSTRUCTIONS`; a subcircuit binds every parameter, such as
  `gemm_coordinate(K=1536, semantics=sm90.wgmma.m64n8k32.e4m3)`. The proof system proves subcircuits.
- **Input** and **input set**: one assignment of values to a subcircuit's input gates, with the output the chip produced
  beside it; an input set is a batch of them for one subcircuit, with their provenance (captured or synthetic).
- **Cell**: one benchmark result: a proof system (an archived one's backend id included) × statement × input set × prover
  hardware.
- **Transfer block**: the transfer gates a network certifier releases on one link, in one direction, at one tick of its
  clock (one bucket, `bucket_ms`): one bucket's row of that direction's record (`WardenParams.grid`), proved as its own proof
  block (`IngressScanBucket`, `EgressBucket`, `scan_zk.py`) (Daniel, 26 Oct). The code's link-window is a sampling block of
  2T transfer blocks, T buckets each way under one registration (`WardenParams.window`), its Program the T ingress then the
  T egress proof blocks chained through the scan state; all are proved today (`scan_zk --draw K` samples), each certified
  one with one liveness bit. The signed `fails` record is the certifier's own. A certifier's *attachment* to a link, its
  32-byte id, is one key and schedule on that link across many periods.
- **Lemma**: any proved statement that isn't a guarantee. A package keeps its lemmas in `SecurityProofs`, outside its spec,
  and they change freely while every guarantee proves the same statement.
- **Spec**: a Lean package's trusted text, all a reader must trust to read its guarantees: its definitions, assumptions and
  guarantees, importing only one another, the `Core` spec and Mathlib (`layers` in `lean-audit.json`). Every line is
  trusted, so it is kept short and the machinery goes in `SecurityProofs`.
- **Lock**: a package's guarantee records in its `lean-audit.json` (`AGENTS.md`, "Lean"): each guarantee's statement and
  every definition it reads. A changed record means a guarantee now says something else. It protects only the guarantees
  and what their statements read, though today it lists many lemmas too.
