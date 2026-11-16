## Narrower senses

- A protocol's **ε_skip**, in Dither's `PROTOCOL.md` and the Credit statement, is that protocol's own skipped share of
  what it credits, the Notation's ε_W; the fleet's ε_skip is ε_W(1 − ε_undecl) plus the wrong certificates.
- `verity.circuits`'s `Circuit` is a Program's canonical gate layout and a `query_ast.GateSet` is a set of its gates; its query `Family` is an indexed sequence of gate sets; a primitive's `family` is the id stem of lazily materialised primitives such as `Const<w>`. The `subcircuits` of a `typed-obligation/v0` object are its shared gate bodies; the name is serialized and stays. `verity.circuits`'s gates are word-level: each outputs one `Value<w>` of w bits, not one bit, and so are the gates of the Lean model in `security/Definitions/Core/` (`Verity.Circuit`), whose Boolean profile is width 1.
- A **template instance** (the queries `Q_template_instance` and `Q_template_instances`) is one occurrence of a subcircuit template in a program, and in a claim id `property/instance`, such as `cr/sha-512`, the instance is the primitive an assumption is about; neither is an input.
- A **universal unit** (`verity.circuits.private_circuits.universal.UniversalUnit`) is a public Boolean Definition, one per size class, that
  runs any **private circuit** of the class, whose description is given as a value. Circuit hiding is
  recursion (Flock's `--hide`), not the universal unit.
- A **transition unit** is one tensor-core accumulator step; a GEMM-coordinate subcircuit chains `K/16` of them, and a "unit circuit" arithmetises one.
- Flock's statement layout (`Flock.Statement`, `Flock.HmRow`, in `verity/protocols/proofs/flock/verifier/`) nests three levels inside one proof:
  - a **unit copy** is one copy of the statement's unit circuit in one slot of the unit range (the spec's "unit",
    `units_per_vu` ≥ 1 of them per VU);
  - a **VU** is one instance's share of the witness: the slots that lay out one sampled proof block, with its rows' digests,
    its unit copies and its outputs (a padding VU lays out none);
  - an **R1CS block** is one diagonal block of the statement's block R1CS, 2^{k_log} witness bits holding G =
    `vus_per_block` VUs, a power of two (the spec's "block"; a statement has 2^{nbl} of them). **Flock's block size
    limit** is k_log ≤ 27 (`MAX_K_LOG`; the range the soundness bound covers, `Stmt.InRange`, with at most 1024 regions
    and m_pts ≤ 64): a proof block whose unit circuit doesn't fit in one R1CS block can't be proved as one.

  An R1CS block is never a proof block; `units_per_vu`, `vus_per_block` and `blocks_real` keep their names.
- Flock's **Merkle cap** (`cap`, `caps`, `recursiveCaps`) is the row of a commitment tree's nodes that stands in for its root, the 2^c nodes at depth c (`Flock.Level.cap`), which a proof's rows open to (`Flock.merkleCheck`); it is not Pearl's "cap" (Older names).
- "Candidate C" and "Candidate V" in older notes are the QuickSilver VOLE research backend, since deleted, not Flock.
- A **pipeline stage** is one buffered step of a CTA's software pipeline, the operands it loads ahead of the tensor cores
  (CUTLASS's `Stages`, Triton's `num_stages`).
- In a kernel, a **warp** is CUDA's 32 threads that issue together, and a **lane** is one thread's index among them.
- A **sliding window** is a model's term: the span of positions a layer's attention reads (a model config's
  `sliding_window`).
- In `adapters/vllm`, "family" and "kind" name circuit Definitions and query families, many of them serialized fields (its README's vocabulary).
- Eight words still carry two senses in the code:
  - **Footprint**: a replay block's is where SPOT reads harm (Two-stage design, below). An HBM footprint
    (`ResourceAccounting.BoundReceipt.footprint`) and a theorem's (its hypotheses, in Flock's `ASSUMPTIONS.md`) keep the word.
  - **Row**: a commitment row, the bits one leaf hashes (`hm96` rows, the `*/row/*` leaf schemes, the row-form
    statements). A matrix's row stays plain English. A workload or a server block is never a row (Older names).
  - **Transcript**: the messages of a proof or a session, what its verifier sees (Flock's, PoUS's transcript hash).
    The values of a run's gates are its trace (`evaluation.reference.Trace`); the Python and Lean evaluators'
    shared refusal text keeps the old word.
  - **Stage**: to stage a statement is to fix and write it before any coins are flipped (the ZK server's staged
    statements).
    A step of a run is a phase (Older names); a stage of a Boolean lowering is a level; a pipeline stage keeps its
    sense (above).
  - **Window**: the code's windows (`verity.commitments.periods`' `Window`, a served window circuit and its window
    record, PoUW's window and its salt) are one period's served steps, and keep the word until renamed. A span of time
    is a period, a server's gates between two memory wipes are a server block, and a sliding window is a model's term
    (above). PoUB's window keeps the word: the span one challenge of the remote bandwidth audit covers.
  - **Escape**: an incorrect proof block's chance to pass the sample is its miss probability (Older names); the
    integrity profile's `escape` method keeps the old word. In vLLM's Program frontend, a host escape
    (`E_OPAQUE_HOST`) is a value read on the host, such as `.item()`, which the frontend refuses.
  - **Acceptance probability**: a prover's success probability, the δ of knowledge soundness's κ(δ). An incorrect proof
    block's chance to pass the sample is its miss probability (Older names), which the Lean's `effAcceptProb` and
    `acceptProb` lemmas and the docstrings of SPOT's Python verifier still call its acceptance probability.
  - **Cap**: SPOT's served cap S, the most served output bits a request counts in the capped two-stage form, and the
    network attester's registered response cap (`NetworkAttesterParams.max_tokens_cap`, the most tokens a response carries)
    keep the word. Pearl's cap is its inert-work bound (Older names), and "cap check" and an "over-cap" tile stay
    retired.

## Older names

Prose and printed text use the current terms; a site that still spells an older name is counted per file in the lint's
list `tests/lint/allowlists/repo_retired_terms.json`. Identifiers and stored record keys keep an older name until its
word map or its format's next version lands; each row below names the ones it covers.

Older names stay where they are baked into a digest, a frozen fixture, a recorded artifact field or a pinned vector. They mean
the term beside them, and `tests/lint/test_repo_retired_terms.py` warns on a new use of each row's quoted names (and of the
identifiers a clause saying "until renamed" lists) in any file it doesn't already allow. It refuses none: the test is
bookkeeping (`tests/conftest.py`), and the check's ratchet gate is off the merge gate.

| Old name, where it stays | Term |
|---|---|
| "Candidate A", "Candidate B"; the `candidate` feature and label keys | the archived proof systems A-GKR, B-Ligero |
| the `candidate` label values `SP1` and `C` (`research.store.vocab.CANDIDATES`) | D-SP1, and the VOLE candidate (not Flock) |
| "C-Flock"; the stored candidate label `C-Flock`, backend ids `C-interactive`, `C-fs` | Flock (Succinct's is "upstream Flock") |
| "sampled proofs", the sampled proof system's name; `sampled_proofs`, `SampledProofs` and the Lake package `core_sampled_proofs`, until renamed; the protocol's id `sampled-proofs` (its protocol entries such as `sampled-proofs:rows-sha512`, its record and component names) and the record ids under `verity/sampled-proofs/`, which keep the name until a format version | SPOT (`spot` in paths and Python names, `Spot` in Lean); a sampled proof block's proof keeps its words |
| `verity.circuits.targets.target.Target`, `TARGETS`, `FIRST`; the result fingerprint's `profile` | subcircuit (a `GemmCoordinate<K>` at one semantics) |
| `Target.family`, `FAMILY_DIGEST`, a gate family such as `veritor.tc-ampere-bf16@1` | the semantics parameter's gate set |
| "backend", "backend family", "configured backend" for Flock; `backends/` (Flock's old directory), `verity_flock.backend`, `BACKENDS` | the proof system |
| "IR", "Verity IR", "IR primitives", "IR evaluator"; `verity.ir` | circuit, Program; `verity.circuits` |
| "instance", "instance set"; the `bench-instances/*` datasets and `benchmarks/numerical/bench-instances/`, the fingerprint's `instances` (`contract.instances_ref`) and its tier (`vu-k1536`, …), the `instance-equiv/v1` and `vllm-vu-set/v1` kinds, the `instances_dataset` / `instances_tier` labels, until a format version | input, input set |
| "configuration" meaning a backend with a scheme; `drilldown.Variant`, `VARIANTS`; `views.Configuration`, `CONFIGURATIONS` | a proof system and a statement |
| `verity_sp1.proofs.statement.Statement` (statement formats 2 and 3 on the wire), its `KindProgram` and "kind" | the SP1-era wire encoding of a batch of proof-block obligations, not a statement in this sense; a kind is a subcircuit |
| "pin", "pinned theorem", "pinned statement"; `pins` in a `lean-audit.json` written before the rename (read as `guarantees`, rewritten by its next `--update`) and in `tools/verity/lean/Facts.lean`'s facts | guarantee |
| "law", "draw law"; `law` identifiers and record keys (`parse_law`, `same_law`, `law_object`, a registration's `law`), the Lean's `Law` | sampling design |
| "escape", "escape probability"; "acceptance probability" for an incorrect proof block's chance to pass the sample, the escape probability's next name, which the Lean's `*_escape` theorems, `effEscape`, `effAcceptProb` and `acceptProb` lemmas keep | miss probability (P_a; `effAcceptProb` is P̄_a at footprint 1), and its complement 1 − P_a the detection probability; acceptance probability keeps its knowledge-soundness sense, a prover's success probability (the δ of κ(δ)) |
| "egress cone" for the served outputs an incorrect proof block can change, and "the egress a wrong u reaches" for their number, in the Lean's docstrings | reach E, and its size \|E\| (the Lean's E(u)) |
| "proof unit", "unit class", and "unit" alone meaning one; `unitClasses`, `verity.circuits.parts`, `LinkWindows.proof_unit`, the `"proof-unit"` kind of Flock's Boolean export, a `Spec`'s public-item reasons (its digest hashes them), and the `vu_*` names (below) | proof block; a unit class is a size class, or Flock's proof-block class |
| "unit" in the audit record's keys `wrong_units` and `proved_units`, until a format version | proof block |
| "wrong unit", "incorrect unit", "faulty unit", "faulty proof block", "faulty set"; the Lean's `wrong` | incorrect proof block (and correct); the incorrect set F |
| "transcript", "committed values" and "the circuit's values" for the values of a run's gates; `ExecutionTranscript` | trace |
| "verifier of record" | the verifier, whose verdict counts and which the guarantees are proved about; other checking code is named for what it is: the Rust cross-check, a Python mirror, or a Python verifier until its Lean one replaces it |
| "δ_link", "the link term"; the Lean's `LinkSound` and `δlink` | ε_bind, the binding error |
| "verification unit" (VU); `vu_*` identifiers, the `verity/sampled-proofs/vu-sample/v1` domain, the `vllm-vu-set/v1` kind, the `vu-k1536` tier | proof block (Flock's statement layout keeps VU: Narrower senses) |
| "M0"; its first statement id, in the Lean verifier's `Flock/Tags.lean` | Flock's circuit prover: module `verity_flock.circuit`, CLI `flock-circuit`, statement `verity/flock-circuit` |
| "salted leaf", "salted commitment" | hiding leaf, hiding commitment |
| "coin source", "randomness source", "coin device" | the challenger's live-coin part |
| "network accounting" for the subprotocol | network attestation (the area and its team keep "network accounting") |
| "network certification", "network certificate" | network attestation: the subprotocol, and a network attester's signed record |
| "certification period" | attestation period |
| "production" ("in production", "the production prover", "production systems") | serving, or proving, by sense; the Rust and CUDA prover |
| "M1", "M2" (elsewhere in the code also unrelated things) | Flock ZK: M1 is its masking, M2 its coin tree |
| `hm96-sha256/v1`, until it is deleted | the hiding leaf `hm96-sha512/v1` |
| "coin server"; the `round_digest` strings of statement identities in `Flock/Tags.lean`, until a format version | the challenger's live-coin part |
| "network transparency"; the `network-transparency/` paths the `Certifier` Lean cites | network accounting for the area, which began as the experimental branch of that name; network attestation for what the paths' note proves, the network attester's timing-channel bound |
| "warden", "network warden", "tap", "the auditor's tap", "uplink tap", "warden/certifier"; what a format fixes, until the network attester's next format version: its domain tags `verity-network-warden/` (`cells-domain/v2`, `cells/v2`, `parameters/v3`, `registration/v2`, `units/v1`, `wiring/v1`) and a stand-in's (`verity/a5/warden-stand-in/v1`, `a5-warden/rest/`), the circuit `WardenRowIdentity` and the names its units are staged under (`unit/warden-row`, `unit/a5-warden`), a link window's kind and key `warden`, a row registration's value `warden/<direction>/<w>` and a fails certificate's link `warden-fails`; what a caller outside the tree gives, until its callers change with it: the auditor gate's verb `warden-rehearse`, the variables `A5_WARDEN_RUN`, `CR_WARDEN_RUN`, `N2_WARDEN_RUN`, `SC_WARDEN_RUN`, `A5_WARDEN_UNITS` and `WARDEN_SERVE_C_ALLOCATOR`, the flags `--warden-run` and `--warden`, the class `--class warden` (`N2_CLASS=warden`), the path `CR_PATH=warden` and the subcommand `negatives.py warden`; what recorded data holds: a kept run's log `warden.json`, its directory `warden/` and its reports' key `warden_run`, which programs read back, the trace corpus's labels `warden-heavy` and `warden-smallkv`, its keys `warden.*` and `meta['warden']`, its watch label `warden:PID` and its value `ingress="warden"`, and the store label value `replay_set=warden/re-verify`; network steganography's own terms, an active warden and a passive warden, stay | the network attester; "the warden's records" are network attestations |
| "network certifier", "the certifier" for the device; what a format fixes, until its next format version: its domain tags `verity-network-certifier/`, the party `network-certifier`, the record keys `"network_certifier"` and `network_certifier_run`; the Lean package `NetworkAttester` and the names that follow it, until lean renames them | the network attester |
| "memory challenger"; what code and records still spell, until its word map or a format version renames them: the party and role `memory-challenger`, the Python module `memory_challenger` (`pous/attester/`, `benchmarks/pous/memory_challenger/`), the Lean's `MemoryChallenger` names and its Lake package `core_memory_challenger`, the signed message `memoryChallengerMessage` and the domain tags `verity/memory-challenger-record/v1` and `verity/pous/memory-challenger` | the memory attester (the challenger, which every coin comes from, keeps its name) |
| "remote bandwidth audit", "remote bandwidth", "remote audit", "proof of useful bandwidth" | PoUB |
| "timed verifier"; the recorded config key `verifier_cpu` (`--verifier-cpu`), until a format version | the memory attester |
| "prover gateway", "gateway"; one recorded statement string of recursive sessions | the ZK server |
| "verifier gateway" | the challenger |
| "batch verifier" (B-Ligero's was another thing) | the verifier |
| "GPU workers" | workers |
| "netlist"; Flock's `verity/flock-netlist/…` tags and domains and the `flock-netlist/v1` file format of its older statements (`netlist-vectors.patch`) | a private circuit, or the circuit's description |
| "trusted text", in the `layers` reasons of `lean-audit.json` | spec |
| "horizon" and "accounting period" for the stretch of time a bound is about; the Lean's `*Horizon*` and `*Period*` names (`ComposePeriod`, `CircuitWorkPeriod`), PoUB's ledger's `horizon` field, until renamed | monitoring period (`horizon` is its pooled reading) |
| "block" for a part of the device graph; `Datacenter.blocks`, `block_of` | zone |
| "block", "thread block" for CUDA's unit of a launch; CUDA's `blockIdx`, `blockDim` | CTA |
| "block" for a kernel's tile of a tensor; the Definitions `AttnBlock_v*`, `AttnBlockSoftcap_v*`, `AttnBlockFA3_v1`, `SparseMlaBlock_v1`, `TritonBlockRMSNorm_v1`, `MoeAlignBlocks_v1`, Triton's `BLOCK_*` | tile |
| "block", "message block" for a hash's compression input; the Lean's `Hm96.block`, `HmRow.blockBit`, `ROW_BLOCK`, `shaBlock`, `NetworkCheck.block`, until renamed; BLAKE3's own terms | chunk (a compression's 1024 bits) |
| "block" for a piece of a counter-mode stream (coins, noise); `Randomness.block`, `J10R.noiseBlock`, `verity.commitments.randomness`'s `blocks()`, until renamed | chunk (of the stream) |
| "block" for a unit of the encoded store of proof of useful space (PoUS); `blocks_per_segment`, `block_bytes`, until a format version | label (a segment's top label) |
| "block" for a unit of a store, registered or appended, and of a fill; the receipt kind `block-sha512`, `BLOCK_BYTES`, the Lean's `Requirement.liveBlocks`/`readBlocks`/`OneBlockEach`, `appendCoinBlocks`, until a format version or their owners rename them | chunk (of the store) |
| "block", "block-scaled", "blockwise", "scale block" for entries under one FP8 or NVFP4 scale; the checkpoint id `verity-fp8-block128/v1`, the Definitions `ScaledMmFp8Block_v1`, `ScaledMmFp8BlockSharedScale_v1`, `MoeExpertFp8BlockCoordinate_v2`, `BlockScaledAlignAdd`, `SiluMulFp8BlockQuant_v1`, vLLM's `weight_block_size` | scale group |
| "block" for a range of a GEMM's inner dimension; the Lean's `D1.nBlocks`, `ThreeBlock` | segment |
| "block" for a page of vLLM's KV cache, and its block pool; vLLM's `block_size`, `block_table`, `num_gpu_blocks` | KV page, page pool |
| "block" for a section of a record ("the workload record's sampling block") | section |
| "block" for a model's transformer layer; the model's `blocks.N` module paths, the tap family `moe_block_stream`, until a format version | layer |
| "block" for G of the network attester's served rows; the Definition `CompileSliceBlocks_v1` | row group |
| "compile unit" | the compile slice's proof block |
| "isolation unit" (IU), "isolation block", "certified block" | inference block |
| "novel communication intensity" (NCI); "baseline" for θ; `Nci*`, `nci-*`, `ius` and the `nci` directories, until renamed | the inference-only policy; the registered information θ |
| "replay unit" (RU), `ReplayUnit`, `n_v`; "audit block"; "coarse unit" and "chain" in the Lean's older two-stage audit | replay block (n_ρ, its count of proof blocks) |
| "replay domain"; the planner's `ReplayDomain`, `Candidate.replay` and `per_replay`, until renamed | replay grain; n_ρ |
| "clustered" sampling, "commit-after-draw"; "deferred interior"; "inner rule", "inner sample", "the second draw", "stage 2"; "the first draw", "stage 1" | a two-stage design; the interior commitment; the second stage; the first stage |
| "leg"; "integrity", "work", "space" and "communication" naming subprotocols; the all-protocols runner's legs, the verdict keys `integrity` and `communication`, the Lean's `Core.Protocol` | subprotocol |
| "stage", outside a sampling design's stages (one-stage, two-stage, the first and second stage) and a pipeline stage; names and record keys that keep it, such as `research run --stage` | phase, or run |
| "row" for a server's window (s, w), "per-row bound" | server block, server-block yield |
| "work", "FLOPs", "MAC-times", "pricing" for what a circuit computes or a protocol credits; pub(μ°), μ°, W_reg and W_b in Dither's docs until restated in AND gates; the Lean's `CircuitWork*` and the device model's `Price`; "κ/2" | size, in AND gates; U/s_×. "Work" stays for resources spent in 𝓜, the slowdown lower bound's unit, so `WorkSound` (freed compute in 𝓜's resources) keeps its name |
| "row" for a workload; vLLM's `verity-vllm row`, `row.py`, `config.RowConfig`, `row_file`, row ids and row dirs, and "row of record" (`manifests/records.json`), in `adapters/vllm/` until vLLM's workloads move | workload; a workload of record |
| "benchmark row", "benchmark cell", "bench cell", "measured cell", "metric row", "metrics row"; "proof table" for the proof benchmarks' results; "row" and "cell" for what a benchmark recorded (a table's row, a commitment row and Dither's cell keep their senses) | a benchmark result, or a metric for one named number of it (Terms) |
| "link-window", "link window"; code and records' `window` (`NetworkAttesterParams.window`), the Lean's `TransferBlocks` and `TransferBlocksAcceptMeans`, until a format version | an execution phase's transfer blocks |
| "window" for a span of time; `Verity.Window`, `verity.commitments.windows`, `window.kind`, until a format version | period: an execution phase, an attestation period (`NetworkCheck.Period`), or T ticks |
| "timestep" (for `every` buckets) | slot stride: `every` buckets (`every` stays until a format version) |
| "Commit window", for the steps one Commit covers | Commit period |
| "session" for anything but Flock's and the ZK server's session (Terms, Session); code and records' `session` in those other senses (`session_id`, the challenger's gate session), until a format version | by sense: a run (one execution of a subprotocol); a release (the ZK server's one outer proof per statement); a network attester's attachment to a link (its 32-byte id); a run's challenge phase (the live challenger's gate session) |
| "deploy", `verity deploy`, "deployment file", "deployment configuration" for running Verity's roles | `verity run CONFIG --party ROLE`; the configuration |
| "ACCEL" naming the protocol, and PoUW's `accel/` directory; the coin domain `verity/pouw-accel/<kind>/v1` in a Lean docstring (`security/Definitions/Pouw/Assumptions/Dither.lean`), until renamed | Dither |
| "the accelerator model", "the machine model" for 𝓜 (`benchmarks/machine_model/` and its `machine-model` command keep the name) | the ACCEL model |
| `ShaSpeed` and Dither's `Pouw.Assumptions.ServiceSound`, the Lean's names until renamed; not H1's premise `Properties.Pouw.MonitoringPeriod.ServiceSound` (a window's one-stage Flock sessions), another premise, which keeps its name | `SAKSha` (SAK for SHA) and `ProofSystemSound` |
| "overhead", "rho" and ρ for the training-cost ratio, overhead(𝒢, 𝒜, η), "the overhead lower bound"; "the circuit-size slowdown" for \|C\|/T_0, "training efficiency" η, "hardware multiple"; the stored key `rho` (the circuit properties' output) and the Lean's ρ in prose, until renamed | slowdown(𝒢, 𝒜, η); the slowdown lower bound (honest overhead γ, and ρ as 𝓜's resource rate, stay) |
| "dampener", "compute dampener" D; the Lean's `Nci.Dampener` and `Dampener*` names, until renamed | the slowdown |
| T_0 and the reference circuit C_0 as overhead's normalizer; T_target ≥ L·T_0/Φ_total | the training cost T_𝒢(η; 𝓕); the time to the task, at least T_𝒢(η; 𝒜) GPU-hours |
| A_phys, A_i(S), δ_i(S), A′ in roman letters; "layer 1", "layer 2", "layer 3"; "the one theorem", "soundness one theorem" | 𝓕, 𝒜_p, δ_p, 𝒜; the feasible set, protocol soundness, the slowdown lower bound |
| ε_gap, and its `gap.*` routes in `benchmarks/slowdown` (`gap.floor`, `gap.clock`) until renamed | ε_undecl (`undecl.gpu`, `undecl.clock`) |
| ε for one GPU, "the uncertified share": model slack, idle pipe, ε_skip and exchanged bits | the free compute at that GPU, ε_undecl + ε_skip; the exchanged bits are fresh input (k_S's r_in and r_side), reported beside it in bits |
| 𝒜_phys, "physical class", "the physical model's class"; "envelope" for 𝒜_p, "subprotocol application" for p; 𝒜′, "restricted class", "monitored class", "training class", "faithfulness" for 𝒜′; "class" for a set of computations; "initial information" for what both sides start from; the Lean's `ResourceAccounting.Overhead.physClass`, `envelope`, `restricted` and `Class` and `benchmarks/slowdown`'s, until renamed | the feasible set 𝓕; protocol p's admissible set 𝒜_p; the admissible set 𝒜; set; the initial state a |
| "server-time" for the cost, "threshold" for T_𝒢(η; S); the Lean's `threshold`, until renamed | GPU-hours; the training cost T_𝒢(η; S) |
| "server-time block", "serving block", "server-window", "physical block", and "window" for a server's gates between two memory wipes; "the serving-block bound"; the module docs of `Verity.Block` and `benchmarks/slowdown`'s `ServingBlock`, until they are renamed | server block; the slowdown lower bound's sum over server blocks |
| "server block" for one server's proof blocks in one circuit, "server-time block" for the steps that circuit spans | a server's proof blocks; the circuit's steps |
| "coverage", "materialization" and "work requirement" for what every successful training run must compute; η_cov's subscript | necessary subcomputation (the covering bounds and the inventory's coverage keep their names) |
| "exclusion bound" and "exclusion" for the result; for the per-block step, "per-block limit", "per-block cap", "window cap", q, q_T(c, k), and `benchmarks/slowdown`'s `Cap`, until renamed | the slowdown lower bound; the server-block yield q_S, and the inference-block yield q(c, k) |
| "registered computation", "registered work" | the registered circuit |
| "training class" for a premise set and the yield it gives; "allowance", "tier", "route" and "application" for a proof of the yields; `benchmarks/slowdown`'s `CLASSES` and `Allowance`, until renamed | pricing argument (`benchmarks/nci`'s tiers are mechanism sets, `benchmarks/slowdown`'s routes free-compute sources) |
| "reference construction", "reference run", "reference circuit" | the reference training algorithm |
| "dark" (compute, fleet, leaves), "unmonitored", "undeclared hardware", ε_dark, L_dark, and the Lean's `Nci.Dampener.unmonitoredShare` and `benchmarks/slowdown`'s `eps_dark`; "monitored" (compute, hardware, a server) | unregistered hardware, ε_unreg, L_unreg; registered hardware (the monitoring period keeps its name) |
| "certified work", "credited work", "accounted compute", "declared compute", W_acc, W_cert for compute; W_cert in the Lean (`Slowdown.Slowdown.residualShare`, `ResourceAccounting.Run.ComputeSound`) and the block record (`verity/block-record/v2`), `w_cert`, until renamed; "certified share", "declared utilization" | verified compute, V ("certified" stays for what a network attester certifies); utilization |
| "slack", "uncertified capacity", "uncertified gates", "unaccounted" (share, compute, gates), "undeclared compute", "skippable work", "residual compute", "monitored residual" (r, r̂) | free compute, U (its registered part is ε_undecl + ε_skip). "Slack" keeps its other senses |
| "unregistered input", "unregistered input bits", "unregistered information", "novel information", "novel bits", "novel input" | fresh input, fresh input bits (in(I) ∖ θ stays as notation) |
| a for the protocol index: 𝒜_a, δ_a | p: 𝒜_p, δ_p (a is the initial state) |
| G for the target or the acceptable outputs | 𝒴, and the task 𝒢 |
| θ for a card's numbers | the card table (θ is the registered inputs) |
| "envelope ISA", "envelope op"; `machine-model cover` and its `envelope` key, until renamed | the ACCEL ops (𝒜_p is the admissible set) |
| "coverage premise", "coverage table"; `machine_model/data/cover/`, until renamed | the card assumption, the opcode table (coverage is the inventory's) |
| "words" for Dither's committed 16-bit sums | prefix sums |
| "staging", "the staging decode" for copying a stored model's bytes into GPU memory and decoding them to the served weights; `benchmarks/torch_gpu/p2_stage`, until renamed | the store's load |
| "post-landing run", "postland" | after-landing run |
| "train" for a batch of PRs landed on one check | candidate |
| "draw" (noun, verb), "drawn proof block", "drawn unit", "sampled unit"; `DrawIsCoin`, `Remote.Draw.Session`, `--draw`, until renamed | the sample, a sampled proof block; to sample (coins are flipped) |
| "components", "product" for a composite's parts and their joining, and the Lean's and the channels' message format's names | subprotocols; composition |
| "firewall"; the record ids and domain tags `firewall/v0` and `verity/firewall-*`, the party name `firewall` and the message type `Firewall`, until a format version; `benchmarks/`'s `MODES=firewall`, `--firewall`, `FC_FIREWALL*` and `flock-firewall`, until renamed | the ZK server |
| `flock-verify`, `remote check` for the verifiers, until renamed | `verity verify proof`, `verity verify remote` |
| "builder", "advice compiler", "circuit builder" for what compiles the advice; `Builder` | registered compiler |
| "declared advice", "declaring" or "a declaration" of advice; "declared circuit", "declared work"; "declaration block", "declaration period" | reported advice; registered circuit, the registered circuit's work; execution block, execution phase |
| a served span: "window", "commit period" for PoUW's span and `window`, "sampling block", until renamed | execution phase |
| "window" for SPOT's served span with its audit; SPOT's records' `window` (`window_root`) until a format version | run (`Spot.Run`); its served part is the run's serving, and its length the run length |
| "sampling period" for work plus its audit | an execution phase and its verification phase |
| "hm96 rows" in protocol prose (the commitments' docs keep it) | hiding commitments |
| "proof service" | proof system |
| "audit coin"; "the audit" as a step | challenger's coins; verification phase |
| "prover" for the party: "the prover's node, machine, host or site", "a machine the prover doesn't control", "nothing the prover runs or hosts", "prover-controlled", "prover hardware", "the prover system" | the developer: the developer's node, machine, host or site, a machine the developer doesn't control, the developer's hardware ("prover" stays for the proving side of a proof: Flock's prover, a proof's messages, a cheating prover in a soundness statement) |
| "deployment" for onsite or remote, "onsite deployment", "remote deployment" (a datacenter running Verity, a deployment record and `verity.runtime` keep the word) | onsite or remote, the list of protocols a configuration names, the Lean's `Composite` |
| "composite protocol" (the Lean's `Composite` keeps its name) | compose, over onsite's or remote's list; protocol soundness is stated per configuration |
| "named premise", and "premise" for a named `Prop`: "the card premise" | assumption, or a proposed assumption until a lock lists it: the card assumption ("premise" stays for a theorem's hypothesis) |
| "before that period's coins exist"; "rejection dominates", "reject dominance" | before any coins are flipped; any rejection rejects |
| tick(g) | t(g) |
| D for the chosen-state budget | Λ (D is the registered circuit) |
| D_b^τ for the PRF distinguisher | 𝒟_b^τ |
| γ for PoUW's gap ("the γ-gap game", G_γ, the Lean's `Gγ*`, `gammaOf`, `*Gamma*`, `reductionGamma`, `Params.γ0`); η, and ε in TT_OUT and TT_NCP, for a failure probability (`εPearlC`, `WindowEps`, `epsL5e`); "weights" for a fixed operand (`Layout.wid`, `Workload.weight`, `weightOK`, `WeightsOK`); Pearl's δ for its noise scale (`Params.δ`). They stay in records and pins stored before the rename (`price_twins.py` reads both spellings), the wire names `weights_root` and `verity/pouw/h1/weights/v1`, Python's `gamma` and `gamma0` fields and the claim id `pouw-gamma-per-coin` | the gap g, with advantage ε = g/(1 − g) (`GGap*`, `gapOf`, `*Gap*`, `reductionGap`, `Params.gap0`); the failure probability δ (`δPearlC`, `ExecutionPhaseDelta`, `deltaL5e`); the fixed operand (`fixedId`, `fixed`, `fixedOK`, `FixedOK`); the noise scale σ (`Params.σ`) |
| Pearl's "debit" and "cap": "debit", "debited element", "unneeded" element, "cap", "cap check", "over-cap" tile; `pearl_c_debit`, `debit_units`, `forming_elements`, `Rule.debitUnits`, `debitOf`, `tileDebitRev1`, `capOK`, `Device.cap`, `CAP`, until renamed | inert work; an inert element; the inert-work share ι; the inert-work bound; a tile over the inert-work bound |
| "subprotocol" for a protocol a configuration lists | protocol |
| "(c,k)-partition" for a partition of a circuit's gates into blocks within the budgets (c, k), and "(c, k)-partition" for the same | the (c,k,n)-inference circuit's partition into (c,k)-inference blocks |

The `veritor.*` gate families (`verity/circuits/targets/target.py`) keep their spelling, which `*_FAMILY_DIGEST`
pins, fixtures and runs bake in; a new family takes a fresh id (`test_targets_frozen_families.py`).
