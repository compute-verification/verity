# Glossary: notation, older names and narrower senses

The rest of the [Glossary](ARCHITECTURE.md#glossary), and it changes with it: a term that replaces an older one gets its row here in the same change.

## Notation

The one symbol table: every protocol states its guarantees in these symbols, one meaning each. A row names its Lean
declaration, or says "not yet in Lean". A gate count that credits or reports is in AND gates, and the
(c, k)-partitions count every gate ("Size, credit and the overhead lower bound", below).

A bound is argued in three parts: **the feasible set** 𝓕 (what the hardware can run at all, from the developer's
initial state), **protocol soundness** (each accepted protocol a's admissible set 𝒜_a holds except with probability δ_a;
`Verity.Soundness`) and **the overhead lower bound** (how much of training's necessary subcomputation the unaccounted
compute can do, the one place the ACCEL model's resources appear). The bounding strategy that gives the third connects
the task and the admissible set, and belongs to neither.

**Circuits**

| Symbol | Meaning | Lean |
|---|---|---|
| C, \|C\|, C(x) | A circuit: gates in topological order, each holding one value. \|C\| is its size, the number of its operation (non-input) gates. C(x) is the value of its output gates on the unregistered input x. | `Verity.Circuit`, `Circuit.computed`, `Circuit.size`, `Circuit.eval` |
| in(S), out(S) | For a set S of gates: the gates outside S with a wire into S; the gates of S read outside it, and its output gates. in(C) is C's input gates and out(C) its output gates. | `Circuit.into`, `Circuit.exits`; `Circuit.isInput`, `Circuit.outputs` |
| θ, x | The registered input gates, their values fixed before the period and the same in every circuit compared; the unregistered input, x : in(C) ∖ θ → values. | `Verity.Declaration.θ`; x is the input part of `Circuit.eval`'s argument |
| D, R = (θ, D, 𝓘) | The declared circuit and its registration, once: θ, D and 𝓘, D's (c, k)-partition into inference blocks. | `Verity.Declaration`; 𝓘 is a `Verity.Partition` |
| (c, k), 𝒞_{c,k} | A partition's gates per block, exactly (padded), and unregistered input bits per block, at most: \|I\| = c and \|in(I) ∖ θ\| ≤ k. 𝒞_{c,k} is the set of circuits with such a partition. | `Verity.Partition`, `Partition.gates`; the (c, k) condition `Partition.IsCK` not yet in Lean |
| A, x(g), t(g) | The actual circuit: everything run in the period, each gate g placed at a device x(g) and a tick t(g). | `Accounting.Placed`, `Accounting.Placed.x`, `Accounting.Placed.t` |
| A[X, J], Φ_X | The operation gates of A placed in the location X (a set of devices) during the ticks J; X's capacity in gates per second, \|A[X, J]\| ≤ Φ_X·\|J\|. Φ_s is a server's, Φ_total the fleet's, and M_s a server's memory in bits. | `Accounting.Placed.region`, `Accounting.Placed.ComputeRate`, `Accounting.Placed.Memory` |
| W_acc(X, J), U(X, J) | The accounted compute at X during J, the declared gate occurrences the protocols guarantee were computed as declared; the unaccounted gates, \|A[X, J]\| − \|A[X, J] ∩ W_acc(X, J)\|: slack on monitored compute, and every gate on dark compute. | the share 1 − W_acc/C: `Nci.Dampener.residualShare` (W_cert there, renamed later); the sets not yet in Lean |
| s_× | The fewest gates of a counted operation, in the same count: U unaccounted gates complete at most U/s_× counted operations. | not yet in Lean |

**The physical model**

| Symbol | Meaning | Lean |
|---|---|---|
| 𝓜, ρ | The ACCEL model: a period's devices, their resources (tensor cores, issue slots, shared-memory and register ports, device memory, links) and each resource's rate ρ, each a ceiling no program exceeds, at the firmware maximum; the card files `verity/hardware/cards/<sku>.toml` hold them. | `Verity.Accel.Machine`, `Verity.Accel.Card.ρ`, `Verity.Accel.H200.card`, `Verity.Accel.Sm120.card` |
| H, a, Λ | The fleet's fixed transition dynamics, its chosen initial state, and the chosen-state budget: A is H unrolled from a over the period (programs and data are in a or the inputs, never in the circuit's structure), and Λ bounds a, attached to the physical region it bounds. | `Accounting.Device`; `Verity.Device`, not yet in Lean; `FixedMachine` and `InitialBudget` not yet in Lean |
| k_exfil(Y), k_infil(Y) | Block Y's exfiltration capacity log2\|O_Y\|, O_Y the observable output behaviours (values, and timing, lengths and routing where an observer can tell them apart) Y can produce and still be accepted; the information reaching Y that its input accounting has not already charged. | not yet in Lean |
| HW, INV | Named assumptions. HW is the hardware model: 𝓜, the device model, links and side-channel rates, what a wipe clears. INV is the inventory's: coverage, dark compute's bound, and the honesty of certifiers, challengers and their hosts (`trusted-host`, which the per-party `Faithful`, `Secret` and `Confined` are to replace). | `Accounting.Hardware`, `Accounting.Placed.Links`, `Accounting.Placed.Causal`; `Accounting.Covered`, `Accounting.TrustedUnits` |

**Sets, protocols and the guarantee**

| Symbol | Meaning | Lean |
|---|---|---|
| 𝓕 | The feasible set: the computations, pairs (C, x), the hardware can run under HW from the developer's initial state a, which Λ bounds; dark compute included. | `Accounting.Overhead.physClass`, renamed later; a set of computations is an `Accounting.Overhead.Class`, renamed later |
| 𝒜_a, δ_a | The admissible set of protocol a and its failure bound: the computations protocol a accepts, except with probability δ_a, Pr[accept ∧ (A, x) ∉ 𝒜_a] ≤ δ_a. | `Accounting.Overhead.envelope`, renamed later; `Verity.Sound`; `Verity.Soundness`, not yet in Lean |
| 𝒜 | The admissible set, 𝓕 ∩ ⋂_a 𝒜_a, which every accepted run lies in except with probability Σ_a δ_a and the cryptographic terms. | `Accounting.Overhead.restricted`, renamed later |
| τ, b | A prover strategy; a server block (below). | `Verity.Game.Strategy`; b as `Verity.Block` |
| δ_coin, δ_crypto | Probability bounds from the live-coin draws, under uniform coins; advantage bounds for SHA-512's binding, knowledge soundness and pseudorandomness, against a stated adversary. | `Verity.Sound`'s δ; `Security/Definitions/Hash/` |
| Break, B^τ, 𝒟_b^τ | The event that the fixed finder exhibits a SHA-512 collision, or that the fixed forger produces a valid LMS signature under a device's registered key on a message the device's log never signed, charged as f(Pr[B^τ collides]) + Pr[the forger succeeds] and never assumed away; B^τ is the fixed finder built from τ; 𝒟_b^τ is the distinguisher built from τ at server block b, Adv_PRF(𝒟_b^τ) the PRF term. | not yet in Lean |
| ε_Wipe,b | The streamed wipe's failure bound at server block b, 2^-128 + Adv_PRF. | `StreamedWipeMeetsServer5GB`, not yet in Lean |

**Training tasks, cost and overhead**

| Symbol | Meaning | Lean |
|---|---|---|
| 𝒢, 𝒴 | The training task: one random input x, drawn once, and a success test on the output read at positions fixed before the draw. The acceptable outputs are a set 𝒴, since many trained models pass, and success means C(x) ∈ 𝒴. Tasks are static; an interactive world is a causal experiment with each candidate's own queries. | `Accounting.Overhead.Task`, `Accounting.Overhead.Succeeds` |
| η, P_α, η_cov | A success probability; the properties P_α a bound lower-bounds pointwise, each that the run computes a necessary subcomputation (below); the exception, Pr[success ∧ ¬⋁_α P_α] ≤ η_cov, uniform over strategies. | not yet in Lean |
| GPU-hours, cost(C) | The unit of cost. Every GPU of a server counts while the server computes anything, because the wipe and the per-block limit are per server, so whole servers are charged: cost(C) is each server's occupied time times its GPUs, summed over the servers, on C's worst input, and never wall-clock. | `Accounting.Overhead.cost`; `Accounting.Overhead.serverTime`, each server's occupied time without its GPUs, renamed later |
| T_𝒢(η; S) | The training cost of a set S of computations: the least GPU-hours at which some computation in S succeeds with probability above η, inf{cost(C) : Pr[(C, x) ∈ S ∧ C(x) ∈ 𝒴] > η}. | `Accounting.Overhead.threshold`, renamed later; `Accounting.Overhead.success` |
| overhead(𝒢, 𝒜, η) | The training cost over the admissible set divided by the training cost over the feasible set, T_𝒢(η; 𝒜)/T_𝒢(η; 𝓕), with the same initial state a and Λ on both sides, defined when the denominator is in (0, ∞). The reference training algorithm bounds the denominator. | `Accounting.Overhead.overhead` |
| slowdown | Wall-clock time only, never overhead: reaching the task takes at least T_𝒢(η; 𝒜) GPU-hours spread over the fleet's GPUs. | — |

**Monitored, dark and unaccounted compute**: monitored compute is the hardware the auditors know about, and dark compute
the hardware they don't. Each share has its denominator stated, an overlap counted once:

| Symbol | Meaning | Lean |
|---|---|---|
| ε_dark | Dark compute's share of the fleet's capacity, Φ(L_dark) = ε_dark·Φ_total, L_dark its devices; mainline 10^-3, range 10^-4 to 10^-2. Dark compute has no server blocks. | `Nci.Dampener.unmonitoredShare` |
| declared compute, ε_undecl | Declared compute is monitored compute running work the developer declares: served inference, and the protocol's own work at the public best's cost. Its operations per second over the card's ceiling is the declared utilization. ε_undecl is the undeclared compute's share of the monitored capacity C_X: monitored compute running nothing declared, such as idle or uncredited pipes, the host CPUs, the clock above the credited one, filler, and the honest implementation's excess over the public best. C_X is the location's capacity in 𝓜's resources at each SKU's ceiling, reported as counted operations (Φ_X, in gates per second, stays for the partition, c_S and the counting). | `Nci.Dampener.residualShare` reads it with ε_skip, as the slack; ε_undecl alone not yet in Lean |
| accounted compute, ε_skip | Accounted compute, W_acc (above), is the declared compute the protocols guarantee was done as declared, so it is at most the declared compute; the gap is skippable work, and ε_skip its share: what PoUW can miss, ε_W of the declared share, ε_W·(1 − ε_undecl), and the wrong certificates. | `Accounting.Compose` (ε_♭ = ε_skip + ε_cert + ε_route) |
| slack | Monitored compute that is not accounted, the undeclared compute plus the skippable work: ε_undecl + ε_skip of the monitored capacity, 1 − W_acc/C_X. It is the number driven to 0; the protocol's own work at the public best's cost is in neither, with γ ≤ 10 allowed. | `Nci.Dampener.residualShare` |
| unaccounted compute | Slack plus dark compute: everything hidden training could use if no per-block limit applied. | not yet in Lean |
| ε_break | The gates of the faulty proof blocks F. | `Nci.Guarantees.Dampener` (its χ) |

**The bounding strategy**, a proof technique that connects the task and the admissible set and belongs to neither:

| Symbol | Meaning | Lean |
|---|---|---|
| server block, c_S, k_S | The gates one monitored server computes between two wipes; dark compute has none. A server block is padded to exactly c_S = ⌊Φ_s·Δ_wipe⌋ gates and reads at most k_S = k_wipe + (r_in + r_side)·Δ_wipe unregistered input bits: what the wipe leaves of M_s (at most 4×10^10 bits plus egress), and the certified ingress and the side channels over the wipe interval Δ_wipe. | `Verity.Block`, `Verity.Blocks`, `Verity.Wipe` (their module's "server-time blocks", renamed later); the sum over server blocks not yet in Lean |
| inference block | A piece of the circuit shaped like inference: at most c gates and k unregistered input bits, one block of D's registered (c, k)-partition 𝓘 ("(c, k)-block" only in technical detail). | 𝓘 is a `Verity.Partition`; the (c, k) condition not yet in Lean |
| N | The necessary subcomputation: what every successful training run must compute (the P_α above), at least N counted products. | not yet in Lean |
| q, q_T(c, k) | The per-block limit: the most counted products one server block can complete; q_T(c, k) is a pricing argument's limit for a block of c gates and k unregistered input bits. | not yet in Lean |
| pricing argument | A proof of the per-block limit that charges each counted product for the input bits it needs: its operands, fan-out, partial sums and recovery. | `Nci.Guarantees.ThreadedReads`, `Nci.Guarantees.ContainedTraining` |
| reference training algorithm | The best known way to train with no protocol; it bounds T_𝒢(η; 𝓕), the training cost over the feasible set. | not yet in Lean |
| overhead lower bound | The result: a lower bound on overhead(𝒢, 𝒜, η), from where the necessary subcomputation can fit ("Size, credit and the overhead lower bound", below). | `Nci.Guarantees.Dampener`, `Nci.Guarantees.DampenerRemote`, renamed later |

## Size, credit and the overhead lower bound

Credit and all reporting are in circuit size, counted in AND gates (tentative).

- **Size** is a circuit's or a region's AND-gate count. AND gates are what C-Flock pays for, so size is the lingua franca
  every protocol credits in and every result reports computation in, in place of work, FLOPs, MAC-times and pricing.
  U unaccounted gates complete at most U/s_× counted operations, s_× the fewest gates of a counted operation in the same
  count; this replaces κ/2.
- **Credit** is a statement about values. An accepted run credits a size S, and its guarantee is
  Pr[accept ∧ the AND gates whose committed values are right number fewer than (1 − ε_skip)·S] ≤ δ_coin + δ_crypto. It
  needs no algorithmic premise: SAK (`SAKResidual`) is a premise of the overhead lower bound only. For the served PoUW,
  the one computational premise left is `SharedNoiseLucks` (the fold receipts' detection) until it moves to Dither's
  no-fold receipt, which takes none. The remote audit (`Accounting.Remote`) bounds no hidden work itself, so its PoUW
  guarantee is the size credit alone.
- **The overhead lower bound** bounds how much hidden training the unaccounted compute could do beside the credited
  work. Training's necessary subcomputation has to fit somewhere. Dark compute does it at the hardware's rate. In
  monitored compute two limits apply, and the smaller counts: each server block completes at most the per-block limit,
  and accounted compute is busy with its declared work, so only slack is free. Onsite the per-block limit is the smaller
  one; remotely, the slack is. The bound is the one place the ACCEL model 𝓜's resources appear (tensor-pipe time, issue
  slots, memory bandwidth), because gates can't measure a device's capacity. Its result is reported as counted
  operations, each labelled with that operation's canonical AND count, never converted to "ANDs of hidden work" (that
  would need a ceiling on ANDs per resource unit over all programs, which is vacuous).
- **What counts every gate.** The (c, k)-partitions (inference blocks, a server block's c_S) and the Lean's counting
  results (`Circuit.size`) count every gate, so a report that reads one converts it explicitly (`Circuit.andSize` beside
  `Circuit.size`, and credit as `Verity.creditSize`: not yet in Lean). In Credit, S is the AND count of the
  registered declared circuit under one public lowering pinned at registration, never the prover's own.
- **Not yet restated.** The Lean's device-model audits still price work per op (`Accounting.Run.ComputeSound` at a
  `Price`, and the covert residual F·T − W_acc, which is one device's slack), and Dither's `PROTOCOL.md` credits at
  pub(μ°) against W_reg until it is restated in AND gates.

Anyone who finds a reason this is wrong says so the moment they find it.

## Workload

A **workload** is what a datacenter runs over a monitoring period: its Programs,
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
| "verification unit" (VU); `vu_*` identifiers, the `verity/sampled-proofs/vu-sample/v1` domain, the `vllm-vu-set/v1` kind, the `vu-k1536` tier | proof block (C-Flock's statement layout keeps VU in its own sense: Narrower senses) |
| "M0"; its first statement id, baked into the Lean verifier's `Flock/Tags.lean` | C-Flock's circuit prover: module `verity_flock.circuit`, CLI `flock-circuit`, statement `verity/flock-circuit` |
| "M1", "M2" (elsewhere in the code these also name unrelated things) | C-Flock ZK |
| `hm96-sha256/v1`, until it is deleted | the hiding leaf `hm96-sha512/v1` |
| "coin server"; the `round_digest` strings of statement identities in `Flock/Tags.lean` | the challenger's live-coin part |
| "network transparency"; the `network-transparency/` paths in network accounting's project store that the `NetworkCertifier` Lean cites | network accounting |
| "warden", "network warden", "tap", "the auditor's tap", "uplink tap", "warden/certifier"; the domain tags `verity-network-warden/row/v1` and `verity-network-warden/record/v1`, the store label value `replay_set=warden/re-verify` | the network certifier; "the warden's records" are network certificates |
| "timed verifier"; the recorded config key `verifier_cpu` (`--verifier-cpu`) | the memory challenger |
| "prover gateway", "gateway"; one recorded statement string of recursive sessions | the firewall |
| "verifier gateway" | the challenger |
| "batch verifier" (B-Ligero's was another thing) | the verifier |
| "GPU workers" | workers |
| "netlist"; C-Flock's `verity/flock-netlist/…` tags and domains and the `flock-netlist/v1` file format of its older statements (`netlist-vectors.patch`) | a private circuit, or the circuit's description |
| "trusted text", in the `layers` reasons of `lean-audit.json` and the module docs of the `Protocol.lean` files | spec |
| "horizon" and "accounting period" for the stretch of time a bound is about; the Lean's `*Horizon*` and `*Period*` names (`HorizonSampling`, `RateHorizon`, `ComposePeriod`, `CircuitWorkPeriod`), the remote bandwidth ledger's `horizon` field, until they are renamed | monitoring period (`horizon` is its pooled reading) |
| "block" for a part of the device graph; `datacenter.physical.Block`, `block_of`, `network_sim`'s `--blocks` flag | zone |
| "block", "thread block" for CUDA's unit of a launch; CUDA's `blockIdx`, `blockDim` | CTA |
| "block" for a kernel's tile of a tensor; the Definitions `AttnBlock_v*`, `AttnBlockSoftcap_v*`, `AttnBlockFA3_v1`, `SparseMlaBlock_v1`, `TritonBlockRMSNorm_v1`, `MoeAlignBlocks_v1`, Triton's `BLOCK_*` | tile |
| "block", "message block" for a hash's compression input; the Lean's `Hm96.block`, `HmRow.blockBit`, `ROW_BLOCK`, `shaBlock`, `NetworkCheck.block`, until they are renamed; BLAKE3's own terms | chunk (a compression's 1024-bit chunk) |
| "block" for a piece of a counter-mode stream (coins, noise); `Randomness.block`, `J10R.noiseBlock`, `verity.ml.randomness`'s `blocks()`, until they are renamed | chunk (of the stream) |
| "block" for a unit of PoUS's encoded store; `blocks_per_segment`, `block_bytes`, until a format version | label (a segment's top label) |
| "block" for a unit of a store, registered or appended, and of a fill; the receipt kind `block-sha512`, `BLOCK_BYTES`, the Lean's `Requirement.liveBlocks`/`readBlocks`/`OneBlockEach`, `Tag.Block`, `appendCoinBlocks`, until a format version or their owners rename them | chunk (of the store) |
| "block", "block-scaled" for entries under one FP8 or NVFP4 scale; the checkpoint id `verity-fp8-block128/v1`, the Definitions `ScaledMmFp8Block_v1`, `ScaledMmFp8BlockSharedScale_v1`, `MoeExpertFp8BlockCoordinate_v2`, `BlockScaledAlignAdd`, `SiluMulFp8BlockQuant_v1`, vLLM's `weight_block_size` | scale group |
| "block" for a range of a GEMM's inner dimension; the Lean's `D1.nBlocks`, `ThreeBlock` | segment |
| "block" for a page of vLLM's KV cache, and its block pool; vLLM's `block_size`, `block_table`, `num_gpu_blocks` | KV page, page pool |
| "block" for a section of a record ("the workload record's sampling block") | section |
| "block" for a model's transformer layer; the model's `blocks.N` module paths, the tap family `moe_block_stream` | layer |
| "block" for a bucket of the network certifier's timing advice, and "the one-block cap"; the Lean's `OccupancyReadsOwnBlock`, `EncardAccSeqsLeBlocks`, until they are renamed | bucket (the τ_b time slot), and "a bucket's cap" |
| "block" for G of the certifier's served rows; the Definition `CompileSliceBlocks_v1` | row group |
| "compile unit" | the compile slice's proof block |
| "physical block" | server block |
| "isolation unit" (IU) | isolation block |
| "replay unit" (RU) | replay block |
| "leg"; "integrity", "work", "space" and "communication" naming subprotocols; the all-protocols runner's legs, the verdict keys `integrity` and `communication`, the Lean's `Core.Protocol` | subprotocol |
| "stage", outside "one-stage"; names and record keys that keep it, such as `research run --stage` | phase, or run |
| "row" for a server's window (s, w), "per-row bound" | server block, per-block limit |
| "work", "FLOPs", "MAC-times", "pricing" for how much a circuit computes or a protocol credits; pub(μ°), μ°, W_reg and W_b in Dither's docs until they are restated in AND gates; the Lean's `CircuitWork*` names and the device model's `Price`; "κ/2" | size, in AND gates; U/s_×. "Work" stays for resources spent in 𝓜, the overhead lower bound's unit, so `WorkSound` (freed compute in 𝓜's resources) keeps its name |
| "row" for a workload; vLLM's `verity-vllm row`, `row.py`, `config.RowConfig`, `row_file`, row ids and row dirs, and "row of record" (`manifests/records.json`), in `integrations/vllm/` and `examples/vllm/` until vLLM's workloads move | workload; a workload of record |
| "link-window", "link window"; code and records' `window` (`WardenParams.window`), the Lean's `LinkWindow` and `LinkWindowAcceptMeans`, until a format version | a sampling block of transfer blocks, over a sampling period |
| "window" for a span of time; `Accounting.Window`, `verity.ml.windows`, `window.kind`, until a format version | period: a sampling period, a certification period (`NetworkCheck.Period`), or a period of T ticks |
| "Commit window", for the steps one Commit covers | Commit period |
| "session"; code and records' `session` (`session_id`, `core-firewall session`, the challenger's gate session), until a format version | by sense: a run (one execution of a subprotocol); a release (the firewall's one outer proof per statement); a certifier's attachment to a link (its 32-byte id); a run's challenge phase (the live challenger's gate session) |
| "deploy", `verity deploy`, "deployment file", "deployment configuration" for running Verity's roles | `verity run CONFIG --party ROLE`; the configuration |
| "ACCEL" naming the protocol, and PoUW's `accel/` directory; the coin domain `verity/pouw-accel/<kind>/v1` in a Lean docstring (`Security/Definitions/Pouw/Assumptions/Dither.lean`), until it is renamed | Dither |
| "the accelerator model", "the machine model" for 𝓜 (`benchmarks/pouw/machine_model/` and its `machine-model` command keep the name) | the ACCEL model |
| `ShaSpeed` and `ServiceSound`, the Lean's names until they are renamed | `SAKSha` (SAK for SHA) and `ProofSystemSound` |
| ρ, "slowdown" and "the circuit-size slowdown" for \|C\|/T_0, "dampener", "compute dampener"; `benchmarks/rho` (its `Slowdown.rho`, `slowdown()`, the `rho` command) until they are renamed, the Lean's `Nci.Dampener` and `Dampener*` names (`DampenerRemotePeriod`) until they are renamed | overhead(𝒢, 𝒜, η); "slowdown" is wall-clock time only, and ρ is 𝓜's resource rate |
| T_0 and the reference circuit C_0 as overhead's normalizer; T_target ≥ L·T_0/Φ_total | the training cost T_𝒢(η; 𝓕), which the reference training algorithm bounds; the time to the task, at least T_𝒢(η; 𝒜) GPU-hours spread over the fleet's GPUs |
| A_phys, A_i(S), δ_i(S), A′ in roman letters; "layer 1", "layer 2", "layer 3"; "the one theorem", "soundness one theorem" | 𝓕, 𝒜_a, δ_a, 𝒜; the feasible set, protocol soundness, the overhead lower bound |
| ε_gap, and its `gap.*` routes in `benchmarks/rho` (`gap.floor`, `gap.clock`) until they are renamed | ε_undecl (`undecl.gpu`, `undecl.clock`) |
| ε for one GPU, "the uncertified share": model slack, idle pipe, ε_skip and exchanged bits | the slack at that GPU, ε_undecl + ε_skip; the exchanged bits are an input term (k_S's r_in and r_side), reported beside it in bits |
| 𝒜_phys, "physical class", "the physical model's class"; the Lean's `Accounting.Overhead.physClass`, until it is renamed | the feasible set 𝓕 |
| "envelope" for 𝒜_a, and "subprotocol application" for its a; the Lean's `Accounting.Overhead.envelope`, `benchmarks/rho`'s `envelopes.py` and `data/envelopes.toml`, until they are renamed | the admissible set 𝒜_a of protocol a |
| 𝒜′, "restricted class", "monitored class", "training class" for 𝒜′; the Lean's `Accounting.Overhead.restricted`, until it is renamed | the admissible set 𝒜 = 𝓕 ∩ ⋂_a 𝒜_a |
| "class" for a set of computations; the Lean's `Accounting.Overhead.Class`, until it is renamed | set |
| "initial information" for what both sides of the overhead start from | the initial state a, which Λ bounds |
| "server-time" for the cost; the Lean's `Accounting.Overhead.serverTime`, until it is renamed and counts each server's GPUs | GPU-hours |
| "threshold" for T_𝒢(η; S); the Lean's `Accounting.Overhead.threshold`, until it is renamed | the training cost T_𝒢(η; S) |
| "server-time block", "serving block", "server-window", and "window" for a server's gates between two wipes; "the serving-block bound"; the module docs of `Verity.Block` and `benchmarks/rho`'s `ServingBlock`, until they are renamed | server block; the overhead lower bound's sum over server blocks |
| "server block" for the proof blocks one server runs in one circuit, and "server-time block" for the steps that circuit spans; the network certifier's `blocks.Server` and `Blocks.window`, and the `server` and `server_time` keys of `verity/blocks/v1`, until a format version | a server's proof blocks; the circuit's steps |
| "certified block" | inference block |
| "coverage", "materialization" and "work requirement" for what every successful training run must compute; η_cov's subscript | necessary subcomputation (the covering bounds and the inventory's coverage keep their names) |
| "exclusion bound" and "exclusion" for the result | the overhead lower bound |
| "exclusion bound" for the per-block step, "per-block cap", "window cap", q_S; `benchmarks/rho`'s `Cap` and `data/exclusion.toml`, until they are renamed | the per-block limit q |
| "training class" for a premise set and the per-block count it gives; "allowance", "tier", "route" and "application" for a proof of the per-block limit; `benchmarks/rho`'s `classes.py`, `CLASSES` and `Allowance`, until they are renamed | pricing argument (`benchmarks/nci`'s tiers are sets of mechanisms, and `benchmarks/rho`'s routes kinds of slack) |
| "reference construction", "reference run", "reference circuit" for the best known training with no protocol | the reference training algorithm |
| "dark fleet", "dark leaves", "undeclared hardware"; the Lean's `Nci.Dampener.unmonitoredShare` | dark compute (ε_dark and L_dark keep their symbols) |
| "certified work", "certified share", W_cert for compute; the Lean's W_cert (`Nci.Dampener.residualShare`, `Accounting.Run.ComputeSound`) and `benchmarks/resource_accounting`'s `w_cert`, until they are renamed | accounted compute, W_acc. "Certified" stays where a certifier certifies something, such as certified ingress and the network certifier |
| "uncertified capacity", "the uncertified share", "the unaccounted share" for ε_undecl + ε_skip; "uncertified gates" | slack; U counts the unaccounted gates |
| G for the target or the acceptable outputs | 𝒴, and the task 𝒢 |
| θ for a card's numbers | the card table (θ is the registered inputs) |
| "envelope ISA", "envelope op" for the ACCEL model's instruction kinds; `machine_model`'s `cover` command and its `envelope` key, until they are renamed | the ACCEL ops (𝒜_a is the admissible set) |
| "coverage premise", "coverage table" for the ACCEL model's; `machine_model/data/cover/`, until it is renamed | the card premise, the opcode table (coverage is the inventory's) |
| "words" for Dither's committed 16-bit sums | prefix sums |
| "staging", "the staging decode" for copying a stored model's bytes into GPU memory and decoding them to the served weights; `benchmarks/torch_gpu/p2_stage`, until it is renamed | the store's load |
| "post-landing run", "postland" | after-landing run |
| "train" for a batch of PRs landed on one check | none: check-mode landing is deleted, and drive lands the reviewed prefix of `next` |
| tick(g) | t(g) |
| D for the chosen-state budget | Λ (D is the declared circuit) |
| D_b^τ for the PRF distinguisher | 𝒟_b^τ |

The `veritor.*` gate families are grandfathered and the prefix is frozen: each existing id
declared in `verity/ml/circuits/targets/target.py` (`veritor.tc-ampere-bf16@1`, the `veritor.tc-bf16.*@1` and
`veritor.tensor-core.*@2` families) keeps its exact spelling, because the `*_FAMILY_DIGEST` pins, fixtures and recorded runs
bake it in, and renaming would invalidate every one of them for no semantic gain. No new family is created under
`veritor.*`: a new family takes a fresh id, and `verity/ml/circuits/tests/test_targets_frozen_families.py` fails on any
`veritor.*` id outside the grandfathered set.

## The ACCEL model and Dither

- **The ACCEL model** 𝓜 is the physical model: a period's devices as the Lean's `Accel` reads them
  (`Security/Definitions/Core/Accel/`), their resources (tensor cores, issue slots, shared-memory and register ports,
  device memory, links) each at its data-sheet rate, read at its floors on the H200 and the RTX PRO 6000.
- **The card table** is one SKU's numbers at its firmware maximum: its per-resource rates ρ, stores, formats and clock
  ceiling (`benchmarks/pouw/machine_model/`'s data). θ is never a card's numbers; it is the registered inputs.
- **The ACCEL ops** are the ACCEL model's six instruction kinds (`Accel.Op`), each priced by a card's rates.
- **The card premise** (`CardWithinAccel`): anything the card computes in time T, the ACCEL machine computes within T at
  the card's rates. Its evidence is **the opcode table**, which places every opcode of the card on the ACCEL ops.
- **Segment**: L consecutive inner indices of one output cell, between two committed prefix sums. A Dither proof block is
  one segment of one cell.
- **Noise period** L′: the period of Dither's noise along K; a segment is t = L/L′ noise periods.
- **Prefix sums** T_s: Dither's committed running sums along a segment, stored as 16-bit elements.
- **Dither** is the protocol, PoUW's no-fold design, at v3.2 (`verity/core/protocols/pouw/dither/PROTOCOL.md` while it is
  experimental; the Lean's `Pouw.Dither`), proved over the ACCEL model. Its receipt is no-fold: each
  segment's 16-bit noised prefix sums are committed as SHA-512 hiding leaves, hm96 rows of 16 KiB, before the check coin,
  and the sampled cells are opened after it. The served PoUW moves to it, and `SharedNoiseLucks`, which the served H1
  and J10-R folds rest on, has no place in it.

## Narrower senses

Some words keep a different, narrower sense:

- A protocol's **ε_skip**, in Dither's `PROTOCOL.md` and the Credit statement, is that protocol's own skipped share of
  what it credits, the Notation's ε_W; the fleet's ε_skip is ε_W(1 − ε_undecl) plus the wrong certificates.
- `verity.ml.circuits`'s `Circuit` is a Program's canonical gate layout and a `GateSelection` is a selected set of its gates; its query `Family` is an indexed sequence of gate sets; a primitive's `family` is the id stem of lazily materialised primitives such as `Const<w>`. The `subcircuits` of a `typed-obligation/v0` object (`verity_sp1.proofs.typed_obligation.Subcircuit`) are its shared gate bodies, and a sampled-proofs obligation's JSON `subcircuit` key holds its gate selection; both names are serialized and stay. `verity.ml.circuits`'s gates are word-level: each outputs one `Value<w>` of w bits, not one bit, and so are the gates of the Lean model in `Security/Definitions/Core/` (`Verity.Circuit`), whose Boolean profile is width 1.
- A **template instance** (the queries `Q_template_instance` and `Q_template_instances`) is one occurrence of a subcircuit template in a program, and in a claim id `property/instance`, such as `cr/sha-512`, the instance is the primitive an assumption is about; neither is an input.
- A **universal unit** (`verity.ml.private_circuits.universal.UniversalUnit`) is a public Boolean Definition, one per size class, that
  runs any **private circuit** of the class, whose description (its gates and wiring) is given as a value. Circuit hiding is
  recursion (C-Flock's `--hide`), not the universal unit.
- A **transition unit** is one tensor-core accumulator step; a GEMM-coordinate subcircuit chains `K/16` of them, and a "unit circuit" arithmetises one.
- C-Flock's statement layout (`verity/ml/flock/verifier/PROTOCOL.md` §16) nests three levels inside one proof:
  - a **unit copy** is one copy of the statement's unit circuit in one slot of the unit range (the spec's "unit",
    `units_per_vu` ≥ 1 of them per VU);
  - a **VU** is one instance's share of the witness: the slots that lay out one drawn proof block, with its rows' digests,
    its unit copies and its outputs (a padding VU lays out none);
  - an **R1CS block** is one diagonal block of the statement's block R1CS, 2^{k_log} witness bits holding
    G = `vus_per_block` VUs, a power of two (the spec's "block"; a statement has 2^{nbl} of them). **C-Flock's block
    size limit** is k_log ≤ 27, at most 2^27 witness bits (`MAX_K_LOG`; the range the soundness bound covers,
    `Stmt.InRange`, with at most 1024 regions and m_pts ≤ 64): a proof block whose unit circuit doesn't fit in one
    R1CS block can't be proved as one.

  So "at least one unit per VU and a power of two of VUs per block" reads "at least one unit copy per VU and a power of
  two of VUs per R1CS block". An R1CS block is never a proof block, and the identifiers (`units_per_vu`,
  `vus_per_block`, `blocks_real`) keep their names.
- "Candidate C" and "Candidate V" in older notes are the QuickSilver VOLE research backend, since deleted, not C-Flock.
- A **pipeline stage** is one buffered step of a CTA's software pipeline, the operands it loads ahead of the tensor cores
  (CUTLASS's `Stages`, Triton's `num_stages`); it is not a phase or a run.
- In a kernel, a **warp** is CUDA's 32 threads that issue together, and a **lane** is one thread's index among them.
- A **sliding window** is a model's term: the span of positions a layer's attention reads (a model config's
  `sliding_window`). It is not a period.
- In `integrations/vllm`, "family" and "kind" name circuit Definitions and query families, many of them serialized fields (see its README's vocabulary).
- Five words still carry two senses in the code; each entry says which sense keeps the word and what the other is called:
  - **Row**: a commitment row, the bits one leaf hashes (hm96 rows, the `*/row/*` leaf schemes, the row-form statements).
    A matrix's row (a slot is one row of a step's activations) stays plain English. A workload or a server block is
    never a row (Older names).
  - **Transcript**: the messages of a proof or a session, what its verifier sees (C-Flock's, PoUS's transcript hash).
    The values of a run's gates are its trace; `evaluate_call`'s `transcript` argument and `ExecutionTranscript` keep
    the old word until they are renamed.
  - **Stage**: to stage a statement is to fix and write it before any coin draws (the firewall's staged statements).
    A step of a run is a phase (Older names); a stage of a Boolean lowering, such as one level of a log shifter, is a
    level; a pipeline stage keeps its sense (above).
  - **Window**: the code's windows (`verity.ml.windows`, a served window circuit and its window record, PoUW's window and
    its salt) are one period's served steps, and keep the word until they are renamed. A span of time is a period, a
    server's gates between two wipes are a server block, and a sliding window is a model's term (above).
  - **Escape**: a faulty proof block's chance to be accepted is its acceptance probability (Older names); the
    integrity profile's `escape` method keeps the old word. In vLLM's Program frontend, a host escape
    (`E_OPAQUE_HOST`) is a value read on the host, such as `.item()`, which the frontend refuses.

## Terms of one area

Vocabulary that one area uses, moved here from `ARCHITECTURE.md`'s Glossary so that it holds the shared terms:

- **Slot**: one sequence of a served batch, one row of a step's activations; a step at batch 4,096 commits 4,096 slots
  (`commit_slots`).
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
- **Input block** (`INPUT_UNIT` in `verity.ml.circuits.partition`): the part of a partition that holds every input gate
  (weights, request inputs, randomness). It is certified by checking each input's commitment root against its external
  anchor (the declared weights root, the client's request commitment, the randomness source), always, never sampled.
- **Pinned**, as an instruction's status (`verity.ml.circuits.silicon.instructions`): its silicon model matched every
  hardware probe, at the level `benchmarks/tc_probe/trust.py` computed and a trust dossier records; the other statuses are
  hypothesis and unprobed. A pinned value or root is another sense: one fixed before the run, which a check compares to.
- **Served MACs**: a hardware count in a circuit's census, the multiply-accumulates of the steps whose role is served
  (`Accounting.Gates.Census.servedMacs`, the census's `served_macs`). It is not size: credit and reports count AND gates.
- **Binding unit** (circuit-check's cache, `circuit_check.cache` and `targets.BINDINGS`): one unit of a binding file, a
  root function, a `_model_roots` entry or a `TEMPLATE_PARAMS` entry; a target's cached pass holds the digest of each
  binding unit whose roots reach it.
- **Transformer sublayer**: one of a transformer layer's two halves, its attention or its FFN (MLP or MoE), each with its
  residual add; in DeepSeek-V4, each is wrapped by the hyper-connections' pre and post mixes.
- **Value acquisition** and **native tap** (`integrations/vllm`'s `acquire/`): value acquisition takes a required value's
  bytes where vLLM computes it, through a Python hook or a native tap, a capture compiled into a kernel. A tap is never
  the network certifier (Older names).
- **Transfer block**: the transfer gates a network certifier releases on one link, in one direction, at one tick of its
  clock (one bucket, `bucket_ms`): one bucket's row of that direction's record (`WardenParams.grid`), proved as its own proof
  block (`IngressScanBucket`, `EgressBucket`, `scan_zk.py`). The code's link-window is a sampling block of
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
- **Checking and landing** (the team skill, "Landing"):
  - **Ready mark**: an owner's statement that a PR's head may land; drive admits only marked heads.
  - **Land verdict**: a reviewer's `land` or `hold` on one PR head, recorded as a label; the owner's own doesn't count.
  - **`next`**: the branch where drive stacks admitted PRs, one merge each, above `main`; `main` fast-forwards to its
    reviewed prefix.
  - **Preflight**: the checks `research queue ready` runs on a head merged onto `next` before it records the mark.
  - **Landing gate**: a script at `main` that every PR's merge onto `next` must pass.
  - **Lean check**: a run of every Lake package's audit (`lean-audit`) on a commit of `next`. A PR that changes a lock, or
    Lean that runs as code, lands only on a passed Lean check at or above it.
  - **After-landing run**: the full check of `main`'s tip after each landing; when it passes, `main` is verified.
  - **Pass**: a part of the check, reused while its declared inputs are unchanged. A verdict is a review's, never a pass.
  - **Pure move**: a change that only renames paths, certified by `move_check`; it lands alone, without review.
- **Running jobs** (`tools/research`, `infra/`):
  - **Attempt**: the evidence store's record of one run, completed or failed, with its outputs; `research run` is the
    command that makes one.
  - **Pool**: a node's named set of run slots (`research run --pool NAME`; `research` is any lane's, where
    `--on auto` spills). Not a "line" or a "queue".
  - **Node lease** and **GPU lease**: a node lease (`~/.research/lease`) is the node's own deadline, after which it stops
    once every run on it has custody; a GPU lease (`gpu-lease`) books a node's GPUs for one workload while it runs.
