# Glossary

The repository's one glossary: code, docs, CLI help and output fields use its terms. When a term changes, its entry
changes here and the older name gets a row under Older names, in the same change.

## Terms

**Parties and devices**

- **Developer** and **auditor**: the protocol's two parties. The developer runs computations in its datacenters, keeps
  private records of them and proves properties of those records. The auditor checks the proofs, with a challenger
  online and a verifier offline, and places its own devices at the developer's site on machines the developer doesn't
  control. Nothing the developer runs or hosts is trusted; trust in a device and its host is the named assumption
  `trusted-host` (`TrustedHost`), which per-party `Faithful`, `Secret` and `Confined` are to replace.
- **Challenger**: the auditor's device that draws live coins, committed in advance, and keeps the record; at the MVP it
  is the device running the server's network certifier, so a server's coins never leave that boundary in the clear. It
  never sees the witness. Its premises `CoinsConfidential` and `OneClock` (one clock for every auditor device of a
  monitoring period: the challenger, every network certifier and the memory challenger) are named premises under `trusted-host`. Private messages travel only on private links
  (`WitnessPathPrivate`, proposed). Fiat–Shamir's challenger keeps its own sense.
- **Verifier**: the auditor's Lean verifier, run offline, and the outcome it writes; C-Flock's is
  `verity/core/service/flock/`, cross-checked by upstream Flock's Rust verifier. As a party in a statement, "the
  verifier" is the auditor's side as a whole.
- **Network certification** is the subprotocol, and **the network certifier** its device: an auditor-trusted device on
  each node's uplink, untrusted by the developer (`verity.core.protocols.certifier`), that attests the traffic it
  carries in **network certificates** (its signed records, witness, never shown to the auditor in the clear) and scrubs
  its timing. It alters traffic to close covert channels, network steganography's active "warden", but
  "network certifier" is its one name.
- **Memory challenger**: an auditor-trusted device beside a network certifier, never a process on the prover's node,
  that runs PoUS's timed rounds and signs each (`MemoryChallenger`); PoUS publishes the verdict.
- **Workers**: the developer's servers and GPUs, which run the computation and nearly all proving, trusted by neither
  party. **Firewall**: the developer-trusted only path from the workers to the auditor. It commits every outbound
  message under a fresh salt, masks and checks the rest, and releases outer proofs, zero-knowledge, one per statement
  (each its *release*), on a fixed schedule. The prover is the workers behind it (P = F ∘ W).
- **Proof service**: the one service behind every subprotocol: the workers behind the firewall, the challenger and the
  verifier. A protocol states only a `Spec`: what it commits, its selection rule, its check as a public Program and its
  public items, each with its reason. The service handles commitments, coins, proving and verifying.

**The lifecycle**

- **Registration**: the developer registers, append-only, its **hardware** (modelled as **the site**, `Verity.Site`,
  recursive, folded from the append-only registration), its **information** θ, the input gates whose values are fixed
  before the period (the models, code and data committed up front), and a **circuit builder** B, the ahead-of-time
  compiler (vLLM's `program/predict/`). The subprotocols' Programs and the proof system are the protocol's own, not registered.
- **Execution**: for each **declared advice** a_j (declaring is the developer's), B outputs the **execution block** E_j,
  its partition 𝓘_j into inference blocks, and its placement loc_j on the site's registered devices. B also outputs the
  cut into proof blocks, which validation checks against public rules (each proof block's outputs within its public
  size class, a count cap). The site is public; a_j, 𝓘_j and loc_j (where on the site each part of the workload runs) are private,
  and only their padded counts are public. For network traffic loc_j carries when: B predicts each transfer's send
  tick, and a_j corrects it. a_j is charged only by its length, a public budget per execution block, and is committed
  after E_j runs and before E_j's challenge. PoUW's own compile is not part of B. The **declared circuit** is D = E_1 ‖ … ‖ E_n, the one use of "declared". The **actual circuit** A is everything the hardware
  computed, hidden work included, each gate placed at a device and a tick (`Accounting.Placed`).
- **Verification** follows each execution in five steps, in order: **commitment**, **challenge** (live coins),
  **proof**, **validation**, where the verifier checks the proofs, and **accounting**, which tallies each subprotocol's
  resource accounts for E_j. The policies read those accounts: the inference-only policy checks each E_j against them.
  Every commitment is bound before any challenge. Execution and verification loop, and the bound over a monitoring
  period (The bound, below) is read from the tallies.

**Time and blocks**

- **Monitoring period**: the one word for a span of time (`Verity.MonitoringPeriod`, over `Verity.TickSpan`, a span of
  ticks). It is what a result is about; results are given for a week, a month and a year, against the adversary
  resource model. A **tick** is one step of the one clock. Every other span is a block's: an execution block, a sampling
  block (the network certifier's is T ticks, about 3.2 s), a server block. A block occupies ticks, and its span is said
  as the block's ("E_j's span"), never as a period or a window of its own. The network certifier's **slot stride** (EVERY
  ticks, holding SLOTS execution blocks) is a parameter, not a period.
- **Horizon** means a monitoring period. A bound over it has two readings: per sampling block, each at its own δ,
  summed; and **pooled**, the total at one δ, since each sampling block draws its own coins. The pooled reading is the
  result: its sampling term doesn't grow with the monitoring period, while a per-tick term such as a link's capacity
  does.
- **Session**: one run of a subprotocol, with its own coins and record. **Run**: one execution of the protocol or a
  circuit (a circuit's run is its trace). "Stage" survives only in "one-stage".
- **Adversary resource model**: the compute the adversary is taken to have, the world's: the census entry
  `world-inference-2027` (`benchmarks/census/adversary.json`).

**Circuits and values**

- **Program** (`verity.ml.circuits.Program`): one circuit, stored compressed as Definitions called from a root body.
  Its input gates are **registered inputs** (`RegisteredInput<w>_v1`: weights, constants, tables, bound when it is
  registered) or **run inputs** (`Input<w>_v1`: a request's data and parameters). Both are values.
- **Operation**: a Definition whose own definition is gates (format constants, SHA-512's round constants, measured
  MUFU tables), one pinned list (`verity.ml.circuits.operations`); its guard `operations.program_data` lists the data a
  program still holds.
- **Circuit**: gates and the wires between them; an *expanded* circuit is written out gate by gate. **Gate**: an input,
  a constant bit of an operation, the AND or XOR of bits, or NOT (`verity.ml.circuits.boolean`). A constant or table a
  program or request brings is a value, never a gate.
- **Value**: the bit a gate takes in one run. The **trace** is a circuit's gate values. Structure (gates, wires, ports,
  shapes) is public; values are always hidden, committed under a hiding leaf and touched only inside a zero-knowledge
  proof. A **transcript** is only the messages between the parties; a **log** is only how an implementation stores it;
  the proof system's internal transcript is always qualified as such.
- **Commitment scheme**: how serving commits values (frame-v3 rows, `vllm-v1`; `verity.ml.commitments`). **Hiding
  leaf**: a leaf whose data statistically hides its value (`hm96-sha512/v1`, N·2^-192 for N leaves).
- **Statement**: what the proof system proves: a circuit plus the commitment scheme binding its inputs and outputs.
  **Proof system**: C-Flock, whose circuit prover proves an expanded circuit (`verity/flock-circuit`) and whose
  zero-knowledge version, C-Flock ZK, the proof service runs.
- **Partition**: a division of a program's computed gates into proof blocks (`verity/partition/v1`), a named query
  the verifier evaluates itself (`Q_call`, `Q_word`); the committed set follows from it.
- **Block**: a part of a circuit, declared or actual, always with a prefix: a **proof block** (one part of a partition,
  proved on its own, outputs within its size class, 16 to 262,144 bits), a **sampling block**, a **server block**, an
  **execution block**, a **replay block** (re-executed to build a sampled proof block's witness), an **inference
  block**, a **multiplication block**, PoUW's **check block**, a **transfer block** and **C-Flock's block**. "Block"
  names nothing else; an outside API's name (`blockIdx`, `block_table`) stays.
- **Faulty proof block**: one whose committed outputs differ from its gates applied to its committed inputs.
- **Workload**: what a datacenter runs over a monitoring period: its Programs, their inputs and their placement, one
  thing at every scale; workloads compose. vLLM's files (`integrations/vllm/workloads/*.json`) each state one.
- **Instrumented program**: the program plus serving's taps and hashing, derived, never authored, and never proved.

**Sampling**

- **Sampling design**: the rule by which the verifier's coins pick proof blocks from the partition: `subset:K`,
  `bernoulli:NUM/DEN`, `stratified:K` (the default), `work:K` or `rate:NUM/DEN[:T]`. The **sample** is the set of proof
  blocks the coins picked. **Selection rule** (`Spec.selection`): all proof blocks, a sampling design, or a timed run's
  sequential sampling.
- **Sampling block**: the proof blocks one sample covers, committed and registered before its coins; stratified by
  subcircuit, with one joint harm bound over its strata, pooled over a monitoring period.
- **Acceptance probability** P_a(B): the chance the sample holds no proof block of B. **Confidence set** M_δ: every B
  with P_a(B) > δ; except with probability δ + ε_ks + ε_bind, an audit rejects or the faulty proof blocks are in M_δ.
- **Integrity profile** (`IntegrityProfile`): the audit's output, per-stratum bounds on faulty proof blocks, which
  consumers weight (exfiltration by output bits, compute by size).
- **Commit-and-prove argument**: what proves the sample, with errors ε_ks, ε_bind and ε_zk. C-Flock instantiates it.

**Subprotocols and composites**

- **Subprotocol**: one of the protocols a composite is made of, each going by its name alone. The one list of them, with
  the composites that include each, is the [composition
  README](verity/core/protocols/composition/README.md#subprotocols).
- **Composite protocol**: onsite, for an auditor with devices at the datacenter, and remote, for one with none. A
  **deployment** is a verifier taken as one value, its subprotocols and the policy's check; protocol soundness is stated
  per deployment.
- **Proof of useful bandwidth** (PoUB): the one bandwidth subprotocol, onsite-only for now: a lower bound on the bits
  read from a node's registered store, applying the bandwidth audit's guarantee (`AuditRO`) per pass.
- **Proof of useful space** (PoUS): a lower bound on the bits a device's memory held, from the memory challenger's
  timed rounds. **Proof of useful work** (PoUW): credits a size S of noised matrix products, at least (1 − ε)·S of
  its AND gates right.
- **ACCEL** and **Dither**: the ACCEL model 𝓜 is the physical model (`Security/Definitions/Core/Accel/`): a period's
  devices, their resources at data-sheet rates, its six ACCEL ops, and each SKU's **card table**; the **card premise**
  (`CardWithinAccel`), evidenced by the **opcode table**. Dither is PoUW's no-fold design (`Pouw.Dither`), proved over
  it: each segment's 16-bit noised **prefix sums** committed before the check coin. A **segment** is L inner indices of
  one cell, t = L/L′ **noise tiles**, L′ a length along K.
- **Memory wipe** and **server block**: a registered server's life is cut into server blocks, the gates it computes
  between two memory wipes (`Verity.Block`, `Verity.Blocks`), each opened by a certified wipe: the device fills its
  store with labels from a fresh seed, the memory challenger runs timed spot reads, and what leaves meanwhile counts as
  egress (`Verity.Wipe`). Only the carried state crosses a boundary, as a committed value; of the rest only what
  `Wipe.SurvivorBound` allows survives. The model has no host store.
- **Policy**: a condition on declarations that the verifier checks in validation from the declared circuit and θ alone,
  with no rounds. **The inference-only policy** is the first: each E_j's partition 𝓘_j is a (c, k)-partition, blocks
  of exactly c gates each reading at most k bits of fresh input (`Nci.Complies`, in an older form; `NciCheck`). An
  inference episode reads little fresh input and a training episode a great deal, so training through inference blocks
  yields little.

**The bound**

- **Feasible set** 𝓕, **admissible set** 𝒜 and **slowdown**: a bound is argued in three parts: 𝓕, what the hardware
  can run from the initial state a under HW, unregistered hardware included; **protocol soundness**, each accepted
  protocol p's admissible set 𝒜_p except with probability δ_p, and 𝒜 = 𝓕 ∩ ⋂_p 𝒜_p; and **the slowdown lower bound**,
  from a bounding strategy. **Slowdown** is T_𝒢(η; 𝒜)/T_𝒢(η; 𝓕), measured in **fleet-time**: elapsed time times the
  whole fleet, registered and unregistered (the Lean's `hardwareMultiple`). The protocols cap the compute rate
  available to training at R_free ≈ (U + ν)Φ, with U = ε_undecl + ε_skip + ε_unreg, so slowdown(G) ≈ max(1,
  ρ_G/(α R_free)) for a run G needing rate ρ_G: tiny runs aren't slowed, frontier runs are slowed by about
  1/(α(U + ν)), and the result is a curve over run size. **Prover overhead** γ is the honest developer's cost, not a
  slowdown.
- **The training task of record**: retrain the latest frontier model from scratch. Formally 𝒢 asks for a loss below
  every registered model's (a registered model is read freely, so reproducing one isn't training), from a random
  initialization and θ with no other carried state, and the monitoring period covers the whole run. The reference
  training algorithm is the frontier recipe, with full-parameter updates.
- **The task assumptions**, three, none naming the protocol. **Active parameters** (N*): any model reaching 𝒢's
  loss uses at least N* parameters per token, with N* above what one server block carries from one block to the next.
  **Incompressible intermediate state** (β): each intermediate value a token's computation passes on, activation or
  gradient alike, carries at least β bits. Together they rule out runs built from pieces that each fit one server
  block (independently trained small experts): every token's computation must cross servers, and each crossing is
  priced through the certified channels, which gives the cut bound. **Reference efficiency**: no run uses the hardware
  more efficiently than the reference, so free compute counts at MFU₀, not at peak. With the near-optimal reference
  (α) these are all the bound assumes about training.
- **Bounding strategy**: a necessary subcomputation (N counted objects every successful run computes) and a pricing
  argument for it, giving the yields. Each server block yields at most min(q_S, its inference-block yields plus its
  free compute), unregistered hardware works at its rate, and the straddle term covers objects split between verified
  and free compute. Onsite q_S is the smaller term; remotely free compute sets the number.
- **Registered hardware**, **fresh input**, **verified compute** and **free compute**: hardware is registered when it is
  in the site, and the rest is unregistered. Information is registered when it is in θ, and the rest is fresh input
  (prompts, request inputs, network bits, what a wipe leaves). Verified compute V is the part of A proved to match D,
  at the public best's cost, prover overhead γ ≤ 10 allowed; **utilization** is the registered programs' MFU. Free
  compute is the rest: ε_undecl, ε_skip and ε_unreg. ε_undecl + ε_skip, its part on registered hardware, goes to 0.
- **Size** and **credit**: size is an AND-gate count, what C-Flock pays for and every result reports computation in.
  A protocol credits a size S: except with probability δ_coin + δ_crypto, an accepted run's right values cover at
  least (1 − ε_skip)·S AND gates (Size, credit and the slowdown lower bound, below).
- **Device model** (`Verity.Device`): S_chip bits on chip and M of device memory; per tick at most BW bits from memory,
  side-channel capacities and each op class's rate. A server's compute is its accelerators': its host CPUs are assumed
  to contribute none (**hosts carry no compute**, a named assumption). **Schedule** (`Accounting.Schedule`): its steps, each charging a
  **resource account** (`Accounting.Resource`): compute in AND gates (PoUW), bandwidth in bits read (PoUB), capacity in
  bit-ticks (PoUS). A **certificate** is how much of its resource an accepted audit's account used (`Run.ComputeSound`,
  `Run.BandwidthSound`, `Run.CapacitySound`); the **covert residual** what the limits leave beside it
  (`CovertResidual`); the **joint audit** all three on one device (`Accounting.Guarantees.Joint`).
- **Logical layer** and **physical layer**: what the cryptography proves with no device premise, and its reading on
  registered hardware down to spatial units, ending in the joint ledger (`Accounting.Guarantees.Fit`).
- **Φ**: a compute rate in gates per second, Φ_X for a location X. **M_s**: server s's memory over every tier, a public
  parameter; at each wipe a server keeps at most 5 GB (4×10^10 bits) plus its egress.

**Lean and assurance**

- **Guarantee**: a proved `Prop` in which named assumptions imply a conclusion, `def G : Prop` in a package's
  `Guarantees`, proved in `SecurityProofs.G`, its statement in the lock. A theorem is one when something relies on it as
  proved: a ledger row, a published table, a claim id in `verity.core.claims`, or code reading its result. A **lemma**
  is any other proved statement.
- **Protocol soundness**: the composed protocol's soundness guarantee, `Verity.Soundness`, which is not yet in Lean:
  when the verifier accepts, the monitoring period's claims hold except with probability δ. AGENTS.md's "Soundness is
  one theorem" is a requirement on it, not its name.
- **Assumption**: a named `Prop` taken as a hypothesis and proved by nothing, stated once in its area's
  `Definitions/<Area>/Assumptions` and listed in its lock. One about a primitive has a claim id such as `cr/sha-512`.
- **Fixed finder** and **fixed forger**: constructions from the prover's strategy that output a SHA-512 collision or an
  LMS forgery; `ecr/sha-512` and `euf-cma/lms-sha512-n64` apply only to them, and their success is the Break term.
- **Honest lemma** and **nontrivial lemma**: a guarantee's hypotheses hold at the honest deployed instance; the
  guarantee fails for a verifier that accepts everything.
- **Spec**: a Lean package's trusted text, its definitions, assumptions and guarantees (`layers` in `lean-audit.json`).
  **Lock**: its guarantee records in `lean-audit.json`. **Trusted computing base**: each spec and the code implementing
  it; the *assurance TCB* is Lean's kernel, the pinned Mathlib and ArkLib, the Lean check and `check`'s gates.

**Engineering**

- **Frontend** and **integration**: a frontend hands Verity a circuit (`verity_vllm.program.frontend`); an integration
  makes Verity work end to end with a real system, vLLM's the only one so far.
- **Configuration**: the only Verity artifact, assigning roles to machines (`verity run CONFIG --party ROLE`).
- **Serving** and **proving**: the live inference path and proof generation, whose costs are serving and proving
  overhead. **Approach**: a line of work with a hypothesis, an owner and a status (`research notes approaches`).
  **Project**, **milestone** and **issue** are Linear's words.
- **Slot**: one sequence of a served batch. **Wire**: an edge carrying its source gate's value. **Port** and
  **element**: a named array of fixed-width elements on a Definition's boundary. **Format**: a label on a port's
  elements (BF16, E4M3, token id), outside the digest. **Boolean Definition**: gates only; a word Definition's words
  read through `boolean.word_type`.
- **Subcircuit template** and **subcircuit**: `gemm_coordinate(K, semantics)`, and one with every parameter bound; the
  proof system proves subcircuits. **Input** and **input set**: one assignment to a subcircuit's inputs, and a batch.
  **Cell**: one benchmark result (proof system × statement × input set × hardware). **Input block** (`INPUT_UNIT`):
  every input gate, certified against its external anchor, never sampled.
- **Pinned**: an instruction whose silicon model matched every probe (`benchmarks/tc_probe/trust.py`); a pinned value is
  one fixed before the run. **Served MACs**: a census count, not size. **Binding unit**: one unit of circuit-check's
  cache. **Transformer sublayer**: attention or FFN with its residual add. **Value acquisition**: taking a value's bytes
  where vLLM computes it, by a hook or a native tap.
- **Transfer block**: the transfer gates a network certifier releases on one link, one direction, one bucket, proved as
  its own proof block (`IngressScanBucket`, `EgressBucket`). The code's link-window is a sampling block of 2T of them.
- **Landing**: a **ready mark** is an owner's statement that a head may land; a **land verdict** a reviewer's `land` or
  `hold`; **`next`** the branch drive stacks admitted PRs on; a **pass** a part of the check reused while its inputs are
  unchanged; a **pure move** a rename `move_check` certifies. **Attempt**: the evidence store's record of one run.
  **Pool**: a node's named run slots. **Node lease** and **GPU lease**: a node's deadline, and a booking of its GPUs.

## Notation

The one symbol table: every protocol states its guarantees in these symbols, one meaning each. A row names its Lean
declaration, or says "not yet in Lean". A gate count that credits or reports is in AND gates; the (c, k)-partitions
count every gate.

**Registration and circuits**

| Symbol | Meaning | Lean |
|---|---|---|
| R = (site, θ, B) | Registration, append-only: the site (the hardware, recursive), the registered information θ and the builder B. | `Verity.Site`, not yet in Lean; `Verity.Declaration` |
| C, \|C\|, C(x) | A circuit: gates in topological order. \|C\| counts its operation gates; C(x) is its output on fresh input x. | `Verity.Circuit`, `Circuit.computed`, `Circuit.size`, `Circuit.eval` |
| in(S), out(S) | The gates outside S wired into S; the gates of S read outside it, and its outputs. | `Circuit.into`, `Circuit.exits`; `Circuit.isInput`, `Circuit.outputs` |
| θ, x | The registered information's input gates, fixed before the period; the fresh input, x : in(C) ∖ θ → values. | `Verity.Declaration.θ`; x is the input part of `Circuit.eval`'s argument |
| a_j, E_j, 𝓘_j, loc_j | Declared advice; B's execution block for it, its partition into inference blocks, and its placement on the site's registered devices. | `Verity.Partition`; the rest not yet in Lean |
| D | The declared circuit, D = E_1 ‖ … ‖ E_n. | `Verity.Declaration.D` |
| (c, k), 𝒞_{c,k} | A partition's budgets per block: c gates exactly (padded), at most k fresh input bits; 𝒞_{c,k} the circuits with such a partition. | `Verity.Partition`, `Partition.gates`; `Partition.IsCK` not yet in Lean |
| A, x(g), t(g) | The actual circuit, each gate g at a device x(g) and a tick t(g). | `Accounting.Placed`, `Accounting.Placed.x`, `Accounting.Placed.t` |
| A[X, J], Φ_X | A's operation gates at location X during ticks J; X's capacity, \|A[X, J]\| ≤ Φ_X·\|J\|. Φ_s a server's, Φ_total the fleet's, M_s a server's memory. | `Accounting.Placed.region`, `Accounting.Placed.ComputeRate`, `Accounting.Placed.Memory` |
| V(X, J), U(X, J) | The verified compute at X during J, as gate occurrences; the free gates, \|A[X, J]\| − \|A[X, J] ∩ V(X, J)\|. | the share 1 − V/C: `Nci.Dampener.residualShare`; the sets not yet in Lean |
| s_× | The fewest gates of a counted operation: U free gates complete at most U/s_× of them. | not yet in Lean |
| monitoring period, tick | The one span of time, and the one clock's step; every other span is a block's. | `Verity.MonitoringPeriod`, `Verity.TickSpan`, not yet in Lean |

**The physical model**

| Symbol | Meaning | Lean |
|---|---|---|
| 𝓜, ρ | The ACCEL model: a period's devices, their resources and each resource's rate ρ at the firmware maximum (`verity/hardware/cards/<sku>.toml`). | `Verity.Accel.Machine`, `Verity.Accel.Card.ρ`, `Verity.Accel.H200.card`, `Verity.Accel.Sm120.card` |
| H, a, Λ | The fleet's fixed dynamics, its chosen initial state, and the bound on a: A is H unrolled from a. | `Accounting.Device`; `Verity.Device`, `FixedMachine` and `InitialBudget` not yet in Lean |
| k_exfil(Y), k_infil(Y) | Block Y's exfiltration capacity log2\|O_Y\| over its accepted observable behaviours; the information reaching Y that its fresh-input budget doesn't count. | not yet in Lean |
| HW, INV | Named assumptions: the hardware model (𝓜, links, side channels, what a wipe clears); the inventory's (coverage, unregistered hardware's bound, the devices' honesty, `trusted-host`). | `Accounting.Hardware`, `Accounting.Placed.Links`, `Accounting.Placed.Causal`; `Accounting.Covered`, `Accounting.TrustedUnits` |

**Sets, protocols and the guarantee**

| Symbol | Meaning | Lean |
|---|---|---|
| 𝓕 | The feasible set: the computations (C, x) the hardware can run under HW from a, unregistered hardware included. | `Accounting.Overhead.physClass`, a set being an `Accounting.Overhead.Class`, both renamed later |
| 𝒜_p, δ_p | Protocol p's admissible set and its failure bound: Pr[accept ∧ (A, x) ∉ 𝒜_p] ≤ δ_p. | `Accounting.Overhead.envelope`, renamed later; `Verity.Sound`; `Verity.Soundness`, not yet in Lean |
| 𝒜 | The admissible set, 𝓕 ∩ ⋂_p 𝒜_p, except with probability Σ_p δ_p and the cryptographic terms. | `Accounting.Overhead.restricted`, renamed later |
| τ, b | A prover strategy; a server block. | `Verity.Game.Strategy`; b as `Verity.Block` |
| δ_coin, δ_crypto | Bounds from the live coins; SHA-512's binding, knowledge soundness and pseudorandomness advantages. | `Verity.Sound`'s δ; `Security/Definitions/Hash/` |
| Break, B^τ, 𝒟_b^τ | A SHA-512 collision by the fixed finder B^τ, or an LMS forgery under a registered key, charged and never assumed away; 𝒟_b^τ the PRF distinguisher at server block b. | not yet in Lean |
| ε_Wipe,b | The streamed wipe's failure bound at server block b, 2^-128 + Adv_PRF. | `StreamedWipeMeetsServer5GB`, not yet in Lean |

**Training tasks, cost and slowdown**

| Symbol | Meaning | Lean |
|---|---|---|
| 𝒢, 𝒴 | The training task: one random input x and a success test C(x) ∈ 𝒴 at positions fixed before x. | `Accounting.Overhead.Task`, `Accounting.Overhead.Succeeds` |
| η, P_α, η_cov | A success probability; the properties a bound lower-bounds, each a necessary subcomputation; Pr[success ∧ ¬⋁_α P_α] ≤ η_cov. | not yet in Lean |
| fleet-time, cost(C) | The unit of cost: elapsed time times the whole fleet, registered and unregistered. | `Accounting.Overhead.cost`; `Nci.Dampener.hardwareMultiple` |
| T_𝒢(η; S) | The training cost of a set S: the least fleet-time at which some computation in S succeeds with probability above η. | `Accounting.Overhead.threshold`, renamed later; `Accounting.Overhead.success` |
| slowdown(𝒢, 𝒜, η) | T_𝒢(η; 𝒜)/T_𝒢(η; 𝓕), the same a and Λ on both sides. | `Accounting.Overhead.overhead`, renamed later |
| R_free, U, ν, α, ρ_G | The compute rate left to training, R_free ≈ (U + ν)Φ, U = ε_undecl + ε_skip + ε_unreg; slowdown(G) ≈ max(1, ρ_G/(α R_free)) for a run G needing rate ρ_G. | `Nci.Dampener.remoteDampener` in an older form; the rest not yet in Lean |

**Registered or not**: hardware, information and compute; each share states its denominator.

| Symbol | Meaning | Lean |
|---|---|---|
| registered hardware, L_unreg | Hardware in the site; unregistered hardware is the rest, its devices L_unreg. | not yet in Lean |
| verified compute, γ | V: served inference and the protocols' own work at the public best's cost; γ the prover overhead, ≤ 10. | its share V/C is 1 − `Nci.Dampener.residualShare` |
| ε_undecl | Registered hardware not running proved work, a share of its capacity C_X: idle pipes, host CPUs, the clock above the credited one, filler, the honest implementation's excess. | with ε_skip, `Nci.Dampener.residualShare` |
| ε_skip | Work the proofs could miss: PoUW's ε_W·(1 − ε_undecl), and the wrong certificates. | `Accounting.Compose` (ε_♭ = ε_skip + ε_cert + ε_route) |
| ε_unreg | Unregistered hardware's share of the fleet's capacity, Φ(L_unreg) = ε_unreg·Φ_total; mainline 10^-3. | `Nci.Dampener.unmonitoredShare`, renamed later |
| ε_break | The gates of the faulty proof blocks F. | `Nci.Guarantees.Dampener` (its χ) |

**Blocks, yields and bounding strategies**: server blocks are A's, inference blocks D's, and they meet in verified
compute; gates on unregistered hardware are in no block.

| Symbol | Meaning | Lean |
|---|---|---|
| server block S, c_S, k_S | A's gates on one registered server between two wipes: c_S = ⌊Φ_s·Δ_wipe⌋ (padded); k_S = k_wipe + (r_in + r_side)·Δ_wipe, what the wipe leaves plus the network bits entering. | `Verity.Block`, `Verity.Blocks`, `Verity.Wipe`; the sum not yet in Lean |
| inference block I, c, k | A block of some 𝓘_j; in verified compute each sits inside one server block. | 𝓘_j is a `Verity.Partition`, nested by `Verity.Partition.Refines` |
| N | The necessary subcomputation: N counted objects (products, products with partial sums, or outputs) every successful run computes. | not yet in Lean |
| pricing argument | How much of a block's fresh-input budget one counted object needs, which gives the yields (HeldRate, for products). | `Nci.Guarantees.ThreadedReads`, `Nci.Guarantees.ContainedTraining` |
| q_S, q(c, k) | The server-block and inference-block yields: the most counted objects one block can complete within its budgets. | not yet in Lean |
| straddle term | Counted objects split between verified and free compute in one block, bounded by the bits crossing. | not yet in Lean |
| T_min, T_ref | The least fleet-time that can fit N; the reference training algorithm's; the slowdown lower bound T_min/T_ref ≤ slowdown(𝒢, 𝒜, η). | not yet in Lean; the result `Nci.Guarantees.Dampener`, `Nci.Guarantees.DampenerRemote`, renamed later |

## Size, credit and the slowdown lower bound

- **Size** is an AND-gate count, in place of work, FLOPs, MAC-times and pricing; U/s_× replaces κ/2. The
  (c, k)-partitions and `Circuit.size` count every gate, so a report converts explicitly (`Circuit.andSize`; credit as
  `Verity.creditSize`, not yet in Lean). S is the AND count of D under one public lowering pinned at registration.
- **Credit** needs no algorithmic premise: SAK (`SAKResidual`) is a premise of the slowdown lower bound only. The served
  PoUW rests on `SharedNoiseLucks` until it moves to Dither's no-fold receipt. PoUB bounds no hidden work itself.
- **The slowdown lower bound** is the one place 𝓜's resources appear, since gates can't measure a device's capacity. It
  reports counted operations, each labelled with its AND count, never "ANDs of hidden work".
- **Not yet restated**: the device-model audits price work per op (`Accounting.Run.ComputeSound` at a `Price`), and
  Dither's `PROTOCOL.md` credits pub(μ°) against W_reg.

## Older names

Older names stay where a digest, a fixture, a recorded field or a pinned vector bakes them in. They mean:

| Old name, where it stays | Term |
|---|---|
| "Candidate A" to "Candidate V"; the `candidate` keys and values, `CandidateA`, `account_candidate_a` | the deleted proof systems A-GKR, B-Ligero, D-SP1 and a VOLE prover, never C-Flock |
| `Target`, `TARGETS`, `FIRST`; the fingerprint's `profile`; `Target.family`, `FAMILY_DIGEST` | subcircuit; the semantics parameter's gate set |
| "backend", "configured backend", "configuration" for C-Flock; `backends/`, `verity_flock.backend`, `drilldown.Variant` | the proof system, and a proof system with a statement |
| "IR", "Verity IR"; `verity.ir` | circuit, Program; `verity.ml.circuits` |
| "instance", "instance set"; the `bench-instances/*` datasets, `instances_*` keys and labels | input, input set |
| `verity_sp1.proofs.statement.Statement` and its "kind" | the SP1-era encoding of proof-block obligations; a kind is a subcircuit |
| "pin", "pinned theorem"; `pins` in an older `lean-audit.json` | guarantee |
| "law", "draw law"; `law` keys and the Lean's `Law` | sampling design |
| "draw", "draws" for the proof blocks picked; "sequential draw" | the sample; sequential sampling |
| "escape", "escape probability"; `*_escape`, `effEscape` | acceptance probability (`effEscape` is P̄_a) |
| "proof unit", and "unit" alone meaning one; `unitClasses`, `verity.ml.circuits.parts`, `wrong_units`, `proved_units`, `LinkWindows.proof_unit`, the `"proof-unit"` export kind, a `Spec`'s public-item reasons, the `vu_*` names | proof block |
| "wrong unit", "incorrect unit", "faulty unit"; the Lean's `wrong` | faulty proof block |
| "transcript", "committed transcript", "committed values" for a run's gate values; `ExecutionTranscript`, `evaluate_call`'s `transcript` | trace |
| "verifier of record" | the verifier; other checkers are the Rust cross-check, a Python mirror or a Python verifier |
| "δ_link"; `LinkSound`, `δlink` | ε_bind |
| "verification unit" (VU); `vu_*`, `vu-sample/v1`, `vllm-vu-set/v1`, `vu-k1536` | proof block (C-Flock's layout keeps VU: Narrower senses) |
| "M0"; "M1", "M2" | C-Flock's circuit prover (`verity_flock.circuit`); C-Flock ZK |
| "coin server"; `round_digest` strings in `Flock/Tags.lean` | the challenger's live-coin part |
| "network transparency"; `network-transparency/` paths | network accounting |
| "warden", "active warden", "network warden", "tap", "uplink tap"; the `warden` modules, `Kind.warden`, `service.WardenProfile`, the `verity-network-warden/*` tags, `replay_set=warden/re-verify` | the network certifier; a warden link is a network certifier link, and its records are network certificates |
| "remote bandwidth audit"; the bandwidth audit over PoUS's store as a subprotocol | proof of useful bandwidth (PoUB), the one bandwidth subprotocol; the bandwidth audit is the guarantee it applies |
| "timed verifier"; `verifier_cpu` | the memory challenger |
| "prover gateway", "gateway"; "verifier gateway"; "batch verifier"; "GPU workers" | the firewall; the challenger; the verifier; workers |
| "netlist"; `verity/flock-netlist/…` tags, `flock-netlist/v1` | a private circuit, or the circuit's description |
| "trusted text" in `layers` reasons | spec |
| "horizon", "accounting period"; `*Horizon*` and `*Period*` names (`HorizonSampling`, `ComposePeriod`), PoUB's `horizon` field | monitoring period; "pooled" for a reading pooled over several |
| "declaration block", "declaration period", "the declaration" for one stretch of declaring | execution block; "declaring" and "declared advice" stay |
| "execution period", "sampling period", "certification period", "Commit period"; `Verity.ExecutionPeriod`, `NetworkCheck.Period` | the block's span (E_j's, a sampling block's); monitoring period for time |
| R = (θ, D, 𝓘) for the registration; the programs and the proof system as registered | R = (site, θ, B); the subprotocols' Programs and the proof system are the protocol's own |
| π for a placement | loc |
| "commit phase", "challenge phase", "response phase" | commitment, challenge, proof and validation |
| "run" for one execution of a subprotocol; code's `session` in other senses (a release, a certifier's attachment, the gate session), until a format version | session; by sense, a release, an attachment or a challenge |
| "block" for a zone of the device graph; `datacenter.physical.Block`, `--blocks` | zone |
| "block", "thread block" for CUDA's launch unit; `blockIdx`, `blockDim` | CTA |
| "block" for a kernel's tile; `AttnBlock_v*`, `SparseMlaBlock_v1`, Triton's `BLOCK_*` | tile |
| "block" for a hash's or a stream's chunk; `Hm96.block`, `shaBlock`, `Randomness.block`, `blocks()` | chunk |
| "block" for PoUS's encoded store; `blocks_per_segment`, `block_bytes` | label |
| "block" for a store's or a fill's unit; `block-sha512`, `BLOCK_BYTES`, `Requirement.liveBlocks`, `Tag.Block` | chunk (of the store) |
| "block", "block-scaled" for entries under one scale; `verity-fp8-block128/v1`, `ScaledMmFp8Block_v1`, `weight_block_size` | scale group |
| "block" for a GEMM's inner range; `D1.nBlocks`, `ThreeBlock` | segment |
| "block", "KV block" for a page of vLLM's KV cache; `block_size`, `block_table` | KV page, page pool |
| "block" for a record's section; for a model's layer, `blocks.N`; for the certifier's row group, `CompileSliceBlocks_v1` | section; layer; row group |
| "compile unit"; "replay unit" (RU) | the compile slice's proof block; replay block |
| "novel communication intensity" (NCI); "isolation unit" (IU), "isolation block", "certified block"; (Γ, α); "baseline" for θ; "novel information", "novel reads", "unregistered input"; `Nci*`, `nci-*`, `ius`, the `nci` directories | the inference-only policy; inference block; (c, k); the registered information θ; fresh input |
| "leg"; "component" for a subprotocol of a composite; "integrity", "work", "space", "communication" naming subprotocols; the runner's legs, the composites' `components`, the verdict keys `integrity` and `communication` | subprotocol |
| "stage" outside "one-stage"; `research run --stage` | phase, or run |
| "row" for a server's window or a workload; vLLM's `row.py`, `RowConfig`, row ids, "row of record" | server block; workload, a workload of record |
| "work", "FLOPs", "MAC-times", "pricing" for how much a circuit computes; pub(μ°), μ°, W_reg, W_b; `CircuitWork*`, `Price`; "κ/2" | size, in AND gates; U/s_×. `WorkSound` keeps its name |
| "link-window"; `window`, `LinkWindow`, `LinkWindowAcceptMeans` | a sampling block of transfer blocks |
| "window" for a span of time; `Verity.Window`, `verity.ml.windows`; "Commit window" | monitoring period, or the block's span; tick span in Lean |
| "deploy", `verity deploy`, "deployment file" | `verity run CONFIG --party ROLE`; the configuration |
| "ACCEL" naming the protocol; `accel/`, `verity/pouw-accel/<kind>/v1`; "the accelerator model", "the machine model" | Dither; the ACCEL model |
| `ShaSpeed`, `ServiceSound` | `SAKSha`, `ProofSystemSound` |
| "overhead" for the training cost, "the overhead lower bound", "overhead calculator", "dampener", "compute dampener", ρ, "training efficiency", "hardware multiple", "the circuit-size slowdown"; `benchmarks/rho`, `Nci.Dampener`, `Accounting.Overhead.overhead`, until renamed | slowdown, in fleet-time; the slowdown lower bound. "Prover overhead" γ keeps its name, and ρ is 𝓜's rate |
| "GPU-hours" for the cost; `Accounting.Overhead.serverTime`; T_0, C_0, T_target | fleet-time; T_𝒢(η; 𝓕), which the reference training algorithm bounds |
| A_phys, A_i(S), δ_i(S), A′; "layer 1" to "layer 3"; "the one theorem" | 𝓕, 𝒜_p, δ_p, 𝒜; the feasible set, protocol soundness, the slowdown lower bound |
| ε_gap, `gap.*`; ε for one GPU, "the uncertified share" | ε_undecl (`undecl.*`); ε_undecl + ε_skip at that GPU |
| "physical class", "envelope" for 𝒜_p, "restricted", "monitored" or "training class", "faithfulness"; `physClass`, `envelope`, `restricted`, `Class`, `envelopes.py` | 𝓕; 𝒜_p; 𝒜; set |
| "server-time block", "serving block", "server-window", "physical block"; `ServingBlock` | server block |
| "server block" for one server's proof blocks; `blocks.Server`, `verity/blocks/v1`'s `server` keys | a server's proof blocks |
| "coverage", "materialization", "work requirement" for what every run must compute | necessary subcomputation |
| "exclusion bound"; "per-block limit", "per-block cap", "window cap", q_T(c, k); `Cap`, `exclusion.toml` | the slowdown lower bound; q_S, q(c, k) |
| "registered computation" | the declared circuit |
| "training class" for a premise set; "allowance", "tier", "route"; `classes.py`, `Allowance` | pricing argument |
| "reference construction", "reference run", "reference circuit" | the reference training algorithm |
| "dark", "unmonitored", "undeclared hardware", ε_dark, L_dark; `unmonitoredShare`, `eps_dark`; "monitored" | unregistered hardware, ε_unreg, L_unreg; registered hardware |
| "certified work", "credited work", "accounted compute", "declared compute", W_cert; `w_cert`; "declared utilization" | verified compute, V; utilization |
| "slack", "uncertified capacity", "unaccounted", "undeclared compute", "skippable work", "monitored residual" | free compute; U counts its gates |
| a for the protocol index; G for the outputs; θ for a card's numbers; D for Λ; D_b^τ; tick(g) | p; 𝒴 and 𝒢; the card table; Λ; 𝒟_b^τ; t(g) |
| "envelope ISA", "envelope op"; "coverage premise", "coverage table"; `cover`, `cover/` | the ACCEL ops; the card premise, the opcode table |
| "words" for Dither's committed sums; "staging" for a store's load | prefix sums; the store's load |

The `veritor.*` gate families are grandfathered and frozen: each id in `verity/ml/circuits/targets/target.py` keeps
its spelling, since pins, fixtures and runs bake it in, and no new family is created under `veritor.*`
(`verity/ml/circuits/tests/test_targets_frozen_families.py`).

## Narrower senses

- A protocol's **ε_skip**, in Dither's `PROTOCOL.md` and Credit, is its own skipped share, ε_W; the fleet's is
  ε_W(1 − ε_undecl) plus the wrong certificates.
- `verity.ml.circuits`'s `Circuit` is a Program's gate layout, its gates word-level (one `Value<w>` each), as are the
  Lean model's (`Verity.Circuit`), whose Boolean profile is width 1. A `GateSelection` is a set of its gates; a
  `typed-obligation/v0`'s `subcircuits` are its shared gate bodies.
- A **template instance** is one occurrence of a template in a program; in a claim id `property/instance`, the instance
  is the primitive. A **universal unit** (`UniversalUnit`) runs any private circuit of a size class; circuit hiding is
  recursion (`--hide`). A **transition unit** is one tensor-core accumulator step.
- C-Flock's statement layout (`verity/ml/flock/verifier/PROTOCOL.md` §16): a **unit copy** is one copy of the unit
  circuit; a **VU** one instance's share of the witness, laying out one sampled proof block; an **R1CS block** one
  diagonal block, 2^{k_log} witness bits holding `vus_per_block` VUs. **C-Flock's block size limit** is k_log ≤ 27
  (`MAX_K_LOG`). An R1CS block is never a proof block.
- A **pipeline stage** is one buffered step of a CTA's pipeline (`num_stages`); a **warp** is CUDA's 32 threads, a
  **lane** one of them; a **sliding window** is a model's attention span, not a period. In `integrations/vllm`,
  "family" and "kind" name Definitions and query families.
- Words with two senses in the code: **row**, a commitment row (a matrix's row stays plain English); **transcript**, the
  messages of a proof or a session (PoUS's transcript hash), `ExecutionTranscript` keeping the word until renamed;
  **stage**, to stage a statement is to fix it before any coin; **window**, the code's served windows
  (`verity.ml.windows`, PoUW's window and salt) until renamed; **escape**, the integrity profile's `escape` method, and
  vLLM's host escape (`E_OPAQUE_HOST`).
