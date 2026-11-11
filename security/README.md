# Security

The Lean statement and proof of what an accepted run of Verity guarantees: for each subprotocol, and then for the composed
protocol, a bound on the probability that the verifier accepts while a claim about the run fails. The terms are the
[Glossary](../GLOSSARY.md)'s. The first section gives the four statements to read first; the next is the whole security
argument, from what makes training hard to the slowdown a secret training run suffers, with its symbols and where each
step stands in Lean.

## Start here

Four statements carry what an accepted run guarantees. Each entry gives its file (under `security/`), Lean name and
one-line statement, and its premises by kind: **domain**, what the statement is about; **real**, a named cryptographic
or trust assumption; **folds into**, a second form of another premise; and **open**, a premise about our own code or
protocol that a proof has yet to discharge.

SPOT, the Sample-and-PrOve Transform, takes a proof system Π for one proof block to SPOT[Π], which proves a partitioned
computation by committing to every proof block and proving only the sampled ones; the deployed one is SPOT[Flock]. A
set B of proof blocks is accepted with probability P_a(B), the chance that the sample misses all of it, and the
**tolerated family** is M_δ = {B : P_a(B) > δ} (`Claims.Sampling.Design.M`, the Lean's "confidence set"). F is the
incorrect set of the trace the extractor recovers.

1. **SPOT.** `Claims.SamplingGeneric.CompilerKS` (`Definitions/Claims/SamplingGeneric.lean`, proved in
   `Proofs/Flock/SamplingGeneric.lean`): for every knowledge-sound Π and every compiled verifier meeting SPOT's contract
   with a binding commitment, at every δ ≥ 0, Pr[accept ∧ F ∉ M_δ] ≤ δ + ε_ks + ε_bind, with ε_ks = κ_δ − δ + √(γ·Pr[B])
   from Π's errors at the sampled proof blocks' sessions and ε_bind = Pr[B_ext]; `CompilerZK` carries Π's zero
   knowledge to SPOT[Π]. Their premises (`Arg.KS`, the binding, the scope, the contract) are their domain. At Flock,
   `Flock.Guarantees.SampledKSLiveV3` (stated in `Proofs/Flock/Bridge/ScopeGate.lean`, proved in
   `Proofs/Flock/Bridge/KSLive.lean`) proves the bound at the deployed sessions:
   Pr[`verify --zk` accepts ∧ the run's population is the registration's ∧ F ∉ M_δ] ≤ δ + ε̄_ks + ε_link, at
   δ = C(n−K₀, k)/C(n, k) for k of n proof blocks sampled uniformly, with ε̄_ks Flock's error averaged over the sessions
   and ε_link the link's term:
   - Domain: the circuit and public files parse, with no shared rows, and k ≤ n; `verify --zk` checks the scope, the
     tags and k's split over the tables (`Flock.KsScope.check`).
   - Real: SHA-512 collision resistance (`TableCRZAt` and `LinkCRZCAt` unfold to it) and `UniformRandomBytes` (the
     verifier's bytes, exactly uniform).
   - Folds into: `RecordCustodyZKJPLaw` (the record is live) into Core's challenger premises.
2. **The recursion.** `Claims.RecGeneric.CompilerKS` (`Definitions/Claims/RecursionGeneric.lean`, proved in
   `Proofs/Flock/RecGeneric/OuterProof.lean`): for every Π_in its repetition extractor shows knowledge sound, binding
   ZK server and knowledge-sound Π_out, the compiled verifier is knowledge sound wherever the compiler's contract holds;
   its premises are its domain. Deployed, `Flock.Guarantees.RecursiveKS` (`Proofs/Flock/Recursive/KSStatement.lean`):
   for every decoder within Flock's list-decoding radius and every dense statement, Pr[the recursive verifier accepts ∧
   the extractor recovers no witness] ≤ κ + √(γ·Pr[B]) plus the inner verifier's bad-abort, forked-coin and first-coin
   terms. `Flock.Guarantees.RecursiveKSDeployed` (`Proofs/Flock/Recursive/KSCoinTreeStatement.lean`) is that bound plus
   2·k(x)·δ₁ at the coins the challenger commits at Hello, in an hm96 tree whose leaves hide within δ₁.
   - Domain: the dense statements (`RecDeploy.DenseAt`); the decoders within the radius (`Claims.Flock.ListWithin`; the
     deployed ones meet it).
   - Real: SHA-512 collision resistance bounds Pr[B], and `Claims.Hm96Hiding` bounds δ₁.
   - Open: none.
3. **Zero knowledge of the deployed system.** `Flock.Guarantees.SessionLoopZKDeployed`
   (`Proofs/Flock/SessionLoop/Statement.lean`): at the ZK server's session loop on the deployed tables, for every t > 0,
   every V* with one more round than the verifier and every event E, Pr[E(real)] ≤ 2·(Pr[E(sim)] + 8(t+1)·n_L(x)·δ₁ +
   2·3^(−t) + 4t·3^t·Pr[fork collision]), with δ₁ hm96's hiding gap at the pinned key. For the recursive audit,
   `Flock.Guarantees.RecursiveZK` (`Proofs/Flock/Recursive/Statements.lean`): if the outer session is
   ε_o-zero-knowledge, every event of the view released at a stop from a public family St is at most |St| times as
   likely as at the simulator's, plus ε_o + n_h·δ_{K*}.
   - Domain: a witness in the relation, at a statement the loop serves, in Flock's scope.
   - Real: `Claims.Hm96Hiding` bounds δ₁ and δ_{K*}, and SHA-512 collision resistance the fork term; at the operating
     system's bytes, `UniformRandomBytesTest` (`recursiveZK_os_gap`).
   - Folds into: `Loop.Src` (the loop's inputs from its host are the tape's) into `trusted-host` and
     `UniformRandomBytesTest`.
   - Open: `SwCausalOn`, the tables' side's one unproved premise; per worker, `LoopInnerRound0`, `LiveRoundEq` and
     `FillRoundEq`; and `OuterViewZK`.
4. **Composed soundness.** `Verity.Soundness` and `Verity.RemoteSoundness` (`Properties/Verity/Soundness.lean`),
   proved as `Verity.soundness_of_subprotocols` and `Verity.remote_soundness_of_subprotocols`
   (`Proofs/Properties/Verity/Soundness.lean`): if each of the composite's subprotocols meets its target, then
   Pr[the composite's verifier accepts files that passed the commit phase's check ∧ some subprotocol's claim fails]
   ≤ Σ_a error_a, where error_a is a's δ plus ε_forge plus its loss at the probability that a's finder outputs a
   SHA-512 collision, and ∞ where the finder or the forger breaks its premise.
   - Real: SHA-512 collision resistance (`Vocab.FinderCR`) and LMS unforgeability (`Hash.Assumptions.LmsUnforgeable`),
     read in the error.
   - Open: the targets (`SpotSound`, `PouwSound`, `InferenceOnlySound`, `PousSound`, `BandwidthAuditSound`,
     `NetworkCertifierSound`, `WipeSound`; remote's are the first three), each a hypothesis until proved
     (PoUW's: `verity/protocols/pouw/verifier/`).

**The real assumptions.** Items 1 to 3 rest on three, each under the lock's `open` until `Definitions/Assumptions.lean`
declares it (The lock): `Claims.ExpectedCR` (SHA-512 collision resistance, which the other SHA-512 shapes
fold into), `Claims.Hm96Hiding` (hm96 leaves hide at the pinned key) and `UniformRandomBytesTest` (the operating
system's bytes are uniform up to η on each test). Beside them the statements read `trusted-host`, Core's challenger
premises (`ChallengerLogsCoins`, `FlipsAfterRecorded`, `VerifierReadsLog`) and `Hash.Assumptions.LmsUnforgeable`.

## The security argument

**Start here: training's bound of record.** Two theorems state it, both proved; (H) and (E) are the first one's named
assumptions.

- **The min-cut theorem**, `Nci.Cut.Guarantees.HE` and `HEShared` (`Definitions/Nci/Guarantees/Cut.lean`, proved in
  `Proofs/Nci/Cut.lean`): under (H) and (E), every partition of a circuit into m blocks, each reading at most k gates
  outside θ, has N ≤ m·(k + α) + (k/2)·√(m·O/(β_w·β_x)), α = ½·√(k³/(β_w·β_x·b)), for the N MACs and O output wires
  of its charged matmuls, so ρ = m·c/|C| is bounded below at every c; `HEShared` is it for matmuls sharing arrays,
  each term times F = √(2(1 − 1/ω)), and `CountHE` is it at the count's verdict, `Nci.CompliesAtMost c k m`.
- **(H), hardness**, `Nci.Cut.Assumptions.Hardness` (`Definitions/Assumptions.lean`), conjectured: every set S of words
  with κ(S) ≤ k has Σ_r β_r·|val(S ∩ A_r)| ≤ κ(S), so words behind a cut of κ gates carry at most κ in values, a value
  of array r costing β_r.
- **(E), exports**, `Nci.Cut.Assumptions.Exports` (`Definitions/Assumptions.lean`), conjectured: every set B of
  computed gates has κ(B → E(B)) ≥ b·|E(B)|, E(B) the outputs a MAC inside B feeds that have no wire in B.
- **The counting bound over kinds of work**, `Nci.Formalism.Guarantees.Counting`
  (`Definitions/Nci/Guarantees/Formalism.lean`, proved in `Proofs/Nci/Formalism/Slowdown.lean`): every partition into
  inference blocks has ρ ≥ max(max_K (|B_K|/|C|)/(cap_K/c), max_J Λ_J), for kinds of work B_K, each with a cap cap_K on
  the gates of it an inference block holds, and kinds J each bounded by Λ_J over the whole partition; `Slowdown` beside
  it takes the token kind's cap from the named assumption `training-token-work`.

The rest of this section is the argument they are read in, from what makes training hard to the slowdown.

This is a conditional argument, not yet a proved theorem. It separates a learning hypothesis from enforcement.
Coverage says that sufficiently cheap successful methods require a minimum weighted count of analysed computation.
Local description and dependency arguments bound how much of that count an inference block can contain: the block must
import the required information or reconstruct it within its gate budget. Protocol accounting then bounds the
contribution of correctly executed inference blocks and charges every remaining contribution separately. Comparing
their combined capacity with the required count gives the slowdown bound. Verification establishes execution, not
uselessness for learning: the local bounds are what limit how much verified execution can help with the target. The
question it rests on:

> What property of a capability-acquiring computation forces it to integrate information that can't be cheaply
> regenerated within small computational compartments that receive little input?

The idea: training repeatedly uses information accumulated since registration. An inference block can do the same only
as far as it can import or reconstruct that information within its two budgets, c gates and k fresh bits. The
restriction bites when useful computation repeatedly depends on more updated information than one block can cheaply
reconstruct, and successful alternative methods can't cheaply avoid that dependence. *Repeatedly* matters: a large
update that is hard to reconstruct but needed once is imported once, at a cost that doesn't grow with training.

The argument has four parts:

1. **Reaching the target requires hard-to-localize computation**: coverage of efficient methods.
2. **Small inference blocks can do little of it**: a local bound from an information premise (primarily update
   hardness), a structural argument (primarily the upstream dependency certificate), Lemma 1 and Theorem 3.
3. **The protocol forces most physical compute into those blocks**: the device model, protocol soundness and the
   accounting.
4. **So reaching the target takes more fleet-time**: the slowdown bound.

They meet in one inequality:

$$ \underbrace{\Omega_*}_{\text{required for covered success}} \;\le\;
   \underbrace{\sum_{I \in \mathcal I_{\mathrm{good}}} q_I}_{\text{possible inside the good blocks}} \;+\;
   \underbrace{Q_{\mathrm{rem}}}_{\text{everything else}} $$

Ω_* comes from part 1, each q_I from part 2 and Q_rem from part 3; this inequality is the canonical statement, and
part 4's slowdown is a consequence of it, not a further source of difficulty. All three terms are one **weighted
count** in one convention: the product arguments count weighted multiplications, the recovery argument counts output
observations, and an application of the argument fixes one convention for Ω_*, every q_I and Q_rem.

The numbers come from the slowdown calculator (`verity/protocols/inference_only/rho`), not from this file.

### Objects

- **Time.** A **tick** is one step of the one clock (`OneClock`, a premise stated in prose, not yet in Lean). A **monitoring period** J is a span
  of consecutive ticks, |J| long; every other span is a block's.
- **Hardware.** Leaves L are devices; a location X is a set of leaves. The **site** is the registered hardware, public;
  hardware outside it is unregistered. Φ_X is X's **capacity**, work a tick at its rated peak (the site entry: SKU,
  clock, rate per operand class). Φ_R and Φ_U are the registered and unregistered capacities, Φ = Φ_R + Φ_U.
- **The actual circuit** A: every gate the hardware computed, hidden work included, each gate g at a leaf x(g) and a
  tick t(g); A[X, I] is its gates at X during the ticks I. The registered compiler's own runs, the developer's preparation and
  anything else the hardware does are gates of A.
- **Initial information**: the registered information θ, input gates whose values are committed before J (models, code,
  data). Nothing else is carried into J.
- **Registration**, once and append-only, in the setup phase: the site, θ and the registered compiler B, a public
  program: the log the Site is folded from (`Verity.Registration`, today with the site's entry only). Today's Lean
  models R = (θ, D, 𝓘), the registered circuit and its inference partition, as the committed record
  (`Verity.Committed`), which B's execution blocks extend.
- **Execution.** For each advice a_j (a string of at most ℓ_a bits, private and committed, ℓ_a public), B
  outputs B(a_j) = (E_j, 𝒫_j, 𝓘_j, loc_j):
  - E_j, the execution block: a Boolean circuit whose inputs are θ, earlier blocks' outputs and fresh input;
  - 𝒫_j, a partition of E_j's gates into proof blocks, each block's outputs within its public size class and their
    number within a public cap;
  - 𝓘_j, a partition of E_j's gates into inference blocks, each of at most c gates reading at most k bits of fresh
    input, which is anything outside θ;
  - loc_j : gates(E_j) → registered devices × ticks, the placement.

  a_j, E_j, 𝒫_j, 𝓘_j and loc_j are private, their padded counts public. The registered circuit is D = E_1 ‖ … ‖ E_n.
  Since E_j's structure is a function of a_j, the advice is fresh information: an inference block whose structure
  depends on k_a bits of a_j reads them as fresh input.
- **Derived quantities**, none assumed:
  - F, the incorrect proof blocks (committed outputs that differ from their gates applied to their committed inputs), and
    the **faulty share** φ, the share of credited work that lies in F;
  - Φ_cred, the rate PoUW credits, and the **credited utilization** u = Φ_cred/Φ_R;
  - Φ_gap, credited work a cheaper route could produce; Φ_bk, bookkeeping (certificate slack and verdict lag);
  - Φ_D, the rate of the registered circuit's correct credited work;
  - q_I, the most of the weighted count inference block I computes whole (part 2); ζ̄ and the counts below use the same
    convention.
- **Updated state.** The trained model's parameter slots i, each with its eligible versions (one per optimizer step
  that writes it) and a **price** p_i in bits: what a joint description of it costs, not its storage width.
- **Counted multiplication**: a product of a version of an updated slot and the operand of a **use** j (one token
  position's input to one layer at one training step, a forward value or a gradient). Each use's operand is computed
  by its own **upstream computation** T_j.
- **Local description length** LDL_{t,R}(y): the length of a shortest program from which a fixed decoder, given the
  registration R, outputs y within t steps.

### The protocol

Remote, for each execution block E_j, in five steps:

1. **Commitment.** The developer commits E_j's values under hiding leaves (every value crossing a proof-block boundary,
   its inputs and outputs), a_j, 𝒫_j and 𝓘_j, before any of E_j's coins.
2. **Challenge.** The challenger, on an auditor host, flips live coins: the sample of proof blocks (by size class) and
   PoUW's noise and check coins.
3. **Proof.** The developer proves each sampled proof block in zero knowledge and answers PoUW's challenges in time.
4. **Validation.** The verifier checks the proofs, the partitions' public rules, 𝓘_j's budgets (c, k) and PoUW's
   credit.
5. **Accounting.** Each subprotocol's account for E_j is tallied: the bound on incorrect proof blocks, the credited work,
   the inference blocks. The policies read the accounts.

The monitoring period is accepted when every subprotocol accepts every E_j, under one commitment that binds them all.
Acceptance alone doesn't require the developer to keep its registered hardware busy: a bound on ρ, which holds for
every admissible run, needs a **utilization policy** in the accounting step, refusing a monitoring period whose
credited utilization u is below a public floor u_min. Without it, the bound below holds only for the accepted run's own
Φ_D.

Onsite adds four subprotocols on devices at the site: memory wipes cut each registered server's life into server
blocks, each carrying at most k_wipe bits across its boundary; network certification bounds each server's ingress to
r_in bits a tick, besides side channels at r_side; and PoUS and PoUB bound the memory and bandwidth used.

### Part 1: reaching the target requires hard-to-localize computation

The setting is fixed first, without assuming the result: the **target** of the task 𝒢 (a loss below every registered
model's, reached within J from θ, with probability above η), the registered baseline, the initial information θ, the
arithmetic, and the range T_max of computation the claim rules out. Information specific to a candidate can't enter
through preparation that isn't charged. The setting does not say that θ can't produce an acceptable model cheaply: that
is the question.

**Coverage of efficient methods** (`Nci.Assumptions.CoversEfficientMethods`). Let Core(A, x) say that the
computation meets the premises of one of part 2's assemblies at the block budgets and carries at least its minimum
weighted total Ω_*. Uniformly over the programs, the initial information and the experiment,

$$ \Pr\big[\,A(x) \text{ reaches the target} \;\wedge\; |A| < T_{\max} \;\wedge\; \neg\,\mathrm{Core}(A, x)\,\big]
   \;\le\; \eta. $$

Each method may have its own covered structure (products, accumulations, recoveries); none has to reproduce a reference
circuit. Coverage doesn't say that slow methods of another structure are impossible, only that methods cheap enough to matter
can't escape every analysed structure. The question it answers is a trade-off: as a method reduces the updated state or
dependence the policy restricts, how much more computation does it need for the same target? That full-parameter
training is the unrestricted optimum isn't enough: if a compact-state method reaches the target at twice the cost and
the policy barely slows it, the slowdown is at most two. A method that reaches the target cheaply with a compact update,
frozen features, another learning procedure or another representation, escaping every lower bound that applies, refutes
it at that target and operating point. Small-state and alternative-precision training experiments test it. It is the
largest scientific uncertainty here; a first result states it for an explicit implementation family.

The earlier work floor (every computation reaching the target does at least M_ref/α of a reference's counted
multiplications) is coverage for one structure: a candidate sufficient condition, not the explanation.

### Part 2: small inference blocks can do little of it

A block can **import** information, spending its k fresh bits, or **reconstruct** it, spending its c gates. A local
bound has three ingredients, each with its own premises: **information** (which jointly selected values are expensive
to reconstruct), **structure** (how much counted computation the held values support) and **counting** (from a
block yield to the circuit). Coverage (part 1) then says which learning methods the bound constrains. The Lean is
`Nci.Capability` (vocabulary), `Nci.Assumptions` (premises) and `Nci.Guarantees` (the repertoire), one module per role;
`security/Definitions/Nci/Capability/README.md` surveys every premise, structure and bound in one ontology, with how to
test each. Improving the structure never strengthens an information premise; a stronger premise is a new hypothesis,
named as one.

**Reconstruction, proved** (`Verity.Guarantees.BlockDescription`, Lemma 1). In a registration R with B inference blocks
of at most c gates and k fresh bits each, if a fixed evaluator of a_blk bits extracts y_I from block I within t steps,

$$ \mathrm{LDL}_{t,R}(y_I) \;\le\; k' := k + \lceil \log_2 B \rceil + a_{\mathrm{blk}}. $$

#### Information: what a block can hold

An **item family** X (`Nci.Capability.Items`) lists things a block may hold, each with eligible **cells** (the gates
holding one version of its value) and a **price** p_i in bits. One premise schema covers every family:

$$ \textbf{description hardness:}\quad \mathrm{LDL}_{t,R}(x_U) \;\ge\; \sum_{i \in U} p_i \;-\; \lambda
   \quad\text{for every selection } U \text{ of } X\text{'s items, one cell each} $$

(`Nci.Assumptions.DescriptionHard`). It covers every selection a block can make, charges an item once however often it
is reused, and is time-bounded: replaying training counts only within t. With Lemma 1 it gives the **held budget**
(`Nci.Guarantees.HeldBudget`): every block holds a selection priced at most h = k' + λ, whether it loads, decompresses
or computes it. A second measure makes a different claim: **min-entropy hardness**
(`Nci.Assumptions.MinEntropyHard`) says the values are unpredictable under the run's randomness, not slow to compute;
it gives the budget h = k + λ with no decoder or time budget (`Nci.Guarantees.StatHeldBudget`), is tested by retraining
with fresh randomness, and fails when training is a deterministic function of registered information. Either measure's
meaning is its family's:

| Family | Premise | Fails where |
|---|---|---|
| updated parameter slots, a version each | update hardness | the update has a short description (low rank, a seed), the prefix replays within t, or the slots are predictable from R |
| operand occurrences, one scalar each | activation hardness | a block cheaply generates operands from what it holds; several uniform rates are refuted by exact decoders |
| both, priced together | joint hardness | the same bits describe both families; an additive budget is a hypothesis of its own |

**Recovery** derives an information bound rather than assuming one. A source z of length ℓ_z with
LDL_{τ,R}(z) ≥ ℓ_z − δ (`Nci.Assumptions.SourceHard`), recoverable from a block's program and a complement of at most
ℓ_z − β_v·|𝒥_I| + amb bits (`Nci.Assumptions.Recovers`, amb the index among at most 2^amb candidates), gives the
**recovered budget** (`Nci.Guarantees.RecoveredBudget`)

$$ \beta_v\,|\mathcal J_I| \;\le\; k' + \delta + \mathrm{amb}; $$

with amb ≤ ε·|𝒥_I| and ε < β_v, |𝒥_I| ≤ (k' + δ)/(β_v − ε): ambiguity lowers the information per output. Correlated
outputs need not shorten the complement one coordinate each (for an exact linear map the rank counts), a bound on the
candidates is not a decoder within τ, and δ is paid in every block's bound.

The smallest sufficient premise comes first: update hardness with a dependency certificate; an activation price only
where it survives joint reconstruction attacks; a joint price only where the additive cost is justified; recovery to
derive rates rather than assume them. No single freshness rate explains everything: an intermediate computable from R,
one computable from held updated state, one needing a large upstream reconstruction and one carrying recoverable source
information are different situations even when a codec gives them the same length.

#### Structure: how much the held values support

**Dependency** (`Nci.Capability.Consumers`, `Nci.Capability.Consumers.Upstream`): a **consumer** is something a block
finishes, such as a use j (one token's input to one layer at one step, `Consumers.ofUses`) or an output accumulating many
products (a weight's gradient over tokens, `Consumers.ofOutputs`). It has an upstream region A_j holding no registered
gate, and required gates G_j ⊆ A_j, at least g of them. The **upstream certificate**, checkable on a named circuit, says
every required gate reaches, inside A_j, a gate the consumer's finishing gates read. **Overlap weights** ω_j ≥ 0
(`Nci.Capability.Consumers.OverlapWeights`) sum to at most 1 over the regions holding any one gate: disjoint regions take
1 each, overlapping ones less. Then:

- **import or recompute** (`Nci.Guarantees.ImportOrRecompute`): each served use either spends an imported gate of its
  region or contains G_j, so the served weight is at most k plus the weight of the recomputed uses;
- **recomputing costs gates** (`Nci.Guarantees.RecomputeByGates`): the recomputed weight is at most c/g;
- **recomputing costs updated state** (`Nci.Guarantees.RecomputeByState`): if every G_j needs updated state priced above
  the held budget, nothing is recomputed.

These hold for the counted uses with an upstream certificate: for those, the block must import the specified context or
perform the certified upstream computation. The held budget is not a storage bound; it is what the description premise
allows. Together, with ω_I the weight of the uses block I serves,

$$ \omega_I \;\le\; k + \frac{c}{g}, \qquad\text{and}\qquad \omega_I \;\le\; k
   \ \text{ when reconstructing the certified upstream computation would expose updated items priced above the held budget.} $$

The bound degrades with the gate budget instead of failing at a threshold: importing spends fresh bits, reconstructing
spends gates. The g gates are the analysed implementation's certificate, not a lower bound on every algorithm computing
the same function, and products split between blocks still need their own account. The weights replace "every
activation is hard" with "enough structurally supported computation": overlapping regions get low weights, and choosing
ω is a linear program. These caps count the uses a block serves, not its products; over counted products the bound of
record is the min-cut theorem (Start here).

#### Counting and the standard assemblies

Theorem 3 (`Verity.Guarantees.Counting`, proved) turns a block yield q on weighted counted products into
|C| ≥ c·Ω/q, Ω the computation's weighted total; coverage must then bound Ω, not the raw product count. The assemblies
used so far:

| Information | Structure | Block yield |
|---|---|---|
| update hardness | dependency, recomputing priced out | k, in served weight |
| none | dependency, recomputing by gates | k + c/g, in served weight |
| a recoverable hard source | outputs | (k' + δ + amb)/β_v |

An assembly counts only when its premises hold for the same blocks, and a refuted premise contributes nothing. The
smallest valid cap wins only among caps on the same count: a served-weight cap and a recovery-output cap bound different
counts, so they are compared only after conversion to one convention, or applied separately, each with its own coverage
statement.

**The budgets are fixed by the execution's boundaries, not chosen per data type.** Unregistered values crossing a
block's boundary count by the bits they represent; a compressed code counts as the code, and decoding it spends gates;
values reconstructed inside the block spend gates. For decoding, whether cached keys and values cost fresh bits or gates
is a question of the cheapest admissible placement and reconstruction, not of which data the argument elects to
charge. The threaded read costs (`Nci.Threading.Threaded`, `Nci.Assumptions.WeightsIncompressible`), under which
`Nci.Guarantees.ThreadedReads` proves q(c, k) = (k/β)·max(a_out, w_out·k/(4·a_in·b_w)), assume read costs rather than
derive them; they are a candidate sufficient condition, not part of the repertoire.

### Part 3: the protocol forces most physical compute into those blocks

The protocol isn't what makes training hard; it makes the hardness matter. Three obligations, separate from parts 1
and 2:

**The device model**, the feasible set 𝓕: compute |A[X, I]| ≤ Φ_X·|I| for every location X and span I; hosts carry no
compute (a named assumption; without it host capacity joins Φ_U); links, memory and causality, which onsite reads; and
the inventory, Φ_U/Φ at most a stated share, an assumption about the world, since no registration covers unregistered
hardware.

**Protocol soundness.** Except on an event of probability at most δ = Σ_p δ_p plus the cryptographic terms, jointly with
acceptance, the run lies in 𝒜 = 𝓕 ∩ ⋂_p 𝒜_p (`Verity.Soundness`; the subprotocols' guarantees are the table under "The
guarantees to read first"):

- **sampled proofs:** F lies in the confidence set M_δ, which bounds the incorrect proof blocks in each size class, and
  with each class's block size the faulty share φ;
- **PoUW:** Φ_D ≥ (1 − φ)·Φ_cred − Φ_gap − Φ_bk, with each credited gate a distinct gate of A;
- **the inference-only policy:** every 𝓘_j is a (c, k)-partition;
- **one registration** across the subprotocols;
- **the utilization policy**, once it exists: u ≥ u_min;
- **onsite:** each server block, of at most c_S gates, reads at most k_S = k_wipe + (r_in + r_side)·Δ_wipe fresh bits.

**The accounting**, deterministic. Every counted multiplication of a successful computation inside A goes to exactly
one of two parts:

- an inference block I of the registered circuit's correct credited work whose effective fresh input k_I + k_{F,I} + k_{a,I}
  (its own fresh input, the bits incorrect proof blocks inject into it, and the advice bits its structure depends on) is
  within the range of part 2's assembly, contributing q_I;
- the remainder: unregistered hardware, registered hardware not running credited work, the incorrect proof
  blocks, the credit gap, the bookkeeping, blocks out of range, and multiplications split between a block and the rest.

Credit isn't counted twice, and padding isn't evidence that hardware ran anything. The remainder does at most ζ̄
counted multiplications per unit of its fleet-time, so Q_rem ≤ ζ̄·(Φ − Φ_D)·|J| (remote). ζ̄ is a justified ceiling,
not a reference's achieved rate: a reference execution shows what is feasible, not what is optimal. **Reference
efficiency**, ζ̄ = ζ_ref with ζ_ref = Ω_ref/C_ref the reference's weighted count over its fleet-time, is a hypothesis;
at peak, ζ̄ = ζ_ref/MFU₀.

### Part 4: so reaching the target takes more fleet-time

The **slowdown** ρ(𝒢, η) is T_𝒢(η; 𝒜)/T_𝒢(η; 𝓕), T_𝒢(η; ·) the least fleet-time at which some computation in the set
reaches the target with probability above η, and **fleet-time** elapsed time times Φ, registered and unregistered.

**Remote.** A covered successful computation carries a weighted count of at least Ω_* (part 1; with every weight 1,
the plain count). The good blocks do at most Σ_I q_I (part 2), and the remainder at most Q_rem (part 3). Replacing the
sum by (q/c)·Φ_D·|J| needs the accounting to show that the qualifying blocks have exactly c gates and correspond one to
one to credited actual computation; until that is proved, Σ_I q_I is the safer form. With that conversion and a uniform
cap q, and f = 1 − Φ_D/Φ the **free share**: the part of the whole fleet's capacity, unregistered hardware included,
not doing correct credited work of the registered circuit (1 − f is the part that does),

$$ \Omega_* \;\le\; \Big(\frac{q}{c}\,\Phi_D + \bar\zeta\,(\Phi - \Phi_D)\Big)\,|J|
   \quad\Longrightarrow\quad
   \Phi\,|J| \;\ge\; \frac{\Omega_*}{\bar\zeta\,f + (q/c)\,(1 - f)}. $$

The denominator is measured: C_ref is the fleet-time at which the reference training algorithm reaches the target with
probability above η, which takes a seed sweep, not one run. It stands in for T_𝒢(η; 𝓕), which it bounds from above,
so the bound is relative to that reference. If f is bounded uniformly over 𝒜 (the utilization policy bounds it),

$$ \rho \;\ge\; \frac{\Omega_*}{C_{\mathrm{ref}}\,\big[\bar\zeta\,f + (q/c)\,(1 - f)\big]}, $$

and the bound holds except with probability η + δ: an accepted run reaches the target in less fleet-time only if it
lacks the covered structure, which coverage allows below T_max with probability at most η
(`Nci.Guarantees.CapabilityFloor`), or lies outside 𝒜, which protocol soundness allows with probability at most δ
(part 3). Let Ω_ref be the reference's count in the same convention and ζ_ref = Ω_ref/C_ref. Under the required-count
floor Ω_* ≥ Ω_ref/α,

$$ \rho \;\ge\; \frac{1}{\alpha\,\big[(\bar\zeta/\zeta_{\mathrm{ref}})\,f + y\big]}, \qquad
   y = \frac{q}{c\,\zeta_{\mathrm{ref}}}\,(1 - f), $$

y the yield of correct credited work, as a share of the reference's rate. The ratio ζ̄/ζ_ref keeps the reference's achieved rate apart from the ceiling on every
residual computation; only reference efficiency (ζ̄ = ζ_ref) reduces the bound to 1/(α·[f + y]). The unregistered share
s ≤ f limits what this bound can certify, to at most 1/(α·(ζ̄/ζ_ref)·s); without a matching construction it says
nothing about the actual slowdown. Without part 2, q/c could reach ζ̄ and
the bound falls to Ω_*/(ζ̄·C_ref), no slowdown at all: part 2 is what keeps the registered circuit's credited work from being training.

**Onsite.** As remote, except that each registered server's remainder is capped per server block: a server block of at
most c_S gates whose effective fresh input k_S is within the assembly's range does at most q_S of the weighted count,
its inference blocks and its remainder together. When k_S is large enough for a server block to hold a token's whole
path, the cap is void, and the available onsite bound reduces to the remote bound.

### What the argument doesn't assume

None of these is assumed silently; each would restate the conclusion:

- that a large final model is hard to compartmentalize: large state can have a short generator, or be used without
  repeated expensive crossings;
- that every activation carries β_x new bits: some are cheaply rebuilt from R and the tokens, others share information
  with held weights or upstream values. Activation hardness states an activation bound as a named hypothesis, and the
  dependency assemblies need none;
- that weights and activations pay independent additive costs: they can describe one another or come from the same
  small input. Joint hardness states it as a named hypothesis;
- that because a whole active path doesn't fit, the local bound holds: that rules out one strategy, not recomputation,
  partial paths, compression or other representations;
- that the reference's achieved rate is a ceiling on every method;
- that nothing derives the capability within the claimed time.

### Where each step stands

| Step | Lean | Status |
|---|---|---|
| The actual circuit | `Accounting.Placed`, `Accounting.Placed.region` | Defined |
| Registered objects | `Verity.Circuit`, `Verity.Partition`, `Verity.Partition.IsCK`, `Verity.RegisteredCircuit`, `Verity.Committed`; the site and its registration log, `Verity.Site` and `Verity.Registration`; B, a_j and loc_j not yet in Lean | Partly defined |
| Coverage | `Nci.Assumptions.CoversEfficientMethods` | A named assumption in the lock; Core not yet read off a run's circuit; untested |
| Lemma 1 | `Verity.Guarantees.BlockDescription` | Proved |
| Information premises | `Nci.Assumptions.DescriptionHard`, `Nci.Assumptions.MinEntropyHard`, `Nci.Assumptions.SourceHard`, `Nci.Assumptions.Recovers`, `Nci.Capability.Items` | Named assumptions in the lock, taken by the repertoire's proofs; no price measured at a decoder budget; several uniform activation rates refuted by exact decoders |
| Structure | `Nci.Capability.Consumers`, `Nci.Capability.Consumers.Upstream`, `Nci.Capability.Consumers.OverlapWeights` | Defined; the certificate not yet checked on a named training circuit |
| The repertoire | `Nci.Guarantees.HeldBudget`, `Nci.Guarantees.StatHeldBudget`, `Nci.Guarantees.RecoveredBudget`, `Nci.Guarantees.ImportOrRecompute`, `Nci.Guarantees.RecomputeByGates`, `Nci.Guarantees.RecomputeByState` | Proved (`Proofs/Nci/Capability/`), over premises not yet tested |
| Theorem 3 | `Verity.Guarantees.Counting` | Proved |
| The threaded refinement | `Nci.Threading.Threaded`, `Nci.Assumptions.WeightsIncompressible`, `Nci.Guarantees.ThreadedReads` | Proved for the circuit in hand under assumed read costs |
| Device model | `Accounting.Placed.ComputeRate`, `Accounting.Placed.Links`, `Accounting.Placed.Memory`, `Accounting.Placed.Causal`, `Accounting.Slowdown.physClass`; hosts carry no compute, `OneClock` and the inventory not yet in Lean | Defined; three premises unstated |
| The composites | `Core.Remote.protocol`, `Core.Onsite.protocol`, `Properties.Composition.AcceptMeans` | Proved for the product rule `check`, which both run |
| The utilization policy | not yet in Lean | Not a policy yet |
| Protocol soundness | `Accounting.Slowdown.restricted`; `Verity.soundness_of_subprotocols`, `Verity.remote_soundness_of_subprotocols` | Proved from each subprotocol's target, each a hypothesis until discharged (`Properties/Verity/README.md`) |
| Sampled proofs | `Flock.Guarantees.SampledKSLiveV3`, `Claims.Sampling.RelS` | Proved; the step from F ∈ M_δ to the credit-weighted φ is open |
| PoUW | `Accounting.Guarantees.Compose.CircuitCredit` | Proved; nothing derives Φ_D from it, or shows each credited gate distinct |
| The inference-only policy | `NciCheck.Properties.SecurityProofs.DescWitnesses`, `Nci.CompliesRate`, `Nci.Complies` | Proved at its rate, with each block within (c, k) on the unrolled check's verdict (the lemma `NciCheck.Properties.SecurityProofs.refines`); the step to a (c, k)-partition of the registered circuit itself is open, and k_F (what incorrect proof blocks inject) and k_a (the advice's share) are unbounded |
| Onsite's k_S | `Pous.Guarantees.StreamedWipe`, `NetTiming.Guarantees.PeriodLeakAccepted` | Proved per window; nothing composes them into k_S |
| Accounting | the assignment of a run's counted multiplications not yet in Lean | Open |
| Rate ceiling ζ̄ | the inline hypotheses of `Nci.Guarantees.DampenerRemote`; a named Prop not yet in Lean | Open |
| The floor | `Nci.Guarantees.CapabilityFloor` | Proved, under coverage |
| Slowdown | `Accounting.Slowdown.slowdown`, `Accounting.Slowdown.threshold`, `Accounting.Slowdown.cost` | Defined, with cost in fleet-time; no theorem reads it |
| The bounds | `Nci.Guarantees.DampenerRemote`, `Nci.Guarantees.DampenerThreaded` | Proved as arithmetic over inline hypotheses, not over the definition above |

## Three trees

- `Definitions/` (package `security`, its root `Definitions.lean`): the spec, the part a reader reads. Each area
  (`Core`, `Pouw`, `Pous`, `NetworkCertifier`, `Accounting`, ...) has its vocabulary, its premises (`Assumptions`) and
  its guarantee statements (`Guarantees`), each a `def G : Prop`. The real assumptions, the few the guarantees rest on
  without proof, are declared in `Definitions/Assumptions.lean` or a module under it; a premise declared elsewhere is
  open (The lock).
- `Properties/` (package `security_properties`): statements about the code's Lean functions, the checks each protocol
  runs, one `Properties.<Protocol>` module per protocol.
- `Proofs/` (package `security_proofs`): every proof, `<Area>.SecurityProofs.G : <Area>.Guarantees.G` for each
  guarantee, and the lemmas behind them.

What may import what (`tests/test_lean_packages.py`): `security` requires only Mathlib; `security_properties` requires
`security` and the code it states properties of, never `security_proofs`; `security_proofs` requires both, the code and
ArkLib; and no code package requires anything here. The lock's `layers` say which spec modules each module prefix may
import, and its `trees` forbid a statement from reading, or a definitions or
properties module from importing, anything in `Proofs/`. Flock is the exception until its spec is extracted: its
guarantees and assumptions are still in `Proofs/Flock`, which `trees` lists by module.

## The lock

Each package's `lean-audit.json` is its lock. For each guarantee it records the signature, its premises (the named
assumptions, the domain hypotheses, the open premises, and the `Prop`s of an assumptions module its statement only reads) and a SHA-256 hash of
its type, and, per module, every definition the statements read with a digest. A named assumption, under `assumptions`, is
declared in `Definitions/Assumptions.lean` (`assumptions_module`) or under it (`Properties`' lock also lists
`Properties/Assumptions.lean`, for one stated over the code, which the spec can't import); every other assumption is open, under `open` with
the Linear issue that discharges it and its due date, and `--update` adds none on its own, so that list only shrinks. A
hypothesis that is part of what its guarantee says (a domain, a scope, the layer below's guarantee) is under `domain`
with a one-line `why` instead, and a guarantee with an honest witness must have that witness conclude it. The
Lean check (`tools/verity/lean/check.py`) builds every file, replays every declaration through Lean's kernel, and fails
when:

- a declaration uses an axiom other than `propext`, `Classical.choice` and `Quot.sound` (so no `sorry`, `axiom` or
  `native_decide`), or an option that skips the kernel;
- a guarantee has a `Prop` hypothesis that is not an assumption: a `Prop` listed by name in the lock, with an owner
  and its definition's digest;
- a premise is listed under the wrong kind (a named assumption declared outside `Definitions/Assumptions`), or an
  open premise is not listed;
- a statement reads a definition outside the spec, or breaks `layers` or `trees`;
- a claim definition lacks its witnesses: a theorem that it fails at a toy instance, and one per deployed record that its
  checkable antecedents hold there;
- a record, or a deployed guarantee's `trust_gap` entry, no longer matches the build.

A changed record means a guarantee now says something else. When one lands, the statement before and after is shown
with each changed definition's source; nothing waits on it, and a guarantee that only moved shows none.

## The guarantees to read first

**Protocol soundness** is one theorem (Start here, item 4), the union bound over a composite's subprotocols at its
compiled verifier: `Verity.soundness_of_subprotocols` at onsite's seven and `Verity.remote_soundness_of_subprotocols` at
remote's three, each subprotocol's target a hypothesis until its own proof discharges it. Start at
`Properties/Verity/README.md`: the statement, the subprotocols each composes, how to read it and what it rests on.

The table lists the subprotocols' own guarantees, each in its own form; each is proved under `Proofs/`. A status names the statement's named assumptions; the
conditions it places on its inputs and setup are left out (the lock lists them).

| Subprotocol | Guarantee | Status |
|---|---|---|
| Sampled proofs | `Flock.Guarantees.SampledKSLiveV3`, stated in `Proofs/Flock/Bridge/ScopeGate.lean`, proved in `Proofs/Flock/Bridge/KSLive.lean` | Proved under named assumptions: `UniformRandomBytes`, `RecordCustodyZKJ`, and SHA-512's collision resistance (`SHA512CRStrict`, `SHA512CRExpected`) at its finders |
| PoUW | `Properties.Pouw.Period.H1CheckedCreditPeriod`, the instance at the deployed H1 tile unit of the general `CheckedCreditPeriod` (both in `Properties/lean-audit.json`), `Properties/Pouw/Period.lean` | Proved under named assumptions: `ServiceSound`, `SampleIsCoin`, `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `StagedPins`, `UnitFixed`, `OperandHMu`, `ReceiptLuckAt`, `GuessLuckAt`. The deployed path doesn't meet `SaltIsLiveCoin` yet, and no served path meets `AcceptedLive` (credit only after `flock-verify --zk --registered` accepts) |
| PoUS | `Pous.Period.PeriodHeld`, `Properties/Pous/Period.lean` | Proved under named assumptions: `CoinCommitHides`, `RingDevice`, and `P2RingAuditSecure`, an open conjecture |
| Bandwidth audit | `Bandwidth.Guarantees.AuditRO`, `Definitions/Accounting/Guarantees/Bandwidth.lean` | Proved, in the random-oracle model |
| Network certification | `NetTiming.Guarantees.PeriodLeakAccepted`, `Definitions/NetworkCertifier/Guarantees/Period.lean` | Proved under named assumptions: `TrustedHost`, and sampled proofs' `Claims.SamplingGeneric.Compiled.KS` and `Claims.Sampling.CommitmentScheme.Binds`; each window is bounded alone, with no union over the period yet |
| Memory wipes | `Pous.Guarantees.StreamedWipe`, `Definitions/Pous/Guarantees/StreamedWipe.lean` | Proved; the PRF enters as its advantage term |
| PoUB | `Properties.RemoteAudit.Line.LineHeld`, `Properties/Accounting/Remote/Line.lean` | Inside protocol soundness: its P2 audit (`RemoteAudit.P2Protocol`, on PoUS's store) is one of onsite's seven subprotocols, so `Verity.soundness_of_subprotocols` at `.onsite` covers it through its target, `BandwidthAuditSound`. This guarantee, in its own form, is proved under named assumptions: `Accounting.Remote.Held.Assumptions`, `Accounting.Remote.Draw.Assumptions` |

PoUW has other guarantees, outside protocol soundness.

## The remote protocol in Lean

Remote (`verity/protocols/remote/verifier`, `Core.Remote.protocol`) is the composite for an auditor with no devices at the
site. It composes three subprotocols, SPOT (one-stage v2; the recursive audit refused), PoUW's period check
(`verity/pouw/period/v1`, onsite's work subprotocol too) and the inference-only policy's check (`NciCheck.Protocol`), and
accepts only when all three accept and the commitment binds them (`Properties.Composition`, proved), and only when the
policy and the one-stage audit commit to one record and every window of the period names it as its Commit record
(`Core.Composition.tiedSame`, `Core.Remote.windows`). Onsite holds the same two ties (`Core.Onsite.ties`) and adds PoUS,
PoUB, network certification and memory wipes, with PoUS and PoUB-P2 on one store.

- **Guarantees:** the table's sampled-proofs row and PoUW's period guarantee,
  `Properties.Pouw.PeriodJ10R.H1J10RCheckedCreditPeriod`, both proved. Neither is stated yet as its target in
  `Verity.remote_soundness_of_subprotocols`, so the two compose by hand. Remote's sampled proofs' one-stage form is
  v2, which runs `flock-verify --zk` on the sessions' files and reads the challenger's signed coins (v1, which takes
  verdicts as evidence, is refused); where v2 accepts, its session is in V3's and V5's events
  (`Verity.Sampled.checkBlobs_event`), and only V5's has a coupling.
- **Assumptions:** those guarantees' named assumptions, `UniformRandomBytes`, `RecordCustodyZKJ`, `SHA512CRStrict`,
  `SHA512CRExpected`, `TileServiceSound`, `TileSampleIsCoin`, `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `UnitFixed` and
  `SharedNoiseLucks`. Trust model: nothing the developer runs or hosts is trusted, and every coin is live,
  with no Fiat–Shamir and no retries. The challenger and the verifier both run on auditor hosts. The challenger, which
  flips each coin after the commitment it challenges, is the one auditor-trusted device remote relies on
  (`trusted-host`); the verifier is the auditor's own check. Two premises about where the challenger runs are
  onsite's (`verity/protocols/spot/verifier/README.md`): `CoinsConfidential`, the named Prop
  `Pouw.Assumptions.CoinsConfidential` (`Definitions/Pouw/Assumptions/Dither.lean`, in the lock, taken by
  `Pouw.Guarantees.Dither.FleetSound`), and `OneClock`, prose only.
  Remote's coin path is three assumptions over the challenger's log (`Definitions/Core/Challenger.lean`), stated in
  `Definitions/Core/Assumptions/Challenger.lean` and listed in the lock: the challenger logs every coin it releases
  (`ChallengerLogsCoins`), flips each only after what it samples is recorded (`FlipsAfterRecorded`), and the verifier
  reads coins only from that log (`VerifierReadsLog`). Under them a sample sampled with a logged coin misses a set fixed
  before it at most as often as the sampling design allows (`Verity.Guarantees.SampleAfterRecorded`, proved); none of
  remote's guarantees reads `CoinsConfidential`, and no `.lean` file defines `OneClock`. Network certifiers and memory
  challengers are onsite's.
- **Not the remote composite.** `Accounting.Remote` and `Properties.Accounting.Remote` are PoUB
  (`verity/protocols/poub/verifier`), which proves the bits a node read from its registered store. It is a
  different protocol from the remote composite, and a subprotocol of onsite only.

## Building and checking

With the toolchain `lean-toolchain` pins:

~~~text
lake build                                    # in security/: the spec
lake build                                    # in security/Properties/, then security/Proofs/ (fetches ArkLib)
python3 tools/verity/lean/check.py Security   # from the repository root
~~~

The check audits `Proofs/` with the spec, from one build, and refuses a machine below its memory floor.
