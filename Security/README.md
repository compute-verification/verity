# Security

The Lean statement and proof of what an accepted run of Verity guarantees: for each subprotocol, and then for the composed
protocol, a bound on the probability that the verifier accepts while a claim about the run fails. The terms are the
<<<<<<< ours
[Glossary](../GLOSSARY.md)'s.
=======
[Glossary](../GLOSSARY.md)'s. The first section is the whole security argument, from what makes training hard to the
slowdown a secret training run suffers, with its symbols and where each step stands in Lean.

## The security argument

This is a conditional argument, not yet a proved theorem. Its scientific content is hypotheses about learning:
information premises, which structural arguments turn into local bounds, and one coverage assumption. The protocol's
part is to make them matter. The question it rests on:

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

Ω_* comes from part 1, each q_I from part 2 and Q_rem from part 3. Part 4 divides by rates; it adds no assumption.

The numbers come from the slowdown calculator (`benchmarks/rho`), not from this file.

### Objects

- **Time.** A **tick** is one step of the one clock (`OneClock`, a premise stated in prose, not yet in Lean). A **monitoring period** J is a span
  of consecutive ticks, |J| long; every other span is a block's.
- **Hardware.** Leaves L are devices; a location X is a set of leaves. The **site** is the registered hardware, public;
  hardware outside it is unregistered. Φ_X is X's **capacity**, work a tick at its rated peak (the site entry: SKU,
  clock, rate per operand class). Φ_R and Φ_U are the registered and unregistered capacities, Φ = Φ_R + Φ_U.
- **The actual circuit** A: every gate the hardware computed, hidden work included, each gate g at a leaf x(g) and a
  tick t(g); A[X, I] is its gates at X during the ticks I. The builder's own runs, the developer's preparation and
  anything else the hardware does are gates of A.
- **Initial information**: the registered information θ, input gates whose values are committed before J (models, code,
  data). Nothing else is carried into J.
- **Registration**, once and append-only: the site, θ and the circuit builder B, a public program.
- **Execution.** For each declared advice a_j (a string of at most ℓ_a bits, private and committed, ℓ_a public), B
  outputs B(a_j) = (E_j, 𝒫_j, 𝓘_j, loc_j):
  - E_j, the execution block: a Boolean circuit whose inputs are θ, earlier blocks' outputs and fresh input;
  - 𝒫_j, a partition of E_j's gates into proof blocks, each block's outputs within its public size class and their
    number within a public cap;
  - 𝓘_j, a partition of E_j's gates into inference blocks, each of exactly c gates (padded) reading at most k bits of
    fresh input, which is anything outside θ;
  - loc_j : gates(E_j) → registered devices × ticks, the placement.

  a_j, E_j, 𝒫_j, 𝓘_j and loc_j are private, their padded counts public. The declared circuit is D = E_1 ‖ … ‖ E_n.
  Since E_j's structure is a function of a_j, the advice is fresh information: an inference block whose structure
  depends on k_a bits of a_j reads them as fresh input.
- **Derived quantities**, none assumed:
  - F, the faulty proof blocks (committed outputs that differ from their gates applied to their committed inputs), and
    the **faulty share** φ, the share of credited work that lies in F;
  - Φ_cred, the rate PoUW credits as declared work, and the **declared utilization** u = Φ_cred/Φ_R;
  - Φ_gap, credited work a cheaper route could produce; Φ_bk, bookkeeping (certificate slack and verdict lag);
  - Φ_D, the rate of correct credited declared work;
  - q_I, the most weighted counted multiplications inference block I computes whole (part 2); ζ̄ and the counts below
    use the same weights.
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
2. **Challenge.** The challenger, on an auditor host, draws live coins: the sample of proof blocks (by size class) and
   PoUW's noise and check coins.
3. **Proof.** The developer proves each sampled proof block in zero knowledge and answers PoUW's challenges in time.
4. **Validation.** The verifier checks the proofs, the partitions' public rules, 𝓘_j's budgets (c, k) and PoUW's
   credit.
5. **Accounting.** Each subprotocol's account for E_j is tallied: the bound on faulty proof blocks, the credited work,
   the inference blocks. The policies read the accounts.

The monitoring period is accepted when every subprotocol accepts every E_j and all committed to one registration.
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
per-block cap to the circuit). Coverage (part 1) then says which learning methods the bound constrains. The Lean is
`Nci.Capability` (vocabulary), `Nci.Assumptions` (premises) and `Nci.Guarantees` (the repertoire), one module per role;
`Security/Definitions/Nci/Capability/README.md` surveys every premise, structure and bound in one ontology, with how to
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
| both, priced together (`Items.sum`) | joint hardness | the same bits describe both families; an additive budget is a hypothesis of its own |

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

**Incidence** (`Nci.Capability.Incidence`): each counted product reads a cell of its slot and of its operand occurrence.
With at most r products per slot-operand pair and at most a_out per operand, a block whose products use U slots and V
operands computes at most (`Nci.Guarantees.IncidenceCap`)

$$ |\mathcal J_I| \;\le\; \min\{\,r\,|U|\,|V|,\ a_{\mathrm{out}}\,|V|\,\}. $$

The budgets decide which (|U|, |V|) are possible (`Nci.Guarantees.OperandCapSeparate`, `Nci.Guarantees.OperandCapJoint`).
Under one joint budget β_w|U| + β_x|V| ≤ h the cap is

$$ q(h) \;=\; \max_{\beta_w u + \beta_x v \le h} \min\{r u v,\ a_{\mathrm{out}} v\}
   \;=\; \begin{cases} \dfrac{r h^2}{4\beta_w\beta_x} & r h \le 2\beta_w a_{\mathrm{out}} \text{ (pairs limit it)} \\[1ex]
   \dfrac{a_{\mathrm{out}}\,(h - \beta_w a_{\mathrm{out}}/r)}{\beta_x} & \text{otherwise (operand supply and fan-out limit it)} \end{cases} $$

and under separate budgets min{r·h_w·h_x/(β_w·β_x), a_out·h_x/β_x}. These are two regimes of one calculation, not two
kinds of training. The rectangle U × V overcounts pairs that never meet in a product; a finer count of the incidence
graph is a structural improvement.

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
  the held budget, nothing is recomputed;
- **weighted products** (`Nci.Guarantees.WeightedProductCap`): weighting product (i, j) by p_i·ω_j, a block's weighted
  products are at most the held slot budget h_w times its served weight.

Together, with Ω_I the block's weighted products,

$$ \Omega_I \;\le\; h_w\Big(k + \frac{c}{g}\Big), \qquad\text{and}\qquad \Omega_I \;\le\; h_w\,k
   \ \text{ when recomputing needs more state than } h_w. $$

The bound degrades with the gate budget instead of failing at a threshold: importing spends fresh bits, reconstructing
spends gates. The g gates are the analysed implementation's certificate, not a lower bound on every algorithm computing
the same function, and products split between blocks still need their own account. The weights replace "every
activation is hard" with "enough jointly priced, structurally supported computation": cheap slots get low prices,
overlapping regions low weights, and choosing ω is a linear program once the prices are justified.

#### Counting and the standard assemblies

Theorem 3 (`Verity.Guarantees.Counting`, proved) turns a per-block cap q on weighted counted products into
|C| ≥ c·Ω/q, Ω the computation's weighted total; coverage must then bound Ω, not the raw product count. The assemblies
used so far:

| Information | Structure | Per-block cap |
|---|---|---|
| update hardness | dependency, recomputing priced out | h_w·k (uniform price β_w: k·(k' + λ_w)/β_w products) |
| update hardness | dependency, recomputing by gates | h_w·(k + c/g) |
| update and activation hardness, separately | incidence | min{r·h_w·h_x/(β_w·β_x), a_out·h_x/β_x} |
| joint hardness | incidence | q(h) |
| activation hardness | incidence, fan-out only | a_out·h_x/β_x |
| a recoverable hard source | outputs | (k' + δ + amb)/β_v |

An assembly counts only when its premises hold for the same blocks; the smallest valid cap wins, and a refuted premise
contributes nothing. Since the first rows are about k², they bind only when a block's own capacity c·ζ̄ is well above
k²/β_w. The threaded read costs (`Nci.Threading.Threaded`, `Nci.Assumptions.WeightsIncompressible`), under which
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

- **sampled proofs:** F lies in the confidence set M_δ, which bounds the faulty proof blocks in each size class, and
  with each class's block size the faulty share φ;
- **PoUW:** Φ_D ≥ (1 − φ)·Φ_cred − Φ_gap − Φ_bk, with each credited gate a distinct gate of A;
- **the inference-only policy:** every 𝓘_j is a (c, k)-partition;
- **one registration** across the subprotocols;
- **the utilization policy**, once it exists: u ≥ u_min;
- **onsite:** each server block, of at most c_S gates, reads at most k_S = k_wipe + (r_in + r_side)·Δ_wipe fresh bits.

**The accounting**, deterministic. Every counted multiplication of a successful computation inside A goes to exactly
one of two parts:

- an inference block I of correct credited declared computation whose effective fresh input k_I + k_{F,I} + k_{a,I}
  (its own fresh input, the bits faulty proof blocks inject into it, and the advice bits its structure depends on) is
  within the range of part 2's assembly, contributing q_I;
- the remainder: unregistered hardware, registered hardware not running credited declared work, the faulty proof
  blocks, the credit gap, the bookkeeping, blocks out of range, and multiplications split between a block and the rest.

Credit isn't counted twice, and padding isn't evidence that hardware ran anything. The remainder does at most ζ̄
counted multiplications per unit of its fleet-time, so Q_rem ≤ ζ̄·(Φ − Φ_D)·|J| (remote). ζ̄ is a justified ceiling,
not a reference's achieved rate: a reference execution shows what is feasible, not what is optimal. **Reference
efficiency**, ζ̄ = ζ_ref with ζ_ref = M_ref/C_ref the reference's rate, is a hypothesis; at peak, ζ̄ = ζ_ref/MFU₀.

### Part 4: so reaching the target takes more fleet-time

The **slowdown** ρ(𝒢, η) is T_𝒢(η; 𝒜)/T_𝒢(η; 𝓕), T_𝒢(η; ·) the least fleet-time at which some computation in the set
reaches the target with probability above η, and **fleet-time** elapsed time times Φ, registered and unregistered.

**Remote.** A covered successful computation carries a weighted total of at least Ω_* counted multiplications
(part 1; with every weight 1, the plain count). The good blocks number at most Φ_D·|J|/c and each does at most q
(part 2), and the remainder at most Q_rem (part 3). With f = 1 − Φ_D/Φ the
share of capacity outside correct credited declared work,

$$ \Omega_* \;\le\; \Big(\frac{q}{c}\,\Phi_D + \bar\zeta\,(\Phi - \Phi_D)\Big)\,|J|
   \quad\Longrightarrow\quad
   \Phi\,|J| \;\ge\; \frac{\Omega_*}{\bar\zeta\,f + (q/c)\,(1 - f)}. $$

A demonstrated execution reaching the target in fleet-time C_ref gives T_𝒢(η; 𝓕) ≤ C_ref. If f is bounded uniformly
over 𝒜 (the utilization policy bounds it),

$$ \rho \;\ge\; \frac{\Omega_*}{C_{\mathrm{ref}}\,\big[\bar\zeta\,f + (q/c)\,(1 - f)\big]}, $$

up to probability η: by coverage, computations without the covered structure reach the target below T_max with probability
at most η (`Nci.Guarantees.CapabilityFloor`). With the earlier work floor, Ω_* = ζ_ref·C_ref/α, and reference efficiency
this is 1/(α·[f + y]), y = (q/(c·ζ_ref))·(1 − f) the declared yield as a share. Without part 2, q/c could reach ζ̄ and
the bound falls to Ω_*/(ζ̄·C_ref), no slowdown at all: part 2 is what keeps declared work from being training.

**Onsite.** As remote, except that each registered server's remainder is capped per server block: a server block of at
most c_S gates whose effective fresh input k_S is within the assembly's range does at most q_S counted multiplications,
declared and residual together. When k_S is large enough for a server block to hold a token's whole path, the cap is
void and onsite gives remote's bound.

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
| Declared objects | `Verity.Circuit`, `Verity.Partition`, `Verity.Partition.IsCK`, `Verity.RegisteredCircuit` (the registration, whose builder names the registered compiler `Verity.Compiler` B with its advice cap ℓa, and the advice a_j reported in the execution phases; its inference blocks 𝓘 and proof blocks P, B's output on θ and the advice), `Verity.Committed.Explains`, `Verity.Registration`; the site and loc_j not yet in Lean | Partly defined |
| Coverage | `Nci.Assumptions.CoversEfficientMethods` | A named assumption in the lock; Core not yet read off a run's circuit; untested |
| Lemma 1 | `Verity.Guarantees.BlockDescription` | Proved |
| Information premises | `Nci.Assumptions.DescriptionHard`, `Nci.Assumptions.MinEntropyHard`, `Nci.Assumptions.SourceHard`, `Nci.Assumptions.Recovers`, `Nci.Capability.Items` | Named assumptions in the lock, taken by the repertoire's proofs; no price measured at a decoder budget; several uniform activation rates refuted by exact decoders |
| Structure | `Nci.Capability.Incidence`, `Nci.Capability.Consumers`, `Nci.Capability.Consumers.Upstream`, `Nci.Capability.Consumers.OverlapWeights` | Defined; the certificate not yet checked on a named training circuit |
| The repertoire | `Nci.Guarantees.HeldBudget`, `Nci.Guarantees.StatHeldBudget`, `Nci.Guarantees.RecoveredBudget`, `Nci.Guarantees.IncidenceCap`, `Nci.Guarantees.OperandCapSeparate`, `Nci.Guarantees.OperandCapJoint`, `Nci.Guarantees.ImportOrRecompute`, `Nci.Guarantees.RecomputeByGates`, `Nci.Guarantees.RecomputeByState`, `Nci.Guarantees.WeightedProductCap` | Proved (`Proofs/Nci/Capability/`), over premises not yet tested |
| Theorem 3 | `Verity.Guarantees.Counting` | Proved |
| The threaded refinement | `Nci.Threading.Threaded`, `Nci.Assumptions.WeightsIncompressible`, `Nci.Guarantees.ThreadedReads` | Proved for the circuit in hand under assumed read costs |
| Device model | `Accounting.Placed.ComputeRate`, `Accounting.Placed.Links`, `Accounting.Placed.Memory`, `Accounting.Placed.Causal`, `Accounting.Overhead.physClass`; hosts carry no compute, `OneClock` and the inventory not yet in Lean | Defined; three premises unstated |
| The composites | `Core.Remote.protocol`, `Core.Onsite.protocol`, `Properties.Composition.AcceptMeans` | Proved; the policy is not yet a subprotocol of either, and neither checks one registration |
| The utilization policy | not yet in Lean | Not a policy yet |
| Protocol soundness | `Accounting.Overhead.restricted`; `Verity.Soundness` not yet in Lean | Open |
| Sampled proofs | `Flock.Guarantees.SampledKSLiveV3`, `Claims.Sampling.RelS` | Proved; the step from F ∈ M_δ to the credit-weighted φ is open |
| PoUW | `Accounting.Guarantees.Compose.CircuitCredit` | Proved; nothing derives Φ_D from it, or shows each credited gate distinct |
| The inference-only policy | `NciCheck.Properties.SecurityProofs.DescWitnesses`, `Nci.Complies` | Proved; k_F (what faulty blocks inject) and k_a (the advice's share) are unbounded |
| Onsite's k_S | `Pous.Guarantees.StreamedWipe`, `NetTiming.Guarantees.PeriodLeakAccepted` | Proved per window; nothing composes them into k_S |
| Accounting | the assignment of a run's counted multiplications not yet in Lean | Open |
| Rate ceiling ζ̄ | the inline hypotheses of `Nci.Guarantees.DampenerRemote`; a named Prop not yet in Lean | Open |
| The floor | `Nci.Guarantees.CapabilityFloor` | Proved, under coverage |
| Slowdown | `Accounting.Overhead.overhead`, `Accounting.Overhead.threshold`, `Accounting.Overhead.cost` | Defined, with cost in server-time, not fleet-time; no theorem reads it |
| The bounds | `Nci.Guarantees.DampenerRemote`, `Nci.Guarantees.DampenerThreaded` | Proved as arithmetic over inline hypotheses, not over the definition above |
>>>>>>> theirs

## Three trees

- `Definitions/` (package `security`, with `Models.lean`): the spec, the part a reader reads. Each area (`Core`, `Pouw`,
  `Pous`, `NetworkCertifier`, `Accounting`, ...) has its vocabulary, its named assumptions (`Assumptions`) and its
  guarantee statements (`Guarantees`), each a `def G : Prop`.
- `Properties/` (package `security_properties`): statements about the code's Lean functions, the checks each component
  runs, one `Properties.<Protocol>` module per component.
- `Proofs/` (package `security_proofs`): every proof, `<Area>.SecurityProofs.G : <Area>.Guarantees.G` for each
  guarantee, and the lemmas behind them.

What may import what (`tests/test_lean_packages.py`): `security` requires only Mathlib; `security_properties` requires
`security` and the code it states properties of, never `security_proofs`; `security_proofs` requires both, the code and
ArkLib; and no code package requires anything here. The lock's `layers` say which spec modules each module prefix may
import (the spec imports only itself and Mathlib), and its `trees` forbid a statement from reading, or a definitions or
properties module from importing, anything in `Proofs/`. C-Flock is the exception until its spec is extracted: its
guarantees and assumptions are still in `Proofs/Flock`, which `trees` lists by module.

## The lock

Each package's `lean-audit.json` is its lock. For each guarantee it records the signature, the named assumptions among
its hypotheses and a SHA-256 hash of its type, and, per module, every definition the statements read with a digest. The
Lean check (`tools/verity/lean/check.py`) builds every file, replays every declaration through Lean's kernel, and fails
when:

- a declaration uses an axiom other than `propext`, `Classical.choice` and `Quot.sound` (so no `sorry`, `axiom` or
  `native_decide`), or an option that skips the kernel;
- a guarantee has a `Prop` hypothesis that is not a named assumption: a `Prop` listed by name in the lock, with an owner
  and its definition's digest;
- a statement reads a definition outside the spec, or breaks `layers` or `trees`;
- a claim definition lacks its witnesses: a theorem that it fails at a toy instance, and one per deployed record that its
  checkable antecedents hold there;
- a record no longer matches the build.

A changed record means a guarantee now says something else. When one lands, the change is shown, the statement before
and after and each changed definition with its source; nothing waits on it, and a guarantee that only moved shows none.

## The guarantees to read first

**Protocol soundness, `Verity.Soundness`: not yet in Lean on `main`.** It will be one theorem over all the subprotocols:
the verifier accepts while a claim of the period fails with probability at most their errors summed, each subprotocol's
guarantee entering as a target until that subprotocol's own proof discharges it. The subprotocols are the table's first
six rows, memory wipes among them; PoUB's row is outside it. Neither composite runs the bandwidth audit over PoUS's store
or memory wipes yet, and onsite composes PoUB instead (below). Its building block on `main` is `Verity.Sound`
(`Definitions/Core/Game.lean`).

The table lists what it rests on, then one guarantee outside it; each is proved under `Proofs/`. A status names the statement's named assumptions; the
conditions it places on its inputs and setup, which the lock also lists, are left out.

| Subprotocol | Guarantee | Status |
|---|---|---|
| Sampled proofs | `Flock.Guarantees.SampledKSLiveV3`, stated in `Proofs/Flock/Bridge/Refused.lean`, proved in `Proofs/Flock/Bridge/KSLive.lean` | Proved under named assumptions: `UniformRandomBytes`, `RecordCustodyZKJ`, and SHA-512's collision resistance for its finders, `TableCRZAt` and `LinkCRZC` |
| PoUW | `Properties.Pouw.Period.H1CheckedCreditPeriod`, the instance at the deployed H1 tile unit of the general `CheckedCreditPeriod` (both in `Properties/lean-audit.json`), `Properties/Pouw/Period.lean` | Proved under named assumptions: `ServiceSound`, `DrawIsCoin`, `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `StagedPins`, `UnitFixed`, `OperandHMu`, `ReceiptLuckAt`, `GuessLuckAt`. The deployed path doesn't meet `SaltIsLiveCoin` yet, and no served path meets `AcceptedLive` (credit only after `flock-verify --zk --registered` accepts) |
| PoUS | `Pous.Period.PeriodHeld`, `Properties/Pous/Period.lean` | Proved under named assumptions: `CoinCommitHides`, `RingDevice`, and `P2RingAuditSecure`, an open conjecture |
| Bandwidth audit | `Bandwidth.Guarantees.AuditRO`, `Definitions/Accounting/Guarantees/Bandwidth.lean` | Proved, in the random-oracle model |
| Network certification | `NetTiming.Guarantees.PeriodLeakAccepted`, `Definitions/NetworkCertifier/Guarantees/Period.lean` | Proved under named assumptions: `TrustedHost`, `Claims.Flock.ListWithin`, and sampled proofs' `Claims.Sampling.KS` and `Claims.Sampling.Commitment.Binds`; each window is bounded alone, with no union over the period yet |
| Memory wipes | `Pous.Guarantees.StreamedWipe`, `Definitions/Pous/Guarantees/StreamedWipe.lean` | Proved; the PRF enters as its advantage term |
| PoUB | `Properties.RemoteAudit.Line.LineHeld`, `Properties/RemoteAudit/Line.lean` | Outside protocol soundness for now, that audit's own guarantee. Proved under named assumptions: `Accounting.Remote.Held.Assumptions`, `Accounting.Remote.Draw.Assumptions` |

PoUW has other guarantees, outside protocol soundness.

## The remote protocol in Lean

Remote (`verity/core/protocols/remote`, `Core.Remote.protocol`) is the composite for an auditor with no devices at the
site. It composes two subprotocols, sampled proofs' one-stage audit and PoUW's window check, and accepts only when both
accept and the commitment binds both (`Properties.Composition`, proved). Onsite adds PoUS, PoUB and network certification; memory wipes are a component of neither yet.

- **Guarantees:** the table's sampled-proofs and PoUW rows, both proved. The composed statement, `Verity.Soundness`, is
  not in Lean yet (above), so the two compose by hand, and nothing ties the deployed verdicts to the theorems: on
  `main`, remote's sampled-proofs component is one-stage v1, which takes the sessions' verdicts as evidence; the
  verifier's run of `flock-verify --zk` produces them, and the link from remote's acceptance to that run's comes with
  v2.
- **Assumptions:** those rows' named assumptions, `UniformRandomBytes`, `RecordCustodyZKJ`, `TableCRZAt`, `LinkCRZC`,
  `ServiceSound`, `DrawIsCoin`, `SaltIsLiveCoin`, `RegisteredBeforeDraw`, `StagedPins`, `UnitFixed`, `OperandHMu`,
  `ReceiptLuckAt` and `GuessLuckAt`. Trust model: nothing the prover runs or hosts is trusted, and every coin is live,
  with no Fiat–Shamir and no retries. The challenger and the verifier both run on auditor hosts. The challenger, which
  draws each coin after the commitment it challenges, is the one auditor-trusted device remote relies on
  (`trusted-host`); the verifier is the auditor's own check. Two premises about where the challenger runs,
  `CoinsConfidential` and `OneClock`, are prose only (`verity/core/protocols/sampled_proofs/README.md`): no `.lean`
  file defines either. Network certifiers and memory challengers are onsite's.
- **Not PoUB.** `Accounting.Remote` and `Properties.RemoteAudit` are PoUB, the proof of useful bandwidth
  (`verity/core/protocols/remote_bandwidth`), which proves the bits a node read from its registered store. It is a
  different protocol from the remote composite, and a component of onsite only.

## Building and checking

With the toolchain `lean-toolchain` pins:

~~~text
lake build                                    # in Security/: the spec
lake build                                    # in Security/Properties/, then Security/Proofs/ (fetches ArkLib)
python3 tools/verity/lean/check.py Security   # from the repository root
~~~

The check audits `Proofs/` with the spec, from one build, and refuses a machine below its memory floor.
